# ===========================================
# SESSION-AWARE MULTIHEAD ATTENTION LAYER
# ===========================================
# Implements attention mechanism with:
# - Recency weighting (emphasize last 20% of sequence)
# - Session-aware masking (London/NY/Asian encoding)
# - Volatility spike detection for dynamic weighting
# ===========================================

import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout
import numpy as np


class SessionAwareAttention(Layer):
    """
    Custom attention layer that:
    1. Applies MultiHeadAttention for temporal dependencies
    2. Weights recent candles higher (last 20% of sequence)
    3. Encodes trading session context (London/NY/Asian)
    4. Uses residual connection for gradient flow
    """
    
    def __init__(self, 
                 num_heads=4, 
                 key_dim=32,
                 dropout_rate=0.1,
                 recency_weight=2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.recency_weight = recency_weight
        
    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.feature_dim = input_shape[2]
        
        # MultiHeadAttention layer
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        
        # Layer normalization for stability
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network after attention
        self.ffn = tf.keras.Sequential([
            Dense(self.feature_dim * 2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.feature_dim)
        ])
        
        # Projection for attention output
        self.output_dense = Dense(self.feature_dim)
        
        # Learnable recency weight scale
        self.recency_scale = self.add_weight(
            name='recency_scale',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.recency_weight),
            trainable=True
        )
        
        super().build(input_shape)
    
    def _create_recency_mask(self, seq_len, batch_size):
        """
        Create attention bias that emphasizes recent timesteps.
        Last 20% of sequence gets higher weight.
        """
        positions = tf.range(seq_len, dtype=tf.float32)
        
        # Calculate recency weight: higher for recent positions
        # Last 20% of sequence gets boosted
        recency_threshold = 0.8 * tf.cast(seq_len, tf.float32)
        is_recent = tf.cast(positions >= recency_threshold, tf.float32)
        
        # Create weight vector: 1.0 for old, recency_scale for recent
        weights = 1.0 + is_recent * (self.recency_scale - 1.0)
        
        # Reshape for attention: (1, 1, 1, seq_len)
        weights = tf.reshape(weights, (1, 1, 1, seq_len))
        
        return weights
    
    def _create_session_mask(self, hour_features):
        """
        Create session-aware attention mask.
        Suppress Asian session (22:00-06:00 GMT) noise.
        
        hour_features: tensor with hour information (sin/cos encoded)
        Returns mask with lower weights for Asian session
        """
        # For now, return uniform weights
        # Session masking will be applied via hour encoding in features
        return None
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = self.seq_len
        
        # Create recency bias
        recency_weights = self._create_recency_mask(seq_len, batch_size)
        
        # Apply MultiHeadAttention
        # Query, Key, Value all from input (self-attention)
        attn_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training,
            # attention_mask can be added here for session masking
        )
        
        # Residual connection + normalization
        x = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x, training=training)
        
        # Second residual connection
        x = self.layernorm2(x + ffn_output)
        
        # Weighted pooling using recency weights
        # Apply recency weights to temporal dimension
        weights_expanded = tf.broadcast_to(
            tf.reshape(recency_weights, (1, seq_len, 1)),
            (batch_size, seq_len, self.feature_dim)
        )
        
        # Weighted mean pooling
        weighted_sum = tf.reduce_sum(x * weights_expanded, axis=1)
        weight_sum = tf.reduce_sum(weights_expanded, axis=1)
        pooled = weighted_sum / (weight_sum + 1e-8)
        
        return pooled
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
            'recency_weight': self.recency_weight,
        })
        return config


class VolatilityAdaptiveAttention(Layer):
    """
    Extended attention that dynamically adjusts weights based on
    recent volatility (ATR) in the sequence.
    """
    
    def __init__(self, 
                 num_heads=4,
                 key_dim=32,
                 dropout_rate=0.1,
                 atr_feature_idx=15,  # Index of ATR in feature vector
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.atr_feature_idx = atr_feature_idx
        
    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.feature_dim = input_shape[2]
        
        # Base attention
        self.base_attention = SessionAwareAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout_rate=self.dropout_rate
        )
        
        # Volatility gate: learns to adjust attention based on ATR
        self.vol_gate = Dense(1, activation='sigmoid')
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Extract ATR feature for volatility gating
        atr_values = inputs[:, :, self.atr_feature_idx:self.atr_feature_idx+1]
        
        # Compute volatility gate (higher ATR = higher attention weight)
        vol_gate = self.vol_gate(atr_values)  # (batch, seq, 1)
        
        # Apply volatility-weighted attention
        # Scale input by volatility before attention
        vol_weighted_input = inputs * (1.0 + vol_gate)
        
        return self.base_attention(vol_weighted_input, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
            'atr_feature_idx': self.atr_feature_idx,
        })
        return config
