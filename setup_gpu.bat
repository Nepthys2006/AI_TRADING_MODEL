@echo off
REM ===========================================
REM GPU Training Setup for RTX 4050
REM ===========================================
REM Fixes TensorFlow/JAX version conflict
REM Enables CUDA GPU acceleration
REM ===========================================

echo ============================================
echo   AI Trading Model - GPU Setup
echo   RTX 4050 Configuration
echo ============================================

REM Create virtual environment (recommended)
echo [1/4] Creating virtual environment...
python -m venv gpu_env
call gpu_env\Scripts\activate.bat

REM Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install compatible TensorFlow with CUDA support
echo [3/4] Installing TensorFlow GPU (this may take a few minutes)...
pip install tensorflow[and-cuda]==2.15.0

REM Install other dependencies
echo [4/4] Installing dependencies...
pip install numpy pandas scikit-learn plotly

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To train, run:
echo   gpu_env\Scripts\activate.bat
echo   python train/train_big_brother.py
echo.
echo To verify GPU:
echo   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo.
pause
