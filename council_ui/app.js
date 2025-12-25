/**
 * AI Trading Council - Frontend Application
 * Handles WebSocket connection and dynamic UI updates
 */

class AICouncilApp {
    constructor() {
        this.ws = null;
        this.models = [];
        this.isProcessing = false;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.connectWebSocket();
    }

    bindElements() {
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            modelsList: document.getElementById('models-list'),
            discussionThread: document.getElementById('discussion-thread'),
            rankingsList: document.getElementById('rankings-list'),
            synthesisContent: document.getElementById('synthesis-content'),
            questionInput: document.getElementById('question-input'),
            submitBtn: document.getElementById('submit-btn'),
            refreshBtn: document.getElementById('refresh-btn'),
            phaseIndicator: document.getElementById('phase-indicator'),
            newConversationBtn: document.getElementById('new-conversation-btn'),
            sessionCount: document.getElementById('session-count')
        };
    }

    bindEvents() {
        this.elements.submitBtn.addEventListener('click', () => this.submitQuestion());
        this.elements.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isProcessing) {
                this.submitQuestion();
            }
        });
        this.elements.refreshBtn.addEventListener('click', () => this.refreshModelStatus());

        // Example question buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.elements.questionInput.value = btn.dataset.question;
                this.submitQuestion();
            });
        });

        // New conversation button - clears history
        if (this.elements.newConversationBtn) {
            this.elements.newConversationBtn.addEventListener('click', () => this.startNewConversation());
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.updateConnectionStatus(true);
            console.log('Connected to AI Council server');
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            console.log('Disconnected from server, reconnecting...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }

    updateConnectionStatus(isOnline) {
        const statusIndicator = document.querySelector('.header-status .status-indicator');
        if (isOnline) {
            statusIndicator.classList.add('online');
            statusIndicator.classList.remove('offline');
            this.elements.connectionStatus.textContent = 'Connected';
        } else {
            statusIndicator.classList.add('offline');
            statusIndicator.classList.remove('online');
            this.elements.connectionStatus.textContent = 'Disconnected';
        }
    }

    handleMessage(data) {
        switch (data.type) {
            case 'model_status':
                this.renderModelStatus(data.data);
                break;
            case 'council_started':
                this.startNewSession(data.message);
                break;
            case 'model_thinking':
                this.showModelThinking(data.model_id, data.model_name);
                break;
            case 'model_response':
                this.addModelResponse(data.data);
                break;
            case 'ranking_started':
                this.showRankingPhase(data.message);
                break;
            case 'ranker_thinking':
                this.showRankerThinking(data.ranker_name);
                break;
            case 'ranking_result':
                this.addRankingResult(data.data);
                break;
            case 'final_rankings':
                this.showFinalRankings(data.data);
                break;
            case 'synthesis_started':
                this.showSynthesisPhase(data.message);
                break;
            case 'synthesis_complete':
                this.showSynthesis(data.data);
                break;
            case 'council_complete':
                this.completeSession(data.message);
                break;
            case 'history_cleared':
                this.handleHistoryCleared(data.message);
                break;
            case 'history_data':
                this.updateSessionCount(data.count);
                break;
            case 'error':
                this.showError(data.message);
                break;
        }
    }

    renderModelStatus(models) {
        this.models = models;

        let html = '';
        models.forEach(model => {
            const initials = this.getInitials(model.name);
            const statusClass = model.online ? 'online' : 'offline';

            html += `
                <div class="model-card" id="model-${this.sanitizeId(model.id)}">
                    <div class="model-avatar" style="background-color: ${model.color}">${initials}</div>
                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <div class="model-specialty">${model.specialty}</div>
                    </div>
                    <div class="model-status ${statusClass}"></div>
                </div>
            `;
        });

        this.elements.modelsList.innerHTML = html;
    }

    getInitials(name) {
        return name.split(' ').map(w => w[0]).join('').substring(0, 2).toUpperCase();
    }

    sanitizeId(id) {
        return id.replace(/[^a-zA-Z0-9]/g, '-');
    }

    submitQuestion() {
        const question = this.elements.questionInput.value.trim();
        if (!question || this.isProcessing) return;

        this.isProcessing = true;
        this.elements.submitBtn.disabled = true;

        this.ws.send(JSON.stringify({
            action: 'start_council',
            question: question
        }));
    }

    refreshModelStatus() {
        this.ws.send(JSON.stringify({ action: 'refresh_status' }));
    }

    startNewSession(message) {
        // Clear previous content
        this.elements.discussionThread.innerHTML = '';
        this.elements.rankingsList.innerHTML = '<div class="empty-state"><span class="empty-icon">⏳</span><p>Waiting for responses...</p></div>';
        this.elements.synthesisContent.innerHTML = '<div class="empty-state"><span class="empty-icon">⏳</span><p>Waiting for synthesis...</p></div>';

        // Set phase to responses
        this.setPhase(1);

        // Add session start message
        this.addSystemMessage(message, true);
    }

    setPhase(phase) {
        const phases = this.elements.phaseIndicator.querySelectorAll('.phase');
        phases.forEach((el, i) => {
            el.classList.remove('active', 'completed');
            if (i + 1 < phase) {
                el.classList.add('completed');
            } else if (i + 1 === phase) {
                el.classList.add('active');
            }
        });
    }

    showModelThinking(modelId, modelName) {
        // Highlight model card
        const modelCard = document.getElementById(`model-${this.sanitizeId(modelId)}`);
        if (modelCard) {
            modelCard.classList.add('thinking');
        }

        // Add thinking indicator
        const model = this.models.find(m => m.id === modelId);
        const color = model ? model.color : '#888';

        const thinkingHtml = `
            <div class="message thinking-message" id="thinking-${this.sanitizeId(modelId)}">
                <div class="message-header">
                    <div class="message-avatar" style="background-color: ${color}">${this.getInitials(modelName)}</div>
                    <div class="message-info">
                        <div class="message-name">${modelName}</div>
                        <div class="message-specialty">Thinking...</div>
                    </div>
                </div>
                <div class="thinking-indicator">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span>Formulating expert analysis...</span>
                </div>
            </div>
        `;

        this.elements.discussionThread.insertAdjacentHTML('beforeend', thinkingHtml);
        this.scrollToBottom();
    }

    addModelResponse(response) {
        // Remove thinking indicator
        const thinkingEl = document.getElementById(`thinking-${this.sanitizeId(response.model_id)}`);
        if (thinkingEl) {
            thinkingEl.remove();
        }

        // Remove thinking state from model card
        const modelCard = document.getElementById(`model-${this.sanitizeId(response.model_id)}`);
        if (modelCard) {
            modelCard.classList.remove('thinking');
        }

        // Add response
        const responseHtml = `
            <div class="message" data-model-id="${response.model_id}">
                <div class="message-header">
                    <div class="message-avatar" style="background-color: ${response.color}">${this.getInitials(response.model_name)}</div>
                    <div class="message-info">
                        <div class="message-name">${response.model_name}</div>
                        <div class="message-specialty">${response.specialty}</div>
                    </div>
                </div>
                <div class="message-content" style="border-color: ${response.color}">
                    ${this.formatResponse(response.response)}
                </div>
            </div>
        `;

        this.elements.discussionThread.insertAdjacentHTML('beforeend', responseHtml);
        this.scrollToBottom();
    }

    formatResponse(text) {
        // Basic formatting - convert line breaks and basic markdown
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    showRankingPhase(message) {
        this.setPhase(2);
        this.addSystemMessage(message, true);
        this.elements.rankingsList.innerHTML = '';
    }

    showRankerThinking(rankerName) {
        this.addSystemMessage(`${rankerName} is evaluating responses...`);
    }

    addRankingResult(data) {
        // Show individual ranking result in the discussion
        if (data.best_insight) {
            const html = `
                <div class="message system-insight">
                    <div class="message-content" style="border-color: var(--gold-primary)">
                        <strong>${data.ranker}'s Top Insight:</strong><br>
                        ${data.best_insight}
                    </div>
                </div>
            `;
            this.elements.discussionThread.insertAdjacentHTML('beforeend', html);
            this.scrollToBottom();
        }
    }

    showFinalRankings(rankings) {
        let html = '';

        rankings.forEach((rank, index) => {
            let positionClass = '';
            if (index === 0) positionClass = 'gold';
            else if (index === 1) positionClass = 'silver';
            else if (index === 2) positionClass = 'bronze';

            html += `
                <div class="ranking-item">
                    <div class="ranking-position ${positionClass}">${index + 1}</div>
                    <div class="ranking-info">
                        <div class="ranking-name">${rank.model_name}</div>
                    </div>
                    <div class="ranking-score">${rank.score}/10</div>
                </div>
            `;
        });

        this.elements.rankingsList.innerHTML = html || '<div class="empty-state"><p>No rankings available</p></div>';
    }

    showSynthesisPhase(message) {
        this.setPhase(3);
        this.addSystemMessage(message, true);
        this.elements.synthesisContent.innerHTML = `
            <div class="thinking-indicator">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
                <span>Synthesizing collaborative insights...</span>
            </div>
        `;
    }

    showSynthesis(synthesis) {
        const formattedSynthesis = this.formatResponse(synthesis);
        this.elements.synthesisContent.innerHTML = `
            <div class="synthesis-text">
                <p>${formattedSynthesis}</p>
            </div>
        `;
    }

    completeSession(message) {
        this.addSystemMessage(message, true);
        this.isProcessing = false;
        this.elements.submitBtn.disabled = false;
        this.elements.questionInput.value = '';
        this.elements.questionInput.focus();

        // Extract session count from message (e.g., "Session #5")
        const match = message.match(/Session #(\d+)/);
        if (match) {
            this.updateSessionCount(parseInt(match[1]));
        }
    }

    addSystemMessage(message, highlight = false) {
        const html = `
            <div class="system-message ${highlight ? 'highlight' : ''}">
                ${message}
            </div>
        `;
        this.elements.discussionThread.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    }

    showError(message) {
        this.addSystemMessage(`⚠️ ${message}`, true);
        this.isProcessing = false;
        this.elements.submitBtn.disabled = false;
    }

    scrollToBottom() {
        this.elements.discussionThread.scrollTop = this.elements.discussionThread.scrollHeight;
    }

    // ======== Conversation Memory Methods ========

    startNewConversation() {
        if (this.isProcessing) return;

        if (confirm('Are you sure you want to start a new conversation? This will clear the AI\'s memory of previous discussions.')) {
            this.ws.send(JSON.stringify({ action: 'clear_history' }));
        }
    }

    handleHistoryCleared(message) {
        // Reset the UI
        this.elements.discussionThread.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🆕</div>
                <h3>New Conversation Started</h3>
                <p>The AI council's memory has been cleared. Previous discussions have been forgotten.</p>
                <p>Ask a new question to begin a fresh discussion!</p>
            </div>
        `;
        this.elements.rankingsList.innerHTML = '<div class="empty-state"><span class="empty-icon">🏆</span><p>Rankings will appear after the council evaluates responses</p></div>';
        this.elements.synthesisContent.innerHTML = '<div class="empty-state"><span class="empty-icon">🤝</span><p>The council\'s synthesized recommendation will appear here</p></div>';

        this.updateSessionCount(0);
        this.addSystemMessage('✨ ' + message, true);
    }

    updateSessionCount(count) {
        if (this.elements.sessionCount) {
            this.elements.sessionCount.textContent = `Session: ${count}`;
            this.elements.sessionCount.classList.toggle('has-history', count > 0);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.councilApp = new AICouncilApp();
});
