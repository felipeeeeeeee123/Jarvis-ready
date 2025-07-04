/**
 * JARVIS v3.0 Dashboard JavaScript
 * Real-time monitoring and control interface
 */

class JarvisDashboard {
    constructor() {
        this.ws = null;
        this.charts = {};
        this.currentSection = 'dashboard';
        this.refreshInterval = 30000; // 30 seconds
        this.autoRefreshTimer = null;
        
        this.init();
    }

    init() {
        console.log('Initializing JARVIS v3.0 Dashboard...');
        
        // Setup navigation
        this.setupNavigation();
        
        // Setup WebSocket connection
        this.setupWebSocket();
        
        // Setup auto-refresh
        this.setupAutoRefresh();
        
        // Initial data load
        this.loadDashboardData();
        
        // Update time display
        this.updateTime();
        setInterval(() => this.updateTime(), 1000);
        
        console.log('Dashboard initialized successfully');
    }

    setupNavigation() {
        // Handle navigation clicks
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.getAttribute('data-section') || 
                               e.target.closest('.nav-link').getAttribute('data-section');
                
                if (section) {
                    this.showSection(section);
                }
            });
        });
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.setupWebSocket(), 5000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }

    setupAutoRefresh() {
        this.autoRefreshTimer = setInterval(() => {
            this.refreshCurrentSection();
        }, this.refreshInterval);
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'system_update':
                this.updateSystemMetrics(data.data);
                break;
            case 'trade_update':
                this.updateTradeData(data.data);
                break;
            case 'alert':
                this.showAlert(data.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.className = `status-indicator pulse ${connected ? 'status-online' : 'status-offline'}`;
        }
    }

    updateTime() {
        const timeElement = document.getElementById('currentTime');
        if (timeElement) {
            timeElement.textContent = new Date().toLocaleTimeString();
        }
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.style.display = 'none';
        });
        
        // Show selected section
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
            targetSection.style.display = 'block';
        }
        
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        const activeLink = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        this.currentSection = sectionName;
        
        // Load section-specific data
        this.loadSectionData(sectionName);
    }

    loadSectionData(section) {
        switch (section) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'trading':
                this.loadTradingData();
                break;
            case 'autonomous':
                this.loadAutonomousData();
                break;
            case 'learning':
                this.loadLearningData();
                break;
            case 'performance':
                this.loadPerformanceData();
                break;
            case 'portfolio':
                this.loadPortfolioData();
                break;
            case 'plugins':
                this.loadPluginsData();
                break;
            case 'models':
                this.loadModelsData();
                break;
            case 'logs':
                this.loadLogsData();
                break;
        }
    }

    refreshCurrentSection() {
        this.loadSectionData(this.currentSection);
    }

    async apiCall(endpoint, options = {}) {
        try {
            const response = await fetch(`/api${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API call failed for ${endpoint}:`, error);
            this.showAlert({
                type: 'error',
                message: `Failed to load data from ${endpoint}`,
                details: error.message
            });
            return null;
        }
    }

    async loadDashboardData() {
        console.log('Loading dashboard data...');
        
        try {
            const status = await this.apiCall('/status');
            if (status) {
                this.updateDashboardMetrics(status);
                this.updateSystemComponents(status);
                this.updateResourceChart(status.system);
                this.updateRecentActivity(status);
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    updateDashboardMetrics(status) {
        // System Health
        const healthElement = document.getElementById('systemHealth');
        if (healthElement && status.system && status.system.system_health) {
            const health = status.system.system_health.score;
            healthElement.textContent = `${Math.round(health)}%`;
            healthElement.style.color = health > 80 ? 'var(--success-color)' : 
                                      health > 60 ? 'var(--warning-color)' : 'var(--danger-color)';
        }

        // Total Return
        const returnElement = document.getElementById('totalReturn');
        if (returnElement && status.performance) {
            const totalReturn = status.performance.total_return || 0;
            returnElement.textContent = `${(totalReturn * 100).toFixed(2)}%`;
            returnElement.style.color = totalReturn > 0 ? 'var(--success-color)' : 'var(--danger-color)';
        }

        // Active Trades
        const tradesElement = document.getElementById('activeTrades');
        if (tradesElement && status.system && status.system.application_metrics) {
            tradesElement.textContent = status.system.application_metrics.active_trades || 0;
        }

        // Autonomous Actions
        const actionsElement = document.getElementById('autonomousActions');
        if (actionsElement && status.autonomous_scheduler) {
            actionsElement.textContent = status.autonomous_scheduler.current_running_actions || 0;
        }
    }

    updateSystemComponents(status) {
        const container = document.getElementById('systemComponents');
        if (!container) return;

        const components = [
            { name: 'System Monitor', status: status.system?.monitoring_active },
            { name: 'Self Healing', status: status.self_healing?.overall_status === 'healthy' },
            { name: 'Model Monitor', status: status.model_monitor?.monitor_running },
            { name: 'Parameter Tuner', status: status.parameter_tuner?.auto_tuning_running },
            { name: 'Plugin Manager', status: status.plugin_manager?.monitoring_enabled },
            { name: 'Learning Engine', status: status.learning_engine?.running }
        ];

        container.innerHTML = components.map(comp => `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span>${comp.name}</span>
                <span class="status-indicator ${comp.status ? 'status-online' : 'status-offline'}"></span>
            </div>
        `).join('');
    }

    updateResourceChart(systemData) {
        const ctx = document.getElementById('resourceChart');
        if (!ctx || !systemData) return;

        if (this.charts.resource) {
            this.charts.resource.destroy();
        }

        const metrics = systemData.system_metrics || {};
        
        this.charts.resource = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'Disk'],
                datasets: [{
                    data: [
                        metrics.cpu_percent || 0,
                        metrics.memory_percent || 0,
                        metrics.disk_percent || 0
                    ],
                    backgroundColor: [
                        'rgba(0, 217, 255, 0.8)',
                        'rgba(0, 255, 136, 0.8)',
                        'rgba(255, 170, 0, 0.8)'
                    ],
                    borderWidth: 2,
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)'
                        }
                    }
                }
            }
        });
    }

    updateRecentActivity(status) {
        const container = document.getElementById('recentActivity');
        if (!container) return;

        const activities = [];
        
        if (status.autonomous_scheduler?.recent_executions) {
            status.autonomous_scheduler.recent_executions.forEach(exec => {
                activities.push({
                    time: new Date(exec.started_at).toLocaleTimeString(),
                    type: 'Autonomous Action',
                    message: `Executed ${exec.action_id}`,
                    status: exec.status
                });
            });
        }

        if (activities.length === 0) {
            activities.push({
                time: new Date().toLocaleTimeString(),
                type: 'System',
                message: 'Dashboard initialized',
                status: 'completed'
            });
        }

        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-dark table-sm">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>Message</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${activities.slice(0, 10).map(activity => `
                            <tr>
                                <td>${activity.time}</td>
                                <td>${activity.type}</td>
                                <td>${activity.message}</td>
                                <td>
                                    <span class="badge ${this.getStatusBadgeClass(activity.status)}">
                                        ${activity.status}
                                    </span>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    async loadTradingData() {
        console.log('Loading trading data...');
        
        try {
            const [strategies, trades] = await Promise.all([
                this.apiCall('/strategies'),
                this.apiCall('/trades?limit=20')
            ]);

            if (strategies) {
                this.updateCurrentStrategy(strategies);
            }

            if (trades) {
                this.updateTradesTable(trades.trades);
            }
        } catch (error) {
            console.error('Error loading trading data:', error);
        }
    }

    updateCurrentStrategy(strategiesData) {
        const element = document.getElementById('currentStrategy');
        if (element && strategiesData) {
            element.textContent = strategiesData.current_strategy || 'Unknown';
        }
    }

    updateTradesTable(trades) {
        const tbody = document.querySelector('#tradesTable tbody');
        if (!tbody || !trades) return;

        tbody.innerHTML = trades.map(trade => `
            <tr>
                <td>${new Date(trade.executed_at).toLocaleString()}</td>
                <td>${trade.symbol}</td>
                <td>
                    <span class="badge ${trade.action === 'buy' ? 'bg-success' : 'bg-danger'}">
                        ${trade.action.toUpperCase()}
                    </span>
                </td>
                <td>${trade.quantity}</td>
                <td>$${trade.price.toFixed(2)}</td>
                <td>${trade.strategy_used}</td>
                <td>
                    <span class="badge ${this.getStatusBadgeClass(trade.status)}">
                        ${trade.status}
                    </span>
                </td>
            </tr>
        `).join('');
    }

    async loadLearningData() {
        console.log('Loading learning data...');
        
        try {
            const insights = await this.apiCall('/learning/insights');
            if (insights) {
                this.updateLearningInsights(insights);
            }
        } catch (error) {
            console.error('Error loading learning data:', error);
        }
    }

    updateLearningInsights(insights) {
        const container = document.getElementById('learningInsights');
        if (!container || !insights) return;

        container.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <div class="text-center">
                        <div class="metric-value">${insights.total_experiences || 0}</div>
                        <div class="metric-label">Learning Experiences</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="text-center">
                        <div class="metric-value">${insights.total_user_ratings || 0}</div>
                        <div class="metric-label">User Ratings</div>
                    </div>
                </div>
            </div>
            <hr>
            <h6>Model Status</h6>
            ${Object.entries(insights.model_status || {}).map(([model, status]) => `
                <div class="d-flex justify-content-between mb-2">
                    <span>${model}</span>
                    <span class="badge ${status.trained ? 'bg-success' : 'bg-warning'}">
                        ${status.trained ? 'Trained' : 'Not Trained'}
                    </span>
                </div>
            `).join('')}
        `;
    }

    getStatusBadgeClass(status) {
        switch (status?.toLowerCase()) {
            case 'completed':
            case 'success':
            case 'filled':
                return 'bg-success';
            case 'running':
            case 'pending':
                return 'bg-warning';
            case 'failed':
            case 'error':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }

    showAlert(alertData) {
        // Create alert element
        const alertHtml = `
            <div class="alert alert-${this.getAlertClass(alertData.type)} alert-dismissible fade show" role="alert">
                <strong>${alertData.type.toUpperCase()}:</strong> ${alertData.message}
                ${alertData.details ? `<br><small>${alertData.details}</small>` : ''}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Find or create alert container
        let alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) {
            alertContainer = document.createElement('div');
            alertContainer.id = 'alertContainer';
            alertContainer.className = 'position-fixed top-0 end-0 p-3';
            alertContainer.style.zIndex = '9999';
            document.body.appendChild(alertContainer);
        }
        
        // Add alert
        alertContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alerts = alertContainer.querySelectorAll('.alert');
            if (alerts.length > 0) {
                alerts[0].remove();
            }
        }, 5000);
    }

    getAlertClass(type) {
        switch (type?.toLowerCase()) {
            case 'success': return 'success';
            case 'warning': return 'warning';
            case 'error': return 'danger';
            case 'info': return 'info';
            default: return 'secondary';
        }
    }

    // Utility methods for other sections (simplified for brevity)
    async loadAutonomousData() { /* Implementation */ }
    async loadPerformanceData() { /* Implementation */ }
    async loadPortfolioData() { /* Implementation */ }
    async loadPluginsData() { /* Implementation */ }
    async loadModelsData() { /* Implementation */ }
    async loadLogsData() { /* Implementation */ }
}

// Global functions for button interactions
async function refreshDashboard() {
    window.dashboard.loadDashboardData();
}

async function executeTrade() {
    const symbol = document.getElementById('tradeSymbol')?.value || 'AAPL';
    
    try {
        const result = await window.dashboard.apiCall('/trades', {
            method: 'POST',
            body: JSON.stringify({ symbol, action: 'auto', quantity: null })
        });
        
        if (result) {
            window.dashboard.showAlert({
                type: 'success',
                message: `Trade executed for ${symbol}`
            });
            // Refresh trading data
            window.dashboard.loadTradingData();
        }
    } catch (error) {
        window.dashboard.showAlert({
            type: 'error',
            message: 'Failed to execute trade',
            details: error.message
        });
    }
}

async function switchStrategy(strategy) {
    try {
        const result = await window.dashboard.apiCall(`/strategies/${strategy}/switch`, {
            method: 'POST'
        });
        
        if (result) {
            window.dashboard.showAlert({
                type: 'success',
                message: `Switched to ${strategy} strategy`
            });
            // Refresh trading data
            window.dashboard.loadTradingData();
        }
    } catch (error) {
        window.dashboard.showAlert({
            type: 'error',
            message: 'Failed to switch strategy',
            details: error.message
        });
    }
}

async function submitFeedback() {
    const category = document.getElementById('feedbackCategory')?.value;
    const rating = parseInt(document.getElementById('feedbackRating')?.value);
    const comments = document.getElementById('feedbackComments')?.value;
    
    if (!category || !rating) {
        window.dashboard.showAlert({
            type: 'warning',
            message: 'Please fill in all required fields'
        });
        return;
    }
    
    try {
        const result = await window.dashboard.apiCall('/learning/rating', {
            method: 'POST',
            body: JSON.stringify({
                interaction_id: `dashboard_${Date.now()}`,
                rating,
                category,
                comments
            })
        });
        
        if (result) {
            window.dashboard.showAlert({
                type: 'success',
                message: 'Feedback submitted successfully'
            });
            // Clear form
            document.getElementById('feedbackComments').value = '';
            document.getElementById('feedbackRating').value = '3';
            // Refresh learning data
            window.dashboard.loadLearningData();
        }
    } catch (error) {
        window.dashboard.showAlert({
            type: 'error',
            message: 'Failed to submit feedback',
            details: error.message
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new JarvisDashboard();
});