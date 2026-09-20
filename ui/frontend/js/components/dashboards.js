import { eventBus } from '../event.js';

export class AgentDashboards {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.initLayout();
        eventBus.on('data:statistics', (data) => this.render(data));
    }

    initLayout() {
        this.container.innerHTML = `
            <div class="stat-card">
                <div class="stat-title">Сессий</div>
                <div class="stat-value" id="stat-sessions">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Успешность</div>
                <div class="stat-value" id="stat-success">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Ср. время (с)</div>
                <div class="stat-value" id="stat-time">0.0</div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Вызовы агентов (шт.)</div>
                <div id="chart-bars"></div>
            </div>
        `;
    }

    render(data) {
        if (!data) return;

        document.getElementById('stat-sessions').textContent = data.total_sessions || 0;
        document.getElementById('stat-success').textContent = `${data.success_rate || 0}%`;
        document.getElementById('stat-time').textContent = data.avg_response_time || 0;

        const chartContainer = document.getElementById('chart-bars');
        chartContainer.innerHTML = ''; 

        if (data.agent_calls && data.agent_calls.length > 0) {
            
            const maxCalls = Math.max(...data.agent_calls.map(a => a.calls));

            data.agent_calls.forEach(item => {
                const widthPercent = maxCalls > 0 ? (item.calls / maxCalls) * 100 : 0;
                
                const row = document.createElement('div');
                row.className = 'bar-row';
                row.innerHTML = `
                    <div class="bar-label" title="${item.agent}">${item.agent}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${widthPercent}%;"></div>
                    </div>
                    <div class="bar-value">${item.calls}</div>
                `;
                chartContainer.appendChild(row);
            });
        } else {
            chartContainer.innerHTML = '<div style="color: #64748b; text-align: center; padding: 10px;">Нет данных</div>';
        }
    }
}