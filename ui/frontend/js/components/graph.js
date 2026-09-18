import { eventBus } from '../event.js';

export class AgentGraph {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.rawEvaluations = {}; // Сюда будем сохранять данные для тултипов
        
        // DataSet - специальные реактивные массивы библиотеки vis-network
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        
        this.createTooltip();
        this.initNetwork();

        // Подписываемся на поступление новых данных для графа
        eventBus.on('data:agents_trace', (data) => this.render(data));
    }

    createTooltip() {
        this.tooltip = document.createElement('div');
        // Стилизуем всплывающее окно прямо в JS для независимости компонента
        this.tooltip.style.cssText = `
            position: absolute;
            display: none;
            background: rgba(30, 41, 59, 0.95);
            color: #fff;
            padding: 10px;
            border-radius: 6px;
            z-index: 1000;
            font-family: monospace;
            font-size: 12px;
            pointer-events: none; /* Чтобы мышь не цеплялась за окно */
        `;
        document.body.appendChild(this.tooltip);
    }

    initNetwork() {
        const data = { nodes: this.nodes, edges: this.edges };
        
        // Настройки физики и внешнего вида графа
        const options = {
            nodes: { shape: 'dot', size: 20, font: { size: 14, color: '#333' } },
            edges: { arrows: 'to', font: { align: 'top', size: 12 } },
            physics: { enabled: true, solver: 'repulsion' },
            interaction: { hover: true, tooltipDelay: 0 } // Включаем события наведения
        };
        
        this.network = new vis.Network(this.container, data, options);

        // Слушатель наведения на узел
        this.network.on('hoverNode', (params) => this.showTooltip(params.node));
        
        // Слушатель ухода курсора с узла
        this.network.on('blurNode', () => { this.tooltip.style.display = 'none'; });
        
        // Привязка окна к координатам мыши при движении
        this.container.addEventListener('mousemove', (e) => {
            if (this.tooltip.style.display === 'block') {
                this.tooltip.style.left = e.pageX + 15 + 'px';
                this.tooltip.style.top = e.pageY + 15 + 'px';
            }
        });
    }

    render(traceData) {
        if (!traceData) return;
        
        // Сохраняем raw_evaluations для тултипов
        this.rawEvaluations = traceData.raw_evaluations || {};

        // 1. Форматируем и обновляем вершины (агентов)
        const updatedNodes = traceData.nodes.map(node => ({
            id: node.id,
            label: node.id,
            // Зеленый если done, иначе оранжевый
            color: node.status === 'done' ? '#10b981' : '#f59e0b' 
        }));
        this.nodes.update(updatedNodes);

        // 2. Форматируем и обновляем ребра (переданные данные)
        const updatedEdges = traceData.edges.map(edge => ({
            id: `${edge.from}-${edge.to}`, // Уникальный ID ребра
            from: edge.from,
            to: edge.to,
            // Склеиваем массив переданных данных в строку для подписи
            label: edge.data_passed ? edge.data_passed.join(', ') : ''
        }));
        this.edges.update(updatedEdges);
    }

    showTooltip(nodeId) {
        const data = this.rawEvaluations[nodeId];
        // Формируем HTML внутренности тултипа. Если данные есть — переводим JSON в читаемый текст
        const content = data ? JSON.stringify(data, null, 2) : 'Нет данных об оценке';
        
        this.tooltip.innerHTML = `
            <strong>Агент: ${nodeId}</strong><br>
            <pre style="margin-top: 5px; color: #38bdf8;">${content}</pre>
        `;
        this.tooltip.style.display = 'block';
    }
}