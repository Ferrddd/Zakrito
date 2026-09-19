import { eventBus } from '../event.js';

export class AgentGraph {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.rawEvaluations = {}; 
        
        
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        
        this.createTooltip();
        this.initNetwork();

        
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
            interaction: { hover: true, tooltipDelay: 0 } 
        };
        
        this.network = new vis.Network(this.container, data, options);

        
        this.network.on('hoverNode', (params) => this.showTooltip(params.node));
        
       
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
        
        
        this.rawEvaluations = traceData.raw_evaluations || {};

        
        const updatedNodes = traceData.nodes.map(node => ({
            id: node.id,
            label: node.id,
            
            color: node.status === 'done' ? '#10b981' : '#f59e0b' 
        }));
        this.nodes.update(updatedNodes);

         
        const updatedEdges = traceData.edges.map(edge => ({
            id: `${edge.from}-${edge.to}`, 
            from: edge.from,
            to: edge.to,
            label: edge.data_passed ? edge.data_passed.join(', ') : ''
        }));
        this.edges.update(updatedEdges);
    }

    showTooltip(nodeId) {
        const data = this.rawEvaluations[nodeId];
        const content = data ? JSON.stringify(data, null, 2) : 'Нет данных об оценке';
        
        this.tooltip.innerHTML = `
            <strong>Агент: ${nodeId}</strong><br>
            <pre style="margin-top: 5px; color: #38bdf8;">${content}</pre>
        `;
        this.tooltip.style.display = 'block';
    }
}