import { eventBus } from './event.js';
import { AgentGraph } from './components/graph.js';
import { AgentDashboards } from './components/dashboards.js';

const graph = new AgentGraph('.agent_tracer');
const dashboards = new AgentDashboards('.dashbords');



//функции динамического обновления DOM на основе JSON

function updateOrchestrator(decision) {
    if (!decision) return;
    const constraintsContainer = document.getElementById('contrains_checked'); 
    
    if (constraintsContainer && decision.constraints_checked) {
 
        constraintsContainer.innerHTML = '<h3>Constraints Checked</h2>';
        decision.constraints_checked.forEach(item => {
            const row = document.createElement('div');
            row.className = `constraint-item priority-${item.priority}`;

            
            const isPass = item.status === 'pass';
            const statusColor = isPass ? '#10b981' : '#ef4444';
            const statusText = isPass ? 'PASS' : 'FAIL';

            row.innerHTML = `
                <span class="constraint-name">${item.constraint}</span>
                <span class="constraint-badge" style="color: ${statusColor}; font-weight: bold;">
                    ${statusText}
                </span>
            `;
            constraintsContainer.appendChild(row);
        });
    }

    document.getElementById('problem-detect').textContent = decision.problem_detected || '';
    document.getElementById('expectation-explain').textContent = decision.explanation || '';
    if (decision.expected_effects) {
        document.getElementById('expec-quality').textContent = decision.expected_effects.quality;
        document.getElementById('expec-prod').textContent = decision.expected_effects.production;
        document.getElementById('expec-equip-risk').textContent = decision.expected_effects.equipment_risk;
    }


    const grid = document.querySelector('.dynemic-parameters-grid');
    const template = document.getElementById('param-row-template');
    
    // удаляем все элементы кроме заголовков
    while (grid.children.length > 4) {
        grid.removeChild(grid.lastChild);
    }

    // клонируем темплейт для каждой рекомендации и вставляем в grid
    decision.recommendation.forEach(rec => {
        const clone = template.content.cloneNode(true);
        clone.querySelector('.param-tag').textContent = rec.tag;
        clone.querySelector('.param-name').textContent = rec.name;
        // показываем изменение значения: текущее -> целевое
        clone.querySelector('.param-val').textContent = `${rec.current_value} => ${rec.target_value}`;
        clone.querySelector('.param-unit').textContent = rec.unit;
        grid.appendChild(clone);
    });
}

function updateTelemetry(inputState) {
    if (!inputState) return;

    //обновление блока ЛИМС
    if (inputState.quality_sources && inputState.quality_sources.length > 0) {
        const lims = inputState.quality_sources[0];
        document.getElementById('lims-param').textContent = lims.parameter;
        document.getElementById('lims-type').textContent = lims.source_type;
        document.getElementById('lims-val').textContent = lims.value;
        document.getElementById('lims-unit').textContent = lims.unit;
        document.getElementById('lims-date').textContent = lims.measured_at;
    }
    
    //обновление предупреждения
    const limsWarning = document.getElementById('LIMS-warning');
    if (inputState.data_quality_warnings && inputState.data_quality_warnings.length > 0) {
        limsWarning.textContent = inputState.data_quality_warnings.join(', ');
        limsWarning.style.display = 'block';
    } else {
        limsWarning.style.display = 'none';
    }

    //обновление таблицы КИП
    const kipsTable = document.querySelector('.kips-table');
    while (kipsTable.children.length > 4) {
        kipsTable.removeChild(kipsTable.lastChild);
    }

    inputState.telemetry_summary.forEach(kip => {  
        kipsTable.insertAdjacentHTML('beforeend', `
            <div class="kip-tag">${kip.tag}</div>
            <div class="kip-name">${kip.name}</div>
            <div class="kip-value">${kip.value}</div>
            <div class="kip-unit">${kip.unit}</div>
        `);
    });
}

//подписываем на шину событий
eventBus.on('data:orchestrator', updateOrchestrator);
eventBus.on('data:telemetry', updateTelemetry);
function restore_from_localstorage() {
    try {
        const  savedData = localStorage.getItem('json_from_server')
        if (savedData) {
            const payload = JSON.parse(savedData)
            if (payload.agents_trace) eventBus.emit('data:agents_trace', payload.agents_trace);
            if (payload.orchestrator_decision) eventBus.emit('data:orchestrator', payload.orchestrator_decision);
            if (payload.input_state) eventBus.emit('data:telemetry', payload.input_state);
            if (payload.agent_statistics) eventBus.emit('data:statistics', payload.agent_statistics);
            console.log('Состояние восстановлено из localStorage');
        }
    } catch(err) {
        console.log('Ошибка изъятия данных из localstorage', err);
    }
}
restore_from_localstorage();

//подключение WebSocket к локальному серверу
const ws = new WebSocket(`ws://${window.location.host}`);

ws.onmessage = (event) => {
    try {
        const payload = JSON.parse(event.data);
        let current_data_state = {};
        const saved = localStorage.getItem('json_from_server');
        if (saved) {
            current_data_state = JSON.parse(saved);
        }
        if (payload.agents_trace) current_data_state.agents_trace = payload.agents_trace;
        if (payload.orchestrator_decision) current_data_state.orchestrator_decision = payload.orchestrator_decision;
        if (payload.input_state) current_data_state.input_state = payload.input_state;
        if (payload.agent_statistics) current_data_state.agent_statistics = payload.agent_statistics;

        localStorage.setItem('json_from_server',JSON.stringify(current_data_state));
        //разделяем JSON
        if (payload.agents_trace) eventBus.emit('data:agents_trace', payload.agents_trace);
        if (payload.orchestrator_decision) eventBus.emit('data:orchestrator', payload.orchestrator_decision);
        if (payload.input_state) eventBus.emit('data:telemetry', payload.input_state);
        if (payload.agent_statistics) eventBus.emit('data:statistics', payload.agent_statistics);
        
    } catch (err) {
        console.error('Ошибка обработки JSON из вебсокета:', err);
    }
};
