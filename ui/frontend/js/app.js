import { eventBus } from './event.js';
import { AgentGraph } from './components/graph.js';

// 1. Инициализируем граф. Передаем CSS-селектор контейнера
const graph = new AgentGraph('.agent_tracer');

// 2. Функции динамического обновления DOM на основе JSON

function updateOrchestrator(decision) {
    if (!decision) return;
    const constraintsContainer = document.getElementById('contrains_checked'); // ID из вашего HTML
    
    if (constraintsContainer && decision.constraints_checked) {
        // 1. Очищаем старое содержимое контейнера
        constraintsContainer.innerHTML = '<h3>Constraints Checked</h2>';

        // 2. Проходимся по массиву ограничений из JSON
        decision.constraints_checked.forEach(item => {
            // item = { constraint: "Сера <= 10 мг/кг", status: "pass", priority: "high" }
            
            // Создаем обертку для одной строки проверки
            const row = document.createElement('div');
            row.className = `constraint-item priority-${item.priority}`;

            // Задаем цвет статуса (pass — зеленый, fail — красный)
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

    // Динамическое обновление грида рекомендаций через <template>
    const grid = document.querySelector('.dynemic-parameters-grid');
    const template = document.getElementById('param-row-template');
    
    // Удаляем все элементы кроме первых 4-х (это заголовки teg, name, value, unit)
    while (grid.children.length > 4) {
        grid.removeChild(grid.lastChild);
    }

    // Клонируем темплейт для каждой рекомендации и вставляем в grid
    decision.recommendation.forEach(rec => {
        const clone = template.content.cloneNode(true);
        clone.querySelector('.param-tag').textContent = rec.tag;
        clone.querySelector('.param-name').textContent = rec.name;
        // Показываем изменение значения: текущее -> целевое
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

//подключение WebSocket к локальному серверу
const ws = new WebSocket(`ws://${window.location.host}`);

ws.onmessage = (event) => {
    try {
        const payload = JSON.parse(event.data);
        
        //разделяем JSON
        if (payload.agents_trace) eventBus.emit('data:agents_trace', payload.agents_trace);
        if (payload.orchestrator_decision) eventBus.emit('data:orchestrator', payload.orchestrator_decision);
        if (payload.input_state) eventBus.emit('data:telemetry', payload.input_state);
        
    } catch (err) {
        console.error('Ошибка обработки JSON из вебсокета:', err);
    }
};

// Для ТЕСТА: Если раскомментировать этот блок, граф и интерфейс отрисуются сразу при загрузке 
// страницы до подключения бэкенда (сюда можно вставить ваш тестовый JSON целиком)
 setTimeout(() => {
     const testJSON = {"meta": {
    "timestamp": "2026-09-12T21:54:30+02:00", //дата и время
    "cycle_id": "cycle_8472",                 //номер итерации
    "has_safe_solution": true,                //есть ли безопасное решение(которе рекомендуем)
    "system_status": "WARNING"                //текущее состяние системы OK/WARNING/CRITICAL
  },

  //входные данные с некоторых тегов(пока что количество неизвестно и вероятно будет динамичным)
  "input_state": {
    "telemetry_summary": [ //значения датчиков телеметрии - высокий приоритет
      {"tag": "T-101", "name": "Температура АВТ", "value": 350.5, "unit": "°C"},
      {"tag": "T-102", "name": "Температура АВТ 2", "value": 351, "unit": "°C"}
    ],
    "quality_sources": [ //качсество топлива(лимсы)
      {
        "source_type": "IEEE",
        "parameter": "Сераs",
        "value": 6.7,
        "unit": "Mг/Kг",
        "measured_at": "2027-09-12T20:00:00+02:00",
        "age_minutes": 114,
        "is_outdated": true
      }
    ],
    "data_quality_warnings": ["Анализ серы устарел (более 2 часов)"] //предупреждения(может и не быть)
  },

  "orchestrator_decision": { //наивысший приоритет
  //если "has_safe_solution": false -- текст сообщения будет, остальные скорее всего будут просто пустыми словарями с этом пункте
    "problem_detected": "риск нарушения ограничения по содержанию серы в товарном дизельном топливе",
    "recommendation": [
      {
        "tag": "F-201",
        "name": "Расход сырья",
        "current_value": 150,
        "target_value": 145,
        "direction": "down",
        "unit": "т/ч"
      }
    ],
    "expected_effects": {
      "quality": "Снижение содержания серы до 8.5 мг/кг",
      "production": "Снижение выработки на 2.3%",
      "equipment_risk": "режим в пределах нормы"
    },
    "constraints_checked": [
      {"constraint": "Сера <= 10 мг/кг", "status": "pass", "priority": "high"},
      {"constraint": "Доли компонентов = 100%", "status": "pass", "priority": "high"}
    ],
    "explanation": "снижение расхода необходимо для удержания серы в норме, так как качество имеет приоритет над производительностью.",
    "overall_confidence": 0.85
  },

  "alternatives": [ // не уверен что это будет, самый низкий приоритет
    {
      "scenario_id": "alt_1",
      "description": "Изменение режима гидроочистки",
      "rejected_reason": "Выход за модельный диапазон температур"
    }
  ],

  "agents_trace": { //трассировка агентов, 
    "nodes": [ //агенты
      {"id": "quality_agent", "status": "done", "confidence": 0.9},
      {"id": "reliability_agent", "status": "done", "confidence": 0.95},
      {"id": "optimization_agent", "status": "done", "confidence": 0.8},
      {"id": "orchestrator", "status": "done", "confidence": 0.85}
    ],
    "edges": [ //связи между агентами и подпись к ним
      {"from": "quality_agent", "to": "optimization_agent", "data_passed": ["forecast_sulfur_10.2"]},
      {"from": "reliability_agent", "to": "optimization_agent", "data_passed": ["risk_index_low"]},
      {"from": "optimization_agent", "to": "orchestrator", "data_passed": ["all"]}
      
    ],
    "raw_evaluations": { // промежуточные значения агентов, могут быть любыми в зависимости от агента
      "quality_agent": {"forecast": {"sulfur": 10.2}, "risk_of_violation": true},
      "reliability_agent": {"risk_index": 0.2, "mode_permissible": true},
      "optimization_agent": {"generated_scenarios": 3, "valid_scenarios": 1}
    }
  },

  "visualization_data": { //пока что вообще не обращай на это внимание
    "pareto_front": [
      {"x_production": 145, "y_sulfur": 9.5, "is_selected": true},
      {"x_production": 150, "y_sulfur": 10.2, "is_selected": false}
    ],
    "trend_charts": {
      "timestamps": ["..."],
      "sulfur_actual": ["..."],
      "sulfur_forecast": ["..."]
    }
  }};
     eventBus.emit('data:agents_trace', testJSON.agents_trace);
     eventBus.emit('data:orchestrator', testJSON.orchestrator_decision);
     eventBus.emit('data:telemetry', testJSON.input_state);
}, 500);
