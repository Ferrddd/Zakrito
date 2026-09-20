# Мультиагентная система рекомендаций для производства ДТ

АВТ → гидроочистка → блендинг. Система получает состояние процесса, оценивает качество и риск выхода за
спецификацию, проверяет ограничения и формирует объяснимую рекомендацию оператору — или честно отказывается,
если надёжной рекомендации нет. Приоритет: **качество и жёсткие ограничения важнее экономики** (ТЗ п.1).

> Статус: черновик README. Разделы помечены ✅ готово / 🟡 заглушка / ⬜ не начато.

## Архитектура

```
 данные (parquet) ──► ProcessState ──► Оркестратор ──┬─► Агент качества      ✅ прогноз серы, риск, отказ
                                          │          ├─► Агент надёжности    🟡 заглушка (normal)
                                          │          └─► Агент оптимизации   🟡 заглушка (нет сценариев)
                                          ▼
                       проверка жёстких ограничений на каждом сценарии
                       (what-if через агент качества; сумма долей = 100%)
                                          ▼
                    Recommendation + аудит-лог (data/converted/audit/cycles.jsonl)
```

Оркестратор (`src/orchestrator/orchestrator.py`) — один цикл: качество → надёжность → есть ли риск →
(оптимизатор → отбрасывание сценариев, нарушающих ограничения → выбор лучшего) → рекомендация. Статусы:
`stable` (действий не нужно), `action`, `risk_no_optimizer`, `no_recommendation` (отказ), `escalate`.
Агенты-заглушки подключаются через конструктор `Orchestrator(quality, reliability=..., optimizer=...)`
и помечаются в `Recommendation.stubbed_agents`.

## Структура

```
src/
  agents/
    base.py                 протокол Agent
    quality/                агент качества (см. его README)
  orchestrator/
    orchestrator.py         цикл принятия решения
    contracts.py            Scenario, Recommendation, протоколы и заглушки надёжности/оптимизации
    adapters.py             AgentReport ↔ общие схемы (agents_schemas.py)
    clients.py              RemoteQualityAgent (HTTP)
  schemas/
    process_state.py        ProcessState / AgentReport
    agents_schemas.py       схемы команды: Snapshot, QualityAssessment, Reliability*, OptimizationInput
  data_pipeline/            загрузка/фичи/сплиты (уже есть в репозитории)
scripts/run_orchestrator.py прогон оркестратора по историческому периоду
tests/
```

## Запуск

```bash
make install                # Python 3.12, venv, зависимости (добавь requirements.additions.txt)
make data_pipeline          # витрина data/converted/telemetry_pac_lims.parquet
make train_quality          # обучение агента качества
make test
make orchestrator_demo      # python -m scripts.run_orchestrator --start 2026-06-01 --steps 12
# опционально: агент качества отдельным сервисом
make quality_api & python -m scripts.run_orchestrator --start 2026-06-01 --remote http://localhost:8001
```

## Допущения (пополнять)

| # | Допущение | Где |
|---|---|---|
| 1 | Сера в ppm ≈ мг/кг; лимит 10 из ТЗ п.4 | `quality/config.yaml` |
| 2 | Риск = тревога классификатора **или** p90 > лимита (консервативно) | `orchestrator.has_quality_risk` |
| 3 | ПАК старше 24 ч → отказ от рекомендации | `QualityAgent.max_pak_age_min` |
| 4 | Пока агент надёжности — заглушка, «normal» ≠ реальная оценка; помечается в выводе | `contracts.NotConnectedReliability` |
| 5 | Технологические диапазоны управляющих тегов не заданы (правило границ ТЗ п.4) — в проверке сценариев TODO | `orchestrator._check` |

## Что дальше

- ⬜ Агент надёжности: `severity_index`, тренд ΔP Р-202 (W10), доля аномальных тегов (схема уже в `agents_schemas.py`).
- ⬜ Агент оптимизации: генерация Δu по управляющим тегам, диапазоны как явные допущения, score/Парето.
- ⬜ Модель плотности D15 (двусторонний допуск) — см. ограничения в README агента качества.
- ⬜ Демо-сценарии: стабильный период / риск / неполные данные / полный цикл; dashboard.
- ⬜ Калибровка `p_violation`, LIMS как контрольный факт в агенте качества.