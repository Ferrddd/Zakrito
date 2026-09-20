# Агент качества (`src/agents/quality`)

Прогнозирует содержание серы в товарном ДТ (`24-2000:Mg.Sulfur`, лимит 10 ppm ≈ мг/кг) на горизонтах 30/60/120 мин,
оценивает риск нарушения спецификации и **умеет отказаться** от прогноза, когда данным доверять нельзя.
Для оркестратора — «чёрный ящик» с двумя методами: `capabilities()` и `evaluate(state[, overrides])`.

## Что внутри

```
ProcessState (одна строка витрины)
      │  observe()                      буфер истории (288 точек = 2 суток, сетка 10 мин)
      ▼
 статус ПАК ──► режим: with_analyzer (ПАК ок) │ blind (ПАК нет / заморожен / вне диапазона)
      ▼
 проверки: прогрев истории → полнота тегов → (what-if: известные теги)  ── иначе abstain
      ▼
 OnlineFeatureBuilder  → те же add_engineered + add_lag_features, что и в обучении
      ▼
 QuantileQualityModel (p10/p50/p90, LightGBM)  +  conformal-сдвиг из calibration.json
 ViolationClassifier  (скор нарушения, порог тревоги из threshold.json)
      ▼
 AgentReport: predictions[по горизонтам], data_quality, confidence, drivers (SHAP), abstain/reason
```

| Файл | Роль |
|---|---|
| `agent.py` | `QualityAgent`: буфер, режимы, отказы, what-if, отчёт |
| `features_online.py` | сборка признаков из буфера (паритет с обучением) |
| `model.py` | квантильная модель + классификатор нарушения, `explain()` (SHAP) |
| `train.py` | обучение (walk-forward CV, holdout, conformal, порог тревоги) |
| `api.py` | опциональный FastAPI-сервис |
| `config.yaml` | все гиперпараметры/пути/горизонты |
| `../../schemas/process_state.py` | контракт `ProcessState` / `AgentReport` |

## Быстрый старт

```bash
make install
make data_pipeline          # собирает data/converted/telemetry_pac_lims.parquet
make train_quality          # сейчас: горизонт 6 (60 мин), blind + with_analyzer, --skip-cv
# все горизонты: python -m src.agents.quality.train --force-rebuild
make test
```

**После обновления `train.py` модели нужно переобучить** — иначе нет `calibration.json` и квантили пойдут без
conformal-сдвига (агент предупредит в логе). Артефакты: `data/converted/models/h{H}_{mode}/`
(`quantile/`, `violation/`, `threshold.json`, `calibration.json`, `feature_spec.json`).

## Использование

In-process (основной путь, API не нужен):

```python
from src.agents.quality.agent import QualityAgent
agent = QualityAgent.load()                 # грузит все горизонты, для которых есть артефакты
agent.warm_up(df.loc[:t])                   # история из витрины (иначе первые ~72 цикла — abstain «прогрев»)
report = agent.evaluate(ProcessState(timestamp=t, tags=row_dict))
report = agent.evaluate(state, overrides={"T5_hdt": 345.0})   # what-if для оптимизатора
```

Как сервис (по желанию):

```bash
make quality_api            # http://localhost:8001/docs
curl -X POST :8001/history  -d '[<ProcessState>, ...]'          # прогрев
curl -X POST :8001/evaluate -d '{"state": {...}, "overrides": null}'
```

`src/orchestrator/clients.py::RemoteQualityAgent` реализует тот же протокол — оркестратору всё равно, где живёт агент.

## Контракт вывода (`AgentReport`)

- `predictions[]` — по горизонту: `p10/p50/p90`, `limit`, `p_violation`, `alert_threshold`, `alert`,
  `p90_over_limit`, `source_model`. **Решение о риске консервативное:** `alert OR p90_over_limit`.
- `data_quality` — `pak_status` (`ok|frozen|out_of_range|missing`), `pak_age_min`, `missing_tags`, `history_points`.
- `confidence` — эвристика 0..1 (режим × затухание по возрасту ПАК × полнота тегов), **не вероятность**.
- `drivers[]` — топ-5 локальных SHAP-вкладов (для самого рискового горизонта).
- `abstain` + `reason` — отказ. Причины: нет прогрева истории; нет >30% текущих тегов; неизвестный тег в сценарии;
  последнее достоверное измерение ПАК старше 24 ч (`max_pak_age_min`).

## Допущения и известные ограничения (читать перед демо)

1. **Паритет признаков не проверен на реальных данных.** Онлайн-сборка переиспользует `add_engineered` /
   `add_lag_features`, но я не видел их код. Первым делом прогоните `tests/test_online_parity.py`
   (сравнивает онлайн-признаки с обучающими на 15 случайных моментах). Если `add_engineered` использует окна
   длиннее 288 точек — увеличьте `max_history_points`.
2. **Квантильные модели без монотонных ограничений** (LightGBM не поддерживает их с `objective="quantile"`).
   Ограничения есть только у классификатора. Поэтому what-if по p50/p90 — слабый сигнал; надёжнее опираться на
   `alert`. Физическую корректность контрфактов (T5↑ → сера↓) гарантирует только классификатор.
3. **`p_violation` — не калиброванная вероятность**: классификатор учится с `scale_pos_weight`. Сравнивайте
   с `alert_threshold`, не интерпретируйте как «32% шанс». Для честных вероятностей нужна изотоническая
   калибровка на val (TODO).
4. **Отказ при старом ПАК (>24 ч) действует и в blind-режиме**, хотя blind-модель ПАК не использует — это
   осознанная консервативная политика (пример из ТЗ п.5); отключается `max_pak_age_min=float("inf")`.
5. **ЛИМС не используется.** Таргет — выгрузка ПАК; приоритет ЛИМС→ПАК из ТЗ п.2 на уровне агента не реализован.
6. **Плотность D15 не поддерживается.** `config_density.yaml` есть, но: двусторонний допуск (820–845 — плейсхолдер),
   а `add_target`/`evaluate_violation`/`best_threshold` считают `value > limit`; данные ПАК по D15 только с
   2025‑03‑05; в конфиге `tags_catalog: config/controllable_tags.yaml`, а у серы `markup/controllable_tags.yaml`
   (и `tags_catalog.py` по умолчанию смотрит в `config/`) — привести к одному пути.
7. Единицы: сера в ppm (по массе ≈ мг/кг); конвертаций нет.

## Тесты

- `tests/test_quality_agent.py` — логика агента на фейковых моделях (режимы, NaN-флаги, отказы, what-if, conformal).
- `tests/test_online_parity.py` — train/serve паритет (нужны данные и обученные модели, иначе skip).