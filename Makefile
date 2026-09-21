PYTHON ?= .venv/bin/python
ROOT_DIR = .

# Параметры демо-прогона оркестратора (можно переопределять: make orchestrator_demo STEPS=50)
START ?= 2026-06-01
STEPS ?= 12
EVERY ?= 1
UI_URL ?= http://localhost:3000/api/update
UI_DUMP ?= data/converted/ui/cycles.jsonl
QUALITY_URL ?= http://localhost:8001

export PYTHONDONTWRITEBYTECODE=1

.PHONY: install update_venv copy_env lint lint-ruff lint-mypy test test_orchestrator \
	data_pipeline train_quality train_reliability diagnose_reliability quality_api backfill_feed_median train_model \
	orchestrator_demo orchestrator_dump orchestrator_ui orchestrator_remote ui demo help \
	docker-up docker-down docker-logs docker-reset

help:
	@echo "install / update_venv / copy_env  - окружение"
	@echo "lint / test / test_orchestrator   - проверки"
	@echo "data_pipeline / train_quality     - данные и обучение агента качества"
	@echo "train_reliability                 - обучить модель ΔP агента надёжности"
	@echo "diagnose_reliability              - диагностика тега ΔP перед обучением"
	@echo "quality_api                       - HTTP-агент качества (порт 8001)"
	@echo "orchestrator_demo                 - прогон оркестратора, вывод в консоль"
	@echo "orchestrator_dump                 - прогон + JSON для UI в $(UI_DUMP)"
	@echo "orchestrator_ui                   - прогон + POST в UI ($(UI_URL))"
	@echo "orchestrator_remote               - прогон через quality_api ($(QUALITY_URL))"
	@echo "ui                                - фронтенд + тестовые данные"
	@echo "docker-up / docker-down / docker-logs - всё одной командой в Docker"

install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

update_venv:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

copy_env:
	cp -n .env.example .env || true

# Линтер
lint-ruff:
	$(PYTHON) -m ruff check --no-cache $(ROOT_DIR) --fix

lint-mypy:
	$(PYTHON) -m mypy --disallow-untyped-defs --cache-dir=/dev/null $(ROOT_DIR)

lint: lint-ruff lint-mypy

# Тесты
test:
	$(PYTHON) -m pytest -v

test_orchestrator:
	$(PYTHON) -m pytest -v tests/test_orchestrator.py tests/test_ui_payload.py

# Запуски
data_pipeline:
	$(PYTHON) -m scripts.run_pipeline

train_quality:
	$(PYTHON) -m src.agents.quality.train --horizon 6 --skip-cv --force-rebuild

# Агент надёжности: модель нормального поведения ΔP (после data_pipeline)
train_reliability:
	$(PYTHON) -m src.agents.reliability.train

# Что за сигнал W10: распределение, месяцы, корреляции, ΔP по возрасту блока
diagnose_reliability:
	$(PYTHON) -m src.agents.reliability.diagnose

backfill_feed_median:
	$(PYTHON) -m scripts.backfill_feed_median

# Агент качества: API и оркестратор
quality_api:
	$(PYTHON) -m uvicorn src.agents.quality.api:app --port 8001

orchestrator_demo:
	$(PYTHON) -m scripts.run_orchestrator --start $(START) --steps $(STEPS) --every $(EVERY)

# Без UI: пишем JSON-payload в файл (JSON Lines), удобно для отладки формата
orchestrator_dump:
	$(PYTHON) -m scripts.run_orchestrator --start $(START) --steps $(STEPS) --every $(EVERY) --ui-dump $(UI_DUMP)

# С UI: сначала поднимите `make ui`, затем в другом терминале этот таргет
orchestrator_ui:
	$(PYTHON) -m scripts.run_orchestrator --start $(START) --steps $(STEPS) --every $(EVERY) --ui-url $(UI_URL)

# Агент качества как отдельный сервис: сначала `make quality_api`
orchestrator_remote:
	$(PYTHON) -m scripts.run_orchestrator --start $(START) --steps $(STEPS) --every $(EVERY) --remote $(QUALITY_URL)

# UI
ui:
	@echo "Запуск фронтенда и тестовых данных"
	cd ui && node server.js & cd ui && node test.js

# Полный демо-сценарий: поднять UI в одном терминале, оркестратор — в другом
demo:
	@echo "1) make ui   2) в другом терминале: make orchestrator_ui STEPS=50 EVERY=6"

# Docker: сборка + подготовка данных/моделей + API + UI + оркестратор
docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f orchestrator quality-api

# сбросить модели и витрину (пересоберутся при следующем docker-up)
docker-reset:
	rm -rf data/converted/models data/converted/dataset_cache data/converted/telemetry_pac_lims.parquet