PYTHON ?= .venv/bin/python
ROOT_DIR = .

export PYTHONDONTWRITEBYTECODE=1

install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

update_venv:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

copy_env:
	cat .env.example >> .env

# Линтер
lint-ruff:
	$(PYTHON) -m ruff check --no-cache $(ROOT_DIR) --fix
lint-mypy:
	$(PYTHON) -m mypy --disallow-untyped-defs --cache-dir=/dev/null $(ROOT_DIR)
lint: lint-ruff lint-mypy

#Тесты
test:
	$(PYTHON) -m pytest -v
#Запуски
data_pipeline:
	$(PYTHON) -m scripts.run_pipeline

train_quality:
	$(PYTHON) -m src.agents.quality.train --horizon 6 --skip-cv --force-rebuild