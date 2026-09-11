PYTHON ?= .venv/bin/python
ROOT_DIR = .

update_venv:
	pip install --upgrade pip
	pip install -r requirements.txt

copy_env:
	cat .env.example >> .env

# Линтер
lint-ruff:
	$(PYTHON) -m ruff check --no-cache $(ROOT_DIR) --fix
lint-mypy:
	$(PYTHON) -m mypy --cache-dir=/dev/null $(ROOT_DIR)
lint: lint-ruff lint-mypy

#Запуски
data_pipeline:
	$(PYTHON) -m src.data_pipeline.loader