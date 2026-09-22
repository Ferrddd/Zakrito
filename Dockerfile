FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 — нужен LightGBM; curl — для healthcheck/отладки
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# код (data/, .venv, ui/node_modules исключены через .dockerignore)
COPY . .

# BASE_DIR в проекте = корень репозитория => /app; данные приходят томом ./data:/app/data
CMD ["python", "-m", "scripts.run_orchestrator", "--help"]