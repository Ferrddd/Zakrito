# Как запускать

## A. Одной командой (Docker) — финальная отправка
```bash
cp .env.example .env            # при желании поменять START/STEPS/SLEEP
# сырые данные -> ./data/raw   (или готовые ./data/converted/telemetry_pac_lims.parquet + models/)
docker compose up --build       # или: make docker-up
```
Порядок: `bootstrap` (витрина → обучение, пропускается если артефакты есть) → `quality-api` (:8001/docs) → `ui` (:3000) → `orchestrator` (проигрывает историю, шлёт циклы в UI).
Для отправки жюри лучше положить в архив готовые `data/converted/models` и `telemetry_pac_lims.parquet` — тогда старт за секунды, без обучения.

## B. Локально (разработка)
```bash
make install && make copy_env
make data_pipeline && make train_quality && make train_reliability
make quality_api                                   # терминал 1
make ui                                            # терминал 2
make orchestrator_ui START=2026-06-01 STEPS=50 EVERY=6   # терминал 3
make test && make lint
```

## Перед отправкой (чек-лист)
1. `docker compose config` — без ошибок; `docker compose up --build` на чистой копии репозитория.
2. `ui/server.js`: порт 3000 и путь эндпоинта = `UI_URL` (сейчас допущение `/api/update`).
3. `tags_catalog` в quality/config.yaml (`markup/…`) и config_density.yaml (`config/…`) — теперь резолвится в обе папки, но файл должен существовать.
4. `requirements.txt` — сверить с вашим (сгенерирован по импортам).
5. Агент надёжности в репозитории должен быть версией, читающей `reliability/config.yaml` (NBM, тег W10). Версия, которую я видел (`agent.py`), использует F19 и NBM не грузит — если это актуальный файл, обученная модель не участвует в работе.
6. `make diagnose_reliability` ссылается на `src.agents.reliability.diagnose` — модуля нет среди файлов; уберите цель или добавьте модуль.
7. Не коммитьте `data/` и `.env`.

## Что улучшено
- Docker: Dockerfile, Dockerfile.ui, docker-compose.yml, .dockerignore, bootstrap.sh (идемпотентно), healthcheck.
- `RemoteQualityAgent.warm_up` через `/history` (было 288 полных evaluate по HTTP).
- `run_orchestrator --sleep` — плавная демонстрация в UI.
- `tags_catalog`: устойчив к `config/` vs `markup/`.
- Makefile: убраны дубли в help, добавлены docker-цели.
