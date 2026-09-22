# Как запускать

Жюри запускает на **Windows 11**. Рекомендуемый и единственный полностью проверенный на Windows путь —
**вариант A (Docker)**: он не зависит от Make/bash/Python-окружения хоста. Вариант B (локально, без
Docker) описан для разработки и требует WSL2 или Git Bash — на «голом» PowerShell/cmd `make`-цели работать
не будут (см. примечание в конце раздела B).

## A. Одной командой (Docker) — рекомендуется для Windows 11

Предварительно: **Docker Desktop** с включённым **WSL2**-бэкендом (Settings → General → *Use the WSL 2
based engine*; ставится автоматически при установке Docker Desktop на Windows 11). Команды ниже
выполняются в PowerShell из корня проекта.

```powershell
Copy-Item .env.example .env      # при желании поменять START/STEPS/SLEEP
# сырые данные -> .\data\raw   (или готовые .\data\converted\telemetry_pac_lims.parquet + models\)
docker compose up --build
```

(Если удобнее через `make` — он тоже работает в PowerShell, если Make установлен, например через
`winget install GnuWin32.Make` или Chocolatey: `make docker-up`.)

Порядок: `bootstrap` (витрина → обучение, пропускается если артефакты есть) → `quality-api` (:8001/docs)
→ `ui` (:3000) → `orchestrator` (проигрывает историю, шлёт циклы в UI). Открыть в браузере:
`http://localhost:3000`.

Для отправки жюри лучше положить в архив готовые `data/converted/models` и `telemetry_pac_lims.parquet` —
тогда старт занимает секунды, без обучения на месте.

Если порты 3000/8001 заняты другим процессом на машине жюри — поменяйте проброс портов в
`docker-compose.yml` или остановите конфликтующую службу перед запуском.

## B. Локально (разработка), без Docker

Использует `make`, что на Windows означает один из двух вариантов:

- **WSL2 (рекомендуется)** — открыть Ubuntu-терминал WSL2, дальше всё как на Linux, команды ниже
  выполняются как есть;
- **Git Bash** — большинство целей отработает, но проверьте, что установлен `make` (в поставке Git for
  Windows его нет по умолчанию — нужно доставить отдельно, например через MSYS2/Chocolatey).

```bash
make install && make copy_env
make data_pipeline && make train_quality && make train_reliability
make quality_api                                   # терминал 1
make ui                                             # терминал 2
make orchestrator_ui START=2026-06-01 STEPS=50 EVERY=6   # терминал 3
make test && make lint
```

Нативный PowerShell/cmd без WSL2 и без Git Bash **не поддерживается** для этого варианта: `make install`
вызывает `python3.12` (на Windows обычно `python` или `py -3.12`), а `make ui` запускает фоновый процесс
через bash-синтаксис `&`, который PowerShell интерпретирует иначе. Если WSL2/Git Bash недоступны —
используйте вариант A (Docker).

## Перед отправкой (чек-лист)
1. `docker compose config` — без ошибок; `docker compose up --build` на чистой копии репозитория
   (в идеале — проверить именно на Windows 11 + Docker Desktop, а не только на Linux/macOS).
2. `ui/server.js`: порт 3000 и путь эндпоинта = `UI_URL` (сейчас допущение `/api/update`).
3. `tags_catalog` в quality/config.yaml (`markup/…`) и config_density.yaml (`config/…`) — резолвится в обе
   папки, но файл должен существовать.
4. `requirements.txt` — сверить с окружением, на котором собирали (сгенерирован по импортам).
5. Агент надёжности в репозитории должен быть версией, читающей `reliability/config.yaml` (NBM, тег W10).
   Версия, которую я видел (`agent.py`), использует F19 и NBM не загружает — если это актуальный файл,
   обученная модель не участвует в работе. Проверить перед отправкой.
6. `make diagnose_reliability` ссылается на `src.agents.reliability.diagnose` — свериться, что модуль
   есть в финальном срезе репозитория.
7. Пути внутри репозитория и в Docker-образах должны быть кросс-платформенными (без хардкода `/` там, где
   это может исполняться вне контейнера на Windows) — при использовании варианта A это не критично,
   т.к. всё выполняется внутри Linux-контейнеров, но скрипты, которые жюри могут запустить руками на
   хосте (не в Docker), должны учитывать Windows-пути.
8. Не коммитьте `data/` и `.env`.

## Что улучшено
- Docker: Dockerfile, Dockerfile.ui, docker-compose.yml, .dockerignore, bootstrap.sh (идемпотентно),
  healthcheck — основной путь запуска не зависит от ОС жюри.
- `RemoteQualityAgent.warm_up` через `/history` (было 288 полных evaluate по HTTP).
- `run_orchestrator --sleep` — плавная демонстрация в UI.
- `tags_catalog`: устойчив к `config/` vs `markup/`.
- Makefile: убраны дубли в help, добавлены docker-цели.