#!/usr/bin/env bash
# Идемпотентная подготовка: витрина -> модели. Уже готовое не пересобирается.
set -euo pipefail
cd /app
CONV=data/converted
PARQ=$CONV/telemetry_pac_lims.parquet
HORIZON="${QUALITY_HORIZON-6}"

if [ ! -f "$PARQ" ]; then
  if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "ОШИБКА: нет $PARQ и пусто data/raw — положите сырые файлы в ./data/raw (или готовый parquet в ./data/converted)" >&2
    exit 1
  fi
  echo ">>> Сборка витрины"
  python -m scripts.run_pipeline
fi

if ! ls $CONV/models/h*_blind/threshold.json >/dev/null 2>&1; then
  echo ">>> Обучение агента качества (горизонт: ${HORIZON:-все})"
  if [ -n "$HORIZON" ]; then
    python -m src.agents.quality.train --horizon "$HORIZON" --skip-cv --force-rebuild
  else
    python -m src.agents.quality.train --skip-cv --force-rebuild
  fi
else
  echo ">>> Модели качества найдены — обучение пропущено"
fi

if [ "${TRAIN_RELIABILITY:-1}" = "1" ] && [ ! -f $CONV/models/reliability_nbm.joblib ]; then
  echo ">>> Обучение NBM надёжности (best-effort)"
  python -m src.agents.reliability.train || echo "WARN: NBM не обучена — агент надёжности работает по правилам"
fi
echo ">>> bootstrap готов"
