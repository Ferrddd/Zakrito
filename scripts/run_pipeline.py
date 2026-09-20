"""Подготовка данных: телеметрия + ПАК (+ ЛИМС) -> data/converted/telemetry_pac_lims.parquet.

Признаки, таргет и сплиты строит src/agents/quality/train.py.
Если сырые данные изменились — после этого запускай train с --force-rebuild.
"""
import logging

from src.data_pipeline.loader import DataPreparer
from src.data_pipeline.pipeline_models import data_pipeline_settings as s

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Показатели ЛИМС для приклейки к телеметрии: {имя колонки: имя показателя в ЛИМС}
LIMS_INDICATORS: dict[str, str] = {}


def main() -> None:
    df = DataPreparer(s).prepare_data(lims_indicators=LIMS_INDICATORS or None, save=True)
    logger.info("Готово: %d строк, %d колонок", len(df), df.shape[1])


if __name__ == "__main__":
    main()