from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Data_Pipeline_Settings(BaseSettings):
    raw_data_path: Path = Path("data/raw")
    converted_data_path: Path = Path("data/converted")

    file_avt: Path = Path(f"{raw_data_path}/avt_tags.csv")
    file_242000: Path = Path(f"{raw_data_path}/242000_tags.csv")
    file_lims: Path = Path(f"{raw_data_path}/ЛИМСы 01.01.2023 - н.в_ (2).xlsx")

    anomaly_threshold: float

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

data_pipeline_settings = Data_Pipeline_Settings()