from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Data_Pipeline_Settings(BaseSettings):
    raw_data_path: str
    converted_data_path: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

data_pipeline_settings = Data_Pipeline_Settings()