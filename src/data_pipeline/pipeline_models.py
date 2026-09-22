from pathlib import Path
 
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
 
BASE_DIR = Path(__file__).resolve().parent.parent.parent
 
 
class DataPipelineSettings(BaseSettings):

 
    raw_data_path: Path = Path("data/raw")
    converted_data_path: Path = Path("data/converted")
 
    name_avt: str = "avt_tags.csv"
    name_242000: str = "242000_tags.csv"
    name_lims: str = "ЛИМСы 01.01.2023 - н.в_ (2).xlsx"
    name_pac: str = "Выгрузка ПАК 01.01.2023 - н.в_.xlsx"
 
    # --- пороги детекторов -------------------------------------------------
    # доля каналов со |robust z| > z_threshold, выше которой строка считается
    # аномальной (не выбрасывается, а помечается)
    anomaly_threshold: float = 0.15
    z_threshold: float = 5.0
 
    # теги массового расхода сырья: простой определяется по ним, а не по z-score
    load_tags: tuple[str, ...] = ("T11_hdt", "F26_hdt")
    # доля от медианного расхода, ниже которой установка считается остановленной
    downtime_load_ratio: float = 0.10
    # минимальная длительность простоя, чтобы не ловить одиночные провалы КИП
    downtime_min_points: int = 6
 
    # ПАК считается замороженным, если значение не менялось >= N точек (6 = 1 час)
    pac_frozen_min_points: int = 6
    # физически допустимые диапазоны поточных анализаторов
    pac_ranges: dict[str, tuple[float, float]] = {
        "24-2000:Mg.Sulfur": (0.5, 100.0),
        "24-2000:D15": (750.0, 950.0),
    }
 
    grid_freq: str = "10min"
 
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
 
    @model_validator(mode="after")
    def _resolve_paths(self) -> "DataPipelineSettings":
        if not self.raw_data_path.is_absolute():
            self.raw_data_path = BASE_DIR / self.raw_data_path
        if not self.converted_data_path.is_absolute():
            self.converted_data_path = BASE_DIR / self.converted_data_path
        self.converted_data_path.mkdir(parents=True, exist_ok=True)
        return self
 
    # производные пути считаются от актуального raw_data_path, а не один раз
    # при определении класса
    @property
    def file_avt(self) -> Path:
        return self.raw_data_path / self.name_avt
 
    @property
    def file_242000(self) -> Path:
        return self.raw_data_path / self.name_242000
 
    @property
    def file_lims(self) -> Path:
        return self.raw_data_path / self.name_lims
 
    @property
    def file_pac(self) -> Path:
        return self.raw_data_path / self.name_pac
 

data_pipeline_settings = DataPipelineSettings()
