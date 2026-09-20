from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. Снапшот системы
# ------------------------------------------------------------------

class DataSourceStatus(str, Enum):
    ok = "ok"
    stale = "stale"
    down = "down"


class DataFreshness(BaseModel):
    lims_age_hours: float | None = Field(
        None, description="Сколько часов прошло с последнего ЛИМС-измерения"
    )
    pac_age_hours: float | None = Field(
        None, description="Сколько часов прошло с последнего ПАК-измерения"
    )
    pac_status: DataSourceStatus = Field(
        DataSourceStatus.ok, description="Статус потокового анализатора: работает / устарел / не отвечает"
    )


class ProcessSnapshot(BaseModel):
    timestamp: datetime = Field(
        ..., description="Момент времени, на который зафиксировано состояние процесса"
    )
    telemetry: dict[str, float] = Field(
        ..., description="Текущие значения всех тегов телеметрии, напр. {'T5': 342.1, 'F1': 12.4, ...}"
    )
    data_freshness: DataFreshness = Field(
        ..., description="Возраст и статус источников качества (ЛИМС/ПАК) на этот момент"
    )

# ------------------------------------------------------------------
# 2. Оценка агента качества
# ------------------------------------------------------------------

class QualitySource(str, Enum):
    lims = "LIMS"
    pac = "PAC"
    vak = "VAK"
    none = "none"

class QualityAssessment(BaseModel):
    timestamp: datetime = Field(..., description="Момент, для которого сделана оценка")

    quality_forecast: dict[str, float] = Field(
        ..., description="Прогноз показателей качества, напр. {'sulfur_ppm': 8.4, 'D15': 831.5, ...}"
    )
    risk_exceed_spec: dict[str, float] = Field(
        ..., description="Риск/вероятность превышения спецификации по каждому показателю. {'sulfur_ppm': 0.1, 'D15': 0.32, ...}"
    )
    confidence: float = Field(..., description="Доверие к прогнозу в целом")

    #Нам необходимо уведомлять о просрочных исследованиях, но наверное оптимизатору это не нужно
    target_source_used: QualitySource = Field(..., description="Какой источник лёг в основу актуального значения (не прогноза)")
    source_age_hours: float | None = Field(None, description="Возраст измерения, использованного как последний факт")
    stale_data_warning: bool = False




# ------------------------------------------------------------------
# 3. Оценка агента надежности
# ------------------------------------------------------------------

class RiskClass(str, Enum):
    normal = "normal"
    warning = "warning"
    critical = "critical"


class ReliabilityAssessment(BaseModel):
    timestamp: datetime

    # Рассчитанный индекс тяжести текущего режима работы
    severity_index: float = Field(..., description="Свёрнутый индекс тяжести режима")

    #Компоненты из которы считался severity_index
    dp_trend_slope: float | None = Field(None, description="Тренд ΔP реактора — сигнал закоксовывания")
    downtime_score: float | None = Field(None, description="Доля тегов вне robust z-порога")

    risk_class: RiskClass
    downtime_flag: bool = False

    limiting_factors: list[str] = Field(
        default_factory=list,
        description="Человекочитаемые причины оценки, напр. ['dp_trend_slope растёт 5ч подряд', 'W7 аномален']"
    )

# ------------------------------------------------------------------
# 4. Вход агента оптимизации
# ------------------------------------------------------------------

class QualitySignal(BaseModel):
    """То немногое из QualityAssessment, что реально нужно оптимизатору для решения."""
    quality_forecast: dict[str, float]
    risk_exceed_spec: dict[str, float]
    confidence: float


class ReliabilitySignal(BaseModel):
    """То немногое из ReliabilityAssessment, что реально нужно оптимизатору."""
    severity_index: float
    risk_class: RiskClass


class OptimizationInput(BaseModel):
    snapshot: ProcessSnapshot
    quality: QualitySignal
    reliability: ReliabilitySignal