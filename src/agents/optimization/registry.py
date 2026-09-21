"""Реестр кандидатных управляющих воздействий (action space) — из исходного optim_agent.

Помечены в справочнике КИП как «управляемая»/«берём». bound_pct — допустимый общий
диапазон от опорного значения (МОДЕЛЬНОЕ допущение, не регламент, см. «правило границ» ТЗ п.4);
delta_pct — макс. шаг за один цикл (ограничение скорости изменения режима).

Отличия от исходника (только привязка к данным):
* добавлено поле installation: имя тега не уникально между установками, а колонка витрины
  получает суффикс _avt/_hdt только у тегов, общих для обеих установок — реальная колонка
  определяется по фактическим колонкам (resolve_columns);
* ключ реестра "F14_avt" сохранён: "F14" зарезервирован под контекст 24-2000 (формулы ВАК);
* для температур delta_pct снижен с 5% до 1% (5% от ~340 °C = 17 °C за цикл — недопустимо резко).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from src.data_pipeline.feature_config import resolve_column_name


@dataclass
class ControlVar:
    tag: str                          # ключ реестра
    description: str
    unit: str
    installation: str = "242000"      # "avt" | "242000"
    source_tag: str | None = None     # реальный tag_id в данных (по умолчанию = tag)
    delta_pct: float = 0.05           # допустимый шаг изменения за один цикл
    bound_pct: float = 0.15           # допустимый общий диапазон от опорного значения
    is_hard_bounded: bool = False     # True, если bounds подтверждены техрегламентом
    hard_bounds: tuple[float, float] | None = None

    @property
    def tag_id(self) -> str:
        return self.source_tag or self.tag

    @property
    def suffix(self) -> str:
        return "avt" if self.installation == "avt" else "hdt"


CONTROL_REGISTRY: dict[str, ControlVar] = {
    # --- АВТ, колонна К-1/К-2 ---
    "F65": ControlVar("F65", "Производительность К-2 по отбензиненной нефти (загрузка АВТ)", "т/ч",
                      installation="avt", bound_pct=0.10),
    "F30": ControlVar("F30", "Расход фракции 290-350°C с установки (точка отбора ДТ)", "т/ч", installation="avt"),
    "F32": ControlVar("F32", "Расход фракции 240-290°C с установки", "т/ч", installation="avt"),
    "F34": ControlVar("F34", "Расход фракции 150-250°C с установки (керосин)", "т/ч", installation="avt"),
    "T33": ControlVar("T33", "Температура низа К-2", "°C", installation="avt", delta_pct=0.01, bound_pct=0.05),
    "F12": ControlVar("F12", "Расход 2-го ЦО в К-2 (циркуляционное орошение)", "т/ч", installation="avt"),
    "F14_avt": ControlVar("F14_avt", "Расход 1-го ЦО в К-2 (тег F14 в avt_tags.csv)", "т/ч",
                          installation="avt", source_tag="F14"),
    "F64": ControlVar("F64", "Расход 3-го ЦО в К-2", "т/ч", installation="avt"),
    # --- Гидроочистка 24-2000 ---
    "F26": ControlVar("F26", "Расход сырья на установку гидроочистки (объёмный)", "м3/ч", bound_pct=0.10),
    "P24": ControlVar("P24", "Расход свежего ВСГ с КЦА (соотношение газ/сырьё)", "нм3/ч", bound_pct=0.20),
    "T16": ControlVar("T16", "Температура стабильного гидрогенизата, низ К-201", "°C", delta_pct=0.01, bound_pct=0.05),
    "T18": ControlVar("T18", "Давление на выходе К-201", "МПа", bound_pct=0.05),
}


def resolve_columns(columns: Iterable[str], registry: Mapping[str, ControlVar] = CONTROL_REGISTRY) -> dict[str, str]:
    """Ключ реестра -> реальная колонка витрины (только для тегов, присутствующих в данных)."""
    cols = list(columns)
    out: dict[str, str] = {}
    for key, cv in registry.items():
        try:
            out[key] = resolve_column_name(f"{cv.tag_id}_{cv.suffix}", cols)
        except KeyError:
            continue
    return out


def build_reference(history: pd.DataFrame, registry: Mapping[str, ControlVar] = CONTROL_REGISTRY) -> dict[str, float]:
    """Опорные значения для bound_pct: медиана истории рабочих периодов (is_valid).
    Передавайте историю ДО момента прогона — иначе диапазоны «знают» будущее."""
    hist = history[history["is_valid"].fillna(False)] if "is_valid" in history.columns else history
    ref: dict[str, float] = {}
    for key, col in resolve_columns(history.columns, registry).items():
        s = hist[col].dropna()
        if len(s) >= 200:
            ref[key] = float(s.median())
    return ref


def tag_directory(columns: Iterable[str], registry: Mapping[str, ControlVar] = CONTROL_REGISTRY) -> dict[str, tuple[str, str]]:
    """колонка -> (человекочитаемое имя, единица) для UI."""
    return {col: (registry[k].description, registry[k].unit) for k, col in resolve_columns(columns, registry).items()}