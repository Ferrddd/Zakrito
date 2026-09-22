

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from src.data_pipeline.feature_config import resolve_column_name


def vak_godt_T90(x: dict[str, float]) -> float:
    return (162.998 + 0.12945 * x["T12"] + 59.57 * x["F15"] + 0.00036 * x["W7"]
            + x["T23"] * 0.26366 - 424.72638 * x["F1"] / x["F26"])


def vak_godt_T50(x: dict[str, float]) -> float:
    return 44.625 + 10.0224 * x["P13"] + 0.06981 * x["F9"] + 0.8052 * x["T6"]


def vak_godt_I250(x: dict[str, float]) -> float:
    return (84.585 - 0.21172 * x["T5"] + 0.12137 * x["T11"] - 0.00014 * x["F25"]
            + 0.56248 * x["F14"] - 0.16317 * x["T23"] + 0.20272 * x["T16"])


def vak_godt_D15(x: dict[str, float]) -> float:
    # использует лаговое лабораторное значение (ЛИМС) — без него не считается
    return 667.881 + 0.15417 * x["LIMS_D15"] + 0.00005 * x["F22"] + 0.10774 * x["T11"]


def vak_godt_cloud_point(x: dict[str, float]) -> float:
    return (x["F22"] + 0.0021 * x["W7"] + 0.00008 * x["F25"] - 0.30656 * x["F1"]
            + 0.12018 * x["T6"] + 0.01916 * x["F9"] - 48.254 - 0.05249 * x["T16"])


def vak_godt_T95(x: dict[str, float]) -> float:
    # использует лаговое лабораторное значение (ЛИМС) — без него не считается
    return 0.03814 * x["F9"] - 9.201 - 0.00002 * x["F2"] + 0.62259 * x["T6"] + 0.48321 * x["LIMS_T95"]


def vak_godt_CFPP(x: dict[str, float]) -> float:
    return (0.22088 * x["T6"] - 102.375 - 47.75834 * x["P8"] + 0.03862 * x["F9"]
            + 43.60207 * x["W7"] + 43.81849 * x["P24"])


def vak_godt_IBP(x: dict[str, float]) -> float:
    return (137.762 - 0.0653 * x["F26"] + 0.00011 * x["F22"] + 5.78137 * x["P13"]
            - 34.58028 * x["P24"] - 0.00993 * x["F14"] - 0.99962 * x["W4"]
            + 0.32232 * x["T23"] - 0.09406 * x["T16"])


VAK_GODT_MODELS: dict[str, Callable[[dict[str, float]], float]] = {
    "T90": vak_godt_T90,
    "T50": vak_godt_T50,
    "I250": vak_godt_I250,
    "D15": vak_godt_D15,
    "CloudPoint": vak_godt_cloud_point,
    "T95": vak_godt_T95,
    "CFPP": vak_godt_CFPP,
    "IBP": vak_godt_IBP,
}

# Теги 24-2000, нужные формулам (без LIMS_*).
VAK_TAGS: tuple[str, ...] = (
    "T12", "F15", "W7", "T23", "F1", "F26", "P13", "F9", "T6", "T5", "T11", "F25",
    "F22", "P8", "W4", "F2", "F14", "P24", "T16",
)


def hdt_column(columns: Iterable[str], tag: str) -> str | None:
    """Тег 24-2000 -> реальное имя колонки витрины ('T5' или 'T5_hdt'), None если нет."""
    try:
        return resolve_column_name(f"{tag}_hdt", columns)
    except KeyError:
        return None


def vak_inputs(tags: Mapping[str, float | None]) -> dict[str, float]:
    """Значения тегов из состояния -> вход формул ВАК (только то, что есть)."""
    cols = [k for k, v in tags.items() if v is not None]
    x: dict[str, float] = {}
    for name in VAK_TAGS:
        col = hdt_column(cols, name)
        if col is not None:
            x[name] = float(tags[col])  # type: ignore[arg-type]
    return x


def vak_predict(tags: Mapping[str, float | None]) -> dict[str, float]:
    """Прогноз ВАК по состоянию. Показатели, для которых не хватает тегов
    (или деление на 0), молча пропускаются — вызывающий сравнивает только общие."""
    x = vak_inputs(tags)
    out: dict[str, float] = {}
    for name, model in VAK_GODT_MODELS.items():
        try:
            out[name] = model(x)
        except (KeyError, ZeroDivisionError):
            continue
    return out


def vak_predict_state(state: Mapping[str, float]) -> dict[str, float]:
    """Прогноз ВАК по словарю с ИМЕНАМИ ТЕГОВ формул (T12, F15, ..., LIMS_D15). Показатели, для которых
    не хватает тегов, пропускаются (агент должен трактовать это как «недостаточно данных»)."""
    x = dict(state)
    out: dict[str, float] = {}
    for name, model in VAK_GODT_MODELS.items():
        try:
            out[name] = model(x)
        except (KeyError, ZeroDivisionError):
            continue
    return out