"""
Агент оптимизации (Optimization Agent) для МАС управления цепочкой
АВТ -> гидроочистка (24-2000) -> блендинг.

Роль в архитектуре (см. ТЗ, раздел 3):
    Вход:  текущее состояние процесса (значения управляемых тегов),
           прогнозы/ограничения от агента качества и агента надёжности.
    Выход: набор допустимых сценариев с метриками + Парето-фронт +
           рекомендация оператору (или явный отказ, если решения нет).

ВАЖНО (допущения, которые нужно проверить/скорректировать под реальные
данные и P&ID перед защитой):
  1. Регрессионные формулы качества взяты из листа "ВАК" файла
     Теги_хакатон.xlsx для узла 24-2000 (гидроочищенный дизель, ГО ДТ).
     Это line-fit модели ("виртуальные анализаторы"), а не физические
     первые принципы — использовать как ОЦЕНКУ, не как истину в
     последней инстанции. Приоритет достоверности по ТЗ: ЛИМС > ПАК > ВАК.
  2. Некоторые имена тегов (например T6) встречаются в справочнике КИП
     как для АВТ, так и для 24-2000 с РАЗНЫМ физическим смыслом.
     В формулах ниже используется 24-2000-контекст (как в листе ВАК).
     Перед использованием на реальных данных сверить с P&ID блока
     гидроочистки/стабилизации.
  3. Границы (bounds) управляемых параметров ниже — МОДЕЛЬНЫЕ допущения
     (± консервативный процент от текущего значения), а НЕ подтверждённые
     промышленные пределы (см. "Правило границ" в ТЗ). Заменить на
     согласованные технологические диапазоны, если они появятся.
  4. Расход/содержание серы не моделируется формулой ВАК (в листе её нет) —
     оно должно приходить как измерение/прогноз от агента качества
     (источник ЛИМС/ПАК), здесь используется как внешний вход quality_forecast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime as _dt
from itertools import product
from typing import Any

# ---------------------------------------------------------------------------
# 1. Управляемые параметры (action space)
# ---------------------------------------------------------------------------
# Помечены в справочнике КИП как "управляемая"/"берём". Bounds — модельное
# допущение (см. пункт 3 выше), delta_pct — на сколько % от текущего можно
# сдвинуть параметр за один шаг оптимизации (ограничение скорости изменения
# режима, чтобы не "прыгать" по технологическому процессу).

@dataclass
class ControlVar:
    tag: str
    description: str
    unit: str
    delta_pct: float = 0.05          # допустимый шаг изменения за один цикл
    bound_pct: float = 0.15          # допустимый общий диапазон от текущего значения
    is_hard_bounded: bool = False    # True, если bounds подтверждены техрегламентом
    hard_bounds: tuple[float, float] | None = None


# Реестр кандидатных управляющих воздействий (сведён из листа КИП).
CONTROL_REGISTRY: dict[str, ControlVar] = {
    # --- АВТ, колонна К-1/К-2 ---
    "F65": ControlVar("F65", "Производительность К-2 по отбензиненной нефти (загрузка АВТ)", "т/ч", bound_pct=0.10),
    "F30": ControlVar("F30", "Расход фракции 290-350°C с установки (точка отбора ДТ)", "т/ч"),
    "F32": ControlVar("F32", "Расход фракции 240-290°C с установки", "т/ч"),
    "F34": ControlVar("F34", "Расход фракции 150-250°C с установки (керосин)", "т/ч"),
    "T33": ControlVar("T33", "Температура низа К-2", "°C", bound_pct=0.05),
    "F12": ControlVar("F12", "Расход 2-го ЦО в К-2 (циркуляционное орошение)", "т/ч"),
    # ВНИМАНИЕ: имя тега "F14" встречается ДВАЖДЫ со своим смыслом в разных
    # установках — F14 в avt_tags.csv (АВТ, расход 1-го ЦО в К-2) и F14 в
    # 242000_tags.csv (используется как контекстный признак в формулах ВАК).
    # Чтобы не путать их в коде, тег АВТ хранится в state под ключом
    # "F14_avt", а "F14" зарезервирован под контекст 24-2000 (см. ниже).
    "F14_avt": ControlVar("F14_avt", "Расход 1-го ЦО в К-2 (тег F14 в avt_tags.csv)", "т/ч"),
    "F64": ControlVar("F64", "Расход 3-го ЦО в К-2", "т/ч"),
    # --- Гидроочистка 24-2000 ---
    "F26": ControlVar("F26", "Расход сырья на установку гидроочистки (объёмный)", "м3/ч", bound_pct=0.10),
    "P24": ControlVar("P24", "Расход свежего ВСГ с КЦА (соотношение газ/сырьё)", "нм3/ч", bound_pct=0.20),
    "T16": ControlVar("T16", "Температура стабильного гидрогенизата, низ К-201", "°C", bound_pct=0.05),
    "T18": ControlVar("T18", "Давление на выходе К-201", "МПа", bound_pct=0.05),
}


# ---------------------------------------------------------------------------
# 2. Суррогатные модели качества (лист "ВАК", узел 24-2000 -> ГО ДТ)
# ---------------------------------------------------------------------------
# Каждая функция принимает словарь текущих значений тегов (control + context)
# и возвращает прогнозное значение показателя. Формулы скопированы из ВАК
# без изменений (только синтаксис Python).

def vak_godt_T90(x: dict[str, float]) -> float:
    return (162.998 + 0.12945 * x["T12"] + 59.57 * x["F15"] + 0.00036 * x["W7"]
            + x["T23"] * 0.26366 - 424.72638 * x["F1"] / x["F26"])

def vak_godt_T50(x: dict[str, float]) -> float:
    return 44.625 + 10.0224 * x["P13"] + 0.06981 * x["F9"] + 0.8052 * x["T6"]

def vak_godt_I250(x: dict[str, float]) -> float:
    return (84.585 - 0.21172 * x["T5"] + 0.12137 * x["T11"] - 0.00014 * x["F25"]
            + 0.56248 * x["F14"] - 0.16317 * x["T23"] + 0.20272 * x["T16"])

def vak_godt_D15(x: dict[str, float]) -> float:
    # использует лаговое лабораторное значение как признак (запаздывание ЛИМС)
    return 667.881 + 0.15417 * x["LIMS_D15"] + 0.00005 * x["F22"] + 0.10774 * x["T11"]

def vak_godt_cloud_point(x: dict[str, float]) -> float:
    return (x["F22"] + 0.0021 * x["W7"] + 0.00008 * x["F25"] - 0.30656 * x["F1"]
            + 0.12018 * x["T6"] + 0.01916 * x["F9"] - 48.254 - 0.05249 * x["T16"])

def vak_godt_T95(x: dict[str, float]) -> float:
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
    "CloudPoint": vak_godt_CFPP if False else vak_godt_cloud_point,  # (см. примечание)
    "T95": vak_godt_T95,
    "CFPP": vak_godt_CFPP,
    "IBP": vak_godt_IBP,
}


# ---------------------------------------------------------------------------
# 3. Спецификация / жёсткие ограничения (минимум по ТЗ, раздел 4)
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    sulfur_max_ppm: float = 10.0
    cfpp_max_c: float | None = None          # подставить из выданной спецификации
    cloud_point_max_c: float | None = None   # подставить из выданной спецификации
    t90_max_c: float | None = None
    blend_fraction_sum_tol: float = 1e-3        # сумма долей блендинга = 100% ± tol


# ---------------------------------------------------------------------------
# 4. Сценарий и его оценка
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    deltas: dict[str, float]                # {tag: новое значение управляемого параметра}
    predicted_quality: dict[str, float] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)   # цели для Парето (минимизация)
    feasible: bool = True
    violations: list[str] = field(default_factory=list)


@dataclass
class OptimizationInput:
    current_state: dict[str, float]                     # текущие значения ВСЕХ тегов (control+context)
    quality_forecast: dict[str, dict[str, Any]]         # от агента качества: {"sulfur_ppm": {"pred":.., "confidence":..}, ...}
    reliability_assessment: dict[str, Any]             # от агента надёжности: {"risk_level":.., "hard_constraints": {...}}
    spec: Spec
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {"quality_risk": 0.4, "throughput": 0.2, "energy": 0.2, "reliability_risk": 0.2}
    )


# ---------------------------------------------------------------------------
# 5. Генерация сценариев
# ---------------------------------------------------------------------------

def generate_scenarios(opt_input: OptimizationInput, controls: list[str] | None = None,
                        n_steps: int = 3) -> list[Scenario]:
    """
    Строит сетку сценариев: для каждого управляемого тега берём n_steps точек
    в пределах допустимого шага delta_pct (не всего диапазона bound_pct —
    это ограничение СКОРОСТИ изменения режима за один цикл принятия решения).
    Полный перебор комбинаций может быть большим — на практике для
    прототипа достаточно варьировать 2-3 ключевых тега одновременно,
    остальные оставлять на текущем значении (частичные сценарии),
    плюс несколько "комбинированных" сценариев вручную.
    """
    controls = controls or list(CONTROL_REGISTRY.keys())
    state = opt_input.current_state

    per_tag_options: dict[str, list[float]] = {}
    for tag in controls:
        cv = CONTROL_REGISTRY[tag]
        current = state.get(tag)
        if current is None:
            continue
        step = current * cv.delta_pct
        options = [current - step, current, current + step]
        if cv.is_hard_bounded and cv.hard_bounds:
            lo, hi = cv.hard_bounds
            options = [max(lo, min(hi, v)) for v in options]
        per_tag_options[tag] = sorted(set(options))

    scenarios: list[Scenario] = []
    # 1) базовый сценарий "ничего не менять" (обязателен — устойчивый режим)
    scenarios.append(Scenario(deltas={tag: state[tag] for tag in per_tag_options}))

    # 2) одиночные изменения по одному тегу (проще объяснить оператору)
    for tag, options in per_tag_options.items():
        for val in options:
            if val == state[tag]:
                continue
            d = {t: state[t] for t in per_tag_options}
            d[tag] = val
            scenarios.append(Scenario(deltas=d))

    # 3) ограниченное число парных комбинаций (не полный грид, чтобы не взорваться комбинаторно)
    tags = list(per_tag_options.keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = tags[i], tags[j]
            for v1, v2 in product(per_tag_options[t1], per_tag_options[t2]):
                if v1 == state[t1] and v2 == state[t2]:
                    continue
                d = {t: state[t] for t in per_tag_options}
                d[t1], d[t2] = v1, v2
                scenarios.append(Scenario(deltas=d))

    return scenarios


# ---------------------------------------------------------------------------
# 6. Оценка сценария: прогноз качества, проверка ограничений, целевые функции
# ---------------------------------------------------------------------------

def evaluate_scenario(scn: Scenario, opt_input: OptimizationInput) -> Scenario:
    state = dict(opt_input.current_state)
    state.update(scn.deltas)

    # --- прогноз качества по ВАК (там, где есть все нужные признаки) ---
    for name, model in VAK_GODT_MODELS.items():
        try:
            scn.predicted_quality[name] = model(state)
        except KeyError:
            # не хватает контекстных тегов в текущем состоянии — пропускаем,
            # это должно логироваться агентом как "недостаточно данных для этого показателя"
            continue

    violations: list[str] = []

    # --- сера: жёсткое ограничение, источник — прогноз агента качества (ЛИМС/ПАК), не ВАК ---
    sulfur_pred = opt_input.quality_forecast.get("sulfur_ppm", {}).get("pred")
    if sulfur_pred is not None and sulfur_pred > opt_input.spec.sulfur_max_ppm:
        violations.append(f"sulfur_ppm={sulfur_pred:.2f} > {opt_input.spec.sulfur_max_ppm}")

    # --- прочие показатели качества, если спецификация задана ---
    if (
        opt_input.spec.cfpp_max_c is not None and
        "CFPP" in scn.predicted_quality and
        scn.predicted_quality["CFPP"] > opt_input.spec.cfpp_max_c
    ):
            violations.append(f"CFPP={scn.predicted_quality['CFPP']:.1f} > {opt_input.spec.cfpp_max_c}")
    if (
        opt_input.spec.cloud_point_max_c is not None and 
        "CloudPoint" in scn.predicted_quality and
        scn.predicted_quality["CloudPoint"] > opt_input.spec.cloud_point_max_c
    ):
            violations.append(f"CloudPoint={scn.predicted_quality['CloudPoint']:.1f} > {opt_input.spec.cloud_point_max_c}")
    if (
        opt_input.spec.t90_max_c is not None and
        "T90" in scn.predicted_quality and
        scn.predicted_quality["T90"] > opt_input.spec.t90_max_c
    ):
            violations.append(f"T90={scn.predicted_quality['T90']:.1f} > {opt_input.spec.t90_max_c}")

    # --- жёсткие ограничения от агента надёжности (например, предельная загрузка) ---
    reliability_bounds = opt_input.reliability_assessment.get("hard_constraints", {})
    if isinstance(reliability_bounds, dict):
        for tag, limit in reliability_bounds.items():
            val = scn.deltas.get(tag)
            if val is None:
                continue
            if isinstance(limit, tuple):
                lo, hi = limit
                if not (lo <= val <= hi):
                    violations.append(f"{tag}={val:.2f} вне допустимого диапазона надёжности {limit}")
            elif isinstance(limit, (int, float)):  # трактуем как верхний предел
                if val > limit:
                    violations.append(f"{tag}={val:.2f} > предел надёжности {limit}")

    scn.violations = violations
    scn.feasible = len(violations) == 0

    # --- целевые функции (всё сводим к МИНИМИЗАЦИИ для унификации Парето) ---
    # 1) quality_risk: степень близости к границам спецификации (0 = далеко от предела, безопасно)
    quality_risk = 0.0
    if sulfur_pred is not None:
        quality_risk += max(0.0, sulfur_pred / opt_input.spec.sulfur_max_ppm - 0.7)  # риск растёт после 70% от предела
    if opt_input.spec.t90_max_c and "T90" in scn.predicted_quality:
        quality_risk += max(0.0, scn.predicted_quality["T90"] / opt_input.spec.t90_max_c - 0.9)

    # 2) throughput: хотим МАКСИМИЗИРОВАТЬ выпуск -> минимизируем "недопроизводство"
    base_f65 = opt_input.current_state.get("F65", 0.0)
    f65 = scn.deltas.get("F65", base_f65)
    throughput_loss = max(0.0, base_f65 - f65)  # чем сильнее срезали загрузку, тем хуже

    # 3) energy proxy: сумма относительных отклонений температурных уставок от текущих
    #    (упрощённый прокси, если реальных данных по энергозатратам нет)
    energy_proxy = 0.0
    for tag in ("T33", "T16", "T18"):
        if tag in scn.deltas and tag in opt_input.current_state:
            base = opt_input.current_state[tag]
            if base:
                energy_proxy += abs(scn.deltas[tag] - base) / abs(base)

    # 4) reliability_risk: берём напрямую из оценки агента надёжности (если дан численный risk_score)
    risk_val = opt_input.reliability_assessment.get("risk_score", 0.0)
    reliability_risk = float(risk_val) if isinstance(risk_val, (int, float, str)) else 0.0

    scn.objectives = {
        "quality_risk": quality_risk,
        "throughput_loss": throughput_loss,
        "energy_proxy": energy_proxy,
        "reliability_risk": reliability_risk,
    }
    return scn


# ---------------------------------------------------------------------------
# 7. Парето-фронт (non-dominated sorting)
# ---------------------------------------------------------------------------

def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """a доминирует b, если a не хуже по всем целям и строго лучше хотя бы по одной.
    Все цели трактуются как МИНИМИЗАЦИЯ (см. evaluate_scenario)."""
    keys = a.keys()
    not_worse = all(a[k] <= b[k] for k in keys)
    strictly_better = any(a[k] < b[k] for k in keys)
    return not_worse and strictly_better


def pareto_front(scenarios: list[Scenario]) -> list[Scenario]:
    """Возвращает недоминируемое множество среди ДОПУСТИМЫХ сценариев.
    Жёсткие ограничения отфильтрованы ДО построения фронта — это ключевое
    требование ТЗ: качество/безопасность нельзя компенсировать другими метриками."""
    feasible = [s for s in scenarios if s.feasible]
    front: list[Scenario] = []
    for s in feasible:
        if not any(dominates(other.objectives, s.objectives) for other in feasible if other is not s):
            front.append(s)
    return front


def pick_recommendation(front: list[Scenario], weights: dict[str, float]) -> Scenario | None:
    """Из Парето-фронта выбирает одно решение через взвешенную сумму
    (простой tie-break для оператора; сам фронт при этом показывается целиком,
    чтобы оператор мог выбрать альтернативу)."""
    if not front:
        return None

    def score(s: Scenario) -> float:
        return sum(weights.get(k, 0.0) * v for k, v in s.objectives.items())

    return min(front, key=score)


# ---------------------------------------------------------------------------
# 8. Публичный интерфейс агента
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    all_scenarios: list[Scenario]
    feasible_scenarios: list[Scenario]
    pareto_front: list[Scenario]
    recommendation: Scenario | None
    message: str


def run_optimization_agent(opt_input: OptimizationInput,
                            controls: list[str] | None = None) -> OptimizationResult:
    raw_scenarios = generate_scenarios(opt_input, controls=controls)
    evaluated = [evaluate_scenario(s, opt_input) for s in raw_scenarios]
    feasible = [s for s in evaluated if s.feasible]
    front = pareto_front(evaluated)
    rec = pick_recommendation(front, opt_input.objective_weights)

    if rec is None:
        message = ("Надёжной рекомендации нет: ни один из рассмотренных вариантов не "
                   "удовлетворяет жёстким ограничениям (качество/надёжность). "
                   "Требуется расширить пространство сценариев или снизить нагрузку установки.")
    else:
        message = "Сформирован Парето-фронт допустимых сценариев, рекомендация выбрана по взвешенному критерию."

    return OptimizationResult(
        all_scenarios=evaluated,
        feasible_scenarios=feasible,
        pareto_front=front,
        recommendation=rec,
        message=message,
    )


# ---------------------------------------------------------------------------
# 9. Загрузка РЕАЛЬНЫХ значений из выданных файлов (телеметрия / ЛИМС / ПАК)
# ---------------------------------------------------------------------------
# Теперь используются все четыре источника: avt_tags.csv, 242000_tags.csv,
# ЛИМС и ПАК. Синхронизация — ПО ВРЕМЕНИ (см. правило ТЗ): для выбранного
# момента лабораторного анализа берётся ближайшая по времени 10-минутная
# точка телеметрии (а не строка с тем же номером).
#
# Управляемые параметры F65/F30/F32/F34/T33/F12/F14/F64 берутся из
# avt_tags.csv (установка АВТ). Управляемые параметры F26/P24/T16/T18 и
# все контекстные признаки формул ВАК (T12, F15, W7, T23, F1, P13, F9, T6,
# T5, T11, F25, F22, P8, W4, F2) берутся из 242000_tags.csv — это тот же
# узел (гидроочистка/стабилизация), для которого построены формулы ВАК,
# поэтому коллизий имён тегов между установками здесь не возникает.

import pandas as pd  # используется только в загрузчиках телеметрии ниже


def load_latest_lims_godt(path: str) -> dict[str, tuple[_dt | None, float | None]]:
    """Читает лист ЛИМС и возвращает последние (по дате) значения показателей
    качества для точки 'Гидроочистка, точка отбора 2, Дизельное топливо'.
    Индексы колонок жёстко привязаны к структуре выданного файла
    (см. заголовки листа) — если структура файла изменится, пересчитать."""
    import openpyxl  # type: ignore[import-untyped]

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Лист1"]

    # (имя показателя ВАК/спецификации -> (индекс колонки с датой, индекс колонки со значением))
    targets = {
        "D15": (84, 85),
        "T95": (92, 93),
        "sulfur_mg_kg": (94, 95),
        "CFPP": (96, 97),
        "CloudPoint": (82, 83),
        "IBP": (88, 89),
        "I250": (90, 91),
        "T50": (100, 101),
        "T90": (104, 105),
    }

    result: dict[str, tuple[_dt | None, float | None]] = {}
    for name, (dc, vc) in targets.items():
        best_date, best_val = None, None
        for row in ws.iter_rows(min_row=4, values_only=True):
            d, v = row[dc], row[vc]
            if (
                isinstance(d, _dt) and v is not None and
                (best_date is None or d > best_date)
            ):
                best_date, best_val = d, v
        result[name] = (best_date, best_val)
    return result


def load_latest_pak_sulfur_d15(path: str) -> dict[str, tuple[_dt | None, float | None]]:
    """Читает лист ПАК (24-2000:Mg.Sulfur, 24-2000:D15) и возвращает
    последние по времени показания поточных анализаторов."""
    import openpyxl  # type: ignore[import-untyped]

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Лист1"]

    best_s_date, best_s_val = None, None
    best_d_date, best_d_val = None, None
    for row in ws.iter_rows(min_row=3, values_only=True):
        d0, v0 = row[0], row[1]
        d1, v1 = row[3], row[4]
        if (
            isinstance(d0, _dt) and
            v0 is not None and
            (best_s_date is None or d0 > best_s_date)
        ):
            best_s_date, best_s_val = d0, v0
        if (
            isinstance(d1, _dt) and
            v1 is not None and
            (best_d_date is None or d1 > best_d_date)
        ):
            best_d_date, best_d_val = d1, v1
    return {"sulfur_ppm": (best_s_date, best_s_val), "D15": (best_d_date, best_d_val)}


def load_telemetry_row_nearest(path: str, target_time: _dt) -> dict[str, Any]:
    """Читает avt_tags.csv / 242000_tags.csv и возвращает СТРОКУ тегов,
    ближайшую по времени к target_time (синхронизация по времени, а не
    по номеру строки — обязательное правило ТЗ). Служебные колонки
    'Unnamed: ...' отбрасываются."""
    df = pd.read_csv(path, parse_dates=["date"])
    idx = (df["date"] - target_time).abs().idxmin()
    row = df.loc[idx]
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    row = row.drop(labels=drop_cols)
    actual_time = row["date"]
    row = row.drop(labels=["date"])
    
    res: dict[str, Any] = {"_actual_time": actual_time}
    for k, v in row.items():
        res[str(k)] = v
    return res


if __name__ == "__main__":
    LIMS_PATH = "/mnt/project/ЛИМСы_01_01_2023__н_в__2.xlsx"
    PAK_PATH = "/mnt/project/Выгрузка_ПАК_01_01_2023__н_в_.xlsx"
    AVT_TAGS_PATH = "/mnt/user-data/uploads/avt_tags.csv"
    GO_TAGS_PATH = "/mnt/user-data/uploads/242000_tags.csv"

    lims = load_latest_lims_godt(LIMS_PATH)
    pak = load_latest_pak_sulfur_d15(PAK_PATH)

    lims_date, lims_sulfur = lims["sulfur_mg_kg"]
    pak_date, pak_sulfur = pak["sulfur_ppm"]
    _, lims_d15 = lims["D15"]
    _, lims_t95 = lims["T95"]

    print("Реальные данные, загруженные из ЛИМС/ПАК:")
    print(f"  ЛИМС сера:  {lims_sulfur} мг/кг  (дата анализа: {lims_date})")
    print(f"  ПАК  сера:  {pak_sulfur} ppm     (дата показания: {pak_date})")
    print(f"  ЛИМС D15:   {lims_d15} кг/м3, ЛИМС T95: {lims_t95} °C")
    print()

    # --- Приоритет источника: ЛИМС > ПАК (см. правило в ТЗ) ---
    # Если ЛИМС "моложе" разумного порога свежести — используем его как контрольный факт,
    # иначе (устарел) — переходим на ПАК с понижением уверенности прогноза.
    freshness_limit_hours = 48
    lims_age_hours = (pak_date - lims_date).total_seconds() / 3600 if (lims_date and pak_date) else None

    if lims_age_hours is not None and lims_age_hours <= freshness_limit_hours:
        sulfur_source, sulfur_value, confidence = "ЛИМС", lims_sulfur, 0.9
    else:
        sulfur_source, sulfur_value, confidence = "ПАК", pak_sulfur, 0.6

    print(f"Источник серы для прогноза: {sulfur_source} = {sulfur_value} (уверенность {confidence})")
    print()

    if lims_date is None:
        raise ValueError("Дата из ЛИМС не найдена.")

    # --- синхронизация телеметрии ПО ВРЕМЕНИ к моменту лабораторного анализа ---
    avt_row = load_telemetry_row_nearest(AVT_TAGS_PATH, lims_date)
    go_row = load_telemetry_row_nearest(GO_TAGS_PATH, lims_date)

    print(f"Ближайшая точка телеметрии АВТ:      {avt_row['_actual_time']} "
          f"(разница с ЛИМС: {abs((avt_row['_actual_time']-lims_date).total_seconds()/3600):.1f} ч)")
    print(f"Ближайшая точка телеметрии 24-2000:  {go_row['_actual_time']} "
          f"(разница с ЛИМС: {abs((go_row['_actual_time']-lims_date).total_seconds()/3600):.1f} ч)")
    print()

    # --- current_state: ПОЛНОСТЬЮ реальные значения, синхронизированные по времени ---
    example_state = {
        # --- управляемые параметры АВТ (avt_tags.csv) ---
        "F65": avt_row["F65"], "F30": avt_row["F30"], "F32": avt_row["F32"], "F34": avt_row["F34"],
        "T33": avt_row["T33"], "F12": avt_row["F12"], "F14_avt": avt_row["F14"], "F64": avt_row["F64"],
        # --- управляемые параметры 24-2000 (242000_tags.csv) ---
        "F26": go_row["F26"], "P24": go_row["P24"], "T16": go_row["T16"], "T18": go_row["T18"],
        # --- контекстные теги моделей ВАК (242000_tags.csv) ---
        "T12": go_row["T12"], "F15": go_row["F15"], "W7": go_row["W7"], "T23": go_row["T23"],
        "F1": go_row["F1"], "P13": go_row["P13"], "F9": go_row["F9"], "T6": go_row["T6"],
        "T5": go_row["T5"], "T11": go_row["T11"], "F25": go_row["F25"], "F22": go_row["F22"],
        "P8": go_row["P8"], "W4": go_row["W4"], "F2": go_row["F2"],
        # F14 в формулах ВАК — это тег 24-2000:F14 (не путать с F14_avt выше, см. предупреждение в шапке файла)
        "F14": go_row["F14"],
        # --- реальные значения из ЛИМС (лаговые признаки в формулах ВАК) ---
        "LIMS_D15": lims_d15 if lims_d15 is not None else 840.0,
        "LIMS_T95": lims_t95 if lims_t95 is not None else 360.0,
    }

    opt_in = OptimizationInput(
        current_state=example_state,
        quality_forecast={"sulfur_ppm": {"pred": sulfur_value, "confidence": confidence,
                                          "source": sulfur_source, "age_hours": lims_age_hours}},
        reliability_assessment={"risk_level": "normal", "risk_score": 0.1, "hard_constraints": {}},
        # t90_max_c не задан спецификацией в пакете -> не проверяем как жёсткое, только сера
        spec=Spec(sulfur_max_ppm=10.0),
    )

    result = run_optimization_agent(opt_in)
    print(result.message)
    print(f"Всего сценариев: {len(result.all_scenarios)}, допустимых: {len(result.feasible_scenarios)}, "
          f"на Парето-фронте: {len(result.pareto_front)}")
    if result.recommendation:
        print("Рекомендованные изменения:", result.recommendation.deltas)
        print("Целевые метрики:", result.recommendation.objectives)