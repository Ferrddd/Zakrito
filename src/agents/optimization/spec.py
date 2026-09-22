
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Spec:
    sulfur_max_ppm: float = 10.0
    cfpp_max_c: float | None = None          # подставить из выданной спецификации
    cloud_point_max_c: float | None = None   # подставить из выданной спецификации
    t90_max_c: float | None = None
    blend_fraction_sum_tol: float = 1e-3     # сумма долей блендинга = 100% ± tol

