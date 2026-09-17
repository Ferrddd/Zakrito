import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from src.data_pipeline import features as F
from src.data_pipeline import splitting as S
from src.data_pipeline.loader import DataPreparer
from src.data_pipeline.pipeline_models import data_pipeline_settings

dp = DataPreparer(data_pipeline_settings)

# --- реальный ПАК
pac = dp.load_pac()
print("PAC:", pac.shape, pac.columns.tolist())

# --- синтетическая телеметрия на той же сетке
idx = pd.date_range("2023-01-01", "2026-08-10", freq="10min")
rng = np.random.default_rng(0)
tel = pd.DataFrame(index=idx)
tel["T5_hdt"] = 340 + rng.normal(0, 1.5, len(idx))
tel["P8_hdt"] = 345 + rng.normal(0, 1.5, len(idx))
tel["F15_hdt"] = 20 + rng.normal(0, 2, len(idx))
tel["T11_hdt"] = 100 + rng.normal(0, 5, len(idx))
tel["F2_hdt"] = 300 + rng.normal(0, 10, len(idx))
tel["P24_hdt"] = 40 + rng.normal(0, 3, len(idx))
tel["W10_hdt"] = 0.8 + rng.normal(0, 0.05, len(idx))
tel["F19_hdt"] = 50 + rng.normal(0, 1, len(idx))
tel["T6_hdt"] = 8 + rng.normal(0, 1, len(idx))   # анализатор серы -> лик
# простой: две недели нулевой загрузки
down = slice("2024-07-01", "2024-07-14")
tel.loc[down, "T11_hdt"] = 0.5
tel.loc[down, "F2_hdt"] = 1.0
tel["on_grid"] = True
tel.index.name = "date"

df = dp.mark_downtime(tel)
df = dp.mark_sensor_anomalies(df)
pacx = pac.drop_duplicates(subset="date", keep="last").set_index("date").reindex(df.index)
df = df.join(pacx, how="left")
df = dp.mark_pac_health(df)
df = dp.add_blocks(df)
print("blocks:", df.block_id.nunique(), "valid share:", round(df.is_valid.mean(),3))
print(df[[c for c in df.columns if "__" in c and "Sulfur" in c]].mean(numeric_only=True))

df = F.add_engineered(df)
df = F.add_lag_features(df, list(F.CONTROLS) + list(F.CONDITION) + ["wabt","gas_to_feed"])
df = F.add_target(df, horizon_points=6)

cols = F.feature_columns(df, horizon_points=6, blind=True)
print("leak check: analyzer in features?", [c for c in cols if c.startswith(("T6_hdt","24-2000:Mg"))])

emb = S.embargo_delta(72, 6)
print("embargo:", emb)
dt_index = pd.DatetimeIndex(df.index)
ho = S.holdout_split(dt_index, "120D", emb)
folds = S.rolling_origin_folds(dt_index, 4, "60D", emb, holdout=ho)
for f in folds + [ho]: print(" ", f)
S.assert_no_leakage(df, ho, cols, "target_6", 6)
Xtr, ytr, Xte, yte = S.train_matrix(df, ho, cols, "target_6")
print("shapes:", Xtr.shape, Xte.shape, "| violations train/test:",
      round((ytr>10).mean(),3), round((yte>10).mean(),3))
