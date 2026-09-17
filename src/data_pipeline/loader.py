
"""Подготовка данных для МАС.
 
Главный принцип: строки НЕ выбрасываются. Временная сетка 10 минут сохраняется
целиком, а все проблемные периоды помечаются булевыми масками. Фильтрация
происходит уже на этапе формирования обучающей выборки — после того, как
лаговые и оконные признаки посчитаны по сегментам без разрывов.
 
Если выбросить строки здесь, rolling(6) перепрыгнет через двухнедельный останов
и смешает два несвязанных режима, причём молча.
"""
 
from __future__ import annotations
 
import logging
import re
from collections.abc import Iterable
 
import numpy as np
import pandas as pd

from src.data_pipeline.pipeline_models import (
    DataPipelineSettings,
    data_pipeline_settings,
)
 
logger = logging.getLogger(__name__)
 
UNNAMED_RE = re.compile(r"^Unnamed", flags=re.IGNORECASE)
 
# Теги, которые по справочнику КИП являются поточными анализаторами серы.
# Это тот же сигнал, что и таргет: в признаки для прогноза они попадать не должны
# (только как «последнее известное измерение» с лагом >= горизонта прогноза).
SULFUR_ANALYZER_TAGS = ("T6_hdt", "W7_hdt", "P13_hdt")
 
# Виртуальные анализаторы из телеметрии (APC): тоже расчётные величины, а не КИП.
VIRTUAL_ANALYZER_TAGS = ("F25_hdt",)
 
 
def robust_z(frame: pd.DataFrame) -> pd.DataFrame:
    """Робастный z-score по столбцам с защитой от вырожденных каналов.
 
    MAD == 0 бывает у почти константных каналов; деление на eps даёт z ~ 1e9 и
    канал начинает «аномалить» в каждой строке. В таком случае откатываемся на
    обычное СКО, а полностью константные каналы дают нулевой z.
    """
    med = frame.median()
    mad = (frame - med).abs().median()
    scale = 1.4826 * mad
 
    fallback = frame.std(ddof=0)
    scale = scale.where(scale > 0, fallback)
    degenerate = ~(scale > 0)
    if degenerate.any():
        logger.warning("Вырожденные каналы (нулевой разброс): %s",
                       list(scale.index[degenerate]))
    scale = scale.where(scale > 0, np.nan)
 
    return (frame - med) / scale
 
 
def frozen_runs(series: pd.Series, min_points: int) -> pd.Series:
    """True там, где значение не менялось min_points подряд идущих точек."""
    values = series.to_numpy()
    changed = pd.Series(values).ne(pd.Series(values).shift()).to_numpy()
    groups = changed.cumsum()
    run_len = pd.Series(values).groupby(groups).transform("size").to_numpy()
    return pd.Series((run_len >= min_points) & series.notna().to_numpy(),
                     index=series.index)
 
 
def _mask_min_duration(mask: pd.Series, min_points: int) -> pd.Series:
    """Оставляет только участки маски длиной >= min_points (убирает одиночные срабатывания)."""
    grp = mask.ne(mask.shift()).cumsum()
    size = mask.groupby(grp).transform("size")
    return mask & (size >= min_points)
 
 
class DataPreparer:
    def __init__(self, settings: DataPipelineSettings = data_pipeline_settings):
        self.s = settings
 
    # ------------------------------------------------------------------ КИП
    def load_telemetry(self) -> pd.DataFrame:
        df_avt = pd.read_csv(self.s.file_avt, engine="pyarrow")
        df_hdt = pd.read_csv(self.s.file_242000, engine="pyarrow")
 
        df_avt = self._drop_service_columns(df_avt)
        df_hdt = self._drop_service_columns(df_hdt)
 
        for df in (df_avt, df_hdt):
            df["date"] = pd.to_datetime(df["date"])
 
        # Явные суффиксы вместо _x/_y: T6_avt — температура низа К1,
        # T6_hdt — поточный анализатор серы. Путать их нельзя.
        df = pd.merge(
            df_avt, df_hdt, on="date", how="inner",
            suffixes=("_avt", "_hdt"), validate="one_to_one",
        )
        return df.sort_values("date").reset_index(drop=True)
 
    @staticmethod
    def _drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in df.columns if UNNAMED_RE.match(str(c))]
        constant = [
            c for c in df.columns
            if c != "date" and df[c].dtype.kind in "if" and df[c].nunique(dropna=True) <= 1
        ]
        if drop or constant:
            logger.info("Удаляю служебные %s и константные %s", drop, constant)
        return df.drop(columns=drop + constant)
 
    def normalize_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит к регулярной 10-минутной сетке: дедуп + reindex.
 
        Пропущенные метки времени появляются как строки с NaN и флагом
        on_grid=False. Так любые оконные операции считаются по реальному
        времени, а не по номеру строки.
        """
        df = df.drop_duplicates(subset="date", keep="last").set_index("date")
        full = pd.date_range(df.index.min(), df.index.max(), freq=self.s.grid_freq)
        n_missing = len(full) - len(df.index.intersection(full))
        off_grid = df.index.difference(full)
        if len(off_grid):
            logger.warning("%d меток вне сетки %s, пример: %s",
                           len(off_grid), self.s.grid_freq, off_grid[:3].tolist())
        df = df.reindex(full)
        df.index.name = "date"
        df["on_grid"] = df.notna().any(axis=1)
        logger.info("Сетка: %d точек, добавлено пропусков: %d", len(df), n_missing)
        return df
 
    # ----------------------------------------------------------------- маски
    def mark_downtime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Простой установки по расходу сырья — физический, а не статистический критерий."""
        load_tags = [t for t in self.s.load_tags if t in df.columns]
        if not load_tags:
            logger.warning("Теги загрузки %s не найдены, простой по расходу не размечен",
                           self.s.load_tags)
            df["mask_downtime"] = False
            return df
 
        load = df[load_tags].astype(float)
        threshold = load.median() * self.s.downtime_load_ratio
        low = (load < threshold).all(axis=1) | load.isna().all(axis=1)
        df["mask_downtime"] = _mask_min_duration(low.fillna(False),
                                                 self.s.downtime_min_points)
        logger.info("Простой: %.2f%% точек", 100 * df["mask_downtime"].mean())
        return df
 
    def mark_sensor_anomalies(self, df: pd.DataFrame,
                              exclude: Iterable[str] = ()) -> pd.DataFrame:
        """Доля каналов с выбросом в строке -> mask_sensor_anomaly.
 
        Статистика считается только по рабочим периодам, иначе простои сдвигают
        медиану и MAD.
        """
        exclude = set(exclude) | {"on_grid", "mask_downtime"}
        cols = [c for c in df.columns
                if c not in exclude and df[c].dtype.kind in "if"]
 
        working = df.loc[~df.get("mask_downtime", pd.Series(False, index=df.index)), cols]
        med = working.median()
        mad = (working - med).abs().median()
        scale = (1.4826 * mad).where(lambda s: s > 0, working.std(ddof=0))
        scale = scale.where(scale > 0, np.nan)
 
        z = (df[cols] - med) / scale
        share = (z.abs() > self.s.z_threshold).sum(axis=1) / max(len(cols), 1)
 
        df["anomaly_share"] = share
        df["mask_sensor_anomaly"] = share > self.s.anomaly_threshold
        logger.info("Аномальные строки: %.2f%%", 100 * df["mask_sensor_anomaly"].mean())
        return df
 
    # ------------------------------------------------------------------ ПАК
    def load_pac(self) -> pd.DataFrame:
        """Читает выгрузку ПАК: блоки (дата, значение), разделённые пустыми колонками.
 
        Пары ищутся по содержимому, а не по фиксированному шагу 3 — иначе
        любое изменение формата выгрузки ломает загрузку молча.
        """
        raw = pd.read_excel(self.s.file_pac, header=None)
        tags = raw.iloc[0]
        data = raw.iloc[2:].reset_index(drop=True)  # строка 1 — единицы измерения
 
        frames: list[pd.DataFrame] = []
        for pos, tag in enumerate(tags):
            if pd.isna(tag) or pos + 1 >= raw.shape[1]:
                continue
            dates = pd.to_datetime(data.iloc[:, pos], errors="coerce")
            values = pd.to_numeric(data.iloc[:, pos + 1], errors="coerce")
            if dates.notna().sum() == 0 or values.notna().sum() == 0:
                continue
            block = pd.DataFrame({"date": dates, str(tag).strip(): values})
            frames.append(block.dropna(subset=["date"]))
            logger.info("ПАК %s: %d точек, %s — %s", str(tag).strip(),
                        len(block), dates.min(), dates.max())
 
        wide = frames[0]
        for block in frames[1:]:
            wide = wide.merge(block, on="date", how="outer")
        return wide.sort_values("date").reset_index(drop=True)
 
    def mark_pac_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """Для каждого тега ПАК: <tag>__frozen, <tag>__out_of_range, <tag>__bad, <tag>__age_min.
 
        Замороженный сигнал — не редкость: в исходной выгрузке сера стоит
        на одном значении 46 суток (март-май 2024) и 12 суток на 18.4 ppm
        (апрель 2026). Без этой маски обе полки уедут в таргет как факт.
        """
        step_min = pd.Timedelta(self.s.grid_freq).total_seconds() / 60
        for tag, (lo, hi) in self.s.pac_ranges.items():
            if tag not in df.columns:
                continue
            series = df[tag].astype(float)
 
            frozen = frozen_runs(series, self.s.pac_frozen_min_points)
            out_of_range = series.notna() & (~series.between(lo, hi))
            bad = frozen | out_of_range | series.isna()
 
            df[f"{tag}__frozen"] = frozen
            df[f"{tag}__out_of_range"] = out_of_range
            df[f"{tag}__bad"] = bad
            # возраст последнего достоверного измерения, минуты
            good_idx = np.where(~bad.to_numpy(), np.arange(len(bad)), -1)
            last_good = pd.Series(good_idx, index=df.index).cummax()
            age = (np.arange(len(df)) - last_good.to_numpy()) * step_min
            df[f"{tag}__age_min"] = np.where(last_good.to_numpy() < 0, np.nan, age)
 
            logger.info("ПАК %s: заморожен %.2f%%, вне диапазона %.2f%%, пропуск %.2f%%",
                        tag, 100 * frozen.mean(), 100 * out_of_range.mean(),
                        100 * series.isna().mean())
        return df
 
 # ----------------------------------------------------------------- ЛИМС
    def load_lims(self) -> pd.DataFrame:
        raw = pd.read_excel(self.s.file_lims, header=[0, 1], skiprows=[2, 3])
        columns = []
        for col in raw.columns:
            # Безопасно извлекаем уровни MultiIndex через индексы кортежа
            installation = str(col[0])
            param = str(col[1])
            if param.endswith(".1"):
                columns.append((installation, param[:-2], "Value"))
            else:
                columns.append((installation, param, "Date"))
        raw.columns = pd.MultiIndex.from_tuples(
            columns, names=["Установка", "Показатель", "Тип"])
        return raw

    def lims_to_long(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Широкий ЛИМС -> длинный: date | installation | indicator | value.
 
        В длинном виде ЛИМС можно merge_asof'ить к телеметрии по времени, чего
        широкий MultiIndex не позволяет. Синхронизация только по времени —
        требование ТЗ.
        """
        records = []
        # Извлекаем уникальные пары установок и показателей по индексам кортежа
        pairs = {(str(col[0]), str(col[1])) for col in wide.columns}
        for inst, ind in sorted(pairs):
            try:
                dates = pd.to_datetime(wide[(inst, ind, "Date")], errors="coerce")
                values = pd.to_numeric(wide[(inst, ind, "Value")], errors="coerce")
            except KeyError:
                logger.warning("У показателя %s / %s нет пары Date+Value", inst, ind)
                continue
            block = pd.DataFrame({"date": dates, "value": values})
            block = block.dropna(subset=["date"])
            block["installation"] = inst
            block["indicator"] = ind
            records.append(block)

        long = pd.concat(records, ignore_index=True)
        return long.sort_values("date").reset_index(drop=True)
 
    def attach_lims(self, df: pd.DataFrame, long: pd.DataFrame,
                    indicators: dict[str, str]) -> pd.DataFrame:
        """Приклеивает последний доступный лабораторный результат + его возраст.
 
        indicators: {'имя колонки': 'Показатель из ЛИМС'}.
        merge_asof direction='backward' гарантирует, что в строку t не попадёт
        анализ, взятый позже t — иначе это утечка из будущего.
        """
        left = df.reset_index()
        for name, indicator in indicators.items():
            right = (long[long["indicator"] == indicator]
                     .loc[:, ["date", "value"]]
                     .dropna()
                     .sort_values("date")
                     .rename(columns={"value": f"lims_{name}"}))
            if right.empty:
                logger.warning("ЛИМС: показатель %s не найден", indicator)
                continue
            right[f"lims_{name}_ts"] = right["date"]
            left = pd.merge_asof(left, right, on="date", direction="backward")
            left[f"lims_{name}_age_min"] = (
                (left["date"] - left[f"lims_{name}_ts"]).dt.total_seconds() / 60)
            left = left.drop(columns=[f"lims_{name}_ts"])
        return left.set_index("date")
 
    # -------------------------------------------------------------- сегменты
    def add_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """block_id — номер непрерывного рабочего сегмента.
 
        Границы: простой, дыра в сетке, длинная аномалия. Все оконные признаки
        считаются внутри block_id, поэтому окно физически не может пересечь
        останов. Дополнительно hours_since_block_start — прокси наработки
        катализатора, без него модель серы разваливается на тесте.
        """
        broken = (df["mask_downtime"] | ~df["on_grid"]).fillna(True)
        df["is_valid"] = ~broken & ~df["mask_sensor_anomaly"].fillna(True)
        df["block_id"] = (broken != broken.shift()).cumsum().where(~broken)
 
        stamps = pd.Series(df.index, index=df.index)
        starts = stamps.groupby(df["block_id"]).transform("min")
        df["hours_since_block_start"] = (
            (stamps - starts).dt.total_seconds() / 3600)
        return df
 
    # ---------------------------------------------------------------- сборка
    def prepare_data(self, lims_indicators: dict[str, str] | None = None,
                     save: bool = True) -> pd.DataFrame:
        df = self.load_telemetry()
        df = self.normalize_grid(df)
        df = self.mark_downtime(df)
        df = self.mark_sensor_anomalies(df)
 
        pac = self.load_pac()
        pac = pac.drop_duplicates(subset="date", keep="last").set_index("date")
        pac = pac.reindex(df.index)  # выравнивание по времени, не по номеру строки
        df = df.join(pac, how="left")
        df = self.mark_pac_health(df)
 
        if lims_indicators:
            long = self.lims_to_long(self.load_lims())
            df = self.attach_lims(df, long, lims_indicators)
 
        df = self.add_blocks(df)
 
        logger.info("Итог: %d строк, пригодных для обучения %d (%.1f%%)",
                    len(df), int(df["is_valid"].sum()),
                    100 * df["is_valid"].mean())
 
        if save:
            out = self.s.converted_data_path / "telemetry_pac_lims.parquet"
            df.to_parquet(out, compression="brotli")
            logger.info("Сохранено: %s", out)
        return df
 
