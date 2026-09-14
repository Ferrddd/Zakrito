import pandas as pd

from src.data_pipeline.pipeline_models import (
    Data_Pipeline_Settings,
    data_pipeline_settings,
)

settings = data_pipeline_settings

class DataPreparer:
    def __init__(self, settings:Data_Pipeline_Settings):
        self.file_avt = settings.file_avt
        self.file_242000 = settings.file_242000
        self.file_lims = settings.file_lims
        self.file_pac = settings.file_pac
        self.converted_data_path = settings.converted_data_path

    # Работа с csv
    def load_data(self) -> pd.DataFrame:
        df_avt = pd.read_csv(self.file_avt, engine='pyarrow')
        df_242000 = pd.read_csv(self.file_242000, engine='pyarrow')

        df = pd.merge(df_avt, df_242000, on='date', how='inner')
        return df


    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', df.columns[74]])
        df["date"] = pd.to_datetime(df["date"])
        return df


    def delete_downtime(self, df:pd.DataFrame) -> pd.DataFrame:
        def robust_z(s:pd.Series) -> pd.Series:
            med, mad = s.median(), (s - s.median()).abs().median()
            return (s - med) / (1.4826 * mad + 1e-9)

        cont_cols = df.drop(columns=["date"]).columns #только столбцы с тэгами
        z = df[cont_cols].apply(robust_z)
        downtime_score = (z.abs() > 5).mean(axis=1)   # проверяет отклоняется ли z-score более чем на 5 стандартных отклонений и считает долю датчиков с таким отклонением с в строке
        df = df[downtime_score <= settings.anomaly_threshold]
        return df


    # Работа с таблицами
    def load_lims(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_lims,
                           header=[0, 1],       # Строка 0 (Установка) и Строка 1 (Показатель) становятся заголовками
                           skiprows=[2, 3])      # Пропускаем строки 2 (единицы измерения) и 3 (статистика "Количество значений:")
        new_columns = []
        for col in df.columns:
            installation = str(col[0])
            param = str(col[1])
            if str(param).endswith('.1'):
                new_columns.append((installation, param.replace('.1', ''), 'Value'))
            else:
                new_columns.append((installation, param, 'Date'))

        df.columns = pd.MultiIndex.from_tuples(new_columns, names=['Установка', 'Показатель', 'Тип'])
        return df

    def load_pac(self) -> pd.DataFrame:
        raw = pd.read_excel(self.file_pac, header=None)
        tags = raw.iloc[0]
        data = raw.iloc[2:].reset_index(drop=True)  # строка 1 — units, пропускаем

        step = 3  # date, value, разделитель — если разделителей больше/меньше, вернёмся к динамическому варианту
        frames = {}
        for start in range(0, len(tags), step):
            tag_name = tags[start]
            if pd.isna(tag_name):
                continue
            date_col, value_col = start, start + 1
            block = data.iloc[:, [date_col, value_col]].copy()
            block.columns = ["date", str(tag_name).strip()]
            block["date"] = pd.to_datetime(block["date"])
            block[str(tag_name).strip()] = pd.to_numeric(block.iloc[:, 1], errors="coerce")
            block = block.dropna(subset=["date"])
            frames[str(tag_name).strip()] = block

        # мёрдж по дате, а не по позиции строки — outer, чтобы не терять точки ни одного тега
        wide = None
        for df in frames.values():
            wide = df if wide is None else wide.merge(df, on="date", how="outer")

        assert wide is not None
        return wide.sort_values("date").reset_index(drop=True)


    # Для демонстрации дописать:
    # is_anomaly() которая проверяет текущий набор на аномальность

    def prepare_data(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.clean_data(df)
        df = self.delete_downtime(df)
        df = self.delete_downtime(df)

        df_pac = self.load_pac()

        df = df.merge(df_pac, how="left")

        df.to_parquet(f"{self.converted_data_path}/telemetry+pac.parquet", compression="brotli")

        return df