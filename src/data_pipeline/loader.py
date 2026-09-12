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
        self.anomaly_threshold = settings.anomaly_threshold

    def load_data(self) -> pd.DataFrame:
        df_avt = pd.read_csv(self.file_avt, engine='pyarrow')
        df_242000 = pd.read_csv(self.file_242000, engine='pyarrow')

        df = pd.merge(df_avt, df_242000, on='date', how='inner')
        return df

    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', df.columns[74]])
        df["date"] = pd.to_datetime(df["date"])

        return df

    def is_anomaly(self, row: pd.Series, medians: pd.Series) -> bool:
        def robust_z(s):
            med, mad = s.median(), (s - s.median()).abs().median()
            return (s - med) / (1.4826 * mad + 1e-9)
        
        row = row[row.index != "date"] #только столбцы с тэгами
        z = row.apply(robust_z)
        downtime_score = (z.abs() > 5).mean(axis=1)   # проверяет отклоняется ли z-score более чем на 5 стандартных отклонений и считает долю датчиков с таким отклонением с в строке

        return downtime_score > 0.3