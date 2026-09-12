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

    def load_data(self) -> pd.DataFrame:
        df_avt = pd.read_csv(self.file_avt, engine='pyarrow')
        df_242000 = pd.read_csv(self.file_242000, engine='pyarrow')

        df = pd.merge(df_avt, df_242000, on='date', how='inner')
        return df

    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', df.columns[74]])
        df["date"] = pd.to_datetime(df["date"])

        return df