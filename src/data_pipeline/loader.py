import pandas as pd

from src.data_pipeline import pipeline_models

settings = pipeline_models.data_pipeline_settings

class DataPreparer:
    def __init__(self, settings):
        self.file_avt = settings.file_avt
        self.file_242000 = settings.file_242000

    def load_data(self):
        df_avt = pd.read_csv(self.file_avt, engine='pyarrow')
        df_242000 = pd.read_csv(self.file_242000, engine='pyarrow')

        df = pd.merge(df_avt, df_242000, on='date', how='inner')
        return df

    def clean_data(self, df):
        df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
        df["date"] = df.to_datetime(df["date"])

        return df