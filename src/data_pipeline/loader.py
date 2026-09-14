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

    def load_data(self) -> pd.DataFrame:
        df_avt = pd.read_csv(self.file_avt, engine='pyarrow')
        df_242000 = pd.read_csv(self.file_242000, engine='pyarrow')

        df = pd.merge(df_avt, df_242000, on='date', how='inner')
        return df

    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', df.columns[74]])
        df["date"] = pd.to_datetime(df["date"])

        return df

    def load_lims(self) -> pd.DataFrame:
        df = pd.read_excel(
        self.file_lims,
        header=[0, 1],       # Строка 0 (Установка) и Строка 1 (Показатель) становятся заголовками
        skiprows=[2, 3]      # Пропускаем строки 2 (единицы измерения) и 3 (статистика "Количество значений:")
    )
        new_columns = []
        for installation, param in df.columns:
            if str(param).endswith('.1'):
                new_columns.append((installation, param.replace('.1', ''), 'Value'))
            else:
                new_columns.append((installation, param, 'Date'))

        df.columns = pd.MultiIndex.from_tuples(new_columns, names=['Установка', 'Показатель', 'Тип'])
        return df

    # Для демонстрации дописать:
    # is_anomaly() которая проверяет текущий набор на аномальность