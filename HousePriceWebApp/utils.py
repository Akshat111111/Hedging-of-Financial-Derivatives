import pickle
import pandas as pd

class Dataset():
    def __init__(self , dataset_path):
        with open(dataset_path, 'rb') as file:
            self.df = pickle.load(file)

    def get_column_names(self):
        column_names = list(self.df.columns)
        column_names.remove('price')
        return column_names
    
    def get_area_type(self):
        return self.df['area_type'].unique()
    
    def get_location(self):
        return tuple(self.df['location'].unique())

DATA = Dataset('dataframe.pkl')
PIPELINE_PATH = 'PIPELINE.pkl'

with open(PIPELINE_PATH , 'rb') as file:
    PIPELINE = pickle.load(file)