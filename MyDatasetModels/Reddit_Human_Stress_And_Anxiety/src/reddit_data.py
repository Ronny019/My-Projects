import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel("../data/Sample Data for Text Analysis.xlsx", sheet_name=None)


def get_data(dataframe):
    text_df = dataframe.iloc[~pd.isnull(dataframe), -1]
    return text_df


X_train = pd.DataFrame()
y_train = pd.DataFrame()

raw_text_df = get_data(dataset['Sample Data'])

raw_text_np = raw_text_df.values
