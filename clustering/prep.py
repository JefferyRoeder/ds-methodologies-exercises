import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


url = env.get_db_url('zillow')

def drop_columns(df):
    df_nulls_c = pd.DataFrame(df.apply(lambda x: len(x) - x.count(),axis=0))
    df_nulls_c['pct_rows_missing'] = df_nulls_c[0] / len(df)
    column_drops = df_nulls_c[df_nulls_c['pct_rows_missing'] >.2]
    column_drops['column_names'] = column_drops.index
    column_drops = list(column_drops.column_names)
    df_new = df.drop(column_drops,axis=1)
    return df_new

def drop_rows(df):
    df_nulls_r = pd.DataFrame(df.apply(lambda x: df.shape[1] -x.count(),axis=1))
    df_nulls_r['pct_rows_missing'] = df_nulls_r[0] / df.shape[1]
    row_drops = df_nulls_r[df_nulls_r['pct_rows_missing'] >.2]
    row_drops['column_names'] = row_drops.index
    row_drops = list(row_drops.column_names)
    df_new = df.drop(row_drops,axis=0)
    return df_new




