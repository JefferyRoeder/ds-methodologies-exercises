import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env
import seaborn as sns

url = env.get_db_url('iris_db')
#
def wrangle_iris():
    df = pd.read_sql("""

SELECT *
FROM measurements m
JOIN species s on s.species_id = m.species_id;
"""
,url)
    print(df.head(3))
    
    print(df.shape)
    
    print(df.columns[0:5])
    
    print(df.dtypes)
    
    print(df.describe())
    return df