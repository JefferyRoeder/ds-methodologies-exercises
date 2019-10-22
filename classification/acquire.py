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
    
    print(df.columns)
    
    print(df.dtypes)
    
    print(df.describe())
    return df


def excel_reader():
    df_excel = pd.read_excel('Excel_Exercises.xlsx',sheet_name='Table1_CustDetails')
    df_excel_sample = pd.read_excel('Excel_Exercises.xlsx',sheet_name='Table1_CustDetails',nrows=100)
    print(df_excel.columns[0:5])
    print(df_excel.dtypes[df_excel.dtypes == object])
    print(df_excel.describe().loc[['min','max']])
    return df_excel, df_excel_sample


def google_sheet():
    google_sheet = "https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357"
    google_sheet = google_sheet.replace("edit#gid","export?format=csv&gid")
    df_google = pd.read_csv(google_sheet)
    print(df_google.iloc[0:3])
    print(df_google.columns)
    print(df_google.dtypes)
    print(df_google.describe(include =[np.number]))
    unique_categories = df_google[['Survived','Pclass','Sex','SibSp','Embarked']]
    unique_categories = [unique_categories[i].unique().tolist() for i in unique_categories.columns]
    print(unique_categories)
    return df_google, unique_categories


url = env.get_db_url('titanic_db')

def wrangle_titanic():
    df = pd.read_sql("""

SELECT *
FROM passengers
"""
,url)
    return df