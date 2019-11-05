import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


url = env.get_db_url('zillow')

def prep_predictions():
    df = pd.read_sql("""

SELECT *
FROM predictions_2017
"""

,url)
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df = df[df.groupby('parcelid')['transactiondate'].transform('max') ==df['transactiondate']]



    return df

def single_unit_properties():
    df =pd.read_sql("""
SELECT p.*,pred_17.logerror, pred_17.transactiondate
FROM predictions_2017 pred_17

LEFT JOIN properties_2017 p on pred_17.parcelid = p.parcelid
LEFT JOIN airconditioningtype a ON p.airconditioningtypeid = a.airconditioningtypeid
WHERE p.`latitude` IS NOT NULL AND p.`longitude` IS NOT NULL
AND
p.propertylandusetypeid in (260,279,261,262,263,273) 
AND 

p.calculatedfinishedsquarefeet IS NOT NULL
and
p.bedroomcnt > 0
and
p.bedroomcnt <12
and
p.bathroomcnt <12
and 
p.bathroomcnt > 0
and
p.taxvaluedollarcnt > 0
    """,url)
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df = df[df.groupby('parcelid')['transactiondate'].transform('max') ==df['transactiondate']]
    return df