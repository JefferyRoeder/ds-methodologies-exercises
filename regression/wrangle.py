import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import wrangle
import env

url = env.get_db_url('telco_churn')
def wrangle_churn():
    df = pd.read_sql("""
SELECT 
c.customer_id, c.monthly_charges, c.tenure, c.total_charges, ct.contract_type, c.internet_service_type_id AS internet_type
FROM customers c
JOIN contract_types ct USING(contract_type_id)
WHERE ct.contract_type = 'Two Year' and c.total_charges != ' '
""",url)

    df.total_charges = df.total_charges.astype(float)
    return df