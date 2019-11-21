import pandas as pd
import requests


#acquire sales
def get_sales():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url + '/api/v1/sales')
    sales = []
    data = response.json()
    page = []

    while page is not None:


        result = response.json()
        data = response.json()['payload']['sales']
        page = result['payload']['next_page']
        if page is not None:
            response = requests.get(base_url + page)
        sales += data
    return pd.DataFrame(sales)

#acquire stores
def get_stores():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url + '/api/v1/stores')
    stores = []
    data = response.json()
    page = []

    while page is not None:


        result = response.json()
        data = response.json()['payload']['stores']
        page = result['payload']['next_page']
        if page is not None:
            response = requests.get(base_url + page)
        stores += data
    return pd.DataFrame(stores)


#acquire items
def get_items():
    base_url = 'https://python.zach.lol'    
    items = []
    data = response.json()
    page = []
    response = requests.get(base_url + '/api/v1/items')
    while page is not None:


        result = response.json()
        data = response.json()['payload']['items']
        page = result['payload']['next_page']
        if page is not None:
            response = requests.get(base_url + page)
        items += data
    return pd.DataFrame(items)



def merge_drop_columns(df):
    df = sales.merge(stores,left_on='store',right_on='store_id',how='left')
    df = df.merge(items,left_on='item',right_on='item_id',how='left')
    df.drop(columns=['store_id','item_id'],inplace=True)
    return df


def get_opsd_data(use_cache=True):
    if use_cache and path.exists('opsd.csv'):
        return pd.read_csv('opsd.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    df.to_csv('opsd.csv',index=False)
    return df