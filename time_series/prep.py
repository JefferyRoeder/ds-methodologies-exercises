import pandas as pd
def get_stores_data():
    items = pd.read_csv('items.csv')
    sales = pd.read_csv('sales.csv')
    stores = pd.read_csv('stores.csv')
    df = sales.merge(stores,left_on='store',right_on='store_id',how='left')
    df = df.merge(items,left_on='item',right_on='item_id',how='left')
    df.drop(columns=['store_id','item_id'],inplace=True)
    df = sales.merge(stores,left_on='store',right_on='store_id',how='left')
    df = df.merge(items,left_on='item',right_on='item_id',how='left')
    df.drop(columns=['store_id','item_id'],inplace=True)
    fmt = '%a, %d %b %Y %H:%M:%S %Z'
    df.sale_date = pd.to_datetime(df.sale_date,format=fmt)
    df = df.set_index('sale_date')
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['sales_total'] = df.sale_amount * df.item_price
    return df

def sales_by_day(df):
    sales_by_day = df.resample('D')[['sales_total']].sum()
    sales_by_day['diff_with_last_day'] = sales_by_day.sales_total.diff()
    return sales_by_day


def split_store_data(df, train_prop=.66): 
    train_size = int(len(df) * train_prop)
    train, test = df[0:train_size].reset_index(), df[train_size:len(df)].reset_index()
    return train, test