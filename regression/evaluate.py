#!/usr/bin/env python
# coding: utf-8

# In[2]:


# prepare environment
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from math import sqrt
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from pydataset import data


# In[4]:


#1 load tips dataset
df = data('tips')
df.head(1)


# In[5]:


x = df.total_bill
y = df.tip


# In[42]:


#2a,b
ols_model = ols('tip~total_bill',data=df).fit()

df['yhat'] = ols_model.predict(pd.DataFrame(df.total_bill))


# In[43]:


df.head(1)
df['residual'] = df['yhat']-df['tip']
df['residual2'] = df.residual **2
df.head(1)


# In[45]:


#4 plot residuals
sns.residplot(df.total_bill,df.tip)


# In[54]:


#5 regression errors, SSE, ESS, TSS, MSE, RMSE
def regression_errors(y,yhat):
    SSE = sum(df['residual2'])
    SSE2 = sum((yhat-y)**2)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(df)
    RMSE = sqrt(MSE)
    return SSE,SSE2,MSE,RMSE,ESS,TSS


# In[55]:


regression_errors(df.tip,df.yhat)


# In[ ]:


#calculate baseline
df['tip_baseline_med'] = df['tip'].median()


# In[11]:


df_baseline = df[['total_bill','tip']]
df_baseline['yhat'] = df_baseline['tip'].mean()
df_baseline['residual'] = df_baseline['yhat'] - df_baseline['tip']
df_baseline['residual2'] = df_baseline['residual'] ** 2
df_baseline.head(1)


# In[47]:


#regression errors
SSE_bl = sum(df_baseline['residual2'])
MSE_bl = SSE_bl/len(df_baseline)
RMSE_bl = sqrt(MSE_bl)
ESS_bl = sum((df_baseline.yhat - df_baseline.tip.mean())**2)
TSS_bl = ESS + SSE
print(SSE_bl,MSE_bl,RMSE_bl)


# In[ ]:





# In[49]:


r2 = ESS/TSS
r2_bl = ESS_bl / TSS_bl
print(r2,r2_bl)


# In[20]:


df.head(5)


# In[40]:


a = sns.scatterplot(x='total_bill',y='tip_percent',data=df)


# In[26]:


ols_model.summary()


# In[36]:


order = ['Thur','Fri','Sat','Sun']
sns.countplot(x='day',hue='smoker',data=df,order=order)


# In[38]:


df['tip_percent'] = df.tip/df.total_bill


# In[39]:





# In[50]:


df['tip_baseline_med'] = df['tip'].median()


# In[51]:


df.head()


# In[ ]:




