#!/usr/bin/env python
# coding: utf-8

# In[164]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import split_scale
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm

import env
import wrangle

url = env.get_db_url('telco_churn')


# In[163]:




# # In[3]:


# #telco db
# #df = wrangle.wrangle_churn()
# #df.drop(columns=['customer_id'],inplace=True)


# # In[16]:


# df['contract_type'] = np.where(df['contract_type']== 'One year',1,df['contract_type'])


# # In[94]:


# df['dum_column'] = 1


# # In[75]:


# #split train and test data
# train, test = split_scale.split_my_data(df)

# #scale with MinMax
# #scaler, train,test = split_scale.standard_scaler(train,test)


# # In[165]:


# #X and y train and test
# train, test = split_scale.split_my_data(df)
# X_train = train.drop(columns='total_charges')
# y_train = train[['total_charges']]
# X_test = test.drop(columns='total_charges')
# y_test = test[['total_charges']]


# type(train)


# # In[77]:


# #1 f regression testing features *unscaled*

# f_selector = SelectKBest(f_regression,k=2)

# f_selector.fit(X_train,y_train)

# f_support = f_selector.get_support()
# f_feature = X_train.loc[:,f_support].columns.tolist()

# print(str(len(f_feature)), 'selected features')
# print(f_feature)


# # In[78]:


# #1 plot correlations of x features to y *unscaled*
# plt.figure(figsize=(6,5))
# cor = train.corr()
# sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
# plt.show()


# # In[79]:


# #2 scaled f reg
# scaler, train,test = split_scale.standard_scaler(train,test)
# #X and y train and test
# X_train = train.drop(columns='total_charges')
# y_train = train[['total_charges']]
# X_test = test.drop(columns='total_charges')
# y_test = test[['total_charges']]


# # In[80]:


# train.head(2)


# # In[81]:


# f_selector = SelectKBest(f_regression,k=2)
# f_selector.fit(X_train,y_train)

# f_support = f_selector.get_support()
# f_feature = X_train.loc[:,f_support].columns.tolist()

# print(str(len(f_feature)), 'selected features')
# print(f_feature)


# # In[82]:


# plt.figure(figsize=(6,5))
# cor = train.corr()
# sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
# plt.show()


# # In[83]:


# import statsmodels.api as sm

# #ols model
# ols_model = sm.OLS(y_train,X_train)

# #fit model
# fit = ols_model.fit()

# fit.summary()


# # In[84]:


# # OLS backward elimination based on pvalue

# cols = list(X_train.columns)
# pmax = 1
# while (len(cols)>0):
#     p = []
#     X_1 = X_train[cols]
#     X_1 = sm.add_constant(X_1)
#     model = sm.OLS(y_train,X_1).fit()
#     p = pd.Series(model.pvalues.values[1:],index = cols)
#     pmax = max(p)
#     feature_with_p_max = p.idxmax()
#     if (pmax>.05):
#         cols.remove(feature_with_p_max)
#     else:
#         break
        
# selected_features_BE = cols
# print(selected_features_BE)


# # In[97]:


from sklearn.linear_model import LassoCV
    
# reg = LassoCV()
# reg.fit(X_train,y_train)

# print("Best alpha using built in LassoCV: %f" % reg.alpha_)
# print("Best score using built in LassoCV: %f" % reg.score(X_train,y_train))
# coef = pd.Series(reg.coef_,index=X_train.columns).sort_values(ascending=False)

# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# # In[96]:





# # In[87]:


# imp_coef = coef

# import matplotlib

# matplotlib.rcParams['figure.figsize'] = (4.0, 5.0)
# imp_coef.plot(kind = "barh")
# plt.title("Feature importance using Lasso Model")


# # In[88]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# model = LinearRegression()

# #Initializing RFE model, with parameter to select top 2 features. 
# rfe = RFE(model, 2)

# #Transforming data using RFE
# X_rfe = rfe.fit_transform(X_train,y_train)  

# #Fitting the data to model
# model.fit(X_rfe,y_train)

# print(rfe.support_)
# print(rfe.ranking_)


# # In[138]:


# number_of_features_list = np.arange(1,5)
# high_score = 0

# number_of_features=0
# score_list = []

# for n in range(len(number_of_features_list)):
#     model = LinearRegression()
#     rfe = RFE(model,number_of_features_list[n])
#     X_train_rfe = rfe.fit_transform(X_train,y_train)
#     X_test_rfe = rfe.transform(X_test)
#     model.fit(X_train_rfe,y_train)
#     score = model.score(X_test_rfe,y_test)
#     score_list.append(score)
#     if(score>high_score):
#         high_score = score
#         number_of_features = number_of_features_list[n]
        
# print("Optimum number of features: %d" %number_of_features)
# print("Score with %d features: %f" % (number_of_features,high_score))


# # In[152]:


# list(range(1,X_train.shape[1]+1))


# In[155]:


#5 function returning number of features from RFE function.
def optimum_features(X_train,y_train,X_test,y_test):
    number_of_features_list = list(range(1,X_train.shape[1]+1))
    high_score = 0

    number_of_features=0
    score_list = []

    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features


# In[156]:


#optimum_features(X_train)


# In[115]:


def top_n_features(num_features,X_train,y_train):
    features = num_features
    reg = LassoCV()
    reg.fit(X_train,y_train)
    coef = pd.Series(reg.coef_,index=X_train.columns).sort_values(ascending=False)
    return coef.head(num_features)


# In[159]:


#top_n_features(optimum_features(X_train))


# In[161]:


#X_train


# In[107]:





# In[ ]:




