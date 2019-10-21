import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale


#plots variable pairs for df
def plot_variable_pairs(df):
    return sns.PairGrid(df).map_diag(plt.hist).map_offdiag(plt.scatter)


#converts months to years
def months_to_years