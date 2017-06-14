"""
author: fangren
"""

import pandas as pd
import numpy as np

def load_data(path = 'data\\WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    df = pd.read_csv(path)
    y = df['Attrition']
    X = df.drop('Attrition', 1)
    return X, y

#
# X, y = load_data()
# X.head(5) # explore the variables
# X.isnull().any()  # check wither any of the variable column has null values
# X.any() == np.inf # check wither any of the variable column has infinite values
#
