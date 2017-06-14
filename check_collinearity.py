
from load_data import load_data
from replace_variables import replace_variables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

save_path = 'report//'

def check_collinearity(X):
    # get column names from the variables
    names = list(X)

    # check collinearity of variables
    corr = X.corr()

    # # # visualize correlation
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.matshow(abs(corr))
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90, fontsize = 9)
    # plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 9)
    # plt.savefig(save_path + 'correlation', dpi = 600)

    correlated = []
    for i in range(corr.shape[0]-1):
        for j in range(i+1, corr.shape[1]):
            if abs(corr.ix[i][j]) > 0.7:
                correlated.append(names[j])
                if names[j] in X:
                    del X[names[j]]

    # delete the variables that do not change with observations
    del X['EmployeeCount']
    del X['Over18']
    del X['StandardHours']
    # visualize correlation after delete dependent variable
    # corr = X.corr()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.matshow(abs(corr))
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90, fontsize = 9)
    # plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 9)
    # plt.savefig(save_path + 'correlation_checked', dpi = 600)
    return X

# X, y = load_data()
# X, y = replace_variables(X, y)
# X = check_collinearity(X)
#
# #print data, data.shape

