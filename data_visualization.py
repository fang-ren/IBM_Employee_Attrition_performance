from load_data import load_data
from check_collinearity import check_collinearity
from data_cleaning import data_cleaning
from replace_variables import replace_variables
import numpy as np
import matplotlib.pyplot as plt
import notebook
import seaborn as sns
import os.path

X, y = load_data()
#X, Y = data_cleaning(X, Y, 5)
X, y = replace_variables(X, y)
X = check_collinearity(X)
names = list(X)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

# # one variable vs another
# save_path = 'report\\variable_correlation'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# for i in range(len(names)):
#     for j in range(i+1, len(names)):
#         print names[i], names[j]
#         sns.jointplot(X[names[i]], X[names[j]], cmap=cmap, kind = 'kde')
#         plt.savefig(os.path.join(save_path, names[i] +' Vs ' + names[j]))
#         plt.close('all')
#
#
# # single variable vs target (attrition)
# save_path = 'report\\single_variable_visualization'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# for i in range(len(names)):
#     sns.jointplot(X[names[i]], y, cmap=cmap, kind = 'kde')
#     plt.savefig(os.path.join(save_path, names[i]))
#     plt.close('all')


# skewness of the target (attrition)
save_path = 'report\\'
if not os.path.exists(save_path):
    os.mkdir(save_path)
sns.distplot(y)
plt.savefig(os.path.join(save_path, 'distribution of target variable'))