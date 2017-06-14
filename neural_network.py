"""
author: fangren
"""

from load_data import load_data
from check_collinearity import check_collinearity
from data_cleaning import data_cleaning
from replace_variables import replace_variables
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

X, y = load_data()
#X, Y = data_cleaning(X, Y, 5)
X, y = replace_variables(X, y)
X = check_collinearity(X)
names = list(X)

X_train, X_test, y_train, y_val = train_test_split(X, y, train_size= 0.75,random_state=0)

# correct the skewness
oversampler=SMOTE(random_state=0)
X_train, y_train = oversampler.fit_sample(X_train,y_train)

# model selection using gridsearchcv
## split training data into development and cross validation
#X_dev, X_cv, y_dev, y_cv = train_test_split(X_train, y_train, train_size= 0.75,random_state=0)

# param_grid = {"n_estimators": [200, 400, 800, 1600],
#               "max_depth": [3, 9],
#               "max_features": [0.3, 1, 3, 10],
#               "warm_start":[True, False],
#               "n_jobs": [1, -1]}

param_grid = {"n_estimators": [200, 400, 800, 1600, 3200],
              "warm_start":[True, False],
              "n_jobs":[1,2],
              "max_features":[0.1, 0.3, 0.6],
              'min_samples_leaf':[2, 4],
              "random_state": [0],
              "max_depth":[3, 9],
              "verbose":[1]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, param_grid)
clf.fit(X_train, y_train)
params_rf = clf.best_params_
print params_rf

seed = 0   # We set our random seed to zero for reproducibility
# Random Forest parameters
# params_rf = {
#     'n_jobs': 1,
#     'n_estimators': 800,
#     'warm_start': True,
#     'max_features': 0.3,
#     'max_depth': 9,
#     'min_samples_leaf': 2,
#     'random_state' : seed,
#     'verbose': 0
# }

rf = RandomForestClassifier(**params_rf)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_val, y_pred)
print score

save_path = 'report\\'
if not os.path.exists(save_path):
    os.mkdir(save_path)

feature_importances = rf.feature_importances_
plt.scatter(range(len(names)), feature_importances, c = feature_importances, cmap = 'jet')
plt.xticks(range(len(names)), names, rotation = 90)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'random_forest_feature_importance'), dpi = 600)