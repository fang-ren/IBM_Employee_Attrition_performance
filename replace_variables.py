"""
author: fangren
"""

from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def replace_variables(X, y):
    mapping_BusinessTravel = {'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2}
    mapping_Department = {'Human Resources':0, 'Research & Development':1, 'Sales':2}
    mapping_EducationField = {'Life Sciences':0, 'Medical':1, 'Human Resources':2, 'Technical Degree':3, 'Marketing':4, 'Other':5}
    mapping_Gender = {'Female':0, 'Male':1}
    mapping_JobRole = {'Healthcare Representative':0, 'Human Resources':1, 'Laboratory Technician':2, 'Manager':3, 'Manufacturing Director':4,
                       'Research Director':5, 'Research Scientist':6, 'Sales Executive':7, 'Sales Representative':8}
    mapping_MaritalStatus = {'Divorced':0, 'Single':1, 'Married':2}
    mapping_Over18 = {'Y':1, 'N':0}
    mapping_OverTime = {'Yes':1, 'No':0}
    mapping_Attrition = {'Yes':1, 'No':0}

    X = X.replace({'BusinessTravel': mapping_BusinessTravel, 'Department': mapping_Department, 'EducationField':mapping_EducationField,
               'Gender':mapping_Gender, 'JobRole':mapping_JobRole, 'MaritalStatus':mapping_MaritalStatus, 'Over18':mapping_Over18,
               'OverTime':mapping_OverTime})
    y = y.apply(lambda x: mapping_Attrition[x])
    return X, y

# X, y = load_data()
# X, y = replace_variables(X, y)