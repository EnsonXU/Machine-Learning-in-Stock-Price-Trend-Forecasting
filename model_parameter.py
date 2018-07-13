# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:50:48 2018

@author: chenxu
"""

import pandas as pd  
#import statsmodels.api as sm  
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("/data/3Mdata0.csv")
data = df 
data = data.dropna(axis=0)
row_count = data.shape[0]  

# set the target column
train_cols = data.columns[1:]  

X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
  
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


model = LogisticRegression(penalty='l2')

param_grid=[{#'penalty': ('l1', 'l2'),  
             'C': (0.01, 0.1, 1, 10),
             'solver':('newton-cg', 'lbfgs'),
             'multi_class':('ovr', 'multinomial')}]
gs = GridSearchCV(estimator=model,
                 param_grid=param_grid,
                )

gs = gs.fit(X_train_std,Y_train)
print(gs.best_score_)
print(gs.best_params_)


rf = RandomForestClassifier()
param_grid=[{ 'max_depth' : [1, 20],  #depth of each decision tree
             'n_estimators': [100, 50],  #count of decision tree
             'max_features': ['sqrt', 'auto', 'log2'],      
             'min_samples_split': [2],      
             'min_samples_leaf': [1, 3, 4],
             'bootstrap': [True, False],}]
gs = GridSearchCV(estimator=rf,
                 param_grid=param_grid,
                )

gs = gs.fit(X_train_std,Y_train)
print(gs.best_score_)
print(gs.best_params_)



clf = svm.SVC()
param_grid=[{ 'C':[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
             'kernel': ('rbf', 'linear'),
             'gamma': [0.001, 0.0001]
             }]
gs = GridSearchCV(estimator=clf,
                 param_grid=param_grid,
                )

gs = gs.fit(X_train_std,Y_train)
print(gs.best_score_)
print(gs.best_params_)




