# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:54:18 2018

@author: chenxu
"""
import pandas as pd  
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("/data/3Mlt2.csv")
data = df 
data = data.dropna(axis=0)
row_count = data.shape[0] 
data['R'] = data['ClPr'].diff(-60)   
data['R'] = np.where(data['R']>0,1,0)
#print(data)

data = data.drop(columns=['ClPr'])
print(data['R'])
#data['Prev_HiPr'] = data['Prev_HiPr'].astype('float64')
#test=data

# set the target column
train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
    
'''
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rf = RandomForestClassifier()
param_grid=[{ 'max_depth' : [10, 100],  #depth of each decision tree
             'n_estimators': [150, 80],  #count of decision tree
             'max_features': ['sqrt', 'auto', 'log2'],      
             'min_samples_split': [2, 3, 4],      
             'min_samples_leaf': [1, 2, 3, 4],
             'max_leaf_nodes': [2, 3, 4],
             'bootstrap': [True, False],}]
gs = GridSearchCV(estimator=rf,
                 param_grid=param_grid,
                )

gs = gs.fit(X_train_std,Y_train)
print(gs.best_score_)
print(gs.best_params_)
'''


pipe_line = Pipeline([("std",StandardScaler()),  
                      ("clf",RandomForestClassifier(bootstrap=True,max_depth=10,max_features='log2',max_leaf_nodes=3, 
                                                    min_samples_leaf=3,min_samples_split=3,
                                                    n_estimators=150))])  
train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=X_train,y=Y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  
 
train_mean = np.mean(train_score,axis=1)  

train_std = np.std(train_score,axis=1)  
test_mean = np.mean(test_score,axis=1)  
test_std = np.std(test_score,axis=1)  
plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score")  
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
         markersize=5,label="test_score")  
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
plt.grid()  
plt.title('Learning_curve of Long-term prediction of Random Forest')
plt.xlabel("train_size")  
plt.ylabel("Score")  
plt.legend(loc="lower right")  
#plt.ylim([0.8,1.0])  
plt.show()  
