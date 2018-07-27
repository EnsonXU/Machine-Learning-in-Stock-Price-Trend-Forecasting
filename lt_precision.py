# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:59:14 2018

@author: chenxu
"""
from sklearn import svm
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


df=pd.read_csv("/data/3Mlt2.csv")
data = df 
data = data.dropna(axis=0)
row_count = data.shape[0] 
data['R'] = data['ClPr'].diff(-60)   
data['R'] = np.where(data['R']>0,1,0)

train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
    
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#fit_intercept = False, C = 1e9
model = LogisticRegression(penalty="l2",C=1,multi_class='ovr',solver='newton-cg')
result = model.fit(X_train_std, Y_train)

prepro =result.predict_proba(X_test_std)
acc = result.score(X_test_std,Y_test) 

print ('Total: %d, Precision: %.2f' % (len(Y_test), acc)) 

train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
model = RandomForestClassifier(bootstrap=False,max_depth=100,max_features='auto',max_leaf_nodes=4, 
                                                    min_samples_leaf=1,min_samples_split=2,
                                                    n_estimators=150)

result = model.fit(X_train_std, Y_train)

prepro =result.predict_proba(X_test_std)
acc = result.score(X_test_std,Y_test) 

print ('Total: %d, Precision: %.2f' % (len(Y_test), acc)) 


train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = svm.SVC(C=1000,gamma=0.001,kernel='rbf')
clf.fit(X_train_std, Y_train)  
p_result = clf.predict(X_test_std)
acc = clf.score(X_test_std,Y_test)      
print ('Total: %d, Precision: %.2f' % (len(Y_test), acc)) 
