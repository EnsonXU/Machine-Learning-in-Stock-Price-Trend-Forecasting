# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:28:15 2018

@author: chenxu
"""
import pandas as pd  
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


value=[0] * 60
totalvalue=[0] * 60
price = []
holding=[0] * 60

#train the model
df=pd.read_csv("/data/3Mtradetrain.csv")
data = df 
data = data.dropna(axis=0)
data['R'] = data['ClPr'].diff(-60)   
data['R'] = np.where(data['R']>0,1,0)
data = data.drop(columns=['ClPr'])
print(data)

test=pd.read_csv("/data/3Mtest.csv")

test = test.dropna(axis=0)
#print(test)
# set the target column
train_cols = data.columns[1:] 
test_cols = test.columns[1:] 

X_train = data[train_cols]
Y_train = data['R']

X_test = test[test_cols]
Y_test = test['ClPr']
#print(X_test)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

#fit_intercept = False, C = 1e9

rf=RandomForestClassifier(bootstrap=True,max_depth=10,max_features='log2',max_leaf_nodes=3, 
                                                    min_samples_leaf=3,min_samples_split=3,
                                                    n_estimators=150)
rf.fit(X_train_std, Y_train)
output = rf.predict(X_test_std)

print(output)
print(test['ClPr'][0])
i=0
for o in output:
    print(o)
    if o == 1 & holding[i] == 0:
        value[i]=value[i]+test['ClPr'][i]*100
        holding+=1
    elif o==0 & holding[i] == 1:
        value[i]=test['ClPr'][i]*100
        holding-=1
    for v in value:
        totalvalue[i] += v
    i+=1
    if i==60:
        i=0
    #print(value)

price = np.array(test['ClPr']).tolist()
#print(price)

print(value)