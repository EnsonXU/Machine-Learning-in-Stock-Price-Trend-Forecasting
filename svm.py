# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:54:01 2018

@author: chenxu
"""

from sklearn import svm
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

train=[]
test=[]  #Array Definition

'''
path1 =  r'/data/001sma_train_svm.csv'     #Address Definition
path2 =  r'/data/001sma_test_svm.csv'


path1 =  r'/data/001c.csv'     #Address Definition
path2 =  r'/data/001c.csv'
'''

df=pd.read_csv("/data/3Mdata.csv")
data = df 
data = data.dropna(axis=0)
row_count = data.shape[0] 

train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = svm.SVC(C=100,gamma=0.001,kernel='linear')
clf.fit(X_train_std, Y_train)  
p_result = clf.predict(X_test_std)
acc = clf.score(X_test_std,Y_test)      
print("Total:",len(Y_test))
print("Precision:",acc)

'''
clf.fit(train[0:train_size,1::], train[0:train_size,0])  
p_result = clf.predict(test[train_size+1:len(test),1::])

hit=0
total=len(p_result)

for i in range(0,total):
    if p_result[i] == test[train_size+1:len(test),0][i]:
        hit+=1

print("Total:",total)
print("Hit:",hit)
print("Precision:",100*hit/total)
'''
