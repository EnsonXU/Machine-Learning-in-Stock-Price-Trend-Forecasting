# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:54:01 2018

@author: chenxu
"""

from sklearn import svm
import csv as csv
import numpy as np

train=[]
test=[]  #Array Definition

'''
path1 =  r'/data/001sma_train_svm.csv'     #Address Definition
path2 =  r'/data/001sma_test_svm.csv'
'''

path1 =  r'/data/001c.csv'     #Address Definition
path2 =  r'/data/001c.csv'

with open(path1, 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    next(reader, None)          #Skip header because file header is not needed
    for row in reader:          #fill array by file info by for loop
        train.append(row)
    train = np.array(train)       	

with open(path2, 'r') as f2:
    reader2 = csv.reader(f2)
    next(reader2, None)  
    for row2 in  reader2:
        test.append(row2)
    test = np.array(test)

clf = svm.SVC()
#clf = svm.NuSVC()

train_size=2000

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


