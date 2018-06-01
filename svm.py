# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:54:01 2018

@author: chenxu
"""

from sklearn import svm
import csv as csv
import numpy as np

train=[]
ans_train=[]
test=[]  
ans_test=[]        #Array Definition

'''
path1 =  r'/data/001sma_train_svm.csv'     #Address Definition
path2 =  r'/data/001sma_train_ans_svm.csv'
path3 =  r'/data/001sma_test_svm.csv'
path4 =  r'/data/001sma_test_ans_svm.csv'
'''

path1 =  r'/data/001nosma_train_svm.csv'     #Address Definition
path2 =  r'/data/001nosma_train_ans_svm.csv'
path3 =  r'/data/001nosma_test_svm.csv'
path4 =  r'/data/001nosma_test_ans_svm.csv'


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
        ans_train.append(row2)
    ans_train = np.array(ans_train)

with open(path3, 'r') as f3:
    reader3 = csv.reader(f3)
    next(reader3, None)  
    for row3 in  reader3:
        test.append(row3)
    test = np.array(test)

with open(path4, 'r') as f4:
    reader4 = csv.reader(f4)
    next(reader4, None)  
    for row4 in  reader4:
        ans_test.append(row4)
    ans_test = np.array(ans_test)
    
'''
print(train)
print(ans_train)

print(test)
print(ans_test)

X = [[0, 0], [1, 1]]
y = [0, 1]
'''

clf = svm.SVC()
#clf = svm.NuSVC()
clf.fit(train, ans_train)  
p_result = clf.predict(test)

hit=0
total=len(p_result)

for i in range(0,len(test)):
    if p_result[i] == ans_test[i]:
        hit+=1

print("Total:",total)
print("Hit:",hit)
print("Precision:",100*hit/total)


