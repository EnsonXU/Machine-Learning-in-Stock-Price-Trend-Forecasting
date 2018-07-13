# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:47:48 2018

@author: chenxu
"""

import numpy as np
import csv as csv
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold 

train=[]
test=[]      

'''  
path1 =  r'/data/001nosma_train.csv'     
path2 =  r'/data/001nosma_test.csv'
'''

'''
path1 =  r'/data/3Mdata0.csv'     
path2 =  r'/data/3Mdata0.csv'
'''
path1 =  r'/data/renshou.csv'     
path2 =  r'/data/renshou.csv'


with open(path1, 'r') as f1:    
    reader = csv.reader(f1)     
    next(reader, None)         
    for row in reader:          
        train.append(row)
    train = np.array(train)       	
	
with open(path2, 'r') as f2:
    reader2 = csv.reader(f2)
    next(reader2, None)  
    for row2 in  reader2:
        test.append(row2)
    test = np.array(test)

#To get the best number for running
parameter_gridsearch = {
                 'max_depth' : [1, 20],  #depth of each decision tree
                 'n_estimators': [80, 50],  #count of decision tree
                 'max_features': ['sqrt', 'auto', 'log2'],      
                 'min_samples_split': [2],      
                 'min_samples_leaf': [1, 3, 4],
                 'bootstrap': [True, False],
                 }

randomforest = RandomForestClassifier(bootstrap=True,max_depth=20,max_features='auto',min_samples_leaf=3, min_samples_split=2, n_estimators=50)

row_count = train.shape[0]  
train_size = math.floor(row_count/2)

crossvalidation = StratifiedKFold(train[0:train_size,0] , n_folds=5)

gridsearch = GridSearchCV(randomforest,             #grid search for algorithm optimization
                               scoring='accuracy',
                               param_grid=parameter_gridsearch,
                               cv=crossvalidation)

gridsearch.fit(train[0:train_size,1::], train[0:train_size,0])    #train[0::,0] is as target

model = gridsearch
parameters = gridsearch.best_params_

print("Best param: ",parameters)
print('Best Score: {}'.format(gridsearch.best_score_))

test_size = train_size+1

output = gridsearch.predict(test[test_size+1:row_count,1::])
#output = gridsearch.predict(test)
#print(output)
correct=0

for i in range(0,len(output)-1):
    if output[i]==test[test_size+1:row_count-1,0][i]:
        correct+=1

total = row_count-1 - test_size

print("Total:",total,"Hit:",correct,"Precision:",100*correct/total)
#print()






