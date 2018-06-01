# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:47:48 2018

@author: chenxu
"""

import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold 

train=[]
test=[]      

'''  
path1 =  r'/data/001nosma_train_rf.csv'     
path2 =  r'/data/001nosma_test_rf.csv'
'''

path1 =  r'/data/001sma_train_rf.csv'     
path2 =  r'/data/001sma_test_rf.csv'


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

randomforest = RandomForestClassifier()
crossvalidation = StratifiedKFold(train[0::,0] , n_folds=5)

gridsearch = GridSearchCV(randomforest,             #grid search for algorithm optimization
                               scoring='accuracy',
                               param_grid=parameter_gridsearch,
                               cv=crossvalidation)


gridsearch.fit(train[0::,1::], train[0::,0])    #train[0::,0] is as target

model = gridsearch
parameters = gridsearch.best_params_

print("Best param: ",parameters)
print('Best Score: {}'.format(gridsearch.best_score_))

output = gridsearch.predict(test[0::,1::])
#output = gridsearch.predict(test)

#print(output)
correct=0
total=len(output)

for i in range(0,len(output)):
    if output[i]==test[0::,0][i]:
        correct+=1

print("Total:",total)
print("Hit:",correct)
print("Precision:",100*correct/total)








