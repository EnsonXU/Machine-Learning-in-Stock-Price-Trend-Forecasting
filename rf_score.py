# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:47:48 2018

@author: chenxu
"""

import numpy as np
import csv as csv
import pandas as pd  
import math
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder 
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

   

df=pd.read_csv("/data/001c.csv")
data = df 
data = data.dropna(axis=0)
#clf = gridsearch
#clf = svm.NuSVC()
X = data.ix[:,1:]  
#提取数据中的字符标签Y  
Y = data.ix[:,0] 
#print(label_y)
#print(X)

#将数据分为训练集和测试集  
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=0)  
pipe_line = Pipeline([("std",StandardScaler()),  
                      ("clf",RandomForestClassifier())])  
train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=train_x,y=train_y,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  
#计算训练集交叉验证的准确率的平均值  
train_mean = np.mean(train_score,axis=1)  
#计算训练集交叉验证的准确率的标准差  
train_std = np.std(train_score,axis=1)  
test_mean = np.mean(test_score,axis=1)  
test_std = np.std(test_score,axis=1)  
plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score")  
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
         markersize=5,label="test_score")  
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
plt.grid()  
plt.xlabel("train_size")  
plt.ylabel("Score")  
plt.legend(loc="lower right")  
#plt.ylim([0.8,1.0])  
plt.show()  


'''

gridsearch.fit(train[0:train_size,1::], train[0:train_size,0])    #train[0::,0] is as target

model = gridsearch
parameters = gridsearch.best_params_

print("Best param: ",parameters)
print('Best Score: {}'.format(gridsearch.best_score_))

x=[]
y=[]

for j in range(0,2000):
    test_size = train_size+j
    
    output = gridsearch.predict(test[test_size+1:row_count,1::])
    #output = gridsearch.predict(test)
    #print(output)
    correct=0
    
    for i in range(0,len(output)-1):
        if output[i]==test[test_size+1:row_count-1,0][i]:
            correct+=1
    
    total = row_count-1 - test_size
    
    #print("Total:",total,"Hit:",correct,"Precision:",100*correct/total)
    #print()
    j+=100
    y.append(100*correct/total)
    x.append(total)

plt.figure()  
plt.plot(x,y)  
plt.xlabel("Test_size")  
plt.ylabel("Precision")  
plt.title("Random_forest")  

'''



