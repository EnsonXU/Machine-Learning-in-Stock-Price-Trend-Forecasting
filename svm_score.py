# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:54:01 2018

@author: chenxu
"""

from sklearn import svm
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder 
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

train=[]
test=[]
data=[]    #Array Definition

data=pd.read_csv("/data/001c.csv") 
data = data.dropna(axis=0)
clf = svm.SVC()
#clf = svm.NuSVC()
X = data.iloc[:,1:]  
#提取数据中的字符标签Y  
y = data.iloc[:,0] 
#print(y)
#print(X)
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=1)  
pipe_line = Pipeline([("std",StandardScaler()),  
                      ("clf",svm.SVC())])  
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

