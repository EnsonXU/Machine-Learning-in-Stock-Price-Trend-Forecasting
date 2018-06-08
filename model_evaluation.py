# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:53:18 2018

@author: chenxu
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression 

'''
def plot_learning_curve(estimator, title, X, y,cv, 
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)  
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, test_scores_mean, label = 'Validation error')
    plt.ylabel('Score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    #plt.ylim(0,0.6)
    plt.show() 
'''

if __name__=='__main__':
    
    data=pd.read_csv("/data/001.csv")

    #print (df.head())
    #show the summary of the data
    '''
    print (df.describe())
    print (df.std())
    '''
    data = data.dropna(axis=0)
    row_count = data.shape[0]  
    #print(row_count)
    X = data.iloc[:,1:]  
    #提取数据中的字符标签Y  
    y = data.iloc[:,0] 
 
    train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=1)  
    train_sizes,train_score,test_score =  learning_curve(LogisticRegression(penalty="l2",random_state=0),X=train_x,y=train_y, train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  
    #计算训练集交叉验证的准确率的平均值  
    train_mean = np.mean(train_score,axis=1)  
    #计算训练集交叉验证的准确率的标准差  
    train_std = np.std(train_score,axis=1)  
    test_mean = np.mean(test_score,axis=1)  
    test_std = np.std(test_score,axis=1)  
    plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score_mean")  
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
    plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
             markersize=5,label="test_score_mean")  
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
    plt.grid()  
    plt.xlabel("train_size")  
    plt.ylabel("Score")  
    plt.legend(loc="lower right")  
    #plt.ylim([0.7,1.0])  
    plt.show()  