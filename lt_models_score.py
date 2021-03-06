# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:31:15 2018

@author: chenxu 
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

if __name__=='__main__':
    
    df=pd.read_csv("/data/3Mlt2.csv")
    data = df 
    data = data.dropna(axis=0)
    row_count = data.shape[0] 
    data['R'] = data['ClPr'].diff(-60)   
    data['R'] = np.where(data['R']>0,1,0)
    #print(data)
    
    data = data.drop(columns=['ClPr'])
    train_cols = data.columns[1:]  
    train_x,test_x,train_y,test_y = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
    
    #evaluation of lg penalty=l2
    pipe_line = Pipeline([("std",StandardScaler()),  
                          ("clf",LogisticRegression(penalty="l2",C=1,multi_class='ovr',solver='newton-cg'))])  
    train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=train_x,y=train_y,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  

    train_mean = np.mean(train_score,axis=1)  
    train_std = np.std(train_score,axis=1)  
    test_mean = np.mean(test_score,axis=1)  
    test_std = np.std(test_score,axis=1)  
    plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score_mean")  
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
    plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
             markersize=5,label="test_score_mean")  
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
    plt.grid() 
    plt.title('Learning_curve of Logistic Regression (l2,60days)')
    plt.xlabel("train_size")  
    plt.ylabel("Score")  
    plt.legend(loc="lower right")  
    #plt.ylim([0.7,1.0])  
    plt.show()  
    
    #evaluation of lg penalty=l1
    pipe_line = Pipeline([("std",StandardScaler()),  
                          ("clf",LogisticRegression(penalty="l1",C=1))])  
    train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=train_x,y=train_y,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  

    train_mean = np.mean(train_score,axis=1)  
    train_std = np.std(train_score,axis=1)  
    test_mean = np.mean(test_score,axis=1)  
    test_std = np.std(test_score,axis=1)  
    plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score_mean")  
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
    plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
             markersize=5,label="test_score_mean")  
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
    plt.grid() 
    plt.title('Learning_curve of Logistic Regression (l1,60days)')
    plt.xlabel("train_size")  
    plt.ylabel("Score")  
    plt.legend(loc="lower right")  
    #plt.ylim([0.7,1.0])  
    plt.show()  
    
    #evaluation of rf
    #n_estimators=80, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='auto' , bootstrap=False
    #bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 80
    pipe_line = Pipeline([("std",StandardScaler()),  
                      ("clf",RandomForestClassifier(bootstrap=False,max_depth=100,max_features='auto',max_leaf_nodes=4, 
                                                    min_samples_leaf=1,min_samples_split=2,
                                                    n_estimators=150))])  
    train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=train_x,y=train_y,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  
 
    train_mean = np.mean(train_score,axis=1)  

    train_std = np.std(train_score,axis=1)  
    test_mean = np.mean(test_score,axis=1)  
    test_std = np.std(test_score,axis=1)  
    plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score")  
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
    plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
             markersize=5,label="test_score")  
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
    plt.grid()  
    plt.title('Learning_curve of Random Forest (60days)')
    plt.xlabel("train_size")  
    plt.ylabel("Score")  
    plt.legend(loc="lower right")  
    #plt.ylim([0.8,1.0])  
    plt.show()  
    
    #evaluation of SVM
    pipe_line = Pipeline([("std",StandardScaler()),  
                      ("clf",svm.SVC(C=1000,gamma=0.001,kernel='rbf'))])  
    train_sizes,train_score,test_score = learning_curve(estimator=pipe_line,X=train_x,y=train_y,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)  

    train_mean = np.mean(train_score,axis=1)  
  
    train_std = np.std(train_score,axis=1)  
    test_mean = np.mean(test_score,axis=1)  
    test_std = np.std(test_score,axis=1)  
    plt.plot(train_sizes,train_mean,color="blue",marker="o",markersize=5,label="train_score")  
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")  
    plt.plot(train_sizes,test_mean,color="green",linestyle="--",marker="s",  
             markersize=5,label="test_score")  
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")  
    plt.grid()  
    plt.title('Learning_curve of SVM (60days)')
    plt.xlabel("train_size")  
    plt.ylabel("Score")  
    plt.legend(loc="lower right")  
    #plt.ylim([0.8,1.0])  
    plt.show()  

