# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:20:56 2018

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

if __name__=='__main__':
    tw_array=[]
    lg=[]
    rfr=[]
    svmr=[]
    
    #91
    for tw in range(1,60):
        #prepare for the train data and test data
        df=pd.read_csv("/data/3Mlt2.csv")
        #df=pd.read_csv("/data/renshou_price.csv")
        data = df 
        data = data.dropna(axis=0)
        row_count = data.shape[0] 
        #data['R'] = data['ClPr'].diff(-tw)  
        data['R'] = data['P'].diff(-tw)
        data['R'] = np.where(data['R']>0,1,0)
        #print(data)
        
        #data = data.drop(columns=['ClPr'])
        data = data.drop(columns=['P'])
        #print(data)
        #data['Prev_HiPr'] = data['Prev_HiPr'].astype('float64')
        #test=data
        
        # set the target column
        train_cols = data.columns[1:]  
        X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
            
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        
        #fit_intercept = False, C = 1e9
        #penalty="l2",C=1,multi_class='ovr',solver='newton-cg'
        model = LogisticRegression()
        result = model.fit(X_train_std, Y_train)
        
        prepro =result.predict_proba(X_test_std)
        acc = result.score(X_test_std,Y_test)   
        
        tw_array.append(tw)
        lg.append(acc)
        #lg.append(hit/len(X_test_std))
        '''
        bootstrap=False,max_depth=100,max_features='auto',max_leaf_nodes=4, 
                                                    min_samples_leaf=1,min_samples_split=2,
                                                    n_estimators=150
                                                '''
        rf=RandomForestClassifier()
        rf.fit(X_train_std, Y_train)
        output = rf.predict(X_test_std)
        acc2 = rf.score(X_test_std,Y_test)
       
        rfr.append(acc2)
        #C=1000,gamma=0.001,kernel='rbf'
        clf = svm.SVC()
        clf.fit(X_train_std, Y_train)  
        p_result = clf.predict(X_test_std)
        acc3 = clf.score(X_test_std,Y_test)
        svmr.append(acc3)
        
    #print(tw_array)
    print(svmr)
    print(rfr)
    print(lg)
    plt.plot(tw_array,rfr,color="green",label="Accu_rf")
    plt.plot(tw_array,lg,color="blue",label="Accu_lg") 
    plt.plot(tw_array,svmr,color="red",label="Accu_svm")
    plt.title('Long-Term prediction accuracy')
    plt.xlabel("time window")  
    plt.ylabel("Accuracy")  
    plt.legend(loc="lower right")  
    #print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)) 
