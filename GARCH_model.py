# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:28:17 2018

@author: chenxu
"""

import pandas as pd  
import numpy as np 
import arch
from arch import arch_model
import statsmodels.api as sm 
import statsmodels.tsa.stattools as ts
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    rawdata=pd.read_csv("/data/3Mprice.csv")
    rawdata = rawdata.dropna(axis=0)
    row_count = rawdata.shape[0] 
    
    data = np.array(rawdata['cpercent']) 
    #print(data)
   
    am = arch_model(data) 
    res = am.fit(update_freq=5)
    print(res.summary())
    
    fig = res.plot(annualize='D')
    
    '''
    train = data[:-1000]
    test = data[-1000:]
    am_GARCH = arch.arch_model(train, mean='AR',lags=34,vol='GARCH') 
    res_GARCH = am_GARCH.fit()
    print(res_GARCH.summary())
    print(res_GARCH.params)
    
    res_GARCH.hedgehog_plot()
    
    pre = res.forecast(horizon=10,start=3000).iloc[3000]
    plt.figure(figsize=(10,4))
    plt.plot(test,label='realValue')
    pre.plot(label='predictValue')
    plt.plot(np.zeros(1000),label='zero')
    plt.legend(loc=0)
    '''
    
    '''
    fig = plt.figure(figsize=(50,5))
    ax1=fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data,lags =50,ax=ax1)
    
    
    ini = res.resid[-34:]
    a = np.array(res.params[1:35])
    w = a[::-1] # 系数
    for i in range(1000):
        new = test[i] - (res.params[0] + w.dot(ini[-34:]))
        ini = np.append(ini,new)
    #print(len(ini))
    at_pre = ini[-1000:]
    at_pre2 = at_pre**2
    #print(at_pre2)

    ini2 = res.conditional_volatility[-2:] #上两个条件异方差值

    for i in range(1000):
        new = 0.000097 + 0.2*at_pre2[i] + 0.5*ini2[-1]
        ini2 = np.append(ini2,new)
        print(new)
    vol_pre = ini2[-1000:]
    print(vol_pre)
    '''
    
    
    '''
    data_log = np.log(data)

    #plt.plot(data_log)
    moving_avg = pd.rolling_mean(data_log,12)
    plt.plot(data_log,color='blue')
    plt.plot(moving_avg,color='red')
    
    data_log_moving_avg_diff = data_log - moving_avg
    
    t = sm.tsa.stattools.adfuller(data_log)  # ADF检验
    print ("p-value: ",t[1])

    result = ts.adfuller(data, 1)
    print (result)
    
    fig = plt.figure(figsize=(50,5))
    ax1=fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data,lags =50,ax=ax1)
    
    order = (14,0)
    model = sm.tsa.ARMA(data,order).fit()
    
    at = data -  model.fittedvalues
    at2 = np.square(at)
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(at,label = 'at')
    plt.legend()
    plt.subplot(212)
    plt.plot(at2,label='at^2')
    plt.legend(loc=0)
    
    m = 50 # 我们检验25个自相关系数
    acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
    out = np.c_[range(1,51), acf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    print(output)
    

    train = data[:-1000]
    test = data[-1000:]
    am = arch.arch_model(train,mean='AR',lags=34,vol='GARCH') 
    res = am.fit()
    #print(res.summary())
    print(res.params)
    res.plot()
    plt.plot(data,color='green')
    res.hedgehog_plot()
    
    
    ini = res.resid[-34:]
    a = np.array(res.params[1:35])
    w = a[::-1] # 系数
    for i in range(1000):
        new = test[i] - (res.params[0] + w.dot(ini[-34:]))
        ini = np.append(ini,new)
    #print(len(ini))
    at_pre = ini[-1000:]
    at_pre2 = at_pre**2
    #print(at_pre2)

    ini2 = res.conditional_volatility[-2:] #上两个条件异方差值

    for i in range(1000):
        new = 0.000097 + 0.2*at_pre2[i] + 0.5*ini2[-1]
        ini2 = np.append(ini2,new)
        print(new)
    vol_pre = ini2[-1000:]
    print(vol_pre)
    
    
    
    for i in range(0,1000):
        if vol_pre[i]>0 && data[i+3000]>0:
            hit+=1
        
    
    #plt.figure(figsize=(15,5))
    plt.plot(data,color='blue',label='origin_data')
    #plt.plot(res.conditional_volatility,color='red',label='conditional_volatility')
    x=range(3000,4000)
    plt.plot(x,vol_pre,'.r',color='yellow',label='predict_volatility')
    plt.legend(loc="lower right")
    '''








