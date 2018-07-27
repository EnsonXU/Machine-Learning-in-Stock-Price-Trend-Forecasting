# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:57:14 2018

@author: chenxu
"""
from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch  # 条件异方差模型相关的库


#rawdata=pd.read_csv("/data/3Mprice.csv")
rawdata=pd.read_csv("/data/renshou_price.csv")
rawdata = rawdata.dropna(axis=0)
row_count = rawdata.shape[0] 

#data = np.array(rawdata['cpercent']) 
data = np.array(rawdata['P']) 
train = data[:-1000]
test = data[-1000:]
print(len(train))


'''
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
'''

am = arch.arch_model(train,mean='AR',lags=34,vol='ARCH',p=4) 
res = am.fit()

#res.hedgehog_plot()

#pre = res.forecast(horizon=60,start=3000)
#print(pre.residual_variance.iloc[-1000:])

pre = res.forecast(horizon=1000,start=3000)

pre = pre.mean.iloc[3000]
print(len(pre))
print(len(test))

hit=0
total=1000
for i in range(0,1000):
    if pre[i]-test[i-1]>0 and test[i]-test[i-1]>0.0:
        hit+=1
    elif pre[i]-test[i-1]<0 and test[i]-test[i-1]<0.0:
        hit+=1
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total))  

'''
plt.figure(figsize=(10,4))
plt.plot(test,label='realValue')
pre.plot(label='predictValue')
plt.plot(np.zeros(10),label='zero')
plt.legend(loc=0)
'''