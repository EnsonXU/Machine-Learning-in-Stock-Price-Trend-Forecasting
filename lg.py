# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:31:15 2018

@author: chenxu
"""
import pandas as pd  
import statsmodels.api as sm  

'''
df=pd.read_csv("/data/001nosma_train.csv")
test=pd.read_csv("/data/001nosma_test.csv")
'''

'''
df=pd.read_csv("/data/001sma_train.csv")
test=pd.read_csv("/data/001sma_test.csv")
'''

df=pd.read_csv("/data/001c.csv")
test=pd.read_csv("/data/001c.csv")

data = df 
data = data.dropna(axis=0)
row_count = data.shape[0]  

# set the target column
train_cols = data.columns[1:]  
#print(data[train_cols])
#print(data['R'])
#print(data[train_cols].iloc[0:2089])
train_size = 2000
logit = sm.Logit(data['R'].iloc[0:train_size], data[train_cols].iloc[0:train_size])  
   
#print(data[train_cols].iloc[0:train_size])
# fit the model  
result = logit.fit()   

test_size = row_count - train_size   
# add intercept?? 
#test['intercept'] = 1.0  
predict_cols = test.columns[1:] 

# predict and save the result in predict column  
output = result.predict(test[predict_cols].iloc[test_size:row_count])  

#calculate the accuracy
hit = 0  

for i in range(test_size,row_count-1):
    if output[i] >= 0.5:
        if test['R'][i]==1:
            hit+=1
    
    if output[i] < 0.5:
        if test['R'][i]==0:
            hit+=1      
      
total = row_count-1 - test_size
    
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total))  

