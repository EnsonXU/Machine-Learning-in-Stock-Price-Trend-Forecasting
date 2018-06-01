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

df=pd.read_csv("/data/001sma_train.csv")
test=pd.read_csv("/data/001sma_test.csv")

#print (df.head())
#show the summary of the data
'''
print (df.describe())
print (df.std())
'''
data = df 
#data = data.dropna(axis=0)
#combos = copy.deepcopy(data) 


#data['intercept'] = 1.0  
#print(data)
 
# set the target column
train_cols = data.columns[1:]  
   
logit = sm.Logit(data['R'], data[train_cols])  
   
# fit the model  
result = logit.fit()   
 
# add intercept?? 
#test['intercept'] = 1.0  
predict_cols = test.columns[1:] 

# predict and save the result in predict column  
test['predict'] = result.predict(test[predict_cols])  
   
#calculate the accuracy
total = 0  
hit = 0  

for value in test.values:  
  predict = value[-1]  
  R = int(value[0])
  total += 1 
  #if the prediction is larger than 0.5, means the price is rising 
  if predict >= 0.5: 
      if R == 1:  
          hit += 1 
   #if the prediction is smaller than 0.5, means the price is falling
  if predict < 0.5:
      if R == 0:
          hit += 1
   
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total))  
