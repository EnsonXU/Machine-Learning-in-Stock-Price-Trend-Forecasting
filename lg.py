# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:31:15 2018

@author: chenxu
"""
import pandas as pd  
#import statsmodels.api as sm  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

'''
df=pd.read_csv("/data/001nosma_train.csv")
test=pd.read_csv("/data/001nosma_test.csv")
'''

'''
df=pd.read_csv("/data/001sma_train.csv")
test=pd.read_csv("/data/001sma_test.csv")
'''

df=pd.read_csv("/data/3Mdata.csv")
test=pd.read_csv("/data/3Mdata.csv")

data = df 
data = data.dropna(axis=0)
row_count = data.shape[0]  

# set the target column
train_cols = data.columns[1:]  
train_cols = data.columns[1:]  
X_train, X_test, Y_train, Y_test = train_test_split(data[train_cols], data['R'], test_size=0.3, random_state=0)
    
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#fit_intercept = False, C = 1e9
model = LogisticRegression(penalty="l2",C=10,solver='newton-cg')
result = model.fit(X_train_std, Y_train)

prepro =result.predict_proba(X_test_std)
acc = result.score(X_test_std,Y_test) 

print ('Total: %d, Precision: %.2f' % (len(Y_test), acc)) 


'''
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
'''
