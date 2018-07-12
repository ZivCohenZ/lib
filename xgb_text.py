# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:19:36 2018

@author: ziv
"""

        
        
import pandas as pd
import numpy as np
import xgboost as xgb   
from sklearn.model_selection import train_test_split
class TextTransform ():
    def __init__(self,file,str_x,rows_to_read,num_of_features,str_class='class',train=True):
        self.num_of_features=num_of_features
        self.str_x=str_x
        self.rows_to_read=rows_to_read
        self.file=file
        self.df=pd.read_csv(self.file,nrows=self.rows_to_read)
        self.num_of_rows=self.df.shape[0]
        self.data = []
        self.str_class=str_class
        self.train=train
        pass
    
    def wordToNum(self):

        for x in self.df[self.str_x].values:
            row = np.zeros(self.num_of_features, dtype=int) 
            for xi, i in zip(list(str(x)), np.arange(self.num_of_features)):
                row[i] = ord(xi)
            self.data.append(row)
            
    def createFeaturMatrix(self):
        if (self.train):
            self.data = np.concatenate((self.data[:-3],self.data[1:-2],self.data[2:-1],self.data[3:]),axis=1)
        else:
            self.data = np.concatenate((np.zeros([2,self.num_of_features],np.int8),self.data,np.zeros([1,self.num_of_features],np.int8)),axis=0)
   
            self.data=np.concatenate((self.data[:-3],self.data[1:-2],self.data[2:-1],self.data[3:]),axis=1)
        
    def targetToNum(self):
        self.target =  pd.factorize(self.df[self.str_class])
        self.labels = self.target[1]
        self.target = self.target[0][2:-1]



if __name__=='__main__':
    
    
    train_data_csv='../input/en_train.csv'
    test_data_csv='../input/en_test_2.csv'
    
    ott_train=TextTransform (train_data_csv,'before',10000,11,str_class='class',train=True)
    ott_test=TextTransform (test_data_csv,'before',10000,11,str_class='class',train=False)
    
    del(train_data_csv,test_data_csv)
    
    ott_train.wordToNum()
    ott_train.createFeaturMatrix()
    ott_train.targetToNum()
   
    
    ott_test.wordToNum()
    ott_test.createFeaturMatrix()
    
    x_train, x_valid, y_train, y_valid= train_test_split(ott_train.data, ott_train.target, test_size=0.2, random_state=2018)
    
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
#    dtrain = xgb.DMatrix(ott_train.data, label=ott_train.target)
#    watchlist = [(dtrain, 'train')]
    param = {
        'eta': 0.4,
        'max_depth':10,
        'objective':'multi:softmax',
        'num_class':len(ott_train.labels),
        'eval_metric':'merror',
        'subsample': 1,
        'colsample_bytree': 1,
        'silent':1,
        'seed':2018,
    }
    nrounds=300
    model = xgb.train(param, dtrain, nrounds, watchlist,verbose_eval=3,early_stopping_rounds=20,)
    
    
    

    
    dtest = xgb.DMatrix(ott_test.data)
    target_pred = model.predict(dtest)
    target_pred = [ott_train.labels[int(x)] for x in target_pred]

    test=ott_test.df

    test['class']=target_pred
    test.to_csv('../input/test_pred_class4.csv')
