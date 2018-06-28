# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:16:05 2018

@author: ziv
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:09:25 2018

@author: ziv
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, cross_val_score


from bayes_opt import BayesianOptimization

import xgboost as xgb

class xgb_model:
    def __init__(self,train,y_train,random_state,num_rounds,num_iter,eta,init_points,silent=1):
        
        self.train = train
        self.y_train=y_train

#    test = pd.read_pickle('test_seringe.pkl')
   
        self.xgtrain = xgb.DMatrix(X, label=y)

    
      
        
        self.RMSEbest = 10.
        self.ITERbest = 0
    
          
        self.num_rounds = num_rounds
        self.random_state = random_state
        self.num_iter = num_iter
        self.init_points = init_points
        self.eta=eta
        self.params = {
            'eta':self.eta,
            'silent': silent,
            'eval_metric': 'rmse',
    #        'verbose_eval': True,
            'seed': self.random_state
        }

        self.xgbBO = BayesianOptimization(self.xgb_evaluate, {'min_child_weight': (0, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 10),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })

        self.xgbBO.maximize(init_points=self.init_points, n_iter=self.num_iter)
    
#        self.xgbBO.points_to_csv('dfg5.csv')
#    xgbBO.explore()
        self.p = self.xgbBO.res['max']
        self.all_res=self.xgbBO.res['all']
#    https://www.kaggle.com/kobakhit/xgboost/code
        self.folds=5
        self.num_boost_round=int(self.ITERbest*(1+(1/self.folds)))




        self.best_RMSE = round((-1.0 * self.xgbBO.res['max']['max_val']), 6)
        self.max_depth = self.xgbBO.res['max']['max_params']['max_depth']
        self.gamma = self.xgbBO.res['max']['max_params']['gamma']
        self.min_child_weight = self.xgbBO.res['max']['max_params']['min_child_weight']
        self.alpha = self.xgbBO.res['max']['max_params']['alpha']
        self.subsample = self.xgbBO.res['max']['max_params']['subsample']
        self.colsample_bytree = self.xgbBO.res['max']['max_params']['colsample_bytree']

#    print("\n Best RMSE value: %f" % best_RMSE)
#    print(" Best XGboost parameters:")
#    print(" max_depth=%d gamma=%f min_child_weight=%f  subsample=%f colsample_bytree=%f" % (int(max_depth), gamma, min_child_weight,  subsample, colsample_bytree))




#    paramt = {
##              'booster' : 'gbtree',
#              'max_depth' : max_depth.astype(int),
#              'gamma' : gamma,
#              'eta' : eta,
##              'objective': 'reg:linear',
#              'nthread' : 8,
#              'silent' : True,
#              'eval_metric': 'rmse',
#              'subsample' : subsample,
#              'colsample_bytree' : colsample_bytree,
#              'min_child_weight' : min_child_weight,
#              'alpha' : alpha.astype(int),
#              'seed' : 1001
#              }
#    xgbr1 = xgb.cv(paramt, xgtrain, num_boost_round=2500, nfold=10,early_stopping_rounds=20)
#   
#    xgbr1
#    xgbr1[['train-rmse-mean', 'test-rmse-mean']].plot()
#    len(xgbr1)
#
#    xgbr1 = xgb.train(paramt, xgtrain, num_boost_round=int(ITERbest*(1+(1/folds))))






        self.model_xgb = xgb.XGBRegressor(colsample_bytree=self.colsample_bytree, gamma=self.gamma, 
                                     learning_rate=self.eta, max_depth=self.max_depth.astype(int), 
                                     min_child_weight=self.min_child_weight, n_estimators= self.num_boost_round,#   len(xgbr1),
                                     reg_alpha=self.alpha, 
                                     subsample=self.subsample, silent=1,
                                      nthread = -1)
        
        
#        self.score = self.rmsle_cv(model_xgb)
#        return (model_xgb)
#        print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
                
    def GetModel(self):
        
        return (self.model_xgb)
  
    def TrainModel(self):
        self.score = self.rmsle_cv(self.model_xgb)
        
        return (self.score)
  
    
    
    def xgb_evaluate(self,min_child_weight,
                     colsample_bytree,
                     max_depth,
                     subsample,
                     gamma,alpha):
        self.RMSEbest
        self.ITERbest
    #    self.params
        self.params['min_child_weight'] = int(min_child_weight)
        self.params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
        self.params['max_depth'] = int(max_depth)
        self.params['subsample'] = max(min(subsample, 1), 0)
        self.params['gamma'] = max(gamma, 0)
        self.params['alpha'] = max(alpha, 0)
        
    #    params['max_delta_step'] = max(alpha, 0)
    
    
        self.cv_result = xgb.cv(self.params, self.xgtrain, num_boost_round=self.num_rounds, nfold=5,
                 seed=self.random_state,
                 callbacks=[xgb.callback.early_stop(50)])
    #    global globvar    # Needed to modify global copy of globvar
    #    globvar = cv_result
        score=self.cv_result['test-rmse-mean'].values[-1]
        if ( score < self.RMSEbest ):
            self.RMSEbest = score
            self.ITERbest = len(self.cv_result)
        return -self.cv_result['test-rmse-mean'].values[-1]
    
       
        
     
    
    
    def rmsle_cv(self,model):
        self.n_folds = 5
    
        kf = KFold(self.n_folds, shuffle=True, random_state=self.random_state).get_n_splits(self.train.values)
        rmse= np.sqrt(-cross_val_score(model, self.train.values, self.y_train, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)


def rmsle_cv(model,train,y_train,n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


if __name__ == '__main__':
    
    
    X = pd.read_pickle('train_seringe.pkl')

#    test = pd.read_pickle('test_seringe.pkl')
    y=pd.read_pickle('y_seringe.pkl')


   
    model=xgb_model(X,y,random_state=11,num_rounds=3000,num_iter=10,eta=0.05,init_points=5)
    model.best_RMSE
    mxgb=model.GetModel()
    score2=model.TrainModel()
    
    score = rmsle_cv(mxgb,X,y,5)
    print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))




    