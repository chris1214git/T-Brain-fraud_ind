import pandas as pd
import os
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from bayes_opt import BayesianOptimization

data_path = '../../data'

random_seed = 20
import json
path = '../code/para_dict/data_list.json'
with open(path,'r',encoding='utf-8') as f:
    para = json.loads(f.read())
    
data_list= para['data_list_FE_AN4']
data_list= para['data_list_FE_AN']
delete_list = para['delete_list_overfit1']

def load_data(data_list):
    data=[]
    for d in data_list:
        x = pd.read_csv('../data/preprocess/{}'.format(d))
        x_null = x.isnull().sum()
        
        print('\n',d,x.shape)
        print("Null columns:\n",x_null[x_null>0])

        if (d=='FE_data1.csv') or (d=='FE_data2.csv'):
            x.fillna(value=-1,inplace=True)

        data.append(x)

    all_data = pd.concat(data,axis=1)
    del data
    all_data_numsum = all_data.isnull().sum()
    print('ALL data shape:',all_data.shape)
    print('ALL data null:')
    print(all_data_numsum[all_data_numsum>0])
    return all_data

all_data = load_data(data_list)
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic',\
                'mchno','etymd','contp','locdt_week']
#                 'ovrlt','insfg','ecfg',\
# 'cano_only_consecutive_stscd2','bacno_consecutive_and_only_ecfg','bacno_consecutive_and_only_ecfg',\
# 'cano_lastday_use_twokind','cano_lastlocdt2','bacno_stscd_equal2','bacno_ecfg_equal1']
## mode

category_list=['csmcu','hcefg','stscd','stocn','etymd','contp','locdt_week']

for c in category_list:
    if all_data[c].dtypes == 'float64':
        all_data[c] = all_data[c].astype('int')
    all_data[c]=all_data[c].astype('category')

for c in all_data.columns[all_data.dtypes==bool]:
    all_data[c]=all_data[c].map({True:1,False:0})


## 切三種不同的訓練集驗證
X_train1 = all_data[all_data['locdt']<=60].drop(columns=delete_list)
y_train1 = all_data[all_data['locdt']<=60]['fraud_ind']
X_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)

y_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']

categorical_features_indices = np.where(X_train1.columns.isin(category_list))[0]
print(X_train1.dtypes[categorical_features_indices])

def lgb_f1_score(y_true, y_pred):
    y_pred = np.round(y_pred) # scikits f1 doesn't like probabilities
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     print()
#     print('tn, fp, fn, tp')
#     print(tn, fp, fn, tp)
    return 'f1', f1_score(y_true, y_pred), True

para_lgb = {
                  'num_leaves':31, 
#                   'max_depth':-1, 
                  'learning_rate':0.1, 
                  'n_estimators':2000,
                  'objective': 'binary',
#                   'subsample': 0., 
#                   'colsample_bytree': 0.5, 
                  'lambda_l1': 10,
                  'lambda_l2': 10,
                  'min_child_weight': 1,
                  'bagging_fraction': 1,  
#                   'bagging_freq': 5,
                  'min_split_gain': 0.5,
#                   'min_child_weight': 1,
#                   'min_child_samples': 5,

                  'random_state': random_seed,
                  'device': 'gpu',
                 }

param_range={
    'num_leaves':(20,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'lambda_l1': (0,30),
    'lambda_l2': (0,30),
    'min_child_weight':(1,3),
}



lgb_clf = LGBMClassifier(**para_lgb)
lgb_clf.fit(X_train1, y_train1,
        eval_set=[(X_train1, y_train1),(X_test1, y_test1)],
        eval_metric=lgb_f1_score,
        early_stopping_rounds=50,
        verbose=5,
#         callbacks=[lgb.record_evaluation(evals_result)]
        )
y_test_pred = lgb_clf.predict(X_test1)
score_max = f1_score(y_test1, y_test_pred)
print(score_max)

