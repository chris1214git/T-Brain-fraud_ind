import pandas as pd
import os
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from bayes_opt import BayesianOptimization
from util import util
from util import build_model

random_seed = 20

para = util.load_json('data_list.json')
param_cat = util.load_json('catboost_para.json')['param_cat3_2']

data_list= para['data_list_FE_AN11']
delete_list = para['delete_list_overfit6']
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic',\
                'mchno','etymd','contp','locdt_week']
bool_list= ['cano_lastlocdt2_shift1','cano_lastlocdt2_shiftm1','bacno_stscd_equal2_shift1','bacno_stscd_equal2_shiftm1',\
            'bacno_ecfg_equal1_shift1','bacno_ecfg_equal1_shiftm1']

all_data = build_model.load_data(data_list)
all_data = build_model.transform_data(all_data,category_list,bool_list)
for c in bool_list:
    if c in all_data.columns:
        all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})
        print(c)
        print(all_data[c].value_counts(dropna=False))
        print(all_data[c].value_counts().head())
    
## 切三種不同的訓練集驗證
X_train1 = all_data[all_data['locdt']<=60].drop(columns=delete_list)
y_train1 = all_data[all_data['locdt']<=60]['fraud_ind']
X_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)
y_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']

# X_train_all = all_data[all_data['locdt']<=90].drop(columns=delete_list)
# y_train_all = all_data[all_data['locdt']<=90]['fraud_ind'] 
X_test_all = all_data[all_data['locdt']>90] .drop(columns=delete_list)
y_test_all = all_data[all_data['locdt']>90]['fraud_ind'] 

categorical_features_indices = np.where(X_train1.columns.isin(category_list))[0]
print(X_train1.dtypes[categorical_features_indices])

model = CatBoostClassifier(**param_cat)
model.fit(X_train1, y_train1,
cat_features=categorical_features_indices,    
eval_set=(X_test1, y_test1),
early_stopping_rounds=800,
verbose=500) 

y_test_pred_cat = model.predict_proba(X_test_all)[:,1]
print(y_test_pred_cat.sum(),y_test_pred_cat.shape[0])

for th in [0.4,0.6,0.8,0.9,0.95]:
    p_id = y_test_pred_cat>(th)
    n_id = y_test_pred_cat<=(th)
    y_test_pred_cat2 = y_test_pred_cat.copy()
    y_test_pred_cat2[p_id]=1
    y_test_pred_cat2[n_id]=0
    print(y_test_pred_cat2.sum(),y_test_pred_cat2.sum()/y_test_pred_cat2.shape[0])
    X_test_all2 = all_data[all_data['locdt']>90]
    X_test_all2 = X_test_all2.loc[p_id]
    X_test_all2['fraud_ind']=1
    X_test_all2.to_csv('../data/preprocess/X_test_select_th{}_AN11.csv'.format(int(th*100)),index=False)
