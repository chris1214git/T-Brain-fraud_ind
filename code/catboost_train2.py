#簡單將某個參數,data list,delete list訓練多次，輸出機率平均

import pandas as pd
import os
import numpy as np
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from bayes_opt import BayesianOptimization
from util import util
from util import build_model
    
para = util.load_json('data_list.json')
param_cat = util.load_json('catboost_para.json')['param_cat3']
iterations = 8000
param_cat['iterations']=iterations

data_path = '../data/preprocess/'
random_seed = 20
bagging_time=10
data_list= para['data_list_FE_AN7']
delete_list = para['delete_list_overfit1']
delete_list = ['bacno', 'locdt', 'loctm', 'cano', 'fraud_ind', 'mchno_fraud_mean', 'mcc_fraud_mean', 'acqic_fraud_mean', 'bacno_lastlocdt', 'cano_lastlocdt', 'cano_lastday_use_twokind', 'csmcu', 'bacno_csmcu_lastday_shiftm1', 'flg_3dsmk_shift1', 'bacno_lastlocdt_shiftm1', 'flbmk_shiftm1', 'contp_shift1', 'flbmk', 'bacno_csmcu_lastday_shift1', 'bacno_lastlocdt_shift1', 'cano_lastlocdt2_shift1', 'cano_lastlocdt2_shiftm1', 'bacno_stscd_equal2', 'bacno_stocn_lastday_shiftm1', 'csmcu_value_counts_all', 'cano_only_consecutive_stscd2_shiftm1', 'bacno_ecfg_mode', 'bacno_stocn_lastday_shift1', 'bacno_consecutive_and_only_ecfg_shift1', 'flbmk_shift1', 'cano_stscd_mode', 'ovrlt', 'cano_only_consecutive_stscd2', 'cano_ecfg_mode', 'cano_csmcu_ismode', 'hcefg_shift1', 'cano_lastday_use_twokind_shiftm1', 'cano_lastday_use_shift1', 'cano_stscd_nunique', 'cano_lastday_use_twokind_shift1', 'bacno_csmcu_ismode', 'insfg_shiftm1', 'ovrlt_shift1', 'bacno_stscd_mode', 'cano_only_consecutive_stscd2_shift1', 'ovrlt_shiftm1', 'bacno_stscd_equal2_shiftm1', 'bacno_hcefg_mode', 'iterm', 'insfg_shift1', 'bacno_stscd_equal2_shift1', 'bacno_ecfg_equal1_shift1']

t = util.get_time_stamp()
print('Now:',t)
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc',\
               'acqic','mchno','etymd','contp','locdt_week']
bool_list= ['cano_lastlocdt2_shift1','cano_lastlocdt2_shiftm1','bacno_stscd_equal2_shift1','bacno_stscd_equal2_shiftm1',\
            'bacno_ecfg_equal1_shift1','bacno_ecfg_equal1_shiftm1']

# main
all_data = build_model.load_data(data_list)
all_data = build_model.transform_data(all_data,category_list,bool_list)

# 切訓練集
X_train = all_data[all_data['locdt']<=90].drop(columns=delete_list)
y_train = all_data[all_data['locdt']<=90]['fraud_ind']
X_test = all_data[all_data['locdt']>90].drop(columns=delete_list)
y_test = all_data[all_data['locdt']>90]['fraud_ind']
test_data_txkey = all_data[all_data['locdt']>90]['txkey'].copy().values
print(X_train.shape)
print(X_test.shape)
categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]

def train_model_all(X_train_all,y_train_all,X_test_all,test_data_txkey,categorical_features_indices,param_cat,submit_file_name):
    y_test_pred_cat_all = []
    for i in range(bagging_time):
        print(i)
        param_cat['random_seed']=i
        model = CatBoostClassifier(**param_cat)

        model.fit(
            X_train_all, y_train_all,
            cat_features=categorical_features_indices,    
            verbose=300
        )
        y_test_pred_cat = model.predict_proba(X_test_all)[:,1]
        y_test_pred_cat_all.append(y_test_pred_cat.copy())

    y_test_pred_cat = np.sum(np.array(y_test_pred_cat_all),axis=0)
    result = y_test_pred_cat/bagging_time
    
    with open('../prediction/submit/{}.csv'.format(submit_file_name),'w') as f:
        writer = csv.writer(f)
        writer.writerow(['txkey','fraud_ind'])
        for i in range(result.shape[0]):
            writer.writerow([test_data_txkey[i], result[i]])
    
    util.describe_model(submit_file_name,param_cat,data_list,delete_list,bagging_time)

submit_file_name='submit_time{}_{}_it{}'.format(t[:4],t[4:],iterations)
train_model_all(X_train,y_train,X_test,test_data_txkey,categorical_features_indices,param_cat,submit_file_name)
