## 分開train
## 加入新feature
# test_data_good_index.npy


import pandas as pd
import os
import numpy as np
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from bayes_opt import BayesianOptimization

data_path = '../../data'

random_seed = 20

import time
t_now = time.localtime( time.time() )
mon = str(t_now.tm_mon) if (t_now.tm_mon)>=10 else '0'+str(t_now.tm_mon)
day = str(t_now.tm_mday) if (t_now.tm_mday)>=10 else '0'+str(t_now.tm_mday)
hour = str(t_now.tm_hour) if (t_now.tm_hour)>=10 else '0'+str(t_now.tm_hour)
minute = str(t_now.tm_min) if (t_now.tm_min)>=10 else '0'+str(t_now.tm_min)
t = mon+day+hour+minute
print('Now:',t)

bagging_time=3
import json
path = '../code/para_dict/data_list.json'
with open(path,'r',encoding='utf-8') as f:
    para = json.loads(f.read())
    
data_list= para['data_list_FE_AN7']
delete_list = para['delete_list_overfit2']

def load_data(data_list):
    data=[]
    for d in data_list:
        x = pd.read_csv('../data/preprocess/{}'.format(d))
        x_null = x.isnull().sum()
        
        print('\n',d,x.shape)
        print("Null columns:\n",x_null[x_null>0])

        if (d=='FE_data1.csv') or (d=='FE_data2.csv'):
            x.fillna(value=-1,inplace=True)
        
        if d[:8]=='FE_data9':
            if d!='FE_data9_raw.csv':
                x = x.drop(columns=['bacno_shift1','bacno_shiftm1'])
        data.append(x)

    all_data = pd.concat(data,axis=1)
    del data
    all_data_numsum = all_data.isnull().sum()
    print('ALL data shape:',all_data.shape)
    print('ALL data null:')
    print(all_data_numsum[all_data_numsum>0])
    return all_data


all_data = load_data(data_list)
X_test_select = pd.read_csv('../data/preprocess/X_test_select_th09_AN7.csv')
X_test_select2 = pd.read_csv('../data/preprocess/X_test_select_th005_AN7.csv')
# print(X_test_select)

category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic',\
                'mchno','etymd','contp','locdt_week']

for c in category_list:
    if all_data[c].dtypes == 'float64':
        all_data[c] = all_data[c].astype('int')
    all_data[c]=all_data[c].astype('category')

for c in all_data.columns[all_data.dtypes==bool]:
    all_data[c]=all_data[c].map({True:1,False:0})

bool_list= ['cano_lastlocdt2_shift1','cano_lastlocdt2_shiftm1','bacno_stscd_equal2_shift1','bacno_stscd_equal2_shiftm1',\
            'bacno_ecfg_equal1_shift1','bacno_ecfg_equal1_shiftm1']
for c in bool_list:
    all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})

    
## 切三種不同的訓練集驗證
X_train = all_data[all_data['locdt']<=90].drop(columns=delete_list)
y_train = all_data[all_data['locdt']<=90]['fraud_ind']
# print(X_test_select.shape)
# for c in X_test_select.columns:
#     print(c)
# print('..................................................................')
# print(X_train.shape)
# for c in X_train.columns:
#     print(c)

X_train = pd.concat([X_train,X_test_select.drop(columns=delete_list)],axis=0)
X_train = pd.concat([X_train,X_test_select2.drop(columns=delete_list)],axis=0)

y_train = pd.concat([y_train,X_test_select['fraud_ind']],axis=0)
y_train = pd.concat([y_train,X_test_select2['fraud_ind']],axis=0)


X_test = all_data[all_data['locdt']>90].drop(columns=delete_list)
y_test = all_data[all_data['locdt']>90]['fraud_ind']
test_data_txkey = all_data[all_data['locdt']>90]['txkey'].copy().values
print(X_train.shape)
print(X_test.shape)

categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]
print(X_train.dtypes[categorical_features_indices])

param_cat={
    'loss_function':'Logloss',
    'eval_metric':'F1',

    'iterations':6000,
    'scale_pos_weight':1,
    'target_border':0.5,
    'random_seed':random_seed,
    'thread_count':1,
    'task_type':"GPU",
    'devices':'0:1',
    'verbose':20,

    'learning_rate':0.03,
    'l2_leaf_reg':1.5928949776908008,#20
    'depth':15,
    'max_leaves':35,
    'bagging_temperature':0.05205316105596142,#10
    'random_strength':10,
    'one_hot_max_size':200,
    'grow_policy':'Lossguide',
}


def train_model_all(X_train_all,y_train_all,X_test_all,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name):
    y_test_pred_cat_all = []
    for i in range(bagging_time):
        param_cat['random_seed']=i
        model = CatBoostClassifier(**param_cat)

        model.fit(
            X_train_all, y_train_all,
            cat_features=categorical_features_indices,    
            verbose=100
        )
        y_test_pred_cat = model.predict_proba(X_test_all)[:,1]
        y_test_pred_cat_all.append(y_test_pred_cat.copy())

    y_test_pred_cat = np.sum(np.array(y_test_pred_cat_all),axis=0)
    for th in [0.25,0.27,0.29,0.31,0.33,0.35,0.37]:
        print('th',th)
        y_test_pred_cat2 = y_test_pred_cat.copy() 
        p_id = y_test_pred_cat>(th*bagging_time)
        n_id = y_test_pred_cat<=(th*bagging_time)
        y_test_pred_cat2[p_id]=1
        y_test_pred_cat2[n_id]=0

        ## write csv
        result = y_test_pred_cat2
        print('{}: prediction positive ratio'.format(result.sum()/result.shape[0]))
        print('{}: training positive ratio'.format(y_train_all.sum()/y_train_all.shape[0]))

        with open('../prediction/submit/{}_th{}.csv'.format(submit_file_name,int(th*100)),'w') as f:
            writer = csv.writer(f)
            writer.writerow(['txkey','fraud_ind'])
            for i in range(result.shape[0]):
                writer.writerow([test_data_txkey[i], result[i]])
                
submit_file_name='submit_add_09_time{}_{}'.format(t[:4],t[4:])
train_model_all(X_train,y_train,X_test,test_data_txkey,0.31,categorical_features_indices,param_cat,submit_file_name)
