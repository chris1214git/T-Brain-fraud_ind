import pandas as pd
import os
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from bayes_opt import BayesianOptimization

data_path = '../../data'

random_seed = 20
import json
path = '../code/para_dict/data_list.json'
with open(path,'r',encoding='utf-8') as f:
    para = json.loads(f.read())
    
data_list= para['data_list_FE_AN9']
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
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic',\
                'mchno','etymd','contp','locdt_week']
#                 'ovrlt','insfg','ecfg',\
# 'cano_only_consecutive_stscd2','bacno_consecutive_and_only_ecfg','bacno_consecutive_and_only_ecfg',\
# 'cano_lastday_use_twokind','cano_lastlocdt2','bacno_stscd_equal2','bacno_ecfg_equal1']
## mode
for c in all_data.columns:
    print(c)
print(all_data.dtypes)

for c in category_list:
    if all_data[c].dtypes == 'float64':
        all_data[c] = all_data[c].astype('int')
    all_data[c]=all_data[c].astype('category')

for c in all_data.columns[all_data.dtypes==bool]:
    all_data[c]=all_data[c].map({True:1,False:0})
    print(all_data[c].value_counts())

bool_list= ['cano_lastlocdt2_shift1','cano_lastlocdt2_shiftm1','bacno_stscd_equal2_shift1','bacno_stscd_equal2_shiftm1',\
            'bacno_ecfg_equal1_shift1','bacno_ecfg_equal1_shiftm1']
for c in bool_list:
    all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})
    print(c)
    print(all_data[c].value_counts(dropna=False))
    
# for c in all_data.columns:
#     print(all_data[c].value_counts().head())
    
## 切三種不同的訓練集驗證
X_train1 = all_data[all_data['locdt']<=60].drop(columns=delete_list)
y_train1 = all_data[all_data['locdt']<=60]['fraud_ind']
X_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)
y_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']

X_train1 = all_data[all_data['locdt']<=30].drop(columns=delete_list)
y_train1 = all_data[all_data['locdt']<=30]['fraud_ind']
X_test1 = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)].drop(columns=delete_list)
y_test1 = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)]['fraud_ind']

categorical_features_indices = np.where(X_train1.columns.isin(category_list))[0]
print(X_train1.dtypes[categorical_features_indices])

param_cat={
    'loss_function':'Logloss',
    'eval_metric':'F1',
    
    'iterations':10000,
    'scale_pos_weight':1,
    'target_border':0.5,
    'random_seed':random_seed,
    'thread_count':1,
    'task_type':"GPU",
    'devices':'0:1',
#     'boosting_type':'Ordered',

    'learning_rate':0.03,
    'l2_leaf_reg':20,#20
    'depth':7,
    'bagging_temperature':0.3,
    'random_strength':10,
    # 'rsm':0.8,

    # 'fold_permutation_block':1,
    # 'feature_border_type':'MinEntropy',
    # 'boosting_type':'Ordered',
    # 'leaf_estimation_backtracking':'Armijo',
    
    'one_hot_max_size':200,
    'grow_policy':'Lossguide',
}

param_cat={
    'loss_function':'Logloss',
    'eval_metric':'F1',
    
    'iterations':10000,
    'scale_pos_weight':1,
    'target_border':0.5,
    'random_seed':random_seed,
    'thread_count':1,
    'task_type':"GPU",
    'devices':'0:1',
#     'boosting_type':'Ordered',

    'learning_rate':0.03,
    'l2_leaf_reg':20,#20
    'depth':7,
    'bagging_temperature':0.3,
    'random_strength':10,
    # 'rsm':0.8,

    # 'fold_permutation_block':1,
    # 'feature_border_type':'MinEntropy',
    # 'boosting_type':'Ordered',
    # 'leaf_estimation_backtracking':'Armijo',
    
    'one_hot_max_size':200,
    'grow_policy':'Lossguide',
}

param_range={
#     'depth':(5,11.9),
    'depth':(5,16.9),
    
#     'max_leaves':(31,31.5),#(20,45),
    'max_leaves':(20,45),
    'l2_leaf_reg':(1,100),#(5,50),
    'bagging_temperature':(0.01,5)#(0.1,5),    
}

def cat_train(depth,max_leaves,l2_leaf_reg,bagging_temperature):
    param_cat['depth']=int(depth)
    param_cat['max_leaves']=int(max_leaves)
    param_cat['l2_leaf_reg']=l2_leaf_reg
    param_cat['bagging_temperature']=bagging_temperature
    
    model = CatBoostClassifier(**param_cat)
    model.fit(X_train1, y_train1,
    cat_features=categorical_features_indices,    
    eval_set=(X_test1, y_test1),
    early_stopping_rounds=1000,
    verbose=500) 
     
    score_max = model.get_best_score()['validation']['F1:use_weights=true']
    print(int(depth),int(max_leaves),l2_leaf_reg,bagging_temperature)
    print(score_max)
    
    with open('./Bayes_result/Bayes_result.txt','a') as f:
        print('depth',int(depth),file=f)
        print('max_leaves',int(max_leaves),file=f)
        print('l2_leaf_reg',l2_leaf_reg,file=f)
        print('bagging_temperature',bagging_temperature,file=f)
        print(score_max,file=f)
        print('',file=f)
    
    return score_max

with open('./Bayes_result/Bayes_result.txt','a') as f:
    print(data_list,file=f)
    print(delete_list,file=f)
    
cat_opt = BayesianOptimization(cat_train,param_range) 
cat_opt.maximize(n_iter=150, init_points=random_seed)
print(cat_opt.max)
with open('./Bayes_result/Bayes_result.txt','a') as f:
    print('Max para',cat_opt.max,file=f)

# model = CatBoostClassifier(**param_cat)
# model.fit(
#     X_train1, y_train1,
#     cat_features=categorical_features_indices,    
#     eval_set=(X_test1, y_test1),
#     verbose=20
# )
# print('Model is fitted: ' + str(model.is_fitted()))
# print('Model params:')
# print(model.get_params())

