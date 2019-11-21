# 3 kind of feature importance to delete features
# object importance


import pandas as pd
import os
import sys
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool

import time
import csv
import eli5
from eli5.sklearn import PermutationImportance

from util import util
from util import build_model

para = util.load_json('data_list.json')
param_cat = util.load_json('catboost_para.json')[sys.argv[3]]
param_cat['iterations']=13000
data_list= para[sys.argv[1]]
delete_list = para[sys.argv[2]]

th=0.31

bagging_time = 3
random_seed = 20

t = util.get_time_stamp()
print('Now:',t)

category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic','mchno','etymd','contp','locdt_week']
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic','mchno','etymd','contp','locdt_week',\
               'leakage_complex1','leakage_complex2','leakage_complex3','leakage_complex4','contp_ecfg','csmcu_ecfg']
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic','mchno','etymd','contp','locdt_week',\
               'leakage_complex1','leakage_complex2','leakage_complex3','leakage_complex4','contp_ecfg','csmcu_ecfg']

# param_cat={
#     'loss_function':'Logloss',
#     'eval_metric':'F1',
    
#     'iterations':6000,
#     'scale_pos_weight':1,
#     'target_border':0.5,
#     'random_seed':random_seed,
#     'thread_count':1,
#     'task_type':"GPU",
#     'devices':'0:1',
#     'verbose':20,

#     # 'min_data_in_leaf':1,
#     # 'has_time':True,

#     'learning_rate':0.03,
#     'l2_leaf_reg':1.5928949776908008,#20
#     'depth':15,
#     'max_leaves':35,
#     'bagging_temperature':0.05205316105596142,#10
#     'random_strength':10,
#     # 'rsm':0.8,

#     # 'fold_permutation_block':1,
#     # 'feature_border_type':'MinEntropy',
#     # 'boosting_type':'Ordered',
#     # 'leaf_estimation_backtracking':'Armijo',
    
#     'one_hot_max_size':200,
#     'grow_policy':'Lossguide',
#     # 'grow_policy':'Depthwise',
# }
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
    all_data_numsum = all_data.isnull().sum()
    print('ALL data null:')
    print(all_data_numsum[all_data_numsum>0])
    return all_data

def parse_validation(all_data,i,delete_list):
    train_data_txkey = all_data[all_data['locdt']<=90]['txkey'].copy().values
    test_data_txkey = all_data[all_data['locdt']>90]['txkey'].copy().values

    if i==0:
        X_train = all_data[all_data['locdt']<=60].drop(columns=delete_list)
        y_train = all_data[all_data['locdt']<=60]['fraud_ind']
        X_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)
        y_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']
    elif i==1:
        X_train = all_data[all_data['locdt']<=45].drop(columns=delete_list)
        y_train = all_data[all_data['locdt']<=45]['fraud_ind']
        X_test = all_data[(all_data['locdt']>45) & (all_data['locdt']<=90)].drop(columns=delete_list)
        y_test = all_data[(all_data['locdt']>45) & (all_data['locdt']<=90)]['fraud_ind']
    elif i==2:
        X_train = all_data[all_data['locdt']<=30].drop(columns=delete_list)
        y_train = all_data[all_data['locdt']<=30]['fraud_ind']
        X_test = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)].drop(columns=delete_list)
        y_test = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)]['fraud_ind']
    elif i==3:
        X_train = all_data[all_data['locdt']<=90].drop(columns=delete_list)
        y_train = all_data[all_data['locdt']<=90]['fraud_ind'] 
        X_test = all_data[all_data['locdt']>90] .drop(columns=delete_list)
        y_test = all_data[all_data['locdt']>90]['fraud_ind'] 

    return X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey
    

def train_model_validation(X_train1, y_train1, X_test1, y_test1, th, categorical_features_indices,param_cat):
    y_test1_pred_all = []
    for i in range(bagging_time):
        param_cat['random_seed'] = i
        model = CatBoostClassifier(**param_cat)

        model.fit(
            X_train1, y_train1,
            cat_features=categorical_features_indices,    
            eval_set=(X_test1, y_test1),
            # early_stopping_rounds=2000,
            verbose=600,
        )
        y_test1_pred = model.predict_proba(X_test1,verbose=True)[:,1]
        y_test1_pred_all.append(y_test1_pred.copy())
    
    y_test1_pred_all = np.array(y_test1_pred_all)
    y_test1_pred = np.sum(y_test1_pred_all,axis=0)
    # print(y_test1_pred_all)
    # print(y_test1_pred)

    p_id = y_test1_pred>(th*bagging_time)
    n_id = y_test1_pred<=(th*bagging_time)
    y_test1_pred[p_id]=1
    y_test1_pred[n_id]=0

    F1_score = f1_score(y_test1, y_test1_pred)

    print('Model is fitted: ' + str(model.is_fitted()))
    print('\nF1 score',F1_score)
    tn, fp, fn, tp = confusion_matrix(y_test1, y_test1_pred).ravel()
    print('tn fp fn tp')
    print(tn, fp, fn, tp)
    print('Percision', tp/(tp+fp))
    print('Recall',tp/(tp+fn))
        
    return model, F1_score


def train_model_all(X_train_all,y_train_all,X_test_all,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name):
    y_test_pred_cat_all = []
    for i in range(bagging_time):
        param_cat['random_seed']=i
        model = CatBoostClassifier(**param_cat)

        model.fit(
            X_train_all, y_train_all,
            cat_features=categorical_features_indices,    
            verbose=600
        )
        y_test_pred_cat = model.predict_proba(X_test_all)[:,1]
        y_test_pred_cat_all.append(y_test_pred_cat.copy())

    y_test_pred_cat = np.sum(np.array(y_test_pred_cat_all),axis=0)
    
    ## write csv
    result = y_test_pred_cat
    print('{}: prediction positive ratio'.format(result.sum()/result.shape[0]))
    print('{}: training positive ratio'.format(y_train_all.sum()/y_train_all.shape[0]))

    with open('../prediction/submit/{}'.format(submit_file_name),'w') as f:
        writer = csv.writer(f)
        writer.writerow(['txkey','fraud_ind'])
        for i in range(result.shape[0]):
            writer.writerow([test_data_txkey[i], result[i]])

# In[34]:

def feature_importance_test(model,X_test1,y_test1,categorical_features_indices,importance_type,print_c=False):
    train_pool=Pool(X_test1, y_test1,cat_features=categorical_features_indices)
    feature_names = X_test1.columns
    delete_col = []
    feature_importances = 0

    if importance_type=='PredictionValuesChange':
        feature_importances = model.get_feature_importance(train_pool,type=importance_type,thread_count=4)
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            if score<0.01:
                delete_col.append(name)
    elif importance_type=='LossFunctionChange':
        feature_importances = model.get_feature_importance(train_pool,type=importance_type,thread_count=4)
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            if score<=0.0:
                delete_col.append(name)    
    elif importance_type=='ShapValues':
        feature_importances = model.get_feature_importance(train_pool,type=importance_type,thread_count=4)
        feature_importances = np.mean(abs(feature_importances),axis=0)
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            if score<0.001:
                delete_col.append(name)
    else:
        print('invalid importance type')
    if print_c==True:
        print('\nFeature importance')
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            print('{}: {}'.format(name, score))
    return delete_col

def main():
    submit_file_name='submit_cat_time{}_{}.csv'.format(t[:4],t[4:])
    all_data = load_data(data_list)

    print('hello')
    for c in all_data.columns:
        # print(c)
        if c in category_list:
            all_data[c]=all_data[c].astype('int')
            all_data[c]=all_data[c].astype('category')
            print(all_data[c].value_counts(dropna=False))

    bool_list= ['cano_lastlocdt2_shift1','cano_lastlocdt2_shiftm1','bacno_stscd_equal2_shift1','bacno_stscd_equal2_shiftm1',\
              'bacno_ecfg_equal1_shift1','bacno_ecfg_equal1_shiftm1']

    for c in bool_list:
        all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})
#         print(c)
#         print(all_data[c].value_counts(dropna=False))
    
    models=[]
    F1_scores=[]
    for i in range(3):
        print('.........................')
        print('validation set',i)
        print('.........................')
        X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey=parse_validation(all_data,i,delete_list)
        categorical_features_indices = np.where(X_train.columns.isin(category_list))[0] 
        model, F1_score = train_model_validation(X_train, y_train, X_test, y_test, th, categorical_features_indices,param_cat)
        F1_scores.append(F1_score)
        models.append(model)

    with open('../prediction/log.txt','a') as f:
        print('{}'.format(submit_file_name),file=f)
        print('data list',sys.argv[1],file=f)
        print('delete_list:',sys.argv[2],file=f)
        print('para:',param_cat,file=f)
        print('F1_score:',F1_scores[0],F1_scores[1],F1_scores[2],file=f)
        print('average F1_score:', (F1_scores[0]+F1_scores[1]+F1_scores[2])/3,file=f)
        print('\n',file=f)

    X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey=parse_validation(all_data,3,delete_list)
    categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]
    train_model_all(X_train,y_train,X_test,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name)

    feature_important_types=['PredictionValuesChange']#,'LossFunctionChange','ShapValues']
    for fi_type in feature_important_types:
        print('.........................')
        print(fi_type)
        print('.........................')
        F1_scores=[]
        delete_cols=[]
        for i in range(3):
            if i==0:
                print_c=True
            else:
                print_c=False
            X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey=parse_validation(all_data,i,delete_list)
            categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]
            delete_col = feature_importance_test(models[i],X_test,y_test,categorical_features_indices,fi_type,print_c)
            delete_cols.append(delete_col)

        delete_col2=[]
        for i in delete_cols[0]:
            if i in delete_cols[1]:
                if i in delete_cols[2]:
                    delete_col2.append(i)

        with open('../prediction/feature_importance_results.txt','a') as f:
            print('{}'.format(submit_file_name),file=f)
            print(fi_type,file=f)
            print('data list',data_list,file=f)
            print('delete_list:\n{}'.format(delete_list),file=f)
            print('Inner delete_col:\n{}'.format(delete_col2),file=f)
            print('\n',file=f)
            
        delete_list2 = delete_list+delete_col2
        
        F1_scores=[]
        for i in range(3):
            print('after deleting less significant features')
            print('.........................')
            print('validation set',i)
            print('.........................')
            X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey=parse_validation(all_data,i,delete_list2)
            categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]
            model, F1_score = train_model_validation(X_train, y_train, X_test, y_test, th, categorical_features_indices,param_cat)
            F1_scores.append(F1_score)
        
        with open('../prediction/log.txt','a') as f:
            print('{} feature_importance:{}'.format(submit_file_name,fi_type),file=f)
            print('data list',sys.argv[1],file=f)
            print('delete_list:',delete_list2,file=f)
            print('para:',param_cat,file=f)
            print('F1_score:',F1_scores[0],F1_scores[1],F1_scores[2],file=f)
            print('average F1_score:', (F1_scores[0]+F1_scores[1]+F1_scores[2])/3,file=f)
            print('\n',file=f)
        
        submit_file_name2='submit_cat_time{}_{}_{}.csv'.format(t[:4],t[4:],fi_type)
        X_train,y_train,X_test,y_test,train_data_txkey,test_data_txkey=parse_validation(all_data,3,delete_list2)
        categorical_features_indices = np.where(X_train.columns.isin(category_list))[0]
        train_model_all(X_train,y_train,X_test,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name2)
    
if __name__ == '__main__':
    main()