import pandas as pd
import os
import numpy as np
import math

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
import eli5
from eli5.sklearn import PermutationImportance
import time
import csv

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

def transform_data(all_data,category_list,bool_list):
    for c in category_list:
        if all_data[c].dtypes == 'float64':
            all_data[c] = all_data[c].astype('int')
        all_data[c]=all_data[c].astype('category')

    for c in all_data.columns[all_data.dtypes==bool]:
        all_data[c]=all_data[c].map({True:1,False:0})

    for c in bool_list:
        if c in all_data.columns:
            all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})
    
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
            verbose=100,
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

        with open('../prediction/submit/{}_th{}'.format(submit_file_name,int(th*100)),'w') as f:
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


def permutation_test(model,X_test1,y_test1):
    perm = PermutationImportance(model1, random_state=random_seed).fit(X_test1, y_test1)
    feature_importance1 = pd.DataFrame({'feature':X_test1.columns.tolist(),'importance':perm.feature_importances_})
    delete_col1 = feature_importance1.iloc[:,0][(feature_importance1['importance'].values)<=0.0000]
    print(delete_col1)

    return delete_col1
