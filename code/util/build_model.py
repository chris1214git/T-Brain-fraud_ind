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

        data.append(x)

    all_data = pd.concat(data,axis=1)
    all_data_numsum = all_data.isnull().sum()
    print('ALL data null:')
    print(all_data_numsum[all_data_numsum>0])
    return all_data

def parse_validation(all_data,delete_list):
    X_train1 = all_data[all_data['locdt']<=60].drop(columns=delete_list)
    y_train1 = all_data[all_data['locdt']<=60]['fraud_ind']
    X_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)
    y_test1 = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']

    X_train2 = all_data[all_data['locdt']<=45].drop(columns=delete_list)
    y_train2 = all_data[all_data['locdt']<=45]['fraud_ind']
    X_test2 = all_data[(all_data['locdt']>45) & (all_data['locdt']<=90)].drop(columns=delete_list)
    y_test2 = all_data[(all_data['locdt']>45) & (all_data['locdt']<=90)]['fraud_ind']

    X_train3 = all_data[all_data['locdt']<=30].drop(columns=delete_list)
    y_train3 = all_data[all_data['locdt']<=30]['fraud_ind']
    X_test3 = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)].drop(columns=delete_list)
    y_test3 = all_data[(all_data['locdt']>30) & (all_data['locdt']<=90)]['fraud_ind']


    test_data_txkey = all_data[all_data['locdt']>90]['txkey'].copy().values
    X_train_all = all_data[all_data['locdt']<=90].drop(columns=delete_list) 
    y_train_all = all_data[all_data['locdt']<=90]['fraud_ind'] 

    X_test_all = all_data[all_data['locdt']>90].drop(columns=delete_list) 
    # y_test_all = all_data[all_data['locdt']>90]['fraud_ind'] 
    return X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2,\
            X_train3, y_train3, X_test3, y_test3, X_train_all, y_train_all, X_test_all, test_data_txkey


def train_model_validation(X_train1, y_train1, X_test1, y_test1, th, categorical_features_indices,param_cat):
    model = CatBoostClassifier(**param_cat)

    model.fit(
        X_train1, y_train1,
        cat_features=categorical_features_indices,    
        eval_set=(X_test1, y_test1),
        early_stopping_rounds=200,
    #     use_best_model=True,
        silent=False,
        verbose=100,
    #     plot=True,
    )
    print('Model is fitted: ' + str(model.is_fitted()))

    train_pool=Pool(X_test1, y_test1, cat_features=categorical_features_indices)
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_test1.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('\nFeature importance')
        print('{}: {}'.format(name, score))

    y_test1_pred = model.predict_proba(X_test1,verbose=True)[:,1]

    y_test1_pred[y_test1_pred>th]=1
    y_test1_pred[y_test1_pred<=th]=0
    F1_score = f1_score(y_test1, y_test1_pred)
    print('\nF1 score',F1_score)

    tn, fp, fn, tp = confusion_matrix(y_test1, y_test1_pred).ravel()
    print('tn fp fn tp')
    print(tn, fp, fn, tp)
    print('Percision', tp/(tp+fp))
    print('Recall',tp/(tp+fn))

    return model, F1_score


def train_model_all(X_train_all,y_train_all,th,categorical_features_indices,param_cat,submit_file_name):
    model = CatBoostClassifier(**param_cat)

    model.fit(
        X_train_all, y_train_all,
        cat_features=categorical_features_indices,    
        verbose=100,
        silent=False
    )
    y_test_pred_cat = model.predict_proba(X_test_all)[:,1]

    y_test_pred_cat[y_test_pred_cat>th]=1
    y_test_pred_cat[y_test_pred_cat<=th]=0


    # ## write csv
    result = y_test_pred_cat
    test_data_txkey = all_data[all_data['locdt']>90]['txkey'].values

    print('{}: prediction positive ratio'.format(result.sum()/result.shape[0]))
    print('{}: training positive ratio'.format(y_train_all.sum()/y_train_all.shape[0]))

    with open('../prediction/{}'.format(submit_file_name),'w') as f:
        writer = csv.writer(f)
        writer.writerow(['txkey','fraud_ind'])
        for i in range(result.shape[0]):
            writer.writerow([test_data_txkey[i], result[i]])

# In[34]:

def permutation_test(model,X_test1,y_test1):
    perm = PermutationImportance(model1, random_state=random_seed).fit(X_test1, y_test1)
    feature_importance1 = pd.DataFrame({'feature':X_test1.columns.tolist(),'importance':perm.feature_importances_})
    delete_col1 = feature_importance1.iloc[:,0][(feature_importance1['importance'].values)<=0.0000]
    print(delete_col1)

    return delete_col1
