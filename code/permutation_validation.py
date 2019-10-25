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

import json
path = '../code/para_dict/data_list.json'
with open(path,'r',encoding='utf-8') as f:
    para = json.loads(f.read())

# 'data_list_FE_AN'
# 'delete_list_overfit1'
data_list= para[sys.argv[1]]
delete_list = para[sys.argv[2]]
th = float(sys.argv[3])
iteration = int(sys.argv[4])
l2_leaf_reg = float(sys.argv[5])

random_seed = 20

t_now = time.localtime( time.time() )
t = str(t_now.tm_mon)+str(t_now.tm_mday)+str(t_now.tm_hour)+str(t_now.tm_min)
print('Now:',t)

category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic','mchno','etymd','contp']
param_cat={
    'loss_function':'Logloss',
    'eval_metric':'F1',
    
    'iterations':iteration,
    'learning_rate':0.1,
    'l2_leaf_reg':3,
    'bagging_temperature':1,
    
    'depth':6,
    'one_hot_max_size':300,    
    'rsm':1,
    'scale_pos_weight':1,
    'target_border':0.5,
    'random_seed':random_seed,
    'verbose':True    
}

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
#         silent=False,
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


def train_model_all(X_train_all,y_train_all,X_test_all,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name):
    model = CatBoostClassifier(**param_cat)

    model.fit(
        X_train_all, y_train_all,
        cat_features=categorical_features_indices,    
#         silent=False,
        verbose=100
    )
    y_test_pred_cat = model.predict_proba(X_test_all)[:,1]

    y_test_pred_cat[y_test_pred_cat>th]=1
    y_test_pred_cat[y_test_pred_cat<=th]=0


    # ## write csv
    result = y_test_pred_cat

    print('{}: prediction positive ratio'.format(result.sum()/result.shape[0]))
    print('{}: training positive ratio'.format(y_train_all.sum()/y_train_all.shape[0]))

    with open('../prediction/{}'.format(submit_file_name),'w') as f:
        writer = csv.writer(f)
        writer.writerow(['txkey','fraud_ind'])
        for i in range(result.shape[0]):
            writer.writerow([test_data_txkey[i], result[i]])

# In[34]:

def permutation_test(model,X_test1,y_test1,categorical_features_indices):
    train_pool=Pool(X_test1, y_test1,cat_features=categorical_features_indices)
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_test1.columns
    delete_col1 = []
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))
        if score<0.1:
            delete_col1.append(name)
    return delete_col1

def main():

    all_data = load_data(data_list)
    all_data[category_list]=all_data[category_list].astype('category')

    X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2,\
            X_train3, y_train3, X_test3, y_test3, X_train_all, y_train_all, X_test_all, test_data_txkey=\
    parse_validation(all_data,delete_list)

    # Train on catboost
    categorical_features_indices = np.where(X_train1.columns.isin(category_list))[0]
    print(X_train1.dtypes[categorical_features_indices])

    model1, F1_score1 = train_model_validation(X_train1, y_train1, X_test1, y_test1, th, categorical_features_indices,param_cat)
    model2, F1_score2 = train_model_validation(X_train2, y_train2, X_test2, y_test2, th, categorical_features_indices,param_cat)
    model3, F1_score3 = train_model_validation(X_train3, y_train3, X_test3, y_test3, th, categorical_features_indices,param_cat)


    submit_file_name='submit_cat_time{}_{}.csv'.format(t[:4],t[4:])
    with open('../prediction/log.txt','a') as f:
        print('{}'.format(submit_file_name),file=f)
        print('data list',data_list,file=f)
        print('delete_list:\n{}'.format(delete_list),file=f)
        print('parameter:\n{}'.format(param_cat),file=f)
        print('th:',th,file=f)
        print('F1_score:',F1_score1,F1_score2,F1_score3,file=f)
        print('average F1_score:', (F1_score1+F1_score2+F1_score3)/3,file=f)
        print('\n',file=f)

    delete_col1 = permutation_test(model1,X_test1,y_test1,categorical_features_indices)
    delete_col2 = permutation_test(model2,X_test2,y_test2,categorical_features_indices)
    delete_col3 = permutation_test(model3,X_test3,y_test3,categorical_features_indices)
    delete_col=[]
    for i in delete_col1:
        if i in delete_col2:
            if i in delete_col3:
                delete_col.append(i)

    with open('../prediction/permutation_results.txt','a') as f:
        print('{}'.format(submit_file_name),file=f)
        print('data list',data_list,file=f)
        print('delete_list:\n{}'.format(delete_list),file=f)
        print('delete_col1:\n{}'.format(delete_col1),file=f)
        print('delete_col2:\n{}'.format(delete_col2),file=f)
        print('delete_col3:\n{}'.format(delete_col3),file=f)
        print('Inner delete_col:\n{}'.format(delete_col),file=f)
        print('\n',file=f)
    
    
    categorical_features_indices_del = np.where(X_train1.drop(columns=delete_col).columns.isin(category_list))[0]
    model1, F1_score1 = train_model_validation(X_train1.drop(columns=delete_col), y_train1, X_test1.drop(columns=delete_col), y_test1, th, categorical_features_indices_del,param_cat)
    model2, F1_score2 = train_model_validation(X_train2.drop(columns=delete_col), y_train2, X_test2.drop(columns=delete_col), y_test2, th, categorical_features_indices_del,param_cat)
    model3, F1_score3 = train_model_validation(X_train3.drop(columns=delete_col), y_train3, X_test3.drop(columns=delete_col), y_test3, th, categorical_features_indices_del,param_cat)

    with open('../prediction/log.txt','a') as f:
        print('{} permutation'.format(submit_file_name),file=f)
        print('data list',data_list,file=f)
        print('delete_list:\n{}'.format(delete_list),file=f)
        print('parameter:\n{}'.format(param_cat),file=f)
        print('th:',th,file=f)
        print('F1_score:',F1_score1,F1_score2,F1_score3,file=f)
        print('average F1_score:', (F1_score1+F1_score2+F1_score3)/3,file=f)
        print('\n',file=f)

    
    train_model_all(X_train_all,y_train_all,X_test_all,test_data_txkey,th,categorical_features_indices,param_cat,submit_file_name)
    submit_file_name='submit_cat_permutation_time{}_{}.csv'.format(t[:4],t[4:])
    train_model_all(X_train_all.drop(columns=delete_col),y_train_all,X_test_all.drop(columns=delete_col),test_data_txkey,th,categorical_features_indices_del,param_cat,submit_file_name)
    
if __name__ == '__main__':
    main()  # 或是任何你想執行的函式
