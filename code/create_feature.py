import pandas as pd
import os
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

import time
import csv
random_seed = 20

t_now = time.localtime( time.time() )
t = str(t_now.tm_mon)+str(t_now.tm_mday)+str(t_now.tm_hour)+str(t_now.tm_min)
print('Now:',t)

th = 0.37

data_list=['raw_data.csv','FE_data1.csv','FE_data2.csv','FE_data3.csv']

## 除掉一些可能會overfit,distribution不同,受時間影響大的feature
delete_list1 = ['bacno','locdt','loctm','cano','fraud_ind']
delete_list2 = ['mchno','acqic','mcc']
delete_list3 = ['stocn','scity','csmcu']
delete_list4 = ['iterm']
delete_list6 = ['mchno_fraud_mean','mcc_fraud_mean','acqic_fraud_mean']
delete_list7 = ['bacno_locdt_skew','bacno_locdt_kurt','cano_locdt_skew','cano_locdt_kurt']
delete_list8 = ['bacno_lastlocdt','cano_lastlocdt']

delete_list5 = ['contp','etymd','hcefg','insfg','ovrlt','flbmk','flg_3dsmk']
# bacno_cano_nunique

delete_list = delete_list1+delete_list2+delete_list3+delete_list4+delete_list6+delete_list7+['txkey']+delete_list8
category_list=['csmcu','hcefg','stscd','scity','stocn','mcc','acqic','mchno','etymd','contp']

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

def get_dummy(x,category_list):
    categorical_features_indices = np.where(x.columns.isin(category_list))[0]
    category_list2 = X_train1.columns[categorical_features_indices]
    x[category_list2] = x[category_list2].astype('category')    
    return pd.get_dummies(x)

def PCA_feature(x,n=2):
    x
    stdsc = StandardScaler() 
    x = stdsc.fit_transform(x)
  
    PCA_model = PCA(n_components=n)
    pca_feature = PCA_model.fit_transform(x)  
    return pca_feature

def Isolation_all_feature(x,c_ratio=0.01):
    clf = IsolationForest(behaviour='new', max_samples=0.8, max_features=1,
                        random_state=random_seed, contamination=c_ratio)
    clf.fit(x)
    y_pred = clf.score_samples(x)
    
    return y_pred

def Kmeans_all_feature(x,n=1):
    stdsc = StandardScaler() 
    x = stdsc.fit_transform(x)

    clf = KMeans(n_clusters=n,random_state=random_seed)
    x2 = clf.fit_transform(x)
    x3 = clf.fit_predict(x)

    distance = np.zeros(x3.shape)
    for i in range(distance.shape[0]):
        distance[i] = x2[i,x3[i]]

    return distance

def main():
    all_data = load_data(data_list)
    all_data[category_list]=all_data[category_list].astype('category')

    X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2,\
            X_train3, y_train3, X_test3, y_test3, X_train_all, y_train_all, X_test_all, test_data_txkey=\
    parse_validation(all_data,delete_list)

    pca2_feature = PCA_feature(all_data.drop(columns=delete_list),2)
    pca3_feature = PCA_feature(all_data.drop(columns=delete_list),3)
    pca2_feature = pd.DataFrame(pca2_feature,columns=['pca2_feature_1','pca2_feature_2'])
    pca3_feature = pd.DataFrame(pca3_feature,columns=['pca3_feature_1','pca3_feature_2','pca3_feature_3'])
    pca_feature = pd.concat([pca2_feature,pca3_feature],axis=1)
    pca_feature.to_csv('../data/preprocess/pca_feature.csv',index=False)
    
    c_ratio = y_train_all.sum()/y_train_all.shape[0]
    isolationtree_all_feature = Isolation_all_feature(all_data.drop(columns=delete_list),c_ratio)
    isolationtree_all_feature = pd.DataFrame(isolationtree_all_feature,columns=['isolationtree_all_feature'])
    isolationtree_feature = isolationtree_all_feature
    
    isolationtree_feature.to_csv('../data/preprocess/isolationtree_feature.csv',index=False)
    
    
    kmeans_all_feature1 = Kmeans_all_feature(all_data.drop(columns=delete_list),1)
    kmeans_all_feature3 = Kmeans_all_feature(all_data.drop(columns=delete_list),3)
    kmeans_all_feature5 = Kmeans_all_feature(all_data.drop(columns=delete_list),5)

    kmeans_all_feature1 = pd.DataFrame(kmeans_all_feature1,columns=['kmeans_all_feature'])
    kmeans_all_feature3 = pd.DataFrame(kmeans_all_feature3,columns=['kmeans_all_feature3'])
    kmeans_all_feature5 = pd.DataFrame(kmeans_all_feature5,columns=['kmeans_all_feature5'])
    kmeans_feature = pd.concat([kmeans_all_feature1,kmeans_all_feature3,kmeans_all_feature5],axis=1)
    
    kmeans_feature.to_csv('../data/preprocess/kmeans_feature.csv',index=False)


    # np.save('isolationtree_bacno_feature.npy',isolationtree_bacno_feature)
    # np.save('kmeans_bacno_feature.npy',kmeans_bacno_feature)
    
    
if __name__ == '__main__':
    main()  # 或是任何你想執行的函式