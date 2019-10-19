#!/usr/bin/env python
# coding: utf-8

# # [玉山人工智慧公開挑戰賽2019秋季賽,真相只有一個 -『信用卡盜刷偵測』](https://tbrain.trendmicro.com.tw/Competitions/Details/10)
# 
# ## <font color=red>任務:預測某刷卡交易是否為盜刷</font>
# 
# ### Task Schedule:
# 1. 讀取資料,將字串轉換成int
# 2. EDA(exploratory data analysis)
# 3. Feature engineering
# 4. 訓練模型,調整參數(預計使用lgb，速度較快)
# 5. 嘗試使用不同模型,做Ensamble(blending, stacking)
# 6. Anomaly detection
# 
# ### 注意事項:
# 1. 因為test data和train data時間不相關,在驗證時採取前60天訓練61~90天驗證,但仍需小心時間差異造成的影響
# 
# ### TODO:
# 1. **EDA(見下方詳細解釋）,找出不適合作為training feature的特徵,加以轉化成高級特徵或刪除**
# 2. **找data leakage**
# 
# 3. Anomaly detection: 看這類的模型能不能取代lgb(似乎是不行，盜刷數據並沒有那麼Anomaly）,但可以嘗試將Anomaly結果當成新feature
# 
# ### <font color=green>Results:</font>
# * 不做處理,直接丟lgb訓練 leaderboard score:0.45
# 

# ## 讀取,轉換字串成可以訓練的資料

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000


# In[2]:


train_data_path = os.path.join(data_path,'train.zip')
train_data = pd.read_csv(train_data_path, encoding = "big5")

test_data_path = os.path.join(data_path,'test.zip')
test_data = pd.read_csv(test_data_path, encoding = "big5")

train_data_num = train_data.shape[0]
test_data_txkey = test_data['txkey'].copy()

train_data = train_data.sort_values(by=['bacno','locdt','loctm']).reset_index(drop=True)
label_data = train_data['fraud_ind'].copy()

all_data = pd.concat([train_data,test_data],axis=0).reset_index(drop=True)
print(all_data.index)
print(train_data.shape)
print(test_data.shape)
print(all_data.shape)


# In[3]:


train_data.head()


# In[4]:


all_data.ecfg = all_data.ecfg.map({'N':0,'Y':1})
all_data.ovrlt = all_data.ovrlt.map({'N':0,'Y':1})
all_data.insfg = all_data.insfg.map({'N':0,'Y':1})
all_data.flbmk = all_data.flbmk.map({'N':0,'Y':1})
all_data.flg_3dsmk = all_data.flg_3dsmk.map({'N':0,'Y':1})
all_data.loctm = all_data.loctm.astype(int)
all_data = all_data.infer_objects()

# print(all_data.dtypes)
print('Missing value training data:\n',train_data.isna().sum()[train_data.isna().sum()>0])
print('Missing value testing data:\n',test_data.isna().sum()[test_data.isna().sum()>0])

## not neccessary to fill null value, since we use lgb model
all_data.flbmk = all_data.flbmk.fillna(value=all_data.flbmk.mean(skipna=True))
all_data.flg_3dsmk = all_data.flg_3dsmk.fillna(value=all_data.flg_3dsmk.mean(skipna=True))


# ## Dirty Data

# In[5]:


weird1 = (all_data['insfg']==1)&(all_data['iterm']==0)
print(weird1.value_counts())
print(all_data[weird1]['fraud_ind'].sum())


# In[6]:


binary_list=['ecfg','insfg','ovrlt','flbmk','flg_3dsmk']
category_list=['contp','etymd','hcefg','stocn','scity','stscd','csmcu']


# ## Feature engineering
# * train & valid only（先不考慮test data)

# ## Bin cut

# In[7]:


## transform large type category features 轉換有大量類別的特徵
## 第一種轉法: bin cut,只留下數量最多的類別,將資料數少的類別都分成同一類,
## 第二轉種法: 根據fraud_ind的bacno數量,決定要留下哪些類別,剩下的分成同一類(要仔細觀察train和valid的關係,避免overfitting)

# th=100
# category_list = all_data['mchno'].value_counts()[:th].index
# all_data2 = all_data.copy()
# all_data2[~all_data['mchno'].isin(category_list)]=-1
# print(all_data2['mchno'].value_counts()[:th])

# th=100
# category_list = all_data['acqic'].value_counts()[:th].index
# all_data2 = all_data.copy()
# all_data2[~all_data['acqic'].isin(category_list)]=-1
# print(all_data2['acqic'].value_counts()[:th])

# th=100
# category_list = all_data['mcc'].value_counts()[:th].index
# all_data2 = all_data.copy()
# all_data2[~all_data['mcc'].isin(category_list)]=-1
# print(all_data2['mcc'].value_counts()[:th])

th=15
category_list = all_data['stocn'].value_counts()[:th].index
all_data2 = all_data.copy()
all_data2[~all_data['stocn'].isin(category_list)]=-1
# print(all_data2['stocn'].value_counts()[:th])

th=20
category_list = all_data['scity'].value_counts()[:th].index
all_data2 = all_data.copy()
all_data2[~all_data['scity'].isin(category_list)]=-1
# print(all_data2['scity'].value_counts()[:th])

th=10
category_list = all_data['csmcu'].value_counts()[:th].index
all_data2 = all_data.copy()
all_data2[~all_data['csmcu'].isin(category_list)]=-1
# print(all_data2['csmcu'].value_counts()[:th])


# In[8]:


one_cut = all_data[all_data['locdt']<=120]['txkey'].max()/20
all_data['txkey_bin'] = all_data['txkey']//one_cut


# In[9]:


all_data['locdt_week'] = all_data['locdt']%7+1
# all_data['locdt_month'] = all_data['locdt']%30+1

all_data['loctm_hr'] = all_data['loctm'].apply(lambda s:s//10000).astype(int)
# all_data['loctm_hr2'] = all_data['loctm'].apply(lambda s:s//1000).astype(int)
# all_data['loctm_hr_sin'] = all_data['loctm_hr'].apply(lambda s:math.sin(s/24*math.pi)).astype(int)
# all_data['loctm_hr2_sin'] = all_data['loctm_hr2'].apply(lambda s:math.sin(s/240*math.pi)).astype(int)

mean_df = all_data.groupby(['bacno'])['cano'].nunique().reset_index()
mean_df.columns = ['bacno', 'cano_not1']
mean_df[mean_df['cano_not1']>1]=0
all_data = pd.merge(all_data, mean_df, on='bacno', how='left')
print(all_data['cano_not1'].value_counts())

mean_df = all_data.groupby(['bacno'])['txkey'].nunique().reset_index()
mean_df.columns = ['bacno', 'txkey'+'_count']
all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

# mean_df = all_data.groupby(['bacno'])['loctm_hr'].mean().reset_index()
# mean_df.columns = ['bacno', 'loctm_hr'+'_mean']
# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

# mean_df = all_data.groupby(['bacno'])['loctm_hr'].var().reset_index()
# mean_df.columns = ['bacno', 'loctm_hr'+'_var']
# mean_df.fillna(value=-1,inplace=True)
# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')


# In[10]:


l_list=['mchno','acqic','mcc','stocn','scity','csmcu']
for l in l_list:
    tmp_df = all_data.groupby([l])['bacno'].nunique().reset_index()
    tmp_df.columns = [l, l+'_bacno_nunique']
    all_data = pd.merge(all_data,tmp_df, on=l, how='left')
    
for l in l_list:
    tmp_df = all_data.groupby([l])['cano'].nunique().reset_index()
    tmp_df.columns = [l, l+'_cano_nunique']
    all_data = pd.merge(all_data,tmp_df, on=l, how='left') 


# ## Personal feature engineering
# 
# #### 由於模型讀沒有辦法讀取用戶歷史記錄，我們手動製作跟用戶相關的歷史特徵
# * 根據bacno or cano製作
# * conam,ecfg,stocn,stscd,csmcu

# In[11]:


## conam
all_data['conam'] = np.log(all_data['conam']+2)

# bacno_mean_conam:某個使用者平均的消費
mean_df = all_data.groupby(['bacno'])['conam'].mean().reset_index()
mean_df.columns = ['bacno','bacno_mean_conam']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

# bacno_scale_conam:某使用者相對自己平均消費的金額
all_data['bacno_scale_conam'] = all_data['conam']-all_data['bacno_mean_conam']

# cano_mean_conam:某個卡片的平均消費
mean_df = all_data.groupby(['cano'])['conam'].mean().reset_index()
mean_df.columns = ['cano','cano_mean_conam']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

# cano_scale_conam:相對卡片自己平均消費的金額
all_data['cano_scale_conam'] = all_data['conam']-all_data['cano_mean_conam']


# In[12]:


## ecfg
## 連續且唯一的ecfg出現，標記1

def one_consecutive_ecfg(s):    
    s2 = s.map({0:' ',1:'1'})
    count=len([x for x in ''.join(s2).split()])
    if count==1:
        return 1
    else:
        return 0
    
bacno_consecutive_ecfg = all_data.groupby(['bacno'])['ecfg'].apply(one_consecutive_ecfg)
bacno_consecutive_ecfg2 = bacno_consecutive_ecfg[bacno_consecutive_ecfg==1]
print(bacno_consecutive_ecfg2.shape[0])

all_data['bacno_consecutive_and_only_ecfg']=0

for i in range(bacno_consecutive_ecfg2.shape[0]):
    if i%(bacno_consecutive_ecfg2.shape[0]//4)==0:
        print(i)
    all_data[all_data['bacno']==bacno_consecutive_ecfg2.index[i]]['bacno_consecutive_and_only_ecfg']=all_data[all_data['bacno']==bacno_consecutive_ecfg2.index[i]]['ecfg']==1
    


# In[13]:


## stocn

## stocn_nunique
tmp_df = all_data.groupby(['bacno'])['stocn'].nunique().reset_index()
tmp_df.columns = ['bacno','bacno_nunique_stocn']
all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')

## stocn_value_counts
bacno_list = all_data['bacno'].unique()
all_data['stocn_value_counts']=0
stocn_value_counts_dict={}

for b in bacno_list:
    stocn_value_counts_dict[b] = all_data[all_data['bacno']==b]['stocn'].value_counts() 

for i in range(all_data.shape[0]):
    if i%500000==0:
        print(i)
    all_data.loc[i,'stocn_value_counts']=stocn_value_counts_dict[all_data.iloc[i]['bacno']][all_data.iloc[i]['stocn']]


# In[15]:


## csmcu

## csmcu_nunique
tmp_df = all_data.groupby(['bacno'])['csmcu'].nunique().reset_index()
tmp_df.columns = ['bacno','bacno_nunique_csmcu']
all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')

print('hi')
## csmcu_value_counts
bacno_list = all_data['bacno'].unique()
all_data['csmcu_value_counts']=0
csmcu_value_counts_dict={}

for b in bacno_list:
    csmcu_value_counts_dict[b] = all_data[all_data['bacno']==b]['csmcu'].value_counts() 

for i in range(all_data.shape[0]):
    if i%500000==0:
        print(i)
    all_data.iloc[i]['csmcu_value_counts']=csmcu_value_counts_dict[all_data.iloc[i]['bacno']][all_data.iloc[i]['csmcu']]


# In[ ]:


## 每個用戶，連續且唯一的stscd==2

def one_consecutive_stscd(s):
#     s2 = s.diff(1)
#     print((s2!=0).sum(skipna=True)!=2)
#     return (s2!=0).sum(skipna=True)!=2
    
    s2 = s.map({0:' ',2:'1'})
    s2 = s2.fillna(value='-1')
    count=len([x for x in ''.join(s2).split()])
    if count==1:
        return 1
    else:
        return 0
    
consecutive_cano = all_data.groupby(['cano'])['stscd'].apply(one_consecutive_stscd)
consecutive_cano2 = consecutive_cano[consecutive_cano==1]
print(consecutive_cano2.shape[0])

all_data['cano_only_consecutive_stscd2']=0

for i in range(consecutive_cano2.shape[0]):
    if i%(consecutive_cano2.shape[0]//4)==0:
        print(i)
    all_data[all_data['cano']==consecutive_cano2.index[i]]['cano_only_consecutive_stscd2']=all_data[all_data['cano']==consecutive_cano2.index[i]]['stscd']==2
    


# ## Data Leakage
# * cano在被盜取後會換卡片，觀察fraud data製作cano相關 features

# In[ ]:


## 某用戶第一次使用該卡片，且最後一天並不是使用該卡片，將最後一天使用該卡片的交易給值1，其餘0
## 用merge會比for迴圈快很多
def lastday_cano(s):
    cano_firstid = s['cano'].iloc[0]    
    if s['cano'].iloc[-1]==cano_firstid:
        return -1
    
    return s[s['cano']==cano_firstid]['locdt'].max()


cano_firstid = all_data.groupby(['bacno'])['cano'].apply(lambda s: s.iloc[0]).reset_index()
cano_lastday = all_data.groupby(['bacno']).apply(lastday_cano).reset_index()
# print(cano_lastday)

cano_lastday_use = pd.merge(cano_firstid, cano_lastday, on='bacno',how='left')
cano_lastday_use.columns = ['bacno', 'cano', 'locdt']
cano_lastday_use['cano_lastday_use']=cano_lastday_use['locdt']!=-1

all_data = pd.merge(all_data,cano_lastday_use,on=['bacno','cano','locdt'], how='left')
all_data['cano_lastday_use'] = all_data['cano_lastday_use'].fillna(value=False)
all_data['cano_lastday_use'] = all_data['cano_lastday_use'].map({True:1,False:0})

print(cano_lastday_use['cano_lastday_use'].sum())
print(all_data['cano_lastday_use'].value_counts())


# ## 和Fraud相關的特徵工程
# #### 先使用於train上 檢查validation結果,小心overfit
# 

# In[ ]:


## 某卡片過去是否有盜刷記錄
## 記錄每個卡片第一天被盜刷的日期,
def cano_find_firstday(d):
    return d[d['fraud_ind']==1]['locdt'].min()
    
cano_firstday = all_data.groupby(['cano']).apply(cano_find_firstday)
cano_hasfraud = cano_firstday[cano_firstday>=0]

all_data['cano_hasfraud_before']=0
for i in range(cano_hasfraud.shape[0]):
    if i%2000==0:
        print(i)
    all_data[(all_data['cano']==cano_hasfraud.index[i])&             (all_data['locdt']>=cano_hasfraud.iloc[i])]['cano_hasfraud_before']=1

print(all_data[all_data['cano']==cano_hasfraud.index[10]][['locdt','fraud_ind']])


# In[ ]:


# for i in range(500):
#     print(i,all_data.groupby(['bacno']).get_group(i)[['ecfg','fraud_ind']])
# mean_df = all_data.groupby(['bacno'])['fraud_ind'].mean().reset_index()
# mean_df.columns = ['bacno', 'loctm_hr'+'_mean']
# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

# print(all_data[['bacno','locdt','loctm']])

# 該交易的歸戶帳號是否曾經被盜刷 0->沒 1->有 -1->無紀錄


# 該交易的歸戶帳號是否曾經被盜刷卻又復原
# 該交易的歸戶帳號是否第一次刷卡
# 該交易的歸戶帳號第幾次刷卡

# 該交易的卡號是否曾經被盜刷
# 該交易的卡號是否曾經被盜刷卻又復原
# 該交易的卡號是否第一次刷卡
# 該交易的卡號第幾次刷卡

# mean_df = all_data.groupby(['bacno']).apply(lambda s:s.mode()).reset_index()
# mean_df.columns = ['bacno', 'stocn'+'_mode']
# mean_df.fillna(-1,inplace=True)
# print(mean_df.stocn_mode.value_counts())
# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

# 消費國別是否跟自己所有消費的眾數不一樣
# 消費城市是否跟自己所有消費的眾數不一樣
# 消費地幣別是否跟自己所有消費的眾數不一樣
# 支付型態是否跟自己所有消費的眾數不一樣
# 分期期數是否跟自己所有消費的眾數不一樣

# 是否第一次網路消費且過去有非網路消費的經驗


# In[ ]:


# data = pd.concat([df[:train_num], train_Y], axis=1)
# for c in df.columns:
#     mean_df = data.groupby([c])['SalePrice'].mean().reset_index()
#     mean_df.columns = [c, f'{c}_mean']
#     data = pd.merge(data, mean_df, on=c, how='left')
#     data = data.drop([c] , axis=1)


# all_data['howmany_cano'] = 
# all_data['howmany_txkey'] = 

## bacno刷卡頻率分佈

# all_data['fraud_before'] =
# all_data['fraud_last_time'] =

# 印出某個被盜刷的人的刷卡使用時間分佈


# In[ ]:


all_data.to_csv('../data/all_data.csv',index=False)


# ## Train on LGB(未調參數)

# In[ ]:


delete_list = ['bacno','locdt','loctm','cano','fraud_ind','iterm']
#txkey大小, cano可能會重複所以重要？


# In[ ]:


X_train = all_data[all_data['locdt']<=60].drop(columns=delete_list)
y_train = all_data[all_data['locdt']<=60]['fraud_ind']
X_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)
y_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']

print(delete_list)
print(X_train.shape)
print(y_train.sum()/y_train.shape[0])
print(y_test.sum()/y_test.shape[0])


import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def lgb_f1_score(y_true, y_pred):
    y_pred = np.round(y_pred) # scikits f1 doesn't like probabilities
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print()
    print('tn, fp, fn, tp')
    print(tn, fp, fn, tp)
    return 'f1', f1_score(y_true, y_pred), True

param_dist_lgb = {
#                   'num_leaves':45, 
#                   'max_depth':5, 
                  'learning_rate':0.1, 
                  'n_estimators':600,
                  'objective': 'binary',
#                   'subsample': 1, 
#                   'colsample_bytree': 0.5, 
#                   'lambda_l1': 0.1,
#                   'lambda_l2': 0,
#                   'min_child_weight': 1,
                  'random_state': random_seed,
                 }
evals_result = {}

lgb_clf = LGBMClassifier(**param_dist_lgb)
lgb_clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_metric=lgb_f1_score,
        early_stopping_rounds=200,
        verbose=True,
        callbacks=[lgb.record_evaluation(evals_result)]
        )
y_test_pred = lgb_clf.predict(X_test)
print('F1',f1_score(y_test, y_test_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print(tn, fp, fn, tp)


# In[ ]:


print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result, metric='f1')
plt.show()

print('Plotting feature importances...')
ax = lgb.plot_importance(lgb_clf, max_num_features=10)
plt.show()

print('Plotting 4th tree...')  # one tree use categorical feature to split
ax = lgb.plot_tree(lgb_clf, tree_index=3, figsize=(15, 15), show_info=['split_gain'])
plt.show()

print('Plotting 4th tree with graphviz...')
graph = lgb.create_tree_digraph(lgb_clf, tree_index=3, name='Tree4')
graph.render(view=True)


# ## Train on catboost
# * https://catboost.ai/docs/concepts/python-reference_parameters-list.html
# * 研究有哪些可以用的function

# In[ ]:


from catboost import CatBoostClassifier, Pool

test_data = catboost_pool = Pool(train_data, 
                                 train_labels)

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
# train the model
model.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)


# In[ ]:


get_ipython().system('pip3 install catboost --user')


# In[ ]:


# 用hinge loss(當SVM)


# In[ ]:


# X_train['cents']
# encoding data

# GroupKfold
# vanilla KFold


# ## write csv

# In[ ]:


# lgb_clf = LGBMClassifier(**param_dist_lgb)
# lgb_clf.fit(train_data,label_data)

# result = lgb_clf.predict(test_data)
# print(result.sum())
# print(result.sum()/result.shape[0])
# print(label_data.sum()/label_data.shape[0])

# test_data_txkey = test_data['txkey'].copy()

# import csv
# with open('../prediction/submit_lgb.csv','w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['txkey','fraud_ind'])
#     for i in range(result.shape[0]):
#         writer.writerow([test_data_txkey[i], result[i]])


# In[ ]:





# ## Anomaly detection
# * one class svm
# * isolation tree
# * replicator NN
# * Kmeans?
# * KNN(take too much time)

# ## 製作特徵
# XGB, PCA, Isolation Forest, Kmean距離？, oneclass SVM?
# 當作新feature

# In[ ]:


import xgboost as xgb
param_dist_xgb = {'learning_rate':0.01, #默认0.3
              'n_estimators':1000, #树的个数
#               'max_depth':5,
#               'min_child_weight':1,
#               'gamma':0.2,
#               'subsample':0.8,
#               'colsample_bytree':0.8,
#               'objective': 'binary:logistic', #逻辑回归损失函数
#               'nthread':4,  #cpu线程数
#               'scale_pos_weight':1,
              'seed':random_seed}  #随机种子

evals_result = {}

xgb_clf = xgb.XGBClassifier(**param_dist_xgb)
xgb_clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_metric=lgb_f1_score,
        early_stopping_rounds=600,
        verbose=True,
#         callbacks=[xgb.record_evaluation(evals_result)]
        )

print('F1',f1_score(y_test, xgb_clf.predict(X_test)))
xgb_X_train = xgb_clf.apply(X_train)
xgb_X_test = xgb_clf.apply(X_test)





