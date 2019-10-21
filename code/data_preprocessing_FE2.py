import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000

all_data = pd.read_csv('../data/preprocess/raw_data.csv')
print(all_data.head())
raw_col_num = all_data.shape[1]


# ## Data Leakage
# * cano在被盜取後會換卡片，觀察fraud data製作cano相關 features


## 某用戶第一次使用該卡片，且最後一天並不是使用該卡片，將最後一天使用該卡片的交易給值1，其餘0
## 用merge會比for迴圈快很多
def lastday_cano(s):
    cano_firstid = s['cano'].iloc[0]    
    if s['cano'].iloc[-1]==cano_firstid:
        return -1
    
    return s[s['cano']==cano_firstid]['locdt'].max()


cano_firstid = all_data.groupby(['bacno'])['cano'].apply(lambda s: s.iloc[0]).reset_index()
cano_lastday = all_data.groupby(['bacno']).apply(lastday_cano).reset_index()

cano_lastday_use = pd.merge(cano_firstid, cano_lastday, on='bacno',how='left')
cano_lastday_use.columns = ['bacno', 'cano', 'locdt']
cano_lastday_use['cano_lastday_use']=cano_lastday_use['locdt']!=-1

all_data = pd.merge(all_data,cano_lastday_use,on=['bacno','cano','locdt'], how='left')
all_data['cano_lastday_use'] = all_data['cano_lastday_use'].fillna(value=False)
all_data['cano_lastday_use'] = all_data['cano_lastday_use'].map({True:1,False:0})

print(cano_lastday_use['cano_lastday_use'].sum())
print(all_data['cano_lastday_use'].value_counts())


def cano_find_firstday(d):
    return d[d['fraud_ind']==1]['locdt'].min()
    
cano_firstday = all_data.groupby(['cano']).apply(cano_find_firstday)
cano_hasfraud = cano_firstday[cano_firstday>=0]

all_data['cano_hasfraud_before']=0
for i in range(cano_hasfraud.shape[0]):
    if i%2000==0:
        print(i)
    all_data[(all_data['cano']==cano_hasfraud.index[i])&             (all_data['locdt']>=cano_hasfraud.iloc[i])]['cano_hasfraud_before']=1

print(all_data[all_data['cano']==cano_hasfraud.index[10]][['locdt','cano','cano_hasfraud_before','fraud_ind']])

## stocn
tmp_df = all_data.groupby(['bacno'])['stocn'].nunique().reset_index()
tmp_df.columns = ['bacno','bacno_stocn_nunique']
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
    all_data.loc[i,'stocn_value_counts']=stocn_value_counts_dict[all_data.loc[i,'bacno']][all_data.loc[i,'stocn']]

## csmcu
tmp_df = all_data.groupby(['bacno'])['csmcu'].nunique().reset_index()
tmp_df.columns = ['bacno','bacno_csmcu_nunique']
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
    all_data.loc[i,'csmcu_value_counts']=csmcu_value_counts_dict[all_data.loc[i,'bacno']][all_data.loc[i,'csmcu']]


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



# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data2.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))