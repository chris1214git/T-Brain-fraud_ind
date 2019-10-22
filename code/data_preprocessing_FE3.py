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



## 每個用戶，連續且唯一的stscd==2

def one_consecutive_stscd(s):
#     s2 = s.diff(1)
#     print((s2!=0).sum(skipna=True)!=2)
#     return (s2!=0).sum(skipna=True)!=2
    
    s2 = s.map({0:' ',2:'1'})
    s2.fillna(value='-1',inplace=True)
    count=len([x for x in ''.join(s2).split()])
    if count==1:
        return 1
    else:
        return 0
    
consecutive_cano = all_data.groupby(['cano'])['stscd'].apply(one_consecutive_stscd)
consecutive_cano2 = consecutive_cano[consecutive_cano==1].reset_index()
consecutive_cano2.columns = ['cano','cano_only_consecutive_stscd2']
consecutive_cano2['stscd']= 2

print(consecutive_cano2)
all_data = pd.merge(all_data,consecutive_cano2,on=['cano','stscd'],how='left')
all_data['cano_only_consecutive_stscd2'].fillna(value=0,inplace=True)

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
bacno_consecutive_ecfg2 = bacno_consecutive_ecfg[bacno_consecutive_ecfg==1].reset_index()

bacno_consecutive_ecfg2.columns = ['bacno','bacno_consecutive_and_only_ecfg']
bacno_consecutive_ecfg2['ecfg']= 1

print(bacno_consecutive_ecfg2)
all_data = pd.merge(all_data,bacno_consecutive_ecfg2,on=['bacno','ecfg'],how='left')
all_data['bacno_consecutive_and_only_ecfg'] = all_data['bacno_consecutive_and_only_ecfg'].fillna(value=0)


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


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data3.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))