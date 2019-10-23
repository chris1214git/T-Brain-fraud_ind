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


## 只有兩種卡片,且只換過一次
## ex: 1 1 1 1 2 2

def find_bacno_cano_twokind(s):
    s2 = s['cano'].diff(1)
    
    if (s2!=0).sum()!=2:
        return -1
    
    cano_firstid = s['cano'].iloc[0]    
    return s[s['cano']==cano_firstid]['locdt'].max()

bacno_cano_twokind_firstcano = all_data.groupby(['bacno'])['cano'].apply(lambda s: s.iloc[0]).reset_index()
bacno_cano_twokind_lastlocdt = all_data.groupby(['bacno']).apply(find_bacno_cano_twokind).reset_index()

bacno_cano_twokind = pd.merge(bacno_cano_twokind_firstcano, bacno_cano_twokind_lastlocdt, on='bacno',how='left')
bacno_cano_twokind.columns = ['bacno', 'cano', 'locdt']

bacno_cano_twokind=bacno_cano_twokind[bacno_cano_twokind['locdt']!=-1]
bacno_cano_twokind['cano_lastday_use_twokind']=1

print(bacno_cano_twokind.head())
print(bacno_cano_twokind.columns)
print(bacno_cano_twokind.shape)

all_data = pd.merge(all_data,bacno_cano_twokind,on=['bacno','cano','locdt'], how='left')
all_data['cano_lastday_use_twokind'] = all_data['cano_lastday_use_twokind'].fillna(value=0)
print(all_data['cano_lastday_use_twokind'].value_counts())
 
    
## 所有卡片最後一天用的時間
## 跟每個人擁有的卡片數量做條件

cano_lastlocdt = all_data.groupby(['cano'])['locdt'].max().reset_index()
cano_lastlocdt.columns = ['cano','locdt']
cano_lastlocdt['cano_lastlocdt'] = 1
all_data = pd.merge(all_data,cano_lastlocdt,on=['cano','locdt'],how='left')
all_data['cano_lastlocdt'] = all_data['cano_lastlocdt'].fillna(value=0)

## 一個人最後一天的消費
bacno_lastlocdt = all_data.groupby(['bacno'])['locdt'].max().reset_index()
bacno_lastlocdt.columns = ['bacno','locdt']
bacno_lastlocdt['bacno_lastlocdt'] = 1
all_data = pd.merge(all_data,bacno_lastlocdt,on=['bacno','locdt'],how='left')
all_data['bacno_lastlocdt'] = all_data['bacno_lastlocdt'].fillna(value=0)

## 兩個條件做exception
all_data['cano_lastlocdt2'] = (all_data['cano_lastlocdt']==1)&(all_data['bacno_lastlocdt']==0)

## stscd
## 唯一一次stscd==2
bacno_lastlocdt = all_data.groupby(['bacno'])['stscd'].apply(lambda s:(s==2).sum()==1).reset_index()
bacno_lastlocdt.columns = ['bacno','bacno_stscd_equal2']
bacno_lastlocdt['stscd']=2

all_data = pd.merge(all_data,bacno_lastlocdt,on=['bacno','stscd'],how='left')
all_data['bacno_stscd_equal2'] = all_data['bacno_stscd_equal2'].fillna(value=False)

## ecfg
## 唯一一次ecfg==1
bacno_lastlocdt = all_data.groupby(['bacno'])['ecfg'].apply(lambda s:(s==1).sum()==1).reset_index()
bacno_lastlocdt.columns = ['bacno','bacno_ecfg_equal1']
bacno_lastlocdt['ecfg']=1

all_data = pd.merge(all_data,bacno_lastlocdt,on=['bacno','ecfg'],how='left')
all_data['bacno_ecfg_equal1'] = all_data['bacno_ecfg_equal1'].fillna(value=False)




# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data4.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))