import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000


# In[2]:
all_data = pd.read_csv('../data/preprocess/raw_data.csv')
print(all_data.head())
print(all_data.shape)

raw_col_num = all_data.shape[1]


## transform large type category features 轉換有大量類別的特徵
## 第一種轉法: bin cut,只留下數量最多的類別,將資料數少的類別都分成同一類,
## 第二轉種法: 根據fraud_ind的bacno數量,決定要留下哪些類別,剩下的分成同一類(要仔細觀察train和valid的關係,避免overfitting)

## some bug~~
# th=15
# category_list = all_data['stocn'].value_counts()[:th].index
# all_data['stocn_bin'] = all_data['stocn']
# all_data[~all_data['stocn'].isin(category_list)]=-1
# # print(all_data['stocn'].value_counts()[:th])

# th=20
# category_list = all_data['scity'].value_counts()[:th].index
# all_data['scity_bin'] = all_data['scity']
# all_data[~all_data['scity'].isin(category_list)]=-1
# # print(all_data['scity'].value_counts()[:th])

# th=10
# category_list = all_data['csmcu'].value_counts()[:th].index
# all_data['csmcu_bin'] = all_data['csmcu']
# all_data[~all_data['csmcu'].isin(category_list)]=-1
# # print(all_data['csmcu'].value_counts()[:th])

# one_cut = all_data[all_data['locdt']<=120]['txkey'].max()/20
# all_data['txkey_bin'] = all_data['txkey']//one_cut

# print(all_data.shape)

#######################################
mean_df = all_data.groupby(['bacno'])['cano'].nunique().reset_index()
mean_df.columns = ['bacno', 'bacno_cano_nunique']
all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

mean_df = all_data.groupby(['bacno'])['txkey'].nunique().reset_index()
mean_df.columns = ['bacno', 'bacno_txkey_nunique']
all_data = pd.merge(all_data, mean_df, on='bacno', how='left')

############### conam ################
all_data['conam_log'] = np.log(all_data['conam']+2)

# bacno_mean_conam:某個使用者平均的消費
mean_df = all_data.groupby(['bacno'])['conam_log'].mean().reset_index()
mean_df.columns = ['bacno','bacno_mean_conam']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

# bacno_scale_conam:某使用者相對自己平均消費的金額
all_data['bacno_scale_conam'] = all_data['conam_log']-all_data['bacno_mean_conam']

# cano_mean_conam:某個卡片的平均消費
mean_df = all_data.groupby(['cano'])['conam'].mean().reset_index()
mean_df.columns = ['cano','cano_mean_conam']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

# cano_scale_conam:相對卡片自己平均消費的金額
all_data['cano_scale_conam'] = all_data['conam']-all_data['cano_mean_conam']

# 每個卡片的消費金額統計值
mean_df = all_data.groupby(['cano'])['conam'].skew().reset_index()
mean_df.columns = ['cano','cano_conam_skew']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['conam'].apply(pd.Series.kurt).reset_index()
mean_df.columns = ['cano','cano_conam_kurt']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['conam'].mean().reset_index()
mean_df.columns = ['cano','cano_conam_mean']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['conam'].var().reset_index()
mean_df.columns = ['cano','cano_conam_var']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

# bacno_ismax_conam:某個使用者最大的消費
mean_df = all_data.groupby(['bacno'])['conam'].max().reset_index()
mean_df.columns = ['bacno','bacno_max_conam']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')
all_data['bacno_ismax_conam']=all_data['conam']==all_data['bacno_max_conam']
# bacno_ismin_conam:某個使用者最小的消費
mean_df = all_data.groupby(['bacno'])['conam'].min().reset_index()
mean_df.columns = ['bacno','bacno_min_conam']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')
all_data['bacno_ismin_conam']=all_data['conam']==all_data['bacno_min_conam']


############### ecfg ###############

# bacno_ratio_ecfg:某個使用者網路消費的頻率
mean_df = all_data.groupby(['bacno'])['ecfg'].apply(lambda s: s.sum()/s.shape[0]).reset_index()
mean_df.columns = ['bacno','bacno_ratio_ecfg']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

# cano_ratio_ecfg:某個使用者網路消費的頻率
mean_df = all_data.groupby(['cano'])['ecfg'].apply(lambda s: s.sum()/s.shape[0]).reset_index()
mean_df.columns = ['cano','cano_ratio_ecfg']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

############### locdt ###############
all_data['locdt_week'] = all_data['locdt']%7+1
all_data['loctm_hr'] = all_data['loctm'].apply(lambda s:s//10000).astype(int)

mean_df = all_data.groupby(['bacno'])['locdt'].skew().reset_index()
mean_df.columns = ['bacno','bacno_locdt_skew']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data.groupby(['bacno'])['locdt'].apply(pd.Series.kurt).reset_index()
mean_df.columns = ['bacno','bacno_locdt_kurt']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data.groupby(['cano'])['locdt'].skew().reset_index()
mean_df.columns = ['cano','cano_locdt_skew']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['locdt'].apply(pd.Series.kurt).reset_index()
mean_df.columns = ['cano','cano_locdt_kurt']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

############### stocn,scity,csmcu ###############

# bacno_mode
mean_df = all_data.groupby(['bacno'])['stocn'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['bacno','bacno_stocn_mode']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data.groupby(['bacno'])['scity'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['bacno','bacno_scity_mode']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data.groupby(['bacno'])['csmcu'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['bacno','bacno_csmcu_mode']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

all_data['bacno_stocn_ismode']=all_data['stocn']==all_data['bacno_stocn_mode']
all_data['bacno_scity_ismode']=all_data['scity']==all_data['bacno_scity_mode']
all_data['bacno_csmcu_ismode']=all_data['csmcu']==all_data['bacno_csmcu_mode']

# cano_mode
mean_df = all_data.groupby(['cano'])['stocn'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['cano','cano_stocn_mode']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['scity'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['cano','cano_scity_mode']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

mean_df = all_data.groupby(['cano'])['csmcu'].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
mean_df.columns = ['cano','cano_csmcu_mode']
all_data = pd.merge(all_data,mean_df,on='cano',how='left')

all_data['cano_stocn_ismode']=all_data['stocn']==all_data['cano_stocn_mode']
all_data['cano_scity_ismode']=all_data['scity']==all_data['cano_scity_mode']
all_data['cano_csmcu_ismode']=all_data['csmcu']==all_data['cano_csmcu_mode']

############### mchno,mcc,acqic ###############
## seems to be bad features

mean_df = all_data[all_data['locdt']<=60].groupby(['mchno'])['fraud_ind'].mean().reset_index()
mean_df.columns = ['bacno','mchno_fraud_mean']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data[all_data['locdt']<=60].groupby(['mcc'])['fraud_ind'].mean().reset_index()
mean_df.columns = ['bacno','mcc_fraud_mean']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

mean_df = all_data[all_data['locdt']<=60].groupby(['acqic'])['fraud_ind'].mean().reset_index()
mean_df.columns = ['bacno','acqic_fraud_mean']
all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

all_data['mchno_fraud_mean'].fillna(value=-1,inplace=True)
all_data['mcc_fraud_mean'].fillna(value=-1,inplace=True)
all_data['acqic_fraud_mean'].fillna(value=-1,inplace=True)

l_list=['mchno','acqic','mcc','stocn','scity','csmcu']
for l in l_list:
    tmp_df = all_data.groupby([l])['bacno'].nunique().reset_index()
    tmp_df.columns = [l, '{}_bacno_nunique'.format(l)]
    all_data = pd.merge(all_data,tmp_df, on=l, how='left')
    
for l in l_list:
    tmp_df = all_data.groupby([l])['cano'].nunique().reset_index()
    tmp_df.columns = [l, '{}_cano_nunique'.format(l)]
    all_data = pd.merge(all_data,tmp_df, on=l, how='left') 


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data1.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))