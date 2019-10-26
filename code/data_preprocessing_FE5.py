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

# 卡片更換次數
def cano_change_count(s):
    s2 = s.diff()
    return s2[s2!=0].shape[0]

def cano_change_ratio(s):
    s2 = s.diff()
    return (s2[s2!=0].shape[0])/(s2.shape[0])

tmp_df = all_data.groupby(['bacno'])['cano'].apply(cano_change_count).reset_index()
tmp_df.columns = ['bacno','bacno_cano_changetimes']
all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')

tmp_df = all_data.groupby(['bacno'])['cano'].apply(cano_change_ratio).reset_index()
tmp_df.columns = ['bacno','cano_change_ratio']
all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')

bacno_list = all_data.bacno.unique()
all_data['bacno_locdt_diff'] = 0
all_data['bacno_loctm_diff'] = 0
# 距離上次刷卡的時間(日期)
for b in bacno_list:
    all_data.loc[all_data['bacno']==b,'bacno_locdt_diff']=all_data.loc[all_data['bacno']==b,'locdt'].diff().fillna(value=-1)
    
# 距離上次刷卡的時間(時間)
for b in bacno_list:
    all_data.loc[all_data['bacno']==b,'bacno_loctm_diff']=all_data.loc[all_data['bacno']==b,'loctm'].diff().fillna(value=-1)

all_data.loc[all_data['bacno_locdt_diff']!=0,'bacno_loctm_diff'] = -1


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data5.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))