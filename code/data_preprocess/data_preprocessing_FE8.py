import pandas as pd
import os
import numpy as np
import math

raw_data_file_path = '../../data/preprocess/raw_data.csv'
save_data_file_path = '../../data/preprocess/FE_data8.csv'
random_seed = 2000

all_data = pd.read_csv(raw_data_file_path)
print(all_data.head())
raw_col_num = all_data.shape[1]


tmp_df = all_data.groupby(['bacno','stocn'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','stocn','locdt']
tmp_df['bacno_stocn_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','stocn','locdt'])
all_data['bacno_stocn_lastday'] = all_data['bacno_stocn_lastday'].fillna(value=0)

tmp_df = all_data.groupby(['bacno','scity'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','scity','locdt']
tmp_df['bacno_scity_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','scity','locdt'])
all_data['bacno_scity_lastday'] = all_data['bacno_scity_lastday'].fillna(value=0)

tmp_df = all_data.groupby(['bacno','csmcu'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','csmcu','locdt']
tmp_df['bacno_csmcu_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','csmcu','locdt'])
all_data['bacno_csmcu_lastday'] = all_data['bacno_csmcu_lastday'].fillna(value=0)


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv(save_data_file_path,index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))