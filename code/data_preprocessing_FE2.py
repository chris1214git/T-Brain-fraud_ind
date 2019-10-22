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


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data2.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))