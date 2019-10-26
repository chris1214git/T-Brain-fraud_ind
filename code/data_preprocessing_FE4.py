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

# 一天內的消費次數
mean_df = all_data.groupby(['bacno','locdt']).apply(lambda x: x.shape[0]).reset_index()
mean_df.columns = ['bacno','locdt','bacno_1_count']
all_data = pd.merge(all_data,mean_df,on=['bacno','locdt'],how='left')

# 兩天內的消費次數
all_data['bacno_2day_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_2day_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']>all_data.at[i,'locdt']-2) &
                                                 (all_data['locdt']<=all_data.at[i,'locdt'])].shape[0]
# 三天內的消費次數
all_data['bacno_3day_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_3day_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']>all_data.at[i,'locdt']-3) &
                                                 (all_data['locdt']<=all_data.at[i,'locdt'])].shape[0]
# 七天內的消費次數
all_data['bacno_7day_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_7day_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']>all_data.at[i,'locdt']-7) &
                                                 (all_data['locdt']<=all_data.at[i,'locdt'])].shape[0]


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data4.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))