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

# 兩天後的消費次數
all_data['bacno_2day_after_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_2day_after_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']<all_data.at[i,'locdt']+2) &
                                                 (all_data['locdt']>all_data.at[i,'locdt'])].shape[0]
# 三天後的消費次數
all_data['bacno_2day_after_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_3day_after_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']<all_data.at[i,'locdt']+3) &
                                                 (all_data['locdt']>all_data.at[i,'locdt'])].shape[0]
# 七天後的消費次數
all_data['bacno_2day_after_count']=0
for i in all_data.index:
    if i%500000==0:
        print(i)
    all_data.at[i,'bacno_7day_after_count'] = all_data[(all_data['bacno']==all_data.at[i,'bacno']) &
                                                 (all_data['locdt']<all_data.at[i,'locdt']+7) &
                                                 (all_data['locdt']>all_data.at[i,'locdt'])].shape[0]


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data4_2.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))