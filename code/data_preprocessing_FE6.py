import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000

# a = pd.Series([1,2,3])
# print(a.is_monotonic_increasing)

all_data = pd.read_csv('../data/preprocess/raw_data.csv')
print(all_data.head())
raw_col_num = all_data.shape[1]


# cano is monotonic increasing
tmp_df = all_data.groupby(['bacno'])['cano'].is_monotonic_increasing.reset_index()
tmp_df.columns = ['bacno','bacno_cano_monoincrease']
all_data = pd.merge(all_data,tmp_df,on=['bacno'],how='left')


# bacno,cano mode
print('mode')
mode_list =['contp','etymd','mchno','acqic','mcc','ecfg','hcefg','stscd']

for m in mode_list:
    mean_df = all_data.groupby(['bacno'])[m].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
    mean_df.columns = ['bacno','bacno_{}_mode'.format(m)]
    all_data = pd.merge(all_data,mean_df,on='bacno',how='left')

    mean_df = all_data.groupby(['cano'])[m].apply(lambda s:pd.Series.mode(s)[0]).reset_index()
    mean_df.columns = ['cano','cano_{}_mode'.format(m)]
    all_data = pd.merge(all_data,mean_df,on='cano',how='left')


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data6.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))