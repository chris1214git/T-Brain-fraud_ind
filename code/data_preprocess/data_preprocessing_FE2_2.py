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

bacno_frequency_encoding_list = ['stocn','csmcu','mchno','acqic','mcc','scity','contp','etymd','stscd','hcefg']


mean_df = all_data.groupby(['cano'])['txkey'].nunique().reset_index()
mean_df.columns = ['cano', 'cano_txkey_nunique']
all_data = pd.merge(all_data, mean_df, on='cano', how='left')

for l in bacno_frequency_encoding_list:
    print(l)
    tmp_df = all_data.groupby(['cano'])[l].nunique().reset_index()
    tmp_df.columns = ['cano','cano_{}_nunique'.format(l)]
    all_data = pd.merge(all_data,tmp_df,on='cano',how='left')

    bacno_list = all_data['cano'].unique()
    all_data['{}_value_counts_cano'.format(l)]=0
    value_counts_dict={}

    for b in bacno_list:
        value_counts_dict[b] = all_data[all_data['cano']==b][l].value_counts() 

    for i in range(all_data.shape[0]):
        if i%500000==0:
            print(i)
        all_data.loc[i,'{}_value_counts_cano'.format(l)]=value_counts_dict[all_data.loc[i,'cano']][all_data.loc[i,l]]        
    
    all_data['{}_ratio_cano'.format(l)]=all_data['{}_value_counts_cano'.format(l)]/all_data['cano_txkey_nunique']

# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data2_2.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))
