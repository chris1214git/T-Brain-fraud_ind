import pandas as pd
import os
import numpy as np
import math
import sys

random_seed = 2000

all_data = pd.read_csv('../../data/preprocess/raw_data.csv')
print(all_data.head())
raw_col_num = all_data.shape[1]


var_list =['stocn','scity','csmcu','mchno','acqic','mcc']

def cal_std(s):
    return np.std(s)

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = x.values
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
#     print(ent)
    return ent

for v in var_list:
    tmp_df = all_data.groupby(['bacno'])[v].apply(calc_ent).reset_index()
    print(tmp_df)
    tmp_df.columns = ['bacno','bacno_{}_entropy'.format(v)]
    all_data = pd.merge(all_data,tmp_df,on=['bacno'],how='left')
    all_data['bacno_{}_entropy'.format(v)].fillna(value=1,inplace=True)
    
for v in var_list:
    tmp_df = all_data.groupby([v])['locdt'].apply(cal_std).reset_index()
    print(tmp_df)
    tmp_df.columns = [v,'{}_locdt_std'.format(v)]
    all_data = pd.merge(all_data,tmp_df,on=[v],how='left')
    all_data['{}_locdt_std'.format(v)].fillna(value=-1,inplace=True)


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../../data/preprocess/FE_data11.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))