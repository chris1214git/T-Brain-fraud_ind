import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000

all_data = pd.read_csv('../../data/preprocess/raw_data.csv')
print(all_data.head())
raw_col_num = all_data.shape[1]


print('diff')
diff_list =['contp','etymd','mchno','acqic','mcc','ecfg','hcefg','stscd','stocn','scity','csmcu']
bacno_list = all_data.bacno.unique()
cano_list = all_data.cano.unique()

print('bacno')
for d in diff_list:
    print(d)
    for b in bacno_list:
        all_data.loc[all_data['bacno']==b,'bacno_{}_diff1'.format(d)]=all_data.loc[all_data['bacno']==b,d].diff(periods=1).fillna(value=-0.5)
        all_data.loc[all_data['bacno']==b,'bacno_{}_diff2'.format(d)]=all_data.loc[all_data['bacno']==b,d].diff(periods=2).fillna(value=-0.5)
        all_data.loc[all_data['bacno']==b,'bacno_{}_diff3'.format(d)]=all_data.loc[all_data['bacno']==b,d].diff(periods=3).fillna(value=-0.5)
    all_data.loc[abs(all_data['bacno_{}_diff1'.format(d)])>=1,'bacno_{}_diff1'.format(d)]=1
    all_data.loc[abs(all_data['bacno_{}_diff2'.format(d)])>=1,'bacno_{}_diff2'.format(d)]=1
    all_data.loc[abs(all_data['bacno_{}_diff3'.format(d)])>=1,'bacno_{}_diff3'.format(d)]=1

# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../../data/preprocess/FE_data7.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))