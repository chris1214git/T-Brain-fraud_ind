import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000

shift_file_num = sys.argv[1]

raw_data = pd.read_csv('../../data/preprocess/raw_data.csv')
bacno_col = raw_data.bacno

shift_data = pd.read_csv('../../data/preprocess/FE_data{}.csv'.format(shift_file_num))
all_data = pd.concat([shift_data,bacno_col],axis=1)
print(all_data.shape)

bacno_list = bacno_col.unique()

raw_col_num = all_data.shape[1]

for c in all_data.columns:
    print(c)
    for b in bacno_list:
        all_data.loc[all_data['bacno']==b,'{}_shift1'.format(c)]=all_data.loc[all_data['bacno']==b,c].shift(periods=1,fill_value=-1)
        all_data.loc[all_data['bacno']==b,'{}_shiftm1'.format(c)]=all_data.loc[all_data['bacno']==b,c].shift(periods=-1, fill_value=-1)

# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../../data/preprocess/FE_data9_{}.csv'.format(shift_file_num),index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))