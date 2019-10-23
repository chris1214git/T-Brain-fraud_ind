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

bacno_list = all_data.bacno.unique()
for b in bacno_list:
    all_data[all_data['bacno']==b]

# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../data/preprocess/FE_data4.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))