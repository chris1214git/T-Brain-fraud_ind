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





# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv(save_data_file_path,index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))