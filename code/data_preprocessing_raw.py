import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data/raw_data'

random_seed = 2000


# In[2]:


train_data_path = os.path.join(data_path,'train.zip')
train_data = pd.read_csv(train_data_path, encoding = "big5")

test_data_path = os.path.join(data_path,'test.zip')
test_data = pd.read_csv(test_data_path, encoding = "big5")

train_data_num = train_data.shape[0]
test_data_txkey = test_data['txkey'].copy()
np.save('../data/preprocess/test_data_txkey',test_data_txkey.values)

train_data = train_data.sort_values(by=['bacno','locdt','loctm']).reset_index(drop=True)
test_data = test_data.sort_values(by=['bacno','locdt','loctm']).reset_index(drop=True)
label_data = train_data['fraud_ind'].copy()

all_data = pd.concat([train_data,test_data],axis=0).reset_index(drop=True)
print(all_data.index)
print(train_data.shape)
print(test_data.shape)
print(all_data.shape)


# In[3]:


# In[4]:


all_data.ecfg = all_data.ecfg.map({'N':0,'Y':1})
all_data.ovrlt = all_data.ovrlt.map({'N':0,'Y':1})
all_data.insfg = all_data.insfg.map({'N':0,'Y':1})
all_data.flbmk = all_data.flbmk.map({'N':0,'Y':1})
all_data.flg_3dsmk = all_data.flg_3dsmk.map({'N':0,'Y':1})
all_data.loctm = all_data.loctm.astype(int)
all_data = all_data.infer_objects()

# print(all_data.dtypes)
print('Missing value training data:\n',train_data.isna().sum()[train_data.isna().sum()>0])
print('Missing value testing data:\n',test_data.isna().sum()[test_data.isna().sum()>0])

## not neccessary to fill null value, since we use lgb model
all_data.flbmk = all_data.flbmk.fillna(value=all_data.flbmk.mean(skipna=True))
all_data.flg_3dsmk = all_data.flg_3dsmk.fillna(value=all_data.flg_3dsmk.mean(skipna=True))

all_data.to_csv('../data/preprocess/raw_data.csv',index=False)
print('saving all_data, shape:{}'.format(all_data.shape))
raw_col_num = all_data.shape[1]
