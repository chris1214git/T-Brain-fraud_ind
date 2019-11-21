import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
# get_ipython().run_line_magic('matplotlib', 'inline')
data_path = '../data'

random_seed = 2000
data_list=["raw_data.csv","FE_data1.csv","FE_data2.csv","FE_data2_2.csv","FE_data3.csv","FE_data4.csv",\
           "FE_data4_2.csv","FE_data5.csv","FE_data6.csv","FE_data7.csv","FE_data7_2.csv",\
            "pca_feature.csv","isolationtree_feature.csv","kmeans_feature.csv","svm_rbf_feature.csv"]

def load_data(data_list):
    data=[]
    for d in data_list:
        x = pd.read_csv('../../data/preprocess/{}'.format(d))
        x_null = x.isnull().sum()
        print('\n',d,x.shape)
        print("Null columns:\n",x_null[x_null>0])

        if (d=='FE_data1.csv') or (d=='FE_data2.csv'):
            x.fillna(value=-1,inplace=True)
        
        if d[:8]=='FE_data9':
            if d!='FE_data9_raw.csv':
                x = x.drop(columns=['bacno_shift1','bacno_shiftm1'])
        data.append(x)

    all_data = pd.concat(data,axis=1)
    del data
    all_data_numsum = all_data.isnull().sum()
    print('ALL data shape:',all_data.shape)
    print('ALL data null:')
    print(all_data_numsum[all_data_numsum>0])
    return all_data

def transform_data(all_data,category_list,bool_list):
    for c in category_list:
        if all_data[c].dtypes == 'float64':
            all_data[c] = all_data[c].astype('int')
        all_data[c]=all_data[c].astype('category')

    for c in all_data.columns[all_data.dtypes==bool]:
        all_data[c]=all_data[c].map({True:1,False:0})

    for c in bool_list:
        if c in all_data.columns:
            all_data[c]=all_data[c].map({'True':1,'False':0,'-1':-1})
    
    return all_data

all_data = load_data(data_list)
bool_cols = all_data.columns[all_data.dtypes==bool].values
for b in bool_cols:
    all_data[b] = all_data[b].map({True:1,False:0}) 

raw_col_num = all_data.shape[1]


def combine_category(arr,f1,f2,cnt1,cnt2):
    max_cnt = cnt1 if cnt1>cnt2 else cnt2
    max_id = 1 if cnt1>cnt2 else 2
    if max_id==1:
        return arr[f2]*max_cnt+arr[f1]
    elif max_id==2:
        return arr[f1]*max_cnt+arr[f2]
all_data['leakage_complex1'] = all_data.apply(combine_category,axis=1,f1='cano_only_consecutive_stscd2',f2='bacno_consecutive_and_only_ecfg',cnt1=2,cnt2=2)
all_data['leakage_complex2'] = all_data.apply(combine_category,axis=1,f1='bacno_stscd_equal2',f2='bacno_ecfg_equal1',cnt1=2,cnt2=2)
all_data['leakage_complex3'] = all_data.apply(combine_category,axis=1,f1='cano_lastlocdt2',f2='leakage_complex2',cnt1=2,cnt2=4)
all_data['leakage_complex4'] = all_data['cano_only_consecutive_stscd2'] + all_data['bacno_consecutive_and_only_ecfg']\
                               + all_data['bacno_stscd_equal2'] + all_data['bacno_ecfg_equal1']\
                               + all_data['cano_lastlocdt2']
all_data['contp_ecfg'] = all_data.apply(combine_category,axis=1,f1='contp',f2='ecfg',cnt1=7,cnt2=2)
all_data['csmcu_ecfg'] = all_data.apply(combine_category,axis=1,f1='csmcu',f2='ecfg',cnt1=76,cnt2=2)

all_data['bacno_mchno_ismode'] = all_data['bacno_mchno_mode']==all_data['mchno']
all_data['bacno_mchno_ismode']=all_data['bacno_mchno_ismode'].map({True:1,False:0})
all_data['cano_acqic_ismode'] = all_data['cano_acqic_mode']==all_data['acqic']
all_data['cano_acqic_ismode']=all_data['cano_acqic_ismode'].map({True:1,False:0})

# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv('../../data/preprocess/FE_data10.csv',index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))