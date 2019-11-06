import pandas as pd
import os
import numpy as np
import math

raw_data_file_path = '../../data/preprocess/raw_data.csv'
raw_data_file_path2 = '../../data/preprocess/FE_data7.csv'
raw_data_file_path3 = '../../data/preprocess/FE_data7_2.csv'
raw_data_file_path4 = '../../data/preprocess/FE_data7_3.csv'
raw_data_file_path5 = '../../data/preprocess/FE_data7_4.csv'

save_data_file_path = '../../data/preprocess/FE_data8.csv'
random_seed = 2000

raw_data = pd.read_csv(raw_data_file_path)
raw_data2 = pd.read_csv(raw_data_file_path2)
raw_data3 = pd.read_csv(raw_data_file_path3)
raw_data4 = pd.read_csv(raw_data_file_path4)
raw_data5 = pd.read_csv(raw_data_file_path5)

all_data = pd.concat([raw_data,raw_data2,raw_data3,raw_data4,raw_data5],axis=1)
print(all_data.shape)
print(all_data.head())
raw_col_num = all_data.shape[1]


diff_list =['contp','etymd','mchno','acqic','mcc','ecfg','hcefg','stscd','stocn','scity','csmcu']

all_data['bacno_diff1_sum']=all_data['bacno_contp_diff1']+all_data['bacno_etymd_diff1']+all_data['bacno_mchno_diff1']+\
                    all_data['bacno_acqic_diff1']+all_data['bacno_mcc_diff1']+all_data['bacno_ecfg_diff1']+all_data['bacno_hcefg_diff1']+\
                    all_data['bacno_stscd_diff1']+all_data['bacno_stocn_diff1']+all_data['bacno_csmcu_diff1']
all_data['bacno_diff2_sum']=all_data['bacno_contp_diff2']+all_data['bacno_etymd_diff2']+all_data['bacno_mchno_diff2']+\
                    all_data['bacno_acqic_diff2']+all_data['bacno_mcc_diff2']+all_data['bacno_ecfg_diff2']+all_data['bacno_hcefg_diff2']+\
                    all_data['bacno_stscd_diff2']+all_data['bacno_stocn_diff2']+all_data['bacno_csmcu_diff2']

all_data['cano_diff1_sum']=all_data['cano_contp_diff1']+all_data['cano_etymd_diff1']+all_data['cano_mchno_diff1']+\
                    all_data['cano_acqic_diff1']+all_data['cano_mcc_diff1']+all_data['cano_ecfg_diff1']+all_data['cano_hcefg_diff1']+\
                    all_data['cano_stscd_diff1']+all_data['cano_stocn_diff1']+all_data['cano_csmcu_diff1']
all_data['cano_diff2_sum']=all_data['cano_contp_diff2']+all_data['cano_etymd_diff2']+all_data['cano_mchno_diff2']+\
                    all_data['cano_acqic_diff2']+all_data['cano_mcc_diff2']+all_data['cano_ecfg_diff2']+all_data['cano_hcefg_diff2']+\
                    all_data['cano_stscd_diff2']+all_data['cano_stocn_diff2']+all_data['cano_csmcu_diff2']

all_data['bacno_diff_m1_sum']=0
all_data['bacno_diff_m2_sum']=0
all_data['cano_diff_m1_sum']=0
all_data['cano_diff_m2_sum']=0
for c in diff_list:
    c1 = 'bacno_{}_diff_m1'.format(c)
    c2 = 'bacno_{}_diff_m2'.format(c)
    c3 = 'cano_{}_diff_m1'.format(c)
    c4 = 'cano_{}_diff_m2'.format(c)
    all_data['bacno_diff_m1_sum']+=all_data[c1]
    all_data['bacno_diff_m2_sum']+=all_data[c2]
    all_data['cano_diff_m1_sum']+=all_data[c3]
    all_data['cano_diff_m2_sum']+=all_data[c4]
    



tmp_df = all_data.groupby(['bacno','stocn'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','stocn','locdt']
tmp_df['bacno_stocn_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','stocn','locdt'])
all_data['bacno_stocn_lastday'] = all_data['bacno_stocn_lastday'].fillna(value=0)

tmp_df = all_data.groupby(['bacno','scity'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','scity','locdt']
tmp_df['bacno_scity_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','scity','locdt'])
all_data['bacno_scity_lastday'] = all_data['bacno_scity_lastday'].fillna(value=0)

tmp_df = all_data.groupby(['bacno','csmcu'])['locdt'].max().reset_index()
tmp_df.columns = ['bacno','csmcu','locdt']
tmp_df['bacno_csmcu_lastday']=1
all_data = pd.merge(all_data,tmp_df,how='left',on=['bacno','csmcu','locdt'])
all_data['bacno_csmcu_lastday'] = all_data['bacno_csmcu_lastday'].fillna(value=0)


# write file
FE_data = all_data.iloc[:,raw_col_num:]
FE_data.to_csv(save_data_file_path,index=False)
print('saving FE_data, shape:{}'.format(FE_data.shape))