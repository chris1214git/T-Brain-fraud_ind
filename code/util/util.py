import time
import os
import json

def get_time_stamp():
    t_now = time.localtime( time.time() )
    mon = str(t_now.tm_mon) if (t_now.tm_mon)>=10 else '0'+str(t_now.tm_mon)
    day = str(t_now.tm_mday) if (t_now.tm_mday)>=10 else '0'+str(t_now.tm_mday)
    hour = str(t_now.tm_hour) if (t_now.tm_hour)>=10 else '0'+str(t_now.tm_hour)
    minute = str(t_now.tm_min) if (t_now.tm_min)>=10 else '0'+str(t_now.tm_min)
    t = mon+day+hour+minute
    return t

def write_submit_file(submit_file_path, submit_file_name, test_data_txkey, result):
    with open(os.path.join(submit_file_path,submit_file_name)+'.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['txkey','fraud_ind'])
        for i in range(result.shape[0]):
            writer.writerow([test_data_txkey[i], result[i]])

def describe_model(submit_file_name,param_cat,data_list,delete_list,bagging_time):
    with open('../prediction/submit_describe.txt','a') as f:
        print(submit_file_name,file=f)
        print('catboost parameter:\n{}'.format(param_cat),file=f)
        print('data list\n',data_list,file=f)
        print('delete list\n',delete_list,file=f)
        print('bagging time',bagging_time,file=f)

def load_json(json_file_name):
    path = '../code/para_dict/'
    with open(os.path.join(path,json_file_name),'r',encoding='utf-8') as f:
        para = json.loads(f.read())
    return para