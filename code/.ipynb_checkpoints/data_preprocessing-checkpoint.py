{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# [玉山人工智慧公開挑戰賽2019秋季賽,真相只有一個 -『信用卡盜刷偵測』](https://tbrain.trendmicro.com.tw/Competitions/Details/10)\n",
    "\n",
    "## <font color=red>任務:預測某刷卡交易是否為盜刷</font>\n",
    "\n",
    "### Task Schedule:\n",
    "1. 讀取資料,將字串轉換成int\n",
    "2. EDA(exploratory data analysis)\n",
    "3. Feature engineering\n",
    "4. 訓練模型,調整參數(預計使用lgb，速度較快)\n",
    "5. 嘗試使用不同模型,做Ensamble(blending, stacking)\n",
    "6. Anomaly detection\n",
    "\n",
    "### 注意事項:\n",
    "1. 因為test data和train data時間不相關,在驗證時採取前60天訓練61~90天驗證,但仍需小心時間差異造成的影響\n",
    "\n",
    "### TODO:\n",
    "1. **EDA(見下方詳細解釋）,找出不適合作為training feature的特徵,加以轉化成高級特徵或刪除**\n",
    "2. **找data leakage**\n",
    "\n",
    "3. Anomaly detection: 看這類的模型能不能取代lgb(似乎是不行，盜刷數據並沒有那麼Anomaly）,但可以嘗試將Anomaly結果當成新feature\n",
    "\n",
    "### <font color=green>Results:</font>\n",
    "* 不做處理,直接丟lgb訓練 leaderboard score:0.45\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 讀取,轉換字串成可以訓練的資料"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import math\n",
    "\n",
    "%matplotlib inline\n",
    "data_path = '../data'\n",
    "\n",
    "random_seed = 2000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/kaiwen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:13: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version\n",
      "of pandas will change to not sort by default.\n",
      "\n",
      "To accept the future behavior, pass 'sort=False'.\n",
      "\n",
      "To retain the current behavior and silence the warning, pass 'sort=True'.\n",
      "\n",
      "  del sys.path[0]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RangeIndex(start=0, stop=1943452, step=1)\n",
      "(1521787, 23)\n",
      "(421665, 22)\n",
      "(1943452, 23)\n"
     ]
    }
   ],
   "source": [
    "train_data_path = os.path.join(data_path,'train.zip')\n",
    "train_data = pd.read_csv(train_data_path, encoding = \"big5\")\n",
    "\n",
    "test_data_path = os.path.join(data_path,'test.zip')\n",
    "test_data = pd.read_csv(test_data_path, encoding = \"big5\")\n",
    "\n",
    "train_data_num = train_data.shape[0]\n",
    "test_data_txkey = test_data['txkey'].copy()\n",
    "\n",
    "train_data = train_data.sort_values(by=['bacno','locdt','loctm']).reset_index(drop=True)\n",
    "label_data = train_data['fraud_ind'].copy()\n",
    "\n",
    "all_data = pd.concat([train_data,test_data],axis=0).reset_index(drop=True)\n",
    "print(all_data.index)\n",
    "print(train_data.shape)\n",
    "print(test_data.shape)\n",
    "print(all_data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>acqic</th>\n",
       "      <th>bacno</th>\n",
       "      <th>cano</th>\n",
       "      <th>conam</th>\n",
       "      <th>contp</th>\n",
       "      <th>csmcu</th>\n",
       "      <th>ecfg</th>\n",
       "      <th>etymd</th>\n",
       "      <th>flbmk</th>\n",
       "      <th>flg_3dsmk</th>\n",
       "      <th>...</th>\n",
       "      <th>iterm</th>\n",
       "      <th>locdt</th>\n",
       "      <th>loctm</th>\n",
       "      <th>mcc</th>\n",
       "      <th>mchno</th>\n",
       "      <th>ovrlt</th>\n",
       "      <th>scity</th>\n",
       "      <th>stocn</th>\n",
       "      <th>stscd</th>\n",
       "      <th>txkey</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6413</td>\n",
       "      <td>1</td>\n",
       "      <td>117264</td>\n",
       "      <td>934.49</td>\n",
       "      <td>5</td>\n",
       "      <td>62</td>\n",
       "      <td>N</td>\n",
       "      <td>4</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>200000.0</td>\n",
       "      <td>275</td>\n",
       "      <td>53099</td>\n",
       "      <td>N</td>\n",
       "      <td>5817</td>\n",
       "      <td>102</td>\n",
       "      <td>0</td>\n",
       "      <td>1549254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>6189</td>\n",
       "      <td>1</td>\n",
       "      <td>117264</td>\n",
       "      <td>939.19</td>\n",
       "      <td>5</td>\n",
       "      <td>62</td>\n",
       "      <td>Y</td>\n",
       "      <td>2</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>221428.0</td>\n",
       "      <td>317</td>\n",
       "      <td>90151</td>\n",
       "      <td>N</td>\n",
       "      <td>1463</td>\n",
       "      <td>102</td>\n",
       "      <td>0</td>\n",
       "      <td>1837177</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6189</td>\n",
       "      <td>1</td>\n",
       "      <td>117264</td>\n",
       "      <td>1267.47</td>\n",
       "      <td>5</td>\n",
       "      <td>62</td>\n",
       "      <td>Y</td>\n",
       "      <td>2</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>25</td>\n",
       "      <td>212635.0</td>\n",
       "      <td>317</td>\n",
       "      <td>90151</td>\n",
       "      <td>N</td>\n",
       "      <td>1463</td>\n",
       "      <td>102</td>\n",
       "      <td>0</td>\n",
       "      <td>1859385</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6231</td>\n",
       "      <td>1</td>\n",
       "      <td>117264</td>\n",
       "      <td>1017.37</td>\n",
       "      <td>5</td>\n",
       "      <td>62</td>\n",
       "      <td>N</td>\n",
       "      <td>5</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>30</td>\n",
       "      <td>200947.0</td>\n",
       "      <td>277</td>\n",
       "      <td>12726</td>\n",
       "      <td>N</td>\n",
       "      <td>5817</td>\n",
       "      <td>102</td>\n",
       "      <td>0</td>\n",
       "      <td>994333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6189</td>\n",
       "      <td>1</td>\n",
       "      <td>117264</td>\n",
       "      <td>613.81</td>\n",
       "      <td>5</td>\n",
       "      <td>62</td>\n",
       "      <td>N</td>\n",
       "      <td>4</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>34</td>\n",
       "      <td>150512.0</td>\n",
       "      <td>263</td>\n",
       "      <td>92571</td>\n",
       "      <td>N</td>\n",
       "      <td>5817</td>\n",
       "      <td>102</td>\n",
       "      <td>0</td>\n",
       "      <td>1639576</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 23 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   acqic  bacno    cano    conam  contp  csmcu ecfg  etymd flbmk flg_3dsmk  \\\n",
       "0   6413      1  117264   934.49      5     62    N      4     N         N   \n",
       "1   6189      1  117264   939.19      5     62    Y      2     N         N   \n",
       "2   6189      1  117264  1267.47      5     62    Y      2     N         N   \n",
       "3   6231      1  117264  1017.37      5     62    N      5     N         N   \n",
       "4   6189      1  117264   613.81      5     62    N      4     N         N   \n",
       "\n",
       "   ...  iterm  locdt     loctm  mcc  mchno  ovrlt  scity  stocn stscd    txkey  \n",
       "0  ...      0      3  200000.0  275  53099      N   5817    102     0  1549254  \n",
       "1  ...      0      4  221428.0  317  90151      N   1463    102     0  1837177  \n",
       "2  ...      0     25  212635.0  317  90151      N   1463    102     0  1859385  \n",
       "3  ...      0     30  200947.0  277  12726      N   5817    102     0   994333  \n",
       "4  ...      0     34  150512.0  263  92571      N   5817    102     0  1639576  \n",
       "\n",
       "[5 rows x 23 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing value training data:\n",
      " flbmk        12581\n",
      "flg_3dsmk    12581\n",
      "dtype: int64\n",
      "Missing value testing data:\n",
      " flbmk        3715\n",
      "flg_3dsmk    3715\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "all_data.ecfg = all_data.ecfg.map({'N':0,'Y':1})\n",
    "all_data.ovrlt = all_data.ovrlt.map({'N':0,'Y':1})\n",
    "all_data.insfg = all_data.insfg.map({'N':0,'Y':1})\n",
    "all_data.flbmk = all_data.flbmk.map({'N':0,'Y':1})\n",
    "all_data.flg_3dsmk = all_data.flg_3dsmk.map({'N':0,'Y':1})\n",
    "all_data.loctm = all_data.loctm.astype(int)\n",
    "all_data = all_data.infer_objects()\n",
    "\n",
    "# print(all_data.dtypes)\n",
    "print('Missing value training data:\\n',train_data.isna().sum()[train_data.isna().sum()>0])\n",
    "print('Missing value testing data:\\n',test_data.isna().sum()[test_data.isna().sum()>0])\n",
    "\n",
    "## not neccessary to fill null value, since we use lgb model\n",
    "all_data.flbmk = all_data.flbmk.fillna(value=all_data.flbmk.mean(skipna=True))\n",
    "all_data.flg_3dsmk = all_data.flg_3dsmk.fillna(value=all_data.flg_3dsmk.mean(skipna=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dirty Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False    1943422\n",
      "True          30\n",
      "dtype: int64\n",
      "0.0\n"
     ]
    }
   ],
   "source": [
    "weird1 = (all_data['insfg']==1)&(all_data['iterm']==0)\n",
    "print(weird1.value_counts())\n",
    "print(all_data[weird1]['fraud_ind'].sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "binary_list=['ecfg','insfg','ovrlt','flbmk','flg_3dsmk']\n",
    "category_list=['contp','etymd','hcefg','stocn','scity','stscd','csmcu']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature engineering\n",
    "* train & valid only（先不考慮test data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bin cut"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "## transform large type category features 轉換有大量類別的特徵\n",
    "## 第一種轉法: bin cut,只留下數量最多的類別,將資料數少的類別都分成同一類,\n",
    "## 第二轉種法: 根據fraud_ind的bacno數量,決定要留下哪些類別,剩下的分成同一類(要仔細觀察train和valid的關係,避免overfitting)\n",
    "\n",
    "# th=100\n",
    "# category_list = all_data['mchno'].value_counts()[:th].index\n",
    "# all_data2 = all_data.copy()\n",
    "# all_data2[~all_data['mchno'].isin(category_list)]=-1\n",
    "# print(all_data2['mchno'].value_counts()[:th])\n",
    "\n",
    "# th=100\n",
    "# category_list = all_data['acqic'].value_counts()[:th].index\n",
    "# all_data2 = all_data.copy()\n",
    "# all_data2[~all_data['acqic'].isin(category_list)]=-1\n",
    "# print(all_data2['acqic'].value_counts()[:th])\n",
    "\n",
    "# th=100\n",
    "# category_list = all_data['mcc'].value_counts()[:th].index\n",
    "# all_data2 = all_data.copy()\n",
    "# all_data2[~all_data['mcc'].isin(category_list)]=-1\n",
    "# print(all_data2['mcc'].value_counts()[:th])\n",
    "\n",
    "th=15\n",
    "category_list = all_data['stocn'].value_counts()[:th].index\n",
    "all_data2 = all_data.copy()\n",
    "all_data2[~all_data['stocn'].isin(category_list)]=-1\n",
    "# print(all_data2['stocn'].value_counts()[:th])\n",
    "\n",
    "th=20\n",
    "category_list = all_data['scity'].value_counts()[:th].index\n",
    "all_data2 = all_data.copy()\n",
    "all_data2[~all_data['scity'].isin(category_list)]=-1\n",
    "# print(all_data2['scity'].value_counts()[:th])\n",
    "\n",
    "th=10\n",
    "category_list = all_data['csmcu'].value_counts()[:th].index\n",
    "all_data2 = all_data.copy()\n",
    "all_data2[~all_data['csmcu'].isin(category_list)]=-1\n",
    "# print(all_data2['csmcu'].value_counts()[:th])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "one_cut = all_data[all_data['locdt']<=120]['txkey'].max()/20\n",
    "all_data['txkey_bin'] = all_data['txkey']//one_cut"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0    1127387\n",
      "0.0      36245\n",
      "Name: cano_not1, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "all_data['locdt_week'] = all_data['locdt']%7+1\n",
    "# all_data['locdt_month'] = all_data['locdt']%30+1\n",
    "\n",
    "all_data['loctm_hr'] = all_data['loctm'].apply(lambda s:s//10000).astype(int)\n",
    "# all_data['loctm_hr2'] = all_data['loctm'].apply(lambda s:s//1000).astype(int)\n",
    "# all_data['loctm_hr_sin'] = all_data['loctm_hr'].apply(lambda s:math.sin(s/24*math.pi)).astype(int)\n",
    "# all_data['loctm_hr2_sin'] = all_data['loctm_hr2'].apply(lambda s:math.sin(s/240*math.pi)).astype(int)\n",
    "\n",
    "mean_df = all_data.groupby(['bacno'])['cano'].nunique().reset_index()\n",
    "mean_df.columns = ['bacno', 'cano_not1']\n",
    "mean_df[mean_df['cano_not1']>1]=0\n",
    "all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n",
    "print(all_data['cano_not1'].value_counts())\n",
    "\n",
    "mean_df = all_data.groupby(['bacno'])['txkey'].nunique().reset_index()\n",
    "mean_df.columns = ['bacno', 'txkey'+'_count']\n",
    "all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n",
    "\n",
    "# mean_df = all_data.groupby(['bacno'])['loctm_hr'].mean().reset_index()\n",
    "# mean_df.columns = ['bacno', 'loctm_hr'+'_mean']\n",
    "# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n",
    "\n",
    "# mean_df = all_data.groupby(['bacno'])['loctm_hr'].var().reset_index()\n",
    "# mean_df.columns = ['bacno', 'loctm_hr'+'_var']\n",
    "# mean_df.fillna(value=-1,inplace=True)\n",
    "# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "l_list=['mchno','acqic','mcc','stocn','scity','csmcu']\n",
    "for l in l_list:\n",
    "    tmp_df = all_data.groupby([l])['bacno'].nunique().reset_index()\n",
    "    tmp_df.columns = [l, l+'_bacno_nunique']\n",
    "    all_data = pd.merge(all_data,tmp_df, on=l, how='left')\n",
    "    \n",
    "for l in l_list:\n",
    "    tmp_df = all_data.groupby([l])['cano'].nunique().reset_index()\n",
    "    tmp_df.columns = [l, l+'_cano_nunique']\n",
    "    all_data = pd.merge(all_data,tmp_df, on=l, how='left') \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Personal feature engineering\n",
    "\n",
    "#### 由於模型讀沒有辦法讀取用戶歷史記錄，我們手動製作跟用戶相關的歷史特徵\n",
    "* 根據bacno or cano製作\n",
    "* conam,ecfg,stocn,stscd,csmcu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "## conam\n",
    "all_data['conam'] = np.log(all_data['conam']+2)\n",
    "\n",
    "# bacno_mean_conam:某個使用者平均的消費\n",
    "mean_df = all_data.groupby(['bacno'])['conam'].mean().reset_index()\n",
    "mean_df.columns = ['bacno','bacno_mean_conam']\n",
    "all_data = pd.merge(all_data,mean_df,on='bacno',how='left')\n",
    "\n",
    "# bacno_scale_conam:某使用者相對自己平均消費的金額\n",
    "all_data['bacno_scale_conam'] = all_data['conam']-all_data['bacno_mean_conam']\n",
    "\n",
    "# cano_mean_conam:某個卡片的平均消費\n",
    "mean_df = all_data.groupby(['cano'])['conam'].mean().reset_index()\n",
    "mean_df.columns = ['cano','cano_mean_conam']\n",
    "all_data = pd.merge(all_data,mean_df,on='cano',how='left')\n",
    "\n",
    "# cano_scale_conam:相對卡片自己平均消費的金額\n",
    "all_data['cano_scale_conam'] = all_data['conam']-all_data['cano_mean_conam']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "33983\n",
      "0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/kaiwen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:21: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8495\n",
      "16990\n",
      "25485\n",
      "33980\n"
     ]
    }
   ],
   "source": [
    "## ecfg\n",
    "## 連續且唯一的ecfg出現，標記1\n",
    "\n",
    "def one_consecutive_ecfg(s):    \n",
    "    s2 = s.map({0:' ',1:'1'})\n",
    "    count=len([x for x in ''.join(s2).split()])\n",
    "    if count==1:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "    \n",
    "bacno_consecutive_ecfg = all_data.groupby(['bacno'])['ecfg'].apply(one_consecutive_ecfg)\n",
    "bacno_consecutive_ecfg2 = bacno_consecutive_ecfg[bacno_consecutive_ecfg==1]\n",
    "print(bacno_consecutive_ecfg2.shape[0])\n",
    "\n",
    "all_data['bacno_consecutive_and_only_ecfg']=0\n",
    "\n",
    "for i in range(bacno_consecutive_ecfg2.shape[0]):\n",
    "    if i%(bacno_consecutive_ecfg2.shape[0]//4)==0:\n",
    "        print(i)\n",
    "    all_data[all_data['bacno']==bacno_consecutive_ecfg2.index[i]]['bacno_consecutive_and_only_ecfg']=all_data[all_data['bacno']==bacno_consecutive_ecfg2.index[i]]['ecfg']==1\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "500000\n",
      "1000000\n",
      "1500000\n"
     ]
    }
   ],
   "source": [
    "## stocn\n",
    "\n",
    "## stocn_nunique\n",
    "tmp_df = all_data.groupby(['bacno'])['stocn'].nunique().reset_index()\n",
    "tmp_df.columns = ['bacno','bacno_nunique_stocn']\n",
    "all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')\n",
    "\n",
    "## stocn_value_counts\n",
    "bacno_list = all_data['bacno'].unique()\n",
    "all_data['stocn_value_counts']=0\n",
    "stocn_value_counts_dict={}\n",
    "\n",
    "for b in bacno_list:\n",
    "    stocn_value_counts_dict[b] = all_data[all_data['bacno']==b]['stocn'].value_counts() \n",
    "\n",
    "for i in range(all_data.shape[0]):\n",
    "    if i%500000==0:\n",
    "        print(i)\n",
    "    all_data.loc[i,'stocn_value_counts']=stocn_value_counts_dict[all_data.iloc[i]['bacno']][all_data.iloc[i]['stocn']]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hi\n",
      "0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/kaiwen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:20: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-15-f4c4f4123af6>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     18\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m%\u001b[0m\u001b[0;36m500000\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m         \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m     \u001b[0mall_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'csmcu_value_counts'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcsmcu_value_counts_dict\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mall_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'bacno'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mall_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'csmcu'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/series.py\u001b[0m in \u001b[0;36m__setitem__\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   1036\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1037\u001b[0m         \u001b[0;31m# do the setitem\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1038\u001b[0;31m         \u001b[0mcacher_needs_updating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_is_chained_assignment_possible\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1039\u001b[0m         \u001b[0msetitem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1040\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcacher_needs_updating\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_check_is_chained_assignment_possible\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   3200\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3201\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_copy\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3202\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_setitem_copy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstacklevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mt\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'referant'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3203\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3204\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_check_setitem_copy\u001b[0;34m(self, stacklevel, t, force)\u001b[0m\n\u001b[1;32m   3243\u001b[0m             \u001b[0;31m# the copy weakref\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3244\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3245\u001b[0;31m                 \u001b[0mgc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcollect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3246\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mgc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_referents\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_copy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3247\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_copy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "## csmcu\n",
    "\n",
    "## csmcu_nunique\n",
    "tmp_df = all_data.groupby(['bacno'])['csmcu'].nunique().reset_index()\n",
    "tmp_df.columns = ['bacno','bacno_nunique_csmcu']\n",
    "all_data = pd.merge(all_data,tmp_df,on='bacno',how='left')\n",
    "\n",
    "print('hi')\n",
    "## csmcu_value_counts\n",
    "bacno_list = all_data['bacno'].unique()\n",
    "all_data['csmcu_value_counts']=0\n",
    "csmcu_value_counts_dict={}\n",
    "\n",
    "for b in bacno_list:\n",
    "    csmcu_value_counts_dict[b] = all_data[all_data['bacno']==b]['csmcu'].value_counts() \n",
    "\n",
    "for i in range(all_data.shape[0]):\n",
    "    if i%500000==0:\n",
    "        print(i)\n",
    "    all_data.iloc[i]['csmcu_value_counts']=csmcu_value_counts_dict[all_data.iloc[i]['bacno']][all_data.iloc[i]['csmcu']]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-16-e65720d61a52>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     23\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m%\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mconsecutive_cano2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m//\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m         \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 25\u001b[0;31m     \u001b[0mall_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mall_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'cano'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0mconsecutive_cano2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'cano_only_consecutive_stscd2'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mall_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mall_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'cano'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0mconsecutive_cano2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'stscd'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     26\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__setitem__\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   3368\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3369\u001b[0m             \u001b[0;31m# set column\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3370\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_set_item\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3371\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3372\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_setitem_slice\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m_set_item\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   3450\u001b[0m         \u001b[0;31m# value exception to occur first\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3451\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3452\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_setitem_copy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3453\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3454\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0minsert\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcolumn\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mallow_duplicates\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.5/site-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_check_setitem_copy\u001b[0;34m(self, stacklevel, t, force)\u001b[0m\n\u001b[1;32m   3243\u001b[0m             \u001b[0;31m# the copy weakref\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3244\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3245\u001b[0;31m                 \u001b[0mgc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcollect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3246\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mgc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_referents\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_copy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3247\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_copy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "## 每個用戶，連續且唯一的stscd==2\n",
    "\n",
    "def one_consecutive_stscd(s):\n",
    "#     s2 = s.diff(1)\n",
    "#     print((s2!=0).sum(skipna=True)!=2)\n",
    "#     return (s2!=0).sum(skipna=True)!=2\n",
    "    \n",
    "    s2 = s.map({0:' ',2:'1'})\n",
    "    s2 = s2.fillna(value='-1')\n",
    "    count=len([x for x in ''.join(s2).split()])\n",
    "    if count==1:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "    \n",
    "consecutive_cano = all_data.groupby(['cano'])['stscd'].apply(one_consecutive_stscd)\n",
    "consecutive_cano2 = consecutive_cano[consecutive_cano==1]\n",
    "print(consecutive_cano2.shape[0])\n",
    "\n",
    "all_data['cano_only_consecutive_stscd2']=0\n",
    "\n",
    "for i in range(consecutive_cano2.shape[0]):\n",
    "    if i%(consecutive_cano2.shape[0]//4)==0:\n",
    "        print(i)\n",
    "    all_data[all_data['cano']==consecutive_cano2.index[i]]['cano_only_consecutive_stscd2']=all_data[all_data['cano']==consecutive_cano2.index[i]]['stscd']==2\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Leakage\n",
    "* cano在被盜取後會換卡片，觀察fraud data製作cano相關 features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## 某用戶第一次使用該卡片，且最後一天並不是使用該卡片，將最後一天使用該卡片的交易給值1，其餘0\n",
    "## 用merge會比for迴圈快很多\n",
    "def lastday_cano(s):\n",
    "    cano_firstid = s['cano'].iloc[0]    \n",
    "    if s['cano'].iloc[-1]==cano_firstid:\n",
    "        return -1\n",
    "    \n",
    "    return s[s['cano']==cano_firstid]['locdt'].max()\n",
    "\n",
    "\n",
    "cano_firstid = all_data.groupby(['bacno'])['cano'].apply(lambda s: s.iloc[0]).reset_index()\n",
    "cano_lastday = all_data.groupby(['bacno']).apply(lastday_cano).reset_index()\n",
    "# print(cano_lastday)\n",
    "\n",
    "cano_lastday_use = pd.merge(cano_firstid, cano_lastday, on='bacno',how='left')\n",
    "cano_lastday_use.columns = ['bacno', 'cano', 'locdt']\n",
    "cano_lastday_use['cano_lastday_use']=cano_lastday_use['locdt']!=-1\n",
    "\n",
    "all_data = pd.merge(all_data,cano_lastday_use,on=['bacno','cano','locdt'], how='left')\n",
    "all_data['cano_lastday_use'] = all_data['cano_lastday_use'].fillna(value=False)\n",
    "all_data['cano_lastday_use'] = all_data['cano_lastday_use'].map({True:1,False:0})\n",
    "\n",
    "print(cano_lastday_use['cano_lastday_use'].sum())\n",
    "print(all_data['cano_lastday_use'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 和Fraud相關的特徵工程\n",
    "#### 先使用於train上 檢查validation結果,小心overfit\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## 某卡片過去是否有盜刷記錄\n",
    "## 記錄每個卡片第一天被盜刷的日期,\n",
    "def cano_find_firstday(d):\n",
    "    return d[d['fraud_ind']==1]['locdt'].min()\n",
    "    \n",
    "cano_firstday = all_data.groupby(['cano']).apply(cano_find_firstday)\n",
    "cano_hasfraud = cano_firstday[cano_firstday>=0]\n",
    "\n",
    "all_data['cano_hasfraud_before']=0\n",
    "for i in range(cano_hasfraud.shape[0]):\n",
    "    if i%2000==0:\n",
    "        print(i)\n",
    "    all_data[(all_data['cano']==cano_hasfraud.index[i])&\\\n",
    "             (all_data['locdt']>=cano_hasfraud.iloc[i])]['cano_hasfraud_before']=1\n",
    "\n",
    "print(all_data[all_data['cano']==cano_hasfraud.index[10]][['locdt','fraud_ind']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for i in range(500):\n",
    "#     print(i,all_data.groupby(['bacno']).get_group(i)[['ecfg','fraud_ind']])\n",
    "# mean_df = all_data.groupby(['bacno'])['fraud_ind'].mean().reset_index()\n",
    "# mean_df.columns = ['bacno', 'loctm_hr'+'_mean']\n",
    "# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n",
    "\n",
    "# print(all_data[['bacno','locdt','loctm']])\n",
    "\n",
    "# 該交易的歸戶帳號是否曾經被盜刷 0->沒 1->有 -1->無紀錄\n",
    "\n",
    "\n",
    "# 該交易的歸戶帳號是否曾經被盜刷卻又復原\n",
    "# 該交易的歸戶帳號是否第一次刷卡\n",
    "# 該交易的歸戶帳號第幾次刷卡\n",
    "\n",
    "# 該交易的卡號是否曾經被盜刷\n",
    "# 該交易的卡號是否曾經被盜刷卻又復原\n",
    "# 該交易的卡號是否第一次刷卡\n",
    "# 該交易的卡號第幾次刷卡\n",
    "\n",
    "# mean_df = all_data.groupby(['bacno']).apply(lambda s:s.mode()).reset_index()\n",
    "# mean_df.columns = ['bacno', 'stocn'+'_mode']\n",
    "# mean_df.fillna(-1,inplace=True)\n",
    "# print(mean_df.stocn_mode.value_counts())\n",
    "# all_data = pd.merge(all_data, mean_df, on='bacno', how='left')\n",
    "\n",
    "# 消費國別是否跟自己所有消費的眾數不一樣\n",
    "# 消費城市是否跟自己所有消費的眾數不一樣\n",
    "# 消費地幣別是否跟自己所有消費的眾數不一樣\n",
    "# 支付型態是否跟自己所有消費的眾數不一樣\n",
    "# 分期期數是否跟自己所有消費的眾數不一樣\n",
    "\n",
    "# 是否第一次網路消費且過去有非網路消費的經驗\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# data = pd.concat([df[:train_num], train_Y], axis=1)\n",
    "# for c in df.columns:\n",
    "#     mean_df = data.groupby([c])['SalePrice'].mean().reset_index()\n",
    "#     mean_df.columns = [c, f'{c}_mean']\n",
    "#     data = pd.merge(data, mean_df, on=c, how='left')\n",
    "#     data = data.drop([c] , axis=1)\n",
    "\n",
    "\n",
    "# all_data['howmany_cano'] = \n",
    "# all_data['howmany_txkey'] = \n",
    "\n",
    "## bacno刷卡頻率分佈\n",
    "\n",
    "# all_data['fraud_before'] =\n",
    "# all_data['fraud_last_time'] =\n",
    "\n",
    "# 印出某個被盜刷的人的刷卡使用時間分佈\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_data.to_csv('../data/all_data.csv',index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train on LGB(未調參數)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "delete_list = ['bacno','locdt','loctm','cano','fraud_ind','iterm']\n",
    "#txkey大小, cano可能會重複所以重要？"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = all_data[all_data['locdt']<=60].drop(columns=delete_list)\n",
    "y_train = all_data[all_data['locdt']<=60]['fraud_ind']\n",
    "X_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)].drop(columns=delete_list)\n",
    "y_test = all_data[(all_data['locdt']>60) & (all_data['locdt']<=90)]['fraud_ind']\n",
    "\n",
    "print(delete_list)\n",
    "print(X_train.shape)\n",
    "print(y_train.sum()/y_train.shape[0])\n",
    "print(y_test.sum()/y_test.shape[0])\n",
    "\n",
    "\n",
    "import lightgbm as lgb\n",
    "from lightgbm.sklearn import LGBMClassifier\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "def lgb_f1_score(y_true, y_pred):\n",
    "    y_pred = np.round(y_pred) # scikits f1 doesn't like probabilities\n",
    "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n",
    "    print()\n",
    "    print('tn, fp, fn, tp')\n",
    "    print(tn, fp, fn, tp)\n",
    "    return 'f1', f1_score(y_true, y_pred), True\n",
    "\n",
    "param_dist_lgb = {\n",
    "#                   'num_leaves':45, \n",
    "#                   'max_depth':5, \n",
    "                  'learning_rate':0.1, \n",
    "                  'n_estimators':600,\n",
    "                  'objective': 'binary',\n",
    "#                   'subsample': 1, \n",
    "#                   'colsample_bytree': 0.5, \n",
    "#                   'lambda_l1': 0.1,\n",
    "#                   'lambda_l2': 0,\n",
    "#                   'min_child_weight': 1,\n",
    "                  'random_state': random_seed,\n",
    "                 }\n",
    "evals_result = {}\n",
    "\n",
    "lgb_clf = LGBMClassifier(**param_dist_lgb)\n",
    "lgb_clf.fit(X_train, y_train,\n",
    "        eval_set=[(X_train, y_train),(X_test, y_test)],\n",
    "        eval_metric=lgb_f1_score,\n",
    "        early_stopping_rounds=200,\n",
    "        verbose=True,\n",
    "        callbacks=[lgb.record_evaluation(evals_result)]\n",
    "        )\n",
    "y_test_pred = lgb_clf.predict(X_test)\n",
    "print('F1',f1_score(y_test, y_test_pred))\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()\n",
    "print(tn, fp, fn, tp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Plotting metrics recorded during training...')\n",
    "ax = lgb.plot_metric(evals_result, metric='f1')\n",
    "plt.show()\n",
    "\n",
    "print('Plotting feature importances...')\n",
    "ax = lgb.plot_importance(lgb_clf, max_num_features=10)\n",
    "plt.show()\n",
    "\n",
    "print('Plotting 4th tree...')  # one tree use categorical feature to split\n",
    "ax = lgb.plot_tree(lgb_clf, tree_index=3, figsize=(15, 15), show_info=['split_gain'])\n",
    "plt.show()\n",
    "\n",
    "print('Plotting 4th tree with graphviz...')\n",
    "graph = lgb.create_tree_digraph(lgb_clf, tree_index=3, name='Tree4')\n",
    "graph.render(view=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train on catboost\n",
    "* https://catboost.ai/docs/concepts/python-reference_parameters-list.html\n",
    "* 研究有哪些可以用的function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from catboost import CatBoostClassifier, Pool\n",
    "\n",
    "test_data = catboost_pool = Pool(train_data, \n",
    "                                 train_labels)\n",
    "\n",
    "model = CatBoostClassifier(iterations=2,\n",
    "                           depth=2,\n",
    "                           learning_rate=1,\n",
    "                           loss_function='Logloss',\n",
    "                           verbose=True)\n",
    "# train the model\n",
    "model.fit(train_data, train_labels)\n",
    "# make the prediction using the resulting model\n",
    "preds_class = model.predict(test_data)\n",
    "preds_proba = model.predict_proba(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip3 install catboost --user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 用hinge loss(當SVM)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# X_train['cents']\n",
    "# encoding data\n",
    "\n",
    "# GroupKfold\n",
    "# vanilla KFold"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## write csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# lgb_clf = LGBMClassifier(**param_dist_lgb)\n",
    "# lgb_clf.fit(train_data,label_data)\n",
    "\n",
    "# result = lgb_clf.predict(test_data)\n",
    "# print(result.sum())\n",
    "# print(result.sum()/result.shape[0])\n",
    "# print(label_data.sum()/label_data.shape[0])\n",
    "\n",
    "# test_data_txkey = test_data['txkey'].copy()\n",
    "\n",
    "# import csv\n",
    "# with open('../prediction/submit_lgb.csv','w') as f:\n",
    "#     writer = csv.writer(f)\n",
    "#     writer.writerow(['txkey','fraud_ind'])\n",
    "#     for i in range(result.shape[0]):\n",
    "#         writer.writerow([test_data_txkey[i], result[i]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Anomaly detection\n",
    "* one class svm\n",
    "* isolation tree\n",
    "* replicator NN\n",
    "* Kmeans?\n",
    "* KNN(take too much time)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 製作特徵\n",
    "XGB, PCA, Isolation Forest, Kmean距離？, oneclass SVM?\n",
    "當作新feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xgboost as xgb\n",
    "param_dist_xgb = {'learning_rate':0.01, #默认0.3\n",
    "              'n_estimators':1000, #树的个数\n",
    "#               'max_depth':5,\n",
    "#               'min_child_weight':1,\n",
    "#               'gamma':0.2,\n",
    "#               'subsample':0.8,\n",
    "#               'colsample_bytree':0.8,\n",
    "#               'objective': 'binary:logistic', #逻辑回归损失函数\n",
    "#               'nthread':4,  #cpu线程数\n",
    "#               'scale_pos_weight':1,\n",
    "              'seed':random_seed}  #随机种子\n",
    "\n",
    "evals_result = {}\n",
    "\n",
    "xgb_clf = xgb.XGBClassifier(**param_dist_xgb)\n",
    "xgb_clf.fit(X_train, y_train,\n",
    "        eval_set=[(X_train, y_train),(X_test, y_test)],\n",
    "        eval_metric=lgb_f1_score,\n",
    "        early_stopping_rounds=600,\n",
    "        verbose=True,\n",
    "#         callbacks=[xgb.record_evaluation(evals_result)]\n",
    "        )\n",
    "\n",
    "print('F1',f1_score(y_test, xgb_clf.predict(X_test)))\n",
    "xgb_X_train = xgb_clf.apply(X_train)\n",
    "xgb_X_test = xgb_clf.apply(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PCA visualization in one person who has fraud data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "def PCA_plot(x,label):\n",
    "    x = x.drop(columns=)\n",
    "    \n",
    "    ## 應該先轉dummy,標準化,再PCA\n",
    "    dummy_list=['contp','etymd','stscd','hcefg']\n",
    "    dummy_list2=['stocn','scity','csmcu']#'mchno','acqic','mcc',\n",
    "    x[dummy_list] = x[dummy_list].astype(object)\n",
    "    x[dummy_list2] = x[dummy_list2].astype(object)\n",
    "    x = pd.get_dummies(x).drop(columns=['mchno','acqic'])    \n",
    "    \n",
    "    from sklearn.preprocessing import StandardScaler \n",
    "    stdsc = StandardScaler() \n",
    "    x = stdsc.fit_transform(x)\n",
    "    print(x.shape,label.sum())\n",
    "\n",
    "    PCA_model = PCA(n_components=2)\n",
    "    train_data_pca = PCA_model.fit_transform(x)\n",
    "    train_data_pca1 = train_data_pca[label==1]\n",
    "    train_data_pca0 = train_data_pca[label==0]\n",
    "    \n",
    "    plt.clf()\n",
    "    plt.figure(figsize=(10,10))\n",
    "    plt.scatter(train_data_pca1[:, 0], train_data_pca1[:, 1], c='r',label='fraud transaction',s=100)\n",
    "    plt.scatter(train_data_pca0[:, 0], train_data_pca0[:, 1], c='b',label='normal transaction',s=3)\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "    \n",
    "bacno_hasfraud = all_data[all_data['fraud_ind']==1]['bacno'].unique()\n",
    "print(bacno_hasfraud.shape[0])\n",
    "print(all_data[all_data['fraud_ind']==1].shape[0])\n",
    "\n",
    "for i in range(bacno_hasfraud.shape[0]):\n",
    "    if all_data[all_data['bacno']==bacno_hasfraud[i]].shape[0]>300:\n",
    "        print('Ploting PCA on bacno-{}'.format(bacno_hasfraud[i]))\n",
    "        PCA_plot(all_data[all_data['bacno']==bacno_hasfraud[i]],all_data[all_data['bacno']==bacno_hasfraud[i]]['fraud_ind'])\n",
    "\n",
    "## TSNE, Kmeans作圖?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Isolation Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import IsolationForest\n",
    "\n",
    "c_ratio = y_train.sum()/y_train.shape[0]\n",
    "# fit the model\n",
    "clf = IsolationForest(behaviour='new', max_samples=0.8, max_features=1,\n",
    "                      random_state=random_seed, contamination=c_ratio)\n",
    "clf.fit(X_train)\n",
    "\n",
    "y_pred_train = clf.predict(X_train)\n",
    "y_pred_test = clf.predict(X_test)\n",
    "\n",
    "y_pred_test2 = -y_pred_test\n",
    "y_pred_test2[y_pred_test2==-1]=0\n",
    "y_pred_test2.sum()\n",
    "\n",
    "y_pred_train2 = -y_pred_train\n",
    "y_pred_train2[y_pred_train2==-1]=0\n",
    "y_pred_train2.sum()\n",
    "\n",
    "from sklearn.metrics import f1_score\n",
    "print(f1_score(y_train, y_pred_train2))\n",
    "print(f1_score(y_test, y_pred_test2))\n",
    "\n",
    "isolationtree_X_train = clf.score_samples(X_train)\n",
    "isolationtree_X_test = clf.score_samples(X_test)\n",
    "\n",
    "print(isolationtree_X_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## One class SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import svm\n",
    "\n",
    "clf = svm.OneClassSVM(nu=0.1, kernel=\"rbf\", gamma='scale',verbose=True, random_state=random_seed)\n",
    "clf.fit(X_train)\n",
    "y_pred_train = clf.predict(X_train)\n",
    "y_pred_test = clf.predict(X_test)\n",
    "y_pred_test.sum()\n",
    "\n",
    "y_pred_train2 = -y_pred_train\n",
    "y_pred_train2[y_pred_train2==-1]=0\n",
    "y_pred_train2.sum()\n",
    "\n",
    "from sklearn.metrics import f1_score\n",
    "print(f1_score(y_train, y_pred_train2))\n",
    "print(f1_score(y_test, y_pred_test2))\n",
    "\n",
    "svm_X_train = clf.score_samples(X_train)\n",
    "svm_X_test = clf.score_samples(X_test)\n",
    "\n",
    "print(isolationtree_X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
