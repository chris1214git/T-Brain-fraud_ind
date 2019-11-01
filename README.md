# T-Brain-fraud_ind

## 官網:
https://tbrain.trendmicro.com.tw/Competitions/Details/10

## 資料夾分類
code:
data_preprocess:資料預處理,特徵工程
EDA:觀測檢查資料分佈,outlier,製作各種表格找資料間的pattern,特徵工程靈感
build_model:建立模型,調整參數,驗證特徵的好壞

data:
    raw_data/:原始dataset
    preprocess/:處理後,人工製造的特徵,執行datapreprocess的產物

prediction:
    log.txt:描述submit_*.csv


## 參考資料:

銀行刷卡原理:
https://progressbar.tw/posts/75

catboost:
https://catboost.ai/docs/concepts/tutorials.html
https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb

lgb:
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier
https://github.com/microsoft/LightGBM/tree/master/examples/python-guide

Kaggle:
https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions?fbclid=IwAR0GXWSIn0YnoqLgNPud7pE2Nz1WbH4yAskm2_qYM7kZ_6fWbYERj22MAIs
https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321
frequency encoding,切四種validation,permutation test將四次validation都沒有變好的特徵刪除

Jupyter:
markdown
https://medium.com/ibm-data-science-experience/markdown-for-jupyter-notebooks-cheatsheet-386c05aeebed

Pyecharts:
https://github.com/pyecharts/pyecharts

異常偵測:

### 程式整理
Readme markdown:
https://github.com/othree/markdown-syntax-zhtw
https://ithelp.ithome.com.tw/articles/10203758

argparse:
https://blog.csdn.net/lanzheng_1113/article/details/77574446

### 關於工作站:
用工作站肯定要學tmux，沒有為什麼，因為它很猛。最猛的是功能是可以關掉本地的電腦，工作站的程式仍然可以持續執行，可以保留工作狀態，且可以切割自己熟悉的工作畫面。
* Tmux
看這個youtube講解，一個小時內就幾乎都懂了
https://www.youtube.com/watch?v=nD6g-rM5Bh0&list=PLT98CRl2KxKGiyV1u6wHDV8VwcQdzfuKe
懂了之後的就存個cheatsheet，忘了的時候打開來看看
http://tmuxcheatsheet.com/

* github
因為我是獨立作業，所以用到的指令很少，最多遇到的問題只有不小心上傳超過100MB的檔案，但這些常見問題google一下就有答案了
https://www.alexkras.com/getting-started-with-git/


## 比賽技巧:

我記得把資料前處理好丟進lightgbm訓練，基本的f1分數只有0.45，算是蠻低的。原因多半是原始資料都是種類很多的類別型資料，很多序號類的特徵，以及有時間序列的問題，
1. 很多序號類型的特徵，這類特徵在test data出現很多train data沒有出現的類別。
2. 是時間序列的資料，建模和做特徵時要注意會不會有leakage和分布相不相同


> EDA(exploratory data analysis):
1. 比較train, validation data, test data各個feature的分佈，看各個feature是否會隨時間不同而改變
->見 function compare_distribution
檢查差異,目測結果:
txkey有一點差異，其他幾乎一樣

2. 比較train, validation data各個feature相對於fraud_ind的關係，看他們與fraud_ind的關係是否會隨時間改變，如果會就不適合做training feature
->見 function analze_distribution
檢查差異，目測結果:
明顯差異,應該去除,或應該抽取更高層的穩定特徵(TODO)
csmcu
mcc
loctm_hr2

無法判讀:應該找出來並想辦法去除或找更高層特徵(TODO)
很多＝＝

3. 比較normal, fraud data的各個feature分佈差異，找有問題的feature!
結果:

4. 但這樣無法看出單一用戶在normal和fraud的關係，所以要另外印出檢查，看有fraud data的用戶，該資料特點在哪
見 ## 以每個bacno來看fraud情況

> Feature engineering:
盡量抽取高級特徵，讓test data的特徵和train data特徵分布相同
抽取時間相關性的特徵，因為這類特徵樹狀模型是學不到的，得人工建立

> Anomaly detection:
因為是盜刷偵測，跟anomaly detection算是蠻相關的，所以我用了一些常用的unsupervised方法做特徵，要記得標準化，都使用scikit，見create_feature.py
1. isolationforest
2. oneclass svm
3. oneclass kmeans(分一類，計算離中心的距離)

> validation:
資料總共是120天，由於如果拿近幾天的資料訓練，拿久遠的資料測試會有資料洩漏的問題，因此只能切(train前/validation後)這樣的切法。
我的方法是切三種，1~60/61~90、1~45/46~90、1~30/31~90，以確保我們訓練的結果夠robust而不是fit到public leaderboard
奇怪的是訓練起來的分數，越少訓練資料反而validation結果越好(1~30天訓練/31~90測試)，推測是近幾天的資料比較雜亂，模型沒有好效果。
此外，validation的分數幾乎都比leaderboard少了0.03左右，估計leaderboard的資料也是屬比較乾淨的。

> catboost:
這個模型default的參數就蠻好的，使用方法官網的 (tutorial)[https://catboost.ai/docs/concepts/tutorials.html] 就講差不多了，且如果用CPU訓練的話能調的參數也不多，因為API接口和其他scikit learn的模型差不多，和xgboost,lgb蠻像的，所以我就隨便調調了，比較特別的是它有cat_feature這個參數。從這篇文章中可以看出這個參數影響模型效果也蠻多的。[https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db]
我還在努力參透xgboost,lgb,catboost的算法差易，希望我能透徹理解什麼時候該用那些模型。

> ensamble:

* 貝葉斯調參


關於工具的學習:
https://xkcd.com/353/

這裡講python套件的學習，python套件很猛，平常想的到會用的運算、演算法、到某個paper的概念幾乎都被包好在某個套件裡，讓我們很幸運地不用一直重新發明一個輪子，而且99%的時間，套件寫同樣演算法的效率會比自己寫的高很多。以資料處理來說，numpy和pandas有很多加快運算的function，像這次比賽資料超過100萬筆，我用for寫一些前處理，跟我用pd.merge寫起來速度大概差了幾十倍，所以說多學著用套件優化後的function肯定好很多。
演算法方面，雖然套件不是萬能的，但以目前的我來說，能夠花一分的力氣很快地玩過各種演算法算是很幸福的事，它讓我能在沒有完全理解演算法的情況下大致的測量每種演算法的效果，在邊用的時候邊學它的原理會比待在教室聽課好玩很多。

學習的方法大致就是打開該套件官網看User Guide或API Reference像挖寶藏一樣的看它有哪些功能，有時候看一看會有 啊!這可以用! 的發現，但如果是完全沒接觸過的套件可以找線上教學，先大概理解這個套件在幹嘛，看有沒有人介紹這套件最常用的是那些功能，再看官網的User Guide。

numpy
https://numpy.org/devdocs/user/quickstart.html
pandas
matplotlib
scikit
catboost,lightgbm


關於前處理:
encoding
http://contrib.scikit-learn.org/categorical-encoding/
transformation:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

scikit
玩玩裡面不同的模型，看他們是幹什麼用的，然後嘗試用進這個project中
https://scikit-learn.org/stable/modules/classes.html#


## 比賽心得:
> 在北京念書，想做點實際的project所以報了這個比賽。
這次做比賽就是希望把平常聽到的做法用出來，從資料的前處理、分析、特徵工程到建模、ensamble。玩玩各種python套件、邊train邊學各種演算法，然後看些blog、kaggle討論或youtuber，是個很有趣的過程。




