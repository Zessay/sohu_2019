## <font size=4>目录说明</font>

- data：保存原始数据、中间生成数据以及实体字典

  + gen：用于保存中间的生成文件，比如经过特征工程之后的训练集和测试集；
  + **nerdict**：实体字典，包括官方实体字典，搜狗、百度输入法词库、爬虫获取的明星词条以及流行的手机和电视节目；

  

- models：用来保存训练得到的模型

  + machine：用来保存`lightgbm, xgboost, catboost`训练得到的预测模型
  + word：用来保存`word2vec, doc2vec, TfIdf, LDA, KMeans `等基于单词得到的中间模型



- process：对训练集和测试集进行预处理的代码
  + `generate_all_tokens.py`：对原始训练集和测试集进行预处理，得到`all_tokens.csv`，包含`newsId, title, content, tokens_with_sw, tokens_without_sw`等列
  + `get_features.py`：基于`all_tokens.csv`进行特征工程，得到`all_train.csv, all_test.csv`
  + `postprocess.py`：对生成的训练集和测试集数据进行一些后处理，包括添加`idf`特征以及平滑之后的转化率`ctr`相关特征
  + `train_models.py`：用于训练并保存`word2vec, doc2vec, tfidf, kmeans, lda`等模型



- result：保存用于提交的结果文件



- stopwords：分词和处理过程需要使用的停止词
  + `simple_stopwords.txt`：常用的中文停止词和符号
  + `post_stopwords.txt`：预测过程中对结果干扰比较大的单词



- train：3种不同的用于训练的模型
  + `cat_train.py`：使用caboost模型训练并得到结果文件
  + `lgb_train.py`：使用lightgbm模型训练并得到结果文件
  + `xgb_train.py`：使用xgboost模型训练并得到结果文件



- utils：一些工具函数
  + `smooth.py`：包含了贝叶斯平滑函数，用于对单词到实体的转化率进行平滑
  + `subsave.py`：包含加载源文件函数，按顺序保存提交文件等函数
  + `threshold.py`：用于搜索最佳分割阈值以及基于搜索结果获取测试集实体的函数



- `config.py`：用于设置源文件以及生成文件的保存路径
- `main.py`：主函数，可以直接通过`python main.py`运行

## <font size=4>运行说明</font>

- 将原始文件`coreEntityEmotion_train.txt`和`coreEntityEmotion_test_stage2.txt`放入`data`文件夹中
- Linux下直接命令行运行`sh run.sh`
- Windows下确保安装git，在git bash下运行`sh run.sh`