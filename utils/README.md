> **smooth.py**：包含了贝叶斯平滑函数，用于对单词到实体的转化率进行平滑
>
> - **`BayesianSmoothing`类**：定义了贝叶斯平滑过程
> - `bys` 函数：用于调用的贝叶斯平滑函数，主要包含3个参数



> **subsave.py**：用于加载源文件以及按顺序保存提交文件
>
> - `save_as_order` 函数：防止生成的文件乱序，按顺序保存文件
> - `loadData` 函数：用于加载原始训练集和测试集文件
> - `read_sub_file` 函数：加载生成的提交文件
> - `extract_ent_emo` 函数：从train_data中提取id、实体以及情感



> **threshold.py**：用于获取最佳分割阈值以及对获取核心实体
>
> - `find_threshold` 函数：根据验证集的预测结果，搜索最佳分割阈值
> - `return_entity`  函数：根据搜索得到的阈值，返回对应的实体