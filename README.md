# TET-Pipeline

> MER-Baseline中修改分离出的文本单模态情感识别模块

### 仓库结构

```
.
├── config_path.py # 原始数据处理所用的路径
├── config.py # # 处理后数据集路径
├── extract_text_dataset.py # 原始数据集ASR，Text单模态数据集构造
├── extract_text_embedding_LZ.py # 从Text数据集提取embedding并存储
├── train.py # 分类网络和模型构建及训练
├── README.md
├── dataset # gitignored: [Text单模态数据集]
├── features # gitignored: [embedding特征]
├── saved-unimodal # gitignored: [训练完成网络]
└── tools # gitignored: [预训练模型及工具库]
```



## 目前数据集已测试预训练模型：

### macbert-large：

run：

```shell
# 提取embedding
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
```

```shell
# 文本单模态predict网络
python -u train.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
```

paper_result：

![net2](https://github.com/WarmCongee/TER-Pipeline/raw/master/img/MacBERT_paper_result.png)

result：

```shell
# train_args_path为该次训练超参数保存path
fscore: 0.4377, valmse: 2.4225, metric: -0.1678,  
train_args_path: ./saved-unimodal/model/cv_features:chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT_f1:0.4378_valmse:2.4225_metric:-0.1679_1683700986.6242754.npz

fscore: 0.4159, valmse: 2.4419, metric: -0.1945, 
train_args_path: ./saved-unimodal/model/cv_features:chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT_f1:0.4159_valmse:2.4419_metric:-0.1945_1683701321.3115225.npz

fscore: 0.4261, valmse: 2.3998, metric: -0.1739, 
train_args_path: ./saved-unimodal/model/cv_features:chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT+chinese-macbert-large-4-UTT_f1:0.4261_valmse:2.3998_metric:-0.1739_1683701999.5011146.npz
```



### roberta-large：

run：

```shell
# 提取embedding
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large' --gpu=0
```

```shell
# 文本单模态predict网络
python -u train.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
```

paper_result：

![RoBERTa](https://github.com/WarmCongee/TER-Pipeline/raw/master/img/RoBERTa_paper_result.png)

result：

```shell
# train_args_path为该次训练超参数保存path
fscore: 0.4245, valmse: 2.3955, metric: -0.1744, 
train_args_path: ./saved-unimodal/model/cv_features:chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT_f1:0.4245_valmse:2.3955_metric:-0.1744_1683703252.0435195.npz

fscore: 0.4247, valmse: 2.3798, metric: -0.1703, 
train_args_path: ./saved-unimodal/model/cv_features:chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT_f1:0.4247_valmse:2.3798_metric:-0.1703_1683703431.8806338.npz

fscore: 0.4237, valmse: 2.3435, metric: -0.1622, 
train_args_path: ./saved-unimodal/model/cv_features:chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT+chinese-roberta-wwm-ext-large-4-UTT_f1:0.4237_valmse:2.3435_metric:-0.1622_1683703848.208033.npz

```



### Code Reference: 

[https: //github.com/zeroQiaoba/MER2023-Baseline](https://github.com/zeroQiaoba/MER2023-Baseline)