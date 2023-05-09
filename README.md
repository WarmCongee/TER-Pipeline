# TET-Pipeline

> MER-Baseline的文本单模态情感识别模块

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



### 目前已测试预训练模型：

macbert-large：