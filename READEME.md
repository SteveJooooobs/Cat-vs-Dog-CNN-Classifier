# Cat vs Dog CNN Classifier

从零搭建的卷积神经网络（CNN），在 Kaggle Dogs vs Cats 数据集上达到 **96.6% 测试准确率**（V3 版本）。

## 版本性能对比

| 指标 | V1/V2 | V3（当前） |
| :--- | :--- | :--- |
| 验证集最佳准确率 | 91.9% | **95.88%** |
| 测试集准确率 | 92.0% | **96.6%** |
| 测试集损失 | 0.2004 | **0.0904** |
| 训练集最终损失 | - | 0.1034 |
| 验证集最终损失 | - | 0.1095 |

### V2 → V3 关键变更

| 参数              | V1  | V3 |
| :---             | :---| :--- |
| 训练轮次          | 60  | **80** |
| Dropout          | 0.0 | **0.1** |
| StepLR step_size | 15  | **25** |
| StepLR gamma    | 0.75 | **0.85** |

## 项目亮点
- 纯手写 PyTorch 模型，未使用任何预训练权重
- 完整的数据增强策略（随机裁剪、翻转、旋转、颜色抖动）
- BatchNorm + Dropout 分阶段调优（V2 关闭→V3 适度回调）
- 学习率调度器精细打磨
- 训练过程可复现，代码结构清晰

## 项目结构
```
CNN_cat_vs_dog/
├── data(需自行创建数据源存储的目录)/
│   
├── src/
│   ├── data/clean_data.py      # 缩减版数据加载、增强、DataLoader
│   ├── data/clean_data_2.py    # 完整版数据加载、动态划分、DataLoader（推荐）
│   ├── train_script.py          # 模型定义、训练循环
│   └── test/test.py             # 测试脚本
├── models/
│   │
│   └── best_catdog_model-v3.pth  # V3 权重（当前最新）
├── requirements.txt
└── README.md
```

## 环境依赖
- Python 3.10+
- PyTorch 2.x
- torchvision
- PIL (Pillow)

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
0. 在kaggle下载的数据如果是只做猫狗分开，不分训练集等，直接解压到row_complate中即可
1. 从 [魔塔社区](https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class) 下载 Dogs vs Cats 数据集
2. 解压后将数据文件夹放入 `data/raw/` 目录（缩减版）或 `data/raw_complate/` 目录（完整版）
3. 按猫/狗分好子文件夹：
```
data/raw/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

## 训练
```bash
cd src
python train_script.py
```

训练完成后，最佳模型保存在 `models/best_catdog_model-v3.pth`。

## 测试
```bash
cd src/test
python test.py
```

## 最终结果（V3）
| 指标 | 数值 |
|------|------|
| 验证集最佳准确率 | **95.88%** |
| 测试集准确率 | **96.6%** |
| 测试集损失 | 0.0904 |
| 训练集最终损失 | 0.1034 |
| 验证集最终损失 | 0.1095 |

## 模型架构
- 4 层卷积 + BatchNorm + MaxPool
- 自适应平均池化 (7×7)
- 2 层全连接 + Dropout(0.1)
- 总参数量：~5.4M


## 作者
[Mjolnir / GitHub:SteveJooooobs]

### 致谢
- 本项目调优过程得到 [DeepSeek](https://chat.deepseek.com/) 的全程技术支持
