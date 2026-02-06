# 花卉识别项目

基于ConvNeXt的花卉分类识别系统

## 项目结构

```
submission/
├── code/
│   ├── model.py          # 模型定义
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 预测脚本
│   ├── utils.py          # 工具函数
│   └── requirements.txt  # 依赖包列表
├── model/
│   ├── convnext_base_flower_recognition.pth  # 模型权重(训练后生成)
│   ├── best_model.pth    # 最佳模型检查点(训练后生成)
│   └── config.json       # 模型配置
└── results/
    └── submission.csv    # 预测结果(预测后生成)
```

## 环境安装

### 方法1: 使用pip安装

```bash
cd submission/code
pip install -r requirements.txt
```

### 方法2: 使用conda创建环境

```bash
# 创建新环境
conda create -n flower_recognition python=3.9

# 激活环境
conda activate flower_recognition

# 安装PyTorch (根据你的CUDA版本选择)
# CPU版本:
conda install pytorch torchvision cpuonly -c pytorch

# GPU版本(CUDA 11.8):
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install pandas Pillow
```

## 环境依赖详细说明

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| torch | >=1.12.0 | 深度学习框架 |
| torchvision | >=0.13.0 | 计算机视觉工具包,包含ConvNeXt模型 |
| pandas | >=1.3.0 | 数据处理,读取CSV标签文件 |
| Pillow | >=9.0.0 | 图像处理 |
| numpy | >=1.21.0 | 数值计算 |

## 使用说明

### 1. 训练模型

```bash
cd submission/code
python train.py
```

训练完成后,模型将保存在 `submission/model/` 目录下。

### 2. 预测

```bash
cd submission
python code/predict.py <测试集目录> <输出文件路径> [可选参数]

# 示例1: 基本使用（推荐）
python code/predict.py ../test ./results/submission.csv

# 示例2: 指定模型路径
python code/predict.py ../test ./results/submission.csv --model_path ./model/best_model.pth

# 示例3: 指定配置文件
python code/predict.py ../test ./results/submission.csv --config_path ./model/config.json
```

参数说明:
- **位置参数**:
  - `test_img_dir`: 测试图片目录（必需）
  - `output_path`: 预测结果输出路径（必需，推荐：`./results/submission.csv`）
  
- **可选参数**:
  - `--model_path`: 模型权重文件路径（默认：`../model/convnext_base_flower_recognition.pth`）
  - `--config_path`: 配置文件路径（默认：`../model/config.json`）
  - `--num_classes`: 类别数量（默认：102，会从config.json自动读取）
  - `--img_size`: 输入图像尺寸（默认：224，会从config.json自动读取）

**输出格式**:
预测结果将保存为CSV文件，包含三列：
- `filename`: 图片文件名
- `category_id`: 预测的类别ID（花卉类别编号）
- `confidence`: 预测置信度（0-1之间）

### 3. 自定义配置

修改 `code/config.py` 文件中的参数来调整训练配置。

## 模型说明

- **基础模型**: ConvNeXt-Base (预训练于ImageNet)
- **迁移学习策略**: 冻结骨干网络,只训练分类器层
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss with Label Smoothing
- **学习率调度**: CosineAnnealingLR
- **数据增强**: 
  - 训练集: RandomResizedCrop, RandomHorizontalFlip
  - 验证集: Resize, CenterCrop

## 注意事项

1. 首次运行会自动重组织数据集结构
2. 确保有足够的磁盘空间(重组织数据会复制文件)
3. 如果使用GPU训练,确保安装了正确的CUDA版本
4. 建议使用GPU进行训练以加速训练过程

## 文件说明

- `model.py`: 包含模型创建和加载函数
- `train.py`: 训练脚本,包含完整的训练流程
- `predict.py`: 预测脚本,支持批量预测
- `utils.py`: 工具函数,包括数据重组织、数据增强等
- `config.py`: 集中管理所有配置参数
