# 训练脚本使用说明

## 快速开始

### 两种使用模式

#### 模式1: 自动划分数据集（推荐新数据）
```bash
# 传入原始图片文件夹和CSV标签文件，自动划分数据集
python train.py train train_labels.csv

# 完整路径示例
python code/submission/code/train.py code/data/train code/data/train_labels.csv
```

#### 模式2: 使用已划分的数据集
```bash
# 如果数据集已经划分好（包含train/val子目录）
python train.py flowerme

# 完整路径示例
python code/submission/code/train.py code/data/flowerme
```

## 工作流程

### 模式1流程（带数据集划分）

当你提供 `labels_file` 参数时：

```
1. 读取原始图片文件夹 + CSV标签文件
   ├── train/              # 原始图片
   │   ├── img001.jpg
   │   ├── img002.jpg
   │   └── ...
   └── train_labels.csv    # 标签文件

2. 自动划分数据集（70% train, 20% test, 10% val）
   保存到: model/dataset/
   ├── train/
   │   ├── category_1/
   │   ├── category_2/
   │   └── ...
   ├── val/
   │   └── ...
   └── test/
       └── ...

3. 使用划分后的数据集训练模型

4. 保存训练结果到 model/runs/
```

### 模式2流程（直接训练）

当你不提供 `labels_file` 参数时：

```
1. 验证数据集目录结构
   flowerme/
   ├── train/
   │   ├── category_1/
   │   ├── category_2/
   │   └── ...
   └── val/
       └── ...

2. 直接训练模型

3. 保存训练结果到 model/runs/
```

## 位置参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `data_dir` | str | ✓ | 原始图片目录 或 已划分的数据集目录 |
| `labels_file` | str | ✗ | CSV标签文件（如果需要自动划分数据集） |

## 可选参数

### 模型参数
```bash
--model yolov8m-cls.pt          # 预训练模型（默认：yolov8m-cls.pt）
--resume path/to/last.pt        # 恢复训练检查点
```

### 训练参数
```bash
--epochs 100                    # 训练轮数（默认：100）
--batch 16                      # 批次大小（默认：16）
--imgsz 224                     # 图像尺寸（默认：224）
--device 0                      # GPU设备（默认：自动）
--workers 8                     # 数据加载线程数（默认：8）
```

### 优化器参数
```bash
--optimizer Adam                # 优化器（默认：Adam）
--lr0 0.001                     # 初始学习率（默认：0.001）
--weight_decay 0.0005           # 权重衰减（默认：0.0005）
--momentum 0.937                # SGD动量（默认：0.937）
```

### 训练控制
```bash
--patience 50                   # 早停耐心值（默认：50）
--seed 0                        # 随机种子（默认：0）
--amp                           # 启用混合精度训练
--plots                         # 生成训练图表
```

### 输出控制
```bash
--project model                 # 项目保存目录（默认：model）
--name runs                     # 运行名称（默认：runs）
```

### 数据增强参数（用于CustomizedDataset）
```bash
--hsv_h 0.015                   # HSV色调增强（默认：0.015）
--hsv_s 0.7                     # HSV饱和度增强（默认：0.7）
--hsv_v 0.4                     # HSV明度增强（默认：0.4）
--fliplr 0.5                    # 水平翻转概率（默认：0.5）
--flipud 0.0                    # 垂直翻转概率（默认：0.0）
--erasing 0.0                   # 随机擦除概率（默认：0.0）
```

## 使用示例

### 示例1: 自动划分并训练（原始数据）
```bash
# 最常用：传入原始图片和标签文件
python train.py train train_labels.csv

# 数据集会自动划分到 model/dataset/
# 训练结果保存到 model/runs/
```

### 示例2: 使用已划分数据集
```bash
# 如果已经有划分好的数据集
python train.py flowerme
```

### 示例3: 自定义训练参数
```bash
python train.py train train_labels.csv \
  --epochs 50 \
  --batch 32 \
  --imgsz 256 \
  --device 0
```

### 示例4: GPU + 混合精度 + 数据增强
```bash
python train.py train train_labels.csv \
  --device 0 \
  --amp \
  --erasing 0.2 \
  --hsv_h 0.02 \
  --fliplr 0.5
```

### 示例5: 完整配置示例
```bash
python train.py train train_labels.csv \
  --model yolov8m-cls.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 224 \
  --optimizer Adam \
  --lr0 0.001 \
  --device 0 \
  --amp \
  --plots \
  --patience 50 \
  --hsv_h 0.015 \
  --hsv_s 0.7 \
  --fliplr 0.5 \
  --erasing 0.2
```

### 示例6: 从检查点恢复训练
```bash
python train.py flowerme --resume model/runs/weights/last.pt
```

### 示例7: Docker容器中使用
```bash
# 自动划分数据集
python /workspace/code/submission/code/train.py \
  /workspace/code/data/train \
  /workspace/code/data/train_labels.csv

# 使用已划分数据集
python /workspace/code/submission/code/train.py \
  /workspace/code/data/flowerme
```

## CSV文件格式要求

如果使用自动划分模式，CSV文件必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `filename` | str | 图片文件名（如 img001.jpg） |
| `category_id` | int | 类别ID |
| `chinese_name` | str | 中文类别名（可选） |
| `english_name` | str | 英文类别名（可选） |

**示例CSV内容:**
```csv
filename,category_id,chinese_name,english_name
img001.jpg,1,玫瑰,rose
img002.jpg,2,郁金香,tulip
img003.jpg,1,玫瑰,rose
```

## 输出文件结构

### 模式1（带数据集划分）
```
model/
├── dataset/                 # 自动划分的数据集
│   ├── train/
│   │   ├── 1/              # 类别目录（使用category_id）
│   │   ├── 2/
│   │   └── ...
│   ├── val/
│   └── test/
├── runs/                    # 训练结果
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.csv
│   └── ...
└── config.json             # 训练配置
```

### 模式2（直接训练）
```
model/
├── runs/                    # 训练结果
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.csv
│   └── ...
└── config.json             # 训练配置
```

## 数据增强说明

训练时自动使用 `CustomizedDataset` 类，包含以下增强：

**训练集增强:**
1. Resize (224x224)
2. RandomHorizontalFlip (p=0.5)
3. RandomVerticalFlip (p=0.0)
4. RandAugment (num_ops=2, magnitude=9)
5. ColorJitter (使用 hsv_h, hsv_s, hsv_v)
6. ToTensor
7. Normalize (ImageNet标准)
8. RandomErasing (p=0.0, 可调整)

**验证集增强:**
1. Resize (224x224)
2. ToTensor
3. Normalize (ImageNet标准)

## 数据集划分说明

### 划分比例（默认）
- **训练集 (train)**: 70%
- **测试集 (test)**: 20%
- **验证集 (val)**: 10%

### 类别目录命名
使用 `category_id` 作为类别目录名，例如：
```
dataset/train/1/     # category_id = 1
dataset/train/2/     # category_id = 2
```

### 划分特点
- 按类别分层划分，保证每个类别的比例相同
- 使用随机种子（可通过 `--seed` 参数控制）确保可重复性
- 自动处理文件复制，不修改原始数据

## 常见问题

### Q1: 如何只划分数据集不训练？
A: 使用 `model.py` 的 split 命令：
```bash
python model.py split --csv train_labels.csv --images_dir train --output_dir flowerme
```

### Q2: 数据集划分保存在哪里？
A: 默认保存在 `model/dataset/` 目录下

### Q3: 可以自定义划分比例吗？
A: 目前固定为 70:20:10，如需自定义请直接使用 `model.py split` 命令

### Q4: CSV文件缺失某些图片怎么办？
A: 程序会自动跳过缺失的文件，并在最后统计中显示缺失数量

### Q5: 如何使用自己已经划分好的数据集？
A: 不传递 `labels_file` 参数，直接传入数据集目录：
```bash
python train.py my_dataset
```

## 最佳实践

### 第一次训练（原始数据）
```bash
# 1. 准备数据
#    - train/ 文件夹包含所有原始图片
#    - train_labels.csv 包含标签信息

# 2. 运行训练（自动划分+训练）
python train.py train train_labels.csv --epochs 50 --device 0 --amp

# 3. 检查输出
#    - model/dataset/  # 划分后的数据集
#    - model/runs/     # 训练结果
#    - model/config.json  # 配置文件
```

### 继续训练或调参
```bash
# 使用已经划分好的数据集
python train.py model/dataset \
  --resume model/runs/weights/last.pt \
  --epochs 100
```

### 生产环境部署
```bash
# 使用最佳模型
best_model = "model/runs/weights/best.pt"
```

## 参数映射（与CustomizedDataset匹配）

| 命令行参数 | CustomizedDataset使用 | 说明 |
|-----------|----------------------|------|
| `--imgsz` | `args.imgsz` | 图像尺寸 |
| `--fliplr` | `args.fliplr` | 水平翻转概率 |
| `--flipud` | `args.flipud` | 垂直翻转概率 |
| `--hsv_h` | `args.hsv_h` | ColorJitter的hue参数 |
| `--hsv_s` | `args.hsv_s` | ColorJitter的saturation参数 |
| `--hsv_v` | `args.hsv_v` | ColorJitter的brightness/contrast参数 |
| `--erasing` | `args.erasing` | RandomErasing概率 |

## 注意事项

1. **数据集结构**: 必须包含 `train/` 和 `val/` 子目录
2. **路径处理**: 支持相对路径和绝对路径
3. **自动保存**: 配置自动保存到 `{project}/config.json`
4. **自定义增强**: 始终使用 `CustomizedTrainer` 和 `CustomizedDataset`

## 帮助信息

```bash
python train.py --help
```

### 模型相关
```bash
--model yolov8m-cls.pt          # 预训练模型路径（默认：yolov8m-cls.pt）
--resume path/to/checkpoint.pt  # 从检查点恢复训练
```

### 训练参数
```bash
--epochs 100                    # 训练轮数（默认：100）
--batch 16                      # 批次大小（默认：16）
--imgsz 224                     # 图像尺寸（默认：224）
--workers 8                     # 数据加载线程数（默认：8）
```

### 优化器参数
```bash
--optimizer Adam                # 优化器选择（默认：Adam）
--lr0 0.001                     # 初始学习率（默认：0.001）
--lrf 0.01                      # 最终学习率因子（默认：0.01）
--momentum 0.937                # SGD动量（默认：0.937）
--weight_decay 0.0005           # 权重衰减（默认：0.0005）
--warmup_epochs 3.0             # 预热epochs（默认：3.0）
```

### 训练控制
```bash
--patience 50                   # 早停耐心值（默认：50）
--val                           # 启用验证（默认：True）
--fraction 1.0                  # 使用数据集比例（默认：1.0，即100%）
```

### 设备和性能
```bash
--device 0                      # GPU设备ID（默认：自动）
--device 0,1                    # 多GPU训练
--device cpu                    # CPU训练
--amp                           # 启用混合精度训练
--deterministic                 # 使用确定性算法（默认：True）
--seed 0                        # 随机种子（默认：0）
```

### 输出控制
```bash
--project model                 # 项目目录（默认：model）
--name runs                     # 运行名称（默认：runs）
--exist_ok                      # 允许覆盖现有目录（默认：True）
--plots                         # 生成训练图表
--verbose                       # 详细输出（默认：True）
```

### 数据增强
```bash
--hsv_h 0.015                   # HSV色调增强（默认：0.015）
--hsv_s 0.7                     # HSV饱和度增强（默认：0.7）
--hsv_v 0.4                     # HSV明度增强（默认：0.4）
--fliplr 0.5                    # 水平翻转概率（默认：0.5）
--flipud 0.0                    # 垂直翻转概率（默认：0.0）
--erasing 0.0                   # 随机擦除概率（默认：0.0）
--multi_scale                   # 启用多尺度训练
--rect                          # 启用矩形训练
```

### 高级参数
```bash
--pretrained                    # 使用预训练权重（默认：True）
--freeze 10                     # 冻结前N层
--cos_lr                        # 使用余弦学习率调度器
--close_mosaic 10               # 最后N个epochs关闭mosaic增强
--save_period 10                # 每N个epochs保存检查点（-1禁用）
--cache ram                     # 缓存数据到内存
--cache disk                    # 缓存数据到磁盘
```

## 完整示例

### 示例1: 快速开始（默认配置）
```bash
python train.py flowerme
```

### 示例2: 自定义基本参数
```bash
python train.py flowerme --epochs 50 --batch 32 --imgsz 256
```

### 示例3: GPU训练 + 混合精度
```bash
python train.py flowerme --device 0 --amp --batch 64
```

### 示例4: 多GPU训练
```bash
python train.py flowerme --device 0,1,2,3 --batch 128
```

### 示例5: 启用数据增强
```bash
python train.py flowerme \
  --hsv_h 0.02 \
  --hsv_s 0.8 \
  --hsv_v 0.5 \
  --fliplr 0.5 \
  --flipud 0.1 \
  --erasing 0.2 \
  --multi_scale
```

### 示例6: 完整配置
```bash
python train.py flowerme \
  --model yolov8m-cls.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 224 \
  --optimizer Adam \
  --lr0 0.001 \
  --weight_decay 0.0005 \
  --patience 50 \
  --device 0 \
  --amp \
  --plots \
  --project model \
  --name runs \
  --hsv_h 0.015 \
  --hsv_s 0.7 \
  --hsv_v 0.4 \
  --fliplr 0.5 \
  --erasing 0.2
```

### 示例7: 从检查点恢复训练
```bash
python train.py flowerme --resume model/runs/weights/last.pt
```

### 示例8: 使用较小数据集进行实验
```bash
python train.py flowerme --fraction 0.1 --epochs 10
```

### 示例9: 迁移学习（冻结骨干网络）
```bash
python train.py flowerme --freeze 10 --epochs 50
```

### 示例10: Docker容器中训练
```bash
# 在容器内
python /workspace/code/submission/code/train.py /workspace/code/data/train

# 或使用相对路径
cd /workspace/code/submission/code
python train.py ../../data/train
```

## 输出文件

训练完成后，以下文件将保存在 `{project}/{name}/` 目录下：

```
model/runs/
├── weights/
│   ├── best.pt              # 最佳模型权重
│   └── last.pt              # 最后一个epoch的权重
├── results.csv              # 训练指标记录
├── args.yaml                # 训练参数配置
├── confusion_matrix.png     # 混淆矩阵（如果启用--plots）
├── results.png              # 训练曲线（如果启用--plots）
└── ...
config.json                  # 完整配置文件
```

## 参数对比表（旧版 vs 新版）

| 旧参数名 | 新参数名 | 变化 |
|---------|---------|------|
| `--data_dir` | `data_dir` (位置参数) | 改为位置参数 |
| `--save_dir` | `--project` | 重命名 |
| `--model_path` | `--model` | 重命名 |
| `--img_size` | `--imgsz` | 重命名 |
| `--batch_size` | `--batch` | 重命名 |
| `--lr` | `--lr0` | 重命名 |
| `--num_workers` | `--workers` | 重命名 |
| N/A | `--lrf` | 新增 |
| N/A | `--momentum` | 新增 |
| N/A | `--warmup_epochs` | 新增 |
| N/A | `--name` | 新增 |
| N/A | `--exist_ok` | 新增 |
| N/A | `--save_period` | 新增 |
| N/A | `--cache` | 新增 |
| N/A | `--fraction` | 新增 |
| N/A | `--freeze` | 新增 |
| N/A | `--cos_lr` | 新增 |
| N/A | `--close_mosaic` | 新增 |
| N/A | `--multi_scale` | 新增 |
| N/A | `--rect` | 新增 |

## 兼容YOLO标准

本训练脚本的参数设计完全兼容 Ultralytics YOLO 的标准参数，可以直接使用 YOLO 官方文档中的参数配置。

## 注意事项

1. **位置参数**: `data_dir` 必须是第一个参数，`labels_file` 是可选的第二个参数
2. **路径处理**: 支持相对路径和绝对路径
3. **默认值**: 所有可选参数都有合理的默认值，可以不指定
4. **自定义数据增强**: 训练时会自动使用 `CustomizedDataset` 类（RandAugment + ColorJitter + RandomErasing）
5. **配置保存**: 训练配置会自动保存到 `{project}/config.json`

## 帮助信息

查看所有可用参数：
```bash
python train.py --help
```
