# 10折交叉验证训练说明

## 修改概述

已将训练脚本修改为使用10折交叉验证，每个epoch随机选择2折作为验证集，8折作为训练集。

## 主要修改

### 1. utils.py 新增功能

#### `create_kfold_splits()` 函数
- 创建10折交叉验证数据划分
- 确保每个类别在各折中均匀分布
- 返回所有折的CSV文件路径和类别映射信息

参数:
- `src_dir`: 原始图像目录
- `dst_dir`: CSV文件保存目录
- `labels_file`: 标签CSV文件
- `n_folds`: 折数（默认10）
- `seed`: 随机种子
- `verify_images`: 是否验证图片有效性

返回:
```python
{
    'fold_csvs': [fold_0.csv, fold_1.csv, ..., fold_9.csv],
    'num_classes': 类别数,
    'class_id_to_idx': 类别ID到索引的映射,
    'idx_to_class_id': 索引到类别ID的映射,
    'class_distribution': 各类别样本数统计,
    'fold_sizes': 各折的样本数列表,
    'n_folds': 10
}
```

#### `KFoldFlowerDataset` 类
- 支持从多个CSV文件加载数据的Dataset类
- 用于合并多个折的数据

### 2. train.py 新增/修改功能

#### `train_model_with_kfold()` 函数
新增的训练函数，支持K折交叉验证：

特点:
- 每个epoch开始时随机选择验证折
- 使用不同的随机种子（seed + epoch）保证可重现性
- 训练集和验证集分别应用对应的数据增强
- 记录每个epoch的折划分情况

参数:
- `fold_csvs`: 所有折的CSV文件路径列表
- `n_folds`: 总折数（默认10）
- `n_val_folds`: 每个epoch用作验证的折数（默认2）
- 其他标准训练参数...

#### main() 函数修改
- 使用 `create_kfold_splits()` 替代原来的 `reorganize_dataset_by_label()`
- 调用 `train_model_with_kfold()` 进行训练

## 工作原理

### 数据划分
1. 读取所有标签数据
2. 按类别分组
3. 每个类别的样本均匀分配到10折中
4. 保存10个CSV文件（fold_0.csv 到 fold_9.csv）

### 训练过程
每个epoch:
1. 随机选择2折（索引）作为验证集
2. 剩余8折作为训练集
3. 创建对应的DataLoader
   - 训练集：RandomResizedCrop + RandomHorizontalFlip
   - 验证集：Resize + CenterCrop
4. 正常训练和验证

### 示例
- Epoch 1: 训练折[0,1,2,3,4,5,6,7], 验证折[8,9]
- Epoch 2: 训练折[1,2,3,4,5,6,8,9], 验证折[0,7]
- Epoch 3: 训练折[0,2,3,4,5,7,8,9], 验证折[1,6]
- ...

## 优势

1. **更充分的数据利用**: 每个样本都有机会被用于训练和验证
2. **更好的泛化能力**: 模型在不同的数据子集上进行验证
3. **减少过拟合风险**: 每个epoch的验证集都不同
4. **保持类别均衡**: 每折中各类别比例一致

## 使用方法

训练命令不变，直接运行:
```bash
python code/train.py --data_dir train --labels_file train_labels.csv --epochs 50
```

训练过程会输出每个epoch使用的折信息:
```
Epoch 1/50
  训练折: [0, 1, 2, 3, 4, 5, 6, 7] (8 折)
  验证折: [8, 9] (2 折)
  训练集大小: 8000
  验证集大小: 2000
```

## 保存的文件

在 `model/` 目录下会生成:
- `fold_0.csv` ~ `fold_9.csv`: 10折数据划分
- `class_mapping.json`: 类别映射信息（包含n_folds字段）
- `convnext_large_flower_plus_recognition.pth`: 最终模型权重
- `best_model.pth`: 检查点（包含fold_splits历史记录）
- `config.json`: 训练配置

## 注意事项

1. **随机性**: 使用seed + epoch确保可重现性
2. **内存**: 每个epoch重新创建DataLoader，内存占用略有增加
3. **时间**: 由于每个epoch都重新加载数据，初始化时间略有增加（但训练时间基本不变）
4. **兼容性**: 保留了原来的 `train_model()` 函数以保持向后兼容

## 测试

可以运行测试脚本验证数据划分:
```bash
python test_kfold.py
```

该脚本会检查:
- 各折样本数
- 类别分布均匀性
- 数据完整性

