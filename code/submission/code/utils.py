import os
import json
import shutil
import random
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, delta=0, verbose=True):
        """
        参数:
        patience: 多少个epoch没有改善后停止训练
        delta: 最小改善幅度
        verbose: 是否打印信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.0
    
    def __call__(self, val_acc, model=None):
        """
        检查是否需要早停
        参数:
        val_acc: 当前验证集准确率
        model: 模型(可选,用于保存最佳模型)
        """
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.val_acc_max = val_acc
            if self.verbose:
                print(f'初始最佳验证准确率: {self.val_acc_max:.2f}%')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_acc_max = val_acc
            if self.verbose:
                print(f'验证准确率提升: {self.val_acc_max:.2f}%')
            self.counter = 0


# 计算并存储平均值和当前值
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 保存配置文件为json格式
def save_config(config, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"配置已保存到: {filepath}")

# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reorganize_dataset_by_label(src_dir, dst_dir, labels_file, val_split=0.2, seed=42, verify_images=True):
    """
    根据标签文件重新组织数据集结构，验证图片有效性，并返回划分后的CSV路径
    自动统计类别数并建立原始类别ID到模型输出索引的映射
    
    参数:
    src_dir: 原始图像文件所在目录
    dst_dir: 重新组织后的目录（CSV文件将保存在此目录）
    labels_file: 包含文件名和类别信息的CSV文件路径
    val_split: 验证集占比（0.0-1.0）
    seed: 随机种子，确保可重现
    verify_images: 是否验证图片有效性（跳过损坏/缺失的图片）
    
    返回:
    dict: {
        'train_csv': 训练集CSV路径,
        'val_csv': 验证集CSV路径,
        'num_classes': 类别数,
        'class_id_to_idx': 类别ID到索引的映射字典,
        'idx_to_class_id': 索引到类别ID的映射字典,
        'train_size': 训练集大小,
        'val_size': 验证集大小,
        'class_distribution': 每个类别的样本数统计
    }
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建目标文件夹
    os.makedirs(dst_dir, exist_ok=True)
    
    # 读取标签文件
    df = pd.read_csv(labels_file)
    print(f"总共找到 {len(df)} 张图片")
    
    # 验证图片有效性
    if verify_images:
        valid_rows = []
        for idx, row in df.iterrows():
            img_path = os.path.join(src_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img.verify()  # 验证图片完整性
                    valid_rows.append(row)
                except Exception as e:
                    print(f"损坏图片已跳过: {img_path} ({e})")
            else:
                print(f"缺失图片已跳过: {img_path}")
        valid_df = pd.DataFrame(valid_rows)
        print(f"有效图片数: {len(valid_df)}")
    else:
        valid_df = df
    
    # 统计所有类别并建立映射(按类别ID排序以保证一致性)
    unique_class_ids = sorted(valid_df['category_id'].unique())
    # 转换为Python原生int类型以支持JSON序列化
    class_id_to_idx = {int(class_id): idx for idx, class_id in enumerate(unique_class_ids)}
    idx_to_class_id = {idx: int(class_id) for class_id, idx in class_id_to_idx.items()}
    num_classes = len(unique_class_ids)
    
    print(f"总共 {num_classes} 个类别")
    print(f"类别ID范围: {min(unique_class_ids)} - {max(unique_class_ids)}")
    print(f"映射示例: {dict(list(class_id_to_idx.items())[:5])}...")
    
    # 统计每个类别的样本数(转换为Python原生int类型)
    class_distribution = {int(k): int(v) for k, v in valid_df['category_id'].value_counts().to_dict().items()}
    
    # 按类别分组,确保每个类别都按相同比例随机划分
    train_rows = []
    val_rows = []
    
    print("\n按类别划分数据集:")
    for cat_id in unique_class_ids:
        group = valid_df[valid_df['category_id'] == cat_id].copy()
        group = group.sample(frac=1, random_state=seed)  # 随机打乱
        
        n_samples = len(group)
        n_train = int(n_samples * (1 - val_split))
        
        # 确保至少有一个样本在训练集（如果该类别有样本）
        if n_train == 0 and n_samples > 0:
            n_train = 1
        
        train_group = group.iloc[:n_train]
        val_group = group.iloc[n_train:]
        
        if len(train_group) > 0:
            train_rows.append(train_group)
        if len(val_group) > 0:
            val_rows.append(val_group)
        
        print(f"  类别 {cat_id}: 总数={n_samples}, 训练={len(train_group)}, 验证={len(val_group)}")
    
    train_df = pd.concat(train_rows).reset_index(drop=True) if train_rows else pd.DataFrame()
    val_df = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame()
    
    print(f"\n训练集总计: {len(train_df)} 张图片")
    print(f"验证集总计: {len(val_df)} 张图片")
    print(f"划分比例: {len(train_df)/(len(train_df)+len(val_df))*100:.1f}% / {len(val_df)/(len(train_df)+len(val_df))*100:.1f}%")
    
    # 保存为CSV
    train_csv = os.path.join(dst_dir, 'train_split.csv')
    val_csv = os.path.join(dst_dir, 'val_split.csv')
    mapping_json = os.path.join(dst_dir, 'class_mapping.json')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # 保存类别映射
    mapping_data = {
        'class_id_to_idx': class_id_to_idx,
        'idx_to_class_id': idx_to_class_id,
        'num_classes': num_classes,
        'class_distribution': class_distribution
    }
    with open(mapping_json, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练集CSV已保存: {train_csv}")
    print(f"验证集CSV已保存: {val_csv}")
    print(f"类别映射已保存: {mapping_json}")
    
    return {
        'train_csv': train_csv,
        'val_csv': val_csv,
        'num_classes': num_classes,
        'class_id_to_idx': class_id_to_idx,
        'idx_to_class_id': idx_to_class_id,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'class_distribution': class_distribution
    }

class FlowerCSVDataset(Dataset):
    """
    基于CSV文件的花卉数据集加载器
    
    参数:
    csv_file: CSV文件路径，包含 filename 和 category_id 列
    img_dir: 图像文件所在目录
    transform: 数据预处理/增强的transforms
    class_id_to_idx: 类别ID到索引的映射字典（可选）
    """
    def __init__(self, csv_file, img_dir, transform=None, class_id_to_idx=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_id_to_idx = class_id_to_idx
        
        # 如果没有提供映射，自动创建
        if self.class_id_to_idx is None:
            unique_ids = sorted(self.data['category_id'].unique())
            self.class_id_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['category_id'])
        
        # 将类别编号转换为索引
        if self.class_id_to_idx:
            label = self.class_id_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms():
    """获取数据预处理和增强的transforms"""
    from torchvision import transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms
