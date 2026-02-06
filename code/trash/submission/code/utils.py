import os
import json
import shutil
import random
import pandas as pd
from collections import defaultdict
from PIL import ImageFile
import torch

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AverageMeter:
    """计算并存储平均值和当前值"""
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


def save_config(config, filepath):
    """保存配置文件为JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"配置已保存到: {filepath}")

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reorganize_dataset_by_label(src_dir, dst_dir, labels_file, val_split=0.2):
    """
    根据标签文件重新组织数据集结构
    
    参数:
    src_dir: 原始图像文件所在目录
    dst_dir: 重新组织后的目录
    labels_file: 包含文件名和类别信息的CSV文件路径
    val_split: 验证集占比
    """
    # 创建目标文件夹
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 读取标签文件
    labels_df = pd.read_csv(labels_file)
    print(f"总共找到 {len(labels_df)} 张图片")
    
    # 按类别分组
    class_groups = defaultdict(list)
    for _, row in labels_df.iterrows():
        filename = row['filename']
        # 使用category_id作为类别名称
        class_name = str(row['category_id'])  # 转换为字符串以用作文件夹名称
        class_groups[class_name].append(filename)
    
    print(f"总共 {len(class_groups)} 个类别")
    
    # 创建类别文件夹并分配文件
    for class_name, files in class_groups.items():
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # 随机分割训练集和验证集
        random.shuffle(files)
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # 复制训练文件
        for file in train_files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(train_dir, class_name, file)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"复制文件 {file} 时出错: {e}")
        
        # 复制验证文件
        for file in val_files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(val_dir, class_name, file)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"复制文件 {file} 时出错: {e}")
    
    print(f"数据集重新组织完成")
    return len(class_groups)


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
