#!/usr/bin/env python3
"""
测试10折交叉验证功能
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from utils import create_kfold_splits
import pandas as pd

def test_kfold_splits():
    """测试10折交叉验证数据划分"""
    print("=" * 60)
    print("测试10折交叉验证数据划分")
    print("=" * 60)
    
    # 假设的参数（请根据实际情况修改）
    data_dir = 'train'
    labels_file = 'train_labels.csv'
    model_dir = 'model'
    
    # 检查文件是否存在
    if not os.path.exists(labels_file):
        print(f"错误: 标签文件不存在: {labels_file}")
        return
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 创建10折划分
    print("\n创建10折交叉验证划分...")
    kfold_info = create_kfold_splits(
        src_dir=data_dir,
        dst_dir=model_dir,
        labels_file=labels_file,
        n_folds=10,
        seed=42,
        verify_images=False  # 为了测试速度，暂不验证图片
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("划分结果统计:")
    print("=" * 60)
    print(f"类别数: {kfold_info['num_classes']}")
    print(f"总折数: {kfold_info['n_folds']}")
    print(f"总样本数: {sum(kfold_info['fold_sizes'])}")
    print(f"\n各折样本数:")
    for i, size in enumerate(kfold_info['fold_sizes']):
        print(f"  第 {i} 折: {size} 样本")
    
    # 验证每个类别在各折中的分布
    print("\n" + "=" * 60)
    print("验证各折中类别分布均匀性:")
    print("=" * 60)
    
    fold_class_counts = []
    for i, csv_file in enumerate(kfold_info['fold_csvs']):
        df = pd.read_csv(csv_file)
        class_counts = df['category_id'].value_counts().to_dict()
        fold_class_counts.append(class_counts)
        print(f"\n第 {i} 折包含 {len(class_counts)} 个类别")
    
    # 检查是否所有类别都出现在各折中
    all_classes = set()
    for counts in fold_class_counts:
        all_classes.update(counts.keys())
    
    print(f"\n总共 {len(all_classes)} 个类别")
    
    # 抽样检查几个类别的分布
    sample_classes = sorted(list(all_classes))[:5]
    print(f"\n抽样检查前5个类别在各折中的分布:")
    for class_id in sample_classes:
        counts = [fold_counts.get(class_id, 0) for fold_counts in fold_class_counts]
        print(f"  类别 {class_id}: {counts} (总计: {sum(counts)})")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_kfold_splits()
