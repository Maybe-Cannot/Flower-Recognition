#!/usr/bin/env python3
"""
可视化10折交叉验证的折划分情况
"""
import random

def visualize_kfold_strategy(n_epochs=10, n_folds=10, n_val_folds=2, seed=42):
    """
    可视化K折交叉验证策略
    
    参数:
    n_epochs: 显示的epoch数
    n_folds: 总折数
    n_val_folds: 每个epoch用于验证的折数
    seed: 随机种子
    """
    print("=" * 80)
    print(f"10折交叉验证策略可视化 (显示前{n_epochs}个epoch)")
    print("=" * 80)
    print(f"总折数: {n_folds}")
    print(f"每epoch训练折数: {n_folds - n_val_folds}")
    print(f"每epoch验证折数: {n_val_folds}")
    print("=" * 80)
    print()
    
    # 统计每折被用作验证集的次数
    val_usage_count = {i: 0 for i in range(n_folds)}
    train_usage_count = {i: 0 for i in range(n_folds)}
    
    for epoch in range(n_epochs):
        # 模拟训练脚本中的逻辑
        random.seed(seed + epoch)
        all_fold_indices = list(range(n_folds))
        random.shuffle(all_fold_indices)
        val_fold_indices = sorted(all_fold_indices[:n_val_folds])
        train_fold_indices = sorted(all_fold_indices[n_val_folds:])
        
        # 统计
        for idx in val_fold_indices:
            val_usage_count[idx] += 1
        for idx in train_fold_indices:
            train_usage_count[idx] += 1
        
        # 可视化
        fold_status = ['T'] * n_folds
        for idx in val_fold_indices:
            fold_status[idx] = 'V'
        
        status_str = ' '.join([f"{i}:{s}" for i, s in enumerate(fold_status)])
        print(f"Epoch {epoch+1:2d}: {status_str}")
        print(f"          训练折: {train_fold_indices}")
        print(f"          验证折: {val_fold_indices}")
        print()
    
    # 显示统计信息
    print("=" * 80)
    print("使用统计 (T=训练次数, V=验证次数):")
    print("=" * 80)
    
    for i in range(n_folds):
        train_count = train_usage_count[i]
        val_count = val_usage_count[i]
        total_count = train_count + val_count
        train_pct = (train_count / n_epochs * 100) if n_epochs > 0 else 0
        val_pct = (val_count / n_epochs * 100) if n_epochs > 0 else 0
        
        print(f"折 {i}: T={train_count:2d} ({train_pct:5.1f}%), V={val_count:2d} ({val_pct:5.1f}%), 总计={total_count:2d}")
    
    print()
    print("=" * 80)
    print("说明:")
    print("  T = 训练集 (Training)")
    print("  V = 验证集 (Validation)")
    print("  每个折在不同epoch中会被用作训练集或验证集")
    print("  随着epoch增加，各折的使用会趋于均衡")
    print("=" * 80)

if __name__ == '__main__':
    # 可视化前20个epoch的折划分
    visualize_kfold_strategy(n_epochs=20, n_folds=10, n_val_folds=2, seed=42)
