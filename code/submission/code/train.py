#!/usr/bin/env python3
"""
花卉分类模型训练脚本
基于ConvNeXt-Base的花卉分类模型训练
"""

import os
import time
import json
import random
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from PIL import Image, ImageFile

from model import create_model
from utils import (
    set_seed, get_data_transforms,
    AverageMeter, calculate_accuracy, save_config,
    EarlyStopping, create_kfold_splits, KFoldFlowerDataset
)

# 设置线程库以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'[Train]', ncols=100)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        acc1, _ = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))

        # 更新进度条显示
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })

    return losses.avg, top1.avg.item()


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    # 创建进度条
    pbar = tqdm(val_loader, desc='  [Val]', ncols=100)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 计算准确率
            acc1, _ = calculate_accuracy(outputs, labels, topk=(1, 5))
            # 更新统计
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            
            # 更新进度条显示
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%'
            })
    
    return losses.avg, top1.avg.item()


def train_model_with_kfold(model, criterion, optimizer, scheduler, fold_csvs, data_dir, class_id_to_idx, 
                          batch_size, num_workers, device, num_epochs=25, n_folds=10, n_val_folds=2, 
                          early_stopping_patience=None, seed=42):
    """
    使用K折交叉验证训练模型，每个epoch随机选择验证折
    
    参数:
    model: 要训练的模型
    criterion: 损失函数
    optimizer: 优化器
    scheduler: 学习率调度器
    fold_csvs: K折CSV文件路径列表
    data_dir: 数据目录
    class_id_to_idx: 类别映射
    batch_size: 批次大小
    num_workers: 数据加载线程数
    device: 训练设备
    num_epochs: 训练轮数
    n_folds: 总折数
    n_val_folds: 每个epoch用作验证的折数
    early_stopping_patience: 早停耐心值
    seed: 随机种子基数
    
    返回:
    model: 训练好的模型
    history: 训练历史
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # 初始化早停机制
    early_stopping = None
    if early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        print(f"EarlyStopping: {early_stopping_patience} epochs")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'fold_splits': []  # 记录每个epoch的折划分
    }
    
    # 获取数据增强
    data_transforms = get_data_transforms()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 每个epoch随机选择验证折（使用epoch作为随机种子以保证可重现性）
        random.seed(seed + epoch)
        all_fold_indices = list(range(n_folds))
        random.shuffle(all_fold_indices)
        val_fold_indices = all_fold_indices[:n_val_folds]
        train_fold_indices = all_fold_indices[n_val_folds:]
        
        val_fold_indices.sort()
        train_fold_indices.sort()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}\t训练折: {train_fold_indices}\t验证折: {val_fold_indices}')
        
        # 记录折划分
        history['fold_splits'].append({
            'epoch': epoch + 1,
            'train_folds': train_fold_indices,
            'val_folds': val_fold_indices
        })
        
        # 创建训练集和验证集
        train_csvs = [fold_csvs[i] for i in train_fold_indices]
        val_csvs = [fold_csvs[i] for i in val_fold_indices]
        
        train_dataset = KFoldFlowerDataset(
            train_csvs, 
            data_dir, 
            transform=data_transforms['train'],
            class_id_to_idx=class_id_to_idx
        )
        val_dataset = KFoldFlowerDataset(
            val_csvs, 
            data_dir, 
            transform=data_transforms['val'],
            class_id_to_idx=class_id_to_idx
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch: [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}% '
              f'Time: {epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            print(f'best acc: {best_acc:.2f}%')
        
        # 早停检查
        if early_stopping is not None:
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                print(f"\nStop! epoch: {epoch+1}\tbest acc: {best_acc:.2f}")
                break
        print()

    if early_stopping is not None and early_stopping.early_stop:
        print(f'训练早停。最佳验证准确率: {best_acc:.2f}%')
    else:
        print(f'训练完成。最佳验证准确率: {best_acc:.2f}%')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, early_stopping_patience=None):
    """
    训练模型的主函数（向后兼容版本）
    参数:
    model: 要训练的模型
    criterion: 损失函数
    optimizer: 优化器
    scheduler: 学习率调度器
    dataloaders: 数据加载器字典
    dataset_sizes: 数据集大小字典
    device: 训练设备
    num_epochs: 训练轮数
    early_stopping_patience: 早停耐心值(None表示不使用早停)
    返回:
    model: 训练好的模型
    history: 训练历史
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # 初始化早停机制
    early_stopping = None
    if early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        print(f"启用早停机制，耐心值: {early_stopping_patience} epochs")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()
        # 训练
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch: [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}% '
              f'Time: {epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            print(f'新的最佳验证准确率: {best_acc:.2f}%')
        
        # 早停检查
        if early_stopping is not None:
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                print(f"\n早停触发! 在第 {epoch+1} 个epoch停止训练")
                print(f"最佳验证准确率: {best_acc:.2f}%")
                break
        
        print()

    if early_stopping is not None and early_stopping.early_stop:
        print(f'训练提前停止。最佳验证准确率: {best_acc:.2f}%')
    else:
        print(f'训练完成。最佳验证准确率: {best_acc:.2f}%')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='花卉分类模型训练')
    parser.add_argument('--data_dir',               type=str,   default=r'train',               help='原始数据集目录')
    parser.add_argument('--labels_file',            type=str,   default=r'train_labels.csv',    help='标签文件路径')
    parser.add_argument('--save_dir',               type=str,   default='model',                help='模型保存目录')
    parser.add_argument('--resume',                 type=str,   default=None,                   help='恢复训练的模型路径')

    parser.add_argument('--model_type',             type=str,   default='convnext_base',        help='模型类型')
    parser.add_argument('--img_size',               type=int,   default=224,                    help='图像尺寸')

    parser.add_argument('--seed',                   type=int,   default=42,                     help='随机种子')
    parser.add_argument('--early_stopping',         type=int,   default=10,                     help='早停耐心值(epochs),设为0或None则不使用早停')
    parser.add_argument('--batch_size',             type=int,   default=64,                     help='批次大小')
    parser.add_argument('--epochs',                 type=int,   default=50,                     help='训练轮数')
    parser.add_argument('--kfold',                  type=int,   default=10,                     help='数据集折数')
    parser.add_argument('--num_workers',            type=int,   default=0,                      help='数据加载线程数')

    parser.add_argument('--lr',                     type=float, default=0.001,                  help='学习率')
    parser.add_argument('--weight_decay',           type=float, default=1e-4,                   help='权重衰减')
    parser.add_argument('--label_smoothing',        type=float, default=0.1,                    help='标签平滑')
    parser.add_argument('--use_advanced_aug',       action='store_true',                        help='使用高级数据增强(RandAugment, ColorJitter, RandomErasing)')

    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # ==================== 路径配置（集中管理） ====================
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据相关路径
    data_dir = args.data_dir  # 原始图像目录
    labels_file = args.labels_file  # 标签CSV文件
    # 模型保存目录（submission/model）
    model_dir = os.path.abspath(os.path.join(script_dir, '..', 'model'))
    # 模型权重保存路径
    model_save_path = os.path.join(model_dir, 'convnext_large_flower_plus_recognition.pth')
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    # 配置文件保存路径
    config_path = os.path.join(model_dir, 'config.json')
    # 创建必要的目录
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("路径配置信息:")
    print(f"  脚本目录: {script_dir}")
    print(f"  数据目录: {data_dir}")
    print(f"  标签文件: {labels_file}")
    print(f"  模型保存目录: {model_dir}")
    print(f"  模型权重: {model_save_path}")
    print(f"  检查点: {checkpoint_path}")
    print(f"  配置文件: {config_path}")
    print("=" * 60)
    # ===========================================================

    # 使用10折交叉验证进行数据集划分
    print("正在创建10折交叉验证数据集划分...")
    kfold_info = create_kfold_splits(
        src_dir=args.data_dir,
        dst_dir=model_dir,
        labels_file=labels_file,
        n_folds=args.kfold,
        seed=args.seed,
        verify_images=True
    )
    
    # 提取信息
    fold_csvs = kfold_info['fold_csvs']
    num_classes = kfold_info['num_classes']
    class_id_to_idx = kfold_info['class_id_to_idx']
    idx_to_class_id = kfold_info['idx_to_class_id']
    fold_sizes = kfold_info['fold_sizes']
    
    print(f"\n类别数: {num_classes}")
    print(f"总样本数: {sum(fold_sizes)}")
    print(f"每折样本数: {fold_sizes}")
    print(f"每个epoch将使用 8 折训练 ({sum(fold_sizes)*0.8:.0f} 样本), 2 折验证 ({sum(fold_sizes)*0.2:.0f} 样本)")

    # 创建模型
    print(f"创建 {args.model_type} 模型...")
    model_ft = create_model(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    print(f"使用设备: {device}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer_ft = optim.Adam(
        model_ft.classifier[2].parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=10, eta_min=0.0001
    )

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint.get('history', history)
            print(f"已加载检查点 (epoch {start_epoch}, best_acc: {best_acc:.2f}%)")
        else:
            print(f"未找到检查点 '{args.resume}'")

    # 训练模型（使用10折交叉验证）
    print("开始训练模型（10折交叉验证）...")
    try:
        model_ft, history = train_model_with_kfold(
            model_ft, 
            criterion, 
            optimizer_ft, 
            exp_lr_scheduler,
            fold_csvs=fold_csvs,
            data_dir=args.data_dir,
            class_id_to_idx=class_id_to_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            num_epochs=args.epochs,
            n_folds=args.kfold,
            n_val_folds=2,
            early_stopping_patience=args.early_stopping,
            seed=args.seed
        )
        final_best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        
        # 保存模型权重
        torch.save(model_ft.state_dict(), model_save_path)
        print(f"模型权重已保存为 {model_save_path}")
        
        # 保存完整检查点
        checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'best_acc': final_best_acc,
            'history': history,
            'config': vars(args)
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存为 {checkpoint_path}")
        config = {
            'model_type': args.model_type,
            'num_classes': num_classes,
            'img_size': args.img_size,
            'best_accuracy': final_best_acc,
            'training_params': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'kfolds': args.kfold,
                'label_smoothing': args.label_smoothing,
                'optimizer': 'Adam',
                'seed': args.seed
            },
            'lr_scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 10,
                'eta_min': 0.0001
            },
            'data_augmentation': {
                'random_resized_crop': args.img_size,
                'random_horizontal_flip': True,
                'resize': 256,
                'center_crop': args.img_size
            },
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'model_details': {
                'pretrained_weights': 'ImageNet',
                'frozen_layers': 'all_except_classifier',
                'classifier_in_features': 1024,
                'classifier_out_features': num_classes
            },
            'class_mapping': {
                'class_id_to_idx': class_id_to_idx,
                'idx_to_class_id': idx_to_class_id
            }
        }
        
        # 保存配置文件
        save_config(config, config_path)
        print(f"配置文件已保存: {config_path}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
