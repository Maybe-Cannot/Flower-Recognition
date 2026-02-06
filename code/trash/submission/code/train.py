#!/usr/bin/env python3
"""
花卉分类模型训练脚本
基于ConvNeXt-Base的花卉分类模型训练
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from PIL import Image, ImageFile

from model import create_model
from utils import (
    set_seed, reorganize_dataset_by_label, get_data_transforms,
    AverageMeter, calculate_accuracy, save_config
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
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
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

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch: [{epoch}][{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Acc@1: {top1.avg:.2f}%')

    return losses.avg, top1.avg.item()


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for inputs, labels in val_loader:
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

    return losses.avg, top1.avg.item()


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
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
    
    返回:
    model: 训练好的模型
    history: 训练历史
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
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
        
        print()

    print(f'训练完成。最佳验证准确率: {best_acc:.2f}%')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='花卉分类模型训练')
    parser.add_argument('--data_dir', type=str, 
                        default=r'train',
                        help='原始数据集目录')
    parser.add_argument('--organized_data_dir', type=str, 
                        default=r'organized_train',
                        help='组织后的数据集目录')
    parser.add_argument('--labels_file', type=str, 
                        default=r'train_labels.csv',
                        help='标签文件路径')
    parser.add_argument('--model_type', type=str, default='convnext_base',
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=102,
                        help='类别数量')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='model',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 数据路径配置
    data_dir = args.data_dir
    organized_data_dir = args.organized_data_dir
    labels_file = args.labels_file
    model_save_path = os.path.join(args.save_dir, 'convnext_base_flower_recognition.pth')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查是否已经重新组织过数据
    if not os.path.exists(organized_data_dir):
        print("正在根据标签文件重新组织数据集...")
        num_classes = reorganize_dataset_by_label(data_dir, organized_data_dir, labels_file)
    else:
        # 计算类别数
        train_classes = os.listdir(os.path.join(organized_data_dir, 'train'))
        num_classes = len(train_classes)
        print(f"使用已有的组织好的数据集,共 {num_classes} 个类别")
    
    # 更新类别数
    args.num_classes = num_classes
    
    # 获取数据转换
    data_transforms = get_data_transforms()
    
    # 加载数据集
    try:
        image_datasets = {}
        for x in ['train', 'val']:
            image_datasets[x] = datasets.ImageFolder(
                os.path.join(organized_data_dir, x),
                transform=data_transforms[x]
            )

        dataloaders = {x: torch.utils.data.DataLoader(
                            image_datasets[x], 
                            batch_size=args.batch_size,
                            shuffle=(x == 'train'), 
                            num_workers=args.num_workers,
                            pin_memory=True
                         ) for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        class_to_idx = image_datasets['train'].class_to_idx

        print(f"训练集大小: {dataset_sizes['train']}")
        print(f"验证集大小: {dataset_sizes['val']}")
        print(f"类别数: {len(class_names)}")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("请检查数据目录结构是否正确")
        return
    
    # 创建模型
    print(f"创建 {args.model_type} 模型...")
    model_ft = create_model(args.num_classes)
    
    # 设置设备
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
    
    # 训练模型
    print("开始训练模型...")
    try:
        model_ft, history = train_model(
            model_ft, 
            criterion, 
            optimizer_ft, 
            exp_lr_scheduler,
            dataloaders,
            dataset_sizes,
            device,
            num_epochs=args.epochs
        )
        
        # 获取最终最佳准确率
        final_best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        
        # 保存最终模型权重
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
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
        checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存为 {checkpoint_path}")
        
        # 保存配置文件
        config = {
            'model_type': args.model_type,
            'num_classes': args.num_classes,
            'img_size': args.img_size,
            'class_to_idx': class_to_idx,
            'best_accuracy': final_best_acc,
            'training_params': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
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
                'classifier_out_features': args.num_classes
            },
            'description': f'{args.model_type} based flower classification model with {args.num_classes} classes'
        }
        save_config(config, os.path.join(args.save_dir, 'config.json'))
        print(f"配置文件已保存")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
