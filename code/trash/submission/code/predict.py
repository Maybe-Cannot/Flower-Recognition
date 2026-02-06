#!/usr/bin/env python3
"""
花卉分类模型预测脚本（模型版，输出格式与示例脚本完全一致）

使用方法:
    python ./code/predict.py <测试集文件夹> <输出文件路径> [--model_path <模型权重路径>]

示例:
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv --model_path ../model/best_model.pth

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名
    - category_id: 预测的类别ID (对应花卉类别编号)
    - confidence: 预测置信度 (0-1之间，保留6位小数)
"""

import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import argparse
from pathlib import Path

from model import load_model


def get_image_files(img_dir, img_extensions=None):
    """获取目录中的所有图片文件"""
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    img_dir_path = Path(img_dir)
    image_files = []

    # 获取所有图片文件
    for ext in img_extensions:
        image_files.extend(img_dir_path.glob(f'*{ext}'))
        image_files.extend(img_dir_path.glob(f'*{ext.upper()}'))

    # 只保留文件名，并排序确保一致性
    image_files = sorted([f.name for f in image_files])

    return image_files


def get_predict_transform(img_size=224):
    """获取预测时的图像预处理transform"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(model, image_path, transform, device, class_to_idx=None):
    """
    对单张图片进行预测
    
    参数:
    model: 训练好的模型
    image_path: 图片路径
    transform: 图像预处理transform
    device: 设备
    class_to_idx: 类别到索引的映射字典
    
    返回:
    predicted_class: 预测的类别ID
    confidence: 预测置信度
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图片 {image_path}: {e}")
        # 返回默认值
        return 0, 0.0
    
    image = transform(image).unsqueeze(0)  # 添加batch维度
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        # 获取概率分布
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    
    # 将预测索引转换回原始类别ID
    if class_to_idx is not None:
        # 创建反向映射
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        predicted_class = int(idx_to_class.get(predicted_idx, predicted_idx))
    else:
        predicted_class = predicted_idx
    
    return predicted_class, confidence


def predict_batch(model, test_dir, output_file, device, transform, class_to_idx=None):
    """
    批量预测测试集图片
    
    参数:
    model: 训练好的模型
    test_dir: 测试集图片目录
    output_file: 输出文件路径
    device: 设备
    transform: 图像变换
    class_to_idx: 类别到索引的映射
    """
    # 获取所有测试图片
    print("正在扫描测试集目录...")
    image_files = get_image_files(test_dir)
    
    if not image_files:
        print(f"错误: 在目录 {test_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print()
    
    results = []
    print("正在生成预测...")
    
    for filename in image_files:
        img_path = os.path.join(test_dir, filename)
        try:
            category_id, confidence = predict_image(
                model, img_path, transform, device, class_to_idx
            )
            results.append({
                'filename': filename,
                'category_id': int(category_id),
                'confidence': round(float(confidence), 6)
            })
        except Exception as e:
            print(f"预测 {filename} 时出错: {e}")
            results.append({
                'filename': filename,
                'category_id': 0,
                'confidence': 0.0
            })
    
    # 保存结果
    df = pd.DataFrame(results)

    # 按照文件名排序并确保列顺序为要求的格式
    if 'filename' in df.columns:
        df = df.sort_values('filename').reset_index(drop=True)
        # 强制列顺序
        df = df[['filename', 'category_id', 'confidence']]
    else:
        df = df.sort_values(df.columns[0]).reset_index(drop=True)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    print()
    print("预测完成!")


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return None


def main():
    """主预测流程"""
    parser = argparse.ArgumentParser(description='花卉分类模型预测')
    # 位置参数：测试集文件夹和输出文件
    parser.add_argument('test_img_dir', type=str, help='测试图片目录')
    parser.add_argument('output_path', type=str, help='预测结果输出路径 (CSV文件)')
    # 可选参数：仅保留模型路径
    parser.add_argument('--model_path', type=str, default='../model/convnext_base_flower_recognition.pth', help='模型权重文件路径')
    args = parser.parse_args()
    
    print(f'测试集目录: {args.test_img_dir}')
    print(f'输出文件: {args.output_path}')
    print(f'模型路径: {args.model_path}')
    print()
    # 检查测试集目录是否存在
    if not os.path.exists(args.test_img_dir):
        print(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return
    # 固定参数
    num_classes = 102
    img_size = 224
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    try:
        model = load_model(args.model_path, num_classes, device)
        print("模型加载成功!")
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return
    print()
    transform = get_predict_transform(img_size)
    # 批量预测
    predict_batch(model, args.test_img_dir, args.output_path, device, transform)


if __name__ == '__main__':
    main()
