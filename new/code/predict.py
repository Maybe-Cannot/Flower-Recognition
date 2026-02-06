#!/usr/bin/env python3
"""
花卉分类模型预测脚本（YOLO版）

使用方法:
    python predict.py <测试集文件夹> <输出文件路径> [--model_path <模型权重路径>]

示例:
    python predict.py ./test_images ./results/submission.csv
    python predict.py ./test_images ./results/submission.csv --model_path ../model/runs/weights/best.pt

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名
    - category_id: 预测的类别ID (对应花卉类别编号)
    - confidence: 预测置信度 (0-1之间，保留6位小数)
    
评估模式:
    如果测试集目录包含子文件夹（按类别组织），会自动启动评估程序
    - 计算整体准确率 (ACC)
    - 输出每个类别的混淆情况
    - 生成详细的错误分析报告
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
from utils import check_has_subfolders, collect_images_with_labels, evaluate_predictions


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


def predict_batch(model, test_img_dir, output_file, imgsz=640, conf_threshold=0.0, image_label_map=None):
    """
    批量预测测试集图片
    
    参数:
    model: 训练好的YOLO模型
    test_img_dir: 测试集图片目录
    output_file: 输出文件路径
    imgsz: 图像尺寸（需与训练时一致）
    conf_threshold: 置信度阈值（可选）
    image_label_map: 图片标签映射（如果提供，则从中获取图片路径）
    """
    # 定义与训练时相同的预处理transforms
    preprocess = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 获取所有测试图片
    print("正在扫描测试集目录...")
    
    if image_label_map:
        # 评估模式：从image_label_map中获取图片路径
        image_files = list(image_label_map.keys())
        img_paths = [image_label_map[f]['path'] for f in image_files]
        print(f"找到 {len(image_files)} 张图片（从子文件夹）")
    else:
        # 预测模式：直接从目录获取图片
        image_files = get_image_files(test_img_dir)
        if not image_files:
            print(f"错误: 在目录 {test_img_dir} 中未找到图片文件")
            return
        img_paths = [os.path.join(test_img_dir, f) for f in image_files]
        print(f"找到 {len(image_files)} 张图片")
    
    print()
    
    results = []
    print("正在生成预测...")
    
    # 分批预测以避免内存溢出
    batch_size = 32  # 每批处理32张图片
    total_batches = (len(img_paths) + batch_size - 1) // batch_size
    
    # 获取设备
    device = next(model.model.parameters()).device
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_files = image_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"处理批次 {batch_num}/{total_batches} ({len(batch_paths)} 张图片)...", end='\r')
        
        # 手动预处理图片
        batch_tensors = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"\n加载图片 {img_path} 失败: {e}")
                # 使用全零张量占位
                batch_tensors.append(torch.zeros((3, imgsz, imgsz)))
        
        # 转换为批次tensor
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model.model(batch_tensor)
            # YOLO分类模型可能返回元组，取第一个元素
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
        
        # 收集结果
        for idx, filename in enumerate(batch_files):
            try:
                # 获取预测结果
                prob = probs[idx]
                class_idx = torch.argmax(prob).item()
                confidence = prob[class_idx].item()
                class_name = model.names[class_idx]
                
                # 尝试将类别名称转换为整数ID
                try:
                    category_id = int(class_name)
                except ValueError:
                    category_id = class_idx
                
                results.append({
                    'filename': filename,
                    'category_id': int(category_id),
                    'confidence': round(float(confidence), 6)
                })
            except Exception as e:
                print(f"\n预测 {filename} 时出错: {e}")
                results.append({
                    'filename': filename,
                    'category_id': 0,
                    'confidence': 0.0
                })
    
    print(f"\n预测完成! 共处理 {len(results)} 张图片")
    print()
    
    # 保存结果
    df = pd.DataFrame(results)
    
    # 按照文件名排序并确保列顺序
    df = df.sort_values('filename').reset_index(drop=True)
    df = df[['filename', 'category_id', 'confidence']]
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到: {output_file}")
    print(f"平均置信度: {df['confidence'].mean():.4f}")
    print()
    
    return df


def main():
    """主预测流程"""
    parser = argparse.ArgumentParser(description='花卉分类模型预测 - YOLO')
    parser.add_argument('test_img_dir',     type=str,                   help='测试图片目录')
    parser.add_argument('output_path',      type=str,                   help='预测结果输出路径 (CSV文件)')
    parser.add_argument('--model_path',     type=str,   default=None,   help='模型权重文件路径')
    parser.add_argument('--imgsz',          type=int,   default=600,    help='图像尺寸（需与训练时一致）')
    parser.add_argument('--device',         type=str,   default='cuda:0', help='推理设备 (cuda:0, cpu)')
    parser.add_argument('--conf',           type=float, default=0.0,    help='置信度阈值')
    
    args = parser.parse_args()
    
    # ==================== 路径配置 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, '..', 'model'))
    result_dir = os.path.abspath(os.path.join(script_dir, '..', 'result'))
    
    # 模型权重路径（优先使用用户指定的，否则使用默认路径）
    if args.model_path:
        model_path = args.model_path if os.path.isabs(args.model_path) else os.path.abspath(args.model_path)
    else:
        # 默认路径: new/model/runs/weights/best.pt
        model_path = os.path.join(model_dir, 'v10m', 'weights', 'best.pt')
    
    # 测试图片目录
    test_img_dir = args.test_img_dir if os.path.isabs(args.test_img_dir) else os.path.abspath(args.test_img_dir)
    
    # 输出CSV路径
    output_path = args.output_path if os.path.isabs(args.output_path) else os.path.abspath(args.output_path)
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    print("=" * 60)
    print("YOLO 花卉分类预测")
    print("=" * 60)
    print(f"  脚本目录: {script_dir}")
    print(f"  测试集目录: {test_img_dir}")
    print(f"  模型权重: {model_path}")
    print(f"  输出文件: {output_path}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  推理设备: {args.device}")
    print(f"  置信度阈值: {args.conf}")
    print("=" * 60)
    print()
    
    # 检查测试集目录
    if not os.path.exists(test_img_dir):
        print(f"错误: 测试集目录不存在: {test_img_dir}")
        return
    
    # 检查是否包含子文件夹（判断是否为带标签的数据集）
    has_labels, subfolders = check_has_subfolders(test_img_dir)
    image_label_map = None
    
    if has_labels:
        print(f"[INFO] 检测到 {len(subfolders)} 个子文件夹，启动评估模式")
        image_label_map = collect_images_with_labels(test_img_dir)
        print(f"[INFO] 收集到 {len(image_label_map)} 张带标签的图片")
        print()
    else:
        print(f"[INFO] 未检测到子文件夹，使用预测模式（不进行评估）")
        print()
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"提示: 请使用 --model_path 参数指定正确的模型路径")
        return
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        model.to(args.device)  # 将模型移到指定设备
        print("模型加载成功!")
        print(f"模型类别数: {len(model.names)}")
        print(f"类别示例: {dict(list(model.names.items())[:5])}...")
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return
    print()
    
    # 批量预测
    predictions_df = predict_batch(
        model, 
        test_img_dir, 
        output_path,
        imgsz=args.imgsz,  # 传递图像尺寸
        conf_threshold=args.conf,
        image_label_map=image_label_map  # 传递图片映射（如果有）
    )
    
    # 如果有标签，进行评估
    if has_labels and image_label_map and predictions_df is not None:
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        evaluate_predictions(predictions_df, image_label_map, output_dir)


if __name__ == '__main__':
    main()



