#!/usr/bin/env python3
# 花卉分类模型训练脚本
# 基于YOLO的花卉分类模型训练

import os
import argparse
from ultralytics import YOLO
from model import CustomizedTrainer
from utils import (save_config, set_seed, validate_dataset_structure, 
                   get_class_names_from_directory, print_dataset_info, handle_dataset_split)


def main():
	"""主训练流程"""
	# ==================== 1. 参数解析 ====================
	parser = argparse.ArgumentParser(description='花卉分类模型训练 - YOLO')
	
	# 获取脚本目录，用于设置默认路径
	script_dir = os.path.dirname(os.path.abspath(__file__))
	default_project = os.path.join(script_dir, '..', 'model')  # 默认保存到 new/model
	default_model = os.path.join(script_dir, '..', 'model', 'yolo11m-cls.pt')  # 默认模型路径
	
	# 位置参数
	parser.add_argument('data_dir', 		type=str, 											help='数据集目录（包含train/val子目录）')
	parser.add_argument('labels_file', 		type=str, 				default=None, nargs='?',	help='标签CSV文件（可选，用于数据集划分）')
	# 输出控制
	parser.add_argument('--project', 		type=str, 				default=default_project, 	help='项目保存目录')
	parser.add_argument('--name', 			type=str, 				default='1112', 			help='运行名称')
	# 模型参数
	parser.add_argument('--model', 			type=str, 				default=default_model, 		help='预训练模型路径')
	parser.add_argument('--resume', 		type=str, 				default=False, 				help='恢复训练启用')
	# 训练参数
	parser.add_argument('--epochs', 		type=int, 				default=50, 				help='训练轮数')  # 范围: 1-1000+, 建议: 50-200, 意义: 完整遍历数据集的次数
	parser.add_argument('--batch', 			type=int, 				default=16, 				help='批次大小')  # 范围: 1-256, 建议: 8-64, 意义: 每次迭代处理的样本数,受显存限制
	parser.add_argument('--imgsz', 			type=int, 				default=600, 				help='图像尺寸')  # 范围: 32-1280, 建议: 224/448/640, 意义: 输入图像边长(像素),需为32的倍数
	parser.add_argument('--device', 		type=str, 				default='cuda:0', 			help='训练设备 (0, 0,1,2,3, cpu)')
	parser.add_argument('--workers', 		type=int, 				default=0, 					help='数据加载线程数（0=主进程加载，避免共享内存问题）')  # 范围: 0-16, 建议: 0-8, 意义: 多进程加载数据,0=单进程
	# 优化器参数
	parser.add_argument('--optimizer', 		type=str, 				default='Adam', 			help='优化器 (SGD, Adam, AdamW, RAdam, NAdam)')  # 可选: Adam/AdamW/SGD/RAdam/NAdam, 建议: Adam(通用)/AdamW(正则化)
	parser.add_argument('--lr0', 			type=float, 			default=0.001, 				help='初始学习率 (SGD=1e-2, Adam=1e-3)')  # 范围: 1e-5 - 1e-1, 建议: SGD=1e-2/Adam=1e-3, 意义: 参数更新步长,过大易震荡,过小收敛慢
	parser.add_argument('--lrf', 			type=float, 			default=0.01, 				help='最终学习率 (占初始lr0的比例)')  # 范围: 0.01 - 1.0, 建议: 0.01 - 0.1, 意义: 训练结束时学习率=lr0*lrf,配合调度器逐步降低学习率
	parser.add_argument('--weight_decay', 	type=float, 			default=0.0005, 			help='权重衰减')  # 范围: 0 - 1e-2, 建议: 1e-5 - 1e-3, 意义: L2正则化系数,防止过拟合
	parser.add_argument('--momentum', 		type=float, 			default=0.937, 				help='SGD动量')  # 范围: 0 - 0.999, 建议: 0.9 - 0.99, 意义: 梯度累积系数,加速收敛并减少震荡(仅SGD)
	parser.add_argument('--cls', 			type=float, 			default=0.5, 				help='分类损失权重')  # 范围: 0 - 1, 建议: 0.5 - 1.0, 意义: 分类损失在总损失函数中的权重,影响正确类别预测的重要性
	# 训练控制
	parser.add_argument('--patience', 		type=int, 				default=5, 				help='早停耐心值')  # 范围: 0-100, 建议: 10-50, 意义: 验证集无改善的最大轮数,0=禁用早停
	parser.add_argument('--seed', 			type=int, 				default=2025, 					help='随机种子')  # 范围: 0-2^32, 建议: 0-9999, 意义: 控制随机性以保证结果可复现
	parser.add_argument('--amp', 			action='store_true', 	default=True,				help='混合精度训练（默认启用）')  # True=启用FP16加速,减少显存并提速1.5-2倍
	parser.add_argument('--plots', 			action='store_true', 	default=False,				help='生成训练图表')  # True=生成训练曲线和可视化图表

	# 数据增强参数（仅用于CustomizedDataset）
	parser.add_argument('--hsv_h', 			type=float, 			default=0.001, 				help='HSV色调增强范围')  # 范围: 0 - 1, 建议: 0.01 - 0.1, 意义: 色调随机偏移范围(0-180度的归一化值)
	parser.add_argument('--hsv_s', 			type=float, 			default=0.1, 				help='HSV饱和度增强范围')  # 范围: 0 - 1, 建议: 0.3 - 0.9, 意义: 饱和度随机变化比例
	parser.add_argument('--hsv_v', 			type=float, 			default=0, 				help='HSV明度增强范围')  # 范围: 0 - 1, 建议: 0.2 - 0.6, 意义: 明度(亮度)随机变化比例
	parser.add_argument('--fliplr', 		type=float, 			default=0.5, 				help='水平翻转概率')  # 范围: 0 - 1, 建议: 0 - 0.5, 意义: 左右镜像翻转的概率
	parser.add_argument('--flipud', 		type=float, 			default=0.001, 				help='垂直翻转概率')  # 范围: 0 - 1, 建议: 0 - 0.5, 意义: 上下镜像翻转的概率(多数场景不建议)
	parser.add_argument('--erasing', 		type=float, 			default=0.0, 				help='随机擦除概率')  # 范围: 0 - 1, 建议: 0 - 0.5, 意义: 随机遮挡图像区域的概率,增强鲁棒性
	args = parser.parse_args()
	
	# ==================== 2. 设置和验证路径 ====================
	# 数据目录处理（用户输入的相对路径基于终端工作路径）
	if os.path.isabs(args.data_dir):
		raw_data_dir = args.data_dir
	else:
		# 用户输入的相对路径，基于当前工作目录
		raw_data_dir = os.path.abspath(args.data_dir)
	
	# 模型保存目录处理
	if os.path.isabs(args.project):
		# 用户提供了绝对路径
		model_dir = args.project
	else:
		# 用户提供了相对路径（基于终端工作路径）或使用默认值（基于脚本位置）
		model_dir = os.path.abspath(args.project)
	os.makedirs(model_dir, exist_ok=True)
	
	# 处理数据集：如果提供了labels_file，则需要先划分数据集
	if args.labels_file:
		# 处理CSV路径（用户输入的相对路径基于终端工作路径）
		if os.path.isabs(args.labels_file):
			csv_path = args.labels_file
		else:
			csv_path = os.path.abspath(args.labels_file)
		
		# 使用utils中的函数处理数据集划分
		data_dir, stats = handle_dataset_split(raw_data_dir, csv_path, model_dir, args.seed)
		if data_dir is None:
			return
	else:
		# 直接使用已经划分好的数据集
		data_dir = raw_data_dir
		stats = {}
	
	# 打印最终路径信息
	print("\n" + "=" * 60)
	print("训练配置:")
	print(f"  数据集目录: {data_dir}")
	print(f"  模型保存目录: {model_dir}")
	print(f"  预训练模型: {args.model}")
	print("=" * 60)
	
	# 验证数据集结构
	if not validate_dataset_structure(data_dir, ['train', 'val']):
		print("[ERROR] 数据集目录结构不正确")
		print("[INFO] 数据集必须包含 train/ 和 val/ 子目录")
		return
	
	print_dataset_info(data_dir)
	
	# 获取类别数
	train_dir = os.path.join(data_dir, 'train')
	num_classes = len(get_class_names_from_directory(train_dir))
	print(f"[INFO] 检测到 {num_classes} 个类别")
	
	# ==================== 3. 加载模型 ====================
	set_seed(args.seed)
	
	if args.resume and os.path.exists(args.resume):
		print(f"[INFO] 从检查点恢复训练: {args.resume}")
		model = YOLO(args.resume)
	else:
		print(f"[INFO] 加载预训练模型: {args.model}")
		model = YOLO(args.model)
	
	# ==================== 4. 训练模型 ====================
	print("\n" + "=" * 60)
	print("开始训练...")
	print(f"  模型: {args.model}")
	print(f"  数据集: {data_dir}")
	print(f"  训练轮数: {args.epochs}")
	print(f"  批次大小: {args.batch}")
	print(f"  图像尺寸: {args.imgsz}")
	print(f"  设备: {args.device or 'auto'}")
	print(f"  混合精度: {args.amp}")
	print("  数据增强: RandAugment + ColorJitter + RandomErasing")
	print("=" * 60 + "\n")
	
	try:
		results = model.train(
			trainer=CustomizedTrainer,  # 使用自定义训练器
			data=data_dir,
			epochs=args.epochs,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
			workers=args.workers,
			optimizer=args.optimizer,
			lr0=args.lr0,
			lrf=args.lrf,
			weight_decay=args.weight_decay,
			momentum=args.momentum,
			patience=args.patience,
			seed=args.seed,
			amp=args.amp,
			plots=args.plots,
			project=model_dir,
			name=args.name,
			exist_ok=True,
			pretrained=True,
			verbose=True,
			val=True,
			save=True,
			cls=args.cls,  # 分类损失权重
			# 自定义数据增强参数
			hsv_h=args.hsv_h,
			hsv_s=args.hsv_s,
			hsv_v=args.hsv_v,
			fliplr=args.fliplr,
			flipud=args.flipud,
			erasing=args.erasing,
		)
		print(f"\n[SUCCESS] 训练完成!")
		# ==================== 5. 保存配置信息 ====================
		config_path = os.path.join(model_dir, 'config.json')
		config = {
			'model': args.model,
			'num_classes': num_classes,
			'data_dir': args.data_dir,
			'training': {
				'epochs': args.epochs,
				'batch': args.batch,
				'imgsz': args.imgsz,
				'optimizer': args.optimizer,
				'lr0': args.lr0,
				'lrf': args.lrf,
				'weight_decay': args.weight_decay,
				'momentum': args.momentum,
				'patience': args.patience,
				'seed': args.seed,
				'amp': args.amp,
				'cls': args.cls,
			},
			'augmentation': {
				'framework': 'Custom (RandAugment + ColorJitter + RandomErasing)',
				'hsv_h': args.hsv_h,
				'hsv_s': args.hsv_s,
				'hsv_v': args.hsv_v,
				'fliplr': args.fliplr,
				'flipud': args.flipud,
				'erasing': args.erasing,
			}
		}
		
		# 如果进行了数据集划分，添加划分信息到config
		if args.labels_file and 'report' in stats:
			config['dataset_split'] = stats['report']
		save_config(config, config_path)
		# 显示模型路径
		best_model = os.path.join(model_dir, args.name, 'weights', 'best.pt')
		last_model = os.path.join(model_dir, args.name, 'weights', 'last.pt')
		print(f"[INFO] 最佳模型: {best_model}")
		print(f"[INFO] 最后模型: {last_model}")
		
	except Exception as e:
		print(f"\n[ERROR] 训练失败: {e}")
		import traceback
		traceback.print_exc()


if __name__ == '__main__':
	main()

