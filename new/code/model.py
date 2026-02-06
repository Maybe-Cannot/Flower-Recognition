"""
数据集处理模块
包含数据集划分和图像处理类
"""

import os
from typing import Dict, List, Tuple, Callable
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from utils import (
	sanitize_class_name, ensure_dir, copy_file, 
	read_csv_labels, get_class_mapping, split_indices
)


# ==================== 自定义数据增强类 ====================
class CustomizedDataset(ClassificationDataset):
	"""自定义分类数据集，增强数据增强功能"""	
	def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
		super().__init__(root, args, augment, prefix)
		
		# 训练时的增强transforms
		train_transforms = T.Compose([
			T.Resize((args.imgsz, args.imgsz)),
			T.RandomHorizontalFlip(p=args.fliplr),
			T.RandomVerticalFlip(p=args.flipud),
			T.RandAugment(num_ops=2, magnitude=9, interpolation=T.InterpolationMode.BILINEAR),
			T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			T.RandomErasing(p=getattr(args, 'erasing', 0.0), scale=(0.02, 0.33), ratio=(0.3, 3.3), inplace=True),
		])
		
		# 验证时的transforms
		val_transforms = T.Compose([
			T.Resize((args.imgsz, args.imgsz)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		self.torch_transforms = train_transforms if augment else val_transforms

class CustomizedTrainer(ClassificationTrainer):
	"""自定义训练器，使用增强的数据集"""
	def build_dataset(self, img_path: str, mode: str = "train", batch=None):
		return CustomizedDataset(root=img_path, args=self.args, augment=(mode == "train"), prefix=mode)

class CustomizedValidator(ClassificationValidator):
	"""自定义验证器，使用增强的数据集"""
	def build_dataset(self, img_path: str, mode: str = "val"):
		return CustomizedDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)
# =========================================================


# 数据集划分工具类,用于按类别划分数据集为train/test/val
class DatasetSplitter:
	# 初始化数据集划分器 
	def __init__(self, images_dir: str, csv_path: str, seed: int = 42):
		"""
		Args:
			images_dir: 源图像所在目录
			csv_path: 标注CSV文件路径 (需包含: filename, category_id, chinese_name, english_name)
			seed: 随机种子，用于可重现性
		"""
		self.images_dir = images_dir
		self.csv_path = csv_path
		self.seed = seed
		self.rows: List[Dict[str, str]] = []
		self.groups: Dict[str, List[Dict[str, str]]] = {}
	
	# 构建输出文件路径
	def build_output_path(self, output_root: str, split: str, class_name: str, filename: str) -> str:
		sanitized = sanitize_class_name(class_name)
		return os.path.join(output_root, split, sanitized, filename)
	
	# 执行数据集划分
	def split_dataset(
		self,
		output_dir: str = "dataset_split",
		class_col: str = "category_id",
		train_ratio: float = 0.7,
		test_ratio: float = 0.2,
	) -> Dict[str, int]:
		"""
		Args:
			output_dir: 输出根目录，将创建 train/test/val 子目录
			class_col: 作为类别子目录名的列 (english_name/chinese_name/category_id)
			train_ratio: 训练集比例
			test_ratio: 测试集比例
		Returns:	统计信息字典 {"train": count, "test": count, "val": count, "missing": count}
		"""
		if not self.rows:  # 读取和分组
			self.rows = read_csv_labels(self.csv_path)
		self.groups = get_class_mapping(self.rows, class_col)
		
		for split in ("train", "test", "val"):  # 准备输出目录
			ensure_dir(os.path.join(output_dir, split))
		
		stats = {"train": 0, "test": 0, "val": 0, "missing": 0}
		missing_files: List[str] = []
		
		for class_name, items in self.groups.items():  # 按类别划分和复制
			n = len(items)
			if n == 0:
				continue
			train_idx, test_idx, val_idx = split_indices(n, train_ratio, test_ratio, self.seed)
			
			def items_from(indices: List[int]):
				for i in indices:
					yield items[i]
			
			for split_name, indices in [("train", train_idx), ("test", test_idx), ("val", val_idx)]:
				for row in items_from(indices):
					filename = row["filename"].strip()
					src_path = os.path.join(self.images_dir, filename)
					if not os.path.isfile(src_path):
						stats["missing"] += 1
						missing_files.append(src_path)
						continue
					dst_path = self.build_output_path(output_dir, split_name, class_name, filename)
					copy_file(src_path, dst_path)
					stats[split_name] += 1
		
		# 输出报告到屏幕并返回报告信息
		report_info = self.save_report(output_dir, stats, missing_files, class_col, train_ratio, test_ratio)
		stats['report'] = report_info  # 添加报告信息到stats中
		return stats
	
	def save_report(self, output_dir: str, stats: Dict[str, int], missing_files: List[str], 
					class_col: str, train_ratio: float, test_ratio: float) -> Dict:
		"""输出数据集划分报告到屏幕，并返回统计信息供config使用"""
		print("\n" + "=" * 60)
		print("数据集划分报告")
		print("=" * 60)
		print(f"源图像目录: {self.images_dir}")
		print(f"标签CSV文件: {self.csv_path}")
		print(f"输出目录: {output_dir}")
		print(f"类别列: {class_col}")
		print(f"随机种子: {self.seed}")
		print(f"\n划分比例:")
		print(f"  训练集: {train_ratio:.1%}")
		print(f"  测试集: {test_ratio:.1%}")
		print(f"  验证集: {1-train_ratio-test_ratio:.1%}")
		print(f"\n统计结果:")
		print(f"  训练集: {stats['train']} 张")
		print(f"  测试集: {stats['test']} 张")
		print(f"  验证集: {stats['val']} 张")
		print(f"  总计: {stats['train'] + stats['test'] + stats['val']} 张")
		print(f"  缺失文件: {stats['missing']} 个")
		if missing_files:
			print(f"\n缺失文件 (前10个):")
			for path in missing_files[:10]:
				print(f"  {path}")
			if len(missing_files) > 10:
				print(f"  ... 还有 {len(missing_files) - 10} 个文件")
		print("=" * 60 + "\n")
		
		# 返回报告信息供config使用
		return {
			'source_images_dir': self.images_dir,
			'csv_file': self.csv_path,
			'output_dir': output_dir,
			'class_column': class_col,
			'seed': self.seed,
			'split_ratios': {
				'train': train_ratio,
				'test': test_ratio,
				'val': 1 - train_ratio - test_ratio
			},
			'statistics': {
				'train': stats['train'],
				'test': stats['test'],
				'val': stats['val'],
				'total': stats['train'] + stats['test'] + stats['val'],
				'missing': stats['missing']
			}
		}
	



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="数据集处理工具：划分数据集或处理图片")
	subparsers = parser.add_subparsers(dest="command", help="选择操作")
	
	split_parser = subparsers.add_parser("split",   help="按子类划分数据集为 train/test/val")  # 数据集划分子命令
	split_parser.add_argument("--csv",              dest="csv_path",    default="train_labels.csv",             help="标注 CSV 路径")
	split_parser.add_argument("--images_dir",                           default="train",                        help="源图像所在目录（CSV 中 filename 对应此目录下的文件）")
	split_parser.add_argument("--output_dir",                           default="flowerme",                     help="输出根目录，将创建 train/test/val 子目录")
	split_parser.add_argument("--class_col",        choices=["english_name", "chinese_name", "category_id"], 
							                                            default="category_id",                  help="作为类别子目录名的列")
	split_parser.add_argument("--seed",             type=int,           default=42,                             help="随机种子")
	split_parser.add_argument("--train_ratio",      type=float,         default=0.7,                            help="训练集比例")
	split_parser.add_argument("--test_ratio",       type=float,         default=0.2,                            help="测试集比例")
	
	process_parser = subparsers.add_parser("process",  help="处理文件夹中的图片")  # 图片处理子命令
	process_parser.add_argument("folder_path",         help="包含图片的文件夹路径")
	process_parser.add_argument("--action",         choices=["validate", "resize"], 
							                                            default="validate",                     help="处理动作")
	process_parser.add_argument("--output_dir",                                                                 help="输出目录（用于resize）")
	process_parser.add_argument("--size",  nargs=2, type=int, 
                                                                        default=[224, 224],                     help="目标尺寸 (宽 高)")
	process_parser.add_argument("--remove_corrupted",   action="store_true",                                    help="删除损坏的图片")
	process_parser.add_argument("--no_recursive",       action="store_true",                                    help="不递归处理子文件夹")
	args = parser.parse_args()
	
	if args.command == "split":
		print("[INFO] 开始数据集划分...")  # 数据集划分
		splitter = DatasetSplitter(
			images_dir=args.images_dir,
			csv_path=args.csv_path,
			seed=args.seed
		)
		stats = splitter.split_dataset(
			output_dir=args.output_dir,
			class_col=args.class_col,
			train_ratio=args.train_ratio,
			test_ratio=args.test_ratio
		)
		print(f"[SUCCESS] 划分完成: {stats}")
		
	else:
		parser.print_help()