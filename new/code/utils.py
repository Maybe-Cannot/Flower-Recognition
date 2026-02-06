# 通用工具函数模块
# 包含数据集处理、文件操作等工具函数


import csv
import os
import re
import shutil
import random
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

# 保存配置文件为JSON
def save_config(config: dict, filepath: str) -> None:
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(config, f, indent=4, ensure_ascii=False)
	print(f"[INFO] 配置已保存: {filepath}")
# 加载JSON配置文件
def load_config(filepath: str) -> dict:
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"配置文件不存在: {filepath}")
	with open(filepath, 'r', encoding='utf-8') as f:
		config = json.load(f)
	print(f"[INFO] 配置已加载: {filepath}")
	return config

# 设置随机种子以确保结果可重现
def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	print(f"[INFO] 随机种子已设置: {seed}")

# 将类别名转换为更适合文件系统的名称
def sanitize_class_name(name: str) -> str:
	"""
	规则：
	- 去除首尾空白
	- 将斜杠、反斜杠、多个空白替换为单个下划线
	- 移除 Windows 不允许的字符 <>:"/\|?*
	- 保留中文及其他 Unicode 字符
	- 若结果为空，则回退为 "unknown"
	"""
	if name is None:
		return "unknown"
	s = str(name).strip()
	s = re.sub(r"[\\/]+", "_", s)  # 统一分隔符为下划线
	s = re.sub(r"\s+", "_", s)
	s = re.sub(r"[<>:\"\\/\|\?\*]", "", s)  # 移除 Windows 非法字符
	s = s.strip(" .")  # 去掉结尾的点和空格
	return s or "unknown"

# 确保目录存在
def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)
# 复制文件到目标位置
def copy_file(src: str, dst: str) -> None:
	ensure_dir(os.path.dirname(dst))
	shutil.copy2(src, dst)

# 读取标签CSV文件
def read_csv_labels(csv_path: str) -> List[Dict[str, str]]:
	"""
	Args:		csv_path: CSV文件路径
	Returns:	包含所有行数据的列表
	"""
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
	
	rows: List[Dict[str, str]] = []
	with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		required_cols = {"filename", "category_id", "chinese_name", "english_name"}
		missing = required_cols - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"CSV 缺少必要列: {missing}. 实际列: {reader.fieldnames}")
		for row in reader:
			rows.append(row)
	
	if not rows:
		raise ValueError("CSV 内无数据行")
	print(f"[INFO] 成功读取 {len(rows)} 行数据")
	return rows

# 按类别分组数据
def get_class_mapping(rows: List[Dict[str, str]], class_col: str = "category_id") -> Dict[str, List[Dict[str, str]]]:
	"""
	Args:
		rows: CSV数据行列表
		class_col: 用作类别的列名
	Returns:		按类别分组的字典
	"""
	valid_cols = {"english_name", "chinese_name", "category_id"}
	if class_col not in valid_cols:
		raise ValueError(f"class_col 必须是 {valid_cols} 之一，当前: {class_col}")
	
	groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
	for r in rows:
		key = r.get(class_col) or r.get("english_name") or r.get("chinese_name") or str(r.get("category_id"))
		key = str(key)
		groups[key].append(r)
	print(f"[INFO] 共分为 {len(groups)} 个类别")
	return groups
# 划分索引为train/test/val
def split_indices(n: int, train_ratio: float = 0.7, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
	"""
	Args:
		n: 总样本数
		train_ratio: 训练集比例
		test_ratio: 测试集比例
		seed: 随机种子
	Returns:
		(train_idx, test_idx, val_idx)
	"""
	idx = list(range(n))
	rng = random.Random(seed)
	rng.shuffle(idx)

	n_train = int(n * train_ratio)
	n_test = int(n * test_ratio)
	n_val = n - n_train - n_test
	train_idx = idx[:n_train]
	test_idx = idx[n_train:n_train + n_test]
	val_idx = idx[n_train + n_test:]
	return train_idx, test_idx, val_idx

# 统计目录中的图片文件数量
def count_images_in_directory(directory: str, extensions: Optional[set] = None) -> int:
	"""
	Args:
		directory: 目录路径
		extensions: 图片扩展名集合
	Returns:	图片文件数量
	"""
	if extensions is None:
		extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
	count = 0
	if os.path.exists(directory):
		for root, dirs, files in os.walk(directory):
			for file in files:
				if os.path.splitext(file)[1].lower() in extensions:
					count += 1
	return count

# 获取数据集统计信息
def get_dataset_statistics(data_dir: str) -> Dict[str, int]:
	"""
	Args:		data_dir: 数据集根目录
	Returns:	包含各个子集图片数量的字典
	"""
	stats = {}
	for split in ['train', 'val', 'test']:
		split_dir = os.path.join(data_dir, split)
		if os.path.exists(split_dir):
			stats[split] = count_images_in_directory(split_dir)
		else:
			stats[split] = 0
	stats['total'] = sum(stats.values())
	return stats

# 验证数据集目录结构
def validate_dataset_structure(data_dir: str, required_splits: List[str] = None) -> bool:
	"""
	Args:
		data_dir: 数据集根目录
		required_splits: 必需的子集列表（默认为 ['train', 'val']）
	Returns:	是否符合要求
	"""
	if required_splits is None:
		required_splits = ['train', 'val']
	if not os.path.exists(data_dir):
		print(f"[ERROR] 数据集目录不存在: {data_dir}")
		return False
	for split in required_splits:
		split_dir = os.path.join(data_dir, split)
		if not os.path.exists(split_dir):
			print(f"[ERROR] 缺少必需的子目录: {split_dir}")
			return False
	print(f"[INFO] 数据集结构验证通过")
	return True

# 从目录结构中获取类别名称
def get_class_names_from_directory(directory: str) -> List[str]:
	"""Args: directory: 包含类别子目录的目录	Returns: 类别名称列表"""
	if not os.path.exists(directory):
		return []
	class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
	class_names.sort()
	return class_names

# 打印数据集信息
def print_dataset_info(data_dir: str) -> None:
	print("\n" + "=" * 60)
	print("数据集信息:")
	print(f"  数据集路径: {os.path.abspath(data_dir)}")
	stats = get_dataset_statistics(data_dir)
	for split, count in stats.items():
		if split != 'total':
			print(f"  {split:10s}: {count:6d} 张图片")
	print(f"  {'total':10s}: {stats['total']:6d} 张图片")
	train_dir = os.path.join(data_dir, 'train')
	if os.path.exists(train_dir):
		class_names = get_class_names_from_directory(train_dir)
		print(f"  类别数量: {len(class_names)}")
	print("=" * 60 + "\n")


# ==================== 预测相关工具函数 ====================
def check_has_subfolders(test_dir: str) -> Tuple[bool, List]:
	"""检查目录是否包含子文件夹（用于判断是否有标签）"""
	from pathlib import Path
	test_path = Path(test_dir)
	subfolders = [d for d in test_path.iterdir() if d.is_dir()]
	return len(subfolders) > 0, subfolders


def collect_images_with_labels(test_dir: str) -> Dict[str, Dict[str, str]]:
	"""
	从带有子文件夹的测试目录收集图片及其真实标签
	
	返回:
	dict: {filename: {'path': full_path, 'true_label': folder_name}}
	"""
	from pathlib import Path
	image_label_map = {}
	img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
	
	test_path = Path(test_dir)
	
	# 遍历所有子文件夹
	for subfolder in test_path.iterdir():
		if not subfolder.is_dir():
			continue
		
		folder_name = subfolder.name
		
		# 遍历子文件夹中的图片
		for img_file in subfolder.iterdir():
			if img_file.suffix.lower() in img_extensions:
				filename = img_file.name
				image_label_map[filename] = {
					'path': str(img_file),
					'true_label': folder_name
				}
	
	return image_label_map


def evaluate_predictions(predictions_df, image_label_map: Dict, output_dir: str) -> float:
	"""
	评估预测结果并生成报告
	
	参数:
	predictions_df: 预测结果DataFrame
	image_label_map: 图片真实标签映射
	output_dir: 报告输出目录
	
	返回:
	float: 整体准确率
	"""
	print("\n" + "=" * 60)
	print("开始评估预测结果")
	print("=" * 60)
	
	# 统计信息
	total_samples = len(predictions_df)
	correct_predictions = 0
	class_correct = defaultdict(int)
	class_total = defaultdict(int)
	confusion_matrix = defaultdict(lambda: defaultdict(int))
	
	# 逐个样本评估
	for _, row in predictions_df.iterrows():
		filename = row['filename']
		pred_label = str(row['category_id'])
		
		if filename not in image_label_map:
			print(f"[WARNING] 文件 {filename} 未在测试集中找到")
			continue
		
		true_label = image_label_map[filename]['true_label']
		
		# 统计总数
		class_total[true_label] += 1
		
		# 判断是否正确
		if pred_label == true_label:
			correct_predictions += 1
			class_correct[true_label] += 1
		
		# 更新混淆矩阵
		confusion_matrix[true_label][pred_label] += 1
	
	# 计算整体准确率
	overall_acc = correct_predictions / total_samples if total_samples > 0 else 0
	
	# 打印结果
	print(f"\n整体评估结果:")
	print(f"  总样本数: {total_samples}")
	print(f"  正确预测: {correct_predictions}")
	print(f"  整体准确率 (ACC): {overall_acc:.4f} ({overall_acc*100:.2f}%)")
	
	# 按类别统计
	print(f"\n各类别准确率:")
	class_acc_list = []
	for true_label in sorted(class_total.keys(), key=lambda x: int(x) if x.isdigit() else x):
		total = class_total[true_label]
		correct = class_correct[true_label]
		acc = correct / total if total > 0 else 0
		class_acc_list.append({
			'class': true_label,
			'total': total,
			'correct': correct,
			'accuracy': acc
		})
		print(f"  类别 {true_label:>4s}: {correct:>4d}/{total:>4d} = {acc:.4f} ({acc*100:.2f}%)")
	
	# 输出错分情况
	print(f"\n错分情况分析（仅显示有错误的类别）:")
	error_details = []
	for true_label in sorted(confusion_matrix.keys(), key=lambda x: int(x) if x.isdigit() else x):
		errors = []
		for pred_label, count in confusion_matrix[true_label].items():
			if pred_label != true_label:
				errors.append((pred_label, count))
		
		if errors:
			errors.sort(key=lambda x: x[1], reverse=True)
			total_errors = sum(e[1] for e in errors)
			print(f"  类别 {true_label} (错误 {total_errors} 个):")
			for pred_label, count in errors[:5]:  # 显示前5个最常见的错分
				print(f"    -> 误判为 {pred_label}: {count} 次")
			
			error_details.append({
				'true_class': true_label,
				'total_errors': total_errors,
				'errors': errors
			})
	
	# 保存详细报告
	report_path = os.path.join(output_dir, 'evaluation_report.txt')
	with open(report_path, 'w', encoding='utf-8') as f:
		f.write("=" * 60 + "\n")
		f.write("花卉分类模型评估报告\n")
		f.write("=" * 60 + "\n\n")
		
		f.write(f"整体评估结果:\n")
		f.write(f"  总样本数: {total_samples}\n")
		f.write(f"  正确预测: {correct_predictions}\n")
		f.write(f"  整体准确率 (ACC): {overall_acc:.4f} ({overall_acc*100:.2f}%)\n\n")
		
		f.write(f"各类别准确率:\n")
		for item in class_acc_list:
			f.write(f"  类别 {item['class']:>4s}: {item['correct']:>4d}/{item['total']:>4d} = {item['accuracy']:.4f} ({item['accuracy']*100:.2f}%)\n")
		
		f.write(f"\n详细错分情况:\n")
		for detail in error_details:
			f.write(f"\n  类别 {detail['true_class']} (总错误: {detail['total_errors']}):\n")
			for pred_label, count in detail['errors']:
				f.write(f"    -> 误判为 {pred_label}: {count} 次\n")
	
	print(f"\n[INFO] 详细评估报告已保存: {report_path}")
	print("=" * 60 + "\n")
	
	return overall_acc


# ==================== 训练相关工具函数 ====================
def handle_dataset_split(raw_data_dir: str, csv_path: str, model_dir: str, seed: int = 2025) -> Tuple[str, Dict]:
	"""
	处理数据集划分逻辑
	
	参数:
	raw_data_dir: 原始图片目录
	csv_path: 标签CSV文件路径
	model_dir: 模型保存目录
	seed: 随机种子
	
	返回:
	(data_dir, stats): 划分后的数据集目录和统计信息
	"""
	from model import DatasetSplitter
	
	print("=" * 60)
	print("数据集划分模式")
	print(f"  原始图片目录: {raw_data_dir}")
	print(f"  标签CSV文件: {csv_path}")
	print("=" * 60)
	
	if not os.path.exists(raw_data_dir):
		print(f"[ERROR] 原始图片目录不存在: {raw_data_dir}")
		return None, {}
	if not os.path.exists(csv_path):
		print(f"[ERROR] 标签CSV文件不存在: {csv_path}")
		return None, {}
	
	# 数据集划分输出目录（保存在model目录下）
	split_output_dir = os.path.join(model_dir, 'dataset')
	
	# 检查数据集是否已经划分
	train_exists = os.path.exists(os.path.join(split_output_dir, 'train'))
	val_exists = os.path.exists(os.path.join(split_output_dir, 'val'))
	test_exists = os.path.exists(os.path.join(split_output_dir, 'test'))
	
	if train_exists and val_exists:
		# 数据集已存在，验证是否有效
		print(f"\n[INFO] 检测到已存在的数据集: {split_output_dir}")
		if validate_dataset_structure(split_output_dir, ['train', 'val']):
			# 统计现有数据集
			train_count = sum([len(files) for _, _, files in os.walk(os.path.join(split_output_dir, 'train'))])
			val_count = sum([len(files) for _, _, files in os.walk(os.path.join(split_output_dir, 'val'))])
			test_count = sum([len(files) for _, _, files in os.walk(os.path.join(split_output_dir, 'test'))]) if test_exists else 0
			
			print(f"[INFO] 数据集已划分:")
			print(f"  训练集: {train_count} 张")
			print(f"  验证集: {val_count} 张")
			if test_exists:
				print(f"  测试集: {test_count} 张")
			print(f"[INFO] 跳过数据集划分，直接使用现有数据集")
			
			stats = {'train': train_count, 'val': val_count, 'test': test_count, 'missing': 0}
			return split_output_dir, stats
		else:
			print(f"[WARNING] 现有数据集结构不正确，将重新划分")
	
	# 执行数据集划分
	print(f"\n[INFO] 开始划分数据集...")
	print(f"[INFO] 输出目录: {split_output_dir}")
	
	splitter = DatasetSplitter(
		images_dir=raw_data_dir,
		csv_path=csv_path,
		seed=seed
	)
	stats = splitter.split_dataset(
		output_dir=split_output_dir,
		class_col='category_id',
		train_ratio=0.7,
		test_ratio=0.2
	)
	
	print(f"[SUCCESS] 数据集划分完成!")
	print(f"  训练集: {stats['train']} 张")
	print(f"  验证集: {stats['val']} 张")
	print(f"  测试集: {stats['test']} 张")
	if stats['missing'] > 0:
		print(f"  [WARNING] 缺失文件: {stats['missing']} 个")
	
	return split_output_dir, stats
