import csv
import os
import re
import shutil
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def sanitize_class_name(name: str) -> str:
	"""将类别名转换为更适合文件系统的名称。

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
	# 统一分隔符为下划线
	s = re.sub(r"[\\/]+", "_", s)
	s = re.sub(r"\s+", "_", s)
	# 移除 Windows 非法字符
	s = re.sub(r"[<>:\"\\/\|\?\*]", "", s)
	s = s.strip(" .")  # 去掉结尾的点和空格（Windows 不允许）
	return s or "unknown"


def read_labels_csv(csv_path: str) -> List[Dict[str, str]]:
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
	return rows


def split_indices(n: int, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
	"""Return shuffled indices split by 70/20/10.

	Returns: (train_idx, test_idx, val_idx)
	"""
	idx = list(range(n))
	rng = random.Random(seed)
	rng.shuffle(idx)

	n_train = int(n * 0.7)
	n_test = int(n * 0.2)
	n_val = n - n_train - n_test

	train_idx = idx[:n_train]
	test_idx = idx[n_train:n_train + n_test]
	val_idx = idx[n_train + n_test:]
	return train_idx, test_idx, val_idx


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str) -> None:
	ensure_dir(os.path.dirname(dst))
	shutil.copy2(src, dst)


def build_output_paths(output_root: str, split: str, class_name: str, filename: str) -> str:
	sanitized = sanitize_class_name(class_name)
	return os.path.join(output_root, split, sanitized, filename)


def run_split(
	csv_path: str = "train_labels.csv",
	images_dir: str = "train",
	output_dir: str = "dataset_cifar_like",
	class_col: str = "category_id",
	seed: int = 42,
) -> Dict[str, int]:
	"""Split dataset by subclass into CIFAR-10 like folders.

	Args:
		csv_path: CSV file path with columns [filename, category_id, chinese_name, english_name]
		images_dir: Directory containing source images (flat, filenames from CSV)
		output_dir: Output root directory to create train/test/val structure
		class_col: Which column to use for class sub-folder names. One of english_name/chinese_name/category_id
		seed: Random seed for reproducibility

	Returns:
		stats dict with counts copied per split.
	"""
	rows = read_labels_csv(csv_path)
	valid_cols = {"english_name", "chinese_name", "category_id"}
	if class_col not in valid_cols:
		raise ValueError(f"class_col 必须是 {valid_cols} 之一，当前: {class_col}")

	# group by class
	groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
	for r in rows:
		key = r.get(class_col) or r.get("english_name") or r.get("chinese_name") or str(r.get("category_id"))
		key = str(key)
		groups[key].append(r)

	# Prepare output dirs
	for split in ("train", "test", "val"):
		ensure_dir(os.path.join(output_dir, split))

	stats = {"train": 0, "test": 0, "val": 0, "missing": 0}
	missing_files: List[str] = []

	for class_name, items in groups.items():
		n = len(items)
		if n == 0:
			continue
		train_idx, test_idx, val_idx = split_indices(n, seed=seed)

		def items_from(indices: List[int]):
			for i in indices:
				yield items[i]

		for split_name, indices in ("train", train_idx), ("test", test_idx), ("val", val_idx):
			for row in items_from(indices):
				filename = row["filename"].strip()
				src_path = os.path.join(images_dir, filename)
				if not os.path.isfile(src_path):
					stats["missing"] += 1
					missing_files.append(src_path)
					continue
				dst_path = build_output_paths(output_dir, split_name, class_name, filename)
				copy_file(src_path, dst_path)
				stats[split_name] += 1

	# Save a report
	report_path = os.path.join(output_dir, "split_report.txt")
	with open(report_path, "w", encoding="utf-8") as f:
		total = sum(stats[s] for s in ("train", "test", "val"))
		f.write("数据集划分报告\n")
		f.write(f"CSV: {os.path.abspath(csv_path)}\n")
		f.write(f"源图像目录: {os.path.abspath(images_dir)}\n")
		f.write(f"输出目录: {os.path.abspath(output_dir)}\n")
		f.write(f"类别列: {class_col}\n")
		f.write(f"随机种子: {seed}\n")
		f.write("\n计数:\n")
		for k in ("train", "test", "val", "missing"):
			f.write(f"{k}: {stats[k]}\n")
		f.write(f"总计(已复制): {total}\n")
		if missing_files:
			f.write("\n缺失文件列表:\n")
			for m in missing_files:
				f.write(m + "\n")

	return stats


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="按子类 7:2:1 划分为 CIFAR-10 风格目录结构")
	parser.add_argument("--csv", dest="csv_path", default="train_labels.csv", help="标注 CSV 路径")
	parser.add_argument("--images_dir", default="train", help="源图像所在目录（CSV 中 filename 对应此目录下的文件）")
	parser.add_argument("--output_dir", default="flowerme", help="输出根目录，将创建 train/test/val 子目录")
	parser.add_argument("--class_col", choices=["english_name", "chinese_name", "category_id"], default="category_id", help="作为类别子目录名的列")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")

	args = parser.parse_args()
	stats = run_split(
		csv_path=args.csv_path,
		images_dir=args.images_dir,
		output_dir=args.output_dir,
		class_col=args.class_col,
		seed=args.seed,
	)
	print("划分完成:", stats)


