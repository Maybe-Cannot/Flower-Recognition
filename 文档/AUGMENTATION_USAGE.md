# 数据增强使用说明

## ✅ 已完成的修改

所有自定义数据增强代码已集成到 `train.py` 文件中，无需额外文件。

### 修改内容：

1. **在 train.py 顶部添加了三个类**:
   - `CustomizedDataset`: 自定义数据集（包含RandAugment、ColorJitter、RandomErasing）
   - `CustomizedTrainer`: 自定义训练器
   - `CustomizedValidator`: 自定义验证器

2. **已有的数据增强参数**:
   ```bash
   --hsv_h      # HSV色调 (默认: 0.015)
   --hsv_s      # HSV饱和度 (默认: 0.7)
   --hsv_v      # HSV明度 (默认: 0.4)
   --fliplr     # 水平翻转概率 (默认: 0.5)
   --flipud     # 垂直翻转概率 (默认: 0.0)
   --erasing    # 随机擦除概率 (默认: 0.0)
   ```

3. **训练自动使用增强数据**:
   - 训练时应用: RandAugment + ColorJitter + RandomErasing + ImageNet标准化
   - 验证时仅应用: Resize + ToTensor + ImageNet标准化

---

## 🚀 使用方法

### 基础训练（默认增强）
```bash
python train.py --data_dir flowerme --epochs 50 --batch_size 64
```

### 启用随机擦除
```bash
python train.py --data_dir flowerme --erasing 0.2 --epochs 50
```

### 自定义所有增强参数
```bash
python train.py \
    --data_dir flowerme \
    --epochs 50 \
    --batch_size 64 \
    --hsv_h 0.02 \
    --hsv_s 0.8 \
    --hsv_v 0.5 \
    --fliplr 0.5 \
    --flipud 0.1 \
    --erasing 0.2
```

---

## 📊 数据增强效果

### 训练时应用:
1. Resize → 224x224
2. RandomHorizontalFlip → 50%
3. RandomVerticalFlip → 0%（可调）
4. **RandAugment** → 自动应用2个随机增强
5. **ColorJitter** → HSV色彩调整
6. ToTensor
7. **Normalize** → ImageNet标准化
8. **RandomErasing** → 随机擦除（可选）

### 验证时应用:
1. Resize → 224x224
2. ToTensor
3. Normalize → ImageNet标准化

---

## ⚙️ 代码结构

整个增强逻辑都在 `train.py` 中:

```python
# train.py 结构
├── 导入 (torch, torchvision.transforms, ultralytics)
├── CustomizedDataset 类 (自定义数据增强)
├── CustomizedTrainer 类 (使用自定义数据集)
├── CustomizedValidator 类 (验证用)
├── parse_args() (包含增强参数)
├── prepare_training_config() (传递增强参数)
├── save_training_config() (保存增强配置到JSON)
└── main() (使用 trainer=CustomizedTrainer)
```

---

## 💡 注意事项

1. **ImageNet标准化**: 使用标准的 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
2. **RandAugment**: 每次随机应用2个增强操作，强度为9
3. **验证集不增强**: 保证评估的公正性
4. **所有参数自动保存**: 训练完成后在 `config.json` 中可查看

---

## 🧪 快速验证

查看 train.py 前65行，确认看到:
```python
class CustomizedDataset(ClassificationDataset):
    """自定义分类数据集，增强数据增强功能"""
    ...
```

运行测试:
```bash
python train.py --help
# 应该能看到 --hsv_h, --hsv_s, --hsv_v, --fliplr, --flipud, --erasing 参数
```

---

## ✨ 对比原始YOLO

| 特性 | 原始YOLO | 当前版本 |
|-----|---------|---------|
| 基础几何变换 | ✅ | ✅ |
| RandAugment | ❌ | ✅ |
| ColorJitter (HSV) | 简单 | ✅ 增强 |
| RandomErasing | ❌ | ✅ |
| ImageNet标准化 | ❌ | ✅ |
| 可配置参数 | 有限 | ✅ 完全可控 |

现在可以直接运行训练了！🎉
