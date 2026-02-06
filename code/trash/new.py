import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('d:/code/Flower-Recognition/code/train_labels.csv')

# 获取所有唯一类别编号，并升序排序
unique_ids = sorted(df['category_id'].unique())

# 生成编号到索引的映射字典
CLASS_ID_TO_IDX = {cid: idx for idx, cid in enumerate(unique_ids)}

print(CLASS_ID_TO_IDX)
print(f'类别总数: {len(CLASS_ID_TO_IDX)}')