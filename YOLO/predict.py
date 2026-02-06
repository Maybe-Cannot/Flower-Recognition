import sys
import os
import pandas as pd
from ultralytics import YOLO

def main():
    model_path = r"best_model.pt"
    # 读取命令行参数
    img_dir = sys.argv[1]
    save_path = sys.argv[2]

    # 确认图片路径存在
    if not os.path.exists(img_dir):
        sys.exit(1)

    # 加载模型
    model = YOLO(model_path)
    # 构建图片通配符路径
    source = os.path.join(img_dir, "*.jpg")
    # 运行预测
    results = model.predict(source, verbose=False)
    # 收集结果
    records = []
    for r in results:
        name = os.path.basename(r.path)
        class_id = r.probs.top1
        class_name = r.names[class_id]
        conf = float(r.probs.top1conf)
        records.append({
            "img_name": name,
            "predicted_class": class_name,
            "confidence": round(conf, 2)
        })

    # 保存结果到 CSV
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()



