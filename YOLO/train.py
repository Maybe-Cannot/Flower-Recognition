from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型（切换到更小的 yolo11n-cls.pt 以减少资源消耗）
    model = YOLO(r"yolo11m-cls.pt")  # 加载预训练模型（建议用于训练）
    # 训练模型（降低 imgsz 和 epochs，启用混合精度以节省内存）
    results = model.train(
        data="flowerme", 
        epochs=10, 
        imgsz=600, 
        device='cuda', 
        plots=False, 
        save=True, 
        half=True, 
        batch=8,
        workers = 0
        )

