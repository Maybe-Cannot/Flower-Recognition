import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes):
    """
    创建ConvNeXt模型
    
    参数:
    num_classes: 分类的类别数
    
    返回:
    model: 配置好的模型
    """
    # 加载预训练模型
    try:
        weights = models.ConvNeXt_Weights.DEFAULT
        model_ft = models.convnext_base(weights=weights)
        print("使用最新的ConvNeXt预训练权重")
    except AttributeError:
        # 对于旧版本的torchvision
        model_ft = models.convnext_base(pretrained=True)
        print("使用旧版pretrained参数")
    
    # 冻结所有参数
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层以适应我们的分类任务
    num_ftrs = model_ft.classifier[2].in_features
    model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    return model_ft


def load_model(model_path, num_classes, device):
    """
    加载已训练的模型
    
    参数:
    model_path: 模型权重文件路径
    num_classes: 分类的类别数
    device: 设备(CPU或GPU)
    
    返回:
    model: 加载了权重的模型
    """
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
