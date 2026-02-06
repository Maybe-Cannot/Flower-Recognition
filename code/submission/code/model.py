import torch
import torch.nn as nn
from torchvision import models

#  创建ConvNeXt模型    num_classes: 分类的类别数    model: 配置好的模型
def create_model(num_classes):
    # 加载预训练模型
    try:
        # 尝试使用新版的加载方式（'weights'）
        model_ft = models.convnext_base(weights='DEFAULT')  # 或者选择具体的权重，如 IMAGENET1K_V1
        print("使用新版的weights参数")
    except TypeError:
        # 如果旧版本的torchvision（没有'weights'参数），使用旧版方式（'pretrained'）
        model_ft = models.convnext_base(pretrained=True)
        print("使用旧版pretrained参数")
    
    # 冻结所有参数
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层以适应我们的分类任务
    num_ftrs = model_ft.classifier[2].in_features
    model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
    return model_ft

# 加载已训练的模型
# model_path: 模型权重文件路径 num_classes: 分类的类别数 device: 设备
def load_model(model_path, num_classes, device):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
