# 计算机视觉基础

计算机视觉（Computer Vision）使计算机能够理解和分析视觉信息。

## 🎯 学习目标

### 1. 基础任务
- **图像分类**：识别图像中的主要对象
- **目标检测**：定位和识别多个对象
- **语义分割**：像素级分类
- **实例分割**：区分同一类别的不同实例

### 2. 核心技术
- 卷积神经网络（CNN）
- 经典架构（ResNet、EfficientNet、ViT）
- 迁移学习
- 数据增强

### 3. 高级应用
- **目标跟踪**
- **图像生成**（GAN、Diffusion）
- **姿态估计**
- **3D视觉**
- **视频理解**

## 📚 主要架构

### 经典CNN架构
| 模型 | 年份 | 主要特点 |
|------|------|---------|
| LeNet | 1998 | 最早的CNN |
| AlexNet | 2012 | 深度学习突破 |
| VGG | 2014 | 简单有效 |
| GoogLeNet | 2014 | Inception模块 |
| ResNet | 2015 | 残差连接 |
| DenseNet | 2017 | 密集连接 |
| EfficientNet | 2019 | 复合缩放 |
| ViT | 2020 | Vision Transformer |

## 🛠️ 技术栈

```python
# 基础库
import cv2
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

# 深度学习
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 目标检测
from detectron2.engine import DefaultPredictor

# 图像分割
from segmentation_models_pytorch import Unet
```

## 📖 学习资源

### 书籍
- 《Computer Vision: Algorithms and Applications》- Szeliski
- 《深度学习视觉》

### 课程
- Stanford CS231n: CNN for Visual Recognition
- MIT 6.819: Dynamic Computer Vision

### 数据集
- ImageNet
- COCO（目标检测、分割）
- PASCAL VOC
- Open Images

## 💡 实践项目

### 初级
- [ ] 图像分类（CIFAR-10、ImageNet）
- [ ] 数据增强实验
- [ ] 迁移学习

### 中级
- [ ] 目标检测（YOLO、Faster R-CNN）
- [ ] 人脸识别
- [ ] OCR文字识别

### 高级
- [ ] 图像分割（U-Net、Mask R-CNN）
- [ ] 风格迁移
- [ ] 图像生成（GAN、Diffusion）
- [ ] 视频分析

## 🔗 核心概念

### 卷积操作
```
输入图像 → 卷积核 → 特征图
```

### 池化层
```
最大池化：保留最强特征
平均池化：保留平均信息
```

### 残差连接
```
x → [F(x)] → + → ReLU → output
  ↑            ↑
  └────────────┘
```

### 注意力机制
```
特征图 → Attention → 加权特征
```

## 📝 学习路径

```
1. 图像基础（OpenCV）
   ↓
2. CNN基础
   ↓
3. 经典架构（ResNet）
   ↓
4. 迁移学习
   ↓
5. 目标检测
   ↓
6. 图像分割
   ↓
7. Vision Transformer
   ↓
8. 实际项目
```

## 💻 编程实践

### 标准流程
1. 数据加载和增强
2. 模型选择/构建
3. 损失函数和优化器
4. 训练和验证
5. 测试和部署

### 数据增强
```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### 迁移学习
```python
# 加载预训练模型
model = models.resnet50(pretrained=True)

# 冻结早期层
for param in model.parameters():
    param.requires_grad = False

# 替换最后的分类层
model.fc = nn.Linear(2048, num_classes)
```

## 🔧 常用工具

- **OpenCV**：图像处理基础
- **Albumentations**：高级数据增强
- **Detectron2**：Facebook的目标检测框架
- **MMDetection**：商汤的检测工具箱
- **Segment Anything Model (SAM)**：Meta的分割模型
