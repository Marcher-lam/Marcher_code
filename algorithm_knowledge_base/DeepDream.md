# DeepDream 学习文档

> 让神经网络"做梦"——可视化 CNN 学到的特征。

> 来源线索：本节内容根据原书中关于"DeepDream"的相关章节（第6章）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** DeepDream 通过反向传播将输入图像的梯度放大并加回原图，使 CNN 学到的特征在图像中"显现"，创造出梦幻般的视觉效果。

**直觉类比：** 想象你看云彩时"看到"了人脸——大脑把随机的形状解读成熟悉的模式。DeepDream 让 CNN 做同样的事：在图像中"寻找"它学过的模式（眼睛、羽毛、建筑），然后把它们放大、叠加到图像上。层数越深，看到的模式越复杂。

**历史背景：** DeepDream 由 Google 工程师 Alexander Mordvintsev 于 2015 年发布，最初用于理解 CNN 内部表征，后成为 AI 艺术的先驱工具。

**算法定位：** 特征可视化、AI 艺术、CNN 解释工具。

**前置知识：** CNN、反向传播、梯度上升、PyTorch。

---

## 2. 核心原理

### 核心思想

选择 CNN 某一层的某个通道，通过**梯度上升**修改输入图像，使该通道的激活值最大化。这相当于问："什么样的输入能让这个神经元最兴奋？"

### 工作流程

1. 前向传播图像到目标层
2. 计算目标通道激活值的梯度（对输入图像）
3. 将梯度加回输入图像（梯度上升）
4. 重复多轮，图像逐渐"充满"目标特征
5. 每几轮做一次高斯模糊（平滑效果）

### 关键概念

- **梯度上升（Gradient Ascent）**：与训练相反，修改输入而非权重
- **多尺度处理**：在不同分辨率上重复增强，产生更丰富的细节
- **层选择**：浅层产生简单纹理，深层产生复杂结构

---

## 3. 数学公式

### 梯度上升

$$x_{t+1} = x_t + \eta \cdot \frac{\partial a_k(x)}{\partial x}$$

其中 $a_k(x)$ 是第 $k$ 个通道的激活值，$\eta$ 是学习率（步长）。

### 多尺度处理

$$x = \text{Upsample}(\text{Dream}(\text{Downsample}(x, s)))$$

在不同尺度 $s$ 下重复做梦过程，然后上采样回原始分辨率叠加。

---

## 4-5. 训练与应用

### 应用场景
1. **AI 艺术**：生成超现实主义图像
2. **特征可视化**：理解 CNN 学到了什么
3. **风格化滤镜**：照片转梦幻风格

---

## 6. 优缺点分析

### 优点
1. **揭示模型内部**：直观展示 CNN 学到的特征
2. **艺术价值**：独特的视觉效果
3. **无需训练**：直接用预训练模型

### 缺点
1. **不可控**：难以精确控制输出内容
2. **过度饱和**：多轮后图像可能过于混乱

---

## 7-8. 代码实现

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

class DeepDream:
    def __init__(self, model=None):
        self.model = model or models.vgg16(pretrained=True).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def dream(self, image_tensor, layer_idx=20, channel_idx=None,
              lr=0.01, iterations=20):
        """对单张图像执行 DeepDream"""
        img = image_tensor.clone().requires_grad_(True)

        for _ in range(iterations):
            self.model.zero_grad()
            # 提取目标层激活
            x = img.unsqueeze(0)
            for i, layer in enumerate(self.model.features):
                x = layer(x)
                if i == layer_idx:
                    if channel_idx is not None:
                        loss = x[0, channel_idx].mean()
                    else:
                        loss = x.mean()
                    loss.backward()
                    break

            gradients = img.grad.data
            img.data += lr * gradients / gradients.norm()
            img.grad.zero_()

        return img.detach()

# 测试
dreamer = DeepDream()
x = torch.randn(3, 224, 224)
result = dreamer.dream(x, layer_idx=20, iterations=10)
print(f"输入: {x.shape} → 输出: {result.shape}")
print(f"变化量: {(result - x).abs().mean():.4f}")
```

---

## 9-14. 练习与路径

**题1：** DeepDream 用梯度上升而非梯度下降，为什么？

**参考答案：** 训练 CNN 时用梯度下降最小化损失。DeepDream 的目标是**最大化**某层激活值，让特定特征在图像中更明显，所以用梯度上升。本质是在输入空间优化而非参数空间。

### 学习路径
- 前置：CNN、反向传播
- 平行：神经风格迁移
- 进阶：特征可视化（Activation Atlas）、GAN 逆映射
- 推荐：Mordvintsev et al., "Inceptionism: Going Deeper into Neural Networks" (2015)
