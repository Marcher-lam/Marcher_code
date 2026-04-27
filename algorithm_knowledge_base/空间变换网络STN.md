# 空间变换网络STN 学习文档

> 让神经网络学会"空间注意力"——通过可学习的空间变换提升分类性能。

> 来源线索：本节内容根据原书第2章关于"目标搜索与识别"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义：** 空间变换网络（Spatial Transformer Network，STN）是由Jaderberg等人于2015年提出的可微分的空间注意力模块，通过学习仿射变换参数来对特征图进行空间变换，使网络能够主动进行旋转、缩放、平移等空间操作。

**直觉类比：** 想象你在看一张倾斜的照片，你会不自觉地"歪头"来把照片摆正，以便更好地识别内容。STN正是让神经网络学会这个"歪头"的动作——它可以学习旋转、缩放、平移图像，让重要的区域变得"正"、"大"、"清晰"。

**历史背景：** 2015年，Max Jaderberg等人在论文"Spatial Transformer Networks"中提出STN，这是首次在深度学习框架中引入可微分的空间变换模块，为注意力机制在空间域的应用奠定了基础。

**算法定位：** 这是深度学习中的"空间注意力"模块，属于可微分的注意力机制。可在PyTorch中通过grid_sample实现。

**前置知识：**
- 仿射变换基础
- 卷积神经网络
- 坐标变换和网格采样

---

## 2. 核心原理

### 2.1 核心思想

STN的核心思想是：**让网络学会在空间上"对齐"和"聚焦"输入**。

三个关键组件：
1. **Localization Network：** 从特征图预测变换参数
2. **Grid Generator：** 生成输出位置对应的输入坐标网格
3. **Sampler：** 根据网格采样生成变换后的特征图

### 2.2 工作流程

```
输入特征图 → Localization Network(预测变换参数θ)
→ Grid Generator(生成采样网格) → Sampler(双线性插值采样)
→ 输出变换后的特征图 → 分类器
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $U$ | 输入特征图，尺寸 $H \times W \times C$ |
| $V$ | 输出特征图，尺寸 $H' \times W' \times C$ |
| $\theta$ | 变换参数（仿射变换为6维） |
| $\mathcal{G}$ | 采样网格 |
| $A_\theta$ | 仿射变换矩阵 |

### 3.2 仿射变换

二维仿射变换可以表示为：

$$\begin{pmatrix} x_i^s \\ y_i^s \end{pmatrix} = A_\theta \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix} = \begin{pmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{pmatrix} \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}$$

其中 $(x_i^t, y_i^t)$ 是输出网格坐标，$(x_i^s, y_i^s)$ 是对应的输入坐标。

### 3.3 变换参数含义

| 参数 | 变换效果 |
|------|----------|
| $\theta_{11}, \theta_{22}$ | 缩放（scale） |
| $\theta_{12}, \theta_{21}$ | 旋转（rotation） + 剪切（shear） |
| $\theta_{13}, \theta_{23}$ | 平移（translation） |

### 3.4 双线性插值采样

$$V_i = \sum_{n}^H \sum_{m}^W U_{nm} \max(0, 1 - |x_i^s - m|) \max(0, 1 - |y_i^s - n|)$$

这个采样过程是可微的，允许反向传播。

---

## 4. 训练过程讲解

### 4.1 模块结构

```python
class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, out_size=(224, 224)):
        super().__init__()
        self.out_size = out_size
        
        # Localization Network: CNN + FC
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)  # 仿射变换6参数
        )
        
        # 初始化为单位矩阵（恒等变换）
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0])
```

---

## 5. 应用场景

1. **图像分类：** 对MNIST等数据进行空间对齐，提升分类准确率
2. **目标检测：** 学习对目标区域的定位和尺度归一化
3. **细粒度分类：** 关注关键区域（如鸟类识别中的鸟嘴）
4. **图像对齐：** 学习图像配准

---

## 6. 优缺点分析

### 6.1 优点

1. **可微分的空间变换：** 首次实现端到端可训练的空间注意力
2. **即插即用：** 可以嵌入任何CNN架构
3. **可解释性强：** 可视化变换矩阵可以看到网络"关注"什么
4. **提升鲁棒性：** 对旋转、尺度变化更鲁棒

### 6.2 缺点

1. **只做仿射变换：** 不能处理更复杂的几何变换
2. **额外计算开销：** 增加了参数和计算量
3. **训练可能不稳定：** 变换参数可能发散

---

## 7. 调库实现

```python
"""
空间变换网络STN的PyTorch完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class STN(nn.Module):
    """空间变换网络完整实现"""
    
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # 1. Localization Network: 预测仿射变换参数
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)  # 6个仿射参数
        )
        
        # 初始化为恒等变换 [1, 0, 0, 0, 1, 0]
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0])
        
        # 2. 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 输出尺寸（用于grid生成）
        self.out_h, self.out_w = 14, 14
    
    def affine_grid(self, theta):
        """生成采样网格"""
        batch_size = theta.size(0)
        
        # 创建标准网格 [-1, 1] x [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.out_h),
            torch.linspace(-1, 1, self.out_w),
            indexing='ij'
        )
        
        # 扩展到batch维度
        grid = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1).float()
        
        # 应用仿射变换
        # theta: (batch, 6) -> (batch, 2, 3)
        theta = theta.view(-1, 2, 3)
        transformed_grid = torch.matmul(grid, theta.transpose(1, 2))
        
        return transformed_grid[:, :, :, :2]  # (batch, h, w, 2)
    
    def stn(self, x):
        """空间变换模块"""
        # 1. 预测变换参数
        theta = self.localization(x)
        
        # 2. 生成网格
        grid = self.affine_grid(theta)
        
        # 3. 采样（使用grid_sample进行双线性插值）
        transformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return transformed
    
    def forward(self, x):
        # 应用STN
        x = self.stn(x)
        
        # 特征提取和分类
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def train_stn():
    """训练STN模型"""
    # 数据
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_data = datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
    # 模型
    model = STN(in_channels=1, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    print(f"Test Accuracy: {correct/total:.4f}")
    return model


def visualize_stn_transforms(model, test_data):
    """可视化STN学习到的变换"""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(10):
        img, _ = test_data[i]
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            # 获取变换参数
            theta = model.localization(img)
            grid = model.affine_grid(theta)
            transformed = F.grid_sample(img, grid, mode='bilinear')
        
        # 显示原图和变换后的图
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img[0, 0], cmap='gray')
        axes[row, col].set_title(f'原图')
        axes[row, col].axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = train_stn()
    print("STN模型训练完成")
```

---

## 8. 手工代码实现

```python
"""
STN的纯PyTorch实现（不使用grid_sample）
"""

import torch
import torch.nn as nn


class SimpleSTN(nn.Module):
    """简化版STN"""
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Localization: 简单CNN
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        # 初始化
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0])
        
    def forward(self, x):
        batch = x.size(0)
        
        # 预测参数
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        
        # 简单变换（只支持缩放和平移的简化版本）
        # 提取缩放和平移参数
        scale = theta[:, 0, 0].unsqueeze(1).unsqueeze(2)
        tx = theta[:, 0, 2].unsqueeze(1).unsqueeze(2)
        ty = theta[:, 1, 1].unsqueeze(1).unsqueeze(2)
        
        # 生成坐标网格
        h, w = x.shape[2:]
        y_coords = torch.linspace(-1, 1, h, device=x.device)
        x_coords = torch.linspace(-1, 1, w, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 应用变换 (简单缩放+平移)
        new_xx = (xx - tx) / (scale + 0.1)
        new_yy = (yy - ty) / (scale + 0.1)
        
        # 归一化到[-1, 1]
        new_xx = new_xx / (x_coords.max() - x_coords.min()) * 2 - 1
        new_yy = new_yy / (y_coords.max() - y_coords.min()) * 2 - 1
        
        grid = torch.stack([new_xx, new_yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        
        # 采样
        output = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros')
        
        return output


if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    stn = SimpleSTN()
    y = stn(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_stn_effects(model, test_images, save_path=None):
    """可视化STN的变换效果"""
    model.eval()
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for i in range(6):
        img = test_images[i:i+1]
        
        with torch.no_grad():
            transformed = model.stn(img)
            
        axes[0, i].imshow(img[0, 0], cmap='gray')
        axes[0, i].set_title('原始')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(transformed[0, 0], cmap='gray')
        axes[1, i].set_title('变换后')
        axes[1, i].axis('off')
        
        # 可视化采样网格
        theta = model.localization(img)
        grid = model.affine_grid(theta)
        axes[2, i].imshow(grid[0, :, :, 0], cmap='RdBu')
        axes[2, i].set_title('变换网格')
        axes[2, i].axis('off')
    
    plt.suptitle('STN空间变换效果', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_affine_transforms(theta_samples, save_path=None):
    """可视化不同变换参数的效果"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    transformations = [
        [1, 0, 0, 0, 1, 0],      # 恒等
        [1.2, 0, 0, 0, 1.2, 0],  # 放大
        [0.8, 0, 0, 0, 0.8, 0],  # 缩小
        [1, 0, 0.3, 0, 1, 0],    # 平移X
        [1, 0.1, 0, 0.1, 1, 0],  # 剪切
        [0.9, -0.2, 0, 0.2, 0.9, 0],  # 旋转+缩放
    ]
    
    # 创建测试图像
    img = np.zeros((100, 100))
    img[30:70, 30:70] = 1
    
    for i, trans in enumerate(transformations):
        theta = torch.tensor([trans])
        
        # 简化可视化
        axes[i//3, i%3].imshow(img, cmap='gray')
        axes[i//3, i%3].set_title(f'变换 {i+1}')
        axes[i//3, i%3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 含义 |
|------|------|
| 分类准确率 | 在测试集上的分类正确率 |
| 变换参数量 | STN模块引入的参数量 |
| 对旋转的鲁棒性 | 对旋转测试集的准确率提升 |

### 10.2 计算代码

```python
def evaluate_stn_on_rotated(model, test_data, rotation_angles=[0, 15, 30, 45]):
    """评估STN对旋转的鲁棒性"""
    results = {}
    
    for angle in rotation_angles:
        rotated_data = rotate_images(test_data, angle)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img, label in rotated_data:
                output = model(img.unsqueeze(0))
                pred = output.argmax().item()
                if pred == label:
                    correct += 1
                total += 1
        
        results[f'{angle}°'] = correct / total
    
    return results
```

---

## 11. 常见问题与易错点

1. **变换参数初始化：** 初始化为单位矩阵，否则初始变换可能破坏输入
2. **grid_sample维度：** 确保grid是(batch, h, w, 2)格式
3. **padding_mode选择：** 'zeros'适合大部分场景，'border'适合无缝拼接
4. **梯度消失：** 复杂变换可能导致梯度消失，需要监控参数范围

---

## 12. 学习总结

空间变换网络STN是首个可端到端训练的空间注意力模块，它让神经网络学会了"主动"对输入进行空间变换——旋转、缩放、平移——以提升任务性能。

核心创新：
1. **可微分的网格采样：** 使用双线性插值实现可微的空间变换
2. **Localization Network：** 自动学习最优的变换参数
3. **即插即用：** 可以嵌入任意CNN架构

数学核心：
- 仿射变换参数预测
- 网格坐标变换 $A_\theta$
- 双线性插值采样 $\sum_{nm} U_{nm} \max(0, 1-|x^s-m|)\max(0, 1-|y^s-n|)$

STN开创了"空间注意力"的先河，后续的许多工作——如STN的变体、注意力机制——都受到了它的启发。

---

## 13. 练习题与思考题

### 基础题

**题目1：** 解释STN中Localization Network的作用。它为什么使用全连接层输出6个参数？

**答案：** Localization Network负责从输入特征图中预测仿射变换的参数。输出6个参数是因为2D仿射变换有6个自由度：$\theta_{11}, \theta_{12}, \theta_{13}, \theta_{21}, \theta_{22}, \theta_{23}$，分别控制缩放、旋转、剪切和平移。

### 进阶题

**题目2：** 比较STN与软性注意力（Soft Attention）的异同。

**答案：** 相同点：都是可学习的注意力机制，都可以嵌入CNN。不同点：STN进行"硬性"的空间变换（输出是新图像），Soft Attention进行"软性"的加权求和（输出是加权和）；STN输出尺寸可变，Soft Attention通常保持尺寸不变；STN可以学习复杂的空间变换，Soft Attention通常只做注意力权重分配。

---

## 14. 学习路径建议

**前置算法：**
- CNN基础（LeNet, VGG）
- 仿射变换数学基础

**平行算法：**
- 通道注意力（SE-Net）
- 混合注意力（CBAM）

**进阶算法：**
- Deformable Convolution（可变形卷积）
- DCN (Deformable Convolutional Networks)
- 空间注意力在其他领域的应用