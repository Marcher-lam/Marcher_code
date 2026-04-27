# SE-Net（Squeeze-and-Excitation Networks）学习文档

> 通道注意力机制，通过学习通道间的重要性关系提升卷积网络性能。

## 1. 算法基础认知

### 一句话定义

SE-Net通过"挤压"和"激励"操作，让网络自适应地学习每个通道的重要性权重，从而强化重要特征、抑制无关特征。

### 直觉类比

就像在交响乐团中，指挥根据音乐需要决定哪些乐器声部需要加强、哪些需要减弱。SE-Net让神经网络也能"指挥"各个特征通道，决定哪些信息更重要。

### 历史背景

- **2017年**：Momenta和牛津大学提出SE-Net
- **2018年**：获得ImageNet图像分类冠军
- **后续发展**：SE模块被广泛集成到ResNet、MobileNet等架构中

### 算法定位

SE-Net是**通道注意力机制**，属于卷积神经网络的轻量级增强模块。

---

## 2. 核心原理

### 核心思想

SE模块的核心是"学习通道权重"——不改变特征图的空间结构，而是为每个通道学习一个重要性系数。

### 工作流程

1. **挤压（Squeeze）**：将空间信息压缩为全局描述
2. **激励（Excitation）**：学习通道间的非线性关系
3. **重标定**：用学到的权重调整原始特征

### 结构图

```
输入特征 → 全局池化 → 全连接→ Sigmoid → 通道权重 → 加权输出
```

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 输入特征，$H \times W \times C$ |
| $U$ | 卷积输出，$H \times W \times C$ |
| $s$ | 通道权重 |
| $\tilde{X}$ | 加权后的输出 |

### 挤压操作

$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W}u_c(i,j)$$

将每个通道的空间信息压缩为一个标量。

### 激励操作

$$s = \sigma(W_2 \delta(W_1 z))$$

- $W_1$: 降维全连接（压缩为$C/r$）
- $\delta$: ReLU激活
- $W_2$: 升维全连接（恢复到$C$）
- $\sigma$: Sigmoid激活

### 重标定

$$\tilde{x}_c = s_c \cdot u_c$$

---

## 4. 调库实现

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """SE注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class SEResNet(nn.Module):
    """带SE模块的ResNet"""
    def __init__(self, num_classes=1000):
        super(SEResNet, self).__init__()
        # 简化的ResNet结构
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.se1 = SEBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.se2 = SEBlock(128)
        
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.se1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.se2(self.conv2(x)))
        x = torch.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 测试
if __name__ == "__main__":
    model = SEBlock(64)
    x = torch.randn(1, 64, 56, 56)
    out = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
```

---

## 5. 手工代码实现

```python
import numpy as np

class NumPySEBlock:
    """纯NumPy实现的SE模块"""
    
    def __init__(self, channels, reduction=16):
        self.channels = channels
        self.reduction = reduction
        
        # 简化的参数初始化
        self.W1 = np.random.randn(channels, channels // reduction) * 0.01
        self.W2 = np.random.randn(channels // reduction, channels) * 0.01
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        batch, c, h, w = x.shape
        
        # Squeeze: 全局平均池化
        z = np.mean(x, axis=(2, 3))  # (batch, channels)
        
        # Excitation: 全连接层
        hidden = self.relu(np.dot(z, self.W1))  # (batch, c//r)
        s = self.sigmoid(np.dot(hidden, self.W2))  # (batch, c)
        
        # 重标定
        output = x * s.reshape(batch, c, 1, 1)
        
        return output

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    se = NumPySEBlock(64)
    x = np.random.randn(2, 64, 32, 32)
    out = se.forward(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
```

---

## 6. 优缺点分析

### 优点

1. **参数少**：仅增加约2%参数量
2. **提升显著**：ImageNet提升1%以上
3. **通用性强**：可嵌入任意CNN架构
4. **可解释**：可视化通道权重

### 缺点

1. **增加计算量**：额外的全连接层
2. **可能过拟合**：小数据集上效果有限
3. **顺序执行**：不能并行

---

## 7. 练习题

1. **基础**：SE模块的"挤压"操作目的是什么？
2. **进阶**：为什么SE模块比Bottleneck更有效？

---

## 8. 学习路径

- 前置：CNN基础、ResNet
- 平行：CBAM
- 进阶：ECA-Net、SRM