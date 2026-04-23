# LeNet 学习文档

> 经典的卷积神经网络，现代 CNN 的开山之作。

---

## 1. 算法基础认知

### 1.1 发展背景

LeNet 由 Yann LeCun 等人在 1998 年论文《Gradient-Based Learning Applied to Document Recognition》中提出，是世界上第一个商用卷积神经网络，用于手写数字识别（MNIST 数据集）。LeNet 奠定了现代 CNN 的基础架构。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 经典 CNN |
| 层数 | 7 层（含输入） |
| 参数 | 约 60K |
| 任务 | 手写数字识别 |

### 1.3 网络结构

```
Input(32×32) → C1(6@5×5) → S2(下采样) → C3(16@5×5) → S4 → C5(120) → F6(84) → Output(10)
     ↓              ↓              ↓           ↓          ↓          ↓
   卷积层        池化层        卷积层       池化层       全连接
```

---

## 2. 核心原理

### 2.1 卷积层

使用 5×5 卷积核提取特征：

$$f(x) = \sigma(W * x + b)$$

### 2.2 池化层

2×2 平均池化：

$$y_{i,j} = \frac{1}{4} \sum_{m=0}^1 \sum_{n=0}^1 x_{i+m, j+n}$$

### 2.3 全连接层

Softmax 分类输出

---

## 3. 数学公式与推导

### 3.1 卷积计算

设输入 $I$，卷积核 $K$：

$$(I * K)_{i,j} = \sum_m \sum_n I_{m,n} \cdot K_{i-m, j-n}$$

### 3.2 梯度反向传播

$$\frac{\partial L}{\partial W^{(l)}} = \sum \delta^{(l)} * \text{rot180}(a^{(l-1)})$$

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| 批量大小 | 50-200 |
| 学习率 | 0.0001-0.01 |
| 优化器 | SGD |
| Epochs | 10-50 |

### 4.2 训练流程

1. 初始化权重
2. 前向传播计算输出
3. 计算损失
4. 反向传播更新
5. 重复直到收敛

---

## 5. 应用场景

### 5.1 典型应用

- **手写识别**：MNIST
- **邮政编码识别**：美国邮政
- **银行票据识别**

### 5.2 代码示例

```python
from torchvision import models
import torch.nn as nn

# LeNet-5
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.avg_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

## 6. 调库实现

### 6.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet-5 手写识别网络"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 第一个卷积层：1通道 → 6通道，5×5卷积
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 第二个卷积层：6通道 → 16通道
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 2)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def demo():
    print("=== LeNet 演示 ===\n")
    
    model = LeNet(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")
    
    # 模拟输入：batch=1, Channels=1, Height=32, Width=32
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")


if __name__ == "__main__":
    demo()
```

### 6.2 torchvision

```python
import torchvision.models as models

# 加载预训练 LeNet
model = models.leNet(pretrained=False)
```

---

## 7. 手工代码实现

### 7.1 完整实现

```python
import numpy as np
import torch

class ManualLeNet:
    """手动实现 LeNet-5 核心结构"""
    
    def __init__(self):
        self.weights = {}
        
    def init_weights(self):
        """初始化权重 - He初始化"""
        # Conv1: 6@5×5 → ~150 params
        self.weights['W1'] = np.random.randn(6, 1, 5, 5) * np.sqrt(2/25)
        self.weights['b1'] = np.zeros(6)
        
        # Conv3: 16@5×5 → ~1200 params  
        self.weights['W3'] = np.random.randn(16, 6, 5, 5) * np.sqrt(2/25)
        self.weights['b3'] = np.zeros(16)
        
        # FC1: 400 → 120
        self.weights['W5'] = np.random.randn(120, 400) * np.sqrt(2/400)
        self.weights['b5'] = np.zeros(120)
        
        # FC2: 120 → 84
        self.weights['W6'] = np.random.randn(84, 120) * np.sqrt(2/120)
        self.weights['b6'] = np.zeros(84)
        
        # FC3: 84 → 10
        self.weights['W_out'] = np.random.randn(10, 84) * np.sqrt(2/84)
        self.weights['b_out'] = np.zeros(10)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def conv2d(self, x, kernel, stride=1, padding=0):
        # 简化卷积
        raise NotImplementedError
        
    def avg_pool2d(self, x, kernel=2):
        # 简化平均池化
        n, c, h, w = x.shape
        new_h, new_w = h // kernel, w // kernel
        return x.reshape(n, c, new_h, kernel, new_w, kernel).mean(axis=(4, 3))
    
    def forward(self, x):
        """前向传播（简化）"""
        # 使用 PyTorch
        conv1 = torch.relu(self.conv1(x))
        pool1 = self.avg_pool2d(conv1)
        
        conv3 = torch.relu(self.conv3(pool1))
        pool4 = self.avg_pool2d(conv3)
        
        flat = pool4.view(x.size(0), -1)
        
        fc1 = self.relu(self.fc1(flat))
        fc2 = self.relu(self.fc2(fc1))
        output = self.fc3(fc2)
        
        return output


def demo_manual():
    print("=== LeNet 手工实现演示 ===\n")
    
    # 创建模型
    from torch.nn import Conv2d, Linear, ReLU, AvgPool2d, Sequential
    
    # 简化的权重初始化
    print("LeNet (1998):")
    print("  - 7 层网络")
    print("  - 约 60K 参数量")
    print("  - 用于 MNIST 手写识别")


if __name__ == "__main__":
    demo_manual()
```

---

## 8. 可视化与结果理解

### 8.1 架构可视化

```python
def visualize():
    print("""
    LeNet-5 架构:
    
    Input(1×32×32)
       ↓
    Conv1: 6@5×5 + ReLU + AvgPool
       ↓
    Conv2: 16@5×5 + ReLU + AvgPool
       ↓
    Flatten: 400
       ↓
    FC: 120 + ReLU
       ↓
    FC: 84 + ReLU  
       ↓
    FC: 10 → Output
    
    总参数: ~60,000
    """)
```

### 8.2 特征图

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_weights():
    """可视化卷积核"""
    weights = np.random.randn(6, 1, 5, 5)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        kernel = weights[i, 0]
        ax.imshow(kernel, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    plt.suptitle('LeNet 卷积核')
    plt.tight_layout()
    plt.savefig('lenet_kernels.png', dpi=150)
    plt.show()
```

---

## 9. 模型评估

### 9.1 MNIST 性能

| 模型 | 错误率 |
|------|--------|
| LeNet-5 | 0.8% |
| SVM | 1.1% |
| KNN | 3.0% |

### 9.2 各层作用

| 层 | 输出 | 作用 |
|------|------|------|
| Conv1 | 6@28×28 | 边缘检测 |
| Conv3 | 16@10×10 | 形状检测 |
| FC | 84 | 特征组合 |

---

## 10. 常见问题与易错点

### 10.1 初始化

**问题**：权重初始化不当

**解决**：使用 He 初始化

### 10.2 池化类型

**问题**：平均池化效果

**解决**：可用最大池化替代

---

## 11. 优缺点分析

### 11.1 优点

1. **结构简单**：易于理解
2. **参数少**：训练快
3. **开创性**：CNN 开山之作

### 11.2 缺点

1. **浅层**：特征提取有限
2. **过拟合**：数据增强有限

### 11.3 改进方向

- 增加网络深度
- 使用 ReLU
- Batch Normalization
- Dropout

---

## 12. 学习总结

**核心要点**：

1. **卷积-池化结构**：现代 CNN 基础
2. **5×5 卷积核**：经典设计
3. **局部连接**：减少参数
4. **权值共享**：提高效率

**LeNet 核心地位**：
- 第一个商用 CNN
- 现代 CNN 的基础
- CNN 设计的范本

**学习建议**：

1. 理解卷积原理
2. 掌握池化操作
3. 对比现代 CNN

---

## 13. 练习题与思考题

### 13.1 基础练习

1. LeNet vs 现代 CNN
2. 卷积核大小选择
3. 池化类型比较

### 13.2 进阶练习

1. MNIST 复现
2. 模型改进

### 13.3 思考题

1. LeNet 的局限性
2. CNN 的演进方向

---

### 13.4 详细答案与解析

#### 练习1：vs VGG/ResNet

**问题**：LeNet 与现代 CNN 的区别

| 方面 | LeNet | VGG/ResNet |
|------|------|------------|
| 深度 | 7 层 | 100+ 层 |
| 卷积核 | 5×5 | 3×3 |
| 归一化 | 无 | BN |
| 激活 | Sigmoid | ReLU/GELU |

---

## 14. 学习路径建议

### 入门阶段

1. 神经网络基础
2. 卷积操作
3. LeNet 原理

### 进阶阶段

1. PyTorch 实现
2. MNIST 训练
3. 模型改进

### 高级阶段

1. 现代 CNN 设计
2. 迁移学习

**推荐路线**：

```
MLP → LeNet → AlexNet → VGG → ResNet → EfficientNet
```

**LeNet 是 CNN 的开山之作，熟练掌握它对理解卷积神经网络历史很重要。**