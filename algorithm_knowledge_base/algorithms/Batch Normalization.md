# Batch Normalization 学习文档

> 深度学习训练成功的关键技术，让神经网络训练更加稳定

---

## 1. 算法基础认知

**一句话定义**：Batch Normalization（批量归一化，简称BatchNorm）是由Google研究员Sergey Ioffe和Christian Szegedy于2015年提出的深度学习技术，通过对每一批次的数据进行均值归零和方差归一化，再学习额外的缩放和平移参数，让神经网络训练更加稳定高效。

**直觉类比**：Batch Normalization就像给深层神经网络"喝杯咖啡提提神"。想象一个长跑运动员跑了很久之后开始"状态下滑"（深层网络训练中的内部协变量偏移——Internal Covariate Shift），每层的输入分布不断变化，导致训练困难。BatchNorm的做法是：每跑一段距离（每个batch），就让大家重新站好队、调整好呼吸（归一化），然后再继续跑。这样每个人都保持在最佳状态，整个团队（网络）就能更稳定、更快地前进。

**历史背景**：
- 2015年，Ioffe和Szegedy在论文"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"中首次提出
- 瞬间成为深度学习最重要的技术创新之一
- 后续发展出Layer Normalization、Instance Normalization、Group Normalization等变体

**核心定位**：
- 类型：深度学习 → 训练技巧
- 作用：加速训练、稳定收敛
- 模型类型：归一化技术

**前置知识**：
- [必备]：深度学习基础（神经网络、反向传播）
- [必备]：概率统计基础（均值、方差）
- [推荐]：优化方法（SGD、Adam）

---

## 2. 核心原理

### 2.1 深度学习的"内部协变量偏移"问题

在深层神经网络中，这个问题很严重：

```
输入 x → 第一层 → 第一层输出 → 第二层输入
                                  ↓
                              分布偏移！
                                  ↓
                             第二层需要适应新的输入分布
```

**问题**：
- 每层参数都在变，导致输入分布不断变化
- 前面层的小变化会被放大
- 底层参数变化影响顶层
- 用较大的学习率会导致训练不稳定
- 用太小的学习率会导致训练太慢

### 2.2 Batch Normalization的核心思想

**核心创新**：在每个mini-batch上对激活进行归一化！

```
原始数据 x        BatchNorm(x)
     ↓                   ↓
均值 = μ          均值 = 0
标准差 = σ        标准差 = 1
     ↓                   ↓
学习 α, β        学习更好的表示
```

**关键步骤**：

1. **计算均值**：$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$

2. **计算方差**：$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$

3. **归一化**：$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

4. **缩放和平移**：$y_i = \gamma \hat{x}_i + \beta$

### 2.3 可学习参数

**为什么需要$\gamma$和$\beta$？**

归一化后数据可能不是最优分布！让网络自己学习最佳分布：

- $\gamma$：缩放因子（标准差的倍数）
- $\beta$：平移因子（均值的偏移）

如果$\gamma = \sigma_B, \beta = \mu_B$，就恢复原始数据。

### 2.4 Inference阶段

训练时有batch，推理时只有单个样本！

**解决方案**：使用训练时累积的移动平均：

$$\mu_{final} = momentum \cdot mu_{previous} + (1-momentum) \cdot mu_{current}$$

$$\sigma_{final}^2 = momentum \cdot \sigma_{previous}^2 + (1-momentum) \cdot \sigma_{current}^2$$

---

## 3. 数学公式与推导

### 3.1 BatchNorm前向传���

**对于单个神经元**：

输入：$B = \{x_1, ..., x_m\}$（m个样本的batch）

1. **均值**：
$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$

2. **方差**：
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

3. **归一化**：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

其中$\epsilon = 1e-8$，防止除零。

4. **输出**：
$$y_i = \gamma \hat{x}_i + \beta$$

### 3.2 向量化版本

如果输入是向量/矩阵：

```python
# 伪代码
def batch_norm(x, gamma, beta, eps=1e-8):
    # x: [batch, features]
    
    # 均值
    mu = x.mean(dim=0)  # [features]
    
    # 方差
    var = x.var(dim=0)  # [features]
    
    # 归一化
    x_norm = (x - mu) / sqrt(var + eps)
    
    # 缩放平移
    y = gamma * x_norm + beta
    
    return y
```

### 3.3 BatchNorm在CNN中

**对于卷积层**：

- 在通道维度上计算均值和方差
- 空间位置共享同一套$\gamma$和$\beta$

```python
# 输入: [batch, channels, height, width]
# 输出: [batch, channels, height, width]
```

### 3.4 反向传播

**梯度计算**（链式法则）：

$$\frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \cdot \gamma$$

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot (-\frac{1}{2})(\sigma^2 + \epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial \mu} = \sum_{i=1}^m \frac{\partial L}{\partial \hat{x}}_i \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}}_i \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{m} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m}$$

### 3.5 推理公式

**固定均值和方差**：

$$y = \gamma \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + \beta$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       输入数据 batch
           │
           ▼
    ┌───────────────┐
    │  计算均值   │ ← μ_B
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  计算方差   │ ← σ²_B
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  归一化     │ ← (x-μ)/σ
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  线性变换   │ ← γx̂ + β
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  激活函数   │ ← ReLU等
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  下一层    │
    └───────────────┘
```

### 4.2 参数更新

**BatchNorm参数**：
- $\gamma$：缩放，随梯度更新
- $\beta$：平移，随梯度更新
- $\mu_{running}$：移动平均的均值（无需反向传播）
- $\sigma^2_{running}$：移动平均的方差（无需反向传播）

**更新公式**：

```python
# 训练时
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

### 4.3 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| momentum | 0.1 | 移动平均动量 |
| eps | 1e-5 | 防止除零 |
| affine | True | 是否学���γ和β |
| track_running_stats | True | 是否追踪移动平均 |

### 4.4 训练技巧

| 技巧 | 说明 |
|------|------|
| 较大的batch | 更稳定的统计 |
| 较大的学习率 | BatchNorm允许 |
| 去掉dropout | BatchNorm有正则化效果 |
| 去掉L2正则 | BatchNorm不需要 |

---

## 5. 应用场景

### 5.1 CNN图像分类

几乎所有现代CNN都使用BatchNorm：

```python
# 典型结构
Conv2d -> BatchNorm2d -> ReLU -> Pooling
```

### 5.2 全连接网络

```python
# NLP / MLP
Linear -> BatchNorm1d -> ReLU
```

### 5.3 RNN/LSTM

RNN中可以使用BatchNorm：

```python
# 在时间步应用
rncell = RNNCell(input_size, hidden_size)
bn = BatchNorm1d(hidden_size)
```

### 5.4 GAN生成

GAN中也常用BatchNorm稳定训练：

```python
# 生成器
ConvTranspose2d -> BatchNorm2d -> ReLU
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **训练加速** | 可以用更大的学习率 |
| **稳定收敛** | 减少梯度消失/爆炸 |
| **正则化** | 类似dropout的效果 |
| **自动初始化** | 对初始化不敏感 |
| **鲁棒** | 对超参数不敏感 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **batch依赖** | 小batch效果差 |
| **不适合RNN** | 序列长度变化 |
| **推理额外步骤** | 需要移动平均 |
| **不适用于分散小batch** | 分布式训练问题 |

### 6.3 改进方案

| 方案 | 说明 |
|------|------|
| LayerNorm | 不依赖batch |
| InstanceNorm | 单样本归一化 |
| GroupNorm | 分组归一化 |
| SyncBatchNorm | 分布式BatchNorm |

---

## 7. 调库实现

### 7.1 PyTorch实现（最常用）

```python
import torch
import torch.nn as nn

# CNN中的BatchNorm
conv = nn.Conv2d(3, 64, 3, padding=1)
bn = nn.BatchNorm2d(64)

# 网络中使用
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# 训练
model = CNN()
x = torch.randn(32, 3, 32, 32)
out = model(x)
print(out.shape)  # torch.Size([32, 64, 32, 32])
```

### 7.2 Keras/TensorFlow实现

```python
# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
])
```

### 7.3 手动实现BatchNorm

```python
import torch
import torch.nn as nn


class BatchNorm1dManual(nn.Module):
    """手动实现BatchNorm1d"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 推理统计量
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training = True
        
    def forward(self, x):
        if self.training:
            # 计算batch统计量
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 使用推理统计量
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放平移
        return self.gamma * x_norm + self.beta


class BatchNorm2dManual(nn.Module):
    """手动实现BatchNorm2d"""
    
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        
        # 推理统计量
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))
        
    def forward(self, x):
        # 保持训练状态
        if self.training:
            # 计算统计量 [channels]
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 形状调整用于广播
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放平移
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        return gamma * x_norm + beta


# 测试
if __name__ == "__main__":
    # 测试1d
    bn1d = BatchNorm1dManual(10)
    x = torch.randn(32, 10)
    out = bn1d(x)
    print(f"BatchNorm1d: {x.shape} -> {out.shape}")
    
    # 测试2d
    bn2d = BatchNorm2dManual(64)
    x = torch.randn(32, 64, 32, 32)
    out = bn2d(x)
    print(f"BatchNorm2d: {x.shape} -> {out.shape}")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
import torch
import torch.nn as nn
import numpy as np


class BatchNormLayer(nn.Module):
    """Batch Normalization层的完整实现"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数：gamma (scale) 和 beta (shift)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # 非可学习参数：用于推理的移动平均
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
    def _check_input(self, x):
        """检查输入维度"""
        raise NotImplementedError
        
    def forward(self, x):
        if self.training:
            # 训练模式：使用batch统计量
            return self._forward_train(x)
        else:
            # 推理模式：使用累积统计量
            return self._forward_eval(x)
            
    def _forward_train(self, x):
        """训练时前向传播"""
        # 计算均值和方差
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        
        # 更新移动平均
        self._update_momentum_stats(mean, var)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.weight * x_norm + self.bias
    
    def _forward_eval(self, x):
        """推理时前向���播"""
        # 使用训练累积的统计量
        mean = self.running_mean
        var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.weight * x_norm + self.bias
    
    def _update_momentum_stats(self, mean, var):
        """更新移动平均"""
        if self.num_batches_tracked == 0:
            self.running_mean = mean.detach()
            self.running_var = var.detach()
        else:
            # 指数移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.detach()
            
        self.num_batches_tracked += 1
    
    def extra_repr(self):
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}'


class BatchNorm1d(BatchNormLayer):
    """1D BatchNorm（如全连接层后）"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps, momentum)
        
    def forward(self, x):
        # x: [batch, features] 或 [batch, seq, features]
        if x.dim() == 3:
            # 3D: [batch, seq, features] -> [batch*seq, features]
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)
            out = super().forward(x)
            return out.reshape(batch_size, seq_len, features)
        else:
            return super().forward(x)


class BatchNorm2d(BatchNormLayer):
    """2D BatchNorm（如卷积层后）"""
    
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__(num_channels, eps, momentum)
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        # 在维度0,2,3上计算，保留channels
        
        if self.training:
            # 计算每个channel的均值和方差
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # 更新移动平均
            self._update_momentum_stats(mean, var)
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放平移
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        return weight * x_norm + bias


def test_batch_norm():
    """测试BatchNorm"""
    print("=== BatchNorm测试 ===\n")
    
    # 测试1D
    x = torch.randn(32, 10)
    bn1d = BatchNorm1d(10)
    
    # 切换到训练模式
    bn1d.train()
    out = bn1d(x)
    print(f"1D: train - {x.shape} -> {out.shape}")
    print(f"  均值: {out.mean(dim=0).abs().max().item():.4f} (应接近0)")
    print(f"  标准差: {out.std(dim=0).mean().item():.4f} (应接近1)")
    
    # 切换到推理模式
    bn1d.eval()
    out = bn1d(x)
    print(f"1D: eval - {x.shape} -> {out.shape}")
    
    # 测试2D
    x = torch.randn(8, 64, 32, 32)
    bn2d = BatchNorm2d(64)
    bn2d.train()
    out = bn2d(x)
    print(f"2D: {x.shape} -> {out.shape}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_batch_norm()
```

---

## 9. 可视化与结果理解

### 9.1 训练曲线对比

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_training_comparison():
    """对比有/无BatchNorm的训练曲线"""
    
    epochs = list(range(50))
    loss_with_bn = np.exp(-np.array(epochs) / 10) + np.random.randn(50) * 0.02
    loss_without_bn = np.exp(-np.array(epochs) / 20) + np.random.randn(50) * 0.05
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_with_bn, label='With BatchNorm', linewidth=2)
    plt.plot(epochs, loss_without_bn, label='Without BatchNorm', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training: With vs Without BatchNorm')
    plt.legend()
    plt.grid(True)
    plt.savefig('bn_comparison.png')
    plt.show()


def plot_distribution_shift():
    """展示内部协变量偏移"""
    
    x = np.random.randn(1000)
    
    # 每层参数变化导致分布偏移
    layers = []
    for i in range(5):
        shift = np.random.randn() * 0.5
        scale = 1 + np.random.randn() * 0.1
        x_new = x * scale + shift
        layers.append(x_new)
        x = x_new
    
    plt.figure(figsize=(12, 4))
    for i, layer_data in enumerate(layers):
        plt.hist(layer_data, bins=50, alpha=0.5, label=f'Layer {i}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Internal Covariate Shift without BatchNorm')
    plt.legend()
    plt.savefig('shift.png')
    plt.show()


if __name__ == "__main__":
    plot_training_comparison()
```

### 9.2 梯度流可视化

```python
def visualize_gradient_flow(model):
    """可视化梯度流"""
    
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.abs().mean().item())
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(grads)), grads)
    plt.xlabel('Layer')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow')
    plt.savefig('gradients.png')
    plt.show()
```

---

## 10. 模型评估

### 10.1 效果评估指标

| 指标 | 衡量内容 |
|------|----------|
| 收敛速度 | 达到相同精度需要的epoch |
| 最终精度 | 最终测试集精度 |
| 稳定性 | 多次训练方差 |

### 10.2 Benchmark对比

| 方法 |收敛速度 | 最终精度 | 训练时间 |
|------|--------|--------|----------|
| 无Norm | 1x | 基准 | 1x |
| BatchNorm | 3x | +2% | 1.1x |
| +更大LR | 5x | +1% | 1.0x |

### 10.3 代码评估

```python
def evaluate_model(model, test_loader):
    """评估模型"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy
```

---

## 11. 常见问题与易错点

### 11.1 小batch效果差

**问题**：batch太小时统计量不稳定

**原因**：均值和方差基于少量样本，不够准确

**解决**：
- 使用GroupNorm
- 增大batch size
- 使用分布式SyncBatchNorm

### 11.2 推理模式错误

**问题**：训练和推理行为不一致

**原因**：忘记切换到eval模式

**解决**：
```python
model.eval()  # 重要！
with torch.no_grad():
    output = model(x)
```

### 11.3 移动平均未更新

**问题**：推理时归一化效果差

**原因**：训练时间太短，还没积累足够的统计量

**解决**：
- 确保训练足够多的epoch
- 或者手动设置running_mean/var

### 11.4 在RNN中使用问题

**问题**：序列长度变化导致统计量不稳定

**原因**：不同batch的序列长度可能不同

**解决**：
- 使用LayerNorm代替
- 或在每个时间步分别应用

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 批次数据的归一化 |
| 核心 | $\hat{x} = \frac{x-\mu}{\sigma}$ |
| 参数 | 可学习γ和β |
| 优势 | 加速训练、稳定收敛 |

### 12.2 公式记忆

**训练时**：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

**推理时**：
$$y = \gamma \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + \beta$$

### 12.3 扩展阅读

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| LayerNorm | 样本内归一化 | RNN/Transformer |
| InstanceNorm | 单通道归一化 | 风格迁移 |
| GroupNorm | 分组归一化 | 小batch |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：BatchNorm的核心作用是什么？

**答案**：通过对每个mini-batch的数据进行归一化，解决了深层神经网络的"内部协变量偏移"问题，使得：
- 可以使用更大的学习率
- 减少对初始化的依赖
- 有一定的正则化效果
- 训练更加稳定

**练习2**：为什么需要γ和β两个可学习参数？

**答案**：归一化后的数据分布可能不是最优的，让网络自己学习最佳的缩放和平移，如果$\gamma = \sigma, \beta = \mu$就恢复到原始分布。

**练习3**：推理时为什么不能用batch的均值和方差？

**答案**：推理时是单样本，batch只有1个样本，统计量完全不可靠，所以用训练时累积的移动平均。

### 13.2 进阶思考

**思考1**：BatchNorm能否用于推理的单样本？

**答案**：不能直接用，因为单样本的均值和方差没有意义。需要使用训练时累积的running_mean和running_var。

**思考2**：BatchNorm和LayerNorm的区别？

**答案**：
- BatchNorm：在batch维度归一化，不同样本共享统计量
- LayerNorm：在特征维度归一化，每个样本独立

**思考3**：小batch下效果差如何解决？

**答案**：使用GroupNorm（在通道分组内归一化）或InstanceNorm。

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 深度学习基础 | 理解网络结构 |
| 3-4 | 优化方法 | 理解梯度下降 |
| 5-6 | BatchNorm论文 | 理解核心思想 |
| 7 | 代码实现 | 跑通demo |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 实现细节 | 完整实现 |
| 2 | 变体学习 | LayerNorm等 |

### 14.3 实战（3周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | CNN实践 | 图像分类 |
| 2 | NLP实践 | 文本分类 |
| 3 | 项目 | 完整训练流程 |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| BatchNorm论文 | https://arxiv.org/abs/1502.03167 |
| PyTorch文档 | https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html |

### B. 参数速查

| 参数 | 默认值 |
|------|--------|
| momentum | 0.1 |
| eps | 1e-5 |
| affine | True |

---

**文档结束**