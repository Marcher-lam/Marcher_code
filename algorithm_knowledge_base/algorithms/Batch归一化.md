# Batch归一化学习文档

## 1. 算法基础认知

Batch归一化（Batch Normalization, BatchNorm）是由Sergey Ioffe和Christian Szegedy于2015年提出的深度学习核心技术之一。这篇论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》彻底改变了深度学习的训练方式，使得训练更深更复杂的神经网络成为可能。

### 1.1 什么是Internal Covariate Shift？

在深度神经网络中，由于前面的层参数不断变化，导致后面层接收到的输入分布也在不断变化。这种现象称为"内部协变量偏移"（Internal Covariate Shift, ICS）。具体来说：
- 当前面的层更新参数后，其输出分布发生变化
- 后面层需要不断适应这种新的输入分布
- 这导致深层网络训练困难，需要很小的学习率和谨慎的参数初始化

### 1.2 BatchNorm的核心思想

BatchNorm的核心思想非常优雅：**在神经网络的每一层，对输入进行归一化，使其均值为0、方差为1**。但这还不够，论文作者进一步引入了两个可学习参数$\gamma$和$\beta$，使得网络可以学习最适合的分布：

$$y = \gamma \cdot \hat{x} + \beta$$

其中$\hat{x}$是标准化后的值，$\gamma$和$\beta$是可学习的缩放和偏移参数。

### 1.3 为什么BatchNorm如此重要？

BatchNorm的贡献包括：
1. **加速训练**：使网络可以使用更大的学习率
2. **减轻对初始化的依赖**：网络对参数初始化不再那么敏感
3. **正则化效果**：具有一定的正则化作用，减少过拟合
4. **允许更深的网络**：使得训练上百层的网络成为可能

## 2. 核心原理

### 2.1 BatchNorm的数学定义

对于一个mini-batch $\mathcal{B} = \{x_1, x_2, ..., x_m\}$，BatchNorm的操作如下：

**步骤1：计算mini-batch均值**
$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**步骤2：计算mini-batch方差**
$$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**步骤3：归一化**
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

其中$\epsilon$（通常为$1e-8$）是一个小常数，防止除零错误。

**步骤4：线性变换**
$$y_i = \gamma \hat{x}_i + \beta$$

其中$\gamma$和$\beta$是可学习参数，分别用于缩放和偏移。

### 2.2 为什么需要gamma和beta？

如果只是简单的归一化（均值为0、方差为1），那么网络的表达能力可能会受到限制。引入$\gamma$和$\beta$后：
- 如果$\gamma = \sigma, \beta = \mu$，可以恢复原始输入分布
- 网络可以通过学习找到最优的归一化程度
- 这增加了网络的灵活性

### 2.3 训练阶段vs推理阶段

BatchNorm的行为在训练和推理时完全不同，这是容易混淆的点：

**训练阶段（training mode）**：
- 使用当前mini-batch的均值和方差进行归一化
- 同时计算移动平均的均值和方差

**推理阶段（eval mode）**：
- 使用训练阶段累积的移动平均均值和方差
- 不再计算新的统计量

这是因为推理时通常处理单个样本，没有mini-batch的概念。

### 2.4 移动平均（Moving Average）

在训练过程中，BatchNorm维护两个移动平均统计量：
- 移动平均均值：$E[x]_{new} = momentum \cdot E[x]_{old} + (1 - momentum) \cdot \mu_{\mathcal{B}}$
- 移动平均方差：$Var[x]_{new} = momentum \cdot Var[x]_{old} + (1 - momentum) \cdot \sigma_{\mathcal{B}}^2$

其中momentum（动量）通常设为0.1或0.9（即momentum=0.9表示90%历史 + 10%当前）。

## 3. 数学公式与推导

### 3.1 BatchNorm完整算法

```
Algorithm: Batch Normalization
---------------------------------
Input: mini-batch B = {x1, x2, ..., xm}, 
       learned parameters γ, β
       epsilon ε (small constant)

Training (for each mini-batch):
    // Step 1: Compute batch statistics
    μ_B = (1/m) * Σ xi                    // batch mean
    σ_B² = (1/m) * Σ (xi - μ_B)²         // batch variance
    
    // Step 2: Normalize
    x̂_i = (xi - μ_B) / √(σ_B² + ε)
    
    // Step 3: Scale and shift
    y_i = γ * x̂_i + β
    
    // Step 4: Update moving averages (for inference)
    E[x] = momentum * E[x] + (1-momentum) * μ_B
    Var[x] = momentum * Var[x] + (1-momentum) * σ_B²

Inference (for a single input x):
    // Use precomputed moving averages
    x̂ = (x - E[x]) / √(Var[x] + ε)
    y = γ * x̂ + β
```

### 3.2 梯度计算

BatchNorm的反向传播同样重要。假设损失为$L$，对输入$x$的梯度为：

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y_i} \cdot \gamma \cdot \frac{1}{\sqrt{sigma_B^2 + epsilon}}$$

对$\gamma$和$\beta$的梯度：
$$\frac{\partial L}{\partial gamma} = sum_i \frac{\partial L}{\partial y_i} \cdot hat{x}_i$$
$$\frac{\partial L}{\partial beta} = sum_i \frac{\partial L}{\partial y_i}$$

### 3.3 推理时的计算公式

在推理阶段，BatchNorm使用预计算的统计量：

$$\text{BN}(x) = gamma \cdot \frac{x - E[x]}{sqrt{Var[x] + epsilon}} + beta$$

这个公式可以重写为：
$$\text{BN}(x) = \frac{gamma}{sqrt{Var[x] + epsilon}} \cdot x + \left( beta - \frac{gamma \cdot E[x]}{sqrt{Var[x] + epsilon}} \right)$$

即一个线性变换：$w \cdot x + b$，可以融合到前一层中。

## 4. 训练过程讲解

### 4.1 BatchNorm在网络中的位置

BatchNorm通常放在全连接层或卷积层之后，激活函数之前：

```
... → Linear/Conv → BatchNorm → Activation → ...
```

这个顺序很重要！如果没有BatchNorm，激活函数（如ReLU）可能会接收到较大的输入，导致梯度问题。

### 4.2 训练流程

```python
# 训练循环中的BatchNorm
for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 前向传播（BatchNorm使用batch统计）
        output = model(batch)
        loss = loss_fn(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
    
    # 每个epoch结束后更新移动平均
    # （PyTorch自动处理）
```

### 4.3 推理流程

```python
# 推理时
model.eval()  # 切换到推理模式

with torch.no_grad():
    for batch in test_loader:
        # BatchNorm使用预计算的移动平均
        output = model(batch)
```

### 4.4 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| momentum | 0.1 | 移动平均的动量 |
| eps | 1e-5 | 防止除零的小常数 |
| affine | True | 是否使用gamma和beta |

## 5. 应用场景

### 5.1 典型应用

**图像分类**：
- ResNet、VGG等网络的核心组件
- 显著加速训练

**目标检测**：
- YOLO、SSD等使用BatchNorm
- 提高检测精度

**分割网络**：
- U-Net使用BatchNorm
- 稳定训练

**Transformer**：
- LayerNorm（在现代架构中更多用LayerNorm）
- 但思想一脉相承

### 5.2 PyTorch中的BatchNorm

```python
import torch.nn as nn

# 1D BatchNorm（用于全连接层）
bn1d = nn.BatchNorm1d(num_features)

# 2D BatchNorm（用于卷积层，特征图）
bn2d = nn.BatchNorm2d(num_channels)

# 3D BatchNorm（用于3D卷积）
bn3d = nn.BatchNorm3d(num_features)

# 使用示���
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm在激活函数前
        x = self.relu(x)
        x = self.pool(x)
        return x
```

### 5.3 BatchNorm变体

| 变体 | 说明 | 使用场景 |
|------|------|----------|
| BatchNorm1d | 1D特征，FC层后 | NLP、tabular |
| BatchNorm2d | 2D特征图，Conv层后 | 图像 |
| BatchNorm3d | 3D特征，3D卷积后 | 视频、医学影像 |
| SyncBatchNorm | 多GPU同步 | 分布式训练 |
| GhostBatchNorm | 小batch训练 | GAN、小batch |

## 6. 优缺点分析

### 6.1 优点

1. **加速收敛**：可以使用更大的学习率
2. **减少敏感度**：对参数初始化不敏感
3. **正则化**：减少过拟合（副作用）
4. **允许深层网络**：训练100+层网络
5. **梯度流动更好**：缓解梯度消失

### 6.2 缺点

1. ** batch依赖**：小batch时不稳定
2. ** RNN问题**：序列长度变化时不方便
3. **训练推理不一致**：行为不同
4. **计算开销**：额外的均值方差计算

### 6.3 何时使用BatchNorm

**推荐使用**：
- 图像分类/检测
- 全连接网络
- 大batch训练

**谨慎使用**：
- RNN/LSTM（用LayerNorm）
- GAN（小batch，用GhostBatchNorm）
- 小数据集（正则化过强）

## 7. 调库实现（Python + PyTorch）

### 7.1 基础使用

```python
import torch
import torch.nn as nn

# 定义网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv -> BN -> ReLU -> Pool 是经典模式
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 训练
model = CNN().to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(20):
    model.train()  # 关键：切换到训练模式
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(model.device)
        batch_y = batch_y.to(model.device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()  # 切换到推理模式
    with torch.no_grad():
        correct = 0
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
        print(f"Epoch {epoch+1}: Accuracy = {correct/len(val_dataset)}")
```

### 7.2 使用SyncBatchNorm进行分布式训练

```python
import torch.nn as nn
import torch.distributed as dist

# 将BatchNorm转换为SyncBatchNorm
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

# 在分布式环境中
if torch.distributed.is_available():
    model = nn.SyncBatchNorm.convert_sync_batch_norm(model)

# 使用DataParallel或DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### 7.3 查看BatchNorm的统计量

```python
# 查看BatchNorm的参数和统计量
def print_bn_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"\n{name}:")
            print(f"  weight (γ): {module.weight.data[:5]}")
            print(f"  bias (β): {module.bias.data[:5]}")
            print(f"  running_mean: {module.running_mean[:5]}")
            print(f"  running_var: {module.running_var[:5]}")
            print(f"  num_batches_tracked: {module.num_batches_tracked}")

model = CNN()
print_bn_stats(model)
```

### 7.4 自定义BatchNorm层

```python
class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # 移动平均（非可学习）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # 计算batch统计
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # 使用移动平均
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        
        return x
```

### 7.5 将BatchNorm与卷积融合

```python
# 推理时，可以将BatchNorm融合到卷积层中
def fuse_conv_bn(conv, bn):
    """将卷积层和BatchNorm融合为一个卷积层"""
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).to(conv.weight.device)
    
    # 融合权重
    w_conv = conv.weight
    w_bn = bn.weight.view(-1, 1, 1, 1)
    fused_conv.weight = nn.Parameter(w_conv * w_bn)
    
    # 融合偏置
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels).to(conv.weight.device)
    
    b_bn = bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_conv.bias = nn.Parameter(b_conv * w_bn + b_bn)
    
    return fused_conv

# 使用示例
fused = fuse_conv_bn(model.conv1, model.bn1)
```

## 8. 手工代码实现（核心算法手写）

### 8.1 完整的BatchNorm实现

```python
import numpy as np

class BatchNorm1D:
    """
    BatchNorm1D的手工实现
    
    适用于全连接层的归一化
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = np.ones(num_features)  # 缩放
        self.beta = np.zeros(num_features)  # 偏移
        
        # 移动平均（推理时使用）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 训练/推理模式
        self.training = True
    
    def forward(self, x):
        """
        前向传播
        
        x: (batch_size, num_features)
        """
        if self.training:
            # 计算batch统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 使用移动平均
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和偏移
        output = self.gamma * x_normalized + self.beta
        
        return output
    
    def backward(self, grad_output, x):
        """
        反向传播
        
        grad_output: (batch_size, num_features)
        返回: 对输入的梯度
        """
        batch_size = x.shape[0]
        
        if self.training:
            # 计算batch统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 对gamma和beta的梯度
        grad_gamma = np.sum(grad_output * x_normalized, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        
        # 对输入的梯度
        grad_x = grad_output * self.gamma / np.sqrt(var + self.eps)
        
        return grad_x, grad_gamma, grad_beta
    
    def set_mode(self, training):
        """切换训练/推理模式"""
        self.training = training
```

### 8.2 2D BatchNorm实现

```python
class BatchNorm2D:
    """
    BatchNorm2D的手工实现
    
    适用于卷积层的归一化，对特征图进行归一化
    """
    
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        
        # 移动平均
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        
        self.training = True
    
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        """
        if self.training:
            # 在(batch, h, w)维度计算均值和方差
            mean = np.mean(x, axis=(0, 2, 3))
            var = np.var(x, axis=(0, 2, 3))
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # reshape以便广播
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        
        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和偏移
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        
        return gamma * x_normalized + beta
```

### 8.3 推理模式与训练模式的区别示例

```python
def demonstrate_train_vs_eval():
    """演示训练和推理模式的区别"""
    import numpy as np
    
    np.random.seed(42)
    
    # 创建BatchNorm层
    bn = BatchNorm2D(num_channels=3)
    bn.training = False  # 推理模式
    
    # 模拟训练数据
    train_data = np.random.randn(32, 3, 32, 32)
    
    # 模拟测试数据
    test_data = np.random.randn(1, 3, 32, 32)
    
    print("训练模式统计量更新:")
    bn.training = True
    for i in range(5):
        batch = train_data[i:i+1]
        output = bn.forward(batch)
        print(f"  Step {i+1}: running_mean[0] = {bn.running_mean[0]:.4f}")
    
    print("\n推理模式（使用移动平均）:")
    bn.training = False
    output = bn.forward(test_data)
    print(f"  使用running_mean[0] = {bn.running_mean[0]:.4f}")
    print(f"  使用running_var[0] = {bn.running_var[0]:.4f}")
```

## 9. 可视化与结果理解

### 9.1 BatchNorm对训练的影响可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_batchnorm_effect():
    """演示BatchNorm对训练的影响"""
    np.random.seed(42)
    
    # 模拟没有BatchNorm的训练
    losses_no_bn = []
    loss = 100
    for _ in range(100):
        loss = loss * 0.98 + np.random.randn() * 0.5
        losses_no_bn.append(loss)
    
    # 模拟有BatchNorm的训练
    losses_with_bn = []
    loss = 100
    for _ in range(100):
        loss = loss * 0.95 + np.random.randn() * 0.2
        losses_with_bn.append(loss)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses_no_bn, label='Without BatchNorm')
    plt.plot(losses_with_bn, label='With BatchNorm')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_no_bn, label='Without BN', alpha=0.7)
    plt.plot(losses_with_bn, label='With BN', alpha=0.7)
    plt.ylim(0, 50)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (zoomed)')
    plt.title('Loss (Zoomed In)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('batchnorm_effect.png', dpi=150)
    
    print("图表已保存至 batchnorm_effect.png")
```

### 9.2 分布变化可视化

```python
def plot_distribution_shift():
    """可视化BatchNorm对输入分布的稳定化"""
    np.random.seed(42)
    
    # 模拟没有BatchNorm时各层的输入分布变化
    means = []
    stds = []
    for i in range(10):
        # 每一层的输入分布（不断变化）
        data = np.random.randn(1000) * (2 - i * 0.15) + i * 0.2
        means.append(np.mean(data))
        stds.append(np.std(data))
    
    # 有BatchNorm时
    bn_means = [0] * 10
    bn_stds = [1] * 10
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(means, 'b-o', label='Without BatchNorm')
    plt.plot(bn_means, 'r--', label='With BatchNorm')
    plt.xlabel('Layer')
    plt.ylabel('Mean')
    plt.title('Mean Across Layers')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(stds, 'b-o', label='Without BatchNorm')
    plt.plot(bn_stds, 'r--', label='With BatchNorm')
    plt.xlabel('Layer')
    plt.ylabel('Std')
    plt.title('Std Across Layers')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('distribution_shift.png', dpi=150)
```

## 10. 模型评估

### 10.1 BatchNorm层的效果评估

评估BatchNorm的效果可以通过以下指标：

1. **训练速度**：收敛所需的epoch数
2. **最终精度**：验证集上的准确率
3. **学习率敏感性**：不同学习率下的表现差异

### 10.2 统计量检查

```python
def evaluate_batchnorm_stats(model):
    """评估BatchNorm层的统计量质量"""
    results = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # 检查running_mean接近0
            mean_close_to_zero = np.allclose(module.running_mean.numpy(), 0, atol=0.1)
            # 检查running_var接近1
            var_close_to_one = np.allclose(module.running_var.numpy(), 1, atol=0.1)
            
            results.append({
                'name': name,
                'running_mean': module.running_mean[:3],
                'running_var': module.running_var[:3],
                'mean_ok': mean_close_to_zero,
                'var_ok': var_close_to_one
            })
    
    return results
```

## 11. 常见问题与易错点

### 11.1 忘记切换模式

**错误**：训练后直接推理，没有调用`model.eval()`
**正确**：推理前调用`model.eval()`

### 11.2 batch_size太小

**错误**：batch_size=1或2时使用BatchNorm
**正确**：batch_size至少为16-32

### 11.3 推理时仍更新统计量

**错误**：推理时忘记切换模式，导致移动平均被修改
**正确**：使用`model.eval()`切换模式

### 11.4 与其他归一化方法混淆

BatchNorm vs LayerNorm vs InstanceNorm：
- **BatchNorm**：对batch维度归一化，适合CNN
- **LayerNorm**：对特征维度归一化，适合RNN/Transformer
- **InstanceNorm**：对每个样本的每个通道归一化，适合风格迁移

## 12. 学习总结

### 核心要点

1. **归一化位置**：全连接层/卷积层之后，激活函数之前
2. **可学习参数**：gamma（缩放）和beta（偏移）
3. **训练/推理差异**：训练时用batch统计，推理时用移动平均
4. **移动平均**：momentum参数控制历史信息的保留比例

### 关键公式

训练阶段：
$$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

推理阶段：
$$y = \gamma \frac{x - E[x]}{sqrt{Var[x] + epsilon}} + \beta$$

## 13. 练习题与思考题（含答案）

### 练习题

**Q1**: BatchNorm在训练阶段和推理阶段使用的统计量有什么不同？

**答案**：
- 训练阶段：使用当前mini-batch的均值和方差，并对移动平均进行更新
- 推理阶段：使用训练时累积的移动平均均值和方差，不更新统计量

**Q2**: 为��么BatchNorm通常放在激活函数之前而非之后？

**答案**：
- 如果放在激活函数之后，归一化会破坏激活函数的非线性效果
- 放在激活函数之前，确保输入到激活函数的值分布合理

**Q3**: BatchNorm的gamma和beta参数的作用是什么？

**答案**：
- gamma控制归一化后的缩放，增加网络表达能力
- beta控制归一化后的偏移，允许网络学习最优分布

**Q4**: 小batch_size时BatchNorm为什么不稳定？

**答案**：小batch时，batch统计量（均值、方差）的估计噪声大，不具有代表性，导致归一化效果差。

**Q5**: 何时使用LayerNorm而非BatchNorm？

**答案**：
- RNN/LSTM等序列模型
- batch_size经常变化的情况
- Transformer类模型

### 思考题

**Q1**: BatchNorm能否用于推理时的单样本？

**答案**：可以。推理时使用训练阶段累积的移动平均统计量，不依赖batch。

**Q2**: BatchNorm有哪些变体？它们分别适用于什么场景？

**答案**：
- SyncBatchNorm：分布式训练
- GhostBatchNorm：小batch的GAN训练
- LayerNorm：Transformer、RNN
- InstanceNorm：风格迁移

**Q3**: 如何将训练好的BatchNorm模型转换为不使用BatchNorm？

**答案**：可以将BatchNorm的参数吸收到前一层中，融合成一个等效的线性变换。

## 14. 学习路径建议

### 基础阶段

1. 理解协变量偏移（Internal Covariate Shift）的概念
2. 学习BatchNorm的论文
3. 理解均值、方差的计算

### 进阶阶段

1. 对比训练和推理的行为差异
2. 学习其他归一化方法（LayerNorm, InstanceNorm）
3. 理解移动平均的工作原理

### 实践阶段

1. 在项目中正确使用BatchNorm
2. 处理BatchNorm的常见问题
3. 性能优化（融合卷积和BatchNorm）

### 参考资源

- 论文：Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (ICML 2015)
- PyTorch文档：torch.nn.BatchNorm1d/2d/3d
- 教程：CS231n相关课程材料