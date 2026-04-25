# Group Normalization 组归一化 学习文档

> 替代Batch Norm，解决小batch问题

---

## 1. 算法基础认知

### 1.1 一句话定义

Group Normalization（组归一化）是2018年由吴育昕等人提出的归一化方法，将通道分组后归一化，不依赖batch大小，解决小batch下BatchNorm效果差的问题。

### 1.2 直觉类比

Group Normalization就像"分组讨论"。BatchNorm要求全组一起讨论（batch内所有样本），但人少时（batch小）讨论结果不可靠。Group Normalization改成每组自己讨论——把通道分成几组，每组内部归一化，这样人多人少都不影响！

想象公司开季度会议：
- BatchNorm = 等全公司100人都到了再统计平均值，人少时结果不准
- GroupNorm = 先按部门分组，每个部门自己统计，人少也不怕

### 1.3 发展背景

- 2018年，吴育昕、何恺明在论文"Group Normalization"中提出
- 同期提出还有Instance Normalization和Layer Normalization
- 解决小batch下BatchNorm失效的根本问题

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 归一化 → 深度学习 |
| 核心优势 | 不依赖batch size |
| 批量独立 | ✓ |
| 计算成本 | O(C·HW) |

---

## 2. 核心原理

### 2.1 归一化方法对比

深度学习中有多种归一化方法，区别在于计算统计量的维度：

| 方法 | 归一化维度 | batch依赖 | 适用场景 |
|------|-----------|---------|---------|
| BatchNorm | (B, C) | 是 | 大batch CV |
| LayerNorm | (C, H, W) | 否 | NLP/小batch |
| InstanceNorm | (C, H, W) | 否 | 风格迁移 |
| GroupNorm | (G, H, W) | 否 | 视频/小batch |
| **SyncBN** | 全局 | ✓ | 多卡同步 |

### 2.2 Group Normalization原理

**核心思想**：将C个通道分成G组，每组独立计算均值和方差。

```
输入: x ∈ R^(B, C, H, W)
通道分组: C → G，每组 G/C 个通道
对每组分别归一化
```

### 2.3 计算流程

```
Step 1: 将C个通道分成G组
       每组大小: C/G

Step 2: 对每个样本、每个组，计算均值
       μ_g = (1/(H·W·C/G)) Σ Σ x_b,c,h,w
       
Step 3: 计算方差       
       σ²_g = (1/(H·W·C/G)) Σ Σ (x_b,c,h,w - μ_g)²
       
Step 4: 归一化
       x̂_b,c,h,w = (x_b,c,h,w - μ_g) / √(σ²_g + ε)

Step 5: 线性变换
       y_b,c,h,w = γ·x̂_b,c,h,w + β
```

---

## 3. 数学公式与推导

### 3.1 基本公式

输入$x \in \mathbb{R}^{B \times C \times H \times W}$，Group Normalization定义为：

$$\text{GN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：

$$\mu_{bg} = \frac{1}{HW \cdot C/G} \sum_{c \in G_g} \sum_{h,w} x_{bchw}$$

$$\sigma^2_{bg} = \frac{1}{HW \cdot C/G} \sum_{c \in G_g} \sum_{h,w} (x_{bchw} - \mu_{bg})^2$$

### 3.2 组划分

令$G$为组数，则每组包含$G/C$个通道。第$g$组覆盖的通道范围是：

$$[g \cdot C/G, (g+1) \cdot C/G)$$

### 3.3 与其他归一化的关系

| 方法 | G值 | 关系 |
|------|------|------|
| InstanceNorm | G=C | 每通道独立归一化 |
| LayerNorm | G=1 | 所有通道一起归一化 |
| GroupNorm | G∈[1,C] | 一般取32或64 |

**特例**：
- 当G=1时，GroupNorm = LayerNorm
- 当G=C时，GroupNorm = InstanceNorm

### 3.4 梯度

反向传播时，需要对$\mu$和$\sigma^2$求导：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} - \frac{\partial L}{\partial \mu} \cdot \frac{1}{N} - \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x-\mu)}{N}$$

---

## 4. 训练过程讲解

### 4.1 Group大小选择

| 通道数C | 推荐G | 每组通道数 |
|---------|-------|----------|
| 32 | 32 | 1 |
| 64 | 32 | 2 |
| 128 | 32 | 4 |
| 256 | 32 | 8 |
| 512 | 32 | 16 |

经验法则：G=32是较好的默认值。

### 4.2 与BatchNorm对比

```python
# 不同的batch size下性能对比
batch_sizes = [1, 2, 4, 8, 16, 32]

for bs in batch_sizes:
    # BatchNorm
    model_bn = nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    
    # GroupNorm
    model_gn = nn.Sequential(
        nn.GroupNorm(32, 64),
        nn.ReLU()
    )
    
    # 测试（实际使用中对比准确率）
    pass
```

### 4.3 PyTorch实现

```python
import torch
import torch.nn as nn

# 基础用法
gn = nn.GroupNorm(num_groups=32, num_channels=64)

# 输入 [B, C, H, W]
x = torch.randn(4, 64, 32, 32)
y = gn(x)

print(f"输入: {x.shape}")
print(f"输出: {y.shape}")
print(f"均值: {y.mean(dim=[0,2,3]).mean():.4f}")  # 应接近0
print(f"方差: {y.var(dim=[0,2,3]).mean():.4f}") # 应接近1
```

---

## 5. 应用场景

### 5.1 视频理解

视频模型通常用小batch（因为显存限制）：

```python
# Video模型适用GroupNorm
class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, 64)  # 视频用GN
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)  # 不依赖batch
        return F.relu(x)
```

### 5.2 高分辨率图像分割

分割网络输入大图像，batch通常为1：

```python
# 分割网络
class Segmentor(nn.Module):
    def __init__(self):
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU()
        )
        # ...更多层
```

### 5.3 小batch训练

显存受限时只能用小batch：

```python
# 显存优化训练
trainloader = DataLoader(dataset, batch_size=2)  # 小batch

model = models.resnet18(weights=None)
# 用GroupNorm替代BatchNorm
model = nn.Sequential(*[
    nn.GroupNorm(32, ch) if isinstance(m, nn.BatchNorm2d) else m
    for m in model.modules()
])
```

### 5.4 对比选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 大batch(>32) | BatchNorm | batch统计稳定 |
| 小batch(<=8) | GroupNorm | 批量独立 |
| 风格迁移 | InstanceNorm | 保留风格信息 |
| NLP | LayerNorm | 序列处理 |
| 视频 | GroupNorm | 小batch多 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 批量独立 | batch大小不影响 |
| 显存友好 | 不存储batch统计量 |
| 训练推理一致 | 无需同步batch |
| 适应小batch | 小batch下依然有效 |
| 视频友好 | 适合时序数据 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 大batch效果略逊 | batch大时不如BN |
| 需调G | 组数需要调参 |
| 每通道参数 | γ,β参数量同BN |

### 6.3 注意事项

- G越大，越接近InstanceNorm
- G=1时，等于LayerNorm
- 默认G=32效果较好

---

## 7. 调库实现（Python + PyTorch）

### 7.1 基本用法

```python
import torch
import torch.nn as nn

# 创建带GroupNorm的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, 64)  # G=32, C=64
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, 128)
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 测试
model = SimpleNet()
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(f"输出: {y.shape}")
```

### 7.2 不同G值对比

```python
# 对比不同G值
g_values = [1, 2, 4, 8, 16, 32, 64]

results = {}
x = torch.randn(4, 64, 32, 32)

for G in g_values:
    gn = nn.GroupNorm(G, 64)
    y = gn(x)
    # 统计特性
    mean = y.mean(dim=[0,2,3)
    var = y.var(dim=[0,2,3])
    results[G] = {'mean': mean.abs().max().item(), 'var': var.abs().max().item()}

print("G\tmean\t\tvar")
for G, r in results.items():
    print(f"{G}\t{r['mean']:.4f}\t\t{r['var']:.4f}")
```

### 7.3 训练示例

```python
import torch.optim as optim

# 简单训练循环
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    
    for batch in trainloader:
        inputs, labels = batch
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: loss={total_loss/len(trainloader):.4f}")
```

### 7.4 BatchNorm转GroupNorm

```python
# 将预训练模型从BatchNorm改为GroupNorm
def convert_bn_to_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # 替换为GroupNorm
            gn = nn.GroupNorm(num_groups, module.num_features)
            setattr(model, name, gn)
        else:
            convert_bn_to_gn(module, num_groups)
    return model

# 使用
model = models.resnet18(pretrained=True)
model = convert_bn_to_gn(model, num_groups=32)
```

---

## 8. 手工代码实现（核心算法手写）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    """Group Normalization - 手工实现"""
    
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # 可学习参数
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 划分组
        G = self.num_groups
        assert C % G == 0, f"C={C} cannot be divided by G={G}"
        channels_per_group = C // G
        
        # 变形: [B, G, C/G, H, W]
        x = x.view(B, G, channels_per_group, H, W)
        
        # 计算均值和方差
        mean = x.mean(dim=[2,3,4], keepdim=True)
        var = x.var(dim=[2,3,4], keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 变形回去: [B, C, H, W]
        x_norm = x_norm.view(B, C, H, W)
        
        # 线性变换
        # 扩展weight和bias到[B,C,H,W]
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        return x_norm * weight + bias


class GroupNormManual:
    """纯numpy手工实现（用于理解原理）"""
    
    def __init__(self, num_groups, num_channels, eps=1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = torch.ones(num_channels)
        self.bias = torch.zeros(num_channels)
    
    def __call__(self, x):
        """前向传播（仅示意，pytorch实现）"""
        return self.forward(x)
    
    def forward(self, x):
        """纯numpy实现"""
        if isinstance(x, torch.Tensor):
            return self._torch_impl(x)
        else:
            return self._numpy_impl(x)
    
    def _torch_impl(self, x):
        """PyTorch实现"""
        B, C, H, W = x.shape
        G = self.num_groups
        Cg = C // G
        
        # Reshape: [B, G, Cg, H, W]
        x = x.view(B, G, Cg, H, W)
        
        # Mean & Var
        mean = x.mean(dim=[2,3,4], keepdim=True)
        var = x.var(dim=[2,3,4], keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(B, C, H, W)
        
        # Scale & Shift
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        return x_norm * weight + bias
    
    def _numpy_impl(self, x):
        """纯numpy实现"""
        # 简化版：假设B=1
        assert x.ndim == 4, "Expected 4D input"
        
        B, C, H, W = x.shape
        G = self.num_groups
        Cg = C // G
        
        # 分组
        x = x.reshape(B, G, Cg, H, W)
        
        # 计算统计量
        mean = x.mean(axis=(2,3,4), keepdims=True)
        var = x.var(axis=(2,3,4), keepdims=True)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 恢复形状
        x_norm = x_norm.reshape(B, C, H, W)
        
        return x_norm


# 测试
if __name__ == "__main__":
    import numpy as np
    
    # PyTorch测试
    x = torch.randn(4, 64, 32, 32, requires_grad=True)
    
    # 手工实现
    gn_manual = GroupNorm(num_groups=32, num_channels=64)
    y_manual = gn_manual(x)
    
    # PyTorch实现
    gn_torch = nn.GroupNorm(num_groups=32, num_channels=64)
    y_torch = gn_torch(x)
    
    print("手工实现:")
    print(f"  shape: {y_manual.shape}")
    print(f"  mean: {y_manual.mean().item():.4f}")
    print(f"  var: {y_manual.var().item():.4f}")
    
    print("\nPyTorch实现:")
    print(f"  shape: {y_torch.shape}")
    print(f"  mean: {y_torch.mean().item():.4f}")
    print(f"  var: {y_torch.var().item():.4f}")
    
    # 差异
    print(f"\n最大差异: {(y_manual - y_torch).abs().max().item():.6f}")
```

---

## 9. 可视化与结果理解

### 9.1 batch大小影响可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成模拟数据
batch_sizes = [1, 2, 4, 8, 16, 32]
means_bn = []
means_gn = []

for bs in batch_sizes:
    x = torch.randn(bs, 64, 32, 32)
    
    # BN
    bn = nn.BatchNorm2d(64)
    y_bn = bn(x)
    means_bn.append(y_bn.mean().item())
    
    # GN
    gn = nn.GroupNorm(32, 64)
    y_gn = gn(x)
    means_gn.append(y_gn.mean().item())

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(batch_sizes))
width = 0.35

ax.bar(x - width/2, means_bn, width, label='BatchNorm', color='steelblue')
ax.bar(x + width/2, means_gn, width, label='GroupNorm', color='coral')

ax.set_xlabel('Batch Size')
ax.set_ylabel('Mean (after normalization)')
ax.set_title('BatchNorm vs GroupNorm: 不同batch下的均值')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('bn_vs_gn.png', dpi=100)
plt.show()
```

### 9.2 不同G值的影响

```python
# 可视化不同G值的效果
import matplotlib.pyplot as plt

G_values = [1, 2, 4, 8, 16, 32, 64]
x = torch.randn(4, 64, 16, 16)

variances = []
for G in G_values:
    gn = nn.GroupNorm(G, 64)
    y = gn(x)
    variances.append(y.var().item())

plt.figure(figsize=(10, 4))
plt.plot(G_values, variances, 'o-')
plt.xlabel('Number of Groups (G)')
plt.ylabel('Variance after normalization')
plt.title('Group Normalization: 不同G值的方差')
plt.grid(True, alpha=0.3)
plt.savefig('gn_g_values.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 理想值 | 说明 |
|------|--------|------|
| 均值 | 0 | 归一化后 |
| 方差 | 1 | 归一化后 |
| 训练速度 | - | 对比batch size |
| 显存 | - | 对比BN |

### 10.2 对比代码

```python
import time

def measure_speed(model, input_tensor, n_iter=100):
    """测量推理速度"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # 计时
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(input_tensor)
    
    return (time.time() - start) / n_iter * 1000  # ms

# 对比
x = torch.randn(4, 64, 224, 224)

model_bn = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU())
model_gn = nn.Sequential(nn.GroupNorm(32, 64), nn.ReLU())

speed_bn = measure_speed(model_bn, x)
speed_gn = measure_speed(model_gn, x)

print(f"BatchNorm: {speed_bn:.2f}ms")
print(f"GroupNorm: {speed_gn:.2f}ms")
```

---

## 11. 常见问题与易错点

### Q1: G值如何选择？

**答案**：默认G=32。通道数能被32整除最好，不能则选择能整除的接近值。

### Q2: 和LayerNorm的区别？

**答案**：LayerNorm是G=1的情况，所有通道一起归一化。GroupNorm分组更灵活。

### Q3: 小batch下一定用GN？

**答案**：推荐使用。batch=1时BN完全失效，GN依然有效。

### Q4: GN需要用eval()模式吗？

**答案**：不需要！GN不依赖batch统计量，训练和推理行为一致。

### Q5: 可以和BN混合使用吗？

**答案**：可以。不同层用不同归一化方法。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 通道分组归一化 |
| 公式 | GN(x) = γ·(x-μ)/σ + β |
| 与BN关系 | 批量独立版本 |
| 与LN关系 | G=1时等于LN |
| 与IN关系 | G=C时等于IN |

### 12.2 公式汇总

均值：
$$\mu = \frac{1}{HW \cdot C/G} \sum x$$

方差：
$$\sigma^2 = \frac{1}{HW \cdot C/G} \sum (x-\mu)^2$$

归一化：
$$\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}}$$

输出：
$$y = \gamma \hat{x} + beta$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. GroupNorm中G=1等价于：
   - A) BatchNorm
   - B) LayerNorm
   - C) InstanceNorm

2. GroupNorm的优势场景是：
   - A) 大batch
   - B) 小batch
   - C) 风格迁移

3. G=C时GroupNorm等价于：
   - A) BatchNorm
   - B) LayerNorm
   - C) InstanceNorm

### 13.2 简答题

1. 为什么GroupNorm在batch=1时依然有效？
2. 比较GroupNorm和LayerNorm的适用场景。

### 13.3 编程题

1. 实现G可学习的GroupNorm。
2. 比较不同归一化方法在CIFAR-10上的效果。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
深度学习基础
    ↓
BatchNorm原理
    ↓
归一化方法对比
    ↓
GroupNorm原理
    ↓
实践调参
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| BatchNorm | batch版本 |
| LayerNorm | G=1 |
| InstanceNorm | G=C |
| SyncBN | 多卡同步BN |

### 14.3 扩展阅读

- Wu, Y., He, K. (2018). Group Normalization. arXiv:1803.08494

---

## 附录

### 参考

1. Wu, Y., He, K. (2018). Group Normalization. arXiv:1803.08494
2. nn.GroupNorm — PyTorch Documentation

---

**文档结束**