# ShuffleNet 学习文档

> 移动端轻量级CNN，用通道混洗优化Group Convolution

---

## 1. 算法基础认知

### 1.1 一句话定义

ShuffleNet是旷视科技于2017年提出的轻量级CNN网络，使用通道混洗（Channel Shuffle）优化Group Convolution，在保持性能的同时大幅减少计算量，适合移动端部署。

### 1.2 直觉类比

ShuffleNet就像一个"高效会议组织者"。普通Group Convolution让每个人只和本组人交流（信息不流通），ShuffleNet做的就是在每次讨论后"混洗"一下，让不同组的人也能交流信息。这样既保持了"分组讨论的高效"（减少了计算），又让信息充分流通（保持了准确率）！

想象你组织一个分组讨论会：
- 普通方式：分成5组，每组只和自己组讨论（效率高但信息不流通）
- ShuffleNet方式：每轮讨论后，成员随机换组——这样每个人都能听到所有组的观点（信息流通了，但仍然保持分组高效！）

### 1.3 发展背景

- 2017年，旷视科技Zhang等人在论文"ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"中提出
- 2018年，ShuffleNet v2发布，进一步优化
- 移动端轻量模型SOTA，长期占据Model Zoo榜首

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 轻量级CNN |
| 输出 | 图像分类/特征 |
| 模型 | 高效mobile-friendly |
| 计算量 | 极低（<300M FLOPs） |

---

## 2. 核心原理

### 2.1 Group Convolution的问题

**标准卷积的计算量**：
$$C_{out} \times C_{in} \times H \times W \times K \times K$$

**Group Convolution的优点**：分组计算，减少参数量

**Group Convolution的缺点**：不同组之间无信息交流！
```
分组1: [A, B, C] → 只和组内交流
分组2: [1, 2, 3] → 只和组内交流
问题：没有跨组信息传递！
```

### 2.2 通道混洗解决方案

**核心思想**：每次Group Convolution后，混洗通道！

```
输入: [g, c//g, h, w]
   ↓
reshape: [g, c//g, h, w]
   ↓
transpose: [c//g, g, h, w]
   ↓
reshape: [c, h, w]
```

### 2.3 ShuffleNet Block架构

```
输入
  │
  ▼
Conv 1x1 (grouped → 压缩通道)  ← 1x1 group conv
  │
  ▼
Channel Shuffle ← 核心创新！
  │
  ▼
3x3 Depthwise Conv (可分离卷积)
  │
  ▼
Conv 1x1 (grouped → 恢复通道)
  │
  ▼
+ 残差连接 (如果 strides=1)
  │
输出
```

### 2.4 vs 其他轻量模型对比

| 模型 | FLOPs | ImageNet精度 | 特点 |
|------|-------|-------------|------|
| MobileNet v1 | 569M | 70.6% | 深度可分离 |
| MobileNet v2 | 300M | 72.0% | 倒残差 |
| **ShuffleNet v2** | **292M** | **72.6%** | 通道混洗 |
| EfficientNet-B0 | 390M | 77.1% | 复合缩放 |

---

## 3. 数学公式与推导

### 3.1 Group Convolution

**输入**：$C_{in}$ 通道，$H \times W$ 空间
**分组**：$g$ 组
**输出**：$C_{out}$ 通道

每个组独立卷积：
$$Y_g = W_g \ast X_g$$

其中 $X_g \in \mathbb{R}^{C_{in}/g \times H \times W}$，$W_g \in \mathbb{R}^{C_{out}/g \times C_{in}/g \times K \times K}$

### 3.2 Channel Shuffle操作

```python
# 伪代码
def channel_shuffle(x, groups):
    batch, channels, h, w = x.size()
    x = x.view(batch, groups, channels // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, -1, h, w)
    return x
```

数学表示：
$$X'_{i,j,k} = X_{transpose(i,j),k}$$

### 3.3 计算量对比

| 操作 | 标准卷积 | Group Conv | ShuffleNet |
|------|----------|-----------|-----------|
| 1x1 conv | $C_{in} \times C_{out}$ | $2 \times C_{in} \times C_{out}/g$ | 相同 |
| 3x3 conv | $9 \times C_{in}$ | $9 \times C_{in}/g$ | 相同 |
| **总计** | $\propto C_{in} \times C_{out}$ | $\propto C_{in} \times C_{out}/g$ | **节省g倍** |

### 3.4 ShuffleNet v2的4条法则

2018年v2版本总结了4条实践法则：

1. **避免用1x1 GConv做分支瓶颈**：用普通1x1 conv
2. **维护通道维度一致性**：concat比add好
3. **减少element-wise操作**：ReLU、BatchNorm开销大
4. **注意FLOPs vs 实际速度**：内存访问也是瓶颈

---

## 4. 训练过程讲解

### 4.1 模型配置

| 版本 | 通道比例 | 输出通道 | FLOPs |
|------|---------|---------|-------|
| v2 x0.5 | 48 | 40M |
| v2 x1.0 | 116 | 146M |
| v2 x1.5 | 176 | 299M |
| v2 x2.0 | 244 | 524M |

### 4.2 训练参数

```python
# ImageNet训练
config = {
    'batch_size': 256,
    'lr': 0.4,              # for SGD with momentum
    'momentum': 0.9,
    'weight_decay': 4e-5,
    'epochs': 250,
    'warmup_epochs': 5,
}
```

### 4.3 数据增强

```python
train_augmentation = [
    RandomResizedCrop(224, scale=(0.08, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.4, contrast=0.4),
    Normalize(mean=[0.485, 0.456, 0.406], 
             std=[0.229, 0.224, 0.225])
]
```

### 4.4 预训练权重

```python
# torchvision
from torchvision.models import shuffenet_v2_x0_5, shuffenet_v2_x1_0

model = shuffenet_v2_x1_0(pretrained=True)
```

---

## 5. 应用场景

### 5.1 移动端部署

```python
# 手机APP
model = shuffenet_v2_x0_5(pretrained=True)
model = model.eval()

# 量化
model = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
```

### 5.2 嵌入式设备

```python
# 树莓派/ Jetson
# 模型导出
torch.jit.save(model, 'shufflenet.pt')
```

### 5.3 快速原型

```python
# 快速验证想法
model = shuffenet_v2_x0_5(pretrained=True)
features = model.forward特征层)
```

### 5.4 实际性能

| 设备 | 模型 | 吞吐量 | 延迟 |
|------|------|--------|------|
| iPhone X | v2 x1 | 100 FPS | 10ms |
| 树莓派4 | v2 x0.5 | 30 FPS | 33ms |
| 服务器 | v2 x1 | 500 FPS | 2ms |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 计算效率高 | 比MobileNet更少的FLOPs |
| 精度高 | ImageNet 72.6% (v2) |
| 实现简单 | Channel Shuffle容易 |
| 适合移动端 |ARM优化好 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 代码复杂度 | 比普通卷积多一步shuffle |
| 显存访问 | shuffle操作有开销 |
| 不如MobileNet流行 | 生态较小 |

### 6.3 注意事项

- Group数不是越大越好（通常4-8）
- ShuffleNet v2比v1更好
- 和MobileNet对比测试实际速度

---

## 7. 调库实现（Python）

### 7.1 torchvision

```python
import torch
from torchvision.models import shuffenet_v2_x1_0

# 加载预训练模型
model = shuffenet_v2_x1_0(pretrained=True)
model.eval()

# 输入
x = torch.randn(1, 3, 224, 224)

# 前向传播
with torch.no_grad():
    output = model(x)

print(f"输出形状: {output.shape}")  # [1, 1000]
```

### 7.2 自定义Block

```python
import torch
import torch.nn as nn

class ShuffleBlock(nn.Module):
    """ShuffleNet v2 Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid = out_channels // 2
        
        # 1x1 conv (grouped, 压缩)
        self.conv1 = nn.Conv2d(in_channels, mid, 1, groups=1)
        self.bn1 = nn.BatchNorm2d(mid)
        
        # 3x3 depthwise
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid)
        self.bn2 = nn.BatchNorm2d(mid)
        
        # 1x1 conv (grouped, 恢复)
        self.conv3 = nn.Conv2d(mid, mid, 1, groups=1)
        self.bn3 = nn.BatchNorm2d(mid)
        
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self._forward_block(x1), x2], dim=1)
        else:
            out = torch.cat([self._forward_block(x), x], dim=1)
        
        return self.relu(out.reshape(x.shape))
    
    def _forward_block(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Channel shuffle (简化版)
        # 实际实现需要正确的reshape和transpose
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        return x


# 测试
block = ShuffleBlock(64, 128, stride=2)
x = torch.randn(1, 64, 56, 56)
out = block(x)
print(f"输出形状: {out.shape}")  # [1, 128, 28, 28]
```

### 7.3 完整模型

```python
class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Stem
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        
        # Stage 2-4
        self.stage2 = nn.Sequential(
            ShuffleBlock(24, 116, stride=2),
            ShuffleBlock(116, 116),
            ShuffleBlock(116, 116),
        )
        
        self.stage3 = nn.Sequential(
            ShuffleBlock(116, 232, stride=2),
            ShuffleBlock(232, 232),
            ShuffleBlock(232, 232),
            ShuffleBlock(232, 232),
        )
        
        self.stage4 = nn.Sequential(
            ShuffleBlock(232, 488, stride=2),
            ShuffleBlock(488, 488),
            ShuffleBlock(488, 488),
        )
        
        # 输出
        self.conv5 = nn.Conv2d(488, 1024, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
```

### 7.4 训练示例

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 模型
model = ShuffleNetV2(num_classes=100)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        images, labels = batch
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}")
```

---

## 8. 手工代码实现（理解原理）

```python
import torch
import torch.nn as nn
import numpy as np

class ChannelShuffle(nn.Module):
    """通道混洗层"""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batch, channels, h, w = x.shape
        groups = self.groups
        
        # 必须是channels可被groups整除
        assert channels % groups == 0, f"channels {channels} not divisible by {groups}"
        
        x = x.reshape(batch, groups, channels // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch, -1, h, w)
        
        return x


class ShuffleNetBlock(nn.Module):
    """ShuffleNet Block"""
    def __init__(self, in_channels, out_channels, stride=1, groups=2):
        super().__init__()
        
        mid = out_channels // 2
        
        # 分支1: 1x1 group conv
        self.gconv1 = nn.Conv2d(in_channels, mid, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(mid)
        
        # 3x3 depthwise
        self.dwconv = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(mid)
        
        # 分支2: 1x1 group conv
        self.gconv2 = nn.Conv2d(mid, mid, 1, groups=groups)
        self.bn3 = nn.BatchNorm2d(mid)
        
        self.stride = stride
        self.shuffle = ChannelShuffle(groups)
        
        # 残差连接
        self.relu = nn.ReLU(inplace=True)
        
        # 如果stride>1，需要下采样
        if stride > 1:
            self.pool = nn.AvgPool2d(stride, stride)
        else:
            self.pool = None
    
    def forward(self, x):
        if self.stride == 1:
            # 分支1
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1 = self.pool(x)
            x2 = x
        
        # 分支1处理
        x1 = self.gconv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        # Shuffle
        x1 = self.shuffle(x1)
        
        # Depthwise
        x1 = self.dwconv(x1)
        x1 = self.bn2(x1)
        
        # 分支2处理
        x1 = self.gconv2(x1)
        x1 = self.bn3(x1)
        x1 = self.relu(x1)
        
        if self.stride > 1:
            x2 = self.pool(x2)
        
        # 合并
        out = torch.cat([x1, x2], dim=1)
        
        return out


# 测试
if __name__ == "__main__":
    block = ShuffleNetBlock(in_channels=24, out_channels=116, stride=2)
    x = torch.randn(1, 24, 56, 56)
    out = block(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    
    # Channel Shuffle测试
    shuffle = ChannelShuffle(groups=2)
    x = torch.randn(1, 4, 8, 8)
    y = shuffle(x)
    print(f"Shuffle输入: {x.shape}")
    print(f"Shuffle输出: {y.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in block.parameters())
    print(f"参数量: {total_params:,}")
```

---

## 9. 可视化与结果理解

### 9.1 模型结构可视化

```python
import matplotlib.pyplot as plt

# 各版本FLOPs对比
versions = ['x0.5', 'x1.0', 'x1.5', 'x2.0']
flops = [40, 146, 299, 524]
accuracy = [60.8, 72.6, 73.7, 74.7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# FLOPs
ax1.bar(versions, flops, color='steelblue')
ax1.set_ylabel('FLOPs (M)')
ax1.set_title('计算量对比')

# 精度
ax2.bar(versions, accuracy, color='coral')
ax2.set_ylabel('ImageNet Top-1 Accuracy (%)')
ax2.set_title('精度对比')

plt.tight_layout()
plt.savefig('shufflenet_comparison.png', dpi=100)
plt.show()
```

### 9.2 特征图可视化

```python
# 可视化中间特征
def visualize_features(model, image):
    model.eval()
    
    hooks = []
    features = []
    
    def hook_fn(module, input, output):
        features.append(output)
    
    # 注册hook到每个stage
    for name, module in model.named_modules():
        if 'stage' in name:
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        output = model(image)
    
    # 绘制特征
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, f in enumerate(features[:4]):
        if len(f.shape) == 4:
            f = f[0].mean(dim=0)
            axes[i//4, i%4].imshow(f.cpu().numpy())
    
    plt.tight_layout()
    plt.savefig('shufflenet_features.png', dpi=100)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| Top-1 Accuracy | ImageNet验证集精度 |
| Top-5 Accuracy | 前5准确率 |
| FLOPs | 浮点运算次数 |
| 参数量 | 模型大小 |
| 推理速度 | FPS / 延迟 |

### 10.2 对比表格

| 模型 | FLOPs | Top-1 | Params |
|------|------|-------|-------|
| MobileNet V3-L | 155M | 75.2% | 5.4M |
| MobileNet V2 | 300M | 72.0% | 3.5M |
| **ShuffleNet V2 x1** | **146M** | **72.6%** | **2.5M** |
| **ShuffleNet V2 x2** | **524M** | **74.7%** | **7.4M** |

### 10.3 评估代码

```python
import time

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            start = time.time()
            outputs = model(images)
            total_time += time.time() - start
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum()
            total += labels.size(0)
    
    accuracy = 100. * correct.item() / total
    avg_time = total_time / total * 1000  # ms
    
    return {
        'accuracy': accuracy,
        'avg_time_ms': avg_time
    }

# 评估
model = shuffenet_v2_x1_0(pretrained=True)
results = evaluate_model(model, test_loader)
print(f"Accuracy: {results['accuracy']:.2f}%")
print(f"Avg Time: {results['avg_time_ms']:.2f}ms")
```

---

## 11. 常见问题与易错点

### Q1: ShuffleNet和MobileNet哪个更好？

**答案**：实际测试为准。ShuffleNet v2在FLOPs接近时通常更快。

### Q2: 为什么需要Channel Shuffle？

**答案**：为了让不同group的特征互相交流，提高表达能力。

### Q3: Group数如何选择？

**答案**：通常4-8。太大可能导致信息丢失。

### Q4: v1和v2的区别？

**答案**：v2改进了架构设计，用实际速度而非FLOPs优化。

### Q5: 移动端部署注意什么？

**答案**：建议用torch.jit导出，量化可进一步加速。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心创新 | Channel Shuffle |
| 计算效率 | Group Conv优化 |
| 版本 | v1, v2 |
| 适合场景 | 移动端部署 |

### 12.2 公式汇总

Channel Shuffle：
$$X'_{i,j,k} = X_{transpose(i,j),k}$$

Group Conv计算：
$$Y_g = W_g \ast X_g$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. Channel Shuffle的主要作用是：
   - A) 减少计算量
   - B) 让不同组特征交流
   - C) 增强正则化

2. ShuffleNet v2相比v1的改进是：
   - A) 更多group
   - B) 更注重实际速度
   - C) 更深的网络

### 13.2 简答题

1. 解释Group Convolution的优缺点。
2. 为什么ShuffleNet比MobileNet快？

### 13.3 编程题

1. 实现一个完整的ShuffleNet Block。
2. 比较ShuffleNet和MobileNet的实际推理速度。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
CNN基础
    ↓
Group Convolution
    ↓
Channel Shuffle
    ↓
ShuffleNet原理
    ↓
移动端部署
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| MobileNet | 竞品 |
| EfficientNet | 更强但复杂 |
| GhostNet | 另一个轻量方案 |

### 14.3 扩展阅读

- Zhang et al. (2017). ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
- Zhang et al. (2018). ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

---

## 附录

### 参考

1. Zhang et al. (2017). ShuffleNet. arXiv:1707.01083
2. Zhang et al. (2018). ShuffleNet V2. arXiv:1807.11164
3. https://github.com/pytorch/vision

---

**文档结束**