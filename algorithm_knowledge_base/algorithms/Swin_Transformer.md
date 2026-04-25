# Swin Transformer 学习文档

> 层级式视觉Transformer，引入移动窗口注意力

---

## 1. 算法基础认知

### 1.1 发展背景

Swin Transformer 由 Microsoft Research Asia 于 2021 年在论文《Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows》中提出，通过移动窗口注意力机制解决了 ViT 的二次复杂度问题，在目标检测和分割任务上取得了 SOTA。相比 ViT 的全局注意力，Swin 使用分块局部注意力并通过移动窗口实现跨块信息交互，实现了线性复杂度的同时保持全局建模能力。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 层级式 Vision Transformer |
| 复杂度 | O(N) 线性 |
| 窗口大小 | 7×7 |
| 移动窗口 | 逐步移动 |

### 1.3 模型系列

| 模型 | 参数量 | GFLOPs | ImageNet Top-1 |
|------|--------|--------|---------------|
| Swin-T | 28M | 4.5 | 81.3% |
| Swin-S | 50M | 8.7 | 83.0% |
| Swin-B | 88M | 15.4 | 83.5% |
| Swin-L | 197M | 37.0 | 84.7% |

---

## 2. 核心原理

### 2.1 移动窗口注意力

Swin 的核心创新是**分层局部注意力 + 移动窗口**：

```
标准注意力: 每个 token 关注所有其他 token (O(N²))
窗口注意力: 每个 token 只关注同窗口内 token (O(N×M))

核心思想：
- Stage 1: 固定 7×7 窗口内的注意力
- Stage 2+: 移动半个窗口，实现跨窗口交互
```

### 2.2 层级结构

```
输入 → Patch Embedding → Stage1 (7×7) → Stage2 (14×14) → Stage3 (28×28) → Stage4 (56×56)
  224×224         56×56        28×28         14×14         7×7
```

### 2.3 关键组件

- **W-MSA (Window-based Multi-Head Self-Attention)**：窗口注意力
- **SW-MSA (Shifted Window MSA)**：移动窗口注意力
- **Relative Position Bias**：相对位置编码

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 输入特征 |
| $B$ | Batch size |
| $H, W$ | 特征图高宽 |
| $C$ | 通道数 |
| $P$ | 窗口大小 |
| $M$ | $P \times P$ 窗口内token数 |
| $h$ | 头的数量 |
| $Q, K, V$ | Query, Key, Value |

### 3.2 窗口注意力公式

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

其中 $B$ 是相对位置偏置：

$$B_{ij} = \tilde{B}_{rel\_i - rel\_j}$$

### 3.3 移动窗口

```
Stage 1: 
┌────┬────┐
│ ██ │ ██ │   固定窗口
├────┼────┤
│ ██ │ ██ │
└────┴────┘

Stage 2:
 ┌┬──┐┌┬──┐
 │█│█││█│█│   移动窗口
 ├┼──┼┼┼──┤
 │█│█││█│█│
 └┴──┘└┴──┘
```

移动偏移量 = floor(P/2)

### 3.4 复杂度分析

| 注意力类型 | 复杂度 |
|------------|---------|
| 全局注意力 | $O(N^2)$ |
| 窗口注意力 | $O(N \cdot M)$ ≈ $O(N)$ |
| 移动窗口 | 同窗口注意力 |

---

## 4. 训练过程讲解

### 4.1 预训练配置

| 参数 | 值 |
|------|-----|
| ImageNet | 1M images |
| Batch Size | 4096 |
| Epochs | 300 |
| Learning Rate | 5e-4 |
| Weight Decay | 0.05 |
| Optimizer | AdamW |
| Scheduler | Cosine |

### 4.2 下游任务微调

- **ImageNet 分类**：直接fine-tune
- **COCO 检测**：使用FPN/蒙版注意力
- **ADE20K 分割**：UperNet

---

## 5. 应用场景

### 5.1 典型应用

- **ImageNet 分类**：图像分类
- **COCO 目标检测**：实例检测
- **ADE20K 语义分割**：场景解析
- **视频动作识别**：时序建模

### 5.2 代码示例

```python
import timm

# 加载 Swin
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

# 推理
output = model(x)
```

---

## 6. 调库实现

### 6.1 基本实现

```python
import torch
import torch.nn as nn
import timm

class SwinTransformerModel:
    """Swin Transformer 模型"""
    
    def __init__(self, model_name='swin_base_patch4_window7_224'):
        self.model_name = model_name
        
        if timm is not None:
            self.model = timm.create_model(model_name, pretrained=True)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        return self.model.forward_features(x)


def demo():
    """演示"""
    print("=== Swin Transformer 演示 ===\n")
    
    if timm is not None:
        swin = SwinTransformerModel()
        params = sum(p.numel() for p in swin.model.parameters())
        print(f"模型: {swin.model_name}")
        print(f"参数量: {params:,}")
    else:
        print("timm 未安装，安装: pip install timm")


if __name__ == "__main__":
    demo()
```

### 6.2 分割实现

```python
"""
针对分割任务的 Swin-UNet 实现
"""

class SwinUNet(nn.Module):
    """Swin-UNet for 医学图像分割"""
    
    def __init__(self, backbone='swin_tiny_patch4_window7_224', num_classes=2):
        super().__init__()
        
        # Backbone
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        
        # 解码器
        feature_channels = [96, 192, 384, 768]  # Swin-T特征通道
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(feature_channels[3], 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + feature_channels[2], 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + feature_channels[1], 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + feature_channels[0], 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        
        # 输出
        self.head = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        
        # 解码
        x = self.decoder4(features[3])
        x = self.decoder3(torch.cat([x, features[2]], dim=1))
        x = self.decoder2(torch.cat([x, features[1]], dim=1))
        x = self.decoder1(torch.cat([x, features[0]], dim=1))
        
        return self.head(x)
```

---

## 7. 手工代码实现

### 7.1 Window Attention 模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowAttention(nn.Module):
    """窗口注意力"""
    
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # 注意力
        q = q * (self.head_dim ** -0.5)
        attn = (q @ k.transpose(-2, -1))
        
        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        # 加权
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def demo_manual():
    """手工实现演示"""
    print("=== Swin Transformer 手工实现演示 ===\n")
    
    # 简化测试
    block = SwinTransformerBlock(dim=96, num_heads=8, window_size=7)
    x = torch.randn(1, 3136, 96)  # 56x56
    
    with torch.no_grad():
        out = block(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")


if __name__ == "__main__":
    demo_manual()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **线性复杂度**：O(N) 而非 O(N²)
2. **层级结构**：多尺度特征
3. **移动窗口**：全局建模能力
4. **SOTA性能**：ImageNet/COCO/ADE20K领先

### 8.2 缺点

1. **实现复杂**：窗口计算和mask
2. **调参敏感**：窗口大小影响大
3. **显存占用**：移动窗口计算

### 8.3 与ViT对比

| 特性 | ViT | Swin |
|------|-----|------|
| 复杂度 | O(N²) | O(N) |
| 感受野 | 全局初始 | 局部+移动 |
| 多尺度 | Patch嵌入 | 金字塔 |
| 数据需求 | 大数据 | 小数据也可 |

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_window_sampling():
    """移动窗口采样可视化"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Stage 1 窗口
    ax = axes[0]
    window1 = np.zeros((14, 14))
    for i in range(0, 14, 7):
        for j in range(0, 14, 7):
            window1[i:i+7, j:j+7] = 1
    ax.imshow(window1, cmap='Blues')
    ax.set_title('Stage 1: Fixed Windows')
    ax.axis('off')
    
    # Stage 2 移动
    ax = axes[1]
    window2 = np.zeros((14, 14))
    offset = 3  # 移动3个位置
    for i in range(0, 14, 7):
        for j in range(0, 14, 7):
            window2[max(0,i-offset):min(14,i-offset+7), 
                   max(0,j-offset):min(14,j-offset+7)] = 1
    ax.imshow(window2, cmap='Greens')
    ax.set_title('Stage 2: Shifted Windows')
    ax.axis('off')
    
    # 特征图
    ax = axes[2]
    features = np.random.rand(56, 56)
    im = ax.imshow(features, cmap='viridis')
    ax.set_title('Feature Maps')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('swin_visualization.png', dpi=150)
    plt.show()


def plot_performance():
    """性能对比"""
    
    models = ['ResNet-50', 'ViT-B/16', 'Swin-T', 'Swin-S', 'Swin-B']
    accuracies = [76.2, 79.9, 81.3, 83.0, 83.5]
    flops = [4.1, 55.8, 4.5, 8.7, 15.4]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 准确率
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('ImageNet Top-1 (%)', color=color)
    ax1.bar(models, accuracies, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(70, 90)
    
    # FLOPs
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('GFLOPs', color=color)
    ax2.plot(models, flops, 'o-', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.savefig('swin_performance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_window_sampling()
    plot_performance()
```

---

## 10. 模型评估

### 10.1 ImageNet 分类

| Model | Top-1 | Top-5 | Params |
|-------|-------|-------|--------|
| Swin-T | 81.3% | 95.3% | 28M |
| Swin-S | 83.0% | 96.2% | 50M |
| Swin-B | 83.5% | 96.5% | 88M |
| Swin-L | 84.7% | 97.1% | 197M |

### 10.2 COCO 检测

| Model | AP | Params |
|-------|-----|--------|
| Swin-T | 45.5 | 38M |
| Swin-B | 51.1 | 110M |

### 10.3 ADE20K 分割

| Model | mIoU | Params |
|-------|------|--------|
| Swin-T | 45.8 | 40M |
| Swin-S | 50.1 | 69M |

---

## 11. 常见问题

### Q1: 如何选择Window Size？

**答案**：常用7×7。更大窗口有利于长距离依赖，但计算量增加。图像较大时可用更大的窗口。

### Q2: Swin适合哪些任务？

**答案**：所有视觉任务。分类、检测、分割、关键点等。特别适合需要多尺度特征的任务。

### Q3: Shifted Window如何实现？

**答案**：通过循环移位和掩码机制实现。需要特殊的mask来处理边界情况和注意力移位。

### Q4: 与ViT相比的优势？

**答案**：1）计算复杂度低：O(N)而非O(N²); 2）局部先验：更适合小数据集; 3）多尺度：更适合下游任务。

### Q5: 如何微调Swin？

**答案**：冻结Encoder只训练分类头，或全参数微调。学习率通常为预训练的1/10。

---

## 12. 学习总结

### 核心要点

1. **移动窗口**：降低复杂度，实现全局建模
2. **层级结构**：多尺度特征金字塔
3. **线性O(N)**：高效可扩展

### Swin创新

- Window Attention替代全局Attention
- Shifted Window实现跨块交互
- 金字塔结构适配多任务

---

## 13. 练习题与思考题

### 练习1：计算

**问题**：224×224图像，window=7，计算Stage1的token数。

<details>
<summary>答案</summary>

Patch size = 4
特征图大小 = 224/4 = 56×56
Window size = 7×7
Token数 = (56/7)² × 7² = 8² × 49 = 3136

</details>

### 练习2：代码

**问题**：实现简单的Window Attention。

<details>
<summary>答案</summary>

```python
def window_attention(q, k, v, window_size):
    # 简化的window attention
    B, N, h, d = q.shape
    attn = (q @ k.transpose(-2, -1)) * (d ** -0.5)
    attn = F.softmax(attn, dim=-1)
    return attn @ v
```

</details>

### 思考题：改进

**问题**：Swin的潜在改进方向？

<details>
<summary>答案</summary>

1. **更大窗口**：如14×14或更大
2. **跨尺度**：Cross-Swin
3. **CNN融合**：ConvSwin
4. **自监督**：MAE预训练

</details>

---

## 14. 学习路径建议

### 14.1 进阶路径

1. ViT基础 → 注意力机制 
2. Swin原理 → 移动窗口实现 
3. 下游任务 → 分割/检测
4. SwinV2/CSwin → 改进版

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| ViT | 基础架构 |
| SwinV2 | 改进版 |
| CSwin | 交叉窗口 |
| MViT | 多尺度版本 |
| BEiT | 预训练 |

### 14.3 扩展阅读

1. Liu et al. (2021). Swin Transformer. arXiv:2103.14030
2. Liu et al. (2022). Swin Transformer V2. arXiv:2111.09883
3. 官方代码：microsoft/Swin-Transformer

---

**文档结束**