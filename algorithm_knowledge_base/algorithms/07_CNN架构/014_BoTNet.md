# BoTNet (Bottleneck Transformer) 学习文档

## 1. 算法基础认知

### 1.1 算法要解决什么问题

在 2021 年之前，CV 领域的两个主流范式——CNN 和 Transformer——有着根本性的差异：

- **CNN**：擅长局部特征提取，通过堆叠卷积层实现层次化表示，计算效率高，但在建模长距离依赖关系时受限
- **Transformer**：通过自注意力机制天然支持全局关系建模，但缺乏 CNN 的局部先验，需要大量数据训练

BoTNet（Bottleneck Transformer）的出发点是一个简单的问题：**能否在 ResNet 的瓶颈块中，将空间卷积替换为自注意力，从而同时获得 CNN 的层次化结构和 Transformer 的全局建模能力？**

### 1.2 核心思路概览

BoTNet 由 Srinivas 等人于 2021 年提出，其设计极为简洁：

```
标准 ResNet Bottleneck: 1×1 conv → 3×3 conv → 1×1 conv
BoTNet Bottleneck:     1×1 conv → MHSA    → 1×1 conv
```

具体来说，将 ResNet 最后三个阶段（C4, C5）中的 3×3 空间卷积替换为**多头自注意力（MHSA）**，其他部分保持不变。

这种设计的精妙之处在于：
1. **即插即用**：不需要改变整体架构，只需要替换一个组件
2. **继承 CNN 的层次结构**：保留了 ResNet 的 stage 设计（下采样、通道数递增）
3. **引入全局建模**：在较高层（特征图尺寸较小，如 14×14）使用自注意力，计算量可控

### 1.3 整体架构

BoTNet 的架构与 ResNet 几乎相同，仅有的区别在最后几个 stage：

```
输入 (224×224×3)
  ↓
Stage 1: 7×7 conv (stride 2) + MaxPool (stride 2) → 56×56
  ↓
Stage 2: 3× ResNet Bottleneck (1×1→3×3→1×1) → 56×56
  ↓
Stage 3: 4× ResNet Bottleneck (1×1→3×3→1×1) → 28×28
  ↓
Stage 4: 6× BoTNet Bottleneck (1×1→MHSA→1×1) → 14×14
  ↓
Stage 5: 3× BoTNet Bottleneck (1×1→MHSA→1×1) → 7×7
  ↓
Average Pooling → Linear → 分类
```

关键信息：
- MHSA 只在 14×14 和 7×7 分辨率上使用（计算量可控）
- 下采样仍然通过 stride=2 的 1×1 卷积实现
- 通道数与 ResNet-50/101 一致

## 2. 核心原理

### 2.1 为什么替换最后的 3×3 卷积？

选择替换最后几个 stage 的 3×3 卷积，而不是所有 stage，原因有三：

**1. 计算效率**
自注意力的复杂度为 O(N²·D)，其中 N 是空间位置的个数。在 ResNet 的不同 stage：
- Stage 3: 56×56 = 3136 个位置 → O(3136²) 不可接受
- Stage 4: 14×14 = 196 个位置 → O(196²) 可以接受
- Stage 5: 7×7 = 49 个位置 → O(49²) 很高效

所以只在特征图较小的 stage 替换。

**2. 语义层次**
浅层网络处理低级特征（边缘、纹理），这些特征需要局部操作（卷积）；深层网络处理高级语义特征，这些特征需要全局关系建模（注意力）。

**3. 渐进式引入**
BoTNet 并不是一次性地用 MHSA 替换所有卷积，而是在较高层逐步引入，保持 CNN 的层次化特征提取能力。

### 2.2 MHSA 与卷积的核心差异

**卷积（3×3 Conv）**：
- 感受野：固定 3×3（可通过堆叠扩大，但效率低）
- 权重共享：在空间位置上共享相同的卷积核
- 计算方式：加权求和（权重由卷积核确定，与输入无关）
- 复杂度：O(K²·C_in·C_out·H·W)

**多头自注意力（MHSA）**：
- 感受野：全局（整个特征图）
- 权重自适应：注意力权重取决于输入本身（Q 和 K 的点积）
- 计算方式：内容自适应加权
- 复杂度：O(H·W·C² + (H·W)²·C)

### 2.3 Positional Encoding 的处理

在 MHSA 中，由于自注意力是置换等变的（permutation equivariant），需要位置编码来提供空间信息。BoTNet 使用了两种位置编码方案：

**方案 1：绝对位置编码**
将可学习的位置编码加到特征图上：
```
x' = x + pos_embed
```

**方案 2：相对位置编码**
在注意力计算中加入位置偏差：
```
Attention(Q,K) = Softmax(QK^T/√d + B)
```
其中 B_ij 表示位置 i 和 j 之间的相对位置偏差。

BoTNet 默认使用绝对位置编码，因为它简单且与 ResNet 的架构兼容。

### 2.4 ResNet 的 Bottleneck 设计回顾

标准 ResNet Bottleneck：
```
输入: C 通道
  ↓
1×1 Conv (C → C/4)  # 降维
  ↓
3×3 Conv (C/4 → C/4)  # 空间特征提取
  ↓
1×1 Conv (C/4 → C)  # 升维
  ↓
+ 残差连接
  ↓
输出: C 通道
```

BoTNet Bottleneck：
```
输入: C 通道
  ↓
1×1 Conv (C → C/4)  # 降维
  ↓
MHSA (C/4 → C/4)  # 空间特征提取（替代 3×3 Conv）
  ↓
1×1 Conv (C/4 → C)  # 升维
  ↓
+ 残差连接
  ↓
输出: C 通道
```

唯一的变化是用 MHSA 替换 3×3 卷积。

## 3. 数学公式与推导

### 3.1 标准 MHSA 在 BoTNet 中的形式

在 BoTNet 的 MHSA 中，输入是一个 2D 特征图 X ∈ ℝ^{H×W×C}。

首先将特征图摊平为序列：
```
X_flat = Reshape(X, (H·W, C))
```

然后计算多头注意力：
```
Q = X_flat · W_Q ∈ ℝ^{N×D}     W_Q ∈ ℝ^{C×D}
K = X_flat · W_K ∈ ℝ^{N×D}     W_K ∈ ℝ^{C×D}
V = X_flat · W_V ∈ ℝ^{N×D}     W_V ∈ ℝ^{C×D}
```

其中 N = H·W，D 是每个头的维度（D = C_red / H，C_red 是注意力输入通道）。

多头注意力输出：
```
head_h = Softmax(Q_h · K_h^T / √d_k) · V_h
MHSA(X) = Concat(head_1, ..., head_H) · W_O
```

### 3.2 相对位置编码

位置编码 B 是一个偏置矩阵，表示位置 i 和 j 之间的相对位置：
```
Attention(Q, K) = Softmax(Q·K^T/√d + B)
```

对于 2D 特征图，B 通常分解为高度和宽度两个方向：
```
B = B_h + B_w
```

其中 B_h[i,j] = p_h[h_i - h_j]，B_w[i,j] = p_w[w_i - w_j]，p_h 和 p_w 是可学习的参数表。

### 3.3 BoTNet Block 的完整前向

```
def forward(x):
    identity = x

    # 步骤 1: 1×1 Conv（降维）
    x = Conv2D(1×1)(x)
    x = BN(x)
    x = ReLU(x)

    # 步骤 2: MHSA（空间特征提取）
    x = MHSA(x + pos_embed)  # 先加位置编码，再做注意力
    # 或者：x = MHSA_with_rel_pos(x)
    x = BN(x)
    x = ReLU(x)

    # 步骤 3: 1×1 Conv（升维）
    x = Conv2D(1×1)(x)
    x = BN(x)

    # 步骤 4: 残差连接
    x = x + identity
    x = ReLU(x)

    return x
```

### 3.4 计算量分析

对比 ResNet-50 的 Bottleneck 和 BoTNet 的 Bottleneck：

**ResNet Bottleneck（Stage 4，C=1024）**：
- 1×1 conv: 1024×256 + 256×1024 = 524,288 个参数
- 3×3 conv: 3×3×256×256 = 589,824 个参数
- 总参数: 1,114,112

**BoTNet Bottleneck（Stage 4，C=1024）**：
- 1×1 conv: 1024×256 + 256×1024 = 524,288 个参数
- MHSA (head=4, head_dim=64): QKV 投影: 256×256×3 = 196,608，输出投影: 256×256 = 65,536
- 总参数: 786,432

MHSA 比 3×3 conv 的参数更少，但 FLOPs 取决于特征图大小（14×14 × 14×14 = 38,416 vs 3×3×196 = 1,764）。

## 4. 训练过程讲解

### 4.1 训练配置

BoTNet 的训练配置与 ResNet 相似，但有些微调：

- **优化器**：SGD with Momentum（momentum=0.9）
- **学习率**：0.1（cosine annealing）
- **权重衰减**：1e-4
- **批次大小**：4096
- **训练轮数**：300 epoch
- **SyncBN**：使用同步 BatchNorm（多 GPU 训练）

### 4.2 初始化策略

BoTNet 的初始化非常重要：
- MHSA 的权重使用 Xavier uniform 初始化
- 残差分支中最后一个 1×1 Conv 的权重初始化为 0（zero-initialization），使每个 block 初始为恒等映射

### 4.3 数据增强

- RandomResizedCrop (224×224)
- RandomHorizontalFlip
- RandAugment
- Label Smoothing (ε=0.1)
- Mixup (optional)

### 4.4 学习率策略

使用 cosine learning rate schedule：
```
lr(t) = 0.5 * lr_0 * (1 + cos(π * t / T))
```

## 5. 应用场景

### 5.1 图像分类

BoTNet 在 ImageNet 上的性能：
- BoTNet-50: 80.5% top-1（ResNet-50: 76.0%）
- BoTNet-101: 81.7% top-1（ResNet-101: 77.4%）
- BoTNet-T3（更大版本）: 83.5% top-1

相比 ResNet 的显著提升（+4.5%），只替换了后两个 stage 的卷积。

### 5.2 目标检测

BoTNet 作为目标检测的 backbone（在 COCO 上）：
- Mask R-CNN + BoTNet-50: 44.4% AP（ResNet-50: 38.9%）
- Cascade R-CNN + BoTNet-101: 48.7% AP

### 5.3 实例分割

在 Mask R-CNN 中作为 backbone：
- Mask R-CNN + BoTNet-50: 39.7% mask AP

## 6. 优缺点分析

### 6.1 优点

1. **简单有效的混合设计**：用 MHSA 替换 3×3 卷积，只改一个组件
2. **继承 CNN 的层次结构**：保留了 ResNet 的 stage 设计、下采样、通道递增
3. **即插即用**：可以直接用于任何基于 ResNet 的架构（检测、分割等）
4. **高性能**：相同计算量下显著优于纯 CNN
5. **训练稳定**：SGD 即可训练，不需要 AdamW

### 6.2 缺点

1. **仅在低分辨率有效**：高分辨率时 MHSA 计算量太大
2. **位置编码处理粗糙**：简单加位置编码可能不够精细
3. **仅替换卷积核**：没有改变整体架构设计，可能存在更优的混合方案
4. **BN vs LN 的冲突**：ResNet 使用 BN，Transformer 通常使用 LN，BoTNet 混用 BN 可能不够理想

## 7. 调库实现

### 7.1 完整的 BoTNet 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHSA(nn.Module):
    """
    多头自注意力模块（用于替换 3×3 卷积）
    支持绝对位置编码和相对位置编码
    """
    def __init__(self, dim: int, num_heads: int = 4, head_dim: int = 64,
                 use_rel_pos: bool = False, height: int = 14, width: int = 14):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_rel_pos = use_rel_pos

        # QKV 投影
        self.qkv = nn.Linear(dim, num_heads * head_dim * 3)
        self.proj = nn.Linear(num_heads * head_dim, dim)

        if use_rel_pos:
            # 相对位置编码
            # 对于 HxW 的特征图，相对位置范围是 [-(H-1), H-1] 和 [-(W-1), W-1]
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * height - 1, head_dim)
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * width - 1, head_dim)
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x (B, C, H, W)
        输出: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # 展平为序列
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # QKV 投影
        qkv = self.qkv(x_flat)  # (B, N, 3*H*Dh)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if self.use_rel_pos:
            # 相对位置编码
            attn = attn + self._get_rel_pos_bias(H, W, device=attn.device)

        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = attn @ V  # (B, H, N, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)

        # 输出投影
        out = self.proj(out)  # (B, N, C)

        # 恢复为 2D 特征图
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out

    def _get_rel_pos_bias(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        计算相对位置编码偏置
        返回: (1, 1, H*W, H*W)
        """
        # 构建相对位置索引
        pos_h = torch.arange(H, device=device)
        pos_w = torch.arange(W, device=device)

        # 每个位置 (i, j) 的相对位置
        rel_h = pos_h[:, None] - pos_h[None, :]  # (H, H)
        rel_w = pos_w[:, None] - pos_w[None, :]  # (W, W)

        # 映射到 [0, 2*H-2] 和 [0, 2*W-2]
        rel_h += H - 1
        rel_w += W - 1

        # 嵌入相对位置
        # rel_pos_h: (2*H-1, Dh), rel_pos_w: (2*W-1, Dh)
        # 扩展为 (H*W, H*W, Dh)
        pos_bias_h = self.rel_pos_h[rel_h.flatten()]  # (H*H, Dh)
        pos_bias_h = pos_bias_h.reshape(H, H, -1)  # (H, H, Dh)
        pos_bias_w = self.rel_pos_w[rel_w.flatten()]  # (W*W, Dh)
        pos_bias_w = pos_bias_w.reshape(W, W, -1)  # (W, W, Dh)

        # 组合为 (H*W, H*W, Dh)
        # 对每个 (i, j) -> (k, l):
        # bias = rel_pos_h[i, k] + rel_pos_w[j, l]
        h_bias = pos_bias_h[:, None, :, None, :].expand(H, W, H, 1, -1)
        w_bias = pos_bias_w[None, :, None, :, :].expand(H, W, 1, W, -1)
        pos_bias = h_bias + w_bias  # (H, W, H, W, Dh)

        # 聚合头维度
        pos_bias = pos_bias.reshape(H * W, H * W, -1)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)  # (1, Dh, HW, HW)

        # 归一化
        return pos_bias @ self.scale


class BoTNetBlock(nn.Module):
    """
    BoTNet Bottleneck Block
    用 MHSA 替换 ResNet Bottleneck 中的 3×3 卷积
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_heads: int = 4,
        head_dim: int = 64,
        use_rel_pos: bool = False,
        height: int = 14,
        width: int = 14,
    ):
        super().__init__()

        # Bottleneck 中间的通道数（通常是 out_channels / 4）
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 替换 3×3 卷积为 MHSA
        if stride == 1:
            # 标准 MHSA（stride=1 时保持尺寸）
            self.mhsa = MHSA(mid_channels, num_heads, head_dim, use_rel_pos, height, width)
        else:
            # stride=2 时，先通过 stride=2 的卷积下采样，再使用 MHSA
            # 或者对注意力池化用 stride 方式
            self.mhsa = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                         stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                MHSA(mid_channels, num_heads, head_dim, use_rel_pos,
                     height // stride, width // stride),
            )

        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # 残差连接的 shortcut
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1×1 conv: 降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # MHSA: 空间特征提取（替代 3×3 conv）
        out = self.mhsa(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1 conv: 升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BoTNet(nn.Module):
    """
    BoTNet: Bottleneck Transformer
    基于 ResNet 架构，将最后两个 stage 中的 3×3 卷积替换为 MHSA
    """
    def __init__(
        self,
        num_classes: int = 1000,
        layers: list = [3, 4, 6, 3],  # ResNet-50 配置
        widths: list = [64, 256, 512, 1024, 2048],  # 各 stage 输出通道
        num_heads: int = 4,
        head_dim: int = 64,
        use_rel_pos: bool = False,
    ):
        super().__init__()

        # Stage 1: 7×7 conv + MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2: 标准 ResNet Bottleneck（不使用 MHSA）
        self.layer1 = self._make_layer(
            64, widths[1], layers[0], stride=1, use_mhsa=False
        )

        # Stage 3: 标准 ResNet Bottleneck（不使用 MHSA）
        self.layer2 = self._make_layer(
            widths[1], widths[2], layers[1], stride=2, use_mhsa=False
        )

        # Stage 4: BoTNet Bottleneck（使用 MHSA）
        # 此时特征图尺寸: 56/4=14, 14×14
        self.layer3 = self._make_layer(
            widths[2], widths[3], layers[2], stride=2, use_mhsa=True,
            height=14, width=14, num_heads=num_heads, head_dim=head_dim
        )

        # Stage 5: BoTNet Bottleneck（使用 MHSA）
        # 此时特征图尺寸: 14/2=7, 7×7
        self.layer4 = self._make_layer(
            widths[3], widths[4], layers[3], stride=2, use_mhsa=True,
            height=7, width=7, num_heads=num_heads, head_dim=head_dim
        )

        # Global Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[4], num_classes)

        self._init_weights()

    def _make_layer(
        self, in_channels, out_channels, blocks, stride=1,
        use_mhsa=False, height=None, width=None, num_heads=4, head_dim=64
    ):
        layers = []

        # 第一个 block 可能需要下采样
        layers.append(
            BoTNetBlock(
                in_channels, out_channels, stride=stride,
                num_heads=num_heads, head_dim=head_dim,
                use_rel_pos=False,
                height=height, width=width,
            ) if use_mhsa else self._make_resnet_block(
                in_channels, out_channels, stride
            )
        )

        # 后续 blocks（stride=1）
        for _ in range(1, blocks):
            layers.append(
                BoTNetBlock(
                    out_channels, out_channels, stride=1,
                    num_heads=num_heads, head_dim=head_dim,
                    use_rel_pos=False,
                    height=height, width=width,
                ) if use_mhsa else self._make_resnet_block(
                    out_channels, out_channels, 1
                )
            )

        return nn.Sequential(*layers)

    def _make_resnet_block(self, in_channels, out_channels, stride):
        """标准 ResNet Bottleneck（无 MHSA）"""
        mid_channels = out_channels // 4
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                     padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def test_botnet():
    """测试 BoTNet 前向传播"""
    model = BoTNet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_botnet()
```

### 7.2 训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 超参数
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=transform_train
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型
model = BoTNet(num_classes=10).to(DEVICE)

# 优化器
optimizer = optim.SGD(
    model.parameters(),
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")

print("训练完成！")
```

## 8. 手工代码实现

### 8.1 核心 MHSA vs Conv2D 对比

```python
def compare_mhsa_vs_conv():
    """
    比较 MHSA 和 3×3 Conv 的计算差异
    """
    import time

    B, C, H, W = 4, 256, 14, 14
    x = torch.randn(B, C, H, W)

    # 3×3 Conv
    conv = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
    # MHSA
    mhsa = MHSA(dim=C, num_heads=4, head_dim=64)

    # 前向传播
    out_conv = conv(x)
    out_mhsa = mhsa(x)

    print(f"Conv 输出形状: {out_conv.shape}")
    print(f"MHSA 输出形状: {out_mhsa.shape}")

    # 参数量对比
    conv_params = sum(p.numel() for p in conv.parameters())
    mhsa_params = sum(p.numel() for p in mhsa.parameters())
    print(f"Conv 参数量: {conv_params:,}")
    print(f"MHSA 参数量: {mhsa_params:,}")

    # 速度对比
    # Conv
    start = time.time()
    for _ in range(100):
        out_conv = conv(x)
    conv_time = time.time() - start

    # MHSA
    start = time.time()
    for _ in range(100):
        out_mhsa = mhsa(x)
    mhsa_time = time.time() - start

    print(f"Conv 100次推理: {conv_time:.4f}s")
    print(f"MHSA 100次推理: {mhsa_time:.4f}s")

    # 关键差异：MHSA 的注意力图
    print("\n关键差异分析:")
    print("Conv: 固定感受野 (3×3)，权重与输入无关")
    print("MHSA: 全局感受野 (14×14)，权重由输入决定")
```

### 8.2 BoTNet 的消融实验

```python
def ablation_study():
    """
    BoTNet 消融实验：在不同 stage 替换卷积为 MHSA 的对比
    """
    results = {}

    # 配置：替换不同的 stage
    configs = [
        ("ResNet-50 (baseline)", [False, False, False, False]),
        ("Replace stage 4 only", [False, False, True, False]),
        ("Replace stage 5 only", [False, False, False, True]),
        ("Replace stages 4&5",   [False, False, True, True]),  # BoTNet-50
    ]

    for name, stages in configs:
        # 简化的参数量计算
        params_stages = {
            "stage2": 256 * 64 + 3 * 3 * 64 * 64 + 64 * 256,  # 标准 Bottleneck
            "stage3": 512 * 128 + 3 * 3 * 128 * 128 + 128 * 512,
            "stage4_conv": 1024 * 256 + 3 * 3 * 256 * 256 + 256 * 1024,
            "stage4_mhsa": 1024 * 256 + (256 * 256 * 3 + 256 * 256) + 256 * 1024,
            "stage5_conv": 2048 * 512 + 3 * 3 * 512 * 512 + 512 * 2048,
            "stage5_mhsa": 2048 * 512 + (512 * 512 * 3 + 512 * 512) + 512 * 2048,
        }

        total = 0
        total += params_stages["stage2"] * 4  # 4 blocks
        total += params_stages["stage3"] * 6  # 6 blocks
        total += (params_stages["stage4_mhsa"] if stages[2] else params_stages["stage4_conv"]) * 3
        total += (params_stages["stage5_mhsa"] if stages[3] else params_stages["stage5_conv"]) * 3

        results[name] = total

    for name, params in results.items():
        print(f"{name}: {params/1e6:.2f}M parameters")
```

## 9. 可视化与结果理解

### 9.1 注意力图 vs 卷积核可视化

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_mhsa_vs_conv():
    """
    可视化 MHSA 的注意力图与卷积核的区别
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 行 1: 3×3 卷积核可视化（固定权重）
    conv_kernel = np.random.randn(3, 3)
    im = axes[0, 0].imshow(conv_kernel, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('3×3 Conv Kernel\n(shared for all locations)')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0])

    # 对于不同位置，卷积核是相同的
    for i in range(3):
        axes[0, i+1].imshow(conv_kernel, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, i+1].set_title(f'Location {i+1}: same kernel')
        axes[0, i+1].axis('off')

    # 行 2: MHSA 注意力图（内容自适应）
    np.random.seed(42)
    for i in range(4):
        # 不同位置，注意力权重不同（取决于该位置的 query 和所有位置的 keys）
        attn_map = np.random.rand(14, 14)
        # 模拟关注不同区域
        center = np.random.randint(0, 14, 2)
        for y in range(14):
            for x in range(14):
                dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                attn_map[y, x] = np.exp(-dist / 2)

        attn_map /= attn_map.sum()
        im = axes[1, i].imshow(attn_map, cmap='viridis')
        axes[1, i].set_title(f'MHSA Location {i+1}\n(content-adaptive)')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.show()


def visualize_botnet_architecture():
    """
    可视化 BoTNet 架构与 ResNet 的对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (title, colors) in zip(axes, [
        ("ResNet-50 Architecture", 'lightblue'),
        ("BoTNet-50 Architecture", 'lightgreen'),
    ]):
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 12)
        ax.set_title(title, fontsize=14)

        stages = [
            ("Stage 1", 0, "7×7 Conv\n+ MaxPool"),
            ("Stage 2", 2, "Bottleneck ×3\n(1×1→3×3→1×1)"),
            ("Stage 3", 4, "Bottleneck ×4\n(1×1→3×3→1×1)"),
            ("Stage 4", 6, "Bottleneck ×6\n(1×1→3×3→1×1)" if "ResNet" in title
                          else "BoT Block ×6\n(1×1→MHSA→1×1)"),
            ("Stage 5", 8, "Bottleneck ×3\n(1×1→3×3→1×1)" if "ResNet" in title
                          else "BoT Block ×3\n(1×1→MHSA→1×1)"),
        ]

        for name, y, desc in stages:
            rect = plt.Rectangle(
                (1, y), 12, 1.5, fill=True,
                facecolor=colors, edgecolor='blue', alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(7, y + 0.75, f"{name}: {desc}",
                   ha='center', va='center', fontsize=10)

        # 标注分辨率变化
        resolutions = [
            (2.5, "56×56"),
            (4.5, "28×28"),
            (6.5, "14×14"),
            (8.5, "7×7"),
        ]
        for y, res in resolutions:
            ax.text(13.5, y + 0.75, res, va='center', fontsize=9, color='red')

        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 ImageNet 评估

```python
from sklearn.metrics import accuracy_score


def evaluate_botnet(model, dataloader, device):
    """
    评估 BoTNet 在 ImageNet 上的性能
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, preds = outputs.topk(5, dim=1)  # Top-5 预测
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Top-1
    top1 = (all_preds[:, 0] == all_labels).float().mean().item()

    # Top-5
    top5 = (all_preds == all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")

    return top1, top5
```

### 10.2 与 ResNet 对比

| 模型 | 参数量 | ImageNet Top-1 | FLOPs | 推理速度 |
|------|--------|----------------|-------|---------|
| ResNet-50 | 25.6M | 76.0% | 4.1G | 1x |
| ResNet-101 | 44.5M | 77.4% | 7.9G | 0.7x |
| BoTNet-50 | 23.5M | 80.5% | 5.2G | 0.85x |
| BoTNet-101 | 41.2M | 81.7% | 9.6G | 0.6x |

BoTNet-50 以更少的参数达到了比 ResNet-101 更高的准确率。

## 11. 常见问题与易错点

### 11.1 MHSA 的输入输出维度

**问题**：MHSA 替换 3×3 卷积时，输入输出通道如何匹配？

**答案**：在 Bottleneck 中，3×3 卷积的输入输出通道都是 `mid_channels`（即 out_channels/4）。MHSA 也保持相同的通道数。

### 11.2 位置编码的使用

**问题**：为什么 BoTNet 需要位置编码而标准的卷积不需要？

**答案**：自注意力是"置换等变"的——如果打乱输入 token 的顺序，输出也会按相同顺序打乱。对于图像来说，这意味着 MHSA 无法区分"左上角的 patch"和"右下角的 patch"。位置编码为每个 token 提供了空间位置信息。

### 11.3 下采样时如何使用 MHSA

**问题**：当 stride=2 进行下采样时，MHSA 如何处理？

**答案**：BoTNet 有两种处理方式：
1. 在 MHSA 之前先使用 stride=2 的 depthwise conv 进行下采样
2. 使用注意力池化（attention pooling）代替下采样

### 11.4 BN 和 LN 的选择

**问题**：BoTNet 使用 BatchNorm 而非 LayerNorm，是否合理？

**答案**：BoTNet 整体保留了 ResNet 的 BN 设计（与卷积配合使用），只在 MHSA 中不使用 normalization（或者使用 BN）。实验表明这在小到中等规模的数据集上工作良好。

## 12. 学习总结

### 12.1 核心贡献

BoTNet 的核心贡献是：

1. **展示了 CNN + Transformer 混合设计的简单有效性**：仅替换一个组件就带来显著提升
2. **验证了"渐进式引入注意力"的策略**：在高语义、低分辨率下使用注意力最有效
3. **桥接了 CNN 和 Transformer 两个领域**：为后续混合模型（如 CoAtNet、ConvNeXt）奠定了基础

### 12.2 设计原则

BoTNet 的成功体现了几个重要原则：
- **最小入侵设计**：尽可能少地修改现有架构
- **渐进式复杂度**：在浅层使用计算高效的卷积，深层使用表达力强的注意力
- **层次化特征**：保留了 ResNet 的多尺度特征金字塔

### 12.3 与 ViT 的关键区别

| 方面 | ViT | BoTNet |
|------|-----|--------|
| 骨干架构 | 纯 Transformer | ResNet + MHSA |
| 局部特征 | patch embedding | 卷积 |
| 全局建模 | 所有层自注意力 | 最后 stages 自注意力 |
| 计算效率 | O(N²) 所有层 | O(N²) 仅在低分辨率层 |
| 数据需求 | 大量 | 适中（继承 CNN 先验） |

## 13. 练习题与思考题

### 13.1 基础题

**题目 1**：BoTNet 将 ResNet 中的哪个组件替换掉了？替换成了什么？

**答案**：将 ResNet Bottleneck 中的 3×3 空间卷积替换成了多头自注意力（MHSA）。

**题目 2**：为什么 BoTNet 只在最后两个 stage 替换 3×3 卷积？

**答案**：1. 计算效率：自注意力 O(N²)，stage 2/3 的特征图尺寸大（56×56, 28×28），O(N²) 不可接受。2. 语义层次：深层特征具有更高的语义层次，适合全局建模。3. stage 4/5 特征图大小为 14×14 和 7×7，计算量可接受。

**题目 3**：BoTNet 的参数量相比标准 ResNet 是增加了还是减少了？为什么？

**答案**：减少了。因为 3×3 卷积的参数量为 3×3×C×C = 9C²，而 MHSA 的 QKV 投影参数量为 3×C×C = 3C²，输出投影为 C×C = C²，总计约 4C²。当 C 较大时，MHSA 的参数量约为 3×3 Conv 的一半。

### 13.2 进阶题

**题目 4**：BoTNet 能否在第一层（7×7 Conv）也使用 MHSA？为什么？

**答案**：不太可行。第一层输入为 224×224×3，MHSA 的复杂度为 O(224² × 224²) × C，计算量巨大。此外，第一层需要提取低级特征（边缘、颜色），卷积比注意力更适合这种任务。

**题目 5**：BoTNet 中的位置编码对于最终性能的影响有多大？

**答案**：位置编码对 BoTNet 的性能影响显著。没有位置编码时，MHSA 无法区分不同位置，性能会下降到略高于随机。绝对位置编码和相对位置编码都能有效提供空间信息，相对位置编码通常略好（约 0.3-0.5%）。

### 13.3 思考题

**题目 6**：如果要将 BoTNet 的设计扩展到更高分辨率（如 448×448），需要做哪些调整？

**答案**：
1. 不再在 stage 4（28×28）使用 MHSA，或者使用改进版注意力（如窗口注意力、轴向注意力）
2. 使用可插值的位置编码（因为分辨率变化时位置编码需要调整）
3. 可能需要在 stage 3 也引入注意力（使用高效变体如 depthwise attention）

**题目 7**：BoTNet 的设计思想是否可以应用于 MobileNet 或 ShuffleNet 等轻量级网络？

**答案**：可以，但需要取舍。MHSA 在轻量级网络中的计算开销相对较高（因为轻量级网络本身的 FLOPs 很小）。可以考虑在最后 1-2 个 stage 使用 MHSA，但需要减小 head_dim 或 num_heads 来控制计算量。

## 14. 学习路径建议

### 14.1 前置知识

1. **ResNet**：理解 Bottleneck 设计、残差连接、stage 结构
2. **多头注意力机制**：理解 QKV 投影、注意力计算
3. **CNN vs Transformer 的区别**：感受野、权重共享、内容自适应

### 14.2 学习步骤

1. **第一步**：阅读原论文《Bottleneck Transformers for Visual Recognition》
2. **第二步**：理解 MHSA 替换 3×3 卷积的动机和方式
3. **第三步**：在 ResNet 基础上实现 BoTNet
4. **第四步**：在 CIFAR-100 上对比 ResNet 和 BoTNet 的性能
5. **第五步**：可视化注意力图，观察 MHSA 的全局关注模式
6. **第六步**：尝试将 MHSA 集成到其他 CNN 架构中

### 14.3 相关论文推荐

- BoTNet (Srinivas et al., 2021)：原论文
- ResNet (He et al., 2016)：基础架构
- ViT (Dosovitskiy et al., 2020)：Vision Transformer
- CoAtNet (Dai et al., 2021)：卷积 + 注意力的进一步探索
- ConvNeXt (Liu et al., 2022)：纯 CNN 的现代化设计

### 14.4 实践建议

1. 在 ImageNet 上对比 ResNet-50 和 BoTNet-50 的训练曲线
2. 在 COCO 上测试 BoTNet 作为检测 backbone 的效果
3. 尝试不同的 MHSA 配置（heads 数、head_dim、是否使用相对位置编码）
4. 将 BoTNet 的思想应用到 ResNeXt、ResNeSt 等变体中
