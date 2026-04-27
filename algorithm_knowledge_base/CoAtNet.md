# CoAtNet (Convolution + Attention Network) 学习文档

## 1. 算法基础认知

### 1.1 算法要解决什么问题

在 BoTNet 成功展示了"用注意力替换卷积"的有效性之后，一个自然的问题是：**如何更系统地将卷积和自注意力融合到一个统一的架构中？**

卷积和自注意力各有优劣：

| 特性 | 卷积 | 自注意力 |
|------|------|---------|
| 局部建模 | ✅ 强（通过小卷积核） | ❌ 弱（需要大模型学习局部模式） |
| 全局建模 | ❌ 弱（需要堆叠很多层） | ✅ 强（直接全局交互） |
| 平移等变性 | ✅ 天生具备 | ❌ 需要位置编码 |
| 输入自适应 | ❌ 权重固定 | ✅ 注意力取决于输入 |
| 计算效率（高分辨率） | ✅ O(K²·H·W) | ❌ O(H²·W²) |
| 计算效率（低分辨率） | ✅ O(K²·H·W) | ✅ 仍能接受 |

CoAtNet（由 Dai 等人于 2021 年提出）的核心目标是：**设计一个能根据分辨率自动平衡卷积和注意力的统一架构，在浅层（高分辨率）使用卷积，在深层（低分辨率）使用注意力，并让两者深度融合。**

### 1.2 核心思路概览

CoAtNet 的设计有两大创新：

**创新 1：深度可分离卷积 + 相对自注意力的统一形式**

CoAtNet 发现了一个重要的数学关系——**带相对位置编码的自注意力可以写成深度可分离卷积的一种广义形式**：

```
深度可分离卷积:     y_i = Σ_j w_{i-j} ⊙ x_j          (权重取决于相对位置)
相对自注意力:       y_i = Σ_j (Softmax(Q_i·K_j/√d + B_{i-j})_j ⊙ v_j   (注意力取决于内容和相对位置)
```

关键区别在于：
- 卷积：权重仅取决于相对位置（w_{i-j}），与内容无关
- 注意力：权重取决于内容（Q_i·K_j）和相对位置（B_{i-j}）

**创新 2：渐进式融合架构**

CoAtNet 设计了 5 个 stage，每个 stage 内部使用不同比例的卷积和注意力：

```
Stage 1 (S1):   Conv  (高分辨率, 纯卷积)
Stage 2 (S2):   Conv  (较高分辨率, 纯卷积)
Stage 3 (S3):   MBConv (中等分辨率, 卷积为主)
Stage 4 (S4):   MBConv + Attention (低分辨率, 混合)
Stage 5 (S5):   Attention (最低分辨率, 纯注意力)
```

其中 MBConv 是 MobileNetV2 中的反向瓶颈块（扩展 → depthwise conv → 压缩）。

### 1.3 整体架构

```
输入 (224×224×3)
  ↓
S1: Conv 3×3 (stride 2) → 112×112×C
  ↓
S2: MBConv × L2 (stride 2) → 56×56×C
  ↓
S3: MBConv × L3 (stride 2) → 28×28×4C
  ↓
S4: MBConv + Attn × L4 (stride 2) → 14×14×8C
  ↓
S5: Attn × L5 → 7×7×16C
  ↓
Global Avg Pooling → FC → 分类
```

## 2. 核心原理

### 2.1 深度可分离卷积回顾

标准卷积：
```
Conv(X)_i = Σ_j W_{i-j} · X_j
```
其中 i 是输出位置，j 是输入位置，W 是卷积核。

深度可分离卷积分为两步：
1. **Depthwise Conv**：每个通道独立进行空间卷积
2. **Pointwise Conv (1×1)**：在通道维度进行线性组合

```
Depthwise(X)_i,c = Σ_j W_{i-j,c} · X_j,c
Pointwise(Y)_i = Y_i · W_pw
```

### 2.2 相对自注意力

带相对位置编码的自注意力：
```
Attention(Q,K,V)_i = Σ_j (Softmax(Q_i·K_j/√d + B_{i-j})_j · V_j
```
其中 B_{i-j} 是位置 i 和 j 之间的相对位置编码。

### 2.3 统一视角

将深度可分离卷积写作：
```
y_i = Σ_j w_{i-j} ⊙ x_j
```

将相对自注意力写作：
```
y_i = Σ_j a_{i,j}(x) ⊙ v_j
```
其中 a_{i,j}(x) = Softmax(Q_i·K_j/√d + B_{i-j})_j

CoAtNet 的核心洞察是：这两个操作可以通过**注意力合并（attentive pooling）** 统一起来，在浅层使用卷积权重（a_{i,j} 只依赖于相对位置），在深层使用内容自适应权重。

### 2.4 MBConv (Mobile Inverted Bottleneck)

MBConv 是 CoAtNet 中用于早期 stage 的基础模块：

```
输入 (H×W×C_in)
  ↓
1×1 Conv (扩展 ×E): C_in → E·C_in + BN + GeLU
  ↓
Depthwise Conv 3×3: E·C_in → E·C_in + BN + GeLU
  ↓
SE Module (Squeeze-and-Excitation)
  ↓
1×1 Conv (压缩): E·C_in → C_out + BN
  ↓
+ 残差连接 (如果 C_in = C_out 且 stride=1)
  ↓
输出 (H'×W'×C_out)
```

### 2.5 CoAtNet Block（带注意力的 MBConv）

在后期 stage，将 depthwise conv 替换为相对注意力（或两者混合）：

```
输入 (H×W×C_in)
  ↓
1×1 Conv (扩展): C_in → E·C_in + BN + GeLU
  ↓
Relative Self-Attention (替换 Depthwise Conv)
  ↓
SE Module (可选)
  ↓
1×1 Conv (压缩): E·C_in → C_out + BN
  ↓
+ 残差连接
  ↓
输出 (H'×W'×C_out)
```

### 2.6 Swish / SiLU 激活函数

CoAtNet 使用 Swish（SiLU）激活函数：
```
Swish(x) = x · σ(x)
```
其中 σ 是 sigmoid 函数。

相比 ReLU，Swish 的优点是：
- 处处可微（ReLU 在 0 处不可导）
- 具有轻微的非零负值（减少神经元死亡）
- 实验表明在注意力模型中表现更好

## 3. 数学公式与推导

### 3.1 相对自注意力公式

输入 X ∈ ℝ^{H×W×C}，展开为序列形式 X ∈ ℝ^{N×C} (N=H·W)：

```
Q = X · W_Q ∈ ℝ^{N×d_k}
K = X · W_K ∈ ℝ^{N×d_k}
V = X · W_V ∈ ℝ^{N×d_v}

相对注意力得分:
A_{ij} = Q_i · K_j / √d_k + B_{ij}

输出:
O_i = Σ_j Softmax(A)_j · V_j

B_{ij} = B_{h_i-h_j, w_i-w_j}  (二维相对位置偏置)
```

### 3.2 二维相对位置编码

B 的参数量为 (2H-1) × (2W-1) × num_heads：

```
B = B_table[h_i - h_j + H - 1, w_i - w_j + W - 1, head]
```

其中 B_table 是可学习的参数表，大小为 (2H-1) × (2W-1) × H。

### 3.3 MBConv + Attention 的完整前向

```python
def forward(self, x):
    identity = x

    # 扩展
    x = self.conv1(x)     # 1×1, C_in → E·C_in
    x = self.norm1(x)
    x = self.act(x)

    if self.use_attn:
        # 相对自注意力（替换 depthwise conv）
        x = self.attn(x)
    else:
        # Depthwise conv
        x = self.conv_dw(x)  # 3×3 depthwise
        x = self.norm_dw(x)
        x = self.act(x)

    # SE 模块
    x = self.se(x)

    # 压缩
    x = self.conv2(x)     # 1×1, E·C_in → C_out
    x = self.norm2(x)

    # 残差连接
    if self.stride == 1 and identity.shape == x.shape:
        x = x + identity

    return x
```

### 3.4 CoAtNet 的计算量配置

CoAtNet 有多个变体，不同变体的计算量不同：

| 模型 | #Params | FLOPs | S1 | S2 | S3 | S4 | S5 |
|------|---------|-------|----|----|----|----|-----|
| CoAtNet-0 | 25M | 4.2G | C=64, L=2 | C=96, L=2 | C=192, L=3 | C=384, L=5 | C=768, L=2 |
| CoAtNet-1 | 42M | 8.4G | C=64, L=2 | C=96, L=2 | C=192, L=6 | C=384, L=10 | C=768, L=2 |
| CoAtNet-2 | 75M | 15.7G | C=64, L=2 | C=128, L=2 | C=256, L=6 | C=512, L=10 | C=1024, L=2 |
| CoAtNet-3 | 168M | 34.9G | C=96, L=2 | C=192, L=2 | C=384, L=6 | C=768, L=14 | C=1536, L=2 |

## 4. 训练过程讲解

### 4.1 训练配置

CoAtNet 在 ImageNet-1K 上的训练配置：

- **优化器**：AdamW (β₁=0.9, β₂=0.999, eps=1e-8)
- **学习率**：5e-4 (cosine decay)
- **权重衰减**：0.05
- **批次大小**：2048
- **训练轮数**：300 epoch + 3 epochs finetune
- **标签平滑**：0.1
- **Dropout**：0.0
- **Stochastic Depth**：0.05 (S3), 0.1 (S4), 0.2 (S5)
- **梯度裁剪**：global norm 5.0

### 4.2 数据增强

- RandomResizedCrop (224×224)
- RandAugment (2, 15) 
- Mixup (α=0.8)
- CutMix (α=1.0)
- RandomErasing (p=0.25)
- ColorJitter (0.4)

### 4.3 ImageNet-21K 预训练

CoAtNet 通常先在 ImageNet-21K 上预训练：
- **学习率**：1e-3
- **训练轮数**：90 epoch
- **输入分辨率**：224×224

然后在 ImageNet-1K 上微调：
- **学习率**：1e-4
- **微调轮数**：30 epoch

### 4.4 分辨率微调

CoAtNet 支持在更高分辨率（如 384×384）上微调：
- 相对位置编码的参数表可以插值适应更高分辨率
- 微调时位置编码重新初始化，保持性能

## 5. 应用场景

### 5.1 图像分类

CoAtNet 在 ImageNet 上的 SOTA 性能：
- CoAtNet-0: 81.6% top-1
- CoAtNet-1: 83.3% top-1
- CoAtNet-2: 84.5% top-1
- CoAtNet-3: 85.5% top-1
- CoAtNet-3 (384): 85.9% top-1
- CoAtNet-3 (512): 86.0% top-1

### 5.2 迁移学习

在多个下游任务上的表现：
- CIFAR-100: 95.6%
- Oxford Flowers: 99.4%
- Stanford Cars: 95.2%
- iNat 2019: 84.9%

### 5.3 目标检测和分割

可以作为 backbone 用于：
- RetinaNet (目标检测)
- Mask R-CNN (实例分割)
- DeepLabV3 (语义分割)

## 6. 优缺点分析

### 6.1 优点

1. **系统性的卷积+注意力融合**：不再是简单的"替换"，而是从数学形式上统一
2. **渐进式架构**：从纯卷积到纯注意力的平滑过渡
3. **相对位置编码高效**：对高分辨率友好
4. **SOTA 性能**：在 ImageNet 上达到 86%+ top-1 准确率
5. **计算效率高**：在相似 FLOPs 下优于同类模型

### 6.2 缺点

1. **架构复杂**：多个 stage 设计、不同 block 类型，实现复杂
2. **预训练依赖**：最优性能需要 ImageNet-21K 预训练
3. **超参数多**：每个 stage 的通道数、层数、是否使用注意力都需要调优
4. **对迁移学习任务**：小数据集上可能不如纯 CNN 稳定（注意力需要足够数据）

## 7. 调库实现

### 7.1 CoAtNet 的完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    """Swish / SiLU 激活函数"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation 模块
    通过全局平均池化 + 两个 FC 层，为每个通道生成自适应权重
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Squeeze: 全局平均池化
        y = x.mean(dim=[2, 3])  # (B, C)
        # Excitation
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(B, C, 1, 1)
        # Scale
        return x * y


class RelativeAttention(nn.Module):
    """
    带相对位置编码的多头自注意力
    用于替换 depthwise convolution
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        height: int = 14,
        width: int = 14,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.height = height
        self.width = width

        # QKV 投影
        self.qkv = nn.Linear(dim, num_heads * head_dim * 3)
        self.proj = nn.Linear(num_heads * head_dim, dim)

        # 相对位置编码表
        # 对于 H×W 的特征图，相对位置范围为 [-(H-1), H-1] 和 [-(W-1), W-1]
        # 每个 head 独立的位置偏置
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(2 * height - 1, 2 * width - 1, num_heads)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def _get_rel_pos_bias(self, H: int, W: int) -> torch.Tensor:
        """
        构建相对位置编码偏置矩阵
        返回: (1, num_heads, H*W, H*W)
        """
        # 位置索引
        pos_h = torch.arange(H, device=self.rel_pos_bias.device)
        pos_w = torch.arange(W, device=self.rel_pos_bias.device)

        # 相对位置: 位置 i 相对于位置 j
        rel_h = pos_h[:, None] - pos_h[None, :]  # (H, H)
        rel_w = pos_w[:, None] - pos_w[None, :]  # (W, W)

        # 映射到 [0, 2H-2] 和 [0, 2W-2]
        rel_h += H - 1
        rel_w += W - 1

        # 索引相对位置编码表
        # rel_pos_bias 形状: (2H-1, 2W-1, num_heads)
        bias = self.rel_pos_bias[rel_h.flatten()][:, rel_w.flatten()]
        # bias 形状: (H*H, W*W, num_heads)

        # 重排: (H*H, W*W, num_heads) -> (1, num_heads, H*W, H*W)
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias

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
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # 相对位置偏置
        pos_bias = self._get_rel_pos_bias(H, W)  # (1, H, N, N)
        attn = attn + pos_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = attn @ V  # (B, H, N, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)
        out = self.proj(out)

        # 恢复 2D 形状
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class CoAtNetBlock(nn.Module):
    """
    CoAtNet Block
    支持 MBConv（纯卷积）和 MBConv + Attention（混合）
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
        use_attn: bool = False,
        num_heads: int = 8,
        head_dim: int = 32,
        height: int = 14,
        width: int = 14,
        se_reduction: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_attn = use_attn

        mid_channels = in_channels * expand_ratio

        # 扩展卷积 (1×1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        # 空间特征提取
        if use_attn:
            # 相对自注意力
            self.attn = RelativeAttention(
                mid_channels, num_heads, head_dim,
                height=height, width=width, dropout=dropout,
            )
        else:
            # Depthwise Conv
            self.conv_dw = nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3,
                stride=stride, padding=1,
                groups=mid_channels, bias=False,
            )
            self.norm_dw = nn.BatchNorm2d(mid_channels)

        self.act = Swish()

        # SE 模块
        self.se = SqueezeExcitation(mid_channels, se_reduction)

        # 压缩卷积 (1×1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 扩展
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # 空间特征提取
        if self.use_attn:
            x = self.attn(x)
        else:
            x = self.conv_dw(x)
            x = self.norm_dw(x)
            x = self.act(x)

        # SE
        x = self.se(x)

        # 压缩
        x = self.conv2(x)
        x = self.norm2(x)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity

        return x


class CoAtNet(nn.Module):
    """
    CoAtNet: 卷积 + 注意力的统一网络
    渐进式地从纯卷积过渡到纯注意力
    """
    def __init__(
        self,
        num_classes: int = 1000,
        num_blocks: list = [2, 2, 3, 5, 2],  # 各 stage 的 block 数量
        channels: list = [64, 96, 192, 384, 768],  # 各 stage 的输出通道
        expand_ratios: list = [1, 4, 4, 4, 4],  # MBConv 扩展倍率
        use_attn_stages: list = [False, False, False, True, True],  # 哪些 stage 使用注意力
        num_heads: int = 8,
        head_dim: int = 32,
    ):
        super().__init__()

        # Stage 0: 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish(),
        )

        # Stages 1-5
        in_channels = channels[0]
        self.stages = nn.ModuleList()
        curr_h, curr_w = 112, 112  # 初始特征图尺寸

        for stage_idx in range(5):
            stage_blocks = []
            out_channels = channels[stage_idx]
            expand = expand_ratios[stage_idx]
            use_attn = use_attn_stages[stage_idx]

            for block_idx in range(num_blocks[stage_idx]):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1

                if stride == 2:
                    curr_h //= 2
                    curr_w //= 2

                block = CoAtNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand,
                    use_attn=use_attn,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    height=curr_h,
                    width=curr_w,
                )
                stage_blocks.append(block)
                in_channels = out_channels

            self.stages.append(nn.Sequential(*stage_blocks))

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def get_stage_features(self, x: torch.Tensor) -> list:
        """
        获取各 stage 的输出特征（用于检测/分割）
        """
        features = []
        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


def test_coatnet():
    """测试 CoAtNet 前向传播"""
    model = CoAtNet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_coatnet()
```

### 7.2 使用 timm 库

```python
import torch
from timm.models import create_model

# 创建 CoAtNet 预训练模型
model = create_model(
    'coatnet_0_rw_224',  # CoAtNet-0
    pretrained=True,
    num_classes=1000,
)

# 查看模型结构
print(f"CoAtNet-0 参数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试前向传播
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"输出形状: {out.shape}")

# 获取多尺度特征
model.eval()
with torch.no_grad():
    # 有些 timm 模型实现了 forward_features 方法
    features = model.forward_features(x)
    print(f"特征形状: {features.shape}")
```

## 8. 手工代码实现

### 8.1 卷积 vs 注意力计算量对比

```python
def compare_complexity():
    """
    对比不同分辨率下卷积和注意力的计算量
    """
    # 假设: C=384, K=3 (卷积核大小), num_heads=8, head_dim=32
    C, K, H, W = 384, 3, 8, 32

    print("不同特征图尺寸下的计算量对比 (FLOPs)")
    print("=" * 70)
    print(f"{'尺寸':<15} {'卷积 (MBConv)':<20} {'注意力 (MHSA)':<20} {'比值':<10}")
    print("=" * 70)

    resolutions = [(112, 112), (56, 56), (28, 28), (14, 14), (7, 7)]

    for h, w in resolutions:
        # 卷积: O(C * K² * H * W)  (depthwise conv)
        # 实际是 depthwise: C * K² * H * W
        conv_flops = C * K * K * h * w * 2  # *2 for multiply-add

        # 注意力: O(4 * C * H * W + 2 * (H * W)² * num_heads)
        # QKV 投影: 3 * C * (num_heads*head_dim) * H * W
        # 注意力矩阵: 2 * (H*W)² * num_heads
        # 输出投影: (num_heads*head_dim) * C * H * W
        attn_flops = 4 * C * C * h * w + 2 * (h * w) ** 2 * 8

        ratio = attn_flops / conv_flops if conv_flops > 0 else float('inf')

        print(f"{h}×{w:<10} {conv_flops/1e6:<20.2f}M {attn_flops/1e6:<20.2f}M {ratio:<10.2f}")

    print("=" * 70)
    print("结论: 高分辨率时卷积效率高，低分辨率时注意力效率可接受")
```

### 8.2 CoAtNet 的消融实验

```python
def coatnet_ablation():
    """
    CoAtNet 消融实验：不同 stage 使用注意力 vs 卷积
    """
    configs = [
        # (use_attn_stages, description)
        ([False, False, False, False, False], "Full Conv (MBOnly)"),
        ([False, False, False, True, True], "CoAtNet Default"),
        ([True, True, True, True, True], "Full Attention"),
        ([False, False, True, True, True], "Attn from S3"),
    ]

    print("CoAtNet 消融实验")
    print("=" * 60)

    for use_attn, desc in configs:
        model = CoAtNet(
            num_classes=100,
            use_attn_stages=use_attn,
        )

        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, 224, 224)

        # 粗略估计 FLOPs
        out = model(x)

        print(f"{desc:<25} | Params: {params/1e6:.2f}M")

    print("=" * 60)
    print("典型结果预期:")
    print("Full Conv: 最低 FLOPs, 较低性能")
    print("CoAtNet Default: 平衡 FLOPs 和性能")
    print("Full Attention: 高 FLOPs, 小数据集上可能过拟合")
```

## 9. 可视化与结果理解

### 9.1 感受野对比

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_receptive_field():
    """
    可视化 CoAtNet 不同 stage 的等效感受野
    卷积层感受野小，注意力层感受野全局
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Stage 2 (纯卷积) - 小感受野
    rf_conv = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            dist = np.sqrt((i - 14)**2 + (j - 14)**2)
            rf_conv[i, j] = max(0, 1 - dist / 5)  # 局部感受野

    im = axes[0].imshow(rf_conv, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Stage 2 (Conv)\nLocal Receptive Field\n(≈ 11×11 after 2 layers)')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])

    # Stage 4 (混合) - 中等感受野
    rf_mixed = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            dist = np.sqrt((i - 7)**2 + (j - 7)**2)
            rf_mixed[i, j] = np.exp(-dist / 10)  # 中间区域注意力强

    im = axes[1].imshow(rf_mixed, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Stage 4 (Conv + Attn)\nMedium Receptive Field\n(content-adaptive)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    # Stage 5 (纯注意力) - 全局感受野
    rf_attn = np.ones((7, 7)) * 0.5
    rf_attn[3, 3] = 1.0  # 中心位置关注自身
    im = axes[2].imshow(rf_attn, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Stage 5 (Attention)\nGlobal Receptive Field\n(all pixels attended)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])

    plt.suptitle('CoAtNet: Progressive Receptive Field Expansion', fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_coatnet_stages():
    """
    可视化 CoAtNet 各 stage 的架构组成
    """
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    stage_info = [
        ("Stem", "Conv 3×3\nstride 2", "112×112", 'lightblue'),
        ("S1", "MBConv ×2\nstride 1", "112×112", 'lightblue'),
        ("S2", "MBConv ×2\nstride 2", "56×56", 'lightblue'),
        ("S3", "MBConv ×3\nstride 2", "28×28", 'lightgreen'),
        ("S4", "Attn ×5\nstride 2", "14×14", 'lightsalmon'),
        ("S5", "Attn ×2\nstride 2", "7×7", 'lightcoral'),
    ]

    for i, (name, desc, res, color) in enumerate(stage_info):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                            facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)

        ax.text(0.5, 0.65, name, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.45, desc, ha='center', va='center', fontsize=9)
        ax.text(0.5, 0.2, res, ha='center', va='center', fontsize=10, color='darkred')

        ax.axis('off')

    plt.suptitle('CoAtNet Architecture: Progressive Conv → Attention Transition', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### 9.2 训练曲线

```python
def plot_training_comparison():
    """
    绘制 CoAtNet 与其他模型的训练曲线对比
    """
    epochs = np.arange(1, 301)

    # 模拟数据
    np.random.seed(42)
    base = 0.76

    # ResNet-50: 收敛慢，最终低
    resnet_acc = base + 0.03 * (1 - np.exp(-epochs / 50)) + 0.01 * np.random.randn(300)
    resnet_acc = np.clip(resnet_acc, 0, 1)

    # ViT-B: 需要更多数据，最终中
    vit_acc = base + 0.01 + 0.04 * (1 - np.exp(-epochs / 70)) + 0.01 * np.random.randn(300)
    vit_acc = np.clip(vit_acc, 0, 1)

    # CoAtNet-0: 收敛快，最终高
    coatnet_acc = base + 0.02 + 0.05 * (1 - np.exp(-epochs / 40)) + 0.01 * np.random.randn(300)
    coatnet_acc = np.clip(coatnet_acc, 0, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, resnet_acc * 100, 'b-', label='ResNet-50', linewidth=2, alpha=0.8)
    plt.plot(epochs, vit_acc * 100, 'g-', label='ViT-B/16', linewidth=2, alpha=0.8)
    plt.plot(epochs, coatnet_acc * 100, 'r-', label='CoAtNet-0', linewidth=2, alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', fontsize=12)
    plt.title('Training Convergence Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(70, 85)

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 ImageNet 评估

```python
from sklearn.metrics import accuracy_score


def evaluate_coatnet(model, dataloader, device):
    """
    评估 CoAtNet 模型
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = outputs.topk(5, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    top1 = (all_preds[:, 0] == all_labels).float().mean().item()
    top5 = (all_preds == all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")

    return top1, top5
```

### 10.2 性能对比

| 模型 | 参数量 | FLOPs | ImageNet Top-1 | ImageNet-21K FT |
|------|--------|-------|----------------|-----------------|
| ResNet-50 | 25.6M | 4.1G | 76.0% | - |
| EfficientNet-B5 | 30M | 9.9G | 83.6% | - |
| ViT-B/16 | 86M | 17.6G | 77.9% | 84.2% |
| BoTNet-50 | 23.5M | 5.2G | 80.5% | - |
| CoAtNet-0 | 25M | 4.2G | 81.6% | - |
| CoAtNet-1 | 42M | 8.4G | 83.3% | - |
| CoAtNet-2 | 75M | 15.7G | 84.5% | 87.1% |
| CoAtNet-3 | 168M | 34.9G | 85.5% | 88.6% |

## 11. 常见问题与易错点

### 11.1 相对位置编码的实现

**问题**：相对位置编码在 batch 中如何高效实现？

**答案**：预计算相对位置索引表，然后在 forward 中查找。对于 H×W 的特征图，预计算一个 (H*W, H*W) 的索引矩阵，然后从 (2H-1)×(2W-1)×num_heads 的参数表中取值。

### 11.2 注意力何时使用

**问题**：在 CoAtNet 中，决定在哪个 stage 使用注意力的依据是什么？

**答案**：主要依据是特征图大小。当特征图 ≤ 14×14 时使用注意力；特征图 ≥ 28×28 时使用卷积。14×14 是经验阈值——此时注意力复杂度 O(196²) 约等于 3×3 卷积的计算量，且注意力能提供全局建模的收益。

### 11.3 Batch 维度处理

**问题**：CoAtNet 使用 BatchNorm 而非 LayerNorm，与传统的 Transformer 不同，为什么？

**答案**：CoAtNet 的早期 stage 主要是卷积层，卷积通常配合 BN。在注意力 stage 中，由于结构仍然是 MBConv 形式（只是替换了 depthwise conv），继续使用 BN 保持一致性。实验表明 BN 在 CoAtNet 中工作良好。

### 11.4 分辨率变化时的位置编码

**问题**：当输入分辨率变化（如 224→384）时，相对位置编码需要如何调整？

**答案**：相对位置编码表 (2H-1)×(2W-1) 需要根据新分辨率重新初始化或插值。通常有两种做法：
1. 对参数表进行双线性插值（bilinear interpolation）
2. 在位置编码表的末尾 pad 零

## 12. 学习总结

### 12.1 核心贡献

CoAtNet 的核心贡献是：

1. **统一了卷积和注意力的数学形式**：通过深度可分离卷积和相对注意力的统一视角
2. **设计了渐进式过渡架构**：从纯卷积（浅层）到纯注意力（深层）
3. **验证了混合设计的有效性**：在 ImageNet 上建立了新的 SOTA

### 12.2 设计哲学

CoAtNet 的设计体现了"因地制宜"的思想：
- 高分辨率 → 低计算开销的卷积（局部优先）
- 低分辨率 → 高表达力的注意力（全局优先）
- 中等分辨率 → 混合使用

### 12.3 与相关方法的对比

| 模型 | 卷积使用 | 注意力使用 | 融合方式 |
|------|---------|-----------|---------|
| BoTNet | ResNet stages | 替换最后 stages 的 3×3 conv | 替换式 |
| CoAtNet | MBConv | 在 MBConv 中替换 depthwise conv | 统一式 |
| ConViT | 门控位置编码 | 标准自注意力 | 门控式 |
| LeViT | 卷积下采样 | 标准自注意力 | 串联式 |

## 13. 练习题与思考题

### 13.1 基础题

**题目 1**：CoAtNet 中卷积和注意力是如何统一的？关键公式是什么？

**答案**：CoAtNet 发现带相对位置编码的自注意力和深度可分离卷积可以统一为"邻域加权和"的形式：
- 深度可分离卷积：y_i = Σ_j w_{i-j} ⊙ x_j（权重仅取决于相对位置）
- 相对自注意力：y_i = Σ_j a(Q_i,K_j,B_{i-j}) · v_j（权重取决于内容和相对位置）
区别在于权重是否与输入内容有关。

**题目 2**：为什么 CoAtNet 在高分辨率（如 112×112）时使用卷积，在低分辨率（如 7×7）时使用注意力？

**答案**：计算效率。自注意力的复杂度为 O(N²)，112×112 时 N=12544，O(12544²) 不可接受。而卷积复杂度 O(K²·H·W) 随分辨率线性增长。7×7 时 N=49，O(49²) 计算量很小，且注意力能提供全局建模能力。

**题目 3**：CoAtNet 中的 SE 模块的作用是什么？

**答案**：SE (Squeeze-and-Excitation) 模块通过全局平均池化获取通道级统计信息，然后通过两个 FC 层生成通道注意力权重。它对每个通道的激活值进行加权，增强重要通道、抑制不重要通道。

### 13.2 进阶题

**题目 4**：CoAtNet 中的相对位置编码相比于 ViT 中的绝对位置编码有什么优势？

**答案**：
1. **平移等变性更好**：相对位置编码关注两个位置之间的相对距离，对平移更鲁棒
2. **分辨率自适应**：可以在不同分辨率之间插值，无需重新训练
3. **参数效率**：参数数量为 (2H-1)×(2W-1)×H，相比绝对位置编码的 N×D，在分辨率变化时更灵活

**题目 5**：如果 CoAtNet 要在视频理解任务中使用，需要做哪些调整？

**答案**：
1. 增加时间维度：将 2D 卷积和注意力扩展为 3D（时空）
2. 相对位置编码扩展为 3D（高度、宽度、时间）
3. Stage 设计可能需要调整（视频分辨率通常较低但帧数多）

### 13.3 思考题

**题目 6**：CoAtNet 的渐进式设计是否可能被一个更通用的"自动选择"机制替代——让网络自己学习在每个位置应该使用卷积还是注意力？

**答案**：这是一个有意义的研究方向。可能的方法包括：
1. **可微架构搜索**：通过架构搜索（NAS）自动决定每层使用 conv 还是 attn
2. **条件计算**：根据输入内容动态选择 conv 或 attn
3. **软加权**：将 conv 和 attn 的输出通过可学习的门控混合

已有工作如 FSA (Flexible Self-Attention) 和 AANet 探索了类似方向。

**题目 7**：CoAtNet 在 ImageNet-21K 上预训练后在 ImageNet-1K 上微调，这种两阶段训练的必要性是什么？

**答案**：两阶段训练的必要性：
1. 注意力组件需要大量数据才能学习良好的全局关系（相比于卷积的强先验）
2. ImageNet-1K (1.2M 图像) 对于大模型（如 CoAtNet-3 的 168M 参数）来说仍然不够
3. ImageNet-21K (14M 图像) 提供了足够的多样性让注意力组件泛化
4. 微调阶段让模型适应目标数据集的分布

这实际上反映了"注意力的数据效率不如卷积"这个根本性问题。

## 14. 学习路径建议

### 14.1 前置知识

1. **ResNet / Bottleneck 设计**：理解残差连接和 stage 结构
2. **MobileNetV2 / MBConv**：理解反向瓶颈块和 depthwise conv
3. **SE-Net**：理解 Squeeze-and-Excitation 模块
4. **ViT 多头注意力**：理解 QKV 机制和自注意力计算
5. **相对位置编码**：理解相对位置编码 vs 绝对位置编码

### 14.2 学习步骤

1. **第一步**：精读原论文《CoAtNet: Marrying Convolution and Attention for All Data Sizes》
2. **第二步**：理解卷积和注意力的统一视角
3. **第三步**：实现 MBConv 和 CoAtNet Block
4. **第四步**：实现完整的 CoAtNet 架构
5. **第五步**：在 CIFAR-100 上对比 CoAtNet 的不同配置
6. **第六步**：使用预训练模型在 ImageNet 上评估
7. **第七步**：探索 CoAtNet 应用于检测/分割任务

### 14.3 相关论文推荐

- CoAtNet (Dai et al., 2021)：原论文
- EfficientNet (Tan & Le, 2019)：MBConv 的提出
- MobileNetV2 (Sandler et al., 2018)：反向瓶颈块
- SE-Net (Hu et al., 2018)：Squeeze-and-Excitation
- BoTNet (Srinivas et al., 2021)：CoAtNet 的前期工作
- ConvNeXt (Liu et al., 2022)：纯 CNN 现代化（受 CoAtNet 启发）
- Swin Transformer (Liu et al., 2021)：窗口注意力（另一种卷积-注意力融合）

### 14.4 实践建议

1. 对比 CoAtNet-0 和 ResNet-50 在相同 FLOPs 下的性能差异
2. 测试不同 expand_ratio（2, 4, 6）对性能和计算量的影响
3. 尝试将相对位置编码替换为其他形式（如 RoPE、ALiBi）
4. 在 COCO 数据集上测试 CoAtNet 作为检测 backbone 的性能
5. 尝试在 CoAtNet 的 attention stage 中使用 Flash Attention 加速
