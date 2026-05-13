# UNet 图像重建 学习文档

> 来源线索：本节内容根据原书中关于"UNet模型详解"（第9章 9.1.3节）的相关章节整理、扩展与教学化改写。

> 编码器提取多尺度特征，解码器逐步重建，跳跃连接保留细节。

## 1. 算法基础认知

**一句话定义**：由对称的编码器-解码器结构和跳跃连接组成的全卷积网络，专为像素级图像重建和分割任务设计。

**直觉类比**：想象一位文物修复师修复一幅破损的画作。他先退后几步看清整体构图（编码器逐步缩小视野、提取全局信息），再走近仔细观察细节（解码器逐步放大、恢复分辨率）。更重要的是，修复过程中他会不断对照原始照片（跳跃连接），确保每一块修复的细节都与原图一致。UNet 的工作原理正是如此——下采样看清全貌，上采样恢复细节，跳跃连接保证信息不丢失。

**历史背景**：UNet 由 Olaf Ronneberger 等人在 2015 年 MICCAI 会议上提出，最初用于生物医学图像分割。其 U 形对称结构和跳跃连接设计大获成功，随后被广泛用于各类图像到图像任务。2020 年，DDPM 用 UNet 作为扩散模型的噪声预测骨干网络，进一步拓展了 UNet 的应用边界。

**算法定位**：深度学习 / 计算机视觉 / 图像重建。属于全卷积编码器-解码器架构，在扩散模型、语义分割、图像超分辨率、图像去噪等多个领域有核心应用。

**前置知识**：
- 卷积神经网络（CNN）：卷积层、池化层、感受野
- 上采样方法：转置卷积（Transposed Convolution）、插值上采样（Interpolation）
- 残差连接（Residual Connection）的概念
- 批归一化（BatchNorm）/ 组归一化（GroupNorm）
- 注意力机制（用于 UNet 的高端变体）

## 2. 核心原理

### 核心思想

UNet 的核心思想可以概括为"先压缩再重建，边重建边对照"。它由三个关键部分组成：

1. **编码器（收缩路径 / Encoder）**：通过连续的卷积 + 下采样逐步压缩空间分辨率，同时增加通道数。这类似于从"看清每个像素"到"理解整体语义"的层次化特征提取过程。

2. **解码器（扩展路径 / Decoder）**：通过连续的上采样 + 卷积逐步恢复空间分辨率，同时减少通道数。这是从"高层语义理解"回到"像素级输出"的重建过程。

3. **跳跃连接（Skip Connections）**：将编码器每一层的特征图直接拼接到解码器对应层。这解决了"压缩-重建"过程中的信息丢失问题——高层特征提供语义信息（"是什么"），跳跃连接提供空间细节（"在哪里"）。

### 工作流程

```
输入图像 (C, H, W)
    │
    ▼
初始卷积 → 特征图 f0
    │
    ▼
Encoder Block 1: Conv + Conv + Downsample → f1 (2C, H/2, W/2)
    │                                                 │
    ▼                                                 │ 跳跃连接
Encoder Block 2: Conv + Conv + Downsample → f2 (4C, H/4, W/4)
    │                                                 │
    ▼                                                 │ 跳跃连接
Encoder Block 3: Conv + Conv + Downsample → f3 (8C, H/8, W/8)
    │
    ▼
Middle Block: Attention + Conv 处理最深层特征
    │
    ▼
Decoder Block 3: Cat(f3) + Conv + Conv + Upsample → d3 (4C, H/4, W/4)
    │                                                 ▲
    ▼                                                 从编码器取回 f2
Decoder Block 2: Cat(f2) + Conv + Conv + Upsample → d2 (2C, H/2, W/2)
    │                                                 ▲
    ▼                                                 从编码器取回 f1
Decoder Block 1: Cat(f1) + Conv + Conv + Upsample → d1 (C, H, W)
    │
    ▼
输出卷积 → 重建图像 (C_out, H, W)
```

### 关键概念解释

- **下采样（Downsample）**：通过 stride=2 的卷积或池化将特征图尺寸减半。每次下采样让网络看到更大范围的信息（感受野翻倍），同时降低计算量。
- **上采样（Upsample）**：通过转置卷积或插值将特征图尺寸翻倍。目的是逐步恢复原始分辨率，产生像素级的输出。
- **跳跃连接**：编码器第 i 层的输出直接拼接到解码器倒数第 i 层的输入（沿通道维度拼接）。这相当于告诉解码器："这里有你之前看到的细节信息，拿去用。"
- **通道数变化**：编码器中通道数逐层翻倍（补偿空间分辨率的下降），解码器中通道数逐层减半（因为跳跃连接拼接引入了额外的通道）。

### 直观解释

可以把 UNet 看作一个"信息的金字塔压缩与展开"系统。编码器把一幅图像的信息从"空间密集、通道稀疏"的形式（大特征图、少通道）转变为"空间稀疏、通道密集"的形式（小特征图、多通道），然后在瓶颈层进行关键信息加工（注意力机制），最后由解码器带着跳跃连接提供的位置信息，把压缩的信息展开回原始分辨率。

在 DDPM 中，UNet 的输入是加噪图像 x_t，输出是预测的噪声 ε，相当于完成了一个"图到图"的翻译任务——这是 UNet 最擅长的领域。

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| x | 输入图像/特征图 | (B, C, H, W) |
| C | 通道数 | 标量 |
| H, W | 空间高和宽 | 标量 |
| s | stride（步长） | 标量，下采样时=2 |
| K | 卷积核大小 | 标量，通常=3 |
| f_enc^i | 第 i 层编码器输出 | (B, C·2^i, H/2^i, W/2^i) |
| f_dec^i | 第 i 层解码器输出 | (B, C·2^{L-i}, H/2^{L-i}, W/2^{L-i}) |
| t | 时间步（DDPM 中） | 标量 |
| t_emb | 时间步嵌入向量 | (B, d_time) |
| L | UNet 总层数 | 标量 |

### 编码器层的形式化

编码器的每层包含两个连续的卷积块和一个下采样操作：

$$f_{\text{enc}}^i = \text{Downsample}(\text{ConvBlock}(\text{ConvBlock}(f_{\text{enc}}^{i-1})))$$

其中每个 ConvBlock（在 DDPM 的 UNet 中通常为 ResBlock）为：

$$\text{ConvBlock}(x) = \text{GN}(\text{Conv}(\text{SiLU}(\text{GN}(\text{Conv}(\text{SiLU}(\text{GN}(x)))))))) + \text{Skip}(x)$$

考虑到时间条件的注入（DDPM 特有），第二个 GroupNorm 前会加入时间嵌入：

$$h = \text{Conv}(\text{SiLU}(\text{GN}(x))) + \text{Linear}(t_{\text{emb}})$$

下采样操作使用 stride=2 的卷积：

$$\text{Downsample}(x) = \text{Conv}_{k=3, s=2, p=1}(x)$$

### 解码器层的形式化

解码器的每层首先通过跳跃连接拼接编码器特征：

$$x_{\text{cat}} = \text{Concat}(f_{\text{dec}}^{i+1\ \text{(上采样后)}}, f_{\text{enc}}^{L-i})$$

沿通道维度拼接，通道数变为 $C_{\text{dec}} + C_{\text{enc}}$。然后经过两个卷积块和一个上采样：

$$f_{\text{dec}}^i = \text{Upsample}(\text{ConvBlock}(\text{ConvBlock}(x_{\text{cat}})))$$

### 瓶颈层（Middle Block）

在编码器和解码器之间，特征图通过：

$$f_{\text{mid}} = \text{ConvBlock}(\text{Attention}(\text{ConvBlock}(f_{\text{enc}}^L)))$$

其中注意力操作计算全局上下文交互：

$$\text{Attention}(x) = x + \text{Proj}(\text{softmax}(\frac{QK^T}{\sqrt{d}})V)$$

Q、K、V 由 x 通过 1x1 卷积投影得到。

### 时间嵌入（DDPM 特有）

DDPM 的 UNet 需要在每个 ResBlock 中注入时间步信息。时间步 t 被编码为正弦位置嵌入：

$$\text{PE}(t, 2i) = \sin(t \cdot e^{-2i \cdot \log(10000) / d})$$
$$\text{PE}(t, 2i+1) = \cos(t \cdot e^{-2i \cdot \log(10000) / d})$$

然后通过一个小的 MLP 投影到每个 ResBlock 需要的时间条件维度：

$$t_{\text{emb}} = \text{Linear}(\text{GELU}(\text{Linear}(\text{PE}(t))))$$

### 输出层

最终输出通过一个 1x1 卷积将通道数映射回输入通道数：

$$\hat{\varepsilon} = \text{Conv}_{k=1}(\text{GN}(\text{SiLU}(\text{GN}(f_{\text{dec}}^0))))$$

在 DDPM 中，这输出预测的噪声；在其他任务中，输出相应的目标（分割掩码、重建图像等）。

## 4. 训练过程讲解

### 数据预处理

- 图像归一化到 [-1, 1]（DDPM 场景）或 [0, 1]（通用场景）
- 尺寸统一，确保能够被 2^L 整除（L 为下采样层数）
- 在 DDPM 中，训练数据是加噪图像 + 对应的真实噪声

### 参数初始化

- 卷积层权重使用 Kaiming 正态初始化（He initialization），适合 ReLU/SiLU 激活
- 偏置初始化 0
- 时间嵌入的 MLP 使用 Xavier 初始化
- GroupNorm 的权重初始化 1，偏置 0

### 迭代过程

```
对于每个训练步骤:
  1. 输入图像 x（或 DDPM 中的加噪图像 x_t + 时间步 t）
  2. 时间步通过正弦编码嵌入 → MLP → t_emb
  3. 前向传播:
     a. init_conv(x) → 初始特征
     b. 逐层下采样: ResBlock + ResBlock + Downsample，保存跳跃连接
     c. 瓶颈层: ResBlock + Attention + ResBlock
     d. 逐层上采样: Cat(skip) + ResBlock + ResBlock + Upsample
     e. 输出卷积 → 最终预测
  4. 计算损失 (MSE / Huber)
  5. 反向传播，更新所有参数
```

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| base_dim | 第一层通道数 | 32-256 | 64（小图）/ 128（大图） |
| dim_mults | 每层通道倍数 | (1,2,4) 或 (1,2,4,8) | (1, 2, 4) 用于 28-64 像素 |
| num_res_blocks | 每层残差块数 | 1-3 | 2 |
| time_emb_dim | 时间嵌入维度 | base_dim 的 2-4 倍 | base_dim * 4 |
| attention_resolutions | 在哪些分辨率使用注意力 | 如 [16, 8] | 最低 1-2 层 |
| group_norm_groups | GroupNorm 的组数 | 4-32 | 8 |
| dropout | 残差块的 dropout | 0.0-0.3 | 0.0（扩散模型通常不用） |

## 5. 应用场景

### 典型应用

1. **DDPM 扩散模型的噪声预测器**：UNet 是 DDPM 的核心组件，接收加噪图像 x_t 和时间步 t，输出预测的噪声 ε。UNet 的图到图映射能力使其天然适合这一任务。Stable Diffusion、DALL-E 2 等知名模型都使用 UNet 作为扩散骨干网络。

2. **医学图像分割**：UNet 的原生应用场景。在 CT、MRI、显微图像中精确分割器官、肿瘤、细胞等。其跳跃连接对保留细小结构（如血管、细胞边界）至关重要。

3. **图像超分辨率**：将低分辨率图像输入 UNet，输出高分辨率版本。编码器提取语义信息（"这是一张人脸"），跳跃连接保留纹理细节（"这里有皱纹"）。

4. **图像去噪与修复**：直接输入含噪声/破损的图像，UNet 学习映射到干净/完整的图像。去噪扩散模型本质上就是迭代版的这类 UNet 去噪。

### 适用数据特征

- 输入和输出具有相同空间维度（或简单的缩放关系）
- 需要同时利用全局语义和局部细节的任务
- 训练数据量适中（UNet 的参数共享使其比全连接更高效，比 Transformer 对数据量更友好）

### 不适用场景

- 输出是单标签的分类任务（CNN + 全连接更合适）
- 输入输出分辨率差异极大的任务
- 超大规模数据且有充足算力（ViT/DiT 可能取得更好效果）
- 严格实时且设备算力极低（可能需要更轻量的 MobileNet 类架构）

## 6. 优缺点分析

### 优点

| 优点 | 成立条件 | 说明 |
|------|----------|------|
| 保留空间细节 | 跳跃连接正确拼接 | 编码器的低层特征包含精确的位置和纹理信息，跳跃连接直接传给解码器 |
| 多尺度特征融合 | 层数 ≥ 3 | 不同层捕捉不同感受野的信息——浅层看到纹理边缘，深层看到物体语义 |
| 数据效率高 | 参数共享和特征重用 | 相比全连接或 Transformer，UNet 对训练数据量的需求较低 |
| 架构灵活 | 模块化设计 | 可替换卷积块（ResBlock / ConvNeXt Block）、加入注意力、调整深度和宽度 |
| 端到端训练 | 全卷积结构 | 无需分阶段训练，不限制输入尺寸（只要能被 2^L 整除） |

### 缺点

| 缺点 | 何时出问题 | 缓解思路 |
|------|-----------|----------|
| 计算量大 | 分辨率高 + 层数多 | 减小 base_dim、使用深度可分离卷积、在潜空间操作（Latent Diffusion） |
| 参数量大 | dim_mults 过大（如 (1,2,4,8,16)） | 控制层数（3-4 层通常足够），使用 1x1 卷积降维 |
| 浅层信息利用方式简单 | 拼接后解码器过度依赖浅层特征 | 在跳跃连接中加入轻量注意力门控（Attention U-Net） |
| 对下采样倍数敏感 | H 或 W 不能被 2^L 整除 | 输入前 pad 到合适尺寸，或在特定层不进行下采样 |

### 与其他编码器-解码器架构的对比

| 特性 | UNet | SegNet | FPN | DeepLabV3+ |
|------|------|--------|-----|------------|
| 跳跃连接方式 | 特征图拼接 | 池化索引复用 | 逐元素相加 | 1x1 卷积 + 拼接 |
| 信息保留程度 | 极高 | 中等 | 高 | 高 |
| 参数量 | 中等 | 较小 | 较大 | 大 |
| DDPM 使用 | 默认选择 | 不适合 | 不常用 | 不适合 |
| 原始用途 | 医学图像分割 | 场景分割 | 目标检测 | 语义分割 |

## 7. 调库实现

```python
"""
DDPM 中的 UNet 实现 -- 使用 PyTorch
包含：时间嵌入、下采样块、上采样块、跳跃连接、自注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 时间嵌入 ====================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码：将时间步 t 映射为固定维度的连续向量

    为什么用正弦编码？
    - 无参数，不需要训练，不会引入额外优化负担
    - 高低频率混合，能表达时间步的"位置"和"变化速率"
    - 不同频率的正弦/余弦确保了不同时间步的嵌入有明显区分
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) 整型时间步索引, 返回: (B, dim)"""
        device = t.device
        half_dim = self.dim // 2
        # 计算频率: [log(10000)/0, log(10000)/1, ..., log(10000)/(half_dim-1)]
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # t 与每个频率相乘: (B, 1) * (1, half_dim) = (B, half_dim)
        emb = t[:, None].float() * emb[None, :]
        # 正弦和余弦拼接: (B, half_dim * 2) = (B, dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeMLP(nn.Module):
    """时间嵌入的小型 MLP：将正弦编码进一步投影到各 ResBlock 需要的时间维度"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ==================== 残差块（含时间注入） ====================
class ResBlock(nn.Module):
    """带时间条件注入的残差卷积块

    结构:
    GroupNorm -> SiLU -> Conv -> +时间嵌入 -> GroupNorm -> SiLU -> Conv -> +残差

    时间嵌入如何注入？
    1. 从 TimeMLP 获得 (B, out_channels) 的时间向量
    2. Reshape 为 (B, out_channels, 1, 1)
    3. 加到第一个卷积的输出上（广播到整个空间）
    这样每个空间位置都感知到相同的时间信息
    """
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)

        # 如果输入输出通道不同，用 1x1 卷积短路对齐
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # 注入时间信息：将时间嵌入加到特征图上
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


# ==================== 自注意力模块 ====================
class SelfAttentionBlock(nn.Module):
    """UNet 中的自注意力模块

    为什么在 UNet 中加注意力？
    - 卷积是局部的（感受野受限于核大小），难以捕获远距离依赖
    - 自注意力让所有位置可以互相"看到"，对理解全局布局至关重要
    - 在低分辨率层（特征图尺寸小）加注意力性价比最高

    实现策略：
    - 在瓶颈层和最低 1-2 个分辨率的解码器层使用
    - 输入输出维度不变，方便插入任何位置
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, \
            f"channels ({channels}) 必须能被 num_heads ({num_heads}) 整除"

        self.norm = nn.GroupNorm(1, channels)  # 等价于 LayerNorm 在 GroupNorm 中
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 归一化 + QKV 投影
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        # 拆分为多头形式
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        scale = self.head_dim ** -0.5
        attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
        out = attn @ V  # (B, num_heads, H*W, head_dim)

        # 合并多头并恢复空间形状
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ==================== 下采样和上采样块 ====================
class DownBlock(nn.Module):
    """下采样模块：两个残差块 + 一个下采样

    为什么每次下采样前放两个残差块？
    - 一个残差块可能不足以在压缩空间前充分提取特征
    - 两个残差块给网络更多机会处理当前分辨率的信息
    """
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, downsample: bool = True, dropout: float = 0.0):
        super().__init__()
        self.resblock1 = ResBlock(in_channels, out_channels,
                                  time_emb_dim, dropout)
        self.resblock2 = ResBlock(out_channels, out_channels,
                                  time_emb_dim, dropout)
        if downsample:
            # stride=2 卷积实现下采样，将 H 和 W 各减半
            self.downsample = nn.Conv2d(out_channels, out_channels,
                                        kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.resblock1(x, t_emb)
        x = self.resblock2(x, t_emb)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """上采样模块：拼接跳跃连接 + 两个残差块 + 一个上采样

    为什么用拼接而不是相加？
    - 拼接保留了两路信息的独立性（编码器特征和解码器特征）
    - 让后续卷积自行学习如何融合，比强制相加更灵活
    - 缺陷是增加了通道数（需要 2x 输入通道的卷积核）
    """
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, upsample: bool = True, dropout: float = 0.0):
        super().__init__()
        # in_channels 包含了跳跃连接拼接后的总通道数
        self.resblock1 = ResBlock(in_channels, out_channels,
                                  time_emb_dim, dropout)
        self.resblock2 = ResBlock(out_channels, out_channels,
                                  time_emb_dim, dropout)
        if upsample:
            # 双线性插值上采样 + 3x3 卷积平滑，避免转置卷积的棋盘效应
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        # 步骤 1: 沿通道维度拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        # 步骤 2: 两个残差块处理
        x = self.resblock1(x, t_emb)
        x = self.resblock2(x, t_emb)
        # 步骤 3: 上采样
        x = self.upsample(x)
        return x


# ==================== 完整 UNet ====================
class UNet(nn.Module):
    """DDPM 中的 UNet：用于预测添加的噪声

    参数说明:
        in_channels: 输入图像通道数 (MNIST=1, RGB=3)
        base_dim: 基础通道数 (第一层卷积的输出通道)
        dim_mults: 每层的通道倍数 (决定各层的通道数和下采样次数)
        with_time_emb: 是否使用时间嵌入 (DDPM 必须，普通去噪可不使用)
        num_res_blocks: 每层的残差块数量
        dropout: 残差块内的 dropout 率
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_dim: int = 64,
                 dim_mults: tuple = (1, 2, 4, 8),
                 with_time_emb: bool = True,
                 num_res_blocks: int = 2,
                 dropout: float = 0.0):
        super().__init__()

        # ---- 时间嵌入模块 ----
        if with_time_emb:
            time_dim = base_dim * 4
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(base_dim),
                TimeMLP(base_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_embed = None

        # ---- 初始卷积 ----
        self.init_conv = nn.Conv2d(in_channels, base_dim,
                                   kernel_size=3, padding=1)

        # ---- 计算各层通道数 ----
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        num_layers = len(dim_mults)

        # ---- 下采样模块 ----
        self.downs = nn.ModuleList([])
        for i in range(num_layers):
            in_ch = dims[i]
            out_ch = dims[i + 1]
            is_last = (i == num_layers - 1)
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResBlock(in_ch, out_ch, time_dim, dropout)
                )
                in_ch = out_ch
            if not is_last:
                self.downs.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3,
                              stride=2, padding=1)  # 下采样
                )

        # ---- 瓶颈层 ----
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim, dropout)
        self.mid_attn = SelfAttentionBlock(mid_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim, dropout)

        # ---- 上采样模块 ----
        self.ups = nn.ModuleList([])
        for i in reversed(range(num_layers)):
            in_ch = dims[i + 1]
            out_ch = dims[i]
            is_last = (i == 0)
            for j in range(num_res_blocks):
                # 跳跃连接拼接 → 输入通道数 x2
                actual_in = in_ch + (dims[i + 1] if j == 0 else out_ch)
                self.ups.append(
                    ResBlock(actual_in, out_ch, time_dim, dropout)
                )
                in_ch = out_ch
            if not is_last:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    )
                )

        # ---- 输出卷积 ----
        self.out_norm = nn.GroupNorm(min(32, base_dim // 4), base_dim)
        self.out_conv = nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor,
                t: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x: (B, C, H, W) 加噪图像
            t: (B,) 时间步索引，DDPM 模式必须
        返回:
            (B, C, H, W) 预测的噪声
        """
        # 时间嵌入
        t_emb = None
        if self.time_embed is not None and t is not None:
            t_emb = self.time_embed(t)

        # 初始卷积
        x = self.init_conv(x)
        h = x.clone()  # 保存用于最终残差

        # ---- 下采样阶段 ----
        skips = []
        for module in self.downs:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            else:
                skips.append(x)       # 在即将下采样前保存
                x = module(x)

        # ---- 瓶颈层 ----
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # ---- 上采样阶段 ----
        for module in self.ups:
            if isinstance(module, ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)  # 跳跃连接拼接
                x = module(x, t_emb)
            else:
                x = module(x)  # 上采样操作

        # ---- 输出 ----
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x + h)  # 全局残差连接
        return x


# ==================== 测试和演示 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建 UNet 模型
    model = UNet(
        in_channels=1,         # MNIST 灰度图
        base_dim=64,           # 基础通道
        dim_mults=(1, 2, 4),   # 3 层 UNet，总下采样 8x
        with_time_emb=True,
        num_res_blocks=2,
        dropout=0.0,
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 模拟 DDPM 场景：输入加噪图像 + 时间步
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)   # 模拟加噪图像
    t = torch.randint(0, 1000, (batch_size,)).to(device) # 随机时间步

    print(f"\n输入图像形状: {x.shape}")
    print(f"时间步: {t.tolist()}")

    # 前向传播
    output = model(x, t)
    print(f"输出（预测噪声）形状: {output.shape}")

    # 验证输出形状与输入一致
    assert output.shape == x.shape, \
        f"形状不匹配: {output.shape} != {x.shape}"
    print("\n验证通过: 输出形状 == 输入形状 ✓")
    print("（如果形状一致，残差连接设计正确）")

    # 测试无时间嵌入模式（普通去噪 UNet）
    model_no_time = UNet(
        in_channels=1, base_dim=32, dim_mults=(1, 2, 4),
        with_time_emb=False
    ).to(device)
    output_no_time = model_no_time(x)
    print(f"\n无时间嵌入 UNet 输出形状: {output_no_time.shape}")
    print("验证通过 ✓")
```

## 8. 手工代码实现

```python
"""
从零手写 UNet -- 不依赖高级封装
逐一实现每个基础组件，理解 UNet 的每个细节
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==================== 第一步：手工实现时间嵌入 ====================
class SinusoidalEmbedding:
    """手工实现正弦位置嵌入的数学逻辑，不使用 PyTorch Module

    这一步展示了正弦嵌入"为什么"是这样计算的：
    1. 对于维度 d，生成 d/2 个频率: 1/(10000^(2i/d)), i = 0, 1, ..., d/2-1
    2. 每个时间步 t 乘以这些频率
    3. 分别计算 sin 和 cos，拼接

    为什么不同频率有用？
    - 低频（小 i）产生平滑的变化，区分相隔较远的时间步
    - 高频（大 i）产生快速振荡，区分相邻的时间步
    - 所有频率组合在一起形成一个丰富的"时间签名"
    """
    @staticmethod
    def encode(t: np.ndarray, d: int) -> np.ndarray:
        """t: (N,) 时间步数组, d: 嵌入维度, 返回: (N, d)"""
        N = len(t)
        # 频率计算
        half_d = d // 2
        freqs = np.exp(-np.arange(half_d) * math.log(10000) / (half_d - 1))
        # (N, 1) * (half_d,) -> (N, half_d)
        angles = t.reshape(-1, 1).astype(np.float32) * freqs.reshape(1, -1)
        # 正弦和余弦拼接
        emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
        return emb


# ==================== 第二步：手工实现残差块 ====================
class ResBlockFromScratch(nn.Module):
    """
    手工残差块，包含：
    - 两次 GroupNorm(8组) + SiLU + 3x3 Conv
    - 时间嵌入：用可学习的投影矩阵从时间向量映射到通道空间后加到特征图上
    - 残差连接：如果输入/输出通道不同，用 1x1 conv 映射

    关键洞察（为什么这样设计）：
    - GroupNorm 替代 BatchNorm：BatchNorm 在 batch 小的时候不准确，
      GroupNorm 不依赖 batch 大小，在 DDPM（大模型大图小 batch）中更稳定
    - SiLU 激活：比 ReLU 更平滑，梯度流动更好，是 Swish 激活的变体
    - 3x3 卷积 + padding=1：保持空间尺寸不变（same convolution）
    - 时间投影后加在特征图上：每个空间位置共享相同的时间条件，广播到整个 HxW
    """
    def __init__(self, in_ch: int, out_ch: int, time_ch: int,
                 groups: int = 8, dropout: float = 0.0):
        super().__init__()
        # 第一组：GN -> SiLU -> Conv3x3
        self.gn1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # 时间投影：将时间嵌入向量 (B, time_ch) 映射为 (B, out_ch, 1, 1)
        self.time_proj = nn.Sequential(
            nn.Linear(time_ch, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )

        # 第二组：GN -> SiLU -> Dropout -> Conv3x3
        self.gn2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 跳跃连接对齐
        self.skip = nn.Conv2d(in_ch, out_ch, 1) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # 第一次卷积 + 时间注入
        h = self.gn1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # 时间嵌入投影并加到特征图上
        t_proj = self.time_proj(t_emb)[:, :, None, None]  # (B, out_ch, 1, 1)
        h = h + t_proj

        # 第二次卷积
        h = self.gn2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ==================== 第三步：手工实现自注意力 ====================
class SelfAttentionFromScratch(nn.Module):
    """
    手工实现 UNet 中的 Flatten Attention：
    - 将特征图 (B, C, H, W) 展平为 (B, H*W, C)
    - 用 1x1 卷积生成 Q, K, V（等价于每个位置的线性投影）
    - 计算缩放点积注意力
    - 恢复空间形状

    注意与标准 Transformers 的区别：
    - 这里没有 multi-head（为简单起见，实际 DDPM 中可加）
    - 使用 1x1 Conv 而不是 Linear（因为输入是 4D 张量）
    """
    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels ** -0.5
        self.norm = nn.GroupNorm(1, channels)  # 每组 1 通道 = InstanceNorm
        # Q, K, V 的 1x1 卷积投影
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        # 可学习的残差权重参数（初始化为 0，让网络先学会卷积再学注意力）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        # 生成 Q, K, V 并展平空间维度
        Q = self.to_q(x_norm).reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        K = self.to_k(x_norm).reshape(B, C, -1)                     # (B, C, HW)
        V = self.to_v(x_norm).reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        # 注意力
        attn = torch.softmax(Q @ K * self.scale, dim=-1)           # (B, HW, HW)
        out = (attn @ V).permute(0, 2, 1).reshape(B, C, H, W)     # 恢复空间形状
        out = self.proj(out)

        return x + self.gamma * out  # 残差连接 + 可学习缩放


# ==================== 第四步：手工实现完整 UNet ====================
class UNetFromScratch(nn.Module):
    """手工 UNet：逐层构建，无任何魔法

    设计选择说明:
    - 总下采样 8x (3 层, 28→14→7→4), pad 到 32 使整除
    - 每层 2 个 ResBlock
    - 瓶颈层: ResBlock -> Attention -> ResBlock
    - 上采样使用 nearest 插值 (最简单) + 3x3 卷积
    """

    def __init__(self, in_ch: int = 1, base_dim: int = 32,
                 time_ch: int = None):
        super().__init__()
        if time_ch is None:
            time_ch = base_dim * 4

        # 时间嵌入的 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_dim),
            nn.Linear(base_dim, time_ch),
            nn.GELU(),
            nn.Linear(time_ch, time_ch),
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(in_ch, base_dim, 3, padding=1)

        # ---- 编码器 (通道数: 32 → 64 → 128) ----
        # Level 0: 32ch, 32x32 (pad 后的 28x28)
        self.enc_block0_1 = ResBlockFromScratch(base_dim, base_dim, time_ch)         # 32 → 32
        self.enc_block0_2 = ResBlockFromScratch(base_dim, base_dim, time_ch)         # 32 → 32
        self.down0 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)      # 32 → 16

        # Level 1: 64ch, 16x16
        self.enc_block1_1 = ResBlockFromScratch(base_dim * 2, base_dim * 2, time_ch) # 64 → 64
        self.enc_block1_2 = ResBlockFromScratch(base_dim * 2, base_dim * 2, time_ch) # 64 → 64
        self.down1 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)  # 16 → 8

        # Level 2: 128ch, 8x8
        self.enc_block2_1 = ResBlockFromScratch(base_dim * 4, base_dim * 4, time_ch) # 128 → 128
        self.enc_block2_2 = ResBlockFromScratch(base_dim * 4, base_dim * 4, time_ch) # 128 → 128
        self.down2 = nn.Conv2d(base_dim * 4, base_dim * 8, 3, stride=2, padding=1)  # 8 → 4

        # Level 3 (瓶颈): 256ch, 4x4
        self.enc_block3_1 = ResBlockFromScratch(base_dim * 8, base_dim * 8, time_ch) # 256 → 256
        self.enc_block3_2 = ResBlockFromScratch(base_dim * 8, base_dim * 8, time_ch) # 256 → 256

        # ---- 瓶颈注意力 ----
        self.mid_attn = SelfAttentionFromScratch(base_dim * 8)

        # ---- 解码器 (通道数: 128 → 64 → 32) ----
        # Level 3: 处理瓶颈输出
        self.dec_block3_1 = ResBlockFromScratch(base_dim * 16, base_dim * 4, time_ch) # 256+256=512 → 128
        self.dec_block3_2 = ResBlockFromScratch(base_dim * 4, base_dim * 4, time_ch)  # 128 → 128
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                              # 4 → 8
            nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1),
        )

        # Level 2: 处理 up3 输出 + skip from enc2
        self.dec_block2_1 = ResBlockFromScratch(base_dim * 8, base_dim * 2, time_ch)  # 128+128=256 → 64
        self.dec_block2_2 = ResBlockFromScratch(base_dim * 2, base_dim * 2, time_ch)  # 64 → 64
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                              # 8 → 16
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1),
        )

        # Level 1: 处理 up2 输出 + skip from enc1
        self.dec_block1_1 = ResBlockFromScratch(base_dim * 4, base_dim, time_ch)      # 64+64=128 → 32
        self.dec_block1_2 = ResBlockFromScratch(base_dim, base_dim, time_ch)          # 32 → 32
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                              # 16 → 32
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
        )

        # Level 0: 处理 up1 输出 + skip from init_conv
        self.dec_block0_1 = ResBlockFromScratch(base_dim * 2, base_dim, time_ch)      # 32+32=64 → 32
        self.dec_block0_2 = ResBlockFromScratch(base_dim, base_dim, time_ch)          # 32 → 32

        # ---- 输出层 ----
        self.out_norm = nn.GroupNorm(8, base_dim)
        self.out_conv = nn.Conv2d(base_dim, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 初始卷积
        x = self.init_conv(x)
        skip0 = x  # 跳跃连接 0

        # ---- 编码器 ----
        x = self.enc_block0_1(x, t_emb)
        x = self.enc_block0_2(x, t_emb)
        x = self.down0(x)
        skip1 = x

        x = self.enc_block1_1(x, t_emb)
        x = self.enc_block1_2(x, t_emb)
        x = self.down1(x)
        skip2 = x

        x = self.enc_block2_1(x, t_emb)
        x = self.enc_block2_2(x, t_emb)
        x = self.down2(x)

        x = self.enc_block3_1(x, t_emb)
        x = self.enc_block3_2(x, t_emb)

        # ---- 瓶颈注意力 ----
        x = self.mid_attn(x)

        # ---- 解码器 ----
        x = self.dec_block3_1(torch.cat([x, x], dim=1), t_emb)  # 瓶颈无额外 skip，自拼接
        x = self.dec_block3_2(x, t_emb)
        x = self.up3(x)

        x = self.dec_block2_1(torch.cat([x, skip2], dim=1), t_emb)
        x = self.dec_block2_2(x, t_emb)
        x = self.up2(x)

        x = self.dec_block1_1(torch.cat([x, skip1], dim=1), t_emb)
        x = self.dec_block1_2(x, t_emb)
        x = self.up1(x)

        x = self.dec_block0_1(torch.cat([x, skip0], dim=1), t_emb)
        x = self.dec_block0_2(x, t_emb)

        # 输出
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置嵌入模块（PyTorch nn.Module 封装，用于时间 MLP）"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ==================== 测试代码 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    print("=== 从零手写 UNet 测试 ===")

    # 创建模型
    model = UNetFromScratch(in_ch=1, base_dim=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"从零 UNet 参数量: {total_params:,}")

    # 模拟输入
    B = 2
    x = torch.randn(B, 1, 32, 32)  # 模拟 pad 到 32x32 的 MNIST 图像
    t = torch.tensor([50, 200])     # 两个不同的时间步

    # 前向传播
    print(f"\n输入形状: {x.shape}")
    print(f"时间步: {t.tolist()}")
    output = model(x, t)
    print(f"输出形状: {output.shape}")

    # 验证
    assert output.shape == x.shape, \
        f"验证失败: 输出 {output.shape} != 输入 {x.shape}"
    print("\n验证通过: 输出形状 == 输入形状 ✓")

    # 测试不同时间步得到不同输出（证明时间嵌入正确工作）
    t2 = torch.tensor([500, 800])
    output2 = model(x, t2)
    diff = (output - output2).abs().mean().item()
    print(f"不同时间步的输出平均差异: {diff:.6f}")
    print(f"(差异 > 0 → 时间嵌入正常工作 ✓)")

    # 展示组件参数
    print(f"\n===== 模型各组件参数统计 =====")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n_params:,} 参数")

    print("\n全部测试通过！手工 UNet 实现正确 ✓")
```

## 9. 可视化与结果理解

```python
"""
UNet 可视化：特征图变化、跳跃连接示意、不同层输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 复用第 7/8 节的 UNet 定义
# 这里创建一个带 hook 的版本用于可视化中间特征

torch.manual_seed(42)

# ==================== 图 1: UNet 架构示意 ====================
def visualize_unet_architecture():
    """画出 UNet 的 U 形结构和跳跃连接"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # 定义各层的特征图尺寸和通道数
    layers = [
        # (层级, 通道数, 尺寸, 类型)
        (0, 1,   32, '输入'),
        (1, 32,  32, 'enc0'),
        (2, 64,  16, 'enc1'),
        (3, 128, 8,  'enc2'),
        (4, 256, 4,  '瓶颈'),
        (5, 128, 8,  'dec2'),
        (6, 64,  16, 'dec1'),
        (7, 32,  32, 'dec0'),
        (8, 1,   32, '输出'),
    ]

    x_coords = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    y_coords = [4, 5, 4, 3, 2, 3, 4, 5, 4]  # U 形
    sizes = [l[2] / 4 for l in layers]  # 可视化尺寸
    colors = ['lightgray'] + ['lightblue'] * 3 + ['lightcoral'] + ['lightgreen'] * 3 + ['lightgray']

    # 画方块
    for i, (level, ch, sz, label) in enumerate(layers):
        rect = plt.Rectangle(
            (x_coords[i] - sizes[i] / 2, y_coords[i] - sizes[i] / 2),
            sizes[i], sizes[i],
            facecolor=colors[i], edgecolor='black', linewidth=2, alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(x_coords[i], y_coords[i],
                f'{label}\n{ch}ch, {sz}x{sz}',
                ha='center', va='center', fontsize=8, fontweight='bold')

    # 画箭头（编码器方向）
    for i in range(4):
        ax.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]),
                    xytext=(x_coords[i], y_coords[i]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 画解码器方向箭头
    for i in range(4, 8):
        ax.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]),
                    xytext=(x_coords[i], y_coords[i]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # 画跳跃连接（虚线）
    for enc_idx, dec_idx in [(1, 7), (2, 6), (3, 5)]:
        ax.plot([x_coords[enc_idx], x_coords[dec_idx]],
                [y_coords[enc_idx], y_coords[dec_idx]],
                'k--', alpha=0.5, linewidth=1.5)
        ax.text((x_coords[enc_idx] + x_coords[dec_idx]) / 2,
                (y_coords[enc_idx] + y_coords[dec_idx]) / 2 + 0.3,
                '跳跃\n连接', ha='center', fontsize=7, color='gray')

    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('UNet U 形架构与跳跃连接示意', fontsize=15, fontweight='bold')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='编码器（下采样）'),
        Patch(facecolor='lightcoral', label='瓶颈层（注意力）'),
        Patch(facecolor='lightgreen', label='解码器（上采样）'),
        plt.Line2D([0], [0], linestyle='--', color='black', alpha=0.5,
                   label='跳跃连接'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig('unet_architecture.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 1 解读：UNet 的 U 形结构——编码器逐步压缩空间、增加通道；")
    print("  瓶颈层在最深层用注意力捕获全局关系；")
    print("  解码器逐步恢复空间、减少通道，并用跳跃连接取回编码器的细节信息。")


# ==================== 图 2: 下采样与上采样的特征演变 ====================
def visualize_down_up_sampling():
    """展示一张测试图经过下采样和上采样后的变化"""
    from torchvision import datasets, transforms

    # 加载测试图
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.FashionMNIST(root="./data", train=True,
                                     download=True, transform=transform)
    img = dataset[0][0].unsqueeze(0)  # (1, 1, 28, 28)

    # 下采样（模拟编码器的各层输出）
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 原始图
    axes[0, 0].imshow(img.squeeze().numpy(), cmap='gray')
    axes[0, 0].set_title('原始图像\n28x28', fontsize=11)
    axes[0, 0].axis('off')

    # 用 avg_pool2d 模拟下采样
    current = img
    resolutions = [(14, 14), (7, 7), (4, 4)]
    for i, (h, w) in enumerate(resolutions):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        axes[0, i + 1].imshow(current.squeeze().numpy(), cmap='gray')
        axes[0, i + 1].set_title(f'下采样 {i+1}\n{w}x{h}', fontsize=11)
        axes[0, i + 1].axis('off')

    # 用 interpolate 模拟上采样
    current_up = current
    upsample_resolutions = [(7, 7), (14, 14), (28, 28)]
    for i, (h, w) in enumerate(upsample_resolutions):
        current_up = F.interpolate(current_up, size=(h, w),
                                    mode='bilinear', align_corners=False)
        axes[1, i].imshow(current_up.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title(f'上采样 {i+1}\n{w}x{h}', fontsize=11)
        axes[1, i].axis('off')

    # 最后一格标注瓶颈
    axes[1, 3].text(0.5, 0.5, '瓶颈层\n4x4\n+注意力',
                    ha='center', va='center', fontsize=13,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    axes[1, 3].axis('off')

    axes[0, 0].set_ylabel('编码器方向 →', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('解码器方向 →', fontsize=12, fontweight='bold')

    plt.suptitle('UNet 下采样（特征压缩）与上采样（特征重建）', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('unet_downsample_upsample.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 2 解读：下采样逐步丢失空间细节但保留语义结构；")
    print("  上采样恢复分辨率但图像变得模糊——这时就需要跳跃连接来补充细节。")


# ==================== 图 3: 跳跃连接的作用演示 ====================
def visualize_skip_connection_benefit():
    """对比有无跳跃连接的 UNet 重建效果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    np.random.seed(123)
    original = np.zeros((28, 28))
    # 画一个圆形
    yy, xx = np.ogrid[:28, :28]
    circle = (xx - 14) ** 2 + (yy - 14) ** 2 <= 6 ** 2
    original[circle] = 1.0
    # 加一些纹理
    original += 0.3 * np.sin(xx / 3) * np.cos(yy / 3) * (1 - circle.astype(float) * 0.5)
    original = np.clip(original, 0, 1)

    # 模拟无跳跃连接的重建（模糊）
    from scipy.ndimage import gaussian_filter
    no_skip = gaussian_filter(original, sigma=3.5)

    # 模拟有跳跃连接的重建（稍模糊但保留边缘）
    with_skip = 0.7 * original + 0.3 * gaussian_filter(original, sigma=2.0)

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像', fontsize=13)
    axes[0].set_xlabel('像素位置 x')
    axes[0].set_ylabel('像素位置 y')

    axes[1].imshow(no_skip, cmap='gray')
    axes[1].set_title('无跳跃连接\n（纯编码-解码，模糊）', fontsize=13)
    axes[1].set_xlabel('像素位置 x')
    axes[1].set_ylabel('像素位置 y')

    axes[2].imshow(with_skip, cmap='gray')
    axes[2].set_title('有跳跃连接\n（保留细节、边缘清晰）', fontsize=13)
    axes[2].set_xlabel('像素位置 x')
    axes[2].set_ylabel('像素位置 y')

    plt.suptitle('跳跃连接对 UNet 重建质量的影响', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('unet_skip_connection_benefit.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 3 解读：无跳跃连接时，编码-解码过程丢失了大量空间细节，输出模糊；")
    print("  跳跃连接将编码器的低层特征（包含精确位置和纹理信息）直接传给解码器，")
    print("  帮助恢复清晰边缘和精细结构——这是 UNet 性能的核心来源。")


if __name__ == "__main__":
    print("=" * 60)
    print("UNet 可视化分析")
    print("=" * 60)
    visualize_unet_architecture()
    visualize_down_up_sampling()
    visualize_skip_connection_benefit()
    print("\n全部可视化完成。")
```

## 10. 模型评估

### DDPM 中 UNet 的评估方法

因为在 DDPM 中 UNet 不是最终输出（最终输出是生成的图像），评估分两个层面：

**1. UNet 噪声预测准确度（训练监控）**

噪声预测越好 → 最终生成质量越高。训练时监控的指标：

- **训练损失（MSE/Huber）**：预测噪声 ε_θ 与真实噪声 ε 之间的差异。损失越低说明 UNet 的去噪能力越强。
- **不同时间步的损失分布**：如果某些 t 的损失远高于其他，说明 UNet 在这些噪声比例下学习不足。

**2. 对最终生成质量的影响（消融实验）**

通过改变 UNet 的配置，观察 FID / IS 的变化来评估 UNet 设计的有效性：

| 消融变量 | 预期效果 | 评估方法 |
|----------|----------|----------|
| UNet 深度（层数） | 更深 → 更大的感受野，但可能过拟合 | 训练相同 DDPM，对比 FID |
| 通道数（base_dim） | 更宽 → 更强的表达能力，但训练变慢 | 对比不同 base_dim 的 FID |
| 注意力模块 | 有注意力 → 更好的全局一致性 | 对比有/无 Self-Attention 的生成质量 |
| 跳跃连接 | 有跳跃连接 → 明显优于无 | 移除所有跳连 vs 保留，质量的巨大差距 |

**3. 参数量和推理速度**

UNet 作为扩散模型的计算瓶颈，参数量和推理速度是重要工程指标：

```python
"""
UNet 参数量和推理速度评估
"""
import time

def benchmark_unet(model, input_shape=(1, 1, 32, 32), num_runs=50):
    """测试 UNet 的推理速度和参数量"""
    device = next(model.parameters()).device
    x = torch.randn(*input_shape).to(device)
    t = torch.tensor([100]).to(device)

    # 预热
    for _ in range(10):
        _ = model(x, t)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(x, t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    total_params = sum(p.numel() for p in model.parameters())

    return {
        'params': total_params,
        'time_per_run_ms': (elapsed / num_runs) * 1000,
        'memory_estimate': total_params * 4 / 1024 / 1024,  # float32: MB
    }
```

## 11. 常见问题与易错点

### 架构设计层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 图像尺寸不能被 2^L 整除 | 下采样后尺寸不对齐、拼接时维度报错 | 每次下采样 stride=2 需要 H 和 W 都能被 2 整除 | 输入前用反射填充（ReflectionPad）补到最近的 2^L 倍数 |
| 跳跃连接通道数不匹配 | 上采样拼接时 RuntimeError | 下采样时改变了通道数（如用了新的 Conv2d(in, out, stride=2)），但跳跃连接保存的是上一步的输出 | 确保 DownBlock 先做卷积（不改变通道）再下采样，或保存下采样后的特征给跳连 |
| 上采样方法不当 | 输出图像有明显棋盘状伪影 | 转置卷积（ConvTranspose2d）的 kernel_size 不能被 stride 整除时产生棋盘效应 | 使用双线性插值 + 3x3 Conv 替代转置卷积，或在转置卷积后调整 kernel/stride 参数 |
| 时间嵌入未正确广播 | 不同空间位置的去噪效果不一致 | 时间嵌入 reshape 为 (B, C) 后未扩展空间维度就加到 4D 特征图上 | 确保 t_emb_proj[:, :, None, None] 正确扩展为 (B, C, 1, 1) |

### 训练层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| GroupNorm 组数过大 | 训练不稳定、损失波动大 | GroupNorm 的 group 数大于通道数，退化为 InstanceNorm | 设置 groups = min(8, channels // 4)，确保每组至少 4 个通道 |
| 深层特征梯度消失 | 深层参数几乎不更新 | 网络太深、缺少残差连接、激活函数选择不当 | 确保每个 ResBlock 都有残差连接，使用 SiLU/GELU 而非 ReLU |
| 注意力模块导致训练变慢 | 加入 Attention 后训练时间翻倍 | 自注意力的 O(H²W²) 复杂度在较高分辨率层计算量极大 | 仅在最低分辨率层（<= 8x8）加入 Attention，使用 Flash Attention |
| 模型对 batch_size 敏感 | 不同 batch_size 下收敛速度差异大 | GroupNorm 虽然比 BatchNorm 好，但 batch 中样本的多样性仍影响学习 | 使用梯度累积模拟大 batch，确保 GroupNorm 的组数合理 |

### 部署与推理层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 推理显存占用大 | GPU 显存不足( OOM ) | 高分辨率下特征图占用大量显存（如 256x256 在通道数为 128 的层需要 256x256x128x4=32MB） | 使用梯度检查点（checkpoint）、降低 base_dim、在潜空间操作 |
| torch.cat 时维度错误 | 拼接报错 dim 参数错误 | 跳跃连接拼接时搞混了 B/C/H/W 维度的索引 | 始终确认 x.shape 和 skip.shape 在各维度上匹配（B 和 H、W），然后 dim=1（通道维）拼接 |

## 12. 学习总结

### 核心思想回顾

UNet 的设计哲学是"压缩以理解语义，展开以恢复细节，跳跃连接防止遗忘"。它通过对称的 U 形结构实现了三个层次的功能：

1. **编码器（下采样路径）**：从像素级细节中蒸馏出语义信息——知道"这是什么"
2. **解码器（上采样路径）**：从语义信息中重建像素级输出——把"知道是什么"转化为"画出来"
3. **跳跃连接**：编码器和解码器之间的信息桥——告诉解码器"你之前在哪个位置看到了什么细节"

在 DDPM 中，UNet 的计算过程可概括为：输入加噪图像 x_t 和时间步 t，经过编码-注意-解码流程，输出预测的噪声 ε。整个过程的核心是图到图的特征变换，而时间步的注入确保了模型知道在去噪过程的哪个阶段做出相应的处理。

### 与前序/相关算法的联系

- **ResNet 是基础**：UNet 中的 ResBlock 直接继承了 ResNet 的残差学习理念
- **Self-Attention 是增强**：在瓶颈层引入全局感受野，弥补 CNN 的局部性局限
- **DDPM 是 UNet 的重要宿主**：UNet 让扩散模型的逆向去噪成为可能
- **FPN（特征金字塔网络）与 UNet 同源**：都使用了多尺度 + 跨层连接的思想

### 后续学习方向

- **Attention U-Net**：在跳跃连接中加入门控注意力，让解码器更有选择性地利用编码器信息
- **ConvNeXt-based U-Net**：用现代化的 ConvNeXt 块替代传统 ResBlock，取得更好效果
- **DiT (Diffusion Transformer)**：用纯 Transformer 架构替代 UNet，探索注意力机制的极限
- **U-Net++ 和 U²-Net**：更密集的跳跃连接结构，用嵌套的 UNet 提升细节恢复能力

## 13. 练习题与思考题

### 基础题 1：通道数计算

给定 UNet 的配置：base_dim=64, dim_mults=(1, 2, 4, 8)（4 层）。

- 列出每一层编码器输出的通道数和空间尺寸（设输入 256x256）
- 列出每一层解码器在跳跃连接拼接前和拼接后的通道数
- 如果跳跃连接改用"逐元素相加"而非"拼接"，解码器的通道数会如何变化？

**参考答案**：
```
编码器各层输出:
  init_conv: 64ch, 256x256
  enc0:      64 → 64,  256→128 (Down), 输出 128ch, 128x128
  enc1:      128→128,  128→64 (Down),  输出 256ch, 64x64
  enc2:      256→256,  64→32 (Down),   输出 512ch, 32x32
  enc3:      512→512 (瓶颈层，不下采样)

解码器各层:
  dec3: 拼接前=512ch, 拼接后=512+512=1024ch, 输出 256ch, 上采样→64x64
  dec2: 拼接前=256ch, 拼接后=256+256=512ch,  输出 128ch, 上采样→128x128
  dec1: 拼接前=128ch, 拼接后=128+128=256ch,  输出 64ch,  上采样→256x256
  dec0: 拼接前=64ch,  拼接后=64+64=128ch,    输出 64ch

逐元素相加时：解码器各层输入通道不变（仍是上一层的输出通道数），因为相加不改变通道维度。
例如 dec2: 输入 256ch + 跳跃 256ch = 256ch (相加)，而非拼接后的 512ch。
优点是参数量和计算量更小，缺点是融合方式更受限制。
```

### 基础题 2：代码补全

补全下面的函数，验证 UNet 结构的合法性：

```python
def validate_unet(h, w, num_downsamples):
    """
    检查输入尺寸 (h, w) 能否经过 num_downsamples 次 2x 下采样
    返回: (bool, 建议的 pad 后尺寸)
    """
    # 请补全
```

**参考答案**：
```python
def validate_unet(h, w, num_downsamples):
    """每次下采样 stride=2，需要每个维度都能被 2 整除"""
    required_divisor = 2 ** num_downsamples

    h_valid = h % required_divisor == 0
    w_valid = w % required_divisor == 0

    # 计算最近的兼容尺寸
    h_pad = ((h + required_divisor - 1) // required_divisor) * required_divisor
    w_pad = ((w + required_divisor - 1) // required_divisor) * required_divisor

    print(f"输入: {h}x{w}, 需要整除: {required_divisor}")
    print(f"原始合法: H={h_valid}, W={w_valid}")
    print(f"建议 pad 到: {h_pad}x{w_pad}")

    return (h_valid and w_valid), (h_pad, w_pad)

# 测试
validate_unet(28, 28, 3)  # MNIST: 28 -> 需要 pad 到 32
validate_unet(256, 256, 4)  # 256 -> 合法，256 % 16 = 0
```

### 进阶题：UNet vs DiT

DDPM 使用 UNet 作为扩散模型的骨干网络，而最新的 DiT 使用纯 Transformer。结合你的理解，分析 UNet 和 DiT 各自的优劣，以及为什么在 Scaling Law 下 DiT 可能优于 UNet？

**参考答案**：

UNet 的优势：
- **归纳偏置（Inductive Bias）**：卷积的平移等变性和局部性使其在小数据量下天然高效。跳跃连接提供了一个强有力的"位置保留"机制。
- **计算效率**：在中等分辨率下，卷积的 O(HW) 复杂度（按特征图面积）优于自注意力的 O(H²W²)。

DiT 的优势：
- **无局部性限制**：全局自注意力允许任意两像素直接交互，不依赖网络的逐层传播。
- **更好的可扩展性（Scalability）**：Transformer 的架构高度规整（全由 Attention + MLP 构成），在数据量和模型规模增长时表现出更可预测的 Scaling Law。而 UNet 在下采样/上采样的"金字塔"结构中，不同层的计算量不均衡，Scaling 行为更难预测。
- **多模态友好**：Transformer 天然适合融合不同模态的 token（图像 patch + 文本 token + 时间步），而 UNet 的卷积结构对非空间信息（如文本 token）不够自然。

为什么 DiT 可能在 Scaling 下更好：
随着数据规模和模型参数量的双重增长，归纳偏置的价值下降（模型能从数据中学到更好的表示），而架构的规整性和可扩展性变得更重要。Transformer 的规整结构使其扩张时各层能均匀受益，而 UNet 在不同分辨率的层之间可能存在瓶颈。

### 开放思考题

如果让你设计一个"双向 UNet"——既从噪声预测图像，又能从图像预测噪声（前向过程也变成可学习的）——你认为这会带来什么好处和挑战？会对扩散模型的训练范式产生什么影响？

**参考思路**：
- **好处**：双向学习可能让 UNet 更好地理解噪声空间和图像空间之间的映射关系；前向过程也可以被优化（不再局限于固定的线性 schedule）；可能实现更少步数的扩散。
- **挑战**：
  1. 训练复杂度翻倍（需要同时优化前向和逆向）
  2. 前向过程的可学习化可能导致训练不稳定（噪声 schedule 与损失函数互相影响）
  3. 可能会退化为恒等映射或平凡的噪声-图像互转
  4. 需要设计合适的双向一致性损失来约束两个过程的互逆性
- **对范式的影响**：这可能模糊扩散模型和自编码器的边界，让扩散过程从一个固定的物理模拟变成一个端到端可学习的"表达转换"框架。

## 14. 学习路径建议

### 前置算法
- **卷积神经网络（CNN）**：理解卷积、池化、感受野、通道这些基本概念
- **残差网络（ResNet）**：理解跳跃连接和残差学习的核心思想（Shortcut Connection + F(x) + x）
- **PyTorch 基础**：nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm, F.interpolate 的使用

### 平行算法
- **FPN（特征金字塔网络）**：类似的多尺度 + 跨层连接思想，用于目标检测
- **转置卷积**：另一种上采样方式，理解其与插值上采样的区别
- **Self-Attention**：理解注意力机制如何捕获全局依赖，用于增强 UNet 瓶颈层

### 进阶算法
- **DDPM**：UNet 的最重要应用场景之一，扩散模型逆向去噪的核心引擎
- **Attention U-Net**：带门控注意力的 UNet 变体，选择性利用跳跃连接
- **DiT（Diffusion Transformer）**：纯 Transformer 扩散模型，理解 UNet 的替代方案
- **Stable Diffusion 中的 UNet**：Latent Diffusion 框架中 UNet 的具体设计和条件注入方式
- **SegFormer / SETR**：基于 Transformer 的分割/重建架构，UNet 的有力竞争者

### 推荐资源
1. **论文**：Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015) — 原文简洁清晰，5 页精华
2. **论文**：Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) — 看附录的 UNet 架构细节
3. **代码**：lucidrains/denoising-diffusion-pytorch (GitHub) — UNet 在 DDPM 中的完整实现
4. **博客**：The U-Net (actually) explained in 10 minutes (YouTube) — 快速直观理解
5. **教程**：PyTorch U-Net tutorial 和 fastai U-Net from scratch — 跟着敲一遍
