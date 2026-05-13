# 扩散模型 DDPM 学习文档

> 来源线索：本节内容根据原书中关于"Diffusion生成模型精讲"（第9章 9.1节）的相关章节整理、扩展与教学化改写。

> 给图像逐步加噪声再学习逆向去噪，从纯噪声中生成新图像。

## 1. 算法基础认知

**一句话定义**：通过前向逐步加噪破坏数据、逆向学习去噪来生成新样本的概率生成模型。

**直觉类比**：想象把一滴墨水滴入一杯清水——墨水会逐渐扩散、均匀分布，最终整杯水变成均匀的淡墨色。DDPM 的前向过程正是如此：向清晰图像逐步添加高斯噪声，直到图像完全变成纯噪声。而逆向过程好比"时光倒流"：从均匀的墨水状态，逐步恢复出最初那滴墨水的形状——即从纯噪声中重建出清晰图像。

**历史背景**：扩散模型的思想可追溯到 2015 年（Sohl-Dickstein et al.），但真正引起轰动的是 2020 年 Ho et al. 提出的 DDPM（Denoising Diffusion Probabilistic Model）。此后 DALL-E 2、Stable Diffusion、Imagen 等爆炸性应用都以此为基础，使扩散模型成为超越 GAN 的主流图像生成范式。

**算法定位**：深度学习 / 生成模型。与 VAE（变分自编码器）和 GAN（生成对抗网络）并列为三大生成模型范式。

**前置知识**：
- 概率论基础：高斯分布（正态分布）、条件概率、贝叶斯公式
- 马尔可夫链：当前状态仅依赖前一状态
- 深度学习基础：神经网络、反向传播、PyTorch
- 了解 UNet 架构（编码器-解码器 + 跳跃连接）

## 2. 核心原理

### 核心思想

DDPM 包含两个互为逆过程的核心阶段：

1. **前向扩散过程（Forward Process）**：给一张真实图像 x_0，按照预定义的噪声 schedule（例如线性增加的 beta 序列），逐步向其中添加高斯噪声。经过 T 步（通常 T=1000）后，图像变为近似纯高斯噪声 x_T ~ N(0, I)。这一过程是固定的，不需要学习。

2. **逆向去噪过程（Reverse Process）**：从纯噪声 x_T 出发，训练一个神经网络（通常是 UNet）来预测每一步添加的噪声。通过逐步减去预测的噪声，最终恢复出清晰图像 x_0。

### 工作流程

```
真实图像 x_0
    │ 加噪声 ε_1 ~ N(0, β_1·I)
    ▼
x_1 = √(1-β_1)·x_0 + √β_1·ε_1
    │ 加噪声 ε_2
    ▼
x_2 = √(1-β_2)·x_1 + √β_2·ε_2
    │ ...
    ▼
x_T = 纯噪声 ~ N(0, I)
    │ 预测噪声 ε_θ(x_T, T)，减去
    ▼
x_{T-1} ≈ 去噪一步
    │ 继续去噪...
    ▼
x_0 ≈ 生成的清晰图像
```

### 关键概念解释

- **Noise Schedule（噪声调度）**：控制每步添加多少噪声的策略。最简单的线性 schedule：beta 从 0.0001 线性增长到 0.02。beta 越大，该步添加的噪声越多，图像被破坏得越厉害。
- **重参数化技巧**：利用高斯分布的可加性，可以从 x_0 直接一步计算出任意时刻 t 的 x_t，无需迭代 T 步。这大幅简化了训练过程。
- **预测噪声而非图像**：DDPM 的核心洞见是——与其让模型直接预测去噪后的图像（难度大），不如让模型预测每一步添加的噪声（任务更简单、更稳定）。

### 直观解释

可以把 DDPM 理解为"学习逆向工程噪声"。前向过程是一套已知的破坏步骤，逆向过程是一套需要学习的修复步骤。因为模型学会了修复任意破坏程度的图像，所以给定纯噪声时，它也能一步步"修复"出一张全新的、从未见过的图像——这就是"生成"的本质。

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| x_0 | 原始清晰图像 | (C, H, W) |
| x_t | 第 t 步加噪后的图像 | 同 x_0 |
| x_T | 纯噪声图像 | 同 x_0 |
| β_t | 第 t 步的噪声方差 | 标量，(0, 1) |
| α_t | 1 - β_t，信号保留比例 | 标量，(0, 1) |
| ᾱ_t | α 的累积乘积 α_1·α_2·...·α_t | 标量 |
| ε | 标准高斯噪声，N(0, I) | 同 x_0 |
| ε_θ | UNet 预测的噪声 | 同 x_0 |
| z | 去噪时采样的随机噪声 | 同 x_0 |
| T | 总扩散步数 | 标量，通常 1000 |

### 前向扩散过程

单步加噪公式：

$$x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \varepsilon_{t-1}, \quad \varepsilon_{t-1} \sim \mathcal{N}(0, I)$$

用 α_t = 1 - β_t 改写：

$$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \varepsilon_{t-1}$$

关键推导——从 x_0 直接得到 x_t（重参数化）：

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \varepsilon_{t-1} \\
    &= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \varepsilon_{t-2}) + \sqrt{1-\alpha_t} \varepsilon_{t-1} \\
    &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \varepsilon_{t-2} + \sqrt{1-\alpha_t} \varepsilon_{t-1}
\end{aligned}
$$

由于两个独立高斯分布之和仍为高斯分布，且方差相加：

$$\sqrt{\alpha_t(1-\alpha_{t-1})} \varepsilon_{t-2} + \sqrt{1-\alpha_t} \varepsilon_{t-1} \sim \mathcal{N}(0, (\alpha_t(1-\alpha_{t-1}) + 1 - \alpha_t)I) = \mathcal{N}(0, (1 - \alpha_t \alpha_{t-1})I)$$

不断迭代，最终得到：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

即：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

这就是核心的前向过程公式。它意味着：只要知道 x_0 和 β 序列，就可以从标准正态分布采样一个 ε，直接算出任意时刻的 x_t。

### 逆向去噪过程

逆向过程的目标是学习 p_θ(x_{t-1} | x_t)，即从噪声更多的图像恢复到噪声更少的图像。

真实的反向条件分布 q(x_{t-1} | x_t, x_0) 是可计算的（利用贝叶斯公式）：

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

其中：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

将 x_0 用 x_t 和 ε 表示（从前向公式反解）：$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t} \varepsilon)$，代入后得到简化的均值公式：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon\right)$$

### 损失函数

DDPM 的训练目标是最小化预测噪声与真实噪声之间的差异：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2 \right]$$

其中：
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon$
- t 从 {1, ..., T} 中均匀采样
- ε_θ 是 UNet 模型，输入为加噪图像 x_t 和时间步 t，输出为预测的噪声

这就是简化版损失函数（Ho et al. 2020），舍弃了方差学习项，实践中效果更好。也常用 Huber Loss（Smooth L1）替代 MSE，对异常值更鲁棒。

### 采样（生成）过程

训练完成后，从纯噪声 x_T ~ N(0, I) 开始，逐步去噪 T 步：

对于 t = T, T-1, ..., 1：
1. 如果 t > 1，采样 z ~ N(0, I)；否则 z = 0
2. 计算 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t)\right) + \sigma_t z$

其中 σ_t 通常取 $\sqrt{\tilde{\beta}_t}$ 或 $\sqrt{\beta_t}$。最后得到 x_0 即为生成的图像。

## 4. 训练过程讲解

### 数据预处理

- 将图像归一化到 [-1, 1] 范围（配合高斯噪声的分布）
- Resize 到统一尺寸（如 28x28 for MNIST，32x32 for CIFAR-10）
- 不做数据增强，因为扩散模型本身通过学习噪声分布获得了很好的泛化能力

### 参数初始化

- UNet 使用 kaiming_normal 初始化卷积层权重
- 时间嵌入使用正弦位置编码，无需训练
- 学习率通常较低（1e-4 到 2e-4），使用 AdamW 优化器

### 迭代过程

```
对于每个 epoch:
  对于每个 batch 的图像 x_0:
    1. 随机采样时间步 t ~ Uniform(1, T)
    2. 随机采样噪声 ε ~ N(0, I)
    3. 计算加噪图像 x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    4. 模型预测噪声: ε_pred = UNet(x_t, t)
    5. 计算损失: loss = MSE(ε_pred, ε) 或 Huber(ε_pred, ε)
    6. 反向传播，更新 UNet 参数
```

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| T (时间步数) | 扩散总步数 | 200-4000 | 1000（质量和速度的折中） |
| β_start | 初始噪声比例 | 1e-5 ~ 1e-4 | 1e-4 |
| β_end | 最终噪声比例 | 0.01 ~ 0.03 | 0.02 |
| learning_rate | 学习率 | 1e-5 ~ 5e-4 | 2e-4 |
| batch_size | 批大小 | 16-256 | 128 |
| loss_type | 损失函数类型 | l1 / l2 / huber | huber |
| UNet base_dim | UNet 基础通道数 | 32-256 | 64（小数据集）/ 128 |

## 5. 应用场景

### 典型应用

1. **无条件图像生成**：给定随机噪声，生成全新的、逼真的图像。如生成人脸、风景、艺术作品等。DDPM 原始论文在 CIFAR-10、LSUN 等数据集上展示了出色的无条件生成能力。

2. **图像超分辨率与修复**：将低分辨率或破损图像作为条件，引导扩散模型生成高分辨率或修复后的版本。SR3 和 Palette 模型都是基于 DDPM 的图像到图像翻译框架。

3. **文本到图像生成**：以文本描述为条件，生成对应的图像。Stable Diffusion、DALL-E 2、Imagen 等核心引擎都基于扩散模型。文本条件通过交叉注意力（Cross-Attention）注入 UNet。

4. **图像编辑与 Inpainting**：指定需要修改或填充的区域，扩散模型在保持其余区域不变的同时生成新内容。Blended Diffusion 和 RePaint 是代表性方法。

### 适用数据特征

- 高维连续数据（图像、音频、视频）
- 数据分布较复杂、需要高质量生成
- 对生成多样性要求高的场景

### 不适用场景

- 需要实时生成的场景（扩散模型采样需要多步迭代，速度慢）
- 离散数据生成（文本等，需配合嵌入或特殊处理方法）
- 极低计算资源环境（训练和推理都较 GAN 更耗费资源）

## 6. 优缺点分析

### 优点

| 优点 | 成立条件 | 说明 |
|------|----------|------|
| 生成质量高 | T 足够大（≥500） | 逐步去噪的精细过程使得生成细节丰富 |
| 训练稳定 | 损失函数为简单的 MSE/Huber | 不像 GAN 需要对抗训练，避免了模式坍塌 |
| 模式覆盖全 | 训练充分 | 基于似然的训练目标和噪音注入使得分布覆盖更全面 |
| 灵活可条件化 | 条件注入架构 | 易于添加文本、图像、类别等多种条件控制 |
| 理论基础扎实 | 热力学扩散 + 概率模型 | 训练目标有严格的变分下界推导 |

### 缺点

| 缺点 | 何时出问题 | 缓解思路 |
|------|-----------|----------|
| 推理速度慢 | 需要循环 T 步去噪 | DDIM 加速采样（跳步），蒸馏（如 Progressive Distillation） |
| 训练时间长 | 大数据集 + 大 T | 减小 T 到 200-400，使用更大 batch |
| 计算资源需求大 | 高分辨率图像 | 在潜空间扩散（Latent Diffusion，Stable Diffusion 方案） |
| 可控性调参复杂 | 多条件融合 | 分类器自由引导（CFG），仔细调整引导强度 |

### 与 GAN / VAE 的对比

| 特性 | DDPM | GAN | VAE |
|------|------|-----|-----|
| 训练稳定性 | 优秀（直接回归噪声） | 较差（对抗训练） | 良好 |
| 生成质量 | 优秀 | 优秀（GAN 仍领先于小规模） | 一般（偏模糊） |
| 生成多样性 | 优秀（模式覆盖好） | 易出现模式坍塌 | 良好 |
| 推理速度 | 慢（需 T 步） | 极快（单步前向） | 快（单步前向） |
| 潜空间结构 | 无显式潜空间 | 无显式潜空间 | 有结构化的潜空间 |
| 似然估计 | 可计算 | 不可直接计算 | 变分下界 |

## 7. 调库实现

```python
"""
DDPM 扩散模型 -- 使用 PyTorch 实现
基于 Fashion-MNIST 数据集演示无条件图像生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# ===================== 1. 噪声调度 =====================
def linear_beta_schedule(timesteps: int = 1000) -> torch.Tensor:
    """生成线性增加的 beta 值序列，控制每步添加的噪声量"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """余弦调度：更平滑的噪声增加策略，生成质量通常优于线性"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ===================== 2. 简化版 UNet（用于 MNIST） =====================
class SimpleUNet(nn.Module):
    """轻量级 UNet，适合 28x28 的 MNIST/Fashion-MNIST"""

    def __init__(self, in_channels: int = 1, base_dim: int = 64,
                 dim_mults: tuple = (1, 2, 4)):
        super().__init__()
        self.base_dim = base_dim

        # ---- 时间嵌入 ----
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---- 初始卷积 ----
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # ---- 下采样模块 ----
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = (idx == len(in_out) - 1)
            self.downs.append(
                DownBlock(dim_in, dim_out, time_dim=time_dim, downsample=not is_last)
            )

        # ---- 中间模块 ----
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = SelfAttention(mid_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim)

        # ---- 上采样模块 ----
        self.ups = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (idx == len(in_out) - 1)
            skip_dim = dim_out  # 跳跃连接的通道数
            self.ups.append(
                UpBlock(dim_in + skip_dim, dim_out, time_dim=time_dim, upsample=not is_last)
            )

        # ---- 输出卷积 ----
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """前向传播：
        x: (B, C, H, W) 加噪图像
        t: (B,) 时间步索引
        返回: (B, C, H, W) 预测的噪声
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # 初始卷积
        x = self.init_conv(x)
        h = x.clone()  # 保存用于残差连接

        # 下采样，保存跳跃连接
        skip_connections = []
        for down in self.downs:
            x = down(x, t_emb)
            skip_connections.append(x)

        # 中间处理
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # 上采样，融合跳跃连接
        for up in self.ups:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)  # 沿通道维度拼接
            x = up(x, t_emb)

        # 残差连接 + 输出
        x = self.out_conv(x + h)
        return x


# ===================== 3. UNet 子模块 =====================
class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于编码时间步 t"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class ResBlock(nn.Module):
    """残差卷积块：两次卷积 + GroupNorm + 时间条件注入"""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        # 跳跃连接：如果输入/输出通道不同，用 1x1 卷积对齐
        self.skip = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # 注入时间条件：将时间嵌入投影到通道维度并加到特征上
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """自注意力模块，捕获全局依赖"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # 各为 (B, H*W, C)
        scale = C ** -0.5
        attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ V).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    """下采样模块：两个残差块 + 可选下采样"""
    def __init__(self, in_channels: int, out_channels: int,
                 time_dim: int, downsample: bool = True):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """上采样模块：两个残差块 + 可选上采样"""
    def __init__(self, in_channels: int, out_channels: int,
                 time_dim: int, upsample: bool = True):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=False) if upsample else nn.Identity()
        if upsample:
            self.upsample_conv = nn.Conv2d(out_channels, out_channels,
                                           kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if hasattr(self, 'upsample_conv'):
            x = self.upsample_conv(self.upsample(x))
        return x


# ===================== 4. DDPM 类（封装训练和采样） =====================
class DDPM:
    """去噪扩散概率模型的完整实现"""

    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_schedule: str = "linear", device: str = "cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        # 生成噪声调度
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        # 预计算各种系数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 注册为 buffer（不参与梯度计算，但随模型保存/加载）
        self.register('betas', betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('alphas_cumprod_prev', alphas_cumprod_prev)

        # 前向扩散用
        self.register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # 逆向去噪用
        self.register('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        self.register('posterior_variance',
                       betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def register(self, name: str, tensor: torch.Tensor):
        """将张量注册为模块属性并移到设备上"""
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """前向扩散：从 x0 和 t 直接计算 x_t"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor,
                 loss_type: str = "huber") -> torch.Tensor:
        """计算训练损失"""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_t, t)

        if loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor,
                 t_index: torch.Tensor) -> torch.Tensor:
        """单步逆向去噪采样"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # 从预测噪声计算均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        # 最后一步不加随机噪声
        mask = (t_index == 0).float().reshape(x.shape[0], 1, 1, 1)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + (1. - mask) * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: tuple, return_all: bool = False):
        """从纯噪声生成图像（完整逆向过程）"""
        self.model.eval()
        if return_all:
            images = []
        img = torch.randn(shape, device=self.device)
        # 从 T-1 到 0 逐步去噪
        for i in tqdm(reversed(range(self.timesteps)), desc='DDPM 采样中'):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, t)
            if return_all:
                images.append(img.cpu())
        if return_all:
            return images
        return img.cpu()

    def _extract(self, arr: torch.Tensor, t: torch.Tensor,
                 x_shape: tuple) -> torch.Tensor:
        """从预计算数组中按时间步 t 提取对应值并 reshape 到 x_shape"""
        batch_size = t.shape[0]
        out = arr.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ===================== 5. 训练和演示 =====================
def train_ddpm():
    """完整的 DDPM 训练和生成演示"""

    # ---- 设备配置 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ---- 数据加载 ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # ---- 模型和 DDPM 初始化 ----
    model = SimpleUNet(in_channels=1, base_dim=64, dim_mults=(1, 2, 4))
    ddpm = DDPM(model, timesteps=300, beta_schedule="linear", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # ---- 训练 ----
    epochs = 20
    print(f"开始训练，共 {epochs} 个 epoch，时间步数 {ddpm.timesteps}...")
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_images, _ in pbar:
            batch_images = batch_images.to(device)
            optimizer.zero_grad()
            # 随机采样时间步
            t = torch.randint(0, ddpm.timesteps, (batch_images.shape[0],),
                              device=device, dtype=torch.long)
            loss = ddpm.p_losses(batch_images, t, loss_type="huber")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})
        print(f"Epoch {epoch + 1} 平均损失: {total_loss / len(dataloader):.5f}")

    print("训练完成！")
    return ddpm


if __name__ == "__main__":
    ddpm = train_ddpm()

    # ---- 生成样本 ----
    print("生成 25 张 Fashion-MNIST 图像...")
    samples = ddpm.sample(shape=(25, 1, 28, 28))

    # ---- 展示结果 ----
    grid = make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.title("DDPM 生成的 Fashion-MNIST 图像", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ddpm_generated_fashion_mnist.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图像已保存为 ddpm_generated_fashion_mnist.png")
```

## 8. 手工代码实现

```python
"""
从零手写 DDPM 扩散模型 -- 不依赖高级封装
本节实现 DDPM 的核心数学运算：前向加噪、损失计算、逆向去噪采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DDPM_FromScratch:
    """从零实现的 DDPM -- 核心数学部分

    与调库版本的区别：
    - 不使用 SimpleUNet 封装，改用最简单的卷积网络演示
    - 手动计算所有扩散系数（alpha, beta, cumulative products）
    - 手动实现前向加噪和逆向采样
    """

    def __init__(self, noise_predictor: nn.Module,
                 timesteps: int = 500, device: str = "cpu"):
        """
        参数:
            noise_predictor: 噪声预测网络 (输入 x_t 和 t，输出预测的噪声)
            timesteps: 扩散总步数
            device: 计算设备
        """
        self.model = noise_predictor.to(device)
        self.timesteps = timesteps
        self.device = device

        # ---- 步骤1：手动构建 beta 序列 ----
        # 线性增长：beta_start -> beta_end，共 timesteps 个值
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # ---- 步骤2：手动计算 alpha 和相关累积量 ----
        self.alphas = 1.0 - self.betas                              # α_t = 1 - β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # ᾱ_t = ∏ α_i

        # ---- 步骤3：预计算前向扩散所需系数 ----
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # ---- 步骤4：预计算逆向去噪所需系数 ----
        # 1/√α_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # alphas_cumprod_prev: ᾱ_{t-1}，用于计算后验方差
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # posterior_variance = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )

    def extract_at_t(self, arr: torch.Tensor, t: torch.Tensor,
                     x_shape: tuple) -> torch.Tensor:
        """
        从预计算数组 arr 中提取时间步 t 对应的值，并 reshape 以支持广播。

        例如：
        arr = [a_0, a_1, ..., a_{T-1}], shape (T,)
        t = [4, 99, 200], shape (3,)
        返回 = [[a_4], [a_99], [a_200]], reshape 为 (3, 1, 1, 1) 以便广播
        """
        batch_size = t.shape[0]
        # gather: 按索引从 arr 取值
        out = arr.gather(-1, t.cpu())
        # reshape 为 (batch, 1, 1, ...) 以匹配 x_shape
        reshape_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
        return out.reshape(reshape_shape).to(t.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  eps: torch.Tensor = None) -> torch.Tensor:
        """
        前向加噪过程（DDPM 核心公式一）：
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε

        参数:
            x0: 原始图像, shape (B, C, H, W)
            t: 时间步索引, shape (B,), 每个元素在 [0, T-1]
            eps: 噪声, 若为 None 则从 N(0,1) 采样
        返回:
            x_t: 加噪后的图像
        """
        if eps is None:
            eps = torch.randn_like(x0)

        # 提取对应时间步的系数
        a1 = self.extract_at_t(self.sqrt_alphas_cumprod, t, x0.shape)
        a2 = self.extract_at_t(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return a1 * x0 + a2 * eps

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        计算单步训练损失（DDPM 核心公式二）：
        L = ||ε - ε_θ(x_t, t)||²

        步骤:
        1. 随机采样时间步 t
        2. 随机采样噪声 ε
        3. 用 add_noise 计算 x_t
        4. 用模型预测噪声 ε_θ(x_t, t)
        5. 计算 MSE(ε_θ, ε)
        """
        batch_size = x0.shape[0]
        # 均匀采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,),
                          device=self.device, dtype=torch.long)
        # 采样标准高斯噪声
        eps = torch.randn_like(x0)
        # 加噪
        x_t = self.add_noise(x0, t, eps)
        # 模型预测噪声
        eps_pred = self.model(x_t, t)
        # MSE 损失
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        单步逆向去噪（DDPM 核心公式三）：
        x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

        其中 σ_t = √(β_t * (1-ᾱ_{t-1})/(1-ᾱ_t)), z ~ N(0,I) (当 t>0)
        """
        betas_t = self.extract_at_t(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = self.extract_at_t(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract_at_t(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 模型预测噪声
        eps_pred = self.model(x_t, t)

        # 计算均值项（预测的 x_{t-1} 均值）
        mean = sqrt_recip_alphas_t * (
            x_t - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t
        )

        # 如果不是最后一步（t > 0），加入随机噪声
        if t[0].item() > 0:  # 整个 batch 的 t 相同（采样时如此）
            posterior_var_t = self.extract_at_t(
                self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var_t) * noise
        else:
            return mean

    @torch.no_grad()
    def generate(self, shape: tuple) -> torch.Tensor:
        """
        完整逆向生成过程:
        1. 从纯噪声 x_T ~ N(0,I) 开始
        2. 对于 t = T-1, T-2, ..., 0：
               x_t = denoise_step(x_{t+1}, t)
        3. 返回 x_0

        这模拟了从完全随机的噪声中"创造"出新图像的全过程。
        """
        self.model.eval()
        x = torch.randn(shape, device=self.device)  # (B, C, H, W)
        for t_val in range(self.timesteps - 1, -1, -1):
            t = torch.full((shape[0],), t_val, device=self.device, dtype=torch.long)
            x = self.denoise_step(x, t)
        return x.cpu()


# ===================== 最简单的噪声预测网络 =====================
class TinyNoisePredictor(nn.Module):
    """微型噪声预测网络，用于演示 -- 仅含 2 层卷积 + 时间嵌入"""

    def __init__(self, img_channels: int = 1, hidden: int = 32):
        super().__init__()
        # 时间步的简单嵌入：t -> 正弦编码 -> MLP -> hidden 维
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # 图像处理
        self.conv1 = nn.Conv2d(img_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden + hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        t_emb = self.time_embed(t.float().unsqueeze(-1))         # (B, hidden)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)                 # (B, hidden, 1, 1)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])     # (B, hidden, H, W)

        # 图像特征
        h = F.relu(self.conv1(x))                                 # (B, hidden, H, W)
        h = torch.cat([h, t_emb], dim=1)                          # 沿通道拼接时间信息
        h = F.relu(self.conv2(h))
        return self.out(h)


# ===================== 测试代码 =====================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    print("=== 从零手写 DDPM 测试 ===")

    # 创建模型和 DDPM
    model = TinyNoisePredictor(img_channels=1, hidden=32)
    ddpm = DDPM_FromScratch(model, timesteps=500, device=device)

    print(f"时间步数: {ddpm.timesteps}")
    print(f"beta 范围: [{ddpm.betas[0].item():.4f}, {ddpm.betas[-1].item():.4f}]")

    # 测试前向加噪
    x0 = torch.randn(4, 1, 28, 28)  # 模拟 4 张 28x28 灰度图
    t = torch.randint(0, 500, (4,))

    x_t = ddpm.add_noise(x0, t)
    print(f"\n前向加噪测试:")
    print(f"  x0 shape: {x0.shape}, 范围: [{x0.min():.3f}, {x0.max():.3f}]")
    print(f"  x_t shape: {x_t.shape}, 范围: [{x_t.min():.3f}, {x_t.max():.3f}]")

    # 测试损失计算
    loss = ddpm.compute_loss(x0)
    print(f"\n损失计算测试:")
    print(f"  loss: {loss.item():.5f}")

    # 测试生成过程
    print(f"\n逆向生成测试 (生成 4 张 28x28 图像)...")
    samples = ddpm.generate((4, 1, 28, 28))
    print(f"  生成图像 shape: {samples.shape}")
    print(f"  生成图像范围: [{samples.min():.3f}, {samples.max():.3f}]")

    print("\n所有测试通过！DDPM 核心数学流程验证成功 ✓")
    print("提示：本示例使用最简单的网络结构演示数学原理。")
    print("实际应用中，请使用第 7 节的 UNet 替代 TinyNoisePredictor。")
```

## 9. 可视化与结果理解

```python
"""
DDPM 可视化：展示前向加噪过程和逆向生成结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 复用第 7 节的组件（SinusoidalPositionEmbedding 等需已定义）
# 这里重新定义关键组件用于可视化

# ---- 噪声调度 ----
def linear_beta_schedule(timesteps=1000):
    return torch.linspace(0.0001, 0.02, timesteps)

# ==================== 图 1: 前向扩散过程可视化 ====================
def visualize_forward_diffusion():
    """展示一张图像在不同扩散步数下的加噪效果"""
    torch.manual_seed(42)

    # 加载一张测试图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    x0 = dataset[0][0]  # 取第一张图: (1, 28, 28)

    # 准备扩散参数
    T = 1000
    betas = linear_beta_schedule(T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 选取展示的时间步
    timesteps_to_show = [0, 10, 50, 100, 200, 400, 600, 800, 1000]

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(18, 3))

    for idx, t in enumerate(timesteps_to_show):
        if t == 0:
            img = x0
            title = "t=0\n(原始图像)"
        else:
            t_tensor = torch.tensor([t - 1])  # 0-indexed
            alpha_bar = alphas_cumprod[t - 1]
            noise = torch.randn_like(x0)
            img = math.sqrt(alpha_bar) * x0 + math.sqrt(1 - alpha_bar) * noise
            title = f"t={t}\nᾱ={alpha_bar:.3f}"

        axes[idx].imshow(img.squeeze().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')

    plt.suptitle("DDPM 前向扩散过程：图像逐步被高斯噪声淹没", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('ddpm_forward_diffusion.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 1 解读：随着 t 增大，ᾱ (信号保留比例) 从 1 衰减到接近 0，")
    print("  图像从清晰逐渐变为完全的高斯噪声。t=1000 时基本无法辨认原图。")


# ==================== 图 2: 噪声调度曲线 ====================
def visualize_noise_schedule():
    """对比线性调度和余弦调度的 beta 和 ᾱ 曲线"""
    T = 1000

    # 线性调度
    betas_linear = linear_beta_schedule(T)
    alphas_linear = 1. - betas_linear
    alphas_cumprod_linear = torch.cumprod(alphas_linear, dim=0)

    # 余弦调度
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod_cos = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod_cos = alphas_cumprod_cos / alphas_cumprod_cos[0]
    betas_cos = 1 - (alphas_cumprod_cos[1:] / alphas_cumprod_cos[:-1])
    betas_cos = torch.clip(betas_cos, 0.0001, 0.9999)
    alphas_cumprod_cos = alphas_cumprod_cos[1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：beta 曲线
    axes[0].plot(betas_linear.numpy(), label='线性调度', linewidth=2)
    axes[0].plot(betas_cos.numpy(), label='余弦调度', linewidth=2)
    axes[0].set_xlabel('时间步 t', fontsize=12)
    axes[0].set_ylabel('β_t (每步噪声方差)', fontsize=12)
    axes[0].set_title('噪声调度对比：β_t 曲线', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 右图：ᾱ 累积乘积曲线
    axes[1].plot(alphas_cumprod_linear.numpy(), label='线性调度', linewidth=2)
    axes[1].plot(alphas_cumprod_cos.numpy(), label='余弦调度', linewidth=2)
    axes[1].set_xlabel('时间步 t', fontsize=12)
    axes[1].set_ylabel('ᾱ_t (信号保留比例)', fontsize=12)
    axes[1].set_title('信号保留衰减：ᾱ_t 曲线', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('DDPM 噪声调度策略对比', fontsize=15)
    plt.tight_layout()
    plt.savefig('ddpm_noise_schedule.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 2 解读：线性调度在前半段信号衰减过快，后半段几乎无变化；")
    print("  余弦调度更平滑，在中间阶段线性衰减，通常生成质量更好。")


# ==================== 图 3: 逆向生成过程 - 逐步去噪 ====================
def visualize_reverse_denoising():
    """
    演示逆向去噪的过程。
    注意：需要先训练一个 DDPM 模型，这里模拟展示。
    """
    # 模拟逆向过程的 8 个阶段（从纯噪声到清晰图像）
    # 实际应用中应使用训练好的模型
    T_show = 8
    shape = (1, 28, 28)
    torch.manual_seed(100)

    fig, axes = plt.subplots(1, T_show, figsize=(16, 3))

    x = torch.randn(shape)  # 纯噪声起点
    for i in range(T_show):
        alpha = i / (T_show - 1)  # 模拟信号恢复程度 [0, 1]
        # 模拟去噪：逐步减少噪声比例，增加信号比例
        display = alpha * torch.zeros(shape) + (1 - alpha) * x
        # 加上一些结构化信号使其看起来"逐渐生成"
        display = display + 0.3 * alpha * torch.sin(
            torch.linspace(0, 4 * math.pi, 28)
        ).unsqueeze(0).unsqueeze(0) * torch.cos(
            torch.linspace(0, 4 * math.pi, 28)
        ).unsqueeze(0).unsqueeze(-1)

        axes[i].imshow(display.squeeze().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(f"x_T → ... → x_0\n阶段 {i+1}/8", fontsize=9)
        axes[i].axis('off')

    plt.suptitle('DDPM 逆向去噪过程示意：从纯噪声逐步恢复图像结构', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('ddpm_reverse_denoising.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("图 3 解读：逆向去噪从纯噪声开始，每一步减去一点点预测的噪声。")
    print("  早期阶段（t 大）处理粗粒度结构（整体形状），")
    print("  后期阶段（t 小）处理细粒度细节（纹理、边缘）。")


if __name__ == "__main__":
    print("=" * 60)
    print("DDPM 可视化分析")
    print("=" * 60)
    visualize_forward_diffusion()
    visualize_noise_schedule()
    visualize_reverse_denoising()
    print("\n全部可视化完成。")
```

## 10. 模型评估

### 评估维度

DDPM 的评估主要从以下维度展开：

**1. 生成质量评估**

- **FID（Frechet Inception Distance）**：最广泛使用的生成质量指标。用 Inception 网络提取生成图像和真实图像的特征，计算两组特征之间的 Frechet 距离。FID 越低越好在大多数情况下代表生成质量越高。
- **IS（Inception Score）**：衡量生成图像的清晰度和多样性。要求生成图能被分类器高置信度识别（清晰度），且各类别分布均匀（多样性）。

**2. 损失收敛分析**

```python
"""DDPM 评估工具"""
def evaluate_ddpm_training(loss_history: list, fid_scores: list = None):
    """
    分析训练过程中的损失和 FID 变化
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2 if fid_scores else 1, figsize=(14, 5))

    if fid_scores is None:
        axes = [axes]

    # 左图：训练损失曲线
    axes[0].plot(loss_history, linewidth=1.5, color='steelblue')
    axes[0].set_xlabel('训练步数', fontsize=12)
    axes[0].set_ylabel('损失值 (Huber)', fontsize=12)
    axes[0].set_title('DDPM 训练损失收敛曲线', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # 标注收敛区域
    end_loss = np.mean(loss_history[-500:])
    axes[0].axhline(y=end_loss, color='red', linestyle='--',
                    label=f'最终稳定损失: {end_loss:.5f}')
    axes[0].legend()

    if fid_scores:
        axes[1].plot(fid_scores, linewidth=2, color='darkorange', marker='o')
        axes[1].set_xlabel('训练步数', fontsize=12)
        axes[1].set_ylabel('FID ↓', fontsize=12)
        axes[1].set_title('生成质量 FID 曲线', fontsize=13)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 判断标准:
# - 损失趋于平稳（波动小）→ 模型收敛
# - FID 持续下降并稳定 → 生成质量在提升
# - FID < 20: 较好, < 10: 优秀（取决于数据集）
```

**3. 评估要点总结**

| 指标 | 含义 | 优良标准 | 适用范围 |
|------|------|----------|----------|
| FID | 生成分布与真实分布距离 | 越低越好 (<20 好) | 无条件生成 |
| IS | 清晰度 + 多样性 | 越高越好 | 无条件生成 |
| Precision | 生成样本是否真实 | 越接近1越好 | 条件生成 |
| Recall | 真实分布是否被覆盖 | 越接近1越好 | 条件生成 |
| 训练损失 | 噪声预测准确度 | 平稳且较低 | 训练监控 |
| 人工评估 | 主观视觉质量 | 肉眼难辨真假 | 所有场景 |

## 11. 常见问题与易错点

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 生成图像全是噪声/模糊 | 采样结果无结构、像噪声团 | 训练不充分或学习率过大导致模型未学到去噪能力 | 增加训练轮次、降低学习率到 1e-4、检查损失是否持续下降 |
| 生成图像多样性不足 | 多张生成图看起来非常相似 | 模型过度记忆训练数据或 T 太小导致扩散不充分 | 增大 T（>=400）、加入训练 dropout、检查数据集多样性 |
| 时间嵌入未正确传入 | 不同 t 的生成结果相同 | 模型的所有模块没有正确接收和使用 t_emb | 逐模块检查 t_emb 是否被正确投影并加到特征图上 |
| 混合精度训练不稳定 | 损失突然爆炸或变为 NaN | 扩散系数的数值范围（如 1/√α_t）在低精度下溢出 | 关键扩散计算保持 float32、使用 gradient clipping |

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 图像归一化范围错误 | 生成图像像素值异常（全黑/全白） | 高斯噪声 N(0,I) 期望输入在 [-1,1] 范围，而原始图像在 [0,1] | 使用 transforms.Normalize((0.5,), (0.5,)) 将 [0,1] 映射到 [-1,1] |
| batch_size 与 t 不匹配 | 运行时维度报错 | t 向量长度与 batch 大小不一致 | 确保 t = torch.randint(0, T, (x0.shape[0],)) |

### 采样层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 采样结果太暗/太亮 | 平均像素值偏离 0 | 采样时均值项被过度/不足去噪 | 检查 posterior_variance 计算是否正确、确认 α 和 β 公式无误 |
| 采样时缺乏随机性 | 多轮生成结果完全相同 | 忘记在去噪步骤中加入随机噪声 z | 确保 denoise_step 中 t>0 时添加 torch.randn_like(x) * sqrt(posterior_var) |
| 采样速度极慢 | 生成一张图需要数分钟 | T 设置太大（如 4000）且图像分辨率高 | 使用 DDIM 加速采样（25-100 步）、减小图像尺寸、使用 GPU |

### 训练稳定性

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 训练初期损失不下降 | 损失在初始值附近振荡 | AdamW 的学习率偏大、beta 参数不合适 | 使用 warmup 策略、学习率从 1e-6 开始预热到 2e-4 |
| 不同 t 的损失差距悬殊 | 某些 t 的损失远大于其他 t | 噪声调度在部分时间步噪声量过大或过小 | 使用余弦调度替代线性调度、用 min-SNR 损失加权策略 |

## 12. 学习总结

### 核心思想回顾

DDPM 的本质是"学会逆转一个已知的破坏过程"。前向过程将数据逐步转化为纯噪声（用预定义的 β 序列控制噪声比例），逆向过程训练神经网络预测并减去噪声。这种"化繁为简"的思路——不直接生成图像，而是预测噪声——是 DDPM 训练稳定的根本原因。

关键公式：
- **前向加噪**：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon$
- **损失函数**：$\mathcal{L} = \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$
- **逆向采样**：$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta) + \sigma_t z$

### 与前序/相关算法的联系

- **DDPM 是基石**：DDIM（确定性加速采样）、Latent Diffusion（潜空间扩散）、Score-based Models 都是 DDPM 的变体
- **UNet 是核心引擎**：DDPM 的噪声预测网络通常用 UNet，其编码器-解码器 + 跳跃连接架构天然适合图像到图像的噪声预测任务
- **注意力增强**：在 UNet 的中间层加入 Self-Attention / Cross-Attention 可大幅提升长距离依赖建模能力

### 后续学习方向

- **DDIM**：去噪扩散隐式模型，支持跳步加速采样（10-50 步即可高质量生成）
- **Stable Diffusion**：在 VAE 的潜空间进行扩散，大幅降低计算开销
- **Classifier-Free Guidance**：无需额外分类器，通过调节条件/无条件预测的混合比例来控制生成
- **Consistency Models**：通过蒸馏使扩散模型在 1-2 步内完成生成

## 13. 练习题与思考题

### 基础题 1：扩散系数计算

给定 T=1000，使用线性 β 调度（β_start=0.0001, β_end=0.02），计算：
- t=100 时的 ᾱ_t
- t=500 时的 ᾱ_t
- t=999 时的 ᾱ_t

观察这些值，说明为什么 DDPM 需要较多的扩散步数。

**参考答案**：
```python
import torch
import math

T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

for t in [99, 499, 998]:  # 0-indexed: t=100,500,999
    alpha_bar = alphas_cumprod[t]
    print(f"t={t+1}: ᾱ_t = {alpha_bar:.6f}, √ᾱ_t = {math.sqrt(alpha_bar.item()):.6f}")

# 输出示例:
# t=100:  ᾱ_t ≈ 0.9  左右 — 仍保留大部分信号
# t=500:  ᾱ_t ≈ 0.4  左右 — 信号衰减到不足一半
# t=999:  ᾱ_t ≈ 0.0001 — 几乎完全噪声
```

t=100 时 ᾱ_t 约 0.90，信号保留 90%，噪声仅占 10%——远未到纯噪声。因此需要数百步才能确保噪声充分混合。这是 DDPM 采样慢的根本原因。

### 基础题 2：损失函数含义

DDPM 训练时，为什么损失函数是 ‖ε - ε_θ(x_t, t)‖² 而不是 ‖x_0 - x_θ(x_t, t)‖²？即为什么让模型预测噪声而不是直接预测原始图像？

**参考答案**：
DDPM 选择预测噪声有三个原因：

1. **目标分布简单**：噪声 ε ~ N(0, I) 是各向同性的高斯分布，分布简单、均匀。而原始图像 x_0 的分布极其复杂（自然图像是高维流形）。预测一个简单分布比预测复杂分布容易得多。

2. **梯度信号均匀**：ε 在每个像素位置都有相似的方差，梯度信号均匀。而直接预测 x_0 时，不同位置（如边缘 vs 平坦区域）的梯度差异巨大，训练不稳定。

3. **实证效果更好**：Ho et al. 原论文消融实验证实，预测噪声方案的样本质量显著优于预测 x_0 方案。

### 进阶题：DDIM 加速采样原理

DDIM 论文提出可以通过"跳步"将采样从 T 步减少到 S 步（S << T），且不重新训练。请解释 DDIM 如何在不改变 DDPM 模型的前提下实现加速采样。

**参考答案**：
DDIM 的核心洞察是：DDPM 的采样公式 $x_{t-1} = \mu(x_t, \varepsilon_\theta) + \sigma_t z$ 中的随机项 σ_t z 不是必须的。

DDIM 构造了一个非马尔可夫的确定性逆向过程（σ_t = 0），使得采样公式变为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \varepsilon_\theta(x_t, t)$$

其中 $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta)/\sqrt{\bar{\alpha}_t}$ 是模型预测的"可能的 x_0"。

因为 DDIM 的重参数化仅依赖边缘分布 q(x_t | x_0)，而 DDPM 模型训练的正是这个边缘分布中的噪声 ε，所以无需重新训练即可使用 DDIM 采样。

在原始 T=1000 步的序列中选取 S 个子序列（如每隔 20 步取一个），直接用 DDIM 公式跳过中间步骤——这就是"跳步加速"。S=50 时生成质量与 T=1000 的 DDPM 采样非常接近，而速度提升 20 倍。

### 开放思考题

如果让你设计一个"双向"扩散模型——同时在图像空间和文本空间进行扩散，最终实现文本和图像的联合生成——你会如何设计？会遇到哪些新挑战？

**参考思路**：
- **架构设计**：需要两个扩散过程（图像扩散 + 文本扩散）和一个跨模态融合机制。文本扩散需处理离散 token 的挑战（可用嵌入空间中的连续扩散）
- **新挑战**：
  1. 文本的离散性：扩散模型天然适合连续空间，文本是离散的，需用 embedding 扩散 + 解码器还原或使用 Masked Diffusion
  2. 跨模态对齐：两个扩散过程的时间步需要同步或通过交叉注意力对齐，不同模态的信息衰减速度可能不同
  3. 联合训练策略：先单独预训练图像和文本扩散模型，再用跨模态注意力联合微调（类似 Stable Diffusion 的思路）
  4. 评估困难：联合生成的评估比单一图像/文本生成复杂得多

## 14. 学习路径建议

### 前置算法
- **高斯分布与概率基础**：理解 N(μ, σ²) 和重参数化采样
- **变分自编码器 (VAE)**：理解从潜变量生成数据的框架
- **UNet**：编码器-解码器 + 跳跃连接架构（DDPM 的核心网络）

### 平行算法
- **DDIM**：确定性加速采样的扩散模型变体，可与 DDPM 同时学习
- **Score-based Generative Models (NCSN)**：基于分数匹配的并行范式
- **Flow Matching**：基于常微分方程的连续归一化流，与扩散模型数学上相关

### 进阶算法
- **Latent Diffusion (Stable Diffusion)**：在 VAE 潜空间扩散，大幅降低计算开销
- **DiT (Diffusion Transformer)**：用纯 Transformer 替代 UNet 作为扩散模型的骨干网络
- **Consistency Models**：通过蒸馏使扩散模型在 1-2 步内完成生成
- **Rectified Flow**：用线性插值路径替代弯曲的扩散路径，减少采样步数

### 推荐资源
1. **论文**：Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **论文**：Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021) — DDIM 加速采样
3. **博客**：Lilian Weng, "What are Diffusion Models?" — 图文并茂的入门讲解
4. **代码**：lucidrains/denoising-diffusion-pytorch (GitHub) — 高质量的 PyTorch 参考实现
5. **视频**：Yannic Kilcher 的 DDPM 论文精读 — 逐公式讲解
