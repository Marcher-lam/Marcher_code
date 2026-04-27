# VQ-VAE（向量量化变分自编码器）学习文档

> 来源线索：本节内容根据原书中关于"VQ-VAE 图像编码与解码"（第13章 13.1-13.2节）的相关章节整理、扩展与教学化改写。

> 将连续图像压缩为离散编码，通过码本向量量化实现高效的有损图像重建。

## 1. 算法基础认知

**一句话定义**：VQ-VAE（Vector Quantized Variational AutoEncoder）是一种将连续潜在表示量化为离散编码的自编码器变体。它使用一个可学习的码本（codebook），将编码器输出的每个连续向量映射为码本中距离最近的离散嵌入向量，再由解码器重建图像。

**直觉类比**：想象你要描述一幅画出差给远方的朋友——你不能直接把整幅画邮寄过去，而是要把画的内容用语言描述成文字（离散符号），对方根据文字再画出类似的画。VQ-VAE 就是把图像"翻译"成离散的 codebook 索引，再根据索引"翻译"回图像。

**历史背景**：2017 年，DeepMind 的 van den Oord 等人提出了 VQ-VAE，这是将向量量化（Vector Quantization）引入生成模型的开创性工作。VQ-VAE 的核心动机是模仿 NLP 中"文本 tokenizer -> Embedding -> 模型处理"的范式，将其移植到图像领域：把图像切分成 Patch，每个 Patch 离散化为一个 token，这样图像就可以像文本一样用离散的序列来表示。后续的 VQ-VAE-2 和 VQ-GAN 进一步提升了生成质量，DALL-E 和 Stable Diffusion 等里程碑模型也都借鉴了离散潜在空间的思想。

**算法定位**：深度学习 / 生成模型 / 离散表示学习。属于自编码器（AE）的扩展变体，结合了向量量化（VQ）技术和直通梯度估计（Straight-Through Estimator）。

**前置知识**：
- 自编码器（Autoencoder）的基本结构：编码器-解码器
- 卷积神经网络（CNN）基础：Conv2d、转置卷积
- PyTorch 张量操作：reshape、argmin、einsum
- 损失函数原理：MSE Loss
- 梯度反向传播机制

## 2. 核心原理

### 核心思想

传统自编码器将图像编码为连续的潜在向量 z，连续表示虽然能保留细节，但不利于模型学习高层次的抽象特征。VQ-VAE 的核心创新是**强制潜在表示为离散形式**：编码器输出不是直接送入解码器，而是先在预定义的码本中查找最近邻，用量化后的向量替换原始编码，形成一个"信息瓶颈"。

这个设计的巧妙之处在于：
- **模仿 NLP 处理范式**：文本通过 tokenizer 变成离散 ID，再通过 Embedding 变成向量。图像也应是离散的（像素值有限），只是数量太大才被视为连续。VQ-VAE 将图像编码成离散 token，使得图像可以像文本一样被序列模型处理。
- **压缩 + 抽象**：编码器将原始像素空间的"大图"压缩为隐空间的"小特征图"，每个位置对应一个 D 维向量，再量化为离散编码。这种压缩保留了关键结构信息，丢弃了冗余细节。

### 工作流程

1. **编码（Encode）**：输入图像 $x$ 经过 CNN 编码器 $E$，得到特征图 $z_e = E(x)$，尺寸为 $h \times w \times D$，即 $h \times w$ 个 D 维向量
2. **向量量化（Vector Quantization）**：对于 $z_e$ 中的每一个 D 维向量，在码本 $E_{\text{codebook}} \in \mathbb{R}^{K \times D}$ 中找最近邻（按 L2 距离），用最近邻的嵌入向量替换之，得到 $z_q$
3. **解码（Decode）**：量化后的 $z_q$ 送入解码器 $G$，输出重建图像 $\hat{x} = G(z_q)$
4. **训练目标**：最小化重构损失 + 码本损失 + 承诺损失

### 关键概念解释

- **码本（Codebook）**：一个 $K \times D$ 的可学习参数矩阵。K 是码本中嵌入向量的个数（如 512、1024、4096），D 是每个嵌入向量的维度（如 256）。类似于 NLP 中的词嵌入矩阵。
- **向量量化（VQ）**：将连续向量替换为码本中最近邻离散向量的操作。本质上是 K-means 聚类的一种在线变体。
- **Straight-Through Gradient（直通梯度）**：argmin 操作不可微，VQ-VAE 使用"直通估计器"——前向传播用量化值 $z_q$，反向传播直接将 $z_q$ 的梯度复制给 $z_e$，从而允许梯度流回编码器。
- **Stop Gradient（sg 算子）**：阻止梯度回传的操作。在 VQ-VAE 中，码本损失对编码器使用 sg，承诺损失对码本使用 sg，实现编码器和码本的解耦优化。

### 直观架构图

```
输入图像 x
    │
    ▼
┌──────────┐
│  Encoder  │  CNN堆叠，逐步下采样
│   (CNN)   │  输出: z_e (h×w×D)
└──────────┘
    │
    ▼  z_e 中每个 D 维向量
┌──────────────────────┐
│  Vector Quantization  │
│  z_q[i] = argmin     │
│  ‖z_e[i] - e_k‖²    │  码本 K×D
│  (最近邻搜索)         │  e_0: [...]
│                      │  e_1: [...]
│  straight-through    │  ...
│  z_q = z_e +         │  e_K-1: [...]
│        (z_q - z_e)   │
│        .detach()     │
└──────────────────────┘
    │
    ▼  z_q (h×w×D)
┌──────────┐
│  Decoder  │  CNN + 上采样
│   (CNN)   │  输出: x̂ (重建图像)
└──────────┘
    │
    ▼
重建图像 x̂
```

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 输入图像 | $(B, C, H, W)$ |
| $\hat{x}$ | 重建图像 | $(B, C, H, W)$ |
| $z_e$ | 编码器输出的连续特征图 | $(B, D, h, w)$ |
| $z_q$ | 量化后的离散特征图 | $(B, D, h, w)$ |
| $E_{\text{cb}}$ | 码本（嵌入表） | $(K, D)$ |
| $e_k$ | 码本中第 k 个嵌入向量 | $(D,)$ |
| $K$ | 码本大小（嵌入向量总数） | 标量 |
| $D$ | 嵌入维度 | 标量 |
| $\beta$ | 承诺损失权重（通常 0.25~2.0） | 标量 |
| $\text{sg}[\cdot]$ | stop gradient 算子 | — |

### 3.1 向量量化过程

编码器输出的特征图 $z_e$ 有 $N = h \times w$ 个 D 维向量 $\{z_e^{(1)}, z_e^{(2)}, ..., z_e^{(N)}\}$。对于每个向量，查找码本中距离最近的嵌入向量：

$$k_i = \underset{k \in \{1,...,K\}}{\arg\min} \|z_e^{(i)} - e_k\|_2^2$$

L2 距离展开（避免直接计算范数的数值优化形式）：

$$\|z_e^{(i)} - e_k\|_2^2 = \|z_e^{(i)}\|_2^2 + \|e_k\|_2^2 - 2 \langle z_e^{(i)}, e_k \rangle$$

在实际代码中，一次计算所有距离矩阵 $d \in \mathbb{R}^{N \times K}$：

$$d = \text{sum}(z_{\text{flat}}^2, \text{dim=1, keepdim=True}) + \text{sum}(E_{\text{cb}}^2, \text{dim=1}) - 2 \cdot z_{\text{flat}} \cdot E_{\text{cb}}^T$$

量化后的特征向量为：

$$z_q^{(i)} = e_{k_i}$$

### 3.2 直通梯度估计（Straight-Through Estimator）

argmin 操作的梯度为 0（除了不可微点），导致梯度无法流回编码器。VQ-VAE 使用直通梯度技巧：

**前向传播**：使用量化值 $z_q$
**反向传播**：梯度从前一层直接"穿过"量化操作，复制到 $z_e$

实现方式：

$$z_q^{\text{output}} = z_e + (z_q - z_e).\text{detach}()$$

- `.detach()` 使得 $(z_q - z_e)$ 在反向传播时梯度为 0
- 前向值等于 $z_q$
- 反向梯度等于 $\frac{\partial \mathcal{L}}{\partial z_q}$（直接传给 $z_e$）

### 3.3 损失函数推导

VQ-VAE 的总损失由三部分构成：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}}$$

**(1) 重构损失（Reconstruction Loss）**

$$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_2^2 \quad \text{或} \quad \text{MSE}(x, \hat{x})$$

用于优化编码器和解码器的参数，确保重建图像尽可能接近原图。

**(2) 码本损失（Codebook Loss / Embedding Loss）**

$$\mathcal{L}_{\text{codebook}} = \|\text{sg}[z_e] - z_q\|_2^2$$

- 仅优化码本参数（编码器梯度被 sg 阻断）
- 让码本中的嵌入向量向编码器输出"靠拢"
- 本质上是 EMA（指数移动平均）更新码本的一种等价形式

**(3) 承诺损失（Commitment Loss）**

$$\mathcal{L}_{\text{commit}} = \beta \cdot \|z_e - \text{sg}[z_q]\|_2^2$$

- 仅优化编码器参数（码本梯度被 sg 阻断）
- $\beta$ 控制编码器"承诺"使用码本的强度
- 防止编码器输出在训练中漂移过大，与码本差距持续扩大

### 3.4 为什么需要三部分损失？

- **只用重构损失**：编码器可以学习到任何连续表示，码本完全不会被使用（argmin 无梯度）
- **只用重构+码本**：码本能学习，但编码器不受约束，可能输出范围无限扩大
- **加入承诺损失**：编码器也受到约束，确保输出始终在码本覆盖范围内

三部分各司其职：重构损失优化编解码质量，码本损失训练码本参数，承诺损失约束编码器行为。

## 4. 训练过程讲解

### 4.1 完整训练步骤

**Step 1：前向传播**
1. 输入图像 $x$ 进入编码器，得到 $z_e$
2. $z_e$ 中的每个向量在码本中搜索最近邻，得到 $z_q$ 和对应的索引
3. 使用直通梯度：$z_q^{\text{output}} = z_e + (z_q - z_e).\text{detach}()$
4. $z_q^{\text{output}}$ 进入解码器，得到重建图像 $\hat{x}$

**Step 2：损失计算**
1. 计算重构损失：$\mathcal{L}_{\text{recon}} = \text{MSE}(x, \hat{x})$
2. 计算码本损失：$\mathcal{L}_{\text{codebook}} = \|\text{sg}[z_e] - z_q\|_2^2$
3. 计算承诺损失：$\mathcal{L}_{\text{commit}} = \beta \cdot \|z_e - \text{sg}[z_q]\|_2^2$
4. 总损失：$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}}$

**Step 3：反向传播**
1. 重构损失的梯度通过解码器 -> 直通梯度 -> 编码器
2. 码本损失的梯度仅更新码本参数
3. 承诺损失的梯度仅更新编码器参数

**Step 4：参数更新**
- 使用 Adam/AdamW 优化器更新所有参数

### 4.2 训练关键点

- **码本初始化**：通常用均匀分布 $\mathcal{U}(-1/K, 1/K)$ 初始化，确保初始值在合理范围内
- **EMA 更新（可选）**：也可以用指数移动平均（EMA）替代码本损失来更新码本，收敛更稳定
- **码本重启**：如果某个码字长时间未被使用，可将其重置为当前 batch 中的一个随机 $z_e$ 向量，防止码本坍塌
- **Beta 调度**：承诺损失权重 $\beta$ 一般固定为 0.25，也可在训练初期设较小值，逐步增大

### 4.3 训练伪代码

```
for epoch in range(num_epochs):
    for x in dataloader:
        z_e = encoder(x)                        # (B, D, h, w)
        z_q, indices = quantize(z_e)             # 最近邻搜索 + 直通梯度
        x_hat = decoder(z_q)                     # 重建图像
        
        loss_recon = MSE(x_hat, x)
        loss_codebook = MSE(z_e.detach(), z_q)
        loss_commit = beta * MSE(z_e, z_q.detach())
        loss = loss_recon + loss_codebook + loss_commit
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5. 应用场景

### 5.1 图像压缩与存储
VQ-VAE 将图像编码为一组离散的整数索引（每个位置一个索引），只存储索引序列和码本即可。解码时用索引查表获取嵌入向量，送入解码器重建图像。相比存储整幅图像，离散索引占用空间极小。例如 $28 \times 28$ 的 MNIST 图像，编码后可能只需 $7 \times 7 = 49$ 个整数就可以近似重建。

### 5.2 无条件/条件图像生成
VQ-VAE 生成的离散 token 序列可以作为自回归模型（如 PixelCNN、GPT、Transformer）的训练目标。先训练 VQ-VAE 获得离散 tokenizer，再训练一个自回归模型来逐 token 地生成新的图像。这是 DALL-E、VQ-GAN 等里程碑式的文生图模型的核心思路。

### 5.3 多模态理解
将图像离散化为 token 后，可以在统一的 token 空间中同时处理图像 token 和文本 token，使得图像和文本可以用同一个 Transformer 进行联合建模（如用于图文检索、视觉问答等任务）。

### 5.4 异常检测
正常图像可以通过 VQ-VAE 较好地重建，而异常图像（训练分布外）在量化时会被映射到不匹配的码字，导致重建误差显著增大。通过监测重构误差可以检测异常。

### 5.5 图像编辑与操作
修改某些位置的离散 token 索引可以局部编辑图像内容（如书中示例：修改部分 token 值，生成图像发生变化），实现可控的图像编辑。

## 6. 优缺点分析

### 优点

| 优点 | 详细说明 |
|------|----------|
| **离散表示** | 将连续图像转换为离散 token 序列，便于与 NLP 方法（自回归、Transformer）结合，统一多模态处理范式 |
| **高效压缩** | 编码后特征图尺寸远小于原图（如 $28 \times 28$ -> $7 \times 7$），结合离散化实现极高压缩比 |
| **生成质量** | 离散潜在空间有助于模型学习高层次语义信息，后续 VQ-GAN 等变体在图像生成上达到了极高质量 |
| **可控生成** | 修改离散 token 可编辑生成结果，为可控图像生成提供了直观的接口 |
| **避免后验坍塌** | 相比 VAE，离散化制造的信息瓶颈使模型难以"偷懒"，迫使编码器提取有意义的特征 |
| **通用性** | 不仅适用于图像，还可扩展到音频、视频等多种模态 |

### 缺点

| 缺点 | 详细说明 |
|------|----------|
| **码本坍塌** | 训练中大量码字可能不被使用（变为"死"码字），需要额外的技术手段来预防 |
| **训练不稳定** | 三部分损失函数需要仔细调参，承诺损失权重 $\beta$ 对结果影响大 |
| **梯度近似** | 直通梯度估计本质上是梯度近似，可能导致编码器学习到次优表示 |
| **码本大小限制** | 码本容量限制了表达能力，太小则重建质量差，太大则搜索效率和利用率问题 |
| **计算开销** | 对整个特征图做 KNN 搜索带来额外计算，特别是大码本时 |
| **模式选择性** | 对某些模式（如未来种细节图案）可能只用少数几个码字，缺乏多样性 |

## 7. 调库实现

下面使用 PyTorch 实现一个完整的 VQ-VAE，并在 MNIST 数据集上进行训练和重建。

```python
"""
VQ-VAE 完整调库实现 (PyTorch)
数据集: MNIST 手写数字
目标: 编码->量化->解码重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ======================== 设备配置 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================== 超参数 ========================
BATCH_SIZE = 128
LATENT_DIM = 64          # 潜在空间每个向量的维度 D
NUM_EMBEDDINGS = 512     # 码本大小 K
COMMITMENT_COST = 0.25   # 承诺损失权重 beta
LR = 2e-4
NUM_EPOCHS = 20          # 为快速验证可设小值
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1

# ======================== 数据集加载 ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    # MNIST 像素范围 [0,1]，不需要额外归一化
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# ======================== Encoder 编码器 ========================
class Encoder(nn.Module):
    """
    将 28x28 的灰度图像编码为 7x7 的特征图，每个位置为 LATENT_DIM 维向量。
    架构: Conv2d + ReLU + Conv2d + ReLU + Conv2d (逐步下采样)
    """
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        # 28x28 -> 14x14
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        # 14x14 -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # 7x7 -> 7x7 (保持尺寸，扩展通道)
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)          # 输出: (B, latent_dim, 7, 7)
        return x

# ======================== Decoder 解码器 ========================
class Decoder(nn.Module):
    """
    将 7x7 的特征图解码回 28x28 的灰度图像。
    架构: ConvTranspose2d + ReLU + ConvTranspose2d + Conv2d + Sigmoid
    """
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        # 7x7 -> 7x7 (保持尺寸)
        self.conv1 = nn.Conv2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        # 7x7 -> 14x14
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # 14x14 -> 28x28
        self.conv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))   # 输出范围 [0,1]
        return x

# ======================== VectorQuantizer 向量量化器 ========================
class VectorQuantizer(nn.Module):
    """
    核心模块: 将连续特征向量量化为码本中最近邻的离散向量。
    支持直通梯度估计 (Straight-Through Estimator)。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 码本: (K, D) 的可学习参数
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 均匀分布初始化
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(self, z):
        """
        参数:
            z: (B, D, H, W) 编码器输出
        返回:
            z_q: (B, D, H, W) 量化后输出，梯度可通过直通估计器传递
            loss: 标量，码本损失 + 承诺损失
            indices: (B, H, W) 每个位置选择的码字索引
        """
        B, D, H, W = z.shape

        # 重排: (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # ---- 计算距离矩阵 ----
        # ‖z_i‖² (N,) 和 ‖e_j‖² (K,) 展开求和
        z_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)     # (N, 1)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)    # (K,)
        # d_ij = ‖z_i‖² + ‖e_j‖² - 2·z_i·e_j^T
        distances = (
            z_sq
            + e_sq.unsqueeze(0)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # (N, K)

        # ---- 找到最近邻 ----
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        z_q_flat = self.embedding(encoding_indices)         # (N, D)

        # ---- 直通梯度估计 ----
        # 前向值 = z_q_flat, 反向梯度直接传给 z_flat
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # 恢复形状
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        indices = encoding_indices.view(B, H, W)

        # ---- 损失计算 ----
        # 码本损失: 让嵌入向量靠近编码器输出 (仅更新码本)
        codebook_loss = F.mse_loss(z_q.detach(), z_flat)
        # 承诺损失: 让编码器输出靠近嵌入向量 (仅更新编码器)
        commitment_loss = F.mse_loss(z_q, z_flat.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        return z_q, loss, indices

# ======================== VQ-VAE 完整模型 ========================
class VQVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64,
                 num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):
        """
        返回:
            x_recon: 重建图像
            vq_loss: 量化损失
            indices: 编码索引
        """
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    def encode(self, x):
        """获取离散编码索引"""
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

    def decode_from_indices(self, indices):
        """从索引直接重建图像"""
        B, H, W = indices.shape
        z_q_flat = self.vq.embedding(indices.view(-1))
        z_q = z_q_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.decoder(z_q)

# ======================== 训练模型 ========================
model = VQVAE(
    in_channels=IMAGE_CHANNELS,
    latent_dim=LATENT_DIM,
    num_embeddings=NUM_EMBEDDINGS,
    commitment_cost=COMMITMENT_COST
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print("开始训练...\n")

train_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_vq_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_recon, vq_loss, _ = model(data)

        recon_loss = F.mse_loss(x_recon, data)
        total_loss = recon_loss + vq_loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_vq_loss += vq_loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Recon: {epoch_recon_loss / len(train_loader):.4f} | "
              f"VQ: {epoch_vq_loss / len(train_loader):.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

print("\n训练完成!")

# ======================== 评估与可视化 ========================
model.eval()

# 取一批测试数据
test_iter = iter(test_loader)
test_images, _ = next(test_iter)
test_images = test_images[:8].to(device)

with torch.no_grad():
    x_recon, vq_loss, indices = model(test_images)
    recon_loss = F.mse_loss(x_recon, test_images)

print(f"\n测试集重构 MSE: {recon_loss.item():.4f}")

# 可视化: 原始 vs 重建
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i in range(8):
    # 原始图像
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap="gray")
    axes[0, i].set_title(f"Original {i+1}" if i == 0 else f"{i+1}")
    axes[0, i].axis("off")
    # 重建图像
    axes[1, i].imshow(x_recon[i].cpu().squeeze(), cmap="gray")
    axes[1, i].set_title(f"Reconstructed {i+1}" if i == 0 else f"{i+1}")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
plt.suptitle("VQ-VAE: MNIST Original vs Reconstructed Images", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("vqvae_reconstruction.png", dpi=100, bbox_inches="tight")
plt.show()

# 可视化: 训练损失曲线
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-o", markersize=4)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("VQ-VAE Training Loss Curve", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vqvae_loss_curve.png", dpi=100, bbox_inches="tight")
plt.show()

# 可视化: 码本利用率分析
with torch.no_grad():
    # 统计每个码字被使用的频率
    all_indices = []
    for data, _ in test_loader:
        data = data.to(device)
        _, _, idx = model(data)
        all_indices.append(idx.cpu())
    all_indices = torch.cat(all_indices, dim=0).flatten()  # 所有位置的索引

    usage_counts = torch.bincount(all_indices, minlength=NUM_EMBEDDINGS)
    used_codewords = (usage_counts > 0).sum().item()
    usage_rate = used_codewords / NUM_EMBEDDINGS * 100

    # 绘制码字使用分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(NUM_EMBEDDINGS), usage_counts.numpy(), width=1.0)
    axes[0].set_xlabel("Codebook Index", fontsize=12)
    axes[0].set_ylabel("Usage Count", fontsize=12)
    axes[0].set_title(f"Codebook Usage Distribution\n"
                      f"({used_codewords}/{NUM_EMBEDDINGS} used, "
                      f"{usage_rate:.1f}% utilization)", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Top-20 码字使用量
    top_k = 20
    sorted_counts, sorted_idx = torch.sort(usage_counts, descending=True)
    axes[1].bar(range(top_k), sorted_counts[:top_k].numpy(), color="steelblue")
    axes[1].set_xlabel("Rank", fontsize=12)
    axes[1].set_ylabel("Usage Count", fontsize=12)
    axes[1].set_title(f"Top-{top_k} Most Used Codewords", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vqvae_codebook_usage.png", dpi=100, bbox_inches="tight")
    plt.show()

# 可视化: 离散 token 修改实验
with torch.no_grad():
    single_img = test_images[0:1]  # (1, 1, 28, 28)
    _, _, orig_indices = model(single_img)
    print(f"\n原始编码索引 (1, 7, 7):")
    print(orig_indices.squeeze().cpu().numpy())

    # 修改部分 token（将左上角的几个索引改为随机值）
    modified_indices = orig_indices.clone()
    modified_indices[0, 0:2, 0:2] = torch.randint(0, NUM_EMBEDDINGS, (2, 2))

    x_recon_orig = model.decode_from_indices(orig_indices)
    x_recon_mod = model.decode_from_indices(modified_indices)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(single_img.cpu().squeeze(), cmap="gray")
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(x_recon_orig.cpu().squeeze(), cmap="gray")
    axes[1].set_title("Reconstructed\n(from original tokens)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(x_recon_mod.cpu().squeeze(), cmap="gray")
    axes[2].set_title("Reconstructed\n(with modified tokens)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("VQ-VAE: Token Modification Experiment", fontsize=14)
    plt.tight_layout()
    plt.savefig("vqvae_token_modification.png", dpi=100, bbox_inches="tight")
    plt.show()

print("\n所有可视化已保存!")
```

## 8. 手工代码实现

下面从零实现 VQ-VAE 的所有核心组件，包括编码器、向量量化器（手工 KNN 搜索 + 直通梯度）、解码器，以及完整的训练测试代码。

```python
"""
VQ-VAE 手工代码实现 (从零搭建)
所有组件均用基础 PyTorch 操作实现，无高层 API 封装。
"""

import torch
import torch.nn.functional as F
import numpy as np

# ======================== 手工 Encoder ========================
class EncoderScratch:
    """
    从零实现的卷积编码器。
    不使用 nn.Module，直接管理权重和偏置。
    结构: Conv(1->32, s=2) + ReLU + Conv(32->64, s=2) + ReLU + Conv(64->64, s=1)
    """
    def __init__(self, in_channels=1, latent_dim=64):
        # 初始化为 Kaiming 分布
        self.conv1_w = torch.randn(32, in_channels, 4, 4) * np.sqrt(2.0 / (in_channels * 4 * 4))
        self.conv1_b = torch.zeros(32)
        self.conv2_w = torch.randn(64, 32, 4, 4) * np.sqrt(2.0 / (32 * 4 * 4))
        self.conv2_b = torch.zeros(64)
        self.conv3_w = torch.randn(latent_dim, 64, 3, 3) * np.sqrt(2.0 / (64 * 3 * 3))
        self.conv3_b = torch.zeros(latent_dim)

        # 标记哪些参数需要梯度
        self.params = [self.conv1_w, self.conv1_b,
                       self.conv2_w, self.conv2_b,
                       self.conv3_w, self.conv3_b]
        for p in self.params:
            p.requires_grad = True

    def forward(self, x):
        """
        x: (B, C, H, W) 输入图像
        返回: (B, D, h, w) 编码特征
        """
        # Conv1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        x = F.conv2d(x, self.conv1_w, self.conv1_b, stride=2, padding=1)
        x = F.relu(x)
        # Conv2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        x = F.conv2d(x, self.conv2_w, self.conv2_b, stride=2, padding=1)
        x = F.relu(x)
        # Conv3: (B, 64, 7, 7) -> (B, D, 7, 7) 保持尺寸
        x = F.conv2d(x, self.conv3_w, self.conv3_b, stride=1, padding=1)
        return x


# ======================== 手工 VectorQuantizer ========================
class VectorQuantizerScratch:
    """
    从零实现的向量量化器。
    核心操作:
    1. 计算每个 z_e 向量到所有码字的 L2 距离
    2. 用 argmin 找最近邻
    3. 直通梯度估计: z_q = z_e + (z_q_raw - z_e).detach()
    4. 计算码本损失和承诺损失
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 码本: (K, D)，均匀初始化
        limit = 1.0 / num_embeddings
        self.codebook = torch.empty(num_embeddings, embedding_dim).uniform_(-limit, limit)
        self.codebook.requires_grad = True
        self.params = [self.codebook]

    def forward(self, z_e):
        """
        z_e: (B, D, H, W)
        返回: z_q, vq_loss, indices
        """
        B, D, H, W = z_e.shape

        # 展平: (B, D, H, W) -> (B, H*W, D) -> (B*H*W, D)
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (N, D)
        N = z_flat.shape[0]
        K = self.num_embeddings

        # ---- 1. 手工计算 L2 距离矩阵 ----
        # d[i][j] = sum_k (z_i[k] - e_j[k])^2
        #          = ‖z_i‖² + ‖e_j‖² - 2 * z_i @ e_j^T
        z_norm_sq = (z_flat * z_flat).sum(dim=1, keepdim=True)     # (N, 1)
        e_norm_sq = (self.codebook * self.codebook).sum(dim=1)     # (K,)
        dot_product = z_flat @ self.codebook.t()                    # (N, K)
        distances = z_norm_sq + e_norm_sq.unsqueeze(0) - 2 * dot_product

        # ---- 2. 找到最近邻索引 ----
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)

        # ---- 3. 查表获取量化值 ----
        z_q_raw = self.codebook[encoding_indices]  # (N, D)

        # ---- 4. 直通梯度估计 ----
        # z_q = z_flat + (z_q_raw - z_flat).detach()
        # 前向: z_q_raw (量化后的值)
        # 反向: 梯度直接传给 z_flat (因为 detach 阻断了 z_q_raw-z_flat 的梯度)
        z_q_flat = z_flat + (z_q_raw - z_flat).detach()

        # ---- 5. 损失计算 ----
        # 码本损失: 码本靠近编码器输出 (stop-gradient on z_e)
        codebook_loss = ((z_flat.detach() - z_q_raw) ** 2).mean()
        # 承诺损失: 编码器承诺使用码本 (stop-gradient on z_q)
        commitment_loss = ((z_q_raw.detach() - z_flat) ** 2).mean()
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # ---- 6. 恢复形状 ----
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(B, H, W)

        return z_q, vq_loss, indices

    def get_codebook_entry(self, indices):
        """通过索引获取码字"""
        return self.codebook[indices]


# ======================== 手工 Decoder ========================
class DecoderScratch:
    """
    从零实现的转置卷积解码器。
    结构: Conv(64->64, s=1) + ReLU + ConvTranspose(64->32, s=2) + ReLU + ConvTranspose(32->1, s=2) + Sigmoid
    """
    def __init__(self, latent_dim=64, out_channels=1):
        # Conv 1x1 保持尺寸
        self.conv1_w = torch.randn(64, latent_dim, 3, 3) * np.sqrt(2.0 / (latent_dim * 3 * 3))
        self.conv1_b = torch.zeros(64)

        # ConvTranspose: 7x7 -> 14x14
        self.conv2_w = torch.randn(64, 32, 4, 4) * np.sqrt(2.0 / (64 * 4 * 4))
        self.conv2_b = torch.zeros(32)

        # ConvTranspose: 14x14 -> 28x28
        self.conv3_w = torch.randn(32, out_channels, 4, 4) * np.sqrt(2.0 / (32 * 4 * 4))
        self.conv3_b = torch.zeros(out_channels)

        self.params = [self.conv1_w, self.conv1_b,
                       self.conv2_w, self.conv2_b,
                       self.conv3_w, self.conv3_b]
        for p in self.params:
            p.requires_grad = True

    def forward(self, z_q):
        """
        z_q: (B, D, H, W)
        返回: (B, C, H_out, W_out) 重建图像
        """
        x = F.conv2d(z_q, self.conv1_w, self.conv1_b, stride=1, padding=1)
        x = F.relu(x)
        x = F.conv_transpose2d(x, self.conv2_w, self.conv2_b, stride=2, padding=1)
        x = F.relu(x)
        x = F.conv_transpose2d(x, self.conv3_w, self.conv3_b, stride=2, padding=1)
        x = torch.sigmoid(x)
        return x


# ======================== 完整 VQ-VAE (手工版) ========================
class VQVAEScratch:
    """组装编码器、量化器和解码器"""
    def __init__(self, in_channels=1, latent_dim=64,
                 num_embeddings=512, commitment_cost=0.25):
        self.encoder = EncoderScratch(in_channels, latent_dim)
        self.vq = VectorQuantizerScratch(num_embeddings, latent_dim, commitment_cost)
        self.decoder = DecoderScratch(latent_dim, in_channels)

    @property
    def params(self):
        """返回所有可训练参数列表 (给优化器用)"""
        all_params = []
        all_params.extend(self.encoder.params)
        all_params.extend(self.vq.params)
        all_params.extend(self.decoder.params)
        return all_params

    def forward(self, x):
        z_e = self.encoder.forward(x)
        z_q, vq_loss, indices = self.vq.forward(z_e)
        x_recon = self.decoder.forward(z_q)
        return x_recon, vq_loss, indices


# ======================== 测试代码 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("VQ-VAE 手工实现测试")
    print("=" * 60)

    # 创建模型
    model = VQVAEScratch(
        in_channels=1, latent_dim=64,
        num_embeddings=512, commitment_cost=0.25
    )
    total_params = sum(p.numel() for p in model.params)
    print(f"模型总参数量: {total_params:,}")

    # 模拟输入
    batch_size = 4
    dummy_input = torch.rand(batch_size, 1, 28, 28)  # MNIST 格式

    # 前向传播测试
    print(f"\n输入形状: {dummy_input.shape}")

    # 手动设置 requires_grad
    dummy_input.requires_grad = True
    x_recon, vq_loss, indices = model.forward(dummy_input)

    print(f"重建输出形状: {x_recon.shape}")
    print(f"编码索引形状: {indices.shape}")
    print(f"量化损失: {vq_loss.item():.6f}")
    print(f"重建输出范围: [{x_recon.min().item():.4f}, {x_recon.max().item():.4f}]")

    # 梯度回传测试
    recon_loss = F.mse_loss(x_recon, dummy_input)
    total_loss = recon_loss + vq_loss
    total_loss.backward()

    has_grad = [p.grad is not None for p in model.params]
    print(f"\n梯度回传测试: {sum(has_grad)}/{len(has_grad)} 个参数有梯度")
    print(f"重构损失: {recon_loss.item():.6f}")
    print(f"总损失: {total_loss.item():.6f}")

    # 验证直通梯度: z_q 到 z_e 的梯度应该非零
    print(f"\n编码器 Conv1 权重梯度范数: {model.encoder.conv1_w.grad.norm().item():.6f}")
    print(f"码本梯度范数: {model.vq.codebook.grad.norm().item():.6f}")
    print(f"解码器 Conv1 权重梯度范数: {model.decoder.conv1_w.grad.norm().item():.6f}")

    # 验证梯度不为零 — 确认直通估计器工作正常
    assert model.encoder.conv1_w.grad.norm() > 0, \
        "ERROR: Encoder received no gradient! Straight-through estimator failed."
    assert model.vq.codebook.grad.norm() > 0, \
        "ERROR: Codebook received no gradient!"

    print("\n" + "=" * 60)
    print("所有测试通过! VQ-VAE 手工实现验证成功。")
    print("=" * 60)
```

## 9. 可视化与结果理解

可视化代码已整合在第 7 节的调库实现中。本节对可视化结果进行解读。

### 9.1 重建效果对比

原始图像和 VQ-VAE 重建图像的并排对比展示了模型的学习质量。对于 MNIST 手写数字：
- 结构完整的数字（如 0、1、7）通常能被几乎完美重建
- 细节复杂的数字（如 8）可能在边缘处略有模糊
- 整体笔画走向和数字识别特征被完好保留

### 9.2 训练损失曲线

训练损失曲线通常呈现以下趋势：
- 前几个 epoch 损失快速下降（模型迅速学习基本重建能力）
- 之后进入缓慢下降的"精调"阶段
- 如果损失在中期反弹，通常是承诺损失权重 $\beta$ 设置不当导致编码器和码本之间出现了拉扯

### 9.3 码本利用率分析

码本使用分布柱状图至关重要，因为它直接反映了"码本坍塌"的严重程度：
- **理想情况**：大多数码字被均衡使用，分布均匀
- **码本坍塌**：只有少数码字（如 10-50 个）被频繁使用，其余永远是"死"码字
- 利用率低于 30% 时通常需要调整策略（如减小码本、增加 $\beta$、使用 EMA 更新）

### 9.4 Token 修改实验

修改部分 token 后重建的图像展示了离散潜在空间的一个有趣特性：
- 修改左上角 token（对应图像左上区域）会导致重建图像左上角失真
- 修改少量的 token 对整体图像影响较小（模型的局部性）
- 修改大量 token 则图像完全破坏

这证实了 VQ-VAE 的离散 token 确实编码了空间的局部信息，而非全局分散表示。这也是为什么 VQ-VAE 的 token 可以像文本 token 一样被序列模型逐位置地建模。

## 10. 模型评估

### 10.1 定量评估指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **MSE / RMSE** | 逐像素的重构误差 | $\text{MSE} = \frac{1}{N}\sum (x_i - \hat{x}_i)^2$ |
| **PSNR** | 峰值信噪比，越高越好 | $10 \cdot \log_{10}(\frac{\text{MAX}^2}{\text{MSE}})$ |
| **SSIM** | 结构相似性，越接近 1 越好 | 考虑亮度、对比度、结构三方面 |
| **码本利用率** | 活跃码字数 / 总码字数 | 反映码本是否坍塌 |
| **Perplexity** | 码字使用分布的熵，越高越均匀 | $\exp(-\sum p_k \log p_k)$ |

### 10.2 定性评估

- **重建保真度**：重建图像是否保留了原图的核心结构和特征
- **泛化能力**：测试集上与训练集上的重建质量差距是否合理
- **生成多样性**：修改 token 后生成的图像是否具有合理的变异性

### 10.3 评估代码

```python
# PSNR 计算
def psnr(x, x_hat):
    mse = F.mse_loss(x_hat, x)
    return 10 * torch.log10(1.0 / mse)

# 码本利用率
def codebook_utilization(model, dataloader, device):
    all_indices = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _, _, idx = model(data)
            all_indices.append(idx.cpu())
    all_indices = torch.cat(all_indices).flatten()
    usage = torch.bincount(all_indices, minlength=model.vq.num_embeddings)
    active = (usage > 0).sum().item()
    return active / model.vq.num_embeddings

# Perplexity
def codebook_perplexity(model, dataloader, device):
    all_indices = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _, _, idx = model(data)
            all_indices.append(idx.cpu())
    all_indices = torch.cat(all_indices).flatten()
    counts = torch.bincount(all_indices, minlength=model.vq.num_embeddings).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # 只保留活跃码字
    perplexity = torch.exp(-(probs * torch.log(probs)).sum())
    return perplexity.item()
```

## 11. 常见问题与易错点

### 问题 1: 码本坍塌（Codebook Collapse）

- **现象**：训练过程中，码本的利用率持续下降，大部分码字从未被使用（成为"死"码字），只有极少数的码字被重复使用。重建图像质量差，模型退化成了只有少量聚类中心的"退化 VQ-VAE"。
- **原因**：
  1. 直通梯度估计只是近似，编码器的实际更新方向可能使输出远离码本
  2. argmin 操作的赢者通吃特性：好的码字变得更好，差的码字变得永远用不上
  3. 承诺损失权重 $\beta$ 太小，编码器输出漂移不受控
  4. 码本初始化和编码器初始化不匹配
- **解决方案**：
  1. 使用 EMA（指数移动平均）更新码本替代梯度更新
  2. 增加承诺损失权重 $\beta$（尝试 0.5~1.0）
  3. 实施码本重启：定期检测未使用的码字，将其重置为随机编码器输出
  4. 减小码本大小（如从 4096 减到 512 或 256）
  5. 使用 FSQ（有限标量量化）作为替代方案 — 它从根本上不会出现码本坍塌

### 问题 2: 训练不稳定 / 损失震荡

- **现象**：训练损失不是平滑下降，而是出现大幅震荡，甚至偶尔飙升。
- **原因**：
  1. 学习率过大，导致编码器和码本之间的动态平衡被打破
  2. 承诺损失和码本损失的比例不合适，两者互相"拉扯"
  3. Batch 中样本的编码分布差异大
- **解决方案**：
  1. 降低学习率（如 2e-4 -> 1e-4），使用 warm-up
  2. 调整 $\beta$，观察 commitment_loss 和 codebook_loss 的比例，理想比值约为 1:1 到 1:5
  3. 增大 batch size 以获得更稳定的统计量

### 问题 3: 直通梯度导致的梯度失真

- **现象**：虽然损失下降，但重建质量提升缓慢，编码器似乎没有充分学习到特征。
- **原因**：Straight-through estimator 将 $\frac{\partial \mathcal{L}}{\partial z_q}$ 原样复制为 $\frac{\partial \mathcal{L}}{\partial z_e}$，但这只有在量化误差很小时才合理。量化误差大时，梯度方向和实际最优方向不一致。
- **解决方案**：
  1. 增加码本大小以减少量化误差
  2. 在训练初期使用更小的 $\beta$（让码本先适应编码器输出），后期增大
  3. 使用 Gumbel-Softmax 等可微的量化替代方案

### 问题 4: 解码器过于强大

- **现象**：码本利用率极低，但重建质量尚可。说明解码器从极少量码字中也能重建出图像，码本的"信息瓶颈"效果被削弱。
- **原因**：解码器容量过大，学会了用少量码字的组合来逼近所有图像，而不是依赖多样化的码字。
- **解决方案**：
  1. 减小解码器的容量（减少通道数或层数）
  2. 减小码本大小，让信息瓶颈更"紧"
  3. 在潜在空间增加噪声（如轻微 dropout）

### 问题 5: 编码器的特征图尺寸选择不当

- **现象**：编码后的 grid 尺寸（h, w）太大导致 token 过多、太小导致信息过于压缩重建质量差。
- **原因**：h 和 w 表示空间分辨率的压缩程度，需要在压缩率和重建质量之间权衡。
- **解决方案**：对于 28x28 图像，通常选择 h=w=7（下采样 4 倍）；对于 256x256 图像，通常选择 h=w=16 或 32。

## 12. 学习总结

VQ-VAE 是连接连续视觉世界和离散符号世界的关键技术。它的核心洞察在于：图像虽然以连续像素值存储，但本质上可以也应该被离散化表示——就像文字被 tokenize 为离散 ID 一样。

通过 CNN 编码器将图像压缩为特征图，再通过向量量化将每个位置的连续向量映射为码本中的离散码字，VQ-VAE 实现了一个有信息瓶颈的自编码器。这个"瓶颈"来自三方面：空间压缩（大图->小特征图）、维度压缩（像素->嵌入向量）、离散化（连续->码字索引）。

训练 VQ-VAE 的最大挑战来自其不可微的量化操作。直通梯度估计器提供了一个简洁的解决方案，但也带来了码本坍塌等问题。理解总损失的三部分构成——重构损失、码本损失、承诺损失——以及它们各自优化的参数子集，是掌握 VQ-VAE 训练动态的关键。

VQ-VAE 的影响力远超其初始提出时的预期：它奠定了后续 VQ-GAN、DALL-E、Stable Diffusion 等里程碑式模型的离散潜在空间基础，证明了"离散化"是处理多种数据模态的统一范式。

## 13. 练习题与思考题

### 13.1 概念理解题

**Q1**: 为什么 VQ-VAE 使用离散编码而不是连续编码？离散编码相比连续编码有什么优势？

**答案**：离散编码的优势在于：(1) 可以像 NLP token 一样被自回归模型处理，实现统一的序列建模范式；(2) 离散化创造了更强的信息瓶颈，迫使模型学习高层抽象而非像素级细节；(3) 离散表示更"自然"——图像的像素值本身也是有限的（如 256^3 种），离散表示更贴近数据的本质结构；(4) 离散编码可以直接用整数索引存储，显著降低了存储和传输成本。

**Q2**: VQ-VAE 中的 EMS/detach() 操作起什么作用？

**答案**：`.detach()` 创建了一个共享数据但梯度被切断的张量副本。在 VQ-VAE 中它的核心用途是"选择性阻断梯度"：
- 在码本损失 `MSE(z_e.detach(), z_q)` 中：阻止梯度流向编码器，仅更新码本
- 在承诺损失 `MSE(z_e, z_q.detach())` 中：阻止梯度流向码本，仅更新编码器
- 在直通梯度 `z_e + (z_q - z_e).detach()` 中：让 z_q_z_e 差异不影响反向传播

这实现了编码器和码本的解耦优化。

### 13.2 数学推导题

**Q3**: 证明 L2 距离矩阵可以通过公式 $\text{distances} = \|z_{\text{flat}}\|^2_2 + \|E_{\text{cb}}\|^2_2 - 2 z_{\text{flat}} E_{\text{cb}}^T$ 高效计算。

**答案**：
对于 $z_i \in \mathbb{R}^D$（第 i 个编码向量）和 $e_j \in \mathbb{R}^D$（第 j 个码字）：

$$\|z_i - e_j\|_2^2 = \sum_{d=1}^{D}(z_{i,d} - e_{j,d})^2$$
$$= \sum_{d=1}^{D}(z_{i,d}^2 - 2z_{i,d}e_{j,d} + e_{j,d}^2)$$
$$= \sum_{d=1}^{D}z_{i,d}^2 + \sum_{d=1}^{D}e_{j,d}^2 - 2\sum_{d=1}^{D}z_{i,d}e_{j,d}$$
$$= \|z_i\|_2^2 + \|e_j\|_2^2 - 2 \langle z_i, e_j \rangle$$

在矩阵形式下，$\|z_{\text{flat}}\|^2_2$ 是 $(N,1)$ 向量（每行的 L2 范数平方），$\|E_{\text{cb}}\|^2_2$ 是 $(K,)$ 向量，$z_{\text{flat}} E_{\text{cb}}^T$ 是 $(N,K)$ 的内积矩阵。通过广播机制，这个公式一次计算所有 N×K 个距离，避免了显式双重循环。

**Q4**: 解释为什么在总损失中加入承诺损失是必要的。如果只用重构损失和码本损失会怎样？

**答案**：如果不加承诺损失（$\beta=0$），编码器不受任何约束。由于码本损失中使用了 `z_e.detach()`，编码器完全不感知码本损失。编码器可以通过不断增大输出幅度来降低重构损失，导致与码本的距离持续扩大：$z_e$ 可能跑到码本完全覆盖不到的区域。结果是量化质量越来越差，重建效果下降。

加入承诺损失后，编码器也受到约束——它必须将输出保持在与码本相近的范围内。这形成了一个"相互制约"的平衡：码本努力靠近编码器输出（码本损失），编码器也努力靠近码本（承诺损失）。

### 13.3 代码实践题

**Q5**: 如果码本大小设置得太小（如 K=8），会发生什么？请分析并验证。

**答案**：当 K 极小时，潜在空间被严重"压扁"——所有输入图像的各个位置只能从 8 个码字中选择。对于 MNIST 的 7×7=49 个位置，每个位置只有 8 种选择，信息容量远不足以编码所有数字的变体。典型结果是：重建图像的各个区域会出现明显的块状伪影、重建质量急剧下降、不同数字变得相似。但码本利用率为 100%（所有 8 个码字肯定都会被使用）。

**Q6**: 编写代码实现 EMA（指数移动平均）来更新码本，替代梯度方式的码本更新。哪种方式更好？为什么？

**答案**：

```python
# EMA 更新码本的实现
def update_codebook_ema(z_e, encoding_indices, codebook, ema_count,
                         ema_weight, decay=0.99, eps=1e-5):
    """
    z_e: (N, D) 编码器输出 (已 flatten)
    encoding_indices: (N,) 每个向量的最近码字索引
    codebook: (K, D) 码本
    ema_count: (K,) 每个码字的使用次数 (EMA)
    ema_weight: (K, D) 每个码字的加权平均 (EMA)
    """
    K = codebook.shape[0]
    # 统计每个码字的 one-hot 使用量
    one_hot = F.one_hot(encoding_indices, num_classes=K).float()  # (N, K)
    # sum_{i: index_i = k} z_e_i
    encodings_sum = one_hot.t() @ z_e  # (K, D)
    # 每个码字被使用的次数
    count = one_hot.sum(dim=0)  # (K,)

    # 指数移动平均更新
    ema_count = decay * ema_count + (1 - decay) * count
    ema_weight = decay * ema_weight + (1 - decay) * encodings_sum

    # 拉普拉斯平滑避免除零
    n = ema_count.sum()
    smoothed_count = (
        (ema_count + eps) / (n + K * eps) * n
    )
    codebook = ema_weight / smoothed_count.unsqueeze(1)
    return codebook, ema_count, ema_weight
```

EMA 方式更好，原因：(1) 更稳定——不像梯度更新那样受学习率和梯度噪声影响；(2) 本质上是 K-means 在线聚类的等价形式，有更清晰的优化目标；(3) 在实践中几乎消除了码本坍塌问题。

### 13.4 思考题

**Q7**: VQ-VAE 和 VAE 的本质区别是什么？为什么 VQ-VAE 中的"Variational"可能有些误导？

**答案**：本质区别在于潜在空间的类型：VAE 的潜在变量是连续的，遵循某个先验分布（通常是标准正态）；VQ-VAE 的潜在变量是离散的，通过确定性量化获得。VAE 使用 KL 散度作为正则化项来约束潜在分布；VQ-VAE 使用向量量化和承诺损失。

VQ-VAE 中的"Variational"确实有些误导——原始 VQ-VAE 论文中，变量是指通过将编码器输出视为某个分类分布的近似后验（即 $q(z|x)$ 是一个 one-hot 分类分布），其中 KL 散度退化为常数。因此 VQ-VAE 实际上更像是一个确定性的自编码器加离散化，而非真正的变分推断方法。VQ-VAE-2 论文也承认了 "VQ-VAE is not exactly a VAE"。

## 14. 学习路径建议

### 已掌握基础后的推荐路线

1. **深入自编码器家族**：对比学习 VAE（变分自编码器）、VQ-VAE、VQ-GAN，理解离散 vs 连续潜在空间的权衡

2. **自回归生成模型**：学习如何用 PixelCNN / Transformer 在 VQ-VAE 的离散编码上进行自回归建模，实现无条件图像生成

3. **FSQ（有限标量量化）**：作为 VQ 的简化替代方案，理解其"四舍五入 + 边界限制"的设计哲学和消除码本坍塌的原理

4. **多模态生成**：了解 DALL-E 如何使用 VQ-VAE tokenizer 生成离散图像 token，与文本 token 统一建模；了解 Stable Diffusion 如何在连续潜在空间中工作并与 VQ 对比

5. **码本进阶技术**：学习 Residual VQ (RVQ)、Product Quantization 等提升码本表达能力的进阶技术

### 前置知识清单（如未掌握建议先补）

- [ ] 自编码器（AE）基本原理
- [ ] 卷积神经网络（CNN）的 Conv2d、ConvTranspose2d
- [ ] PyTorch 的自动微分机制和 detach 操作
- [ ] MSE Loss 和梯度下降
- [ ] K-means 聚类（帮助理解向量量化的本质）
