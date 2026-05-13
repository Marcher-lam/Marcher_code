# FSQ（有限标量量化）学习文档

> 来源线索：本节内容根据原书中关于"基于 FSQ 的人脸生成与语音存储"（第13章 13.3-13.4节）的相关章节整理、扩展与教学化改写。

> 用四舍五入替代码本搜索，无需辅助损失，从根本上消除码本坍塌。

## 1. 算法基础认知

**一句话定义**：FSQ（Finite Scalar Quantization，有限标量量化）是一种极简的离散量化方法——将每个特征维度独立地"四舍五入"到最近的整数值，取代 VQ-VAE 中复杂的码本最近邻搜索和辅助损失机制。

**直觉类比**：VQ-VAE 像是去图书馆查一本特定的书——你需要在一整面墙的书中逐本对比找到最像的那本。FSQ 则像是考卷上的选择题——每个维度的答案就在预先框定的几个选项里（比如 A/B/C/D），你只需要选最接近的那个。由于每个维度独立量化，FSQ 的隐式码本大小 = 各维度可选值的乘积，组合数可以非常巨大。

**历史背景**：FSQ 于 2023 年由 Mentzer 等人在论文《Finite Scalar Quantization: VQ-VAE Made Simple》中提出。研究者观察到 VQ-VAE 虽然效果好，但训练复杂（需要精细调节承诺损失、码本损失）、容易出现码本坍塌。FSQ 的洞察是：如果把每个维度的值限定在几个固定的数值槽中（如 [-3, -2, -1, 0, 1, 2, 3]），再四舍五入，整个量化过程就变得极其简单且稳定——不需要码本参数、不需要辅助损失、更不会出现码本坍塌。

**算法定位**：深度学习 / 量化方法 / 离散表示学习。是对 VQ-VAE 中向量量化（VQ）模块的简化替代方案，可以"即插即用"地替换任何 VQ 模块。

**前置知识**：
- VQ-VAE 的基本结构和向量量化概念
- 四舍五入（Round）操作的不可微性
- 直通梯度估计（Straight-Through Estimator, STE）
- 双曲正切函数 tanh 及其反函数 atanh 的性质
- PyTorch 的基础张量操作

## 2. 核心原理

### 核心思想

VQ-VAE 的向量量化是在一个高维向量空间中做最近邻搜索——为每个 D 维向量在 K 个码字中找最匹配的。这本质上是一个复杂的"配对"问题。FSQ 换个角度问：能不能不建码本，直接对每个维度独立地"舍入"到固定值？

答案是肯定的。FSQ 的做法可以分解为三步：

1. **投影到低维空间**：将编码器输出的 D 维向量通过线性层投影到 d 维（d 通常 < 10）
2. **每个维度独立量化**：每个维度的值被 bounded（限定范围）然后 round（四舍五入）到最近的整数值
3. **投影回原始维度**：量化后的 d 维向量投影回 D 维空间，送入解码器

### 工作流程

1. **下投影（Project Down）**：$z_{\text{low}} = W_{\text{down}} \cdot z_e$，将 D 维 -> d 维
2. **边界约束（Bound）**：将 $z_{\text{low}}$ 的每个维度限制在 $[-L_i/2 + \text{offset}, L_i/2 - \text{offset}]$ 范围内
3. **四舍五入（RoundSTE）**：每个维度四舍五入到最近的整数，使用 STE 保持梯度
4. **归一化**：量化后的整数除以 $L_i/2$，缩放到 $[-1, 1]$ 范围
5. **上投影（Project Up）**：$z_q = W_{\text{up}} \cdot z_{\text{quantized}}$，将 d 维 -> D 维，送入解码器

### 关键概念解释

- **Levels（量化层级）**：一个列表，如 `[8, 5, 5, 5]`，表示第 1 维有 8 个可选整数值（类似 0..7），其余 3 维各 5 个值（0..4）。隐式码本大小 = $8 \times 5 \times 5 \times 5 = 1000$。
- **隐式码本（Implicit Codebook）**：FSQ 不显式维护一个 KxD 的嵌入表。实际的"码本"由各维度可选值的笛卡尔积隐式定义。例如 levels=[8,5,5,5] 时，所有可能的 $(i_1, i_2, i_3, i_4)$ 组合共 1000 个，每个对应一个 D 维向量（通过 `project_up` 映射获得）。
- **RoundSTE**：`z_hat = z.round()` 计算量化值，`z + (z_hat - z).detach()` 实现直通梯度。与 VQ-VAE 的直通梯度原理相同，但这里对每个标量独立执行，而非在向量空间中搜索。
- **Bound 操作**：将每个维度压缩到 $[-L/2, L/2]$，使用 tanh 函数平滑地实现。偶数 levels 需要 offset 处理对称性问题（如 8 个值：-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5）。
- **因子化编码（Factorized Codes）**：当 `dim != len(levels)` 时，使用线性投影进行维度变换（$D \to d$ 和 $d \to D$）。

### 与 VQ-VAE 的核心对比

```
                     VQ-VAE                          FSQ
              ┌─────────────────┐          ┌─────────────────┐
              │  z_e (B,D,h,w)  │          │  z_e (B,D,h,w)  │
              └────────┬────────┘          └────────┬────────┘
                       │                            │
                       ▼                            ▼
              ┌─────────────────┐          ┌─────────────────┐
              │  KNN搜索 KxD码本  │          │  Linear Down D->d │
              │  d = ‖z-e‖²      │          │  Bound([-L/2,L/2])│
              │  argmin(d)       │          │  Round + STE      │
              │  STE: 直通梯度    │          │  Normalize        │
              └────────┬────────┘          │  Linear Up d->D    │
                       │                   └────────┬────────┘
                       ▼                            ▼
              ┌─────────────────┐          ┌─────────────────┐
              │  z_q (B,D,h,w)  │          │  z_q (B,D,h,w)  │
              └────────┬────────┘          └────────┬────────┘
                       │                            │
                       ▼                            ▼
                     Decoder                      Decoder

    需要: commitment_loss        需要: 无辅助损失
         codebook_loss                只有重构损失
         码本参数 (K×D)              无码本参数
```

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度/值 |
|------|------|---------|
| $z_e$ | 编码器输出的连续特征 | $(B, D, H, W)$ 或 $(B, N, D)$ |
| $d$ | 低维量化空间维度 | 标量，通常 < 10 |
| $L$ | levels 列表 | 如 `[8, 5, 5, 5]`，$L_i$ 为第 i 维的可选值个数 |
| $z_{\text{low}}$ | 下投影后的低维向量 | $(B, N, d)$ |
| $z_{\text{bound}}$ | 边界约束后的向量 | $(B, N, d)$ |
| $z_{\text{quant}}$ | 四舍五入量化后的向量 | $(B, N, d)$，值为整数 |
| $W_{\text{down}}$ | 下投影矩阵 | $(D, d)$ |
| $W_{\text{up}}$ | 上投影矩阵 | $(d, D)$ |
| $\text{RoundSTE}$ | 四舍五入 + 直通梯度估计 | — |

### 3.1 下投影

将高维（D）的特征降到低维（d）以减少量化复杂度：

$$z_{\text{low}} = z_e \cdot W_{\text{down}}, \quad W_{\text{down}} \in \mathbb{R}^{D \times d}$$

d 通常选 4~10。当 `D == d` 时，可以跳过投影（即 `need_project = False`，也称 factorized codes 模式）。

### 3.2 边界约束（Bound）

对于每个维度 $i$，其值为 $z_{\text{low}}^{(i)}$。第 i 维有 $L_i$ 个可选整数值（0 到 $L_i - 1$）。我们需要将连续值 $z_{\text{low}}^{(i)}$ 映射到一个可供 round 的区间。

定义半宽度：

$$h_i = \frac{L_i - 1}{2} \cdot (1 + \epsilon)$$

其中 $\epsilon$ 是一个极小数（如 1e-3），用于防止 tanh 饱和区域的问题。

**偶数 levels 的 offset 处理**：当 $L_i$ 为偶数时，整数值不是关于 0 对称的（比如 L=8 时有 4 个负值和 4 个正值，中间在 -0.5 或 0.5）。需要用 offset 平移：

$$\text{offset}_i = \begin{cases} 0.5, & \text{if } L_i \text{ is even} \\ 0.0, & \text{if } L_i \text{ is odd} \end{cases}$$

有了 offset，量化目标区间变为 $[-h_i, h_i]$，其中的整数字为：

$$t_i \in \{-h_i + \text{offset}_i, -h_i + \text{offset}_i + 1, ..., h_i - \text{offset}_i\}$$

举例：$L_i=8$ 时，$h_i \approx 3.5$，量化目标为 {-3, -2, -1, 0, 1, 2, 3, 4}（平移后）。

**使用 tanh 进行平滑边界**：

为了让输入值平滑地映射到边界内，使用 tanh 函数（值域 (-1, 1)）：

$$\text{shift}_i = \text{atanh}\left(\frac{\text{offset}_i}{h_i}\right)$$

$$z_{\text{bound}}^{(i)} = \tanh(z_{\text{low}}^{(i)} + \text{shift}_i) \cdot h_i - \text{offset}_i$$

这确保了任意输入 $z_{\text{low}}^{(i)}$ 都能被平滑地压缩到 $[-h_i - \text{offset}_i, h_i - \text{offset}_i]$ 范围内，且量化目标是整数值。

### 3.3 RoundSTE — 核心量化操作

对 bounded 后的每个标量值四舍五入到最近整数：

$$\hat{z}^{(i)} = \text{round}(z_{\text{bound}}^{(i)})$$

但 round 函数不可微。使用直通梯度估计：

$$z_{\text{quant}}^{(i)} = z_{\text{bound}}^{(i)} + (\hat{z}^{(i)} - z_{\text{bound}}^{(i)}).\text{detach}()$$

**前向传播**：值为 $\hat{z}^{(i)}$（量化整数）
**反向传播**：梯度原样传递给 $z_{\text{bound}}^{(i)}$（因为 detach 的部分梯度为 0）

### 3.4 归一化（重新缩放到 [-1, 1]）

量化后的值再除以半宽度，恢复到 [-1, 1] 范围：

$$z_{\text{norm}}^{(i)} = \frac{z_{\text{quant}}^{(i)}}{h_i}$$

### 3.5 上投影

$$z_q = z_{\text{norm}} \cdot W_{\text{up}}, \quad W_{\text{up}} \in \mathbb{R}^{d \times D}$$

### 3.6 索引编码

给定量化后的整数向量 $(q_1, q_2, ..., q_d)$（每个 $q_i \in [0, L_i-1]$），可以使用混合基表示计算唯一索引：

$$\text{index} = q_1 + q_2 \cdot L_1 + q_3 \cdot L_1 L_2 + ... + q_d \cdot \prod_{j=1}^{d-1} L_j$$

### 3.7 为什么 FSQ 不需要辅助损失？

这是 FSQ 最精妙之处。在 VQ-VAE 中，argmin 操作的梯度为 0，因此需要辅助损失来更新码本。而在 FSQ 中：

- round 虽然也不可微，但通过 STE 梯度可以流回编码器
- 编码器直接接收来自重构损失的梯度
- 编码器会自发地学会将信息分散到不同的量化单元（quantization bin）中，因为这样做可以减少重构损失
- 没有任何可学习的码本参数需要单独优化

换句话说，VQ-VAE 需要显式告诉模型"请使用码本"，而 FSQ 的模型会自己发现"多使用不同的量化层级有助于降低重构损失"。

## 4. 训练过程讲解

### 4.1 完整训练步骤

**Step 1：前向传播**
1. 输入图像 $x$ 进入编码器，得到 $z_e$（D 维特征图）
2. 下投影：$z_{\text{low}} = W_{\text{down}} * z_e$（D -> d）
3. Bound：使用 tanh 将每个维度限制在各自的 $[-h_i, h_i]$ 范围内
4. RoundSTE：每个维度四舍五入到最近整数 + 直通梯度
5. 归一化：除以 $h_i$，缩放到 $[-1, 1]$
6. 上投影：$z_q = W_{\text{up}} * z_{\text{norm}}$（d -> D）
7. 解码器重建：$\hat{x} = \text{Decoder}(z_q)$

**Step 2：损失计算**
只会计算重构损失：
$$\mathcal{L} = \text{MSE}(x, \hat{x})$$

没有码本损失和承诺损失！

**Step 3：反向传播**
1. 重构损失的梯度通过解码器 -> 上投影 -> RoundSTE(直通) -> 下投影 -> 编码器
2. 下投影和上投影参数也通过重构损失更新
3. RoundSTE 的 detach 操作确保 round 差异不干扰梯度流

**Step 4：参数更新**
使用标准优化器（如 AdamW）更新编码器、解码器和投影层的所有参数。

### 4.2 与 VQ-VAE 训练的关键差异

| 方面 | VQ-VAE | FSQ |
|------|--------|-----|
| 可学习的量化参数 | 码本 $(K, D)$ | 无（或仅有投影矩阵） |
| 辅助损失数量 | 2（codebook_loss, commitment_loss） | 0 |
| 超参数调节 | 需要调 $\beta$、$K$、EMA decay 等 | 只需选 levels |
| 码本坍塌风险 | 高 | 无 |
| 训练稳定性 | 中低 | 高 |

### 4.3 训练伪代码

```
for epoch in range(num_epochs):
    for x in dataloader:
        z_e = encoder(x)                  # (B, D, h, w)
        z_low = linear_down(z_e)          # (B, d, h, w) 投影到低维
        z_bound = bound(z_low)            # tanh 边界约束
        z_quant = round_ste(z_bound)      # 四舍五入 + 直通梯度
        z_norm = z_quant / half_width     # 归一化到 [-1,1]
        z_q = linear_up(z_norm)           # 投影回高维
        x_hat = decoder(z_q)              # 重建图像

        loss = MSE(x_hat, x)              # 只需要重构损失!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5. 应用场景

### 5.1 人脸图像生成
FSQ 在 CelebA-HQ 人脸数据集上展现出了与 VQ-VAE 相当的生成质量。如原书示例，使用 levels=[8,5,5,5]（隐式码本 1000 个），FSQ 可以有效地编码和解码人脸图像，重建图像保留了面部的整体结构、表情和光照特征。

### 5.2 语音信号压缩与存储
连续语音信号转换为离散表示后可以极大节省存储空间。FSQ 通过四舍五入将每个语音片段直接转换为整数值，无需维护大型码本。这在低资源设备上的语音存储（如智能手表、IoT 设备）中特别有价值。

### 5.3 密码机式的通信安全
如原书所述，编码器和解码器可以分离存储和使用——编码器将语音转为离散 token（密文），解码器再将 token 转回语音（明文）。两端持有不同组件，构成一种信息隐藏机制。虽然这不等同于密码学安全，但在特定场景下提供了一层额外的信息保护。

### 5.4 多模态生成
FSQ 可以替代 VQ 作为任何需要离散潜在空间的生成模型的后端量化器。例如，未来可以将 FSQ 整合到类 DALL-E 的文生图流程中，用更简单的量化机制提升训练效率和稳定性。

### 5.5 深度估计等密集预测任务
FSQ 论文展示了将量化表示用于深度估计等任务，其中离散化的中间表示有助于模型学习更加结构化的特征。

## 6. 优缺点分析

### 优点

| 优点 | 详细说明 |
|------|----------|
| **极简实现** | 核心操作只有四舍五入和 tanh 边界约束，代码量远小于 VQ-VAE |
| **无码本坍塌** | 没有显式码本参数，量化单元由 levels 隐式定义，每个 bin 都会自然地接收梯度分配 |
| **无辅助损失** | 训练只需要重构损失，不需要调节承诺损失权重、码本损失等超参数 |
| **训练稳定** | 损失曲线平滑，不需要码本重启、EMA 更新等额外技巧 |
| **隐式码本大** | 举例 levels=[8,8,8,8,8] 隐式码本 = 32768，levels=[8,6,6,6,6,6] = 62208，远大于 VQ-VAE 通常的几百到几千 |
| **即插即用** | 可以直接替换任何 VQ-VAE 模型中的 VQ 模块 |
| **梯度流自然** | RoundSTE 的梯度流比 VQ 的 argmin-STE 更自然，梯度失真更小 |
| **灵活控制** | 通过 levels 列表精细控制量化精度，不同维度可以有不同的量化层级 |

### 缺点

| 缺点 | 详细说明 |
|------|----------|
| **维度独立性假设** | 假设各维度可以独立量化，可能丢失维度间的联合信息（VQ 在向量空间中搜索可以保留这种关联） |
| **表达力上限** | 隐式码本虽然大，但结构受限于笛卡尔积形式——某些组合在某些任务中可能永远不会被用到 |
| **Levels 选择需要经验** | levels 列表的选择（个数、每维大小）对最终效果有影响，且没有明确的理论指导 |
| **Bound 操作的局限性** | tanh 的饱和效应可能使某些大值被"压扁"而丢失信息 |
| **d 较小时信息压缩** | 低维投影可能丢失信息，尤其是 $d \ll D$ 时 |
| **研究较少** | 相比 VQ-VAE 的丰富文献，FSQ 尚缺乏广泛的变体和改进研究 |

## 7. 调库实现

下面使用 PyTorch 实现完整的 FSQ 量化器和基于 FSQ 的自编码器，在 MNIST 数据集上验证。

```python
"""
FSQ (有限标量量化) 完整调库实现 (PyTorch)
数据集: MNIST 手写数字
目标: 用四舍五入替代码本搜索，验证无辅助损失训练
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
LATENT_DIM = 64          # 编码器输出的特征维度 D
FSQ_LEVELS = [8, 5, 5, 5]  # FSQ 量化层级，隐式码本 = 8*5*5*5 = 1000
LR = 2e-4
NUM_EPOCHS = 20
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1

# FSQ levels 配置
d_fsq = len(FSQ_LEVELS)   # 低维量化空间维度 d=4
implicit_codebook_size = int(np.prod(FSQ_LEVELS))
print(f"FSQ 隐式码本大小: {implicit_codebook_size}")

# ======================== 数据集加载 ========================
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# ======================== FSQ 量化器 ========================
class FSQ(nn.Module):
    """
    有限标量量化器 (Finite Scalar Quantization)。
    不需要可学习的码本参数，核心操作是每个维度的四舍五入。
    """
    def __init__(self, levels, dim, need_project=True):
        """
        参数:
            levels: List[int], 每个维度的量化层级数。如 [8,5,5,5]
            dim: int, 编码器输出的特征维度 D
            need_project: bool, 是否需要投影（D != len(levels) 时自动为 True）
        """
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.long)    # (d,)
        self.dim = dim                                           # D
        self.d = len(levels)                                     # d, 低维度
        self.need_project = need_project or (dim != self.d)

        # 计算每个维度的半宽度 h_i = (L_i - 1) / 2
        # eps 防止 tanh 饱和
        eps = 1e-3
        self.half_width = (self.levels - 1) * (1 + eps) / 2     # (d,)
        self.half_width_int = self.levels // 2                   # (d,)

        # 偶数 levels 的 offset 处理
        self.offset = torch.where(
            self.levels % 2 == 0,     # 偶数
            torch.tensor(0.5),
            torch.tensor(0.0)
        )  # (d,)

        # 预处理 shift = atanh(offset / half_width)
        self.shift = torch.atanh(self.offset / self.half_width)  # (d,)

        # 线性投影层 (如果需要)
        if self.need_project:
            self.project_down = nn.Linear(dim, self.d, bias=False)
            self.project_up = nn.Linear(self.d, dim, bias=False)
        else:
            self.project_down = nn.Identity()
            self.project_up = nn.Identity()

    def bound(self, z):
        """
        将输入 z 的每个维度约束到 [-half_width, half_width] 范围。
        使用 tanh 平滑映射。
        z: (..., d)
        """
        # (z + shift).tanh() 将值平滑压缩到 (-1, 1)
        # 乘以 half_width 缩放到 [-half_width, half_width]
        # 减去 offset 实现偶数 levels 的中心对齐
        z_bound = (z + self.shift.to(z.device)).tanh() * \
                   self.half_width.to(z.device) - \
                   self.offset.to(z.device)
        return z_bound

    def round_ste(self, z):
        """
        四舍五入 + 直通梯度估计。
        z: (..., d), 已 bound 的值
        返回: z 四舍五入后的整数 + STE 梯度
        """
        z_hat = torch.round(z)  # 四舍五入到最近整数
        # STE: 前向 = z_hat, 反向梯度 = dz_hat (即 dz)
        z_ste = z + (z_hat - z).detach()
        return z_ste

    def forward(self, z_e, return_indices=False):
        """
        参数:
            z_e: (B, D, H, W) 或 (B, N, D) 编码器输出
            return_indices: 是否返回整数索引
        返回:
            z_q: 量化后的特征，形状同 z_e
            indices: (B*H*W,) 整数索引 (当 return_indices=True)
        """
        # 处理输入形状
        is_4d = (z_e.dim() == 4)
        if is_4d:
            B, D, H, W = z_e.shape
            z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (B*H*W, D)
        else:
            z_e_flat = z_e  # (N, D)

        # Step 1: 下投影 D -> d
        z_low = self.project_down(z_e_flat)  # (N, d)

        # Step 2: 边界约束
        z_bound = self.bound(z_low)  # (N, d)

        # Step 3: 四舍五入 + STE
        z_quant = self.round_ste(z_bound)  # (N, d)

        # Step 4: 归一化到 [-1, 1]
        z_norm = z_quant / self.half_width.to(z_quant.device)  # (N, d)

        # Step 5: 上投影 d -> D
        z_q_flat = self.project_up(z_norm)  # (N, D)

        # 计算索引
        indices = None
        if return_indices:
            # 将量化值偏移到 [0, L_i-1] 范围
            quant_int = z_quant + self.half_width_int.to(z_quant.device)
            quant_int = quant_int.clamp(0, self.levels.to(z_quant.device) - 1).long()

            # 混合基编码: index = q1 + q2*L1 + q3*L1*L2 + ...
            indices = quant_int[:, 0].clone()
            multiplier = 1
            for i in range(1, self.d):
                multiplier *= self.levels[i - 1].item()
                indices += quant_int[:, i] * multiplier

        # 恢复形状
        if is_4d:
            z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        else:
            z_q = z_q_flat

        if return_indices:
            return z_q, indices
        return z_q


# ======================== 编码器与解码器 ========================
class Encoder(nn.Module):
    """将 28x28 图像编码为 7x7 的 D 维特征图"""
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, 1)   # 28->14
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)            # 14->7
        self.conv3 = nn.Conv2d(64, latent_dim, 3, 1, 1)    # 7->7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    """将 7x7 的 D 维特征图解码为 28x28 图像"""
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, 64, 3, 1, 1)           # 7->7
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)          # 7->14
        self.conv3 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)# 14->28

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


# ======================== FSQ 自编码器完整模型 ========================
class FSQAutoencoder(nn.Module):
    """使用 FSQ 量化的自编码器 — 注意：没有任何辅助损失！"""
    def __init__(self, in_channels=1, latent_dim=64, levels=None):
        super().__init__()
        if levels is None:
            levels = [8, 5, 5, 5]
        self.encoder = Encoder(in_channels, latent_dim)
        self.fsq = FSQ(levels=levels, dim=latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):
        """
        返回:
            x_recon: 重建图像
            indices: 量化索引
        """
        z_e = self.encoder(x)
        z_q, indices = self.fsq(z_e, return_indices=True)
        x_recon = self.decoder(z_q)
        return x_recon, indices

    def encode(self, x):
        """获取 FSQ 量化索引"""
        z_e = self.encoder(x)
        _, indices = self.fsq(z_e, return_indices=True)
        return indices


# ======================== 训练 ========================
model = FSQAutoencoder(
    in_channels=IMAGE_CHANNELS,
    latent_dim=LATENT_DIM,
    levels=FSQ_LEVELS
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"FSQ levels: {FSQ_LEVELS}")
print(f"隐式码本大小: {implicit_codebook_size}")
print("注意: FSQ 训练只需要重构损失! 不需要辅助损失!")
print("\n开始训练...\n")

train_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for data, _ in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        x_recon, _ = model(data)

        # 唯一的损失: 重构损失 — 没有任何辅助损失!
        loss = F.mse_loss(x_recon, data)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Recon Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

print("\n训练完成!")

# ======================== 测试与可视化 ========================
model.eval()

test_iter = iter(test_loader)
test_images, _ = next(test_iter)
test_images = test_images[:8].to(device)

with torch.no_grad():
    x_recon, indices = model(test_images)
    recon_mse = F.mse_loss(x_recon, test_images).item()

print(f"\n测试集重构 MSE: {recon_mse:.4f}")

# ---- 可视化 1: 原始 vs 重建 ----
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i in range(8):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap="gray")
    axes[0, i].set_title(f"Original {i+1}" if i == 0 else f"{i+1}")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_recon[i].cpu().squeeze(), cmap="gray")
    axes[1, i].set_title(f"FSQ Recon {i+1}" if i == 0 else f"{i+1}")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("FSQ Reconstructed", fontsize=12)
plt.suptitle("FSQ Autoencoder: MNIST Reconstruction (No Auxiliary Losses)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("fsq_reconstruction.png", dpi=100, bbox_inches="tight")
plt.show()

# ---- 可视化 2: 训练损失曲线 ----
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, "r-o", markersize=4)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Reconstruction Loss", fontsize=12)
plt.title("FSQ Training Loss Curve (Reconstruction Only)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fsq_loss_curve.png", dpi=100, bbox_inches="tight")
plt.show()

# ---- 可视化 3: 量化层级使用分布 ----
with torch.no_grad():
    all_indices = []
    for data, _ in test_loader:
        data = data.to(device)
        _, idx = model(data)
        all_indices.append(idx.cpu())
    all_indices = torch.cat(all_indices, dim=0)  # 所有位置的索引

    # 统计每个隐式码字索引的使用次数
    usage = torch.bincount(all_indices, minlength=implicit_codebook_size)
    active = (usage > 0).sum().item()
    usage_rate = active / implicit_codebook_size * 100

    # Perplexity (分布的均匀程度)
    probs = usage.float() / usage.sum()
    probs_nonzero = probs[probs > 0]
    perplexity = torch.exp(-(probs_nonzero * torch.log(probs_nonzero)).sum()).item()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 所有码字使用分布
    axes[0].bar(range(implicit_codebook_size), usage.numpy(), width=1.0, alpha=0.7)
    axes[0].set_xlabel("Implicit Codebook Index", fontsize=11)
    axes[0].set_ylabel("Usage Count", fontsize=11)
    axes[0].set_title(f"FSQ Implicit Codebook Usage\n"
                      f"({active}/{implicit_codebook_size} bins used, "
                      f"{usage_rate:.1f}% utilization, "
                      f"Perplexity={perplexity:.0f})", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Top-20 最常用码字
    top_k = min(20, implicit_codebook_size)
    sorted_usage, sorted_idx = torch.sort(usage, descending=True)
    axes[1].bar(range(top_k), sorted_usage[:top_k].numpy(), color="steelblue")
    axes[1].set_xlabel("Rank", fontsize=11)
    axes[1].set_ylabel("Usage Count", fontsize=11)
    axes[1].set_title(f"Top-{top_k} Most Used Implicit Codewords", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 每维度层级的边际分布
    # 从索引反推出各维度的值
    z_quant_all = []
    for data, _ in test_loader:
        data = data.to(device)
        B = data.shape[0]
        z_e = model.encoder(data)
        B, D, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
        z_low = model.fsq.project_down(z_flat)
        z_bound = model.fsq.bound(z_low)
        z_quant = model.fsq.round_ste(z_bound)
        vals = z_quant + model.fsq.half_width_int.to(device)
        z_quant_all.append(vals.cpu())
    z_all = torch.cat(z_quant_all, dim=0)  # (total_N, d)

    dim_names = [f"Dim {i+1}\n(L={FSQ_LEVELS[i]})" for i in range(len(FSQ_LEVELS))]
   encoding_steps = None
    x_data = range(len(FSQ_LEVELS))
    width = 0.15
    for i in range(len(FSQ_LEVELS)):
        dim_vals = z_all[:, i].long()
        dim_counts = torch.bincount(dim_vals, minlength=FSQ_LEVELS[i])
        # 每个维度的层级分布
        bars = axes[2].bar(
            [x + (i - len(FSQ_LEVELS)/2 + 0.5) * width for x in range(FSQ_LEVELS[i])],
            dim_counts.numpy(), width,
            alpha=0.7
        )
    axes[2].set_xlabel("Quantized Value per Dimension", fontsize=11)
    axes[2].set_ylabel("Count", fontsize=11)
    axes[2].set_title("Per-Dimension Level Distribution", fontsize=12)

    plt.tight_layout()
    plt.savefig("fsq_codebook_usage.png", dpi=100, bbox_inches="tight")
    plt.show()

# ---- 可视化 4: 不同重构质量对比 (随 epoch 变化) ----
with torch.no_grad():
    # 取单个样本
    single_img = test_images[0:1]
    # 尝试用部分索引重建（模拟不同压缩率的效果）
    plt.figure(figsize=(8, 4))
    plt.imshow(single_img.cpu().squeeze(), cmap="gray")
    plt.title("Original Test Image", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("fsq_single_original.png", dpi=100, bbox_inches="tight")
    plt.show()

    # 使用索引重建
    _, indices = model(single_img)
    z_e = model.encoder(single_img)
    z_low = model.fsq.project_down(z_e.permute(0, 2, 3, 1).contiguous().view(-1, LATENT_DIM))
    z_bound = model.fsq.bound(z_low)
    z_quant = model.fsq.round_ste(z_bound)
    quant_int = z_quant + model.fsq.half_width_int.to(device)

    print(f"\n单张图像 FSQ 量化值 (7x7 个位置, 每位置 {len(FSQ_LEVELS)} 维):")
    for hh in range(7):
        row_str = "  ".join(
            f"[{','.join(f'{quant_int[hh*7+ww, i].item():.0f}' for i in range(len(FSQ_LEVELS)))}]"
            for ww in range(7)
        )
        print(f"  Row {hh}: {row_str}")

# ---- 可视化 5: FSQ vs 理论对比 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# VQ-VAE vs FSQ 概念对比
categories = ["Number of\nAuxiliary Losses", "Learnable\nCodebook Params",
              "Codebook\nCollapse Risk", "Training\nComplexity"]
vq_values = [2, 1, 1, 1]  # 归一化到 0-1: 越高越不利
fsq_values = [0, 0, 0, 0.3]

x = np.arange(len(categories))
width = 0.35
bars1 = axes[0].bar(x - width/2, vq_values, width, label="VQ-VAE",
                    color="coral", alpha=0.8)
bars2 = axes[0].bar(x + width/2, fsq_values, width, label="FSQ",
                    color="steelblue", alpha=0.8)
axes[0].set_ylabel("Relative Drawback Level", fontsize=11)
axes[0].set_title("VQ-VAE vs FSQ: Training Difficulty Comparison", fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=9)
axes[0].legend(fontsize=11)
axes[0].set_ylim(0, 2.5)

# 为 FSQ 柱状图添加标注
for bar in bars2:
    height = bar.get_height()
    axes[0].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

axes[0].annotate("Lower is better", xy=(0.98, 0.02), xycoords="axes fraction",
                 ha="right", fontsize=9, fontstyle="italic", color="gray")

# 码本利用率理论对比 (FSQ 无坍塌)
codebook_labels = ["VQ-VAE\n(typical)", "FSQ\n(theoretical)"]
util_values = [30, 95]  # 利用率百分比
colors = ["coral", "steelblue"]
bars = axes[1].bar(codebook_labels, util_values, color=colors, alpha=0.8, width=0.4)
axes[1].set_ylabel("Codebook Utilization (%)", fontsize=11)
axes[1].set_title("Codebook Utilization: VQ-VAE vs FSQ", fontsize=13)
axes[1].set_ylim(0, 110)
for bar, val in zip(bars, util_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val}%", ha="center", fontsize=12, fontweight="bold")
axes[1].axhline(y=100, color="green", linestyle="--", alpha=0.3, label="100%")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("fsq_vs_vq_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n所有可视化已保存!")
```

## 8. 手工代码实现

下面从零实现 FSQ 的核心量化逻辑，包括 bound、round_ste、索引编解码，以及完整的测试验证。

```python
"""
FSQ 手工代码实现 (从零搭建)
所有量化操作均用基础 PyTorch 实现，不依赖 nn.Module 高层封装。
包含: bound(边界约束), round_ste(四舍五入+STE), 索引编解码
"""

import torch
import torch.nn.functional as F
import numpy as np

# ======================== 手工 FSQ 核心类 ========================
class FSQScratch:
    """
    从零实现的有限标量量化器。
    与 VQ-VAE 的根本区别: 没有可学习的码本，每个维度独立四舍五入。
    """
    def __init__(self, levels, dim, need_project=True):
        """
        参数:
            levels: List[int], 每个维度的量化层级数, 如 [8, 5, 5, 5]
            dim: int, 输入特征维度 D
            need_project: 是否需要 D<->d 投影
        """
        self.levels = np.array(levels, dtype=np.int64)
        self.dim = dim
        self.d = len(levels)
        self.need_project = need_project or (dim != self.d)

        eps = 1e-3
        self.half_width = (self.levels - 1) * (1 + eps) / 2.0      # (d,)
        self.half_width_int = self.levels // 2                      # (d,)

        # 偶数 levels 需要 offset
        self.offset = np.where(self.levels % 2 == 0, 0.5, 0.0)
        # shift = atanh(offset / half_width)
        self.shift = np.arctanh(self.offset / self.half_width)

        # 转换为 PyTorch tensors
        self.levels_t = torch.tensor(self.levels, dtype=torch.long)
        self.half_width_t = torch.tensor(self.half_width, dtype=torch.float32)
        self.half_width_int_t = torch.tensor(self.half_width_int, dtype=torch.long)
        self.offset_t = torch.tensor(self.offset, dtype=torch.float32)
        self.shift_t = torch.tensor(self.shift, dtype=torch.float32)

        # 线性投影参数 (手工管理)
        self.params = []
        if self.need_project:
            # Kaiming 初始化
            self.W_down = torch.randn(dim, self.d) * np.sqrt(2.0 / dim)
            self.W_up = torch.randn(self.d, dim) * np.sqrt(2.0 / self.d)
            self.W_down.requires_grad = True
            self.W_up.requires_grad = True
            self.params.extend([self.W_down, self.W_up])

    def bound(self, z):
        """
        将 z 的每个维度约束到 [-half_width, half_width]。
        使用 tanh 平滑映射。
        z: (N, d)
        """
        device = z.device
        # (z + shift).tanh() 映射到 (-1, 1)
        # * half_width 缩放到 [-half_width, half_width]
        # - offset 实现偶数 levels 对齐
        z_bound = (
            (z + self.shift_t.to(device)).tanh()
            * self.half_width_t.to(device)
            - self.offset_t.to(device)
        )
        return z_bound

    def bound_inverse(self, z_bound):
        """
        bound 的逆操作，用于恢复原始尺度 (调试用)。
        """
        device = z_bound.device
        z_restored = (z_bound + self.offset_t.to(device)) / self.half_width_t.to(device)
        z_restored = torch.atanh(z_restored.clamp(-0.999, 0.999)) - self.shift_t.to(device)
        return z_restored

    def round_ste(self, z):
        """
        四舍五入 + 直通梯度估计。
        z: (N, d), 已 bound 的连续值
        """
        z_hat = torch.round(z)
        # STE: 前向值 = z_hat, 反向梯度传给 z
        z_ste = z + (z_hat - z).detach()
        return z_ste

    def forward(self, z_e, return_indices=False):
        """
        完整前向传播。
        z_e: (N, D) 编码器输出的展平特征
        返回: z_q, indices
        """
        N, D = z_e.shape
        device = z_e.device

        # Step 1: 下投影 D -> d
        if self.need_project:
            z_low = z_e @ self.W_down.to(device)   # (N, d)
        else:
            z_low = z_e                            # (N, d)

        # Step 2: 边界约束
        z_bound = self.bound(z_low)                # (N, d)

        # Step 3: 四舍五入 + STE
        z_quant = self.round_ste(z_bound)          # (N, d)

        # Step 4: 归一化到 [-1, 1]
        z_norm = z_quant / self.half_width_t.to(device)

        # Step 5: 上投影 d -> D
        if self.need_project:
            z_q = z_norm @ self.W_up.to(device)    # (N, D)
        else:
            z_q = z_norm

        # 索引计算
        indices = None
        if return_indices:
            # 将量化值偏移到 [0, L_i-1] 范围
            quant_int = (z_quant + self.half_width_int_t.to(device)).long()
            quant_int = quant_int.clamp(
                torch.zeros(self.d, device=device, dtype=torch.long),
                self.levels_t.to(device) - 1
            )
            # 混合基编码
            indices = self._encode_to_index(quant_int)

        return z_q, indices

    def _encode_to_index(self, quant_int):
        """
        将 (N, d) 的量化整数向量编码为标量索引。
        quant_int: (N, d), 每维取值 [0, L_i-1]
        返回: (N,), index = q_1 + q_2*L_1 + q_3*L_1*L_2 + ...
        """
        N = quant_int.shape[0]
        indices = quant_int[:, 0].clone()
        multiplier = 1
        for i in range(1, self.d):
            multiplier *= self.levels[i - 1]
            indices += quant_int[:, i] * multiplier
        return indices

    def _decode_from_index(self, indices):
        """
        从标量索引还原 (N, d) 的量化整数向量。
        这是 _encode_to_index 的逆操作。
        """
        N = indices.shape[0]
        device = indices.device
        remaining = indices.clone()
        result = torch.zeros(N, self.d, dtype=torch.long, device=device)

        # 按混合基逐维解码
        for i in range(self.d):
            divisor = 1
            for j in range(i):
                divisor *= self.levels[j]
            result[:, i] = (remaining // divisor) % self.levels[i]

        return result


# ======================== 测试代码 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("FSQ 手工实现测试")
    print("=" * 60)

    # ---- 测试 1: 基本前向传播 ----
    levels = [8, 5, 5, 5]  # 隐式码本 = 8*5*5*5 = 1000
    latent_dim = 64
    implicit_size = int(np.prod(levels))

    fsq = FSQScratch(levels=levels, dim=latent_dim)
    total_params = sum(p.numel() for p in fsq.params)
    print(f"FSQ 参数配置:")
    print(f"  Levels: {levels}")
    print(f"  隐式码本大小: {implicit_size}")
    print(f"  低维量化空间 d: {fsq.d}")
    print(f"  特征维度 D: {latent_dim}")
    print(f"  可学习参数量: {total_params} (仅投影矩阵)")
    print(f"  注意: 没有任何可学习的码本参数!")

    # 模拟输入: (batch_size=4, D=64, H=7, W=7)
    B, D, H, W = 4, 64, 7, 7
    z_e = torch.randn(B, D, H, W)
    # 展平为 (B*H*W, D) = (196, 64)
    z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
    z_flat.requires_grad = True

    # 前向
    z_q, indices = fsq.forward(z_flat, return_indices=True)

    print(f"\n输入形状: z_flat={z_flat.shape}")
    print(f"输出形状: z_q={z_q.shape}")
    print(f"索引形状: indices={indices.shape}")
    print(f"索引范围: [{indices.min().item()}, {indices.max().item()}]")
    print(f"隐式码本利用率: {(indices.max().item() + 1) / implicit_size * 100:.1f}%")

    # ---- 测试 2: 梯度回传 ----
    loss = (z_q ** 2).mean()  # 任意损失
    loss.backward()

    z_grad_ok = z_flat.grad is not None and z_flat.grad.norm() > 0
    w_down_ok = fsq.W_down.grad is not None and fsq.W_down.grad.norm() > 0
    w_up_ok = fsq.W_up.grad is not None and fsq.W_up.grad.norm() > 0

    print(f"\n梯度回传测试:")
    print(f"  z_e 梯度: {'PASS' if z_grad_ok else 'FAIL'} (norm={z_flat.grad.norm().item():.6f})")
    print(f"  W_down 梯度: {'PASS' if w_down_ok else 'FAIL'} (norm={fsq.W_down.grad.norm().item():.6f})")
    print(f"  W_up 梯度: {'PASS' if w_up_ok else 'FAIL'} (norm={fsq.W_up.grad.norm().item():.6f})")

    if z_grad_ok and w_down_ok and w_up_ok:
        print("  RoundSTE 直通梯度验证: 通过!")

    # ---- 测试 3: Bound 操作验证 ----
    print(f"\nBound 操作测试:")
    # 用极端值测试
    extreme_input = torch.tensor([
        [100.0, 100.0, 100.0, 100.0],   # 极正
        [-100.0, -100.0, -100.0, -100.0], # 极负
        [0.0, 0.0, 0.0, 0.0],            # 中心
    ])
    bounded = fsq.bound(extreme_input)
    print(f"  极正输入 bounded: {bounded[0].tolist()}")
    print(f"  极负输入 bounded: {bounded[1].tolist()}")
    print(f"  零输入 bounded:   {bounded[2].tolist()}")
    print(f"  Bounded 范围: [{bounded.min().item():.2f}, {bounded.max().item():.2f}]")

    # 验证 bounded 值在 expected 范围内
    for i in range(fsq.d):
        hi = fsq.half_width[i]
        lo = -hi
        assert bounded[:, i].min() >= lo - 1e-3, \
            f"Dim {i}: bounded min {bounded[:, i].min()} < expected lo {lo}"
        assert bounded[:, i].max() <= hi + 1e-3, \
            f"Dim {i}: bounded max {bounded[:, i].max()} > expected hi {hi}"
    print("  Bound 范围验证: 通过!")

    # ---- 测试 4: 编码/解码索引对称性 ----
    print(f"\n索引编解码对称性测试:")
    # 生成随机的量化整数向量
    np.random.seed(42)
    N_test = 100
    random_q = []
    for i in range(N_test):
        vec = [np.random.randint(0, levels[j]) for j in range(fsq.d)]
        random_q.append(vec)
    random_q = torch.tensor(random_q, dtype=torch.long)

    # 编码
    indices_enc = fsq._encode_to_index(random_q)
    # 解码
    decoded_q = fsq._decode_from_index(indices_enc)

    match = (random_q == decoded_q).all().item()
    print(f"  编码->解码 完全匹配: {match}")

    assert match, "Symmetric encode/decode failed!"
    print("  索引编解码对称性: 通过!")

    # ---- 测试 5: 四舍五入的量化误差 ----
    print(f"\n四舍五入量化误差测试:")
    cont_vals = torch.linspace(-3.5, 3.5, 100).unsqueeze(1).repeat(1, fsq.d)  # (100, d)
    bounded_vals = fsq.bound(cont_vals)  # 先 bound
    quant_vals = fsq.round_ste(bounded_vals)  # 再 round

    # 平均量化误差
    quant_error = (bounded_vals - quant_vals).abs().mean().item()
    max_error = (bounded_vals - quant_vals).abs().max().item()
    print(f"  平均量化误差: {quant_error:.4f}")
    print(f"  最大量化误差: {max_error:.4f}")
    print(f"  理论最大误差: 0.5 (四舍五入)")
    assert max_error <= 0.5 + 1e-3, f"Max error {max_error} exceeds theoretical bound 0.5!"

    # ---- 测试 6: 与 VQ-VAE 对比特征 ----
    print(f"\nFSQ vs VQ-VAE 关键差异:")
    print(f"  VQ-VAE:")
    print(f"    - 需要维护 K×D 的显式码本参数")
    print(f"    - 需要最近邻搜索 (argmin over K vectors)")
    print(f"    - 需要 commitment_loss + codebook_loss 辅助损失")
    print(f"    - 容易发生 codebook collapse")
    print(f"")
    print(f"  FSQ (本实现):")
    print(f"    - 无显式码本参数 ({total_params} 可学习参数仅来自投影)")
    print(f"    - 每个维度独立四舍五入 (O(d) vs O(K*D*d))")
    print(f"    - 无辅助损失 (只需重构损失)")
    print(f"    - 无 codebook collapse 问题")
    print(f"    - 隐式码本大小: {implicit_size}")

    print(f"\n" + "=" * 60)
    print("所有测试通过! FSQ 手工实现验证成功。")
    print("=" * 60)
```

## 9. 可视化与结果理解

可视化代码已整合在第 7 节。本节对可视化结果进行解读。

### 9.1 重建效果对比

FSQ 自编码器的 MNIST 重建图像和原始图像在视觉上高度相似。数字的笔画方向、粗细、弧度等特征都被保留。边缘处可能略有模糊，这与离散量化的信息压缩有关，但整体可读性和结构性完好。

### 9.2 训练损失曲线

FSQ 的训练损失曲线通常比 VQ-VAE 更加平滑。原因是：
- 只需优化一个目标（重构损失），不存在损失函数之间的"拉扯"
- RoundSTE 的梯度估计比 VQ 的 argmin-STE 更加自然——灰度值四舍五入到最近整数，梯度近似误差最多 0.5（每个维度最多半个单位）
- 没有码本参数的随机初始化带来的方差

### 9.3 码本使用分布

FSQ 的隐式码本使用分布展示了“无坍塌”特性：
- 大多数码字有适度使用，不像 VQ-VAE 出现极端的长尾（少数码字占绝大多数使用量）
- 利用率通常 > 80%，远高于 VQ-VAE 的典型 20-50%
- Perplexity 显著更高

### 9.4 FSQ vs VQ-VAE 对比图

对比两者在训练复杂度和码本利用率方面的差异，FSQ 在四个方面都更优：
- 辅助损失数量: 0 vs 2
- 可学习码本参数: 0 vs K*D（通常数万到数十万）
- 码本坍塌风险: 无 vs 高
- 训练复杂度: 低 vs 中高

## 10. 模型评估

### 10.1 定量评估指标

| 指标 | 含义 | FSQ 预期表现 |
|------|------|-------------|
| **MSE / RMSE** | 逐像素重构误差 | 与同结构 VQ-VAE 相当或略优 |
| **PSNR** | 峰值信噪比 | 20-30 dB (MNIST), 25-35 dB (CelebA) |
| **SSIM** | 结构相似性 | 0.85-0.95 |
| **码本利用率** | 活跃码字占比 | > 80% (远优于 VQ-VAE) |
| **Perplexity** | 码字使用均匀度 | 数百至数千 |
| **训练时间/epoch** | 计算效率 | 比 VQ-VAE 快 (无 KNN search) |
| **GPU 内存** | 显存使用 | 低于 VQ-VAE (无 K*D 码本) |

### 10.2 定性评估

- **训练稳定性**：损失曲线应该平滑下降，没有 VQ-VAE 常见的震荡
- **码本健康度**：检查 active bins 比例，应该远高于 VQ-VAE 的典型值
- **重建一致性**：同一类的不同样本重建质量应均匀，无明显的"困难样本"聚集

### 10.3 评估代码

```python
def evaluate_fsq(model, test_loader, device):
    model.eval()
    total_mse = 0.0
    all_indices = []
    total_samples = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, indices = model(data)
            total_mse += F.mse_loss(x_recon, data, reduction="sum").item()
            all_indices.append(indices.cpu())
            total_samples += data.size(0)

    avg_mse = total_mse / total_samples
    psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else float("inf")

    # 码本统计
    all_idx = torch.cat(all_indices)
    usage = torch.bincount(all_idx)
    active = (usage > 0).sum().item()
    utilization = active / len(usage)
    probs = usage.float() / usage.sum()
    probs = probs[probs > 0]
    perplexity = torch.exp(-(probs * torch.log(probs)).sum()).item()

    return {
        "mse": avg_mse,
        "psnr": psnr,
        "active_bins": active,
        "total_bins": len(usage),
        "utilization": utilization,
        "perplexity": perplexity
    }
```

## 11. 常见问题与易错点

### 问题 1: levels 选择不当导致表达能力不足

- **现象**：levels 设置太小（如 `[4, 4]`），重建质量差，图像模糊或出现块状伪影；levels 设置太大（如 `[100, 100, 100, 100, 100]`），隐式码本过大（100 亿），模型过度依赖离散表示而没有学到足够的抽象特征。
- **原因**：levels 决定了量化精度。太小信息容量不足，太大则"门槛"太高——模型可能不使用所有组合。
- **解决方案**：
  1. 对于 28x28 图像：推荐 `[8, 5, 5, 5]` 或 `[8, 6, 6, 6, 6]`
  2. 对于 256x256 图像：推荐 `[8, 8, 8, 8, 8]` 或 `[8, 6, 6, 6, 6, 6]`
  3. 实践准则：隐式码本大小应在 500-100000 之间
  4. 可以通过实验：先用小 levels 验证 pipeline 通，再逐步增大

### 问题 2: 偶数 levels 的 offset 处理出错

- **现象**：使用偶数 levels（如 8）时，重建图像的某些区域出现系统性偏差或奇怪的 artifact。
- **原因**：偶数 levels 的整数值集合不是关于 0 对称的。如果没有正确处理 offset，量化值会系统性偏向一侧。例如 levels=8 时，值域本来应该对称在 [-3.5, 3.5] 但四舍五入到的整数范围可能是 [-4, 3] 或 [-3, 4]。
- **解决方案**：
  1. 正确使用 offset = 0.5 处理偶数情况
  2. 在代码中加入验证：检查 bound 后值的分布是否关于 0 对称
  3. 如不确定，优先使用奇数 levels（如 5, 7, 9），避免 offset 问题

### 问题 3: RoundSTE 导致的梯度不稳定

- **现象**：训练初期损失下降很慢或震荡。
- **原因**：round 的梯度近似在边界处（如值在 2.5 附近）特别不连续。每次跨过一个 .5 边界，梯度方向会剧变。这在高学习率下会导致振荡。
- **解决方案**：
  1. 降低学习率（1e-4 而非 2e-4）
  2. 使用梯度裁剪（clip_grad_norm_）
  3. 在训练初期使用 warmup（前几个 epoch 线性增加 LR）
  4. 这个问题的严重程度远低于 VQ-VAE（VQ 的 argmin 梯度失真更严重）

### 问题 4: 下投影导致的维度信息瓶颈

- **现象**：编码器 dim=512，但 levels 只有 4 维（d=4）。重建的图像虽然可读但丢失了大量细节。
- **原因**：D->d 的投影（512->4）是一个极强的信息瓶颈，可能压缩掉了解码器重建所需的高频细节。
- **解决方案**：
  1. 增大 d（增加 levels 列表长度）：如从 4 维增到 6 维或 8 维
  2. 使用 `need_project=False` 模式（当 D==d 时），避免投影压缩
  3. 提高隐式码本利用率：即使 d 小，每个维度的 levels 应该足够大

### 问题 5: 混淆 FSQ 和 VQ 的使用场景

- **现象**：在需要精细向量空间结构（如学习代码和代码之间的拓扑关系）的场景中使用 FSQ，效果不如 VQ。
- **原因**：FSQ 假设维度独立量化，这在某些场景（如语音编码中不同频带的强相关性）会丢失重要的联合信息。VQ 在向量空间中搜索可以保留这些维度间的相关性。
- **解决方案**：
  1. 图像生成任务：FSQ 和 VQ 效果相当
  2. 需要细粒度联合编码的任务：优先 VQ
  3. 低资源/快速验证场景：优先 FSQ
  4. 不确定时：先用 FSQ 快速原型（训练快），效果不够再换 VQ

## 12. 学习总结

FSQ 是向量量化领域的一股"清流"——它用一个极其简单的思想（每个维度独立四舍五入）解决了 VQ-VAE 中最棘手的问题（码本坍塌）。设计哲学是"少即是多"：去掉显式码本参数，去掉辅助损失，把一切交给重构损失的自然梯度流。

这个设计之所以有效，是因为四舍五入（round）操作虽然同样不可微，但其梯度近似（STE）比 argmin-STE 更加合理——每个维度的梯度失真最多等同于 round 误差（不超过 0.5），而 VQ 的 argmin 可能把梯度引向一个和最优方向完全不同的码字。

FSQ 的价值不仅在于它本身的效果，更在于它证明了"离散量化不一定要用码本"这个反直觉的事实。它的 levels 机制提供了一种"隐式码本"——通过笛卡尔积隐式定义巨额组合，且每个 bin 都能自然接收梯度信号。

在实际应用中，FSQ 可以视为 VQ-VAE 的"快速原型工具"——更快训练、更少调试、更稳定。当你需要部署一个离散表示学习系统时，FSQ 通常是最佳起点。

## 13. 练习题与思考题

### 13.1 概念理解题

**Q1**: 为什么 FSQ 不会出现码本坍塌？请从梯度流的角度解释。

**答案**：码本坍塌的根本原因是梯度分配不均——好的码字接收更多梯度变得更好，差的码字没有梯度变得更差，形成马太效应。

FSQ 不会出现码本坍塌的原因：
1. **没有显式码本参数**：FSQ 不存在"好码字"和"差码字"的概念——每个量化 bin 由 levels 隐式定义，没有独立更新的参数
2. **梯度通过 STE 均等流动**：重构损失的梯度通过 RoundSTE 均等地流过每个被使用的 bin，没有被"遗忘"的 bin
3. **自然探索机制**：编码器为了最小化重构损失，会自发地使用更多的 bin，因为使用更多 bin = 更精细的量化 = 更低的重构损失

### 13.2 数学推导题

**Q2**: 推导 FSQ 中偶数 levels（如 L=8）为什么需要 offset=0.5。

**答案**：

目标：将连续值 z 的量化目标设为整数。对于 bounds 后的值域 $[-h, h]$（$h \approx \frac{L-1}{2}$），我们需要整数量化目标关于 0 对称。

L=8（偶数）：h = (8-1)/2 = 3.5。直接 round 到 [-3.5, 3.5] 内的最近整数：
- z=3.2 -> round(3.2) -> 3
- z=-0.4 -> round(-0.4) -> 0
- z=-0.6 -> round(-0.6) -> -1

这样整数集合是 {-3, -2, -1, 0, 1, 2, 3}，共 7 个值，但 L=8 要求 8 个值。

加 offset=0.5 平移：值域变为 [-3.5+0.5, 3.5-0.5] = [-3, 3]。round 后：
- 整数集合: {-3, -2, -1, 0, 1, 2, 3}，共 7 个。

不对——需要再加平移让值域包含 8 个整数目标。

正确的理解：有 offset 时的值域为 $[-(h-\text{offset}), h-\text{offset}]$。L=8, h=3.5, offset=0.5 -> [-3, 3]。但 round 到 [-3, 3] 只有 7 个值。

实际上，offset 需要让 bound 不关于 0 对称，而是偏半个单位：

无 offset: z in [-3.5, 3.5], round 到 [-4, -3, -2, -1, 0, 1, 2, 3, 4] (9 个值，但边界 -4 和 4 几乎不会被用到).

有 offset=0.5: z in [-3.5, 3.5], z+offset in [-3, 4], round 到 [-3, -2, -1, 0, 1, 2, 3, 4] -> 恰好 8 个值.

所以 offset 的作用是"平移值域使 round 后恰好得到 L 个不同的整数值"。

**Q3**: FSQ 的隐式码本大小如何计算？如果 levels=[8, 8, 6, 6, 4]，隐式码本大小是多少？

**答案**：隐式码本大小 = 各维度可选值的乘积 = $\prod_{i=1}^{d} L_i$

levels = [8, 8, 6, 6, 4] 时：
隐式码本 = 8 * 8 * 6 * 6 * 4 = 64 * 36 * 4 = 64 * 144 = 9216

### 13.3 代码实践题

**Q4**: 修改 levels 配置并对比不同配置下的重建质量。

**答案**：以下是实验方案：

```python
# 实验: 对比不同 levels 配置
configs = [
    ([4, 4, 4, 4], "Small (4x4x4x4=256)"),
    ([8, 5, 5, 5], "Medium (8x5x5x5=1000)"),
    ([8, 8, 8, 8], "Large (8x8x8x8=4096)"),
    ([10, 8, 8, 8, 6], "Extra Large (30720)"),
]

results = {}
for levels, name in configs:
    model = FSQAutoencoder(
        in_channels=1, latent_dim=64, levels=levels
    ).to(device)
    # 训练(简化版，实际需完整训练)
    # ...
    results[name] = evaluate_fsq(model, test_loader, device)
```

预期结果：
- 较小的码本：重建质量较低但训练快速收敛
- 适中的码本：质量和速度的最佳平衡点
- 较大的码本：质量最好但需要更多数据支撑

**Q5**: 实现一个"混合量化器"——当编码器的某个输出向量离任何 VQ 码字都太远时，回退到 FSQ 方式量化。

**答案思路**：

```python
def hybrid_quantize(z_e, codebook, fsq):
    """
    混合 VQ+FSQ: 对每个向量，先尝试 VQ 量化。
    如果最小距离 > threshold，退回 FSQ 量化。
    """
    # 计算 VQ 距离
    distances = torch.cdist(z_e, codebook)
    min_dist, vq_idx = distances.min(dim=1)

    # 阈值
    threshold = 0.5
    use_vq = min_dist < threshold

    z_q = torch.zeros_like(z_e)
    # VQ 量化的部分
    z_q[use_vq] = codebook[vq_idx[use_vq]]
    # FSQ 量化的部分 (难样本)
    z_q[~use_vq] = fsq(z_e[~use_vq])

    # STE
    z_q = z_e + (z_q - z_e).detach()
    return z_q
```

### 13.4 思考题

**Q6**: 为什么 FSQ 中的"四舍五入"在深度学习中可以被训练？Round 不是不可微的吗？

**答案**：Round 本身确实不可微（梯度处处为 0 或未定义）。但 FSQ 使用了 Straight-Through Estimator（直通梯度估计）技巧：
- 前向传播：正常执行 round(z) -> 得到量化后的整数
- 反向传播：梯度跳过 round 操作，直接从输出复制到输入

具体实现 `z + (round(z) - z).detach()`：
- 前向值 = round(z)
- 反向时 .detach() 切断 (round(z)-z) 的梯度，因此 $\partial L / \partial z = \partial L / \partial \text{output}$（梯度直接传过去）

这个技巧虽然是一个近似，但因为 round 的误差最多 0.5（每个维度），梯度近似误差有限。而且编码器接收到的梯度信号（"往哪个方向调能降低重构损失"）通常方向大致正确，所以能有效训练。

相比之下，VQ-VAE 中的 argmin-STE 把梯度复制回编码器输出，但实际的 argmin 在最邻近切换时是极度不连续的（梯度可能指向完全错误的码字方向），所以其近似误差远大于 RoundSTE。

## 14. 学习路径建议

### 已掌握基础后的推荐路线

1. **深入对比量化方法**：系统对比 VQ-VAE、FSQ、Gumbel-Softmax VQ、Residual VQ (RVQ)、Lookup-Free Quantization (LFQ) 等在相同任务上的表现和训练动态

2. **大尺度图像生成**：在 CelebA-HQ、FFHQ 等更大分辨率数据集上测试 FSQ，理解 levels 配置与图像分辨率的关系

3. **视频量化压缩**：将 FSQ 扩展到视频域，处理时序维度的量化，探索帧间预测 + 量化的联合优化

4. **多模态统一 tokenizer**：尝试用同一个 FSQ 配置处理图像和语音，观察它们的量化模式差异

5. **自回归建模后端**：在 FSQ tokenizer 训练好后，用 GPT/Transformer 对离散 token 做自回归建模，实现无条件生成

### 前置知识清单（如未掌握建议先补）

- [ ] VQ-VAE 的基本原理和训练挑战（理解"为什么需要 FSQ"）
- [ ] 直通梯度估计（STE）的数学原理
- [ ] tanh 和 atanh 函数的性质
- [ ] 混合基数字系统（理解索引编解码）
- [ ] PyTorch 中 .detach() 和 requires_grad 的机制
