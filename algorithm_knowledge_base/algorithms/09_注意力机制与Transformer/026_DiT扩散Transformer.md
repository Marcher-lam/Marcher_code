# DiT 扩散Transformer 学习文档

> 来源线索：本节内容根据原书中关于"DiT 扩散Transformer"（第9章 9.2节）的相关章节整理、扩展与教学化改写。
> 用Transformer替代UNet做扩散模型，结合adaLN实现可控图像生成。

## 1. 算法基础认知

**一句话定义**：DiT（Diffusion Transformer）是完全基于Transformer架构构建的扩散模型，用Transformer替代传统的UNet来预测扩散过程中的噪声。

**直觉类比**：如果把传统的UNet扩散模型比作一个擅长局部修复的画师，那么DiT就是一个具有全局视野的艺术总监——它能看到整幅画的全局关系，理解每个区域与其他区域的联系，从而更精准地指导去噪过程。

**历史背景**：2023年，William Peebles（FAIR）和Saining Xie（NYU）在论文《Scalable Diffusion Models with Transformers》中提出了DiT架构。该论文系统性地证明了Transformer在扩散模型中的有效性，并探索了自适应层归一化（adaLN）等条件注入策略。DiT的提出标志着扩散模型正式进入"Transformer时代"，为后续Sora（OpenAI的视频生成模型）等大尺度生成模型奠定了架构基础。

**算法定位**：深度学习 / 生成模型 / 扩散模型 / Transformer。属于可控制条件图像生成模型，通过预测噪声实现从纯噪声到目标图像的逐步去噪。

**前置知识**：
- 扩散模型（DDPM）的基本原理：前向加噪、反向去噪
- Transformer架构：自注意力、前馈网络、层归一化
- ViT（Vision Transformer）：图像的Patch Embedding处理方式
- PyTorch张量操作与神经网络模块基础

## 2. 核心原理

### 核心思想

DiT的核心思想可以用一句话概括：**把扩散模型中预测噪声的网络，从卷积UNet换成Transformer**。

传统的DDPM使用UNet作为噪声预测网络，UNet通过编码器-解码器结构和跳跃连接来处理图像。但UNet本质上是一个卷积架构，它的感受野受限于卷积核大小，需要多层堆叠才能建立全局关联。

DiT的洞察是：Transformer天然具有全局注意力机制，每个token都能直接关注所有其他token。将图像切成Patch（类似ViT），每个Patch作为一个token，然后用Transformer的自注意力来处理这些Patch之间的关系——这样模型就能在每一步去噪时都拥有"全局视野"。

### 工作流程

整个DiT系统由以下核心模块组成：

1. **Patch Embedding（图像分块嵌入）**：用卷积将输入图像切成固定大小的Patch，每个Patch通过线性层映射为高维嵌入向量，加上位置编码（Position Embedding）。

2. **时间嵌入（Time Embedding）**：扩散过程的时间步t是一个标量（如t=500表示第500步）。通过正弦-余弦编码将t映射为高维向量。

3. **标签嵌入（Label Embedding）**：如果要做条件生成（如"生成数字3的图片"），将类别标签通过Embedding层映射为向量。

4. **条件融合**：将时间嵌入和标签嵌入相加，得到条件向量cond。这个cond会被送入每个DiTBlock中，通过adaLN机制调控Transformer的行为。

5. **DiTBlock（核心处理单元）**：接收Patch嵌入和条件向量，内部包含：
   - **adaLN调制**：根据条件向量生成6个调制参数（缩放、偏移、门控各两组）
   - **多头自注意力（MHA/MQA）**：Patch之间的全局交互
   - **前馈网络（SwiGLU MLP）**：逐Patch的非线性变换
   - **残差连接**：两次残差（注意力后一次，MLP后一次）

6. **输出重建**：处理后的Patch嵌入经过层归一化、线性投影，重组回原始图像的空间分辨率。

### 关键概念解释

- **adaLN（Adaptive Layer Normalization）**：不是使用固定的缩放/偏移参数，而是根据条件向量（时间+标签）动态生成归一化参数。这样每个时间步、每个类别都能有不同的归一化行为，使条件信息深度融入Transformer的每一层。
- **为什么用MQA**：在DiTBlock中，自注意力通常用MQA（多查询注意力）而非标准MHA。因为扩散模型的推理需要逐步迭代（通常1000步），MQA可以显著减少KV缓存的显存开销。
- **Patch化处理**：与ViT一致，用步长=stride的卷积实现Patch切分和初始嵌入的合并计算。

### 几何/直观解释

```
输入图像 (1, 28, 28)
       |
   Patch Embedding (Conv2d kernel=4, stride=4)
       |
  7x7=49个Patch, 每个16维 → Linear → 64维嵌入
       |
  + 位置编码 (49, 64)
       |
  ┌───→ DiTBlock 1 ←─── 条件向量 = 时间嵌入 + 标签嵌入
  │         |
  │    adaLN调制 → MQA注意力 → 残差 → adaLN调制 → MLP → 残差
  │
  ├───→ DiTBlock 2 ←─── (同一个条件向量)
  │
  ├───→ DiTBlock 3 ←─── (同一个条件向量)
       |
  LayerNorm → Linear(64 → 16) → 重组为 (1, 28, 28)
       |
  输出的就是预测的噪声
```

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_0$ | 原始干净图像 | $(B, C, H, W)$ |
| $x_t$ | 第t步加噪图像 | $(B, C, H, W)$ |
| $t$ | 扩散时间步 | 标量, $t \in [0, T-1]$ |
| $y$ | 条件标签（可选） | $(B,)$ |
| $P$ | Patch大小 | 标量，如4 |
| $N$ | Patch数量 | $(H/P) \times (W/P)$ |
| $d$ | 嵌入维度（emb_size） | 标量，如64 |
| $\epsilon$ | 真实噪声 | $(B, C, H, W)$ |
| $\epsilon_\theta$ | 模型预测噪声 | $(B, C, H, W)$ |
| $\alpha_t$ | 第t步信噪保留率 | 标量 |
| $\bar{\alpha}_t$ | 累积信噪保留率 | $\prod_{s=1}^{t} \alpha_s$ |

### 前向扩散过程（加噪）

给定干净图像 $x_0$，直接从 $x_0$ 采样 $x_t$：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$，$\beta_s$ 是预定义的噪声调度参数（如线性从0.0001到0.02）。

### DiT模型的前向计算

**Step 1 - Patch Embedding**：
$$\text{Patches} = \text{Conv2d}(x_t), \quad \text{Conv2d}: (C, H, W) \to (C \cdot P^2, N_h, N_w)$$
$$\text{Emb} = \text{Linear}(\text{Patches}) + \text{PosEmb}, \quad \text{Linear}: \mathbb{R}^{C \cdot P^2} \to \mathbb{R}^{d}$$

**Step 2 - 条件嵌入**：
$$t_{\text{emb}} = \text{TimeEmbedding}(t) = [\sin(\omega_k \cdot t), \cos(\omega_k \cdot t)]_{k=0}^{d/2-1}$$
其中 $\omega_k = \exp(-k \cdot \frac{\ln 10000}{d/2 - 1})$

$$y_{\text{emb}} = \text{Embedding}(y)$$
$$\text{cond} = t_{\text{emb}} + y_{\text{emb}}$$

**Step 3 - DiTBlock处理**：

每个DiTBlock内部执行两次调制-处理-残差循环：

第一次（注意力分支）：
$$[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \text{adaLN\_modulation}(\text{cond})$$
$$h = \text{MQA}(\text{LayerNorm}(x) \odot (1 + \gamma_1) + \beta_1)$$
$$x = x + \alpha_1 \cdot h$$

第二次（MLP分支）：
$$h = \text{MLP}(\text{LayerNorm}(x) \odot (1 + \gamma_2) + \beta_2)$$
$$x = x + \alpha_2 \cdot h$$

**Step 4 - 输出重建**：
$$\hat{\epsilon} = \text{Unpatchify}(\text{Linear}(\text{LayerNorm}(x)))$$

### 损失函数

训练目标是让模型预测的噪声尽可能接近真实噪声：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|_1 \right]$$

通常使用L1 Loss（MAE），也可以使用L2 Loss（MSE）。L1 Loss对异常值更鲁棒，在图像生成任务中常产生更清晰的边缘。

### 反向去噪过程（推理/采样）

从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪 $T$ 步得到 $x_0$：

第 $t$ 步去噪（从 $x_t$ 得到 $x_{t-1}$）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t, y) \right)$$

$$x_{t-1} = \begin{cases}
\mu_\theta(x_t, t) + \sigma_t \cdot z, & t > 0, z \sim \mathcal{N}(0, I) \\
\mu_\theta(x_t, t), & t = 0
\end{cases}$$

其中 $\sigma_t^2 = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$ 是后验方差。

## 4. 训练过程讲解

### 训练数据准备

1. 从数据集中采样一批干净图像 $x_0$ 和对应标签 $y$（如果有条件生成任务）
2. 为每张图像随机采样一个时间步 $t \in [0, T-1]$，$T$ 通常设为1000
3. 对每个 $x_0$ 采样随机噪声 $\epsilon \sim \mathcal{N}(0, I)$
4. 根据公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 生成加噪图像

### 训练循环

```
for epoch in 1..EPOCH:
    for (x_0, y) in DataLoader:
        t ~ Uniform(0, T-1)           # 随机采样时间步
        ε ~ N(0, I)                    # 随机生成噪声
        x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε  # 加噪
        ε_pred = DiT(x_t, t, y)        # 模型预测噪声
        loss = |ε - ε_pred|_1           # L1损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 关键训练细节

- **像素值归一化**：将图像从 [0, 1] 归一化到 [-1, 1]，以匹配高斯噪声的数值范围。
- **随机时间步**：每次迭代随机采样t，确保模型学会处理所有噪声水平。
- **学习率**：通常使用 $10^{-3}$ 到 $10^{-4}$ 的Adam优化器。
- **批次大小**：受显存限制，图像生成任务通常用较大的batch（如128-256）。
- **模型保存**：定期保存检查点（如每20个epoch），便于恢复和评估。

### 训练技巧

- 对较大图像（如256x256），先在小分辨率上预训练，再逐步提升分辨率（渐进式训练）。
- 使用指数移动平均（EMA）的模型参数进行最终采样，可以提高生成质量。
- 对条件生成任务，可以随机丢弃部分标签（label dropout），使模型同时支持无条件和条件生成（Classifier-Free Guidance的基础）。

## 5. 应用场景

### 核心应用场景

1. **条件图像生成**：根据类别标签生成指定类型的图像（如MNIST数字、ImageNet类别）。这是DiT最基础的应用。

2. **文本到图像生成**：将文本描述编码后作为条件输入DiT，实现文生图。Stable Diffusion 3、Sora等模型都采用了DiT架构或其变体。

3. **图像超分辨率**：以低分辨率图像为条件，生成高分辨率细节。DiT的全局注意力可以更好地恢复纹理一致性。

4. **图像修复与补全**：将已知区域作为条件，生成缺失区域的内容。

5. **视频生成**：将多帧图像作为序列输入Transformer，利用DiT架构生成连贯的视频帧（OpenAI Sora的核心架构）。

### 典型使用场景特征

- **需要全局一致性**：如整幅图的风格统一、光照一致
- **需要精细控制**：通过不同条件向量精确控制生成内容
- **需要可扩展性**：模型规模可随计算资源线性扩展（Transformer的天然优势）
- **多条件融合**：时间、文本、类别等多种条件可以自然地通过嵌入相加来融合

## 6. 优缺点分析

### 优点

1. **全局感受野**：Transformer的自注意力机制天然提供全局交互，每个Patch都能直接关注所有其他Patch，这在生成大尺寸、高分辨率图像时优势明显（相比UNet的局部卷积）。

2. **可扩展性强**：DiT遵循Transformer的scaling law，增大参数量和训练数据能持续提升生成质量。这也是Sora等大模型选择DiT架构的原因。

3. **架构统一**：文本、图像、视频等不同模态都使用同一套Transformer架构处理，降低了多模态系统的设计复杂度。

4. **灵活的条件注入**：adaLN机制使条件信息能深度嵌入每一层，而非仅在输入层注入。时间、文本、类别等条件可以自然融合。

5. **无需特殊结构**：不像UNet需要精心设计下采样/上采样路径和跳跃连接，DiT结构更"扁平"，便于工程实现和调优。

### 缺点

1. **计算复杂度高**：自注意力的复杂度是 $O(N^2)$（N为Patch数）。对256x256图像、Patch=4，N=4096，注意力矩阵大小达到4096x4096，计算开销很大。

2. **缺乏归纳偏置**：相比UNet的局部卷积偏置（假设邻近像素相关），Transformer没有任何空间先验，需要更多数据来学习空间结构。

3. **高分辨率挑战**：对高清图像（如1024x1024），Patch数量可能达到数万，自注意力的平方复杂度会变得不可承受。需要配合窗口注意力等优化手段。

4. **训练成本高**：扩散模型本身就需要大量采样步数（1000步），再加上Transformer的计算量，训练成本显著高于GAN等单步生成模型。

5. **对位置编码敏感**：模型需要依赖位置编码来理解Patch的空间位置关系，不同的位置编码方案（可学习/正弦/旋转）对效果有显著影响。

## 7. 调库实现

```python
"""
DiT (Diffusion Transformer) - PyTorch nn 模块实现
使用 PyTorch 内置组件构建DiT的各个子模块，可运行
Python 3.9+, PyTorch 2.0+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 时间嵌入模块 - 正弦/余弦位置编码风格的时间编码
# ============================================================
class TimeEmbedding(nn.Module):
    """
    将扩散时间步t（标量）编码为高维向量。
    仿照Transformer位置编码的方式，使用不同频率的正弦和余弦函数。
    """
    def __init__(self, emb_size):
        super().__init__()
        # 嵌入维度的一半，分别用于sin和cos
        self.half_emb_size = emb_size // 2
        # 计算指数衰减的频率: exp(-k * ln(10000) / (half_emb_size - 1))
        # 低频捕捉长周期模式，高频捕捉短周期模式
        half_emb = torch.exp(
            torch.arange(self.half_emb_size, dtype=torch.float32) *
            (-math.log(10000.0) / (self.half_emb_size - 1))
        )
        # register_buffer使得half_emb随模型保存/加载，但不参与梯度计算
        self.register_buffer('half_emb', half_emb)

    def forward(self, t):
        """
        Args:
            t: (batch_size,) 时间步张量，每个元素是[0, T-1]的整数
        Returns:
            (batch_size, emb_size) 时间嵌入向量
        """
        t = t.view(t.size(0), 1).float()  # (batch, 1)
        # 广播乘法: (batch, 1) * (1, half_emb_size) -> (batch, half_emb_size)
        half_emb_t = self.half_emb.unsqueeze(0) * t  # (batch, half_emb_size)
        # 拼接sin和cos，得到完整嵌入向量
        embs_t = torch.cat([half_emb_t.sin(), half_emb_t.cos()], dim=-1)
        return embs_t  # (batch, emb_size)


# ============================================================
# 2. SwiGLU 前馈网络 - 门控线性单元变体
# ============================================================
class SwiGLU(nn.Module):
    """
    SwiGLU激活的前馈网络。
    相比标准ReLU-MLP，SwiGLU引入了门控机制，能选择性通过信息。
    公式: output = (xW1 ⊙ SiLU(xW2)) W3
    """
    def __init__(self, hidden_size, expansion_factor=4):
        super().__init__()
        inner_size = hidden_size * expansion_factor
        # 门控分支：负责决定哪些信息通过
        self.w1 = nn.Linear(hidden_size, inner_size, bias=False)
        # 值分支：负责提供实际信息
        self.w2 = nn.Linear(hidden_size, inner_size, bias=False)
        # 输出投影
        self.w3 = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: w2(x) * silu(w1(x)) 然后投影回原维度
        gate = F.silu(self.w1(x))  # 门控信号，范围(0, +∞)，软门控
        value = self.w2(x)          # 线性变换的值
        return self.w3(gate * value)


# ============================================================
# 3. 多查询注意力 (MQA) - 所有头共享KV
# ============================================================
class MultiHeadAttention_MQA(nn.Module):
    """
    多查询注意力：多个Query头共享一组Key和Value。
    MHA的KV缓存需要存储h组KV，MQA只需要1组，显存节省h倍。
    在扩散模型的迭代推理中尤为重要。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个Query头的维度

        # Query: 每个头独立, h * d_k = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Key: 所有头共享一组, 输出维度仅为 d_k
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        # Value: 所有头共享一组
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: 可选的注意力掩码
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Query: 多个头 -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

        # Key/Value: 共享 -> (batch, 1, seq_len, d_k)
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_k)
        k = k.transpose(1, 2)  # (batch, 1, seq_len, d_k)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_k)
        v = v.transpose(1, 2)

        # 计算注意力分数: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        # 加权聚合: (batch, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, v)

        # 合并多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)


# ============================================================
# 4. 调制函数 - adaLN的缩放与偏移
# ============================================================
def modulate(x, gamma, beta):
    """
    Adaptive LayerNorm调制。
    公式: x * (1 + gamma) + beta
    gamma/beta来自条件向量，使归一化行为动态适应时间和标签。
    """
    return x * (1 + gamma) + beta


# ============================================================
# 5. DiTBlock - DiT的核心构建块
# ============================================================
class DiTBlock(nn.Module):
    """
    DiT的基本处理单元，类似Transformer的Encoder Layer。
    内部流程:
        x -> LayerNorm -> modulate(γ1,β1) -> MQA -> ×α1 -> +残差
           -> LayerNorm -> modulate(γ2,β2) -> SwiGLU -> ×α2 -> +残差
    """
    def __init__(self, emb_size=64, head_num=4):
        super().__init__()
        self.emb_size = emb_size

        # adaLN调制网络: 从条件向量(emb_size)生成6个调制参数
        # 参数: γ1(缩放), β1(偏移), α1(门控) — 用于注意力分支
        #       γ2(缩放), β2(偏移), α2(门控) — 用于MLP分支
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, emb_size * 6, bias=True)
        )

        # 注意力分支
        self.layer_norm = nn.RMSNorm(emb_size)
        self.mha = MultiHeadAttention_MQA(d_model=emb_size, num_heads=head_num)

        # MLP分支
        self.mlp = SwiGLU(hidden_size=emb_size)
        self.last_norm = nn.RMSNorm(emb_size)

    def forward(self, x, cond):
        """
        Args:
            x: (batch, num_patches, emb_size) Patch嵌入序列
            cond: (batch, emb_size) 条件向量（时间+标签嵌入之和）
        Returns:
            (batch, num_patches, emb_size)
        """
        # 从条件向量生成6个调制参数
        # 需要unsqueeze因为cond是(batch, emb)，adaln_modulation期望(batch, emb)
        cond_params = self.adaln_modulation(cond)  # (batch, emb_size * 6)
        # 拆分为6组，每组emb_size维
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.split(
            cond_params, self.emb_size, dim=-1
        )
        # 扩展维度以匹配 x 的 (batch, num_patches, emb_size)
        # gamma/beta/alpha: (batch, emb_size) -> (batch, 1, emb_size)
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)

        # ----- 注意力分支 -----
        x_residual = x
        x_norm = self.layer_norm(x)
        x_modulated = modulate(x_norm, gamma1, beta1)
        x_attn = self.mha(x_modulated)
        x = x_residual + alpha1 * x_attn

        # ----- MLP分支 -----
        x_residual = x
        x_norm = self.layer_norm(x)
        x_modulated = modulate(x_norm, gamma2, beta2)
        x_mlp = self.mlp(x_modulated)
        x = self.last_norm(x_residual + alpha2 * x_mlp)

        return x


# ============================================================
# 6. DiT 完整模型
# ============================================================
class DiT(nn.Module):
    """
    Diffusion Transformer 完整模型。
    输入: 加噪图像x_t + 时间步t + 标签y
    输出: 预测的噪声ε
    """
    def __init__(self, img_size=28, patch_size=4, channel=1,
                 emb_size=64, label_num=10, dit_num=3, head=4):
        super().__init__()
        # 图像分块相关参数
        self.patch_size = patch_size
        self.patch_count = img_size // patch_size  # 每行/列的Patch数
        self.channel = channel

        # Patch Embedding: 用Conv2d同时实现切分和初始嵌入
        # 输入: (B, channel, H, W), 输出: (B, channel*P^2, H/P, W/P)
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel * patch_size ** 2,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        # 将每个Patch的原始像素值映射到emb_size维嵌入空间
        self.patch_emb = nn.Linear(channel * patch_size ** 2, emb_size)

        # 可学习的位置编码: 为每个Patch位置提供一个独特的向量
        self.patch_pos_emb = nn.Parameter(
            torch.randn(1, self.patch_count ** 2, emb_size) * 0.02
        )

        # 时间嵌入: 将标量时间步编码为emb_size维向量
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size)
        )

        # 标签嵌入: 将类别标签映射为emb_size维向量
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)

        # 堆叠多个DiTBlock组成深度Transformer
        self.dits = nn.ModuleList([
            DiTBlock(emb_size, head) for _ in range(dit_num)
        ])

        # 最终归一化和输出投影
        self.ln = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)

    def forward(self, x, t, y):
        """
        Args:
            x: (batch, channel, H, W) 加噪图像
            t: (batch,) 时间步
            y: (batch,) 类别标签
        Returns:
            (batch, channel, H, W) 预测的噪声
        """
        # 条件嵌入: 时间 + 标签
        t_emb = self.time_emb(t)   # (batch, emb_size)
        y_emb = self.label_emb(y)  # (batch, emb_size)
        cond = t_emb + y_emb       # (batch, emb_size)

        # Patch Embedding
        x = self.conv(x)  # (batch, channel*P^2, H/P, W/P)
        # 重新排列: (B, C*P^2, H/P, W/P) -> (B, H/P, W/P, C*P^2)
        x = x.permute(0, 2, 3, 1)
        # 展平空间维度: (B, (H/P)*(W/P), C*P^2)
        x = x.reshape(x.size(0), self.patch_count * self.patch_count, -1)
        # 线性嵌入 + 位置编码
        x = self.patch_emb(x) + self.patch_pos_emb  # (B, N, emb_size)

        # 通过所有DiTBlock
        for dit in self.dits:
            x = dit(x, cond)

        # 输出重建
        x = self.ln(x)  # (B, N, emb_size)
        x = self.linear(x)  # (B, N, C*P^2)

        # 还原为图像空间: (B, N, C*P^2) -> (B, C, H, W)
        x = x.view(x.size(0), self.patch_count, self.patch_count,
                   self.channel, self.patch_size, self.patch_size)
        # (B, PC_h, PC_w, C, P_h, P_w) -> (B, C, PC_h, P_h, PC_w, P_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # 合并为完整图像
        x = x.reshape(x.size(0), self.channel,
                      self.patch_count * self.patch_size,
                      self.patch_count * self.patch_size)
        return x


# ============================================================
# 7. 扩散过程的辅助函数
# ============================================================
class DiffusionHelper:
    """
    扩散模型的前向加噪和反向去噪辅助函数。
    """
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device

        # 线性噪声调度: β_t 从 beta_start 线性增长到 beta_end
        betas = torch.linspace(beta_start, beta_end, T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 累积 ᾱ_t

        # 为去噪过程预计算一些有用变量
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), alphas_cumprod[:-1]
        ])
        # 后验方差
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    def forward_add_noise(self, x, t):
        """
        前向加噪: x_0 → x_t
        Args:
            x: (B, C, H, W) 干净图像
            t: (B,) 时间步
        Returns:
            x_t: 加噪后的图像
            noise: 添加的噪声（作为训练目标）
        """
        noise = torch.randn_like(x)
        # 提取对应时间步的累积alpha，调整形状以广播
        alpha_bar = self.sqrt_alphas_cumprod[t].view(x.size(0), 1, 1, 1)
        sigma_bar = self.sqrt_one_minus_alphas_cumprod[t].view(x.size(0), 1, 1, 1)
        x_t = alpha_bar * x + sigma_bar * noise
        return x_t, noise

    @torch.no_grad()
    def backward_denoise(self, model, x, y, show_steps=False):
        """
        反向去噪: x_T → x_0, 使用DiT模型预测并去除噪声。
        Args:
            model: DiT模型实例
            x: (B, C, H, W) 纯噪声图像
            y: (B,) 条件标签
            show_steps: 是否返回中间步骤
        Returns:
            denoised_x: 去噪后的图像
            steps: (可选) 中间步骤列表
        """
        model.eval()
        steps = [x.clone()] if show_steps else None
        batch_size = x.size(0)

        for time in reversed(range(self.T)):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)

            # 模型预测噪声
            pred_noise = model(x, t, y)

            # 计算均值 μ_θ(x_t, t)
            alpha = self.alphas[t].view(batch_size, 1, 1, 1)
            alpha_cumprod = self.alphas_cumprod[t].view(batch_size, 1, 1, 1)
            sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(batch_size, 1, 1, 1)

            mean = sqrt_recip_alpha * (
                x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod) * pred_noise
            )

            # 添加噪声（除了最后一步）
            if time > 0:
                posterior_var = self.posterior_variance[t].view(batch_size, 1, 1, 1)
                z = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * z
            else:
                x = mean

            x = torch.clamp(x, -1.0, 1.0)  # 保持像素值在有效范围

            if show_steps:
                steps.append(x.clone())

        if show_steps:
            return x, steps
        return x


# ============================================================
# 8. 运行示例：演示模型的完整工作流程
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("DiT (Diffusion Transformer) 模型演示")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # ---- 模型初始化 ----
    model = DiT(
        img_size=28,    # MNIST图像大小
        patch_size=4,   # 每4x4像素为一个Patch
        channel=1,      # 灰度图像
        emb_size=64,    # 嵌入维度
        label_num=10,   # 10个数字类别
        dit_num=3,      # 3个DiTBlock
        head=4          # 4个注意力头
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ---- 测试前向传播 ----
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)  # 随机图像
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # 随机时间步
    y = torch.randint(0, 10, (batch_size,)).to(device)   # 随机标签

    noise_pred = model(x, t, y)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {noise_pred.shape}")  # 应与输入形状相同

    # ---- 测试扩散加噪 ----
    dh = DiffusionHelper(T=1000, device=device)
    x_clean = torch.randn(batch_size, 1, 28, 28).to(device) * 0.1
    x_noisy, noise = dh.forward_add_noise(x_clean, t)
    print(f"加噪后形状: {x_noisy.shape}")
    print("前向传播测试通过!")
```

## 8. 手工代码实现

```python
"""
DiT 手工代码实现 - 从零实现核心组件
包含可学习的PosEncoding、自适应LayerNorm、简化的DiTBlock
不依赖nn.Transformer，但保留清晰的模块化结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 手工实现: 正弦-余弦时间编码
# 不使用nn.Module包装，直接展示底层计算逻辑
# ============================================================
def sinusoidal_time_encoding(t, emb_size):
    """
    手工实现正弦-余弦时间编码。

    原理: 使用不同频率的正弦/余弦函数将标量t映射为高维向量。
    低频捕捉全局时间信息，高频捕捉局部时间差异。

    Args:
        t: (batch_size,) 时间步张量
        emb_size: 嵌入维度
    Returns:
        (batch_size, emb_size)
    """
    batch_size = t.shape[0]
    half_dim = emb_size // 2

    # 计算频率: ω_k = 1 / (10000^(2k/d))
    # 对数空间均匀采样
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    ).to(t.device)

    # t * ω_k, shape: (batch, half_dim)
    args = t.float().unsqueeze(1) * freq.unsqueeze(0)

    # 拼接sin和cos
    encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return encoding


# ============================================================
# 手工实现: 缩放点积注意力
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None, dropout=0.0):
    """
    手工实现缩放点积注意力。

    公式: Attention(Q,K,V) = softmax(QK^T / √d_k) · V

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: 可选的注意力掩码
        dropout: dropout概率
    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k) 注意力权重
    """
    d_k = Q.size(-1)

    # QK^T: 计算Query和Key的相似度
    # Q: (B, H, Lq, dk) × K^T: (B, H, dk, Lk) → scores: (B, H, Lq, Lk)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 缩放: 除以√d_k防止点积值过大
    # 当d_k很大时，点积方差为d_k，softmax后梯度接近0
    scores = scores / math.sqrt(d_k)

    # 应用掩码: 将需要忽略的位置设为极小值
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax归一化: 每行的权重和为1
    attn_weights = F.softmax(scores, dim=-1)

    # Dropout: 随机丢弃部分注意力连接，正则化
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    # 加权求和: attn_weights(B,H,Lq,Lk) × V(B,H,Lk,dv) → output(B,H,Lq,dv)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ============================================================
# 手工实现: 自适应层归一化 (adaLN)
# ============================================================
class AdaLayerNorm(nn.Module):
    """
    自适应层归一化:
    不是学习固定的γ和β（标准LayerNorm的做法），
    而是根据条件向量cond动态生成γ和β。

    这使得每个时间步和每个条件类别的归一化行为不同，
    条件信息得以深度嵌入网络每一层。
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        # 标准LayerNorm的基础参数（学习全局行为）
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # 从条件向量生成自适应缩放和偏移参数
        self.ada_scale = nn.Linear(cond_dim, dim)
        self.ada_shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        """
        Args:
            x: (batch, seq_len, dim)
            cond: (batch, cond_dim)
        Returns:
            (batch, seq_len, dim)
        """
        # 先做标准归一化
        x = self.norm(x)
        # 从条件生成自适应参数
        scale = self.ada_scale(cond).unsqueeze(1)  # (batch, 1, dim)
        shift = self.ada_shift(cond).unsqueeze(1)  # (batch, 1, dim)
        # 应用自适应缩放和偏移
        return x * (1 + scale) + shift


# ============================================================
# 手工实现: 简化版DiT（核心逻辑完整）
# ============================================================
class SimpleDiT(nn.Module):
    """
    从零实现的简化版DiT模型。
    保留DiT的核心设计理念:
    - Patch Embedding + 位置编码
    - 时间编码 + 标签编码的条件机制
    - adaLN调制
    - 自注意力 + 前馈网络
    - 重复堆叠的Transformer块
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=1,
                 emb_dim=64, num_labels=10, depth=3, num_heads=4):
        super().__init__()
        assert img_size % patch_size == 0, "img_size必须能被patch_size整除"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.emb_dim = emb_dim

        # ----- Patch Embedding -----
        # 每个Patch的扁平化像素数
        patch_dim = in_channels * patch_size ** 2
        self.patch_to_embed = nn.Linear(patch_dim, emb_dim)
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim) * 0.02)

        # ----- 时间编码 -----
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        # ----- 标签编码 -----
        self.label_embed = nn.Embedding(num_labels, emb_dim)

        # ----- DiT块（简化为两层调制） -----
        self.blocks = nn.ModuleList([
            DiTBlockSimple(emb_dim, num_heads, emb_dim) for _ in range(depth)
        ])

        # ----- 输出层 -----
        self.final_norm = nn.LayerNorm(emb_dim)
        self.to_pixels = nn.Linear(emb_dim, patch_dim)

    def patchify(self, x):
        """
        将图像切分为Patch序列。
        (B, C, H, W) → (B, N, C*P*P)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0

        # 使用unfold提取每个Patch，然后展平
        # reshape: (B, C, H, W) → (B, C, H/P, P, W/P, P)
        x = x.reshape(B, C, H // P, P, W // P, P)
        # permute and reshape: → (B, H/P, W/P, C*P*P)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // P) * (W // P), C * P * P)
        return x

    def unpatchify(self, x):
        """
        将Patch序列还原为图像。
        (B, N, C*P*P) → (B, C, H, W)
        """
        B = x.shape[0]
        C = self.in_channels
        P = self.patch_size
        H_div_P = self.img_size // P

        # (B, N, C*P*P) → (B, H/P, W/P, C, P, P)
        x = x.reshape(B, H_div_P, H_div_P, C, P, P)
        # → (B, C, H/P, P, W/P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # → (B, C, H, W)
        x = x.reshape(B, C, self.img_size, self.img_size)
        return x

    def forward(self, x, t, y):
        """
        Args:
            x: (B, C, H, W) 加噪图像
            t: (B,) 时间步
            y: (B,) 标签
        Returns:
            (B, C, H, W) 预测噪声
        """
        B = x.shape[0]

        # 1. Patch嵌入 + 位置编码
        x = self.patchify(x)                         # (B, N, C*P*P)
        x = self.patch_to_embed(x)                   # (B, N, emb_dim)
        x = x + self.pos_embed                       # 添加位置信息

        # 2. 构建条件向量: 时间 + 标签
        t_enc = sinusoidal_time_encoding(t, self.emb_dim).to(x.device)
        t_emb = self.time_mlp(t_enc)                 # (B, emb_dim)
        y_emb = self.label_embed(y)                  # (B, emb_dim)
        cond = t_emb + y_emb                         # (B, emb_dim) 简单相加融合

        # 3. 通过DiT块序列
        for block in self.blocks:
            x = block(x, cond)

        # 4. 输出去像素空间
        x = self.final_norm(x)                       # (B, N, emb_dim)
        x = self.to_pixels(x)                        # (B, N, C*P*P)
        x = self.unpatchify(x)                       # (B, C, H, W)

        return x


class DiTBlockSimple(nn.Module):
    """
    简化版DiTBlock。
    保留了两次调制-处理-残差的模式，使用标准LayerNorm + 自适应调制。
    """
    def __init__(self, dim, num_heads, cond_dim):
        super().__init__()
        self.dim = dim

        # adaLN调制: 从条件向量生成6个参数
        self.adaln_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 6)
        )

        # 注意力: 标准多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            batch_first=True
        )

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cond):
        """
        Args:
            x: (B, N, dim) 序列表示
            cond: (B, dim) 条件向量
        Returns:
            (B, N, dim)
        """
        # 生成调制参数
        params = self.adaln_mod(cond)  # (B, dim*6)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.split(
            params, self.dim, dim=-1
        )

        def modulate(x, gamma, beta):
            return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # ---- 注意力分支 ----
        residual = x
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, gamma1, beta1)
        attn_out, _ = self.attention(x_mod, x_mod, x_mod)
        x = residual + alpha1.unsqueeze(1) * attn_out

        # ---- MLP分支 ----
        residual = x
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, gamma2, beta2)
        mlp_out = self.mlp(x_mod)
        x = residual + alpha2.unsqueeze(1) * mlp_out

        return x


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("手工实现 SimpleDiT 测试")
    print("=" * 60)

    torch.manual_seed(42)

    # 初始化模型
    model = SimpleDiT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        emb_dim=128,
        num_labels=10,
        depth=2,
        num_heads=4
    )
    params_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params_count:,}")

    # 测试输入
    batch_size = 2
    x_test = torch.randn(batch_size, 3, 32, 32)
    t_test = torch.randint(0, 1000, (batch_size,))
    y_test = torch.randint(0, 10, (batch_size,))

    # 前向传播
    training = True  # 全局变量用于attention dropout
    output = model(x_test, t_test, y_test)
    print(f"输入形状:  {x_test.shape}")
    print(f"输出形状:  {output.shape}")
    assert output.shape == x_test.shape, "输出形状应与输入一致!"

    # 时间编码测试
    t_enc = sinusoidal_time_encoding(t_test, 128)
    print(f"时间编码形状: {t_enc.shape}")

    print("\n所有测试通过! SimpleDiT手工实现正确。")
```

## 9. 可视化与结果理解

```python
"""
DiT 可视化 - 展示模型组件和扩散过程
包括: 时间编码热力图、注意力权重可视化、扩散加噪/去噪过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 使用前面定义的正弦-余弦时间编码函数
def _sinusoidal_time_encoding(t_values, emb_size):
    half_dim = emb_size // 2
    freq = torch.exp(-math.log(10000.0) *
                     torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
    args = t_values.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def main():
    # 设置matplotlib中文支持（如不支持英文显示也可以）
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ============================================================
    # 图1: 时间编码热力图 - 展示不同时间步的嵌入向量
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    # 采样1000个时间步中均匀取50个
    t_steps = torch.linspace(0, 999, 50, dtype=torch.long)
    encoding = _sinusoidal_time_encoding(t_steps, 64)

    im = ax1.imshow(encoding.numpy(), aspect='auto', cmap='RdBu_r',
                    extent=[0, 64, 999, 0])
    ax1.set_xlabel('Embedding Dimension (d=64)')
    ax1.set_ylabel('Time Step t')
    ax1.set_title('Time Encoding Heatmap\n(Sinusoidal, Red=Positive, Blue=Negative)')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # ============================================================
    # 图2: 噪声调度曲线 - 展示扩散过程的关键参数
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    T = 1000
    betas = torch.linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    t_range = np.arange(T)
    ax2.plot(t_range, betas.numpy(), label=r'$\beta_t$ (Noise level)', linewidth=1.5, alpha=0.7)
    ax2.plot(t_range, alphas_cumprod.numpy(),
             label=r'$\bar{\alpha}_t$ (Signal retained)', linewidth=1.5)
    ax2.set_xlabel('Time Step t')
    ax2.set_ylabel('Value')
    ax2.set_title('Diffusion Noise Schedule\n(Signal Decay & Noise Growth)')
    ax2.legend(loc='center right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T)

    # ============================================================
    # 图3: 一步加噪的影响 - 不同t对相同图像的加噪效果
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    # 使用简单的2D图案（圆+方块）来演示
    canvas = np.zeros((32, 32))
    # 画一个圆
    cy, cx = 12, 12
    y, x = np.ogrid[:32, :32]
    mask_circle = (x - cx)**2 + (y - cy)**2 <= 49
    canvas[mask_circle] = 1.0
    # 画一个方块
    canvas[20:28, 20:28] = 1.0

    img = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 展示原图和几个关键扩散步骤
    t_demo = [0, 50, 200, 500, 900]
    alphas_cp = alphas_cumprod.numpy()
    images_to_show = []
    labels = ['t=0 (Clean)']

    for i, ti in enumerate(t_demo):
        if ti == 0:
            noisy = img.clone()
            labels.append(f't={ti}')
        else:
            noise = torch.randn_like(img) * 0.5  # scale down for visibility
            alpha_t = alphas_cp[ti]
            noisy = np.sqrt(alpha_t) * img.numpy() + np.sqrt(1 - alpha_t) * noise.numpy()
        if ti > 0:
            images_to_show.append(np.clip(noisy[0, 0], -1, 1))
            labels.append(f't={ti}')

    for i, (img_data, label) in enumerate(zip(images_to_show, labels[1:])):
        ax = ax3.inset_axes([0.05 + i * 0.19, 0.05, 0.18, 0.9])
        ax.imshow(img_data, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(label, fontsize=8)
        ax.axis('off')

    ax3.set_title('Progressive Noise Addition\n(Same Image, Different t)', fontsize=10)
    ax3.axis('off')

    # ============================================================
    # 图4: adaLN调制参数可视化 - 不同时间步产生的调制参数
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])
    # 模拟adaln_modulation网络
    class AdaLNVis(nn.Module):
        def __init__(self, emb_size=64):
            super().__init__()
            self.net = nn.Sequential(nn.SiLU(), nn.Linear(emb_size, emb_size * 6))
        def forward(self, cond):
            return self.net(cond)

    adaln = AdaLNVis(64)
    # 对10个均匀分布的时间步生成调制参数
    t_vis = torch.linspace(0, 999, 10, dtype=torch.long)
    t_enc_vis = _sinusoidal_time_encoding(t_vis, 64)
    params = adaln(t_enc_vis).detach().numpy()  # (10, 384)

    # 只看gamma1和gamma2的平均绝对值（表示调制强度）
    gamma1_mean = np.abs(params[:, :64]).mean(axis=1)
    gamma2_mean = np.abs(params[:, 64:128]).mean(axis=1)

    ax4.plot(t_vis.numpy(), gamma1_mean, 'o-', label=r'$|\gamma_1|$ (Attention branch)',
             markersize=6, linewidth=1.5)
    ax4.plot(t_vis.numpy(), gamma2_mean, 's--', label=r'$|\gamma_2|$ (MLP branch)',
             markersize=6, linewidth=1.5)
    ax4.set_xlabel('Time Step t')
    ax4.set_ylabel('Mean Absolute Modulation Strength')
    ax4.set_title('adaLN Modulation Strength vs. Time Step\n(Larger=Stronger Conditioning)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # 图5: 模拟的去噪过程 - 从噪声到清晰
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])
    # 模拟逐步去噪: 从t=999的高噪声线性插值到t=0的清晰图像
    n_frames = 8
   yr = np.zeros((n_frames, 32, 32))
    # 构造一个"目标"图案
    target = np.zeros((32, 32))
    target[8:24, 8:24] = 0.8
    target[12:20, 12:20] = 1.0

    for i in range(n_frames):
        t_equiv = 999 - i * (999 // (n_frames - 1))
        noise_level = t_equiv / 999.0
        yr[i] = (1 - noise_level) * target + noise_level * np.random.randn(32, 32) * 0.3
        yr[i] = np.clip(yr[i], -1, 1)

    for i in range(n_frames):
        ax_inset = ax5.inset_axes(
            [0.05 + (i % 4) * 0.24, 0.55 - (i // 4) * 0.48, 0.22, 0.42]
        )
        ax_inset.imshow(yr[i], cmap='gray', vmin=-1, vmax=1)
        t_equiv = 999 - i * (999 // (n_frames - 1))
        ax_inset.set_title(f't={t_equiv}', fontsize=7)
        ax_inset.axis('off')
    ax5.set_title('Denoising Process Simulation\n(From Pure Noise to Clean Image)',
                  fontsize=10)
    ax5.axis('off')

    # ============================================================
    # 图6: 不同噪声水平下的信噪比曲线
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])
    snr_values = alphas_cumprod.numpy() / (1 - alphas_cumprod.numpy() + 1e-8)
    snr_db = 10 * np.log10(snr_values)

    ax6.plot(t_range, snr_db, linewidth=1.5, color='darkgreen')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='SNR=0 dB')
    ax6.fill_between(t_range, snr_db, -50, alpha=0.2, color='green')
    ax6.set_xlabel('Time Step t')
    ax6.set_ylabel('SNR (dB)')
    ax6.set_title('Signal-to-Noise Ratio over Diffusion Steps\n(Higher=Cleaner Image)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, T)

    # ============================================================
    # 图7: 注意力模式对比 - 不同头关注不同区域
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 0])
    # 模拟4个头的注意力模式
    n_patches = 16  # 4x4的Patch网格
    # 构造不同的注意力模式
    attn_patterns = [
        np.eye(n_patches) * 0.8 + 0.2 / n_patches,  # Head1: 局部/自身关注
        np.ones((n_patches, n_patches)) / n_patches, # Head2: 均匀关注
    ]
    # Head3: 第一行特别关注最后一行
    attn3 = np.ones((n_patches, n_patches)) / n_patches * 0.1
    attn3[0:4, 12:16] = 0.8 / 4
    attn_patterns.append(attn3)
    # Head4: 对角线带状关注
    attn4 = np.zeros((n_patches, n_patches))
    for i in range(n_patches):
        for j in range(max(0, i-2), min(n_patches, i+3)):
            attn4[i, j] = 1.0 / min(5, 2*i+3)
    attn_patterns.append(attn4)

    titles = ['Head 1: Local/Self', 'Head 2: Uniform',
              'Head 3: Cross-region', 'Head 4: Banded']

    for i, (pattern, title) in enumerate(zip(attn_patterns, titles)):
        ax_i = ax7.inset_axes(
            [0.02 + (i % 2) * 0.5, 0.52 - (i // 2) * 0.5, 0.46, 0.44]
        )
        im_i = ax_i.imshow(pattern, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
        ax_i.set_title(title, fontsize=7)
        ax_i.set_xlabel('Key Position', fontsize=6)
        ax_i.set_ylabel('Query Position', fontsize=6)
        ax_i.tick_params(labelsize=5)
    ax7.set_title('Multi-Head Attention Pattern Examples\n(Different Heads Learn Different Dependencies)',
                  fontsize=10)
    ax7.axis('off')

    # ============================================================
    # 图8: 模型规模与生成质量关系（示意）
    # ============================================================
    ax8 = fig.add_subplot(gs[2, 1])
    # 模拟数据: DiT论文中的FID随模型规模变化趋势（示意）
    model_sizes = [1, 4, 16, 64, 256, 1024]  # 百万参数
    fid_scores = [45, 30, 18, 10, 5.5, 2.8]  # FID越低越好（示意数据）

    ax8.plot(model_sizes, fid_scores, 'o-', color='darkblue',
             markersize=10, linewidth=2, markerfacecolor='lightblue')
    ax8.set_xlabel('Model Parameters (Millions)')
    ax8.set_ylabel('FID Score (Lower=Better)')
    ax8.set_xscale('log')
    ax8.set_title('DiT Scaling Behavior\n(Quality Improves with Model Size)')
    ax8.grid(True, alpha=0.3)
    # 添加标注
    ax8.annotate('Small', xy=(1, 45), fontsize=8, ha='center',
                 xytext=(1, 48), arrowprops=dict(arrowstyle='->', alpha=0.5))
    ax8.annotate('Sora-scale', xy=(1024, 2.8), fontsize=8, ha='center',
                 xytext=(300, 8), arrowprops=dict(arrowstyle='->', alpha=0.5))

    # ============================================================
    # 图9: DiT vs UNet架构对比示意
    # ============================================================
    ax9 = fig.add_subplot(gs[2, 2])
    # 左侧: UNet架构示意
    unet_depths = [64, 128, 256, 512, 256, 128, 64]
    unet_positions = [0, 1, 2, 3, 2, 1, 0]
    ax9.fill_between([0, 1, 2, 3, 2, 1, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [6, 5, 4, 3, 4, 5, 5.5],
                     step='mid', alpha=0.3, color='blue', label='UNet (Hourglass)')
    ax9.plot([0, 1, 2, 3, 2, 1, 0.5], [6, 5, 4, 3, 4, 5, 5.5],
             'o-', color='blue', linewidth=2, markersize=8)

    # 右侧: DiT架构示意（扁平）
    dit_depths = [64, 64, 64, 64, 64, 64]  # 所有层维度相同
    ax9.fill_between([3.5, 4, 5, 6, 7, 8, 8.5], [3, 3, 3, 3, 3, 3, 3],
                     [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
                     alpha=0.3, color='red', label='DiT (Flat Transformer)')
    ax9.plot([3.5, 4, 5, 6, 7, 8, 8.5], [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
             's-', color='red', linewidth=2, markersize=8)

    ax9.set_ylim(0.5, 6.5)
    ax9.set_xlim(-0.5, 9)
    ax9.set_xlabel('Layer Depth')
    ax9.set_ylabel('Feature Dimension')
    ax9.set_title('Architecture Comparison: UNet vs DiT\n(DiT: Flat Transformer, UNet: Hourglass)')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.set_yticks([1, 2, 3, 4, 5, 6])
    ax9.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DiT (Diffusion Transformer) - Comprehensive Visualization',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('dit_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化图表已保存为 'dit_visualization.png'")
    plt.show()


if __name__ == '__main__':
    main()
```

## 10. 模型评估

### 评估指标

#### 1. FID (Frechet Inception Distance)
FID是评估生成图像质量最常用的指标。它计算生成图像和真实图像在Inception网络特征空间中的Frechet距离（Wasserstein-2距离）。

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

其中 $\mu_r, \Sigma_r$ 和 $\mu_g, \Sigma_g$ 分别是真实图像和生成图像在Inception特征空间的均值和协方差。

- **FID 越低越好**：< 10 为优秀，10-30 为良好，> 50 为较差
- DiT-XL/2 在 ImageNet 256x256 上达到 FID ~ 2.27（state-of-the-art）
- **缺点**：对数据量敏感、对不同分辨率需要调整

#### 2. IS (Inception Score)
评估生成图像的清晰度和多样性。

$$\text{IS} = \exp(\mathbb{E}_x [D_{KL}(p(y|x) \| p(y))])$$

- **IS 越高越好**：反映每张图分类置信度高（清晰）且类别分布均匀（多样）
- **局限**：不适合非ImageNet类别的数据集

#### 3. LPIPS (Learned Perceptual Image Patch Similarity)
衡量生成图像与参考图像的感知相似度。对于条件生成任务很重要。

#### 4. sFID (Spatial FID)
在Inception网络的不同空间层级计算FID，获得多尺度质量评估。

#### 5. Precision & Recall
- **Precision**：生成图像中有多少是"真实"的（质量）
- **Recall**：真实数据分布中有多少被生成模型覆盖（多样性）

### 训练监控指标

| 指标 | 含义 | 健康值 |
|------|------|--------|
| Loss (L1) | 噪声预测误差 | 持续下降至稳定 |
| Gradient Norm | 梯度大小 | 10^-3~10^-2 |
| Perplexity (条件生成) | 条件控制的确定程度 | 随epoch下降 |

### 评估注意事项

1. **采样步数**：评估时通常使用全部T步（或DDIM的少量步数），需保持一致
2. **Guidance Scale**：Classifier-Free Guidance的scale参数影响FID和IS的平衡
3. **评估样本量**：至少50000张生成图像以获得统计稳定的FID
4. **随机种子**：固定随机种子以保证结果可复现

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| Patch数量与位置编码不匹配 | 前向传播报错，维度无法相加 | img_size不能被patch_size整除 | 确保img_size % patch_size == 0 |
| 时间嵌入维度不对 | 时间编码的维度错误 | emb_size为奇数时half_emb维度不等于emb_size//2的预期 | emb_size必须为偶数；或使用 `emb_size - emb_size // 2` |
| 加噪图像值域不匹配 | 训练loss很高且不收敛 | 图像像素[0,1]而噪声是N(0,I)，值域不匹配 | 将图像归一化到[-1,1]：`x = x * 2 - 1` |
| adaLN条件注入忘加SiLU | 调制效果差，图像模糊 | 没有非线性激活，条件向量表达能力不足 | 确保adaln_modulation包含SiLU()激活 |
| 输出维度不是(C,H,W)格式 | 损失计算时维度错误 | unpatchify时维度排序或reshape有误 | 逐行验证permute和reshape的维度变换 |
| 去噪生成的图像全部偏灰 | 生成的图像缺乏对比度 | clamp范围设为[0,1]但模型输出在[-1,1] | clamp设为[-1,1]；保存前再线性映射到[0,1] |
| MQA中K/V维度被广播错误 | 注意力输出维度不对 | n头Q对1组KV时，KV的heads维度为1，需要broadcast | 确保K/V在heads维度为1，matmul会自动广播 |
| 训练到后期loss反增 | loss下降后又回升 | 学习率过大导致震荡，或过拟合 | 使用余弦退火学习率调度器；增大数据集或数据增强 |
| 生成图像有棋盘格伪影 | 图像出现规则的格子状伪影 | Patch边界处信息不连续 | 检查Conv2d的stride和kernel_size一致；考虑patch_overlap |
| 条件生成不受标签控制 | 改变标签生成的图像不变 | 标签嵌入被忽略或权重太小 | 检查label_emb是否正确传入；增大α参数范围 |
| 去噪步数影响巨大 | 不同采样步数效果差异很大 | T设置或去噪方差公式有误 | T通常设为1000；验证posterior_variance公式 |
| 批大小过小导致不稳定 | loss曲线波动剧烈 | batch过小导致梯度估计方差大 | 使用梯度累积（gradient accumulation）模拟大batch |

## 12. 学习总结

### 核心思想回顾

DiT（Diffusion Transformer）的核心创新在于用Transformer架构替换扩散模型中的UNet噪声预测网络。这一替换看似简单，实则是架构范式的转变：

1. **Patch化处理**：将图像切分为固定大小的Patch（类似ViT），每个Patch作为一个序列token
2. **全局注意力**：Transformer的自注意力让每个Patch都能在每一步去噪时利用"全局视野"，这是UNet的局部卷积无法做到的
3. **adaLN条件注入**：通过自适应层归一化，将时间步和标签信息深度注入Transformer的每一层，实现精细的条件控制
4. **Scaling Law**：Transformer架构遵循良好的扩展规律，增加参数量能持续提升生成质量

### 与前序/相关算法的联系

- **DDPM是基础**：DiT继承了DDPM的加噪/去噪框架，只替换了噪声预测模型
- **ViT是模板**：DiT的Patch Embedding直接来自Vision Transformer
- **MQA是优化**：使用多查询注意力减少推理时的KV缓存开销
- **Sora是扩展**：OpenAI的Sora将DiT架构从2D图像扩展到3D时空视频生成

### 后续学习方向

- Classifier-Free Guidance：无条件/条件混合采样，提升可控性和质量的平衡
- DDIM（去噪扩散隐式模型）：减少采样步数，加速推理
- Latent Diffusion（潜在扩散模型）：在压缩的潜在空间做扩散（Stable Diffusion的方案）
- DiT的视频/3D变体：Sora、WALT等
- Flash Attention集成：加速DiT的训练和推理

## 13. 练习题与思考题

### 基础题1：参数计算

一个DiT模型参数如下：img_size=32, patch_size=4, channel=3, emb_size=256, label_num=100, dit_num=12, head=8。计算：
- 模型有多少个Patch？
- Patch Embedding层（patch_to_embed）的参数量？
- 位置编码的参数量？

**参考答案**：
- Patch数 = (32/4) x (32/4) = 8 x 8 = 64个
- patch_dim = 3 x 4 x 4 = 48; patch_to_embed: 48 x 256 = 12,288（不含bias则为12288，含bias加256）
- 位置编码: 1 x 64 x 256 = 16,384个参数（nn.Parameter）
- 共: 12,288 + 256 + 16,384 = 28,928（仅输入嵌入部分）

### 基础题2：代码补全

以下代码实现对单张图像的加噪过程补全：

```python
def add_noise_one_step(x_0, t, alphas_cumprod):
    """
    x_0: (1, C, H, W) 干净图像, 归一化到[-1,1]
    t: 标量, 时间步
    alphas_cumprod: (T,) 累积alpha
    """
    # 补全下面代码
    # ...
    return x_t, noise
```

**参考答案**：
```python
def add_noise_one_step(x_0, t, alphas_cumprod):
    noise = torch.randn_like(x_0)
    alpha_bar = alphas_cumprod[t]
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise
```

### 进阶题：分析与设计

DiT使用adaLN进行条件注入。另一个常见方案是直接在输入层将条件向量拼接到Patch嵌入上（如 "class token"拼接）。请对比两种方案的优劣。

**参考答案**：

| 对比维度 | adaLN（DiT方案） | 输入拼接方案 |
|----------|-----------------|-------------|
| 条件注入深度 | 每层都注入，深度影响 | 仅输入层，后续层间接获取 |
| 参数量 | 每层增加6*dmodel参数 | 仅输入层增加少量参数 |
| 条件控制力度 | 强，可精细调节每层行为 | 弱，容易在深层被稀释 |
| 训练难度 | 略高，需平衡各层调制 | 较低，结构简单 |
| 适用场景 | 条件与输出关系复杂的任务 | 简单条件映射任务 |

- adaLN胜在强控制力，适合DiT这种条件（时间步）与输出（噪声模式）强相关的场景
- 输入拼接适合条件信息的语义层次较高（如文本描述）的场景，此时可配合交叉注意力使用

### 开放思考题

DiT论文发现：增加Transformer深度（dit_num）和增加嵌入维度（emb_size）对FID的改善效果不同。你认为更大的深度和更大的宽度各自的优势是什么？在实际部署中应该如何权衡？

**参考思路**：
- **深度优势**：更深的网络能学习更抽象的层次化特征，对复杂分布建模更有优势；类似于CNN中深层特征更有语义含义
- **宽度优势**：更宽的嵌入能提供更大的表示容量，每个token能承载更丰富的信息
- **权衡考量**：
  1. 深度增加需要更多显存（激活值存储），宽度增加需要更多计算（矩阵乘法）
  2. 深度有利全局建模，宽度有利局部模式
  3. 部署时需根据硬件限制：GPU显存大则优先深度，计算能力强则优先宽度
  4. DiT论文建议均衡增长（类似Transformer的scaling law）

## 14. 学习路径建议

### 前置算法
- **扩散模型（DDPM）**：理解加噪/去噪框架、噪声预测目标
- **Vision Transformer（ViT）**：理解图像Patch Embedding和位置编码
- **自注意力机制**：理解QKV计算、缩放、softmax
- **多头注意力（MHA）**：理解多头并行的特征提取

### 平行算法
- **Stable Diffusion**：在潜在空间做扩散，配合文本编码器
- **DDIM**：确定性采样，减少去噪步数
- **U-ViT**：将UNet和ViT结合的扩散方案
- **MDT (Masked Diffusion Transformer)**：引入掩码自编码的DiT变体

### 进阶算法
- **Sora**：OpenAI的视频生成模型，DiT的3D时空扩展
- **Flux**：Black Forest Labs的大规模DiT文生图模型
- **PixArt-α**：高效DiT训练策略
- **SD3 (Stable Diffusion 3)**：基于MMDiT的多模态DiT
- **Classifier-Free Guidance**：无条件/条件混合采样

### 推荐资源
1. **论文**：《Scalable Diffusion Models with Transformers》（Peebles & Xie, 2023）—— DiT原论文，详细讨论架构设计和Scaling行为
2. **代码**：Meta官方DiT开源实现（GitHub: facebookresearch/DiT）
3. **博客**：Understanding Diffusion Models: A Unified Perspective (Calvin Luo)
4. **视频**：Yannic Kilcher的DiT论文解读（YouTube）
5. **进阶**：Sora技术报告 —— 了解DiT如何扩展到视频生成
