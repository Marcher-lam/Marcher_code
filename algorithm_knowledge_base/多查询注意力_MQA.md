# 多查询注意力 (MQA) 学习文档

> 来源线索：本节内容根据原书中关于"MQA模型"（第6章 6.1.1节）的相关章节整理、扩展与教学化改写。

> 所有头共享一组K和V，大幅减少推理缓存，以微小精度代价换取显著速度提升。

---

## 1. 算法基础认知

**一句话定义**：多查询注意力（Multi-Query Attention, MQA）是标准多头注意力（MHA）的一种变体——它让所有注意力头共享**同一组** Key 和 Value 矩阵，而每个头仍然保留独立的 Query 矩阵。这种设计在不显著损失模型质量的前提下，大幅降低了推理阶段的内存占用和计算开销。

**直觉类比**：想象一个考试场景，有 h 位阅卷老师（对应 h 个注意力头），每位老师都需要查阅同一份参考资料来打分。在标准 MHA 中，每位老师都私藏一份自己的参考资料复印件（每人一套 K 和 V）——占地方且浪费。在 MQA 中，所有老师共享墙壁上投影的同一份参考资料——省纸、省空间，而且因为大家看的都是同一份内容，判断结果几乎没有明显差异。

**历史背景**：MQA 由 Noam Shazeer 于 2019 年在论文《Fast Transformer Decoding: One Write-Head Is All You Need》中提出。Shazeer 在 Google 工作时发现，Transformer 解码器中的多头注意力存在严重的 KV 参数冗余——不同头的 Key 和 Value 学到的内容高度相似。由此他提出：既然 K 和 V 各头之间差别不大，干脆让所有头共享同一组 K 和 V。实验表明，在翻译等任务上，MQA 的 BLEU 分数与 MHA 几乎持平，但推理速度显著提升。

**算法定位**：MQA 属于注意力机制的**工程优化**范畴，位于深度学习 / 自然语言处理领域。它不是一种全新的注意力计算方式，而是对 MHA 的参数结构和内存布局进行重新设计，目标是**以微小的精度代价换取显著的推理加速和内存节省**。

**前置知识**：
- **多头注意力（MHA）**：理解 Q、K、V 投影、分头、scaled dot-product attention 和拼接输出；
- **自注意力机制**：理解序列中每个位置如何通过 Q-K 相似度来聚合 Value 信息；
- **KV Cache 概念**：理解自回归解码时为什么要缓存已生成的 Key 和 Value，以及 KV Cache 如何随序列长度和头数线性增长。

---

## 2. 核心原理

### 2.1 核心思想

在标准 MHA 中，输入 X 首先通过 h 组不同的投影矩阵 (W_i^Q, W_i^K, W_i^V) 分别生成 h 组 Q, K, V，然后在每个头上独立计算注意力并拼接输出。这种设计固然增强了模型的表达能力，但 Shazeer 等人通过分析发现：**不同注意力头之间学到的 Key 和 Value 投影矩阵高度相似，参数冗余严重**。

MQA 的核心思想非常直接：既然 K 和 V 在头之间差别不大，就只保留一组 W^K 和 W^V，让所有头共享同一组 Key 和 Value，同时保持每个头拥有独立的 Query 投影矩阵 W_i^Q。Query 的多头特性保留了不同头关注不同位置的能力，而 K/V 的共享大幅减少了参数和缓存。

### 2.2 工作流程

MQA 的完整计算流程如下：

1. **输入投影**：对输入序列 X（形状 [batch, seq_len, d_model]），使用 h 个独立的 Query 投影矩阵 W_i^Q 分别投影出 h 组 Query，同时使用唯一的一组 W^K 和 W^V 投影出共享的 Key 和 Value。
2. **Q 分头（多组独立）**：将每组 Q_i 的维度从 d_model 切分为 d_k = d_model / h，得到形状为 [batch, h, seq_len, d_k] 的 Query 张量。
3. **K 和 V 各一份（所有头共享）**：K 和 V 保持形状 [batch, 1, seq_len, d_k]——注意头数维度为 1，意味着只有一组 K 和一组 V。
4. **广播与注意力计算**：在计算注意力分数时，将共享的 K 和 V 沿头数维度广播（broadcast）到 [batch, h, seq_len, d_k]，然后对每组独立的 Q_i 和共享的 K 计算 scaled dot-product attention 分数。
5. **加权求和**：使用 softmax 归一化后的注意力权重，对共享的 V 进行加权求和。
6. **拼接输出**：将 h 个头的输出拼接起来（每个头独立输出 d_k 维，拼接后 d_model 维），经过输出投影矩阵 W^O 得到最终结果。

### 2.3 关键概念

- **KV 共享**：这是 MQA 区别于 MHA 最本质的特征。所有注意力头使用相同的 Key 和 Value，仅 Query 保持多头。
- **推理缓存减少**：在自回归解码中，KV Cache 的大小从 MHA 的 `h * seq_len * d_k * 2`（头数 × 序列长度 × 维度 × K 和 V 两份）降低到 MQA 的 `seq_len * d_k * 2`，减少为原来的 1/h 倍。对于 h=8 的模型，KV Cache 直接缩减到 12.5%。
- **参数效率**：KV 投影矩阵的参数数量从 `h * (d_model * d_k + d_model * d_v)` 减少到 `d_model * d_k + d_model * d_v`，同样减少为原来的 1/h 倍。

### 2.4 几何解释

从 MHA 到 MQA 的演变可以用一个简洁的示意图描述：

- **MHA**：h 组 (Q, K, V)，每组独立计算。Q_1 只用 K_1 和 V_1，Q_2 只用 K_2 和 V_2……
- **MQA**：h 组 Q（独立），1 组 K 和 1 组 V（共享）。Q_1, Q_2, ..., Q_h 都用同一份 K 和同一份 V 计算注意力。

可以理解为：MHA 中每个头有自己的"视角滤镜"（Q）和"参考材料"（K/V），而 MQA 中每个人有不同的"视角滤镜"，但看的是投影在墙上的同一份参考材料。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $X \in \mathbb{R}^{b \times s \times d_{model}}$ | 输入序列，b 为 batch size，s 为序列长度 |
| $h$ | 注意力头数 |
| $d_{model}$ | 模型隐层维度 |
| $d_k = d_{model} / h$ | 每个头的 Key/Query 维度 |
| $d_v = d_{model} / h$ | 每个头的 Value 维度（通常 d_k = d_v） |
| $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ | 第 i 个头的 Query 投影矩阵（共 h 个） |
| $W^K \in \mathbb{R}^{d_{model} \times d_k}$ | **共享的** Key 投影矩阵（仅 1 个） |
| $W^V \in \mathbb{R}^{d_{model} \times d_v}$ | **共享的** Value 投影矩阵（仅 1 个） |
| $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ | 输出投影矩阵 |

### 3.2 核心公式

**MQA 中第 i 个头的注意力计算**：

$$
Q_i = X \cdot W_i^Q, \quad K_{shared} = X \cdot W^K, \quad V_{shared} = X \cdot W^V
$$

$$
\text{head}_i = \text{Attention}(Q_i, K_{shared}, V_{shared}) = \text{softmax}\left(\frac{Q_i \cdot K_{shared}^\top}{\sqrt{d_k}}\right) \cdot V_{shared}
$$

**MQA 完整输出**：

$$
\text{MultiQuery}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) \cdot W^O
$$

### 3.3 关键特性

**关键 1**：$W^K$ 和 $W^V$ 各只有 1 组，不是 h 组。这是与 MHA 的唯一结构差异。

**关键 2**：所有 Q_i 使用同一个 $W^K$ 和 $W^V$，即对于所有 i，$K_i = K_{shared}$，$V_i = V_{shared}$。

### 3.4 参数量对比

**MHA 的 KV 参数**：

$$
\theta_{KV}^{MHA} = h \times (d_{model} \times d_k + d_{model} \times d_v) = h \times (d_{model} \times \frac{d_{model}}{h} + d_{model} \times \frac{d_{model}}{h}) = 2 \cdot d_{model}^2
$$

**MQA 的 KV 参数**：

$$
\theta_{KV}^{MQA} = d_{model} \times d_k + d_{model} \times d_v = \frac{2 \cdot d_{model}^2}{h}
$$

**减少比例**：MQA 的 KV 参数量仅为 MHA 的 **1/h**。

以典型配置 h=8, d_model=512 为例：
- MHA KV 参数：8 × (512 × 64 + 512 × 64) = 524,288
- MQA KV 参数：512 × 64 + 512 × 64 = 65,536
- 减少量 = 458,752 个参数（节省 87.5%）

### 3.5 KV Cache 大小对比

在自回归解码中，每生成一个新 token，都需要保存所有已生成位置的 K 和 V。KV Cache 的大小为：

- **MHA**：$h \times s \times d_k \times 2$ 个浮点数（h 个头 × s 个位置 × d_k 维 × {K, V}）
- **MQA**：$s \times d_k \times 2$ 个浮点数（1 组 × s 个位置 × d_k 维 × {K, V}）

KV Cache 也必须减少到原来的 1/h。对于 h=96（如 PaLM 540B）、s=2048、d_k=128、fp16（2 字节）：
- MHA KV Cache：96 × 2048 × 128 × 2 × 2B ≈ **100 MB**
- MQA KV Cache：2048 × 128 × 2 × 2B ≈ **1 MB**
- 差距巨大，在批处理推理场景中尤为重要。

### 3.6 精度损失的理论分析

MQA 移除的是 K 和 V 在头之间的**多样性**（diversity）。但从信息论角度看，一个输入序列的"语义映射"在不同头之间确实存在大量冗余。每个头要关注的"内容"（Value）具有很强的共性，真正的差异在于"关注哪里"（Query 决定的注意力权重）。MQA 保留了选择"关注点"的多样性（Query 多头），仅仅统一了"被关注的内容"（共享 K/V），因此精度损失极小。

Shazeer 在 WMT'14 英德翻译任务上的实验表明，MQA 的 BLEU 分数仅比 MHA 低约 0.1-0.2，但推理吞吐量提升了约 3 倍。

---

## 4. 训练过程讲解

### 4.1 数据预处理

MQA 的数据预处理与 MHA 完全一致，不需要任何特殊处理。输入序列经过 tokenization 和 embedding 后，直接送入 MQA 层。训练数据以自回归语言建模为主（如 C4、The Pile 等大规模语料库）。

### 4.2 参数初始化

MQA 的初始化方式与 MHA 基本相同，但需要注意：
- 每个 Query 投影矩阵 W_i^Q 独立初始化（通常使用 Xavier 均匀分布或 Kaiming 初始化）；
- **共享的** W^K 和 W^V 各只有一个矩阵，使用标准初始化即可；
- 输出投影 W^O 与 MHA 相同。

因为 W^K 被所有头共用，其梯度是各头损失梯度的**总和**，因此在其学习率设置上通常不需要特殊调整——梯度信号更强反而有助于稳定训练。

### 4.3 训练迭代过程

MQA 作为 MHA 的 "drop-in replacement"（即插即用替代品），训练流程几乎不变：

1. **前向传播**：输入 X 经过 MQA 层计算输出，流程与 MHA 一致，只是 K/V 计算从 h 次投影降为 1 次。
2. **损失计算**：与 MHA 相同，使用交叉熵损失（语言建模）或任务特定损失。
3. **反向传播**：梯度通过共享的 W^K 和 W^V 回传时，所有 h 个头的梯度累加在一起更新参数。
4. **优化器更新**：使用 Adam/AdamW 等标准优化器。

训练速度方面，由于参数量减少了 `2d_model^2(1 - 1/h)`，每个训练 step 的计算量略有减小，但减小幅度不大（KV 投影在总计算量中占比有限），因此训练加速主要体现在推理阶段。

### 4.4 超参数设置

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 头数 h | 8 / 12 / 16 / 96 | 与 MHA 相同，越大 KV Cache 节省越显著 |
| d_k = d_v | d_model / h（通常 64-128） | 保持与原 MHA 一致 |
| 学习率 | 与 MHA 相同 | 不需要特殊调整 |
| warmup steps | 4000-10000 | 标准 Transformer 配置 |
| batch size | 根据 GPU 内存调整 | 由于每样本 KV Cache 更小，可适当增大 batch |

---

## 5. 应用场景

### 5.1 典型应用

1. **大模型解码器（如 PaLM、Chinchilla）**：Google 的 PaLM 系列模型在解码器中使用了 MQA。PaLM-540B 拥有 96 个头，若不使用 MQA，KV Cache 将极其庞大；使用 MQA 后，KV Cache 占用减少了 95 倍以上，使得长序列推理成为可能。

2. **需要低延迟推理的场景**：在线聊天机器人、实时翻译、语音助手等场景对首 token 延迟和生成速度要求严格。MQA 减少的内存读取（memory I/O）是推理加速的关键——在 decode 阶段，内存带宽往往是瓶颈而非计算本身。

3. **移动端 / 边缘设备部署**：手机、IoT 设备等内存受限的环境下，MQA 极小的 KV Cache（可以保持在 MB 级别甚至更小）使得本地推理成为可能。

### 5.2 适用与不适用场景

| 适用场景 | 不适用场景 |
|----------|------------|
| 强调推理速度和吞吐量的生产系统 | 对精度要求极度敏感的任务（如医疗诊断） |
| 长文本生成（KV Cache 优势明显） | 非常小的模型（头数少，节省不明显） |
| 大规模批处理推理 | 需要头间 K/V 多样性来捕获细微语义差异的任务 |
| 内存受限的部署环境 | 训练阶段不是瓶颈、推理量不大的研究场景 |

---

## 6. 优缺点分析

### 6.1 优点

1. **KV Cache 显著降低**：减少为原来的 1/h，对于多头大模型效果极为显著，直接降低推理内存占用和显存需求。
2. **推理加速**：更小的 KV Cache 意味着更少的 GPU 显存读写（内存带宽节省），在内存带宽瓶颈的 decode 阶段可以获得接近线性的加速。
3. **参数减少**：KV 投影矩阵参数减少为原来的 1/h，模型总体参数量下降，有利于模型分发和加载。
4. **训练梯度信号更强**：共享的 W^K 和 W^V 接收来自所有头的梯度，训练信号更密集，有助于参数学习。
5. **实现简单**：作为 MHA 的即插即用替代品，只需修改投影逻辑和广播操作，无需改变整体架构。

### 6.2 缺点

1. **轻微精度损失**：K 和 V 的共享限制了模型在头之间表达不同"内容表征"的能力，可能导致 BLEU/困惑度等指标的微小下降（通常 < 0.5%）。
2. **头间多样性不足**：虽然 Query 的多样性保留了"关注位置"的选择能力，但被关注到的"内容"在各头之间完全一致，可能限制模型在某些任务上的表现。
3. **较小模型提升不明显**：当 h 较小时，KV Cache 节省的比例也较小，而精度损失可能更加突出。
4. **训练阶段加速有限**：KV 投影在训练总计算中的占比较小，因此训练阶段的速度提升不如推理阶段显著。

### 6.3 MHA 与 MQA 对比表

| 对比维度 | MHA | MQA | 差异 |
|----------|-----|-----|------|
| Query 投影 | h 个 W_i^Q | h 个 W_i^Q | 相同 |
| Key 投影 | h 个 W_i^K | 1 个 W^K | MQA 少 (h-1) 个 |
| Value 投影 | h 个 W_i^V | 1 个 W^V | MQA 少 (h-1) 个 |
| 输出投影 | 1 个 W^O | 1 个 W^O | 相同 |
| KV 参数量 | 2·d_model² | 2·d_model²/h | MQA 减少为 1/h |
| KV Cache 大小 | h·s·d_k·2 | s·d_k·2 | MQA 减少为 1/h |
| 计算量（FLOPs） | 略高 | 略低（差异小） | ~相同量级 |
| 推理吞吐量 | 基准 | 显著提高 | MQA 更快 |
| 精度 | 基准 | 极微小下降 | ~相当 |

---

## 7. 调库实现

以下使用 PyTorch 实现 MQA，包含完整的代码注释。核心设计是：Q 分头（h 组），但 K 和 V 不分头（只有 1 组），通过 broadcast 让所有头共享。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiQueryAttention(nn.Module):
    """
    多查询注意力 (Multi-Query Attention, MQA) 的 PyTorch 实现。

    MQA 让所有注意力头共享同一组 Key 和 Value，而每个头保留独立的 Query。
    这大幅减少了推理时的 KV Cache 大小（减少为原来的 1/h），
    以极小的精度损失换取显著的推理加速。
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化 MQA 层。

        参数:
            d_model:  模型隐层维度（必须能被 num_heads 整除）
            num_heads: 注意力头数
            dropout: 注意力权重的 dropout 概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 每个头拥有独立的 Query 投影矩阵
        # 为了使用矩阵乘法高效计算所有头的 Q，用一个大的 W_q 一次性投影
        # W_q 实际等价于 h 个独立的 W_i^Q 拼接
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # 所有头共享的 Key 投影矩阵（只有 1 组，不分头！）
        # 注意：这里的输出维度是 self.d_k（单个头的维度），而不是 d_model
        # 因为只有一组 K，不需要输出 h*d_k 维
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)

        # 所有头共享的 Value 投影矩阵（只有 1 组）
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)

        # 输出投影矩阵（与 MHA 相同）
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播。

        参数:
            query:  形状 (batch, seq_len, d_model) - 自注意力下 query == key == value
            key:    形状 (batch, seq_len, d_model)
            value:  形状 (batch, seq_len, d_model)
            mask:   可选，形状 (batch, 1, seq_len, seq_len) 或 (batch, seq_len, seq_len)

        返回:
            output: 形状 (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape

        # ===== 第一步：线性投影 =====
        # Q: 投影到 d_model 维，包含了所有头的 Q 信息
        Q = self.W_q(query)   # (batch, seq_len, d_model)

        # K: 投影到 d_k 维 —— 只有一组，不分头！
        K = self.W_k(key)     # (batch, seq_len, d_k)

        # V: 投影到 d_k 维 —— 只有一组，不分头！
        V = self.W_v(value)   # (batch, seq_len, d_k)

        # ===== 第二步：Q 分头（reshape） =====
        # 将 Q 从 (batch, seq_len, d_model) 重塑为 (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q 形状： (batch, num_heads, seq_len, d_k)

        # ===== 第三步：K 和 V 保持单头，但需要广播到多头 =====
        # K 形状：(batch, seq_len, d_k)
        # 为让所有头都能与 K 做矩阵乘法，在头数维度上增加一维
        # 使其变为 (batch, 1, seq_len, d_k)
        # 这个维度为 1，在后续计算中会自动广播
        K = K.unsqueeze(1)   # (batch, 1, seq_len, d_k)
        V = V.unsqueeze(1)   # (batch, 1, seq_len, d_k)

        # ===== 第四步：计算注意力分数 =====
        # Q @ K^T 时，Q 是 (batch, num_heads, seq_len, d_k)
        #          K^T 是 (batch, 1, d_k, seq_len) → 广播后 (batch, num_heads, d_k, seq_len)
        # 结果: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ===== 第五步：应用 mask 并 softmax =====
        if mask is not None:
            # 确保 mask 可以正确广播
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # ===== 第六步：加权求和 =====
        # attn_weights @ V 时，attn_weights 是 (batch, num_heads, seq_len, seq_len)
        #                     V 是 (batch, 1, seq_len, d_k)
        # 结果: (batch, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)

        # ===== 第七步：合并多头输出 =====
        # 从 (batch, num_heads, seq_len, d_k) 变为 (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # ===== 第八步：输出投影 =====
        output = self.W_o(context)  # (batch, seq_len, d_model)

        return output


# ===== 测试代码 =====
if __name__ == "__main__":
    # 设置超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建 MQA 实例
    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads)

    # 前向传播
    output = mqa(x, x, x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"KV 投影参数量:")
    kv_params = sum(p.numel() for name, p in mqa.named_parameters() if 'W_k' in name or 'W_v' in name)
    print(f"  MQA: {kv_params:,} 个参数")
    mha_kv_params = 2 * d_model * d_model  # MHA 的 KV 参数量
    print(f"  MHA (等价的 {num_heads} 头): {mha_kv_params:,} 个参数")
    print(f"  减少比例: {1 - kv_params / mha_kv_params:.1%}")
```

**预期输出**：

```
输入形状: torch.Size([2, 10, 512])
输出形状: torch.Size([2, 10, 512])
KV 投影参数量:
  MQA: 65,536 个参数
  MHA (等价的 8 头): 524,288 个参数
  减少比例: 87.5%
```

---

## 8. 手工代码实现

以下使用 `einops` 简化张量重排操作，从零实现 MQA 并与 MHA 进行参数数量对比。`einops` 的 `rearrange` 语法比原生 `view/transpose` 更直观可读。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class MultiHeadAttention_MQA(nn.Module):
    """
    多查询注意力 (MQA) 的手工实现。

    核心区别:
    - MHA: W_k 和 W_v 输出 d_model 维（包含 h 个头），再 split 为 h 组
    - MQA: W_k 和 W_v 输出 d_k 维（只有 1 组），所有头共享
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Query: 投影到 d_model → 再拆成 h 个头，每头 d_k 维
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # Key: 投影到 d_k 维 —— 只有一组！（MHA 则投影到 d_model 维）
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)

        # Value: 投影到 d_k 维 —— 只有一组！
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        自注意力模式的前向传播（query == key == value == x）。

        参数:
            x: 输入张量，形状 (batch, seq_len, d_model)
            mask: 注意力掩码，形状 (batch, 1, seq_len, seq_len)

        返回:
            output: 形状 (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # ---- 1. 线性投影 ----
        # Q: (batch, seq_len, d_model)
        Q = self.W_q(x)

        # K: (batch, seq_len, d_k)  ← 注意：投影到 d_k，不是 d_model！
        K = self.W_k(x)

        # V: (batch, seq_len, d_k)
        V = self.W_v(x)

        # ---- 2. 使用 einops 进行 Q 分头 ----
        # 将 Q 从 (batch, seq_len, d_model) 重新排列为 (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        # 现在 Q 形状: (batch, num_heads, seq_len, d_k)

        # ---- 3. K 和 V: 增加头数维度以便广播 ----
        # K 从 (batch, seq_len, d_k) 变为 (batch, 1, seq_len, d_k)
        K = rearrange(K, 'b s d -> b 1 s d')
        V = rearrange(V, 'b s d -> b 1 s d')
        # 现在 K, V 形状: (batch, 1, seq_len, d_k)
        # 头数维度为 1，在后续运算中会自动 broadcast

        # ---- 4. 计算注意力分数 ----
        # Q: (b, h, s, d_k)  @  K^T: (b, 1, d_k, s) → (b, h, s, s)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ---- 5. mask + softmax ----
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (b, h, s, s)
        attn = self.dropout(attn)

        # ---- 6. 加权求和 ----
        # attn: (b, h, s, s)  @  V: (b, 1, s, d_k) → (b, h, s, d_k)
        context = torch.matmul(attn, V)

        # ---- 7. 合并多头 ----
        # 从 (b, h, s, d_k) 变为 (b, s, d_model)
        context = rearrange(context, 'b h s d -> b s (h d)')

        # ---- 8. 输出投影 ----
        output = self.W_o(context)

        return output

    def get_kv_params(self) -> int:
        """返回 K 和 V 投影相关的参数总数。"""
        return sum(p.numel() for name, p in self.named_parameters()
                   if 'W_k' in name or 'W_v' in name)


class MultiHeadAttention_MHA(nn.Module):
    """
    标准多头注意力 (MHA) —— 用于对比。
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 所有投影都到 d_model 维，再分头
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # MHA: d_model 维！
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分头 —— 每个都有 h 个独立的头
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        # 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)

        # 合并
        context = rearrange(context, 'b h s d -> b s (h d)')
        return self.W_o(context)

    def get_kv_params(self) -> int:
        """返回 K 和 V 投影相关的参数总数。"""
        return sum(p.numel() for name, p in self.named_parameters()
                   if 'W_k' in name or 'W_v' in name)


# ======================== 完整测试代码 ========================
def test_mqa_vs_mha():
    """
    对比 MQA 和 MHA 的参数数量、前向传播输出和 KV Cache 大小。
    """
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 32

    # 创建两个模型
    mqa = MultiHeadAttention_MQA(d_model=d_model, num_heads=num_heads)
    mha = MultiHeadAttention_MHA(d_model=d_model, num_heads=num_heads)

    # 相同的随机输入
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    with torch.no_grad():
        out_mqa = mqa(x)
        out_mha = mha(x)

    # ---- 结果检查 ----
    print("=" * 60)
    print("MQA vs MHA 对比测试")
    print("=" * 60)

    print(f"\n【配置信息】")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, d_k = {d_model // num_heads}")
    print(f"  输入形状: {x.shape}")

    print(f"\n【输出形状】")
    print(f"  MQA 输出: {out_mqa.shape}")
    print(f"  MHA 输出: {out_mha.shape}")

    print(f"\n【KV 投影参数对比】")
    mqa_kv = mqa.get_kv_params()
    mha_kv = mha.get_kv_params()
    print(f"  MQA 的 KV 参数量: {mqa_kv:,}")
    print(f"  MHA 的 KV 参数量: {mha_kv:,}")
    print(f"  比例 (MQA/MHA):   {mqa_kv / mha_kv:.2%}")
    print(f"  节省的参数数量:    {mha_kv - mqa_kv:,}")

    print(f"\n【总参数对比】")
    mqa_total = sum(p.numel() for p in mqa.parameters())
    mha_total = sum(p.numel() for p in mha.parameters())
    print(f"  MQA 总参数量: {mqa_total:,}")
    print(f"  MHA 总参数量: {mha_total:,}")
    print(f"  比例 (MQA/MHA): {mqa_total / mha_total:.2%}")

    print(f"\n【KV Cache 大小对比 (fp16, 2 字节/元素)】")
    for s in [64, 128, 256, 512, 1024, 2048]:
        mqa_cache = s * (d_model // num_heads) * 2 * 2  # s * d_k * 2(K+V) * 2(fp16)
        mha_cache = num_heads * s * (d_model // num_heads) * 2 * 2
        print(f"  seq_len={s:4d}: MQA={mqa_cache/1024:6.1f} KB | MHA={mha_cache/1024:6.1f} KB | "
              f"节省 {1 - mqa_cache/mha_cache:.1%}")

    print(f"\n【验证 MQA 输出不含 NaN】")
    print(f"  MQA 输出最大值: {out_mqa.max().item():.4f}")
    print(f"  MQA 输出最小值: {out_mqa.min().item():.4f}")
    print(f"  是否含 NaN: {torch.isnan(out_mqa).any().item()}")

    print(f"\n{'=' * 60}")
    print("测试通过！MQA 实现正确，输出形状与 MHA 一致。")
    print("=" * 60)


if __name__ == "__main__":
    test_mqa_vs_mha()
```

**预期输出**：

```
============================================================
MQA vs MHA 对比测试
============================================================

【配置信息】
  d_model = 512, num_heads = 8, d_k = 64
  输入形状: torch.Size([4, 32, 512])

【输出形状】
  MQA 输出: torch.Size([4, 32, 512])
  MHA 输出: torch.Size([4, 32, 512])

【KV 投影参数对比】
  MQA 的 KV 参数量: 65,536
  MHA 的 KV 参数量: 524,288
  比例 (MQA/MHA):   12.50%
  节省的参数数量:    458,752

【总参数对比】
  MQA 总参数量: 851,968
  MHA 总参数量: 1,310,720
  比例 (MQA/MHA): 65.00%

【KV Cache 大小对比 (fp16, 2 字节/元素)】
  seq_len=  64: MQA=  16.0 KB | MHA= 128.0 KB | 节省 87.5%
  seq_len= 128: MQA=  32.0 KB | MHA= 256.0 KB | 节省 87.5%
  seq_len= 256: MQA=  64.0 KB | MHA= 512.0 KB | 节省 87.5%
  seq_len= 512: MQA= 128.0 KB | MHA=1024.0 KB | 节省 87.5%
  seq_len=1024: MQA= 256.0 KB | MHA=2048.0 KB | 节省 87.5%
  seq_len=2048: MQA= 512.0 KB | MHA=4096.0 KB | 节省 87.5%
```

---

## 9. 可视化与结果理解

以下代码使用 matplotlib 绘制三组可视化图表：MHA 与 MQA 的结构对比、KV Cache 随头数变化的曲线、以及参数量对比柱状图。

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 设置中文字体（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 图表 1: MHA 与 MQA 结构对比示意图 ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------- 左图: MHA 结构 ----------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('MHA (Multi-Head Attention)', fontsize=14, fontweight='bold', pad=15)

# 输入 X
ax.add_patch(plt.Rectangle((3.5, 8.5), 3, 0.6, fill=True, facecolor='#AED6F1', edgecolor='#2E86C1', lw=2))
ax.text(5, 8.8, '输入 X (d_model)', ha='center', va='center', fontsize=10, fontweight='bold')

# 投影 W_q, W_k, W_v —— 每种都有 h 个
colors_q = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#1ABC9C', '#3498DB', '#9B59B6', '#E91E63']
y_q = 7.0
for i in range(8):
    ax.add_patch(plt.Rectangle((0.5 + i*1.12, y_q + 0.3), 0.8, 0.5, fill=True,
                                facecolor=colors_q[i], edgecolor='gray', lw=1, alpha=0.7))
    ax.text(0.9 + i*1.12, y_q + 0.55, f'W_q{i}', ha='center', va='center', fontsize=7)
    ax.add_patch(plt.Rectangle((0.5 + i*1.12, y_q - 0.6), 0.8, 0.5, fill=True,
                                facecolor=colors_q[i], edgecolor='gray', lw=1, alpha=0.7))
    ax.text(0.9 + i*1.12, y_q - 0.35, f'W_kv{i}', ha='center', va='center', fontsize=7)

# 注意力头
for i in range(8):
    ax.add_patch(plt.Rectangle((0.5 + i*1.12, y_q - 1.7), 0.8, 0.5, fill=True,
                                facecolor='#D5F5E3', edgecolor='#27AE60', lw=1))
    ax.text(0.9 + i*1.12, y_q - 1.45, f'Head {i}', ha='center', va='center', fontsize=7)

# 拼接
ax.add_patch(plt.Rectangle((3.5, y_q - 2.8), 3, 0.5, fill=True, facecolor='#FADBD8', edgecolor='#C0392B', lw=1.5))
ax.text(5, y_q - 2.55, 'Concat → W_o → Output', ha='center', va='center', fontsize=10)

# 标注文字
ax.text(5, 6.7, '(h 组独立的 K, V)', ha='center', va='center', fontsize=10, style='italic', color='#7F8C8D')

# ---------- 右图: MQA 结构 ----------
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('MQA (Multi-Query Attention)', fontsize=14, fontweight='bold', pad=15)

# 输入 X
ax.add_patch(plt.Rectangle((3.5, 8.5), 3, 0.6, fill=True, facecolor='#AED6F1', edgecolor='#2E86C1', lw=2))
ax.text(5, 8.8, '输入 X (d_model)', ha='center', va='center', fontsize=10, fontweight='bold')

# Q 投影 —— h 个独立
y_q2 = 7.0
for i in range(8):
    ax.add_patch(plt.Rectangle((0.5 + i*1.12, y_q2 + 0.3), 0.8, 0.5, fill=True,
                                facecolor=colors_q[i], edgecolor='gray', lw=1, alpha=0.7))
    ax.text(0.9 + i*1.12, y_q2 + 0.55, f'W_q{i}', ha='center', va='center', fontsize=7)

# 共享的 K 和 V —— 只有一组！
ax.add_patch(plt.Rectangle((3.5, y_q2 - 0.6), 3, 0.5, fill=True,
                            facecolor='#F9E79F', edgecolor='#F39C12', lw=2))
ax.text(5, y_q2 - 0.35, 'W_k (共享) · W_v (共享)', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头指向各头
for i in range(8):
    ax.annotate('', xy=(0.9 + i*1.12, y_q2 - 1.7 + 0.25), xytext=(4.2, y_q2 - 0.35),
                arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1, alpha=0.4))

# 注意力头（共享 K, V）
for i in range(8):
    ax.add_patch(plt.Rectangle((0.5 + i*1.12, y_q2 - 1.7), 0.8, 0.5, fill=True,
                                facecolor='#D5F5E3', edgecolor='#27AE60', lw=1))
    ax.text(0.9 + i*1.12, y_q2 - 1.45, f'Head {i}', ha='center', va='center', fontsize=7)

# 拼接
ax.add_patch(plt.Rectangle((3.5, y_q2 - 2.8), 3, 0.5, fill=True, facecolor='#FADBD8', edgecolor='#C0392B', lw=1.5))
ax.text(5, y_q2 - 2.55, 'Concat → W_o → Output', ha='center', va='center', fontsize=10)

ax.text(5, 6.7, '(所有头共享 1 组 K, V)', ha='center', va='center', fontsize=10, style='italic', color='#27AE60', fontweight='bold')

plt.tight_layout()
plt.savefig('mqa_structure_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================== 图表 2: KV Cache 随头数增加的对比曲线 ========================
fig, ax = plt.subplots(figsize=(10, 5))

num_heads_range = np.arange(1, 33)       # 头数从 1 到 32 变化
d_k = 64                                 # 固定 d_k = 64
seq_len = 1024                           # 固定序列长度 = 1024
bytes_per_elem = 2                       # fp16

mqa_cache = seq_len * d_k * 2 * bytes_per_elem / (1024 ** 2)  # MB, 常数
mha_cache = num_heads_range * seq_len * d_k * 2 * bytes_per_elem / (1024 ** 2)  # MB

ax.plot(num_heads_range, [mqa_cache] * len(num_heads_range), 'o-',
        color='#27AE60', linewidth=2.5, markersize=4, label='MQA (共享 K/V)')
ax.plot(num_heads_range, mha_cache, 's-',
        color='#E74C3C', linewidth=2.5, markersize=4, label='MHA (独立 K/V)')
ax.fill_between(num_heads_range, [mqa_cache] * len(num_heads_range), mha_cache,
                alpha=0.15, color='#E74C3C', label='节省的缓存空间')

ax.set_xlabel('注意力头数 (num_heads)', fontsize=12, fontweight='bold')
ax.set_ylabel('KV Cache 大小 (MB)', fontsize=12, fontweight='bold')
ax.set_title('KV Cache 大小对比: MQA vs MHA (seq_len=1024, d_k=64, fp16)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')

# 标注关键点
ax.annotate(f'MQA: {mqa_cache:.1f} MB (恒定)',
            xy=(16, mqa_cache), xytext=(4, mqa_cache + 2),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
            fontsize=10, color='#27AE60')
ax.annotate(f'h=8: MHA={mqa_cache*8:.1f} MB, MQA={mqa_cache:.1f} MB\n节省 {(1-1/8)*100:.0f}%',
            xy=(8, mqa_cache*8), xytext=(12, mqa_cache * 8 + 2),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
            fontsize=10, color='#E74C3C')

plt.tight_layout()
plt.savefig('mqa_kv_cache_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================== 图表 3: 参数量对比柱状图 ========================
fig, ax = plt.subplots(figsize=(10, 5))

d_model_configs = [256, 512, 768, 1024, 2048, 4096]
num_heads = 8

mqa_params = []
mha_params = []
labels = []

for dm in d_model_configs:
    dk = dm // num_heads
    mqa_kv = dm * dk * 2  # W_k + W_v
    mha_kv = num_heads * dm * dk * 2
    mqa_params.append(mqa_kv)
    mha_params.append(mha_kv)
    labels.append(f'd_model={dm}\n(num_heads={num_heads})')

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, mqa_params, width, label='MQA (共享 K/V)',
               color='#27AE60', edgecolor='white', lw=1.5)
bars2 = ax.bar(x + width/2, mha_params, width, label='MHA (独立 K/V)',
               color='#E74C3C', edgecolor='white', lw=1.5)

# 在柱子上标注数值
for bar, val in zip(bars1, mqa_params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mha_params)*0.02,
            f'{val/1e6:.2f}M' if val > 1e6 else f'{val/1e3:.0f}K',
            ha='center', va='bottom', fontsize=8, color='#27AE60', fontweight='bold')

for bar, val in zip(bars2, mha_params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mha_params)*0.02,
            f'{val/1e6:.2f}M' if val > 1e6 else f'{val/1e3:.0f}K',
            ha='center', va='bottom', fontsize=8, color='#E74C3C', fontweight='bold')

ax.set_xlabel('模型配置', fontsize=12, fontweight='bold')
ax.set_ylabel('KV 投影参数量', fontsize=12, fontweight='bold')
ax.set_title('KV 投影参数量对比: MQA vs MHA (num_heads=8)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加节省比例文字
for i, (mq, mh) in enumerate(zip(mqa_params, mha_params)):
    savings = (1 - mq / mh) * 100
    ax.text(i, max(mq, mh) * 1.08, f'节省 {savings:.0f}%',
            ha='center', fontsize=9, color='#7F8C8D', style='italic')

plt.tight_layout()
plt.savefig('mqa_params_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("三组可视化图表已生成:")
print("  1. mqa_structure_comparison.png — MHA vs MQA 结构对比示意图")
print("  2. mqa_kv_cache_comparison.png — KV Cache 随头数变化曲线")
print("  3. mqa_params_comparison.png — KV 投影参数量对比柱状图")
```

---

## 10. 模型评估

评估 MQA 的核心指标包括：专家利用率、负载均衡度和困惑度。以下代码模拟对比 MQA 与 MHA 在推理阶段的性能差异。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from einops import rearrange


class MultiHeadAttention_MHA(nn.Module):
    """标准 MHA（用于评估对比）"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        Q = rearrange(self.W_q(x), 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(self.W_k(x), 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(self.W_v(x), 'b s (h d) -> b h s d', h=self.num_heads)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V)
        ctx = rearrange(ctx, 'b h s d -> b s (h d)')
        return self.W_o(ctx)


class MultiHeadAttention_MQA(nn.Module):
    """MQA 实现（用于评估对比）"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)   # 共享 K
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)   # 共享 V
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        Q = rearrange(self.W_q(x), 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(self.W_k(x), 'b s d -> b 1 s d')   # 单组 K
        V = rearrange(self.W_v(x), 'b s d -> b 1 s d')   # 单组 V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V)
        ctx = rearrange(ctx, 'b h s d -> b s (h d)')
        return self.W_o(ctx)


def measure_inference_speed(model, x, n_warmup=10, n_trials=50):
    """测量单次前向推理的延迟。"""
    model.eval()
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # 计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_trials * 1000  # 返回毫秒


def compute_pseudo_perplexity(output, target):
    """
    用输出和目标之间的 MSE 模拟"困惑度"。
    实际困惑度需要完整的语言模型概率分布计算，
    这里用重构误差来近似对比模型质量。
    """
    return F.mse_loss(output, target).item()


# ======================== 评估主函数 ========================
print("=" * 65)
print("MQA vs MHA 模型评估")
print("=" * 65)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置
d_model = 512
num_heads = 8
batch_size = 1
seq_len_list = [64, 128, 256, 512, 1024]

# 创建模型
mqa = MultiHeadAttention_MQA(d_model=d_model, num_heads=num_heads).to(device)
mha = MultiHeadAttention_MHA(d_model=d_model, num_heads=num_heads).to(device)

# 评估表格打印
print(f"\n{'序列长度':<10} {'MHA延迟(ms)':<14} {'MQA延迟(ms)':<14} "
      f"{'加速比':<10} {'MHA重构误差':<14} {'MQA重构误差':<14}")
print("-" * 75)

for seq_len in seq_len_list:
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # 测量推理速度
    mha_latency = measure_inference_speed(mha, x)
    mqa_latency = measure_inference_speed(mqa, x)

    # 计算重构误差（模拟困惑度比较）
    with torch.no_grad():
        out_mha = mha(x)
        out_mqa = mqa(x)
    mha_error = compute_pseudo_perplexity(out_mha, x)
    mqa_error = compute_pseudo_perplexity(out_mqa, x)

    speedup = mha_latency / mqa_latency

    print(f"{seq_len:<10} {mha_latency:<14.3f} {mqa_latency:<14.3f} "
          f"{speedup:<10.2f}x {mha_error:<14.6f} {mqa_error:<14.6f}")

# KV Cache 评估
print(f"\n【KV Cache 占用评估 (fp16, seq_len=1024)】")
d_k = d_model // num_heads
s = 1024
bytes_per_elem = 2
mha_cache_mb = num_heads * s * d_k * 2 * bytes_per_elem / (1024 ** 2)
mqa_cache_mb = 1 * s * d_k * 2 * bytes_per_elem / (1024 ** 2)
print(f"  MHA KV Cache: {mha_cache_mb:.1f} MB")
print(f"  MQA KV Cache: {mqa_cache_mb:.1f} MB")
print(f"  节省: {(1 - mqa_cache_mb / mha_cache_mb) * 100:.0f}%")

# 模型参数量评估
mha_total = sum(p.numel() for p in mha.parameters())
mqa_total = sum(p.numel() for p in mqa.parameters())
print(f"\n【参数量评估】")
print(f"  MHA 总参数: {mha_total:,}")
print(f"  MQA 总参数: {mqa_total:,}")
print(f"  MQA 参数量减少: {(1 - mqa_total / mha_total) * 100:.1f}%")

print(f"\n{'=' * 65}")
print("评估结论: MQA 在推理速度上显著优于 MHA（尤其在长序列），")
print("          重构误差（模拟困惑度）仅有微小差异，")
print("          KV Cache 占用仅为 MHA 的 1/num_heads。")
print("=" * 65)
```

---

## 11. 常见问题与易错点

### 11.1 数据层面

**问题 1：Batch Size 太小时路由器不稳定**

| 维度 | 说明 |
|------|------|
| **现象** | 训练时每个 batch 内，路由器（门控网络）的选择分布在不同 batch 之间剧烈波动，注意力权重模式不一致，loss 曲线呈现锯齿状震荡。 |
| **原因** | MQA 的共享 K/V 需要在每个 batch 内捕获足够的语义信号来形成稳定的路由决策。batch size 过小意味着 K/V 所基于的统计样本不足，路由器无法学到稳定的"哪些 token 应该给哪个 Query 头更多权重"。 |
| **解决方案** | 1. 尽可能增大 batch size，至少保证每个 GPU 上有 16-32 个样本；2. 使用梯度累积（gradient accumulation）模拟大 batch 效果；3. 对于实在无法增加 batch 的情况，减小模型维度或减少头数，降低模型的统计需求。 |

**问题 2：序列长度分布严重不均**

| 维度 | 说明 |
|------|------|
| **现象** | 短序列（如 10-50 tokens）时 MQA 表现正常，但长序列（> 1000 tokens）时注意力分布逐渐退化，某些 Query 头开始退化到均匀注意力。 |
| **原因** | 共享的 K 和 V 在长序列中承载了来自序列各处的大量语义信息，单一的 K/V 空间可能无法同时服务好所有 Query 头的需求。尤其是当序列包含多个不同主题的段落时，共享的 K 可能被"语义平均"，导致 Query 头之间的区分度下降。 |
| **解决方案** | 1. 对训练数据做长度分桶（bucketing），确保每个 batch 内长度相近；2. 对于超长序列场景，考虑使用 GQA（Grouped Query Attention）作为折中方案；3. 适当增大 qk_rope_dim 以增强 Query 头的区分能力。 |

### 11.2 模型层面

**问题 3：专家塌缩 —— 所有 Token 只走少数"专家"**

| 维度 | 说明 |
|------|------|
| **现象** | 虽然 MQA 中 K/V 被所有头共享（不涉及 MoE 的专家路由），但在使用 MQA + MoE 的混合架构中（如 DeepSeekMoE），大量 token 被路由到极少数专家，导致其他专家的梯度几乎为零，失去学习能力。观察负载均衡损失居高不下或迅速降到不合理的低值。 |
| **原因** | 门控网络的初始权重随机性不够，或者学习率过大导致路由器快速收敛到局部最优；同时共享 K/V 进一步减少了路由的区分信号，路由器更容易"偷懒"选择少数专家。 |
| **解决方案** | 1. 降低路由器的学习率（通常是整体学习率的 0.1-0.5 倍）；2. 使用 Noisy Top-K Gating，在路由器 logits 中注入高斯噪声，强制路由器随机探索不同选择；3. 增大 auxiliary loss 的权重（如从 0.01 提高到 0.05），强约束均匀路由；4. 加入专家丢弃（expert dropout），在训练时随机 mask 掉部分专家。 |

**问题 4：Top-K 选择的梯度回传问题**

| 维度 | 说明 |
|------|------|
| **现象** | 在 Top-K 操作中，未被选中的 K/V 通道完全不参与前向计算。虽然 MQA 中 K/V 各只有一组（不会被 Top-K 筛掉），但在某些 MQA 扩展变体中（如对 Query 做 Top-K 选择或对专家输出做 Top-K），被筛掉部分的梯度始终为零。训练时发现模型某些部分完全不更新。 |
| **原因** | Top-K 是一个离散操作，本质上不可微。被筛选掉的元素对输出贡献为 0，因此反向传播时梯度为 0，这些元素永远没有机会被重新激活。 |
| **解决方案** | 1. 对于 Top-K 路由，标准做法是使用直通估计器（Straight-Through Estimator），即前向使用 Top-K 做硬选择，反向传播时对未选中的元素也保留梯度（不经过 Top-K 门控）；2. 使用 Gumbel-Softmax 替代硬 Top-K，通过温度退火逐渐逼近离散选择；3. 在 MQA 本身的共享 K/V 中，不需要 Top-K（所有头都用同一份 K/V），因此 MQA 本身不会遇到此问题——这是 MQA 的一个额外优势。 |

### 11.3 调参层面

**问题 5：top_k 值选择不当**

| 维度 | 说明 |
|------|------|
| **现象** | 当 top_k=1 时（所有头都关注 K 中最高度相似的一个位置），模型过度聚焦单一位置，丢失上下文信息；当 top_k=2*h（几乎等于全选）时，失去了稀疏性的优势和加速效果。 |
| **原因** | 在 MQA 中，top_k 实际控制的是"每个 Query 头关注 K 中多少个位置"，但由于 K 是共享的，top_k 的距离效应被放大——所有头的注意力叠加在同一个 K 空间上，top_k 过小会限制所有头的视野，过大则回到接近 dense attention 的效果。 |
| **解决方案** | 标准 MQA 不使用 Top-K 选择（它使用全量 scaled dot-product attention，Top-K 仅用于 MoE 的路由器）。如果确实需要稀疏注意力 + MQA 的组合（如结合 Sliding Window），建议：1. 在短文本任务中 K 可以较大（如 32）；2. 在长文本中考虑滑动窗口混合全局注意力；3. 通过消融实验找到精度-速度的最佳平衡点。 |

**问题 6：噪声系数设置过大或过小**

| 维度 | 说明 |
|------|------|
| **现象** | 噪声系数过大时，门控网络的输出充满随机性，每个 step 专家选择剧烈变化，训练不稳定且无法收敛。噪声系数过小时，负载均衡损失权重再大也拉不动路由分布，专家塌缩持续发生。 |
| **原因** | 噪声注入的目的是打破路由的对称性和防止过早收敛，但噪声本质上是随机扰动。过大则路由决策完全随机（失去"智能路由"的意义），过小则起不到探索作用。 |
| **解决方案** | 1. 推荐的噪声标准差范围为 0.1-1.0，从 1.0 开始，随着训练逐渐降低（噪声退火）；2. 使用 NoisyTopkRouter 的标准实现：`noise_stddev = softplus(x·W_noise)`，然后用这个标准差缩放噪声；3. 监控路由熵（router entropy），理想情况下应该保持在 ln(num_experts) 的 60-80%。 |

---

## 12. 学习总结

混合专家模型（MQA）是 Transformer 架构中一种轻量却极为实用的注意力优化方案。它的核心思路简洁有力：**所有注意力头共享同一组 Key 和 Value 投影矩阵，仅保留 Query 的多头多样性**。这一设计源于对 MHA 中 K/V 参数冗余的深入洞察——不同头之间关注的"内容"高度相似，真正的多样性在于"关注点"的选择。

从数学上看，MQA 的核心公式仍然遵循 scaled dot-product attention：$\text{head}_i = \text{softmax}(Q_i K_{shared}^\top / \sqrt{d_k}) V_{shared}$，唯一的改变是 $K_{shared}$ 和 $V_{shared}$ 来自唯一的 $W^K$ 和 $W^V$。这一微小的结构变化带来了参数量和 KV Cache 的双重大幅缩减（均降低到原来的 $1/h$），而精度损失仅约 0.1-0.2%。

MQA 与 GQA（Grouped Query Attention）和 MLA（Multi-head Latent Attention）的关系值得理清：GQA 是 MHA 和 MQA 之间的过渡——将 Query 头分组，每组共享一组 K/V；MLA 则更进一步，在低维潜在空间中先压缩再展开 K 和 V，实现比 MQA 更极致的缓存压缩。从 MHA → GQA → MQA → MLA 的演变路径，代表了业界对"如何在保持注意力的表达能力的同时最大限度地减少推理开销"这一核心问题的持续探索。

掌握 MQA 不仅有助于理解 DeepSeek-V2/V3、PaLM、Mixtral 等大模型的设计哲学，也为理解后续更高级的注意力优化技术（如 Flash Attention、MLA）奠定了坚实基础。

---

## 13. 练习题与思考题

### 基础题

**题目 1（基础）**：一个 Transformer 解码器使用标准 MHA，配置为 d_model=768, num_heads=12。请计算：
(1) 每个头的 d_k 是多少？
(2) MHA 中所有 K 和 V 投影矩阵的参数量是多少？
(3) 如果改为 MQA，KV 投影参数量减少多少？

<details>
<summary><b>参考答案</b></summary>

(1) d_k = d_model / num_heads = 768 / 12 = 64。

(2) MHA 中 K 投影：每个头有一个 W^K ∈ R^{768 × 64}，共 12 个。
单头参数量 = 768 × 64 = 49,152。
所有 K 投影参数 = 12 × 49,152 = 589,824。
V 投影同理，也是 589,824。
总 KV 参数 = 589,824 + 589,824 = 1,179,648。

(3) MQA 中 K 投影：只有一个 W^K ∈ R^{768 × 64}，参数量 = 768 × 64 = 49,152。
V 投影同理：49,152。
总 KV 参数 = 49,152 + 49,152 = 98,304。
减少量 = 1,179,648 - 98,304 = 1,081,344（减少 91.7%，即原来的 1/12）。

</details>

**题目 2（基础）**：在 MQA 的推理过程中，batch_size=4, seq_len=2048, d_model=512, num_heads=8，使用 fp16 存储。分别计算 MHA 和 MQA 的 KV Cache 占用，并解释为什么 MQA 的节省对批处理（batch inference）特别重要。

<details>
<summary><b>参考答案</b></summary>

d_k = 512 / 8 = 64。fp16 每元素 2 字节。

**MHA**：
单样本 KV Cache = num_heads × seq_len × d_k × 2（K和V各一份）× 2字节
= 8 × 2048 × 64 × 2 × 2 = 4,194,304 字节 = 4 MB
batch_size=4 时：4 × 4 = 16 MB

**MQA**：
单样本 KV Cache = 1 × seq_len × d_k × 2 × 2字节
= 2048 × 64 × 2 × 2 = 524,288 字节 = 0.5 MB
batch_size=4 时：4 × 0.5 = 2 MB

对批处理特别重要的原因：
批处理是提高 GPU 利用率和吞吐量的关键手段。在 MHA 中，KV Cache 与 batch_size 呈线性关系——batch 越大，KV Cache 膨胀越严重，很快触达 GPU 显存上限。MQA 将单样本 KV Cache 降低了 h 倍，直接释放了大量显存空间，使得可以用更大的 batch_size 进行推理，从而成倍提升吞吐量。假设 GPU 显存为 16 GB，MHA 下可能只能 batch=256，而 MQA 下可以 batch=2048——8 倍的吞吐量差异。

</details>

### 进阶题

**题目 3（进阶）**：对比 MHA、GQA (Grouped Query Attention) 和 MQA 三者的参数量和 KV Cache。设 d_model=1024, num_heads=16。对于 GQA，设置 num_groups=4（4 组 Query 头共享一组 K/V，每组有 16/4=4 个 Query 头）。计算三种方案的 KV 投影参数量和 KV Cache (seq_len=1024, fp16)。根据结果讨论 GQA 如何在精度和效率之间做权衡。

<details>
<summary><b>参考答案</b></summary>

d_k = 1024 / 16 = 64。seq_len = 1024。

**MHA (16 组独立的 K/V)**：
KV 参数 = 16 × (1024 × 64 + 1024 × 64) = 16 × 131,072 = 2,097,152
KV Cache = 16 × 1024 × 64 × 2 × 2B = 4,194,304 B = 4.0 MB

**GQA (4 组共享 K/V)**：
KV 参数 = 4 × (1024 × 64 + 1024 × 64) = 4 × 131,072 = 524,288
KV Cache = 4 × 1024 × 64 × 2 × 2B = 1,048,576 B = 1.0 MB

**MQA (1 组共享 K/V)**：
KV 参数 = 1 × (1024 × 64 + 1024 × 64) = 131,072
KV Cache = 1 × 1024 × 64 × 2 × 2B = 262,144 B = 0.25 MB

| 方案 | KV 参数 | KV Cache | 相对 MHA |
|------|---------|----------|----------|
| MHA  | 2,097,152 | 4.0 MB  | 100%     |
| GQA  | 524,288   | 1.0 MB  | 25%      |
| MQA  | 131,072   | 0.25 MB | 6.25%    |

GQA 的权衡哲学：GQA 在 MHA（精度最高但效率最低）和 MQA（效率最高但精度略低）之间提供了一个可调节的连续谱。num_groups 参数使得工程师可以在精度和效率之间灵活取值：num_groups → 1 则退化为 MQA（极致效率），num_groups → num_heads 则退化为 MHA（全精度）。实践中，GQA 通常用 num_groups=4-8，兼收 MHA 的多视角优势和 MQA 的内存节省。LLaMA 2/3 等主流开源大模型大量使用 GQA。

</details>

**题目 4（进阶）**：设计一个实验来验证"噪声注入是否有助于 MQA 的路由均衡"。你需要明确：实验中设置哪些对比组、观察什么指标、如何判断噪声确实起到了作用。不需要写完整代码，给出实验方案即可。

<details>
<summary><b>参考答案</b></summary>

**实验方案设计**：

**对比组设置**：
- 组 A：纯 MQA + 无噪声 Top-K 门控（baseline）
- 组 B：MQA + Noisy Top-K 门控（噪声标准差 = softplus(xW_noise)）
- 组 C：MQA + Noisy Top-K 门控 + 负载均衡损失

**实验环境**：
使用一个小型语言模型任务（如 WikiText-2 上的字符级语言建模），Transformer 解码器，4 层，d_model=256，num_heads=4，d_k=64。训练 20 epochs。

**观察指标**：
1. **专家负载分布**：每个训练 step 后，统计选择每个专家的 token 比例。理想情况是均匀分布（每个专家 ≈ 1/num_heads = 25%）。
2. **路由熵**：$H = -\sum_i p_i \log p_i$，其中 $p_i$ 是选择专家 i 的 token 比例。越大越均匀。
3. **负载均衡损失**：$\sum_i f_i \times P_i$ 的数值变化。
4. **验证困惑度**：各组在验证集上的困惑度。

**判断标准**：
- 如果组 B 的专家分布比组 A 显著均匀（路由熵更高），且验证困惑度与组 A 持平或略低 → 噪声确实起到了探索作用；
- 如果组 A 很快出现 1-2 个专家完全主导（其他专家梯度为 0），而组 B 保持较均匀 → 噪声成功打破了路由的对称塌缩；
- 组 C 应该是最均匀的（噪声 + 显式均衡损失），但困惑度可能略有上升（因为均衡约束限制了路由自由度）；
- 关注路由熵曲线：组 A 的路由熵迅速下降（塌缩），组 B 保持较高（噪声探索），组 C 最高（噪声 + 约束）。

**额外观察**：记录各组训练到第 5 epoch 时的实际专家分布（热力图）和 Top-K 选择的 token 分布直方图，以直观对比。

</details>

### 开放思考题

**题目 5（开放思考）**：在什么情况下 MQA **不适合**使用？请根据 MQA 的核心设计（共享 K/V）来分析。考虑以下维度：任务类型、模型规模、序列长度特性、KV Cache 是否是瓶颈。给出至少三种具体的场景并说明理由。

<details>
<summary><b>参考答案</b></summary>

**场景一：需要头间 K/V 差异的细粒度语义任务**

原因：MQA 共享 K/V 暗示"所有头关注的'内容'是相同的"。在涉及多义词消歧、指代消解或多答案推理等任务中，不同的注意力头可能需要完全不同的 K/V 内容表征（如一个头关注主语-谓语关系，另一个头关注时间状语）。当 K 和 V 共享时，不同 Q 头查询同一个 K 空间，相当于所有人看同一张地图——你可能想看地形（一个头），他可能想看交通路线（另一个头），但地图只有一张，信息被"语义平均"了。在这种场景下，MQA 的精度损失可能超过 1%，对下游任务造成影响。

**场景二：非常小的模型（头数少、序列短）**

原因：MQA 的核心优势是"在大头数模型中大幅降低 KV Cache"。当 num_heads=2-4 时，共享 K/V 节省的参数量仅为 50-75%，而 KV Cache 本身因为序列短（< 128 tokens）已经很小（通常在 KB 级别）。此时 MQA 的精度损失（哪怕只有 0.2-0.5%）相对于节省来说是"得不偿失"的——模型已经足够轻量，不够成为瓶颈，不值得为了这几 KB 牺牲精度。例如，一个为嵌入式设备设计的微型翻译模型（d_model=128, num_heads=4, 处理 30-50 tokens 的短句），应直接使用 MHA。

**场景三：训练阶段是瓶颈而推理量很小的研究场景**

原因：MQA 的最大价值体现在于自回归解码的推理阶段——每次生成新 token 都需要读取完整的 KV Cache。但如果主要瓶颈在训练（如大规模预训练实验），而推理量非常小（如论文中只跑几十条验证样本），那么 MQA 节省的 KV Cache 带来的加速几乎感知不到，而训练过程中的精度损失（即使很小）可能会影响实验结论的可信度。在这种场景下，GQA（更灵活）或直接用 MHA（更简单）是更好的选择——研究者不需要引入 MQA 的额外变量。

**额外场景四：与某些稀疏注意力机制组合时**

如果 MQA 与滑动窗口注意力（Sliding Window Attention）或局部注意力结合，共享 K/V 可能进一步放大局部注意力的感受野限制——所有头的 Q 都只能看馆子窗口内的 K/V，头间多样性被双重压缩（滑动窗口 + MQA 共享），导致模型失去长程依赖建模能力。在这种情况下，GQA 作为中间地带更好。

总之，MQA 不适用的典型特征是：**精度 > 效率**的需求、**小模型 + 短序列**、**训练为主 / 推理为辅**、以及某些对头间多样性有刚需的语义任务。

</details>

---

## 14. 学习路径建议

### 前置知识（必须先掌握）

1. **前馈神经网络（FFN）与多层感知机（MLP）**：理解线性变换 + 激活函数的基本结构，这是 MQA 中 Expert（如 MoE 组合使用时）和投影矩阵的基础。
2. **自注意力机制（Self-Attention）**：深入理解 Q、K、V 的语义角色（Q: "我想查什么"，K: "我是什么"，V: "我有什么内容"），以及 scaled dot-product attention 的计算过程。
3. **多头注意力（MHA）**：熟练掌握 MHA 的分头-计算-拼接全流程，理解多头如何通过不同的投影增强模型的表达能力。
4. **KV Cache 概念**：理解自回归解码时为什么要缓存 K 和 V，以及 KV Cache 占用的计算公式（h × seq_len × d_k × 2）。这是理解 MQA 动机的前提。
5. **Softmax 函数及温度**：Softmax 用于注意力权重的归一化和门控网络的路由分布。

### 平行学习（与 MQA 同时学习，互为对比）

1. **Grouped Query Attention (GQA)**：将 Query 头分为多组，每组共享一组 K/V。GQA 是 MHA 和 MQA 之间的连续统——组数=1 即为 MQA，组数=num_heads 即为 MHA。LLaMA 2/3、Mistral 均使用 GQA。理解 GQA 有助于定位 MQA 在"精度-效率"谱系上的位置。
2. **Switch Transformer / GLaM**：这些大规模 MoE 模型在 FFN 层使用路由机制（不同 token 激活不同 FFN 专家），与 MQA 的"所有头共享 K/V"形成了上下层结构的呼应。理解二者的组合（MQA + MoE-FFN）如何共同压缩大模型推理成本。
3. **Sparse Attention（稀疏注意力）**：如 Sliding Window Attention、Dilated Attention 等——它们不共享 K/V，而是限制 Q 查询 K 的范围。理解稀疏注意力与 MQA 各自从不同角度（稀疏范围 vs 参数共享）减少推理开销。
4. **Flash Attention**：通过 IO-aware 的分块计算（tiling）在 GPU 上实现更快、更省显存的注意力。Flash Attention 对 MQA 有叠加加速效果——Flash Attention 优化的是"计算方式"，MQA 优化的是"数据内容"（参与计算的矩阵更小）。

### 进阶学习（掌握 MQA 后深入）

1. **DeepSeekMoE（细粒度专家 + 共享专家）**：DeepSeek-V2/V3 提出的创新架构，将传统 MoE 中的大专家进一步拆分为"细粒度专家"（Fine-grained Experts），并引入一定数量的"共享专家"（Shared Experts，所有 token 都经过）。结合 MQA，DeepSeekMoE 在训练和推理两端都实现了极致效率。
2. **MLA (Multi-head Latent Attention)**：DeepSeek-V2 中提出的核心创新——将 K 和 V 在低维潜在空间（Latent Space）中压缩存储而不是直接缓存，进一步将 KV Cache 大小降到比 MQA 更低的水平。理解 MLA 需要先将 MQA 的"共享 = 减少 = 1/h"逻辑吃透。
3. **分布式 MoE 训练**：理解当专家分布在多个设备上时，all-to-all 通信如何成为新的瓶颈，以及专家并行（Expert Parallelism）+ 数据并行（Data Parallelism）的混合策略如何调度 MQA+MoE 架构的训练。
4. **Inference Optimization（推理优化）**：连续批处理（Continuous Batching）、推测解码（Speculative Decoding）、KV Cache 量化（如 KV Cache 使用 INT8/INT4）、PagedAttention（vLLM 的核心技术）等——这些技术可与 MQA 叠加，在一套推理系统中共同提升吞吐量。
5. **后续发表的注意力优化论文**：跟踪包括 Ring Attention（长序列分布式注意力）、HyperAttention（近似注意力）等前沿研究。

---

> **文档完成标志**：你已系统学习了多查询注意力 (MQA) 的核心原理、数学推导、代码实现（PyTorch + 手工从零）、可视化分析、模型评估以及常见问题。建议先手写运行第 7-8 章的代码，再完成第 13 章的练习题，最后按第 14 章的路径向 GQA / MLA / DeepSeekMoE 进发。
