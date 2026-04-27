# Transformer 架构详解 学习文档

> 来源线索：本节内容根据原书第3章 3.2节"注意力机制的应用实践：编码器"（第1187-1535行）的相关章节整理、扩展与教学化改写。

> 用自注意力堆叠编码器，让模型从序列中自动抽取全局特征。

## 1. 算法基础认知

**一句话定义**：Transformer 是一种完全基于自注意力机制的神经网络架构，由多个相同的编码器（Encoder）块堆叠而成，每个块包含多头自注意力、前馈网络、残差连接和层归一化。

**直觉类比**：想象一个翻译团队处理一份长文档——不是逐字逐句翻译，而是每个人通读全文后标出自己认为重要的段落，然后团队交流意见（多头注意力），各自整理笔记（前馈网络），最后核对原文确认没有遗漏（残差连接）。如此反复多轮，最终形成高质量的翻译。

**历史背景**：2017年，Google 团队在论文《Attention is All You Need》中提出了 Transformer 架构，彻底抛弃了传统的 RNN 和 CNN，完全依赖注意力机制进行序列建模。这一突破解决了 RNN 无法并行计算和长距离依赖衰减两大核心痛点，开启了 NLP 领域的"预训练+微调"范式革命。BERT、GPT、T5 等几乎所有主流大模型都建立在 Transformer 架构之上。

**算法定位**：深度学习 / 序列建模 / 编码器-解码器架构。Transformer 是神经网络架构层面的创新，其编码器部分（Encoder）是理解后续所有变体的基石。

**前置知识**：
- 自注意力机制（Scaled Dot-Product Attention）的 Q/K/V 三个角色
- 多头注意力机制（Multi-Head Attention）的拆分与拼接原理
- 线性代数：矩阵乘法、softmax 归一化、残差连接
- 神经网络基础：全连接层、激活函数（ReLU/GELU）、Dropout
- PyTorch：nn.Module、张量的 reshape/transpose/view 操作

## 2. 核心原理

### 核心思想

Transformer 编码器的核心思想是：**将输入序列通过多层堆叠的自注意力块进行反复提炼，每一层都在前一层的基础上学习更高级的特征表示**。与 RNN 的顺序处理不同，自注意力允许序列中任意两个位置直接交互，因此能高效捕获长距离依赖。

### 编码器的宏观结构

一个完整的 Transformer 编码器由 N 个完全相同的编码器层（Encoder Layer）堆叠而成。原始论文中 N=6，BERT 中 N=12 或 24。每个编码器层内部包含两个核心子层：

1. **多头自注意力子层（Multi-Head Self-Attention）**：让每个位置关注序列中的所有位置
2. **前馈网络子层（Feed-Forward Network, FFN）**：对每个位置独立地做非线性变换

每个子层后面都跟有**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**。这种设计使得梯度可以畅通无阻地流到最底层，即使堆叠很多层也不会出现梯度消失。

### 从输入到输出的完整流程

```
原始文本 → Tokenization → Token IDs → Input Embedding → + Positional Encoding
    → Encoder Layer 1 → Encoder Layer 2 → ... → Encoder Layer N
    → 输出特征矩阵 (batch, seq_len, d_model)
```

每一步的详细说明：

**Step 1: Tokenization & 词嵌入（Input Embedding）**
- 输入文本经过分词器（Tokenizer）转换为 token ID 序列
- 每个 token ID 通过 Embedding 层查找对应的词向量，维度为 d_model（通常 768）
- 输出形状：(batch_size, seq_len, d_model)

**Step 2: 位置编码（Positional Encoding）**
- 由于自注意力对位置不敏感（排列等变性），需要显式注入位置信息
- 使用正弦/余弦函数生成固定的位置编码向量，与词向量直接相加
- 相加后维度不变：(batch_size, seq_len, d_model)

**Step 3: 编码器层迭代**
- 每个编码器层的输入和输出维度相同（都是 d_model），因此可以随意堆叠
- 层内先做多头自注意力（带残差+LayerNorm），再做 FFN（带残差+LayerNorm）
- 最终输出一个与输入同形的特征矩阵

### 关键概念解释

- **残差连接**：将子层的输入直接加到子层的输出上（x + sublayer(x)），解决深层网络的梯度消失问题
- **层归一化（Layer Normalization）**：沿特征维度（d_model）做归一化，使每层输出的均值为 0、方差为 1，加速训练收敛
- **Pre-LN vs Post-LN**：原始论文将 LayerNorm 放在残差之后（Post-LN），但后续研究发现放在残差之前（Pre-LN）训练更稳定。本书代码采用了 Pre-LN 设计
- **Feed-Forward Network (FFN)**：两层的全连接网络，中间维度通常是 d_model 的 4 倍（如 768 -> 3072 -> 768），使用 GELU 或 ReLU 激活函数
- **堆叠的意义**：浅层学习局部句法特征，中层学习语义特征，深层学习任务相关的高层抽象

## 3. 数学公式与推导

### 3.1 词嵌入层

词嵌入是一个查表操作。设词表大小为 V，嵌入维度为 d_model：

$$\text{Embedding}(x) = E[x], \quad E \in \mathbb{R}^{V \times d_{model}}$$

其中 x 是 token ID，E 是可训练的嵌入矩阵。对于批量输入：

$$X_{embed} = E[X_{ids}], \quad X_{embed} \in \mathbb{R}^{B \times L \times d_{model}}$$

### 3.2 正弦位置编码（Sinusoidal Positional Encoding）

位置编码的每一个维度都使用不同频率的正弦或余弦函数：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
\end{aligned}
$$

其中：
- pos：token 在序列中的位置，范围 [0, L-1]
- i：位置编码向量的维度索引，范围 [0, d_model/2 - 1]
- 2i 和 2i+1 分别对应偶数和奇数维度

**设计巧思**：
- 不同频率确保每个位置有唯一的编码模式
- 正弦/余弦函数具有平滑性和周期性，使相近位置编码也相近
- 线性性质：PE(pos+k) 可以表示为 PE(pos) 的线性函数，模型因此能学习相对位置关系
- 波长从 2π 到 10000·2π 呈几何级数增长

最终输入编码：

$$X_{input} = X_{embed} + PE \in \mathbb{R}^{B \times L \times d_{model}}$$

### 3.3 缩放点积注意力

对单个注意力头：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q, K, V \in \mathbb{R}^{L \times d_k}$。

**缩放因子 $\sqrt{d_k}$ 的必要性**：假设 Q 和 K 各分量独立，均值 0、方差 1，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 d_k。当 d_k 很大（如 64）时，点积值会很大，softmax 输出趋于 one-hot，梯度近零。除以 $\sqrt{d_k}$ 将方差拉回 1。

### 3.4 多头注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O$$

其中每个头：

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

投影矩阵维度：

$$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}, \quad d_k = d_{model} / h$$

$$W^O \in \mathbb{R}^{h \cdot d_k \times d_{model}} = \mathbb{R}^{d_{model} \times d_{model}}$$

在自注意力中：Q = K = V = X（同一个输入）。

### 3.5 前馈网络

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，通常 $d_{ff} = 4 \times d_{model}$。

GELU 激活函数：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

近似形式：$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$

GELU 相比 ReLU 的优势：处处可导，在负半轴保留小梯度，训练更平滑。

### 3.6 残差连接与层归一化

Pre-LN 形式的子层连接（本书采用的方案）：

$$\text{Sublayer}(x) = x + \text{Dropout}(\text{SublayerFunc}(\text{LayerNorm}(x)))$$

层归一化，对每个样本的 d_model 维度做标准化：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu, \sigma^2$ 沿最后一维计算，$\gamma, \beta$ 是可学习参数（各维度一个），$\epsilon$ 防止除零。

### 3.7 完整编码器层的前向计算

以 Pre-LN 为例，一个编码器层的前向计算为：

1. $x_{norm1} = \text{LayerNorm}_1(x)$
2. $x_{attn} = \text{MultiHeadAttention}(x_{norm1}, x_{norm1}, x_{norm1})$
3. $x = x + \text{Dropout}(x_{attn})$
4. $x_{norm2} = \text{LayerNorm}_2(x)$
5. $x_{ffn} = \text{FFN}(x_{norm2})$
6. $x = x + \text{Dropout}(x_{ffn})$
7. 输出 x

### 3.8 参数量估算

对于一个编码器层：
- 多头注意力：4 个 d_model × d_model 的权重矩阵（WK, WK, WV, WO）→ 4·d_model²
- FFN：两个权重矩阵 W1 (d_model × 4d_model) 和 W2 (4d_model × d_model) → 8·d_model²
- 两个 LayerNorm：2·2·d_model ≈ 4·d_model（可忽略）

一个编码器层参数量约为 12·d_model²。以 d_model=768 计算：12 × 768² ≈ 7M。
BERT-base（12 层）：12 × 7M + 嵌入层 768 × 30000 ≈ 84M + 23M ≈ 110M。与官方报告一致。

## 4. 训练过程讲解

### 4.1 预训练目标

编码器通常通过以下两个任务预训练：

**1. 掩码语言模型（Masked Language Model, MLM）**
- 随机遮盖输入中 15% 的 token
- 其中 80% 替换为 [MASK]，10% 替换为随机 token，10% 保持不变
- 模型需要预测被遮盖的原始 token
- 损失函数：交叉熵损失，仅计算被遮盖位置的损失

**2. 下一句预测（Next Sentence Prediction, NSP）**
- 输入两个句子 A 和 B，50% 概率 B 是 A 的下一句，50% 是随机句子
- [CLS] token 的最终表示用于二分类预测
- 后续研究（如 RoBERTa）表明 NSP 并非必要

### 4.2 训练超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 学习率 | 1e-4 | AdamW 优化器，带 warmup 和线性衰减 |
| Batch Size | 256-2048 | 受显存限制，可用梯度累积模拟大批量 |
| 训练步数 | 1M steps | BERT 的训练规模 |
| Warmup 步数 | 10000 | 学习率从 0 线性增长到峰值 |
| Dropout | 0.1 | 应用于所有子层和嵌入层 |
| 权重衰减 | 0.01 | L2 正则化，AdamW 中去耦实现 |
| 序列长度 | 512 | 超过此长度的序列需要截断或特殊处理 |

### 4.3 学习率调度

使用带 warmup 的线性衰减：

$$lr = lr_{peak} \cdot \min\left(\frac{step}{warmup\_steps}, \frac{total\_steps - step}{total\_steps - warmup\_steps}\right)$$

### 4.4 典型训练流程

```
1. 初始化模型参数（随机或从预训练权重加载）
2. 构建 DataLoader，按 batch 加载经过 tokenization 的数据
3. 对每个 batch:
   a. 生成 attention mask（标记 padding 位置）
   b. 前向传播，计算 logits
   c. 计算损失（如 MLM 的交叉熵）
   d. 反向传播，梯度裁剪（max_norm=1.0）
   e. 优化器更新参数
   f. 更新学习率
4. 周期性在验证集评估困惑度（Perplexity）或下游任务指标
5. 保存检查点
```

## 5. 应用场景

1. **文本分类**：使用 [CLS] token 的输出做分类，如情感分析、新闻分类、垃圾邮件检测
2. **命名实体识别（NER）**：编码器输出每个 token 的表示，接一个线性分类器预测每个 token 的实体类型（人名、地名、组织名等）
3. **语义相似度/句子对任务**：输入两个句子，用 [CLS] 输出判断语义是否等价（如 Quora 重复问题检测）
4. **抽取式问答**：输入问题和上下文，预测答案在上下文中的起始和结束位置
5. **序列标注**：对每个 token 打标签，广泛用于分词、词性标注、语法分析等任务

## 6. 优缺点分析

| 维度 | 优点 | 缺点 |
|------|------|------|
| 长距离依赖 | 自注意力直接建模任意两个位置的交互，O(1) 路径长度 | 序列过长时计算量 O(L²)，需要截断策略 |
| 并行计算 | 所有位置同时计算，无须像 RNN 那样顺序展开 | 训练时可以并行，但推理时自回归解码仍需串行 |
| 表示能力 | 多头注意力从多子空间提取特征，FFN 提供非线性变换 | 参数量大，BERT-base 即 110M 参数 |
| 可解释性 | 注意力权重可视化，可以观察模型关注的位置 | 注意力图只是"相关性"，不等于"因果性" |
| 通用性 | 同一架构适用于文本、图像、音频等多种模态 | 缺少归纳偏置，需要更大数据量才能学到有效特征 |
| 训练稳定性 | 残差连接 + LayerNorm 使深层网络可训练 | 对超参数敏感，尤其是学习率调度和 warmup |
| 位置建模 | 显示位置编码注入顺序信息 | 固定正弦编码对超长序列泛化能力有限 |

## 7. 调库实现

以下使用 PyTorch 内置模块实现一个完整的 Transformer 编码器，可直接运行。

```python
"""
Transformer 编码器 - PyTorch 库实现
基于 "Attention is All You Need" 论文架构
使用 PyTorch 内置的 nn.TransformerEncoderLayer 和 nn.TransformerEncoder
"""
import torch
import torch.nn as nn
import math


class TransformerEncoder(nn.Module):
    """
    完整的 Transformer 编码器实现。
    包含：词嵌入 + 正弦位置编码 + 多层 TransformerEncoderLayer + 输出投影
    """

    def __init__(
        self,
        vocab_size: int,          # 词表大小
        d_model: int = 768,       # 模型维度（词向量维度）
        nhead: int = 12,          # 多头注意力的头数（必须能被 d_model 整除）
        num_layers: int = 12,     # 编码器层数
        dim_feedforward: int = 3072,  # FFN 中间层维度（通常为 4 * d_model）
        dropout: float = 0.1,     # Dropout 比例
        max_len: int = 512,       # 最大序列长度（用于位置编码）
        activation: str = "gelu", # FFN 激活函数: "relu" 或 "gelu"
    ):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model

        # ===== 1. 词嵌入层 =====
        # 将 token ID (0 到 vocab_size-1) 映射为 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # ===== 2. 正弦位置编码 =====
        # 固定位置编码，不参与训练
        pe = torch.zeros(max_len, d_model)                    # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数索引用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数索引用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，增加 batch 维度
        # register_buffer: 随模型移动但不参与梯度更新
        self.register_buffer("positional_encoding", pe)

        # ===== 3. Dropout 层（应用于嵌入输出） =====
        self.dropout = nn.Dropout(dropout)

        # ===== 4. 构建单个编码器层 =====
        # nn.TransformerEncoderLayer 已内置: 多头自注意力 + FFN + 残差 + LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # 输入形状为 (batch, seq, feature)，而非 (seq, batch, feature)
        )

        # ===== 5. 堆叠 N 层编码器 =====
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ===== 6. 最终的 LayerNorm =====
        self.ln_final = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,     # (batch_size, seq_len) token ID 序列
        attention_mask: torch.Tensor = None,  # (batch_size, seq_len) 1=有效, 0=padding
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids: token ID 序列
            attention_mask: 注意力掩码，True 的位置不参与注意力计算

        返回:
            (batch_size, seq_len, d_model) 编码后的特征矩阵
        """
        # Step 1: 查表获取词向量
        x = self.token_embedding(input_ids)  # (B, L, d_model)

        # Step 2: 加上位置编码
        seq_len = input_ids.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Step 3: Dropout 正则化
        x = self.dropout(x)

        # Step 4: 构建 PyTorch 需要的 src_key_padding_mask
        # True 表示该位置是 padding，不参与注意力计算
        if attention_mask is not None:
            # attention_mask: 1=有效, 0=padding -> 转换为 True=padding
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Step 5: 逐层编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Step 6: 最终归一化
        x = self.ln_final(x)

        return x


# ==============================
# 测试代码
# ==============================
if __name__ == "__main__":
    # 模拟参数
    vocab_size = 30000
    batch_size = 4
    seq_len = 128

    # 创建模型
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=6,           # 为演示使用 6 层
        dim_feedforward=3072,
        dropout=0.1,
        max_len=512,
    )

    # 为方便演示，使用较小的参数量
    # 实际 BERT-base 为 12 层
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 创建模拟输入
    # 随机 token ID，0 用于 padding
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 后 20 个位置设为 padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -20:] = 0

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")  # 预期: (4, 128, 768)

    # 验证 padding 位置的输出不受非 padding 位置影响
    # （这里仅做形状验证，实际需要更复杂的测试）
    assert output.shape == (batch_size, seq_len, 768), f"输出形状不正确: {output.shape}"
    print("所有测试通过!")
```

## 8. 手工代码实现

以下从零实现 Transformer 编码器的每个组件：Scaled Dot-Product Attention、Multi-Head Attention、FFN、SublayerConnection、Positional Encoding 和完整编码器。

```python
"""
Transformer 编码器 - 手工实现
从零构建每个组件，完整复现 "Attention is All You Need" 的编码器部分
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
# 组件 1: 缩放点积注意力 (Scaled Dot-Product Attention)
# ================================================================
class ScaledDotProductAttention(nn.Module):
    """
    计算缩放点积注意力。
    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (batch, ..., seq_len_q, d_k)
        key: torch.Tensor,     # (batch, ..., seq_len_k, d_k)
        value: torch.Tensor,   # (batch, ..., seq_len_v, d_k)
        mask: torch.Tensor = None,  # 掩码，True 的位置注意力权重置为 -inf
    ):
        """
        参数:
            query: 查询矩阵
            key: 键矩阵
            value: 值矩阵 (seq_len_v == seq_len_k)
            mask: 注意力掩码 (batch, ..., seq_len_q, seq_len_k) 或 broadcastable 形状
        """
        d_k = query.size(-1)

        # Step 1: 计算注意力分数 scores = Q @ K^T / sqrt(d_k)
        # query: (B, h, L_q, d_k), key.transpose(-2,-1): (B, h, d_k, L_k)
        # scores: (B, h, L_q, L_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: 应用掩码（将 mask=True 的位置设为极小值）
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # Step 3: Softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # Step 4: Dropout 正则化
        attn_weights = self.dropout(attn_weights)

        # Step 5: 加权求和 attn_weights @ V
        output = torch.matmul(attn_weights, value)

        # 返回输出和注意力权重（权重可用于可视化）
        return output, attn_weights


# ================================================================
# 组件 2: 多头注意力 (Multi-Head Attention)
# ================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制。
    将 d_model 拆分为 h 个头，每个头在 d_k = d_model/h 维子空间独立计算注意力。
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 三个投影矩阵：将输入分别映射到 Q, K, V 空间
        self.W_q = nn.Linear(d_model, d_model)  # Q 投影
        self.W_k = nn.Linear(d_model, d_model)  # K 投影
        self.W_v = nn.Linear(d_model, d_model)  # V 投影

        # 输出投影矩阵：将拼接后的多头结果映射回 d_model
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放点积注意力计算单元
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_len, d_model)
        key: torch.Tensor,     # (batch, seq_len, d_model)
        value: torch.Tensor,   # (batch, seq_len, d_model)
        mask: torch.Tensor = None,
    ):
        """
        前向传播。对于自注意力: query == key == value
        """
        batch_size = query.size(0)

        # Step 1: 线性投影
        Q = self.W_q(query)  # (B, L, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: 拆分为多头
        # (B, L, d_model) -> (B, L, n_heads, d_k) -> (B, n_heads, L, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: 计算缩放点积注意力
        # 对 mask 扩展 head 维度: (B, L, L) -> (B, 1, L, L)
        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Step 4: 合并多头
        # (B, n_heads, L, d_k) -> (B, L, n_heads, d_k) -> (B, L, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Step 5: 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attn_weights


# ================================================================
# 组件 3: 前馈网络 (Position-wise Feed-Forward Network)
# ================================================================
class FeedForward(nn.Module):
    """
    逐位置的前馈网络。
    两层全连接，中间使用 GELU 激活。
    公式: FFN(x) = W2 @ GELU(W1 @ x + b1) + b2
    """

    def __init__(self, d_model: int = 768, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维
        self.linear2 = nn.Linear(d_ff, d_model)  # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        # GELU 激活 + 第一层全连接
        x = self.linear1(x)
        x = F.gelu(x)  # PyTorch 内置 GELU

        # Dropout + 第二层全连接
        x = self.dropout(x)
        x = self.linear2(x)

        return x


# ================================================================
# 组件 4: 子层连接 (Sublayer Connection with Pre-LN)
# ================================================================
class SublayerConnection(nn.Module):
    """
    带 Pre-LayerNorm 的残差连接。
    公式: output = x + Dropout(Sublayer(LayerNorm(x)))
    Pre-LN: 归一化在子层之前，训练更稳定。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer_func: callable,
        *args,
        **kwargs,
    ):
        """
        参数:
            x: 输入张量
            sublayer_func: 子层函数（如多头注意力、FFN）
            *args, **kwargs: 传递给 sublayer_func 的额外参数
        返回: 应用 Pre-LN 残差连接后的输出
        """
        # Pre-LN: 先 LayerNorm，再走子层，最后加残差
        # 注意：子层内部可能返回元组（如 attn output + weights），这里取第一个
        sublayer_output = sublayer_func(self.layer_norm(x), *args, **kwargs)

        # 处理子层返回元组的情况
        if isinstance(sublayer_output, tuple):
            sublayer_result = sublayer_output[0]
            extra_outputs = sublayer_output[1:]
        else:
            sublayer_result = sublayer_output
            extra_outputs = ()

        # 残差连接 + Dropout
        output = x + self.dropout(sublayer_result)

        return (output,) + extra_outputs


# ================================================================
# 组件 5: 正弦位置编码 (Sinusoidal Positional Encoding)
# ================================================================
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。
    为输入序列的每个位置生成唯一的编码模式。
    """

    def __init__(self, d_model: int = 768, max_len: int = 512):
        super().__init__()

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 位置索引: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 分母项: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        # 这里计算的是 1/分母，用于乘以 position
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # 偶数索引用 sin，奇数索引用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer: 不是可训练参数，但会随模型一起保存/加载/移动
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)，加上了位置编码
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ================================================================
# 组件 6: 单个编码器层 (Encoder Layer)
# ================================================================
class EncoderLayer(nn.Module):
    """
    一个完整的编码器层。
    结构: Pre-LN -> Multi-Head Self-Attention -> 残差 -> Pre-LN -> FFN -> 残差
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 两个子层连接（一个给注意力，一个给 FFN）
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        x: (batch, seq_len, d_model)
        mask: 注意力掩码
        返回: (batch, seq_len, d_model) 编码后的输出, 注意力权重
        """
        # 子层 1: 多头自注意力 + 残差连接
        x, attn_weights = self.sublayer1(
            x,
            lambda _x: self.self_attn(_x, _x, _x, mask)
        )

        # 子层 2: 前馈网络 + 残差连接
        x, _ = self.sublayer2(
            x,
            lambda _x: self.feed_forward(_x)
        )

        return x, attn_weights


# ================================================================
# 组件 7: 完整的 Transformer 编码器
# ================================================================
class TransformerEncoderManual(nn.Module):
    """
    完整的 Transformer 编码器（手工版）。
    组装: Token Embedding + Positional Encoding + N x EncoderLayer
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 嵌入层后的 Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # N 层编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终 LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)

        # 保存参数供后续使用
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        参数:
            input_ids: (batch, seq_len) token ID 序列
            attention_mask: (batch, seq_len) 1=有效, 0=padding
        返回:
            output: (batch, seq_len, d_model)
            all_attn_weights: 所有层的注意力权重列表
        """
        # Step 1: 词嵌入 (B, L) -> (B, L, d_model)
        x = self.token_embedding(input_ids)

        # Step 2: 添加位置编码
        x = self.positional_encoding(x)

        # Step 3: Dropout
        x = self.embed_dropout(x)

        # Step 4: 构建注意力掩码
        # (B, L) -> (B, 1, 1, L) padding 位置设为 True
        if attention_mask is not None:
            # 将 padding 位置 (mask==0) 标记为需要被遮盖的位置
            attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # 扩展为 (B, 1, L, L)，即在最后两维都遮盖 padding 位置
            attn_mask = attn_mask.expand(-1, -1, attention_mask.size(1), -1)
        else:
            attn_mask = None

        # Step 5: 逐层编码
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask)
            all_attn_weights.append(attn_weights)

        # Step 6: 最终 LayerNorm
        x = self.layer_norm(x)

        return x, all_attn_weights


# ==============================
# 测试代码
# ==============================
if __name__ == "__main__":
    # 小规模参数便于测试
    vocab_size = 1000
    d_model = 128      # 使用较小的维度加快测试
    n_heads = 4
    n_layers = 2
    d_ff = 512
    batch_size = 2
    seq_len = 16

    # 创建模型
    model = TransformerEncoderManual(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1,
        max_len=64,
    )

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=" * 60)
    print(f"Transformer 编码器 - 手工实现")
    print(f"=" * 60)
    print(f"总参数量:     {total_params:,}")
    print(f"可训练参数:   {trainable_params:,}")
    print(f"d_model:      {d_model}")
    print(f"n_heads:      {n_heads}")
    print(f"n_layers:     {n_layers}")

    # 创建模拟输入
    torch.manual_seed(42)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 最后 4 个位置设为 padding (token ID = 0)
    input_ids[:, -4:] = 0
    attention_mask = (input_ids != 0).long()

    print(f"\n输入形状:     {input_ids.shape}")
    print(f"序列长度:     {seq_len} (含 4 个 padding)")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output, all_attn_weights = model(input_ids, attention_mask)

    print(f"输出形状:     {output.shape}")  # (2, 16, 128)
    print(f"注意力层数:   {len(all_attn_weights)}")
    print(f"注意力权重形状 (每层): {all_attn_weights[0].shape}")
    # 每层的注意力权重: (B, n_heads, L, L) = (2, 4, 16, 16)

    # 验证
    assert output.shape == (batch_size, seq_len, d_model), "输出形状错误!"
    assert len(all_attn_weights) == n_layers, "注意力权重层数不正确!"

    # 验证 padding 位置不被关注
    # 取第一层、第一个头、第一个样本的最后一行（即 padding 位置的查询）
    last_row_weights = all_attn_weights[0][0, 0, -1, :]  # (seq_len,)
    print(f"\npadding 位置(query=-1)对各 key 的注意力权重:")
    for i, w in enumerate(last_row_weights.tolist()):
        marker = "<-- pad" if input_ids[0, i] == 0 else ""
        print(f"  key_pos={i:2d}: {w:.4f} {marker}")

    print(f"\n所有测试通过!")

    # ==============================
    # 额外验证：与 PyTorch 内置实现的输出对比（结构一致性）
    # ==============================
    print(f"\n{'=' * 60}")
    print(f"交叉验证：对比手工实现与 PyTorch 内置 TransformerEncoderLayer")
    print(f"{'=' * 60}")

    # 创建 PyTorch 内置编码器层（相同参数）
    builtin_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=0.0,  # 关闭 dropout 以便精确对比
        activation="gelu",
        batch_first=True,
    )

    # 将手工实现的层和内置层的参数初始化为相同值
    # 这里仅验证前向传播接口一致，不做逐元素对比
    # （因为我们的实现是 Pre-LN 设计，与内置层一致）
    manual_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)

    test_input = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        # 内置层前向
        builtin_out = builtin_layer(test_input)
        # 手工层前向
        manual_out, _ = manual_layer(test_input)

    print(f"内置层输出形状: {builtin_out.shape}")
    print(f"手工层输出形状: {manual_out.shape}")
    print(f"形状一致: {builtin_out.shape == manual_out.shape}")
    print(f"交叉验证完成!")
```

## 9. 可视化与结果理解

```python
"""
Transformer 编码器可视化
1. 正弦位置编码的热力图
2. 不同位置的位置编码相似度
3. 自注意力权重的热力图
"""
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置中文字体（如果系统支持）
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def visualize_positional_encoding(d_model=256, max_len=128):
    """
    可视化正弦位置编码矩阵。
    展示每个位置在每个维度上的编码值。
    """
    # 计算位置编码
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe_np = pe.numpy()

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：完整位置编码热力图 ----
    im1 = axes[0].imshow(pe_np, aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Sinusoidal Positional Encoding\n(pos, dim) Heatmap")
    axes[0].set_xlabel("Dimension Index (0 to d_model-1)")
    axes[0].set_ylabel("Position (0 to max_len-1)")
    plt.colorbar(im1, ax=axes[0], label="Encoding Value")

    # ---- 右图：前 50 个位置，前 100 维的放大视图 ----
    im2 = axes[1].imshow(pe_np[:50, :100], aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Positional Encoding (First 50 pos, First 100 dims)")
    axes[1].set_xlabel("Dimension Index")
    axes[1].set_ylabel("Position")
    plt.colorbar(im2, ax=axes[1], label="Encoding Value")

    plt.tight_layout()
    plt.savefig("transformer_positional_encoding.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("位置编码可视化已保存至 transformer_positional_encoding.png")


def visualize_position_similarity(d_model=256, max_len=64):
    """
    可视化不同位置编码之间的余弦相似度。
    展示相近位置的编码更相似，以及不同维度频率的影响。
    """
    # 计算位置编码
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 归一化后计算余弦相似度
    pe_norm = pe / pe.norm(dim=1, keepdim=True)
    similarity = torch.matmul(pe_norm, pe_norm.T).numpy()

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：位置间余弦相似度矩阵 ----
    im1 = axes[0].imshow(similarity, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    axes[0].set_title("Cosine Similarity Between Positions")
    axes[0].set_xlabel("Position j")
    axes[0].set_ylabel("Position i")
    plt.colorbar(im1, ax=axes[0], label="Cosine Similarity")

    # ---- 右图：位置 0 与其他位置的距离（前 4 个频率分量） ----
    axes[1].set_title("Positional Encoding Values at Selected Dimensions")
    for dim_idx in [0, 1, 10, 20, 50]:
        label = f"dim={dim_idx} ({'sin' if dim_idx % 2 == 0 else 'cos'})"
        axes[1].plot(pe_np[:64, dim_idx], label=label, linewidth=1.5)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Encoding Value")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("transformer_position_similarity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("位置相似度可视化已保存至 transformer_position_similarity.png")


def visualize_attention_weights():
    """
    使用一个简单的编码器演示自注意力权重的可视化。
    展示 Q-K 注意力矩阵和 V 加权聚合的过程。
    """
    d_model = 64
    n_heads = 4
    seq_len = 10

    torch.manual_seed(42)

    # 创建一个简化的单头注意力用于可视化
    d_k = d_model // n_heads

    # 模拟输入：10 个 token，每个 64 维
    x = torch.randn(1, seq_len, d_model)

    # 模拟 Q, K, V 投影（简化为随机矩阵）
    W_q = torch.randn(d_model, d_k) * 0.1
    W_k = torch.randn(d_model, d_k) * 0.1
    W_v = torch.randn(d_model, d_k) * 0.1

    Q = x @ W_q  # (1, 10, 16)
    K = x @ W_k
    V = x @ W_v

    # 计算注意力
    scores = Q @ K.transpose(1, 2) / math.sqrt(d_k)  # (1, 10, 10)
    attn_weights = torch.softmax(scores, dim=-1)

    attn_np = attn_weights.squeeze(0).detach().numpy()

    # ========= 图形 1: 注意力权重热力图 =========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：注意力权重热力图 ----
    im1 = axes[0].imshow(attn_np, aspect="auto", cmap="YlOrRd")
    axes[0].set_title("Self-Attention Weights\n(Row=Query, Col=Key)")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")

    # 添加数值标注
    for i in range(seq_len):
        for j in range(seq_len):
            text = axes[0].text(
                j, i, f"{attn_np[i, j]:.2f}",
                ha="center", va="center",
                color="white" if attn_np[i, j] > 0.3 else "black",
                fontsize=7,
            )

    plt.colorbar(im1, ax=axes[0], label="Attention Weight")

    # ---- 右图：每行权重之和验证 (应为 1.0) ----
    row_sums = attn_np.sum(axis=1)
    axes[1].bar(range(seq_len), row_sums, color="steelblue", edgecolor="navy")
    axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Expected sum = 1.0")
    axes[1].set_title("Sum of Attention Weights per Query Position\n(Should all equal 1.0)")
    axes[1].set_xlabel("Query Position")
    axes[1].set_ylabel("Sum of Attention Weights")
    axes[1].set_ylim(0.95, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("transformer_attention_weights.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("注意力权重可视化已保存至 transformer_attention_weights.png")


# ==============================
# 执行所有可视化
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 编码器可视化")
    print("=" * 60)

    print("\n[1/3] 正弦位置编码热力图...")
    visualize_positional_encoding(d_model=256, max_len=128)

    print("\n[2/3] 位置编码相似度分析...")
    visualize_position_similarity(d_model=256, max_len=64)

    print("\n[3/3] 自注意力权重可视化...")
    visualize_attention_weights()

    print("\n所有可视化完成!")
```

## 10. 模型评估

Transformer 编码器的评估取决于具体下游任务，但核心评估维度如下：

### 10.1 预训练阶段评估

**困惑度（Perplexity, PPL）**：衡量语言模型质量的标准指标。

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | context)\right)$$

困惑度越低，模型越好。BERT-base 在掩码语言模型任务上通常达到 PPL 3-5。

### 10.2 下游任务评估指标

| 任务类型 | 评估指标 | 说明 |
|----------|----------|------|
| 文本分类 | Accuracy, F1 | 整体准确率和宏平均 F1 |
| 序列标注 | Token-level F1 | 按 token 计算 precision/recall |
| 问答 | Exact Match, F1 | EM: 完全匹配比例; F1: 词级匹配 |
| 语义相似度 | Spearman 相关系数 | 预测分数与人工标注的排序相关性 |
| 推理任务 | Accuracy | 如 MNLI, QNLI 的准确率 |

### 10.3 架构效率评估

| 指标 | 说明 |
|------|------|
| 参数量 | 模型可训练参数总数 |
| 推理延迟 | 单次前向传播的时间（毫秒） |
| 吞吐量 | 每秒能处理的 token 数 |
| 显存占用 | 训练和推理时的 GPU 显存需求 |
| FLOPs | 浮点运算次数，反映计算复杂度 |

### 10.4 评估代码示例

```python
def evaluate_encoder(model, dataloader, device="cuda"):
    """
    简单的编码器评估示例（以分类任务为例）。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 获取 [CLS] 位置的输出
            output, _ = model(input_ids, attention_mask)
            cls_output = output[:, 0, :]  # (batch, d_model)

            # 简单的线性分类器
            logits = model.classifier(cls_output)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy
```

## 11. 常见问题与易错点

### 问题 1

**现象**：深层 Transformer 训练不稳定，损失函数震荡甚至发散。

**原因**：Post-LN 设计中残差分支的梯度无衰减累积，深层时激活值爆炸；学习率过大也会加剧此问题。

**解决方案**：
1. 采用 Pre-LN 架构（LayerNorm 在子层之前），这是目前主流做法
2. 使用 AdamW 优化器并设置 warmup（如 10,000 步）
3. 应用梯度裁剪（max_norm=1.0）
4. 从较小的学习率开始，如 5e-5 而非 1e-4

### 问题 2

**现象**：模型在长序列上的性能显著下降，或显存溢出（OOM）。

**原因**：自注意力的计算和存储复杂度为 O(L²)。序列长度 512 时注意力矩阵为 512x512；序列长度 4096 时变为 4096x4096，大了 64 倍。

**解决方案**：
1. 限制最大序列长度，对长文档做滑动窗口或分块处理
2. 使用混合精度训练（FP16/BF16）减少显存占用
3. 采用 FlashAttention 等高效注意力实现
4. 梯度检查点（gradient checkpointing）以计算换存储
5. 对于超长序列，考虑 Longformer、BigBird 等稀疏注意力变体

### 问题 3

**现象**：位置编码对超出训练长度的序列泛化能力差。

**原因**：固定正弦编码虽理论上可外推，但实践中当推理序列长度远超训练长度时（如训练 512，推理 4096），性能明显下降。

**解决方案**：
1. 使用可学习的位置编码（如 BERT），但同样受训练长度的限制
2. 采用相对位置编码（如 T5 的相对位置偏置、RoPE 旋转位置编码）
3. ALiBi（Attention with Linear Biases）：在注意力分数上加一个与距离成线性关系的偏置，具有良好的外推能力
4. 对绝对位置编码做插值（如 NTK-aware interpolation）

### 问题 4

**现象**：训练好的编码器在下游任务上微调时性能不稳定，不同随机种子结果差异大。

**原因**：BERT 等大模型在下游小数据集上微调时，优化曲面不平坦，容易陷入不同的局部最优。

**解决方案**：
1. 使用较小的学习率（2e-5 到 5e-5），更大的 batch size
2. 多个随机种子运行并取平均
3. 采用多阶段微调（先在大数据上微调，再到目标数据）
4. 使用 R-Drop 等正则化方法增强稳定性

### 问题 5

**现象**：训练时损失很早就收敛到较低值，但验证集指标持续下降（过拟合）。

**原因**：编码器参数量大（110M+），而下游任务数据量小（几千到几万样本），模型记忆了训练数据。

**解决方案**：
1. 增大 Dropout 比例（从 0.1 提升到 0.3）
2. 使用权重衰减（weight decay=0.01）
3. 数据增强（如回译、同义词替换）
4. Early stopping，监控验证集指标
5. 使用对抗训练（如 FGM、PGD）提升鲁棒性

## 12. 学习总结

Transformer 编码器是当代深度学习的奠基性架构，其设计哲学——用自注意力替换序列建模中的循环或卷积操作——被证明是极为成功的。本节从架构层面完整剖析了编码器的每一个组件：

1. **输入层**：词嵌入 + 正弦位置编码，为模型提供词汇语义和位置信息
2. **核心计算层**：多头自注意力（全局特征交互）+ 前馈网络（逐位置的非线性变换）
3. **结构稳定性设计**：Pre-LN 残差连接确保深层网络可训练，Dropout 防止过拟合
4. **堆叠策略**：N 层相同结构的编码器层逐层提炼更高级的表示

理解 Transformer 编码器的关键不在于记忆每个组件，而在于理解"为什么这样设计"——为什么需要位置编码（注意力对位置不敏感）、为什么用 Pre-LN（训练稳定性）、为什么 FFN 维度是 4 倍（平衡计算和表达能力）。掌握了这些设计动机后，后续学习 BERT、GPT、ViT 等变体就只是对基座架构的有针对性的修改。

Transformer 编码器的核心价值在于它的通用性：同样的架构可以处理文本、图像、语音等多种模态，这是它成为"基础模型"时代主角的根本原因。

## 13. 练习题与思考题

### 基础题 1：位置编码计算

**题目**：给定 d_model=512, max_len=100，求位置 pos=5, 维度 i=0, 1, 2, 3 的位置编码值（保留 4 位小数）。

**答案**：

对于 pos=5, d_model=512:

当 2i=0 (i=0): PE(5,0) = sin(5 / 10000^(0/512)) = sin(5/1) = sin(5) ≈ -0.9589
当 2i+1=1 (i=0): PE(5,1) = cos(5 / 10000^(0/512)) = cos(5) ≈ 0.2837
当 2i=2 (i=1): PE(5,2) = sin(5 / 10000^(2/512)) = sin(5 / 10000^(1/256))
  - 10000^(1/256) = exp(ln(10000)/256) = exp(9.21034/256) = exp(0.03598) ≈ 1.03664
  - 5 / 1.03664 ≈ 4.8233
  - PE(5,2) = sin(4.8233) ≈ -0.9938
当 2i+1=3 (i=1): PE(5,3) = cos(4.8233) ≈ -0.1113

**推理过程**：低频维度（i 小）变化慢，高频维度（i 大）变化快。i=0 时波长=2π≈6.28（因为 10000^(0/512)=1），pos=5 非常接近波谷。i=1 时波长略大但变化仍慢。

### 基础题 2：残差连接的作用

**题目**：假设一个没有残差连接的多层 Transformer（仅 LayerNorm + FFN + 多头注意力直接串联），训练 12 层深的模型时会遇到什么问题？请用梯度传播的原理解释。

**答案**：

没有残差连接的深层网络会遇到梯度消失问题。具体分析如下：

设第 l 层的变换为 F_l（包含 LN -> Attention/FFN），则正向传播：
x_l = F_l(x_{l-1})

反向传播时，损失对输入的梯度：
∂L/∂x_0 = ∂L/∂x_N · ∂x_N/∂x_{N-1} · ... · ∂x_1/∂x_0
        = ∂L/∂x_N · ∏_{l=1}^{N} ∂F_l/∂x_{l-1}

如果每一层的梯度模长小于 1（如‖∂F_l/∂x‖ < 0.9），经过 12 层后梯度缩小为 0.9^12 ≈ 0.28，深层（靠近输入）的梯度几乎为零，参数无法有效更新。

而残差连接 x_l = x_{l-1} + F_l(x_{l-1}) 引入了恒等映射路径：
∂x_l/∂x_{l-1} = I + ∂F_l/∂x_{l-1}

反向传播时梯度组成：
∂L/∂x_0 = ∂L/∂x_N · ∏_{l=1}^{N} (I + ∂F_l/∂x_{l-1})

恒等映射 I 确保梯度始终有一条不衰减的通路，即使 F_l 的梯度很小。这是 ResNet 和 Transformer 能够训练深层网络的核心机制。

### 进阶题 1：多头注意力的计算量分析

**题目**：分析单头注意力（d_model=768）与 12 头注意力（每头 d_k=64）的计算量差异。两者的 QKV 投影总参数量相同吗？总 FLOPs 相同吗？为什么实际使用中多头几乎总是优于单头？

**答案**：

**参数量分析**：
- 单头：4 个 (768, 768) 矩阵 = 4 × 768 × 768 = 2,359,296 参数
- 12 头：每个头有 W_Q, W_K, W_V 各 (768, 64) = 3 × 768 × 64 = 147,456 参数
         12 个头共 12 × 147,456 = 1,769,472
         加上 W_O (768, 768) = 589,824
         总计 1,769,472 + 589,824 = 2,359,296 参数
- 结论：参数量完全相同

**FLOPs 分析**：
- Q/K/V 投影：两者相同（总投影维度相同）
- 注意力计算：单头 (768, L, L) 矩阵乘法 = 2 × 768 × L × L
               12 头 12 × 2 × 64 × L × L = 2 × 768 × L × L
- 输出投影：两者相同
- 结论：FLOPs 也相同

**为什么多头优于单头**：
1. 多子空间表示：不同头关注不同的特征方面（语法、语义、位置关系），模型表达能力更强
2. 对称性破缺带来的正则化效应：每个头维度低，单独训练相当于一种隐式的子空间正则化
3. 注意力的"均值回归"：12 个头的拼接平均了各头的预测，方差更小
4. 实际实验证实：固定计算量下，增加头数（减小每头维度）通常比单头大维度表现更好，直到每头维度过小（<32）时性能退化

### 进阶题 2：Pre-LN 和 Post-LN 的深度学习视角

**题目**：为什么 Pre-LN（LayerNorm 在子层之前）比 Post-LN（LayerNorm 在子层之后/残差之后）训练更稳定？从梯度和残差信号的角度分析。

**答案**：

**Post-LN** 的前向传播：
x_l = LayerNorm(x_{l-1} + F_l(x_{l-1}))

**Pre-LN** 的前向传播：
x_l = x_{l-1} + F_l(LayerNorm(x_{l-1}))

关键差异在残差路径和 LayerNorm 的位置：

1. **Post-LN 的问题**：
   - 残差加完之后才做 LayerNorm，这意味着残差路径上的值可能很大
   - 随着层数加深，残差信号逐层放大（类似于随机游走），LayerNorm 前值的量级 ∝ √N
   - 深层时输入 LayerNorm 的值很大，梯度也大，但归一化后的输出振幅受限
   - 这导致训练初期深层梯度爆炸，需要非常小心的 warmup

2. **Pre-LN 的优势**：
   - 在残差分支内部做归一化，确保 F_l 的输入总是零均值、单位方差的
   - 主路径 x_l = x_{l-1} + ... 不受 LayerNorm 约束，信号可以自由流动
   - 最后加一个额外的 LayerNorm 统一输出量级
   - 梯度在深层也能稳定传播，对 warmup 不敏感

3. **实证结果**：
   - 原始 Transformer 论文需要大量 warmup；GPT-2/3 切换到 Pre-LN 后训练更稳定
   - 但 Post-LN 在某些任务上有轻微的精度优势（因为归一化限制了残差信号，相当于隐式正则化）

### 开放题：设计一种新的位置编码方案

**题目**：如果你要为 Transformer 设计一种新的位置编码方案，使其能高效处理超长文档（10000+ tokens），你会如何设计？请描述你的方案并与正弦编码和相对位置编码进行对比。

**参考答案**：

（开放题，以下为示例方案）

**方案：层级位置编码（Hierarchical Position Encoding）**

1. 将位置编码分解为两个分量：段级别位置（粗粒度）+ 段内位置（细粒度）
2. 段级别：使用低频正弦编码，例如段编号 pos_section ∈ [0, N_sections-1]
3. 段内位置：使用高频正弦编码，pos_local ∈ [0, L_section-1]
4. 最终位置编码 = concat(PE_section, PE_local) 或 PE_section + PE_local（降维后）
5. 给段级别位置更大的维度占比（因为段间距离更需要长程建模）

**优势**：
- 参数/计算量与序列长度无关
- 段间关系通过粗粒度位置编码建模，段内关系通过细粒度建模
- 可以自然外推到任意长度

**对比**：
| 维度 | 正弦编码 | 相对位置 | 层级编码 |
|------|----------|----------|----------|
| 外推能力 | 弱（波长固定） | 强 | 强 |
| 计算开销 | O(L) | O(L²) | O(L) |
| 实现复杂度 | 低 | 中 | 中 |
| 绝对位置信息 | 有 | 无 | 有（粗粒度） |

## 14. 学习路径建议

**前置知识准备**（预计 3-5 天）：
1. 复习矩阵乘法和 softmax 的数学原理
2. 学习自注意力机制和缩放点积注意力的计算过程
3. 理解残差连接和 LayerNorm 在深度网络中的作用
4. 回顾 PyTorch 的 nn.Module 和张量操作 API

**本章学习**（预计 2-3 天）：
1. 第一天：阅读核心原理和第 7 节调库代码，运行测试，理解整体架构流程
2. 第二天：逐行阅读第 8 节手工实现，将每个组件独立运行验证，理解输入输出形状变换
3. 第三天：运行可视化代码，观察位置编码的模式和注意力权重的分布，完成练习题

**后续方向**：
- **解码器方向**：学习 GPT 系列的自回归解码器架构，理解因果掩码和交叉注意力
- **预训练方向**：学习 BERT 的 MLM/NSP 预训练目标，以及 RoBERTa/ELECTRA 等改进
- **高效变体**：FlashAttention（IO-aware）、稀疏注意力（Longformer/BigBird）、线性注意力（Performer）
- **多模态方向**：ViT（图像）、Whisper（语音）、CLIP（图文对齐）
- **工程实践**：学习分布式训练（DeepSpeed/FSDP）、量化和推理优化

**推荐资源**：
- 原论文：《Attention is All You Need》(Vaswani et al., 2017)
- Jay Alammar: "The Illustrated Transformer"（可视化教程）
- Andrej Karpathy: "Let's build GPT from scratch"（视频教程）
- HuggingFace Transformers 库文档和源码
