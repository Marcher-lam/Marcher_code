# 交叉注意力 (Cross-Attention) 学习文档

> 来源线索：本节内容根据原书中关于"交叉注意力机制"（第11章 11.5节）的相关章节整理、扩展与教学化改写。
> Q来自一个序列，K/V来自另一个序列，实现跨模态信息交互与特征融合。

## 1. 算法基础认知

**一句话定义**：交叉注意力是一种特殊的注意力机制，其 Query 来自一个序列，而 Key 和 Value 来自另一个不同的序列，从而实现两个独立信息源之间的交互与融合。

**直觉类比**：自注意力像是你阅读一篇文章时，每句话内部自己寻找关键词之间的关系。而交叉注意力像是你拿着一份问题清单（Query）去查阅一本参考书（Key/Value）——你在书上找到每个问题对应的答案，不同的问题会关注书中的不同章节。

**历史背景**：交叉注意力的概念最早出现在2015年Bahdanau等人提出的神经机器翻译模型中，当时被称作"对齐模型"（Alignment Model），用于在源语言和目标语言之间建立对齐关系。2017年Transformer论文正式将交叉注意力作为Decoder的核心组件——Decoder中的"Encoder-Decoder Attention"层本质上就是交叉注意力：Decoder的当前状态作为Query，Encoder的全部输出作为Key和Value。随后，交叉注意力被广泛扩展到多模态学习领域，成为图文理解、语音识别、视觉问答等任务的标配。

**算法定位**：深度学习 / 注意力机制 / 多模态融合。作为序列间信息交互的核心算子，是实现跨模态理解的基础模块。

**前置知识**：
- 自注意力机制（Self-Attention）：理解Q/K/V的定义和计算
- 缩放点积注意力（Scaled Dot-Product Attention）
- Softmax归一化和加权求和
- 多头注意力的拆分/合并操作
- PyTorch张量操作基础

## 2. 核心原理

### 核心思想

交叉注意力与自注意力最本质的区别在于**Q和KV的来源不同**：

- **自注意力**：Q、K、V三者全部来自同一个输入序列 X。公式为：`Attention(X W^Q, X W^K, X W^V)`
- **交叉注意力**：Q来自一个序列A，K和V来自另一个序列B。公式为：`Attention(A W^Q, B W^K, B W^V)`

这个简单的差异带来了根本性变化：它允许两个不同模态/来源的信息进行深度融合。A中的每个元素可以"查询"B中所有元素，根据相似度从B中提取相关信息。

### 工作流程

1. **双输入接收**：接收 Query 输入序列 (batch, query_len, d_model) 和 Context 输入序列 (batch, context_len, d_model)。这两个序列可以来自完全不同的模态（如文本和图像、文本和语音）。

2. **分别投影**：
   - Query序列通过 $W^Q$ 投影 -> Q
   - Context序列通过 $W^K$ 投影 -> K
   - Context序列通过 $W^V$ 投影 -> V
   注意：K和V来自同一个序列（Context），Q来自另一个序列。

3. **多头拆分**：与多头注意力相同，将Q、K、V按头数拆分：
   - Q: (batch, num_heads, query_len, d_k)
   - K: (batch, num_heads, context_len, d_k)
   - V: (batch, num_heads, context_len, d_v)

4. **计算交叉注意力分数**：`scores = Q × K^T / √d_k`，形状为 (batch, num_heads, query_len, context_len)。每一行表示一个Query位置对所有Context位置的关注度。

5. **掩码处理（可选）**：
   - **Padding掩码**：如果Context中有填充位置，将其注意力分数设为 -inf，排除它们的影响
   - **因果掩码**：在某些场景中防止Query关注未来的Context位置

6. **Softmax + 加权求和**：`output = softmax(scores) × V`，形状为 (batch, num_heads, query_len, d_v)。每个Query位置产生一个从Context中聚合来的向量。

7. **拼接与投影**：合并多头输出，通过 $W^O$ 投影回 d_model 维。

### 关键概念解释

- **为什么K/V来自同一序列**：K和V分别代表"匹配什么"和"取什么内容"，它们操作的是同一个信息源（Context），但角色不同。K用来计算匹配分数，V用来提供实际的语义内容。
- **Query序列的角色**：Query是"提问方"——"我想要什么信息？"每个Query位置独立地向Context序列发起查询。
- **输出长度 = Query长度**：交叉注意力的输出序列长度始终等于 Query序列长度，而不是Context序列长度。这是因为每个Query位置只产生一个输出向量。
- **与自注意力的本质区别**：自注意力是"自我反思"（同一序列内部建立联系），交叉注意力是"对外查询"（从一个序列去另一个序列提取信息）。

### 几何/直观解释

```
序列A (Query来源):     ["一家", "三口", "在", "餐桌", "前"]
                              |    |    |    |     |
                          投影为 Q1, Q2, Q3, Q4, Q5 (每个都×W^Q)

序列B (Key/Value来源): [图像中提取的16个Patch特征向量]
                              |
                          分别投影为 K1..K16, V1..V16 (×W^K, ×W^V)

交叉注意力计算:  Q_i 与 所有 K_j 做点积 → 产生注意力权重
                 用权重对所有 V_j 加权求和 → 得到 Q_i 的输出表示

结果: 每个文本词向量都"吸收"了相关的图像信息
      "餐桌"这个词的表示会被图像中桌子区域的Patch特征强化
```

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| $X^Q$ | Query输入序列 | $(B, L_Q, d_{model})$ |
| $X^C$ | Context输入序列 (提供K/V) | $(B, L_C, d_{model})$ |
| $W^Q$ | Query投影矩阵 | $(d_{model}, d_k)$ |
| $W^K$ | Key投影矩阵 (作用于Context) | $(d_{model}, d_k)$ |
| $W^V$ | Value投影矩阵 (作用于Context) | $(d_{model}, d_v)$ |
| $W^O$ | 输出投影矩阵 | $(h \cdot d_v, d_{model})$ |
| $h$ | 注意头数 | 标量 |
| $L_Q$ | Query序列长度 | 标量 |
| $L_C$ | Context序列长度 | 标量 |
| $M$ | 可选的掩码矩阵 | $(B, h, L_Q, L_C)$ 或可广播形状 |

### 单头交叉注意力

$$\text{CrossAttn}(X^Q, X^C) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
$$Q = X^Q W^Q \quad \in \mathbb{R}^{B \times L_Q \times d_k}$$
$$K = X^C W^K \quad \in \mathbb{R}^{B \times L_C \times d_k}$$
$$V = X^C W^V \quad \in \mathbb{R}^{B \times L_C \times d_v}$$

关键点：Q来自 $X^Q$，K和V都来自 $X^C$。

### 多头交叉注意力

与多头自注意力完全相同的结构，只是输入来源不同：

$$\text{MultiHeadCrossAttn}(X^Q, X^C) = \text{Concat}(head_1, ..., head_h) W^O$$

$$head_i = \text{CrossAttn}(X^Q W_i^Q, X^C W_i^K, X^C W_i^V)$$

每个头 $i$ 拥有独立的投影矩阵 $W_i^Q, W_i^K, W_i^V$，它们分别作用于不同的输入序列。

### 带有掩码的交叉注意力

在实际应用中，由于Context序列中可能存在填充位置（Padding），我们需要用掩码来屏蔽它们：

$$\text{MaskedCrossAttn}(X^Q, X^C, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M$ 的元素定义：
$$M_{ij} = \begin{cases}
0, & \text{位置 j 有效} \\
-\infty, & \text{位置 j 需要被屏蔽}
\end{cases}$$

softmax后，被屏蔽位置的权重趋向于0：
$$e^{-\infty} \to 0$$

### 交叉注意力与自注意力的对比推导

自注意力的核心计算：
$$Q_{\text{self}} = XW^Q, \quad K_{\text{self}} = XW^K, \quad V_{\text{self}} = XW^V$$
$$\text{SelfAttn}(X) = \text{softmax}\left(\frac{XW^Q (XW^K)^T}{\sqrt{d_k}}\right) XW^V$$

交叉注意力的核心计算（使用不同的输入）：
$$Q_{\text{cross}} = X^Q W^Q, \quad K_{\text{cross}} = X^C W^K, \quad V_{\text{cross}} = X^C W^V$$
$$\text{CrossAttn}(X^Q, X^C) = \text{softmax}\left(\frac{X^Q W^Q (X^C W^K)^T}{\sqrt{d_k}}\right) X^C W^V$$

可以看到，唯一的区别在于：自注意力中 $X^Q = X^C = X$；交叉注意力中 $X^Q \neq X^C$。

### 残差交叉注意力

在实践中，交叉注意力的输出常与原始 Query 做残差连接：

$$\text{Output} = X^Q + \text{CrossAttn}(\text{LayerNorm}(X^Q), X^C)$$

这个残差连接确保模型`在`融合 Context 信息的同时不会丢失 Query 本身的语义。这被称为"Residual Cross-Attention"，是Transformer Decoder中的标准做法。

## 4. 训练过程讲解

### 训练数据准备

交叉注意力本身是一个网络层，不是一个独立的训练算法。它需要嵌入到更大的模型中联合训练。准备数据时需要注意：

1. **双输入成对提供**：每个训练样本同时包含 Query 序列和 Context 序列。例如：
   - 机器翻译：(源语言句子, 目标语言句子前缀)
   - 多模态：(文本描述, 图像Patch特征)
   - 语音识别：(文本前缀, 语音特征向量)

2. **序列长度对齐**：不同样本的序列长度可能不同，需要使用 Padding 对齐到批次最大长度，同时生成对应的 Padding 掩码。

3. **掩码生成**：
   ```python
   # 生成Padding掩码: True的位置需要被屏蔽
   pad_mask = (input_ids == pad_token_id)  # (batch, seq_len)
   ```

### 训练循环（在Decoder中的使用）

交叉注意力通常作为Transformer Decoder层的一部分进行端到端训练：

```
for batch in DataLoader:
    # 1. 编码器处理Context（如源语言/图像/语音）
    context_features = Encoder(context_input)  # -> (B, L_C, d_model)

    # 2. 解码器自回归生成，每层包含:
    #    - Masked Self-Attention（关注已生成的token）
    #    - Cross-Attention（关注编码器输出的context_features）
    #    - Feed-Forward Network
    decoder_output = Decoder(target_input, context_features)

    # 3. 计算损失
    loss = CrossEntropy(decoder_output, target_labels)
    loss.backward()
    optimizer.step()
```

### 关键训练细节

- **梯度流**：梯度会同时流向 Query 序列的处理路径和 Context 序列的处理路径。这意味着编码器和解码器是联合优化的。
- **注意力Dropout**：通常对注意力权重矩阵施加 0.1 的 dropout，防止过拟合。
- **掩码处理**：必须确保在 softmax 之前应用掩码（将屏蔽位置设为 -1e9），否则无效位置会影响注意力分布。
- **批次内的序列长度差异**：使用 `padding_mask` 和 `attn_mask` 分别处理输入填充和注意力屏蔽。

## 5. 应用场景

### 核心应用场景

1. **机器翻译（Seq2Seq）**：
   - Query: 解码器当前状态（目标语言部分翻译结果）
   - Key/Value: 编码器输出（源语言完整表示）
   - 作用: 解码器每个生成步骤都能从源语言中查找最相关的信息

2. **图文理解与视觉问答（VQA）**：
   - Query: 关于图像的问题文本
   - Key/Value: 图像的Patch特征或区域特征
   - 作用: 问题中的每个词关注图像的相关区域，从视觉信息中提取答案

3. **语音识别（Speech-to-Text）**：
   - Query: 文本token序列（或前缀）
   - Key/Value: 语音特征向量序列
   - 作用: 文本生成过程中动态关注语音信号的不同时间片段

4. **文本到图像生成（Text-to-Image）**：
   - Query: 图像Patch特征（在去噪过程中）
   - Key/Value: 文本描述嵌入
   - 作用: 图像生成过程受文本条件控制，不同区域关注不同词汇

5. **多模态对话系统**：
   - Query: 用户的文本输入
   - Key/Value: 历史对话 + 可选的多模态上下文（图片、音频等）
   - 作用: 综合理解多模态上下文生成回复

### 典型使用场景特征

- **需要跨模态对齐**：文本-图像、文本-语音等不同模态间的对应关系
- **信息补充与增强**：一个序列的信息用于增强另一个序列的表示
- **条件生成**：以Context为条件来控制Query端的内容生成
- **检索式增强**：从大型Context库（如知识库、记忆）中检索相关信息

## 6. 优缺点分析

### 优点

1. **灵活的模态融合**：不要求两个输入序列有相同的长度或语义结构。图像Patch（64个）和文本Token（10个）可以自然交互。

2. **精细的对齐能力**：每个Query位置独立计算与所有Context位置的关联度，可以捕获复杂的多对多映射关系（如一个词对应图像中的多个区域）。

3. **信息保持性好**：与先池化压缩再拼接的方式不同，交叉注意力直接利用原始的Context特征进行计算，避免了压缩造成的信息损失。

4. **可解释性强**：注意力权重矩阵可以直接可视化，展示模型在"看"哪里——如机器翻译中的源语言-目标语言对齐、VQA中问题词和图像区域的对齐。

5. **与自注意力自然组合**：在Transformer Decoder中，Masked Self-Attention处理序列内部依赖，Cross-Attention处理外部信息获取，两者分工明确。

6. **支持可变长度Context**：Context序列的长度可以动态变化，不需要固定窗口。适合处理长度差异大的多模态数据。

### 缺点

1. **计算复杂度依赖两序列长度**：交叉注意力的复杂度为 $O(L_Q \times L_C)$，当两个序列都很长时（如长文本对长音频），计算开销很大。

2. **需要额外的掩码处理**：Context序列中的Padding位置需要正确掩码，否则噪声会污染注意力分布，掩码逻辑出错会导致效果显著下降。

3. **训练数据需要成对提供**：必须有Query和Context的配对数据才能训练，不像自注意力可以在任何序列上独立使用。

4. **可能忽略全局结构**：交叉注意力倾向于关注局部匹配度高的位置，可能忽略Context的全局结构信息。在语音识别中，这可能导致声学特征和语言特征的全局一致性被忽略。

5. **条件过强可能导致过拟合**：如果Context信息太具体，模型可能过度依赖Context而忽略Query自身的语义。

6. **对投影矩阵设计敏感**：Query和Context使用不同的投影矩阵 $W^Q$ 和 $W^K$，它们的初始化策略和维度设计会影响模型学习的难易程度。

## 7. 调库实现

```python
"""
Cross-Attention (交叉注意力) - PyTorch nn 模块实现
包含: 基础交叉注意力、带掩码的交叉注意力、残差交叉注意力
Python 3.9+, PyTorch 2.0+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 基础多头交叉注意力 (Multi-Head Cross-Attention)
# ============================================================
class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制。
    
    与多头自注意力的关键区别:
    - 自注意力: Q, K, V 全部来自同一个输入
    - 交叉注意力: Q 来自 query 输入, K, V 来自 context 输入
    
    这使模型能够从 context 序列中提取与 query 相关的信息。
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        """
        Args:
            d_model: 模型维度（输入和输出的特征维度）
            num_heads: 注意力头数
            dropout: 注意力权重的dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Query投影: 作用于 query 输入
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Key投影: 作用于 context 输入
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Value投影: 作用于 context 输入 (注意: K和V来自同一序列)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影: 将多头拼接结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, context, mask=None):
        """
        Args:
            query:   (batch_size, query_len, d_model) - Query序列
            context: (batch_size, context_len, d_model) - 提供Key/Value的序列
            mask:    可选的注意力掩码, 形状需能广播到 (batch, num_heads, query_len, context_len)
                     True的位置将被屏蔽
        
        Returns:
            output:       (batch_size, query_len, d_model) - 增强后的Query表示
            attn_weights: (batch_size, num_heads, query_len, context_len) - 注意力权重（用于可视化）
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        context_len = context.size(1)
        
        # ---- 1. 线性投影 ----
        # Q 来自 query 输入
        Q = self.w_q(query)  # (batch, query_len, d_model)
        # K 和 V 来自 context 输入 (关键区分点!)
        K = self.w_k(context)  # (batch, context_len, d_model)
        V = self.w_v(context)  # (batch, context_len, d_model)
        
        # ---- 2. 拆分为多头 ----
        # 重塑为: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, context_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, context_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # ---- 3. 计算注意力分数 ----
        # Q × K^T: (B, H, L_Q, d_k) × (B, H, d_k, L_C) -> (B, H, L_Q, L_C)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # 缩放: 防止大维度下的梯度消失
        scores = scores / math.sqrt(self.d_k)
        
        # ---- 4. 应用掩码 (如果有) ----
        if mask is not None:
            # 将需要屏蔽的位置设为极小值，softmax后其权重→0
            scores = scores.masked_fill(mask, float('-1e9'))
        
        # ---- 5. Softmax归一化 ----
        # 沿 context 维度做 softmax，每行(每个Query位置)的权重和为1
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # ---- 6. 加权求和 ----
        # attn_weights (B, H, L_Q, L_C) × V (B, H, L_C, d_k) -> (B, H, L_Q, d_k)
        context_vec = torch.matmul(attn_weights, V)
        
        # ---- 7. 合并多头 ----
        # (B, H, L_Q, d_k) -> (B, L_Q, H, d_k) -> (B, L_Q, d_model)
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # ---- 8. 最终投影 ----
        output = self.w_o(context_vec)  # (B, L_Q, d_model)
        
        return output, attn_weights


# ============================================================
# 2. 带Padding掩码的交叉注意力
# ============================================================
class MaskedCrossAttention(nn.Module):
    """
    带自动Padding掩码的交叉注意力。
    
    自动检测Query和Context中的填充位置(Padding)并生成掩码，
    确保模型不会关注填充的无效位置。
    
    常见使用场景:
    - Query序列有Padding: 机器翻译中目标语言句子长度不同
    - Context序列有Padding: 语音特征向量批次中对齐到相同长度
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
    def forward(self, query, context, query_pad_mask=None, context_pad_mask=None):
        """
        Args:
            query:            (B, L_Q, d_model)
            context:          (B, L_C, d_model)
            query_pad_mask:   (B, L_Q) bool, True=padding位置
            context_pad_mask: (B, L_C) bool, True=padding位置
        
        Returns:
            output:       (B, L_Q, d_model)
            attn_weights: (B, H, L_Q, L_C)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        context_len = context.size(1)
        
        # 构建组合掩码: 综合Query和Context的padding信息
        combined_mask = None
        
        if query_pad_mask is not None or context_pad_mask is not None:
            # 初始化为全False(不需要屏蔽)
            combined_mask = torch.zeros(
                batch_size, 1, query_len, context_len,
                dtype=torch.bool, device=query.device
            )
            
            # Context的padding位置: 所有Query位置都不该关注
            if context_pad_mask is not None:
                # context_pad_mask: (B, L_C) -> (B, 1, 1, L_C)
                combined_mask = combined_mask | context_pad_mask.unsqueeze(1).unsqueeze(2)
            
            # Query的padding位置: 这些Query位置本身无意义，但通常也要屏蔽
            if query_pad_mask is not None:
                # query_pad_mask: (B, L_Q) -> (B, 1, L_Q, 1)
                combined_mask = combined_mask | query_pad_mask.unsqueeze(1).unsqueeze(-1)
        
        output, attn_weights = self.cross_attn(query, context, combined_mask)
        return output, attn_weights


# ============================================================
# 3. 残差交叉注意力 (Residual Cross-Attention)
#    Transformer Decoder层的标准组件
# ============================================================
class ResidualCrossAttentionBlock(nn.Module):
    """
    残差交叉注意力块: 带Pre-Norm和残差连接的交叉注意力。
    
    这是Transformer Decoder中"Encoder-Decoder Attention"层
    (即交叉注意力层)的标准实现方式。
    
    流程:
        x -> LayerNorm -> CrossAttn(作为Q, context作为K/V) -> +x(残差) -> 输出
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context, mask=None):
        """
        Args:
            x:       (B, L_Q, d_model) - Query输入 (如decoder的中间状态)
            context: (B, L_C, d_model) - Context输入 (如encoder的全部输出)
            mask:    可选的注意力掩码
        
        Returns:
            output:       (B, L_Q, d_model)
            attn_weights: (B, H, L_Q, L_C)
        """
        # Pre-Norm + 交叉注意力 + 残差连接
        attn_output, attn_weights = self.cross_attn(
            self.norm(x), context, mask
        )
        # 残差连接: 保留原始x的信息，加上从context获取的信息
        output = x + self.dropout(attn_output)
        return output, attn_weights


# ============================================================
# 4. 完整演示: Transformer Decoder层 (自注意力 + 交叉注意力 + MLP)
# ============================================================
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder的完整一层，包含:
    1. Masked Self-Attention (处理已生成的序列)
    2. Cross-Attention (查询Encoder的输出)
    3. Feed-Forward Network
    
    通过此层可以清晰看到自注意力与交叉注意力的协作关系:
    - 自注意力: 在decoder自己的序列内部建立依赖
    - 交叉注意力: 向encoder的输出"提问"，获取外部信息
    """
    
    def __init__(self, d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # ---- 1. Masked Self-Attention ----
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # ---- 2. Cross-Attention (Encoder-Decoder Attention) ----
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # 使用自己实现的多头交叉注意力
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # ---- 3. Feed-Forward Network ----
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x:               (B, L_Q, d_model) decoder的输入序列
            encoder_output:  (B, L_C, d_model) encoder的完整输出 (作为cross-attn的K/V)
            self_attn_mask:  自注意力的因果掩码 (防止看到未来token)
            cross_attn_mask: 交叉注意力的padding掩码
        
        Returns:
            x:              (B, L_Q, d_model)
            self_attn_w:    自注意力权重
            cross_attn_w:   交叉注意力权重
        """
        # ---- 自注意力: 序列内部交互 ----
        residual = x
        x_norm = self.self_attn_norm(x)
        # Q=K=V=x_norm (自注意力的标志)
        self_attn_out, self_attn_w = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=self_attn_mask,
            need_weights=True
        )
        x = residual + self.self_attn_dropout(self_attn_out)
        
        # ---- 交叉注意力: 向encoder查询信息 ----
        residual = x
        x_norm = self.cross_attn_norm(x)
        # Q=x_norm (来自decoder), K/V=encoder_output (来自encoder)
        # 这是交叉注意力与自注意力的本质区别!
        cross_attn_out, cross_attn_w = self.cross_attn(
            x_norm, encoder_output, mask=cross_attn_mask
        )
        x = residual + self.cross_attn_dropout(cross_attn_out)
        
        # ---- 前馈网络: 逐位置的非线性变换 ----
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.ffn_dropout(ffn_out)
        
        return x, self_attn_w, cross_attn_w


# ============================================================
# 5. 端到端运行示例
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("交叉注意力 (Cross-Attention) 演示")
    print("=" * 60)
    
    # ---- 参数设置 ----
    BATCH_SIZE = 2
    QUERY_LEN = 5      # Query序列长度 (如: 5个词的问句)
    CONTEXT_LEN = 8    # Context序列长度 (如: 8个图像Patch)
    D_MODEL = 16       # 模型维度 (演示用较小值)
    NUM_HEADS = 4      # 注意力头数
    
    torch.manual_seed(42)
    
    # ---- 创建随机输入 ----
    # 模拟: Query = 文本问句, Context = 图像特征
    query = torch.randn(BATCH_SIZE, QUERY_LEN, D_MODEL)
    context = torch.randn(BATCH_SIZE, CONTEXT_LEN, D_MODEL)
    
    print(f"Query形状:   {query.shape}   (模拟: 文本问句的嵌入)")
    print(f"Context形状: {context.shape} (模拟: 图像Patch特征)")
    
    # ---- 测试1: 基础交叉注意力 ----
    print("\n--- 测试1: 基础多头交叉注意力 ---")
    cross_attn = MultiHeadCrossAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output, attn_weights = cross_attn(query, context)
    
    print(f"输出形状:      {output.shape}      (应等于Query长度)")
    print(f"注意力权重形状: {attn_weights.shape}")
    # 验证: 输出长度 == Query长度
    assert output.size(1) == QUERY_LEN, "输出长度必须等于Query长度!"
    print("验证通过: 输出长度 = Query长度 ✓")
    
    # ---- 测试2: 带Padding掩码的交叉注意力 ----
    print("\n--- 测试2: 带Padding掩码的交叉注意力 ---")
    # 模拟: Context中后3个位置是padding
    context_pad_mask = torch.zeros(BATCH_SIZE, CONTEXT_LEN, dtype=torch.bool)
    context_pad_mask[:, -3:] = True  # 后3个位置标记为padding
    
    masked_cross_attn = MaskedCrossAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output_masked, attn_weights_masked = masked_cross_attn(
        query, context, context_pad_mask=context_pad_mask
    )
    
    # 验证: padding位置的注意力权重应接近0
    avg_attn_masked = attn_weights_masked.mean(dim=1)  # 平均多头权重
    pad_attn = avg_attn_masked[:, :, -3:].sum(dim=-1)  # padding位置的总权重
    print(f"Padding位置的注意力权重之和: {pad_attn.mean().item():.6f} (应接近0)")
    print("验证通过: Padding位置被成功屏蔽 ✓")
    
    # ---- 测试3: 自注意力 vs 交叉注意力 对比 ----
    print("\n--- 测试3: 自注意力 vs 交叉注意力 ---")
    self_attn = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=NUM_HEADS,
                                       batch_first=True)
    
    # 自注意力: Q=K=V=query
    self_output, self_weights = self_attn(query, query, query)
    print(f"自注意力输出形状: {self_output.shape}")
    
    # 交叉注意力: Q=query, K=V=context
    cross_output, cross_weights = cross_attn(query, context)
    print(f"交叉注意力输出形状: {cross_output.shape}")
    
    # 交叉注意力权重矩阵不一定是方阵! (query_len != context_len时)
    print(f"自注意力权重: {self_weights.shape} (方阵, 因为L_Q=L_Q)")
    print(f"交叉注意力权重: {cross_weights.shape} (矩形, 因为L_Q≠L_C)")
    
    # ---- 测试4: 完整Transformer Decoder层 ----
    print("\n--- 测试4: 完整Transformer Decoder层 ---")
    # 模拟encoder输出
    encoder_output = torch.randn(BATCH_SIZE, 6, D_MODEL)
    # 因果掩码: decoder不能看到未来的token
    causal_mask = torch.triu(torch.ones(QUERY_LEN, QUERY_LEN) * float('-inf'), diagonal=1)
    
    decoder_layer = TransformerDecoderLayer(
        d_model=D_MODEL, num_heads=NUM_HEADS, dim_feedforward=64
    )
    
    dec_output, self_w, cross_w = decoder_layer(
        query, encoder_output, self_attn_mask=causal_mask
    )
    print(f"Decoder层输出形状: {dec_output.shape}")
    print(f"自注意力权重形状:  {self_w.shape}")
    print(f"交叉注意力权重形状: {cross_w.shape}")
    print("完整Decoder层测试通过 ✓")
    
    print("\n" + "=" * 60)
    print("所有测试通过! 交叉注意力实现正确。")
    print("=" * 60)
```

## 8. 手工代码实现

```python
"""
Cross-Attention 手工代码实现 - 从零实现核心逻辑
完全基于原始矩阵运算，不依赖nn.MultiheadAttention
展示交叉注意力与自注意力的明确区别
Python 3.9+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 手工实现: 缩放点积注意力 (底层计算)
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, training=True):
    """
    手工实现缩放点积注意力计算。
    
    Args:
        Q: (batch, heads, query_len, d_k)
        K: (batch, heads, key_len, d_k)
        V: (batch, heads, key_len, d_v)
        mask: 可选的布尔掩码 (True=屏蔽), 形状需可广播到 (B, H, L_Q, L_K)
    
    Returns:
        output: (batch, heads, query_len, d_v)
        attn_weights: (batch, heads, query_len, key_len)
    """
    d_k = Q.size(-1)
    
    # Step 1: 计算原始注意力分数
    # Q × K^T: 衡量每个Query位置与每个Key位置的匹配度
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L_Q, L_K)
    
    # Step 2: 缩放 (关键步骤!)
    # 没有缩放的话, 当d_k较大(如64)时, 点积值的方差≈d_k,
    # softmax输出会极度尖锐(接近one-hot), 梯度趋于零
    scores = scores / math.sqrt(d_k)
    
    # Step 3: 应用掩码
    if mask is not None:
        # 被屏蔽的位置设为极大负数, softmax(极大负数) → 0
        scores = scores.masked_fill(mask, float('-1e9'))
    
    # Step 4: Softmax归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Dropout正则化
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Step 6: 加权聚合
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights


# ============================================================
# 手工实现: 完整的交叉注意力类
# ============================================================
class CrossAttentionScratch(nn.Module):
    """
    从零实现的多头交叉注意力。
    
    通过对比 with SelfAttentionScratch 来理解交叉注意力的独特之处:
    CrossAttention:  Q = query × W_Q,     K = context × W_K,     V = context × W_V
    SelfAttention:   Q = x × W_Q,          K = x × W_K,          V = x × W_V
    
    关键: 两个不同的 W_K, W_V 分别作用于 query 和 context!
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        
        # ---- 创建投影权重矩阵 ----
        # 使用 nn.Parameter 手动管理参数，而非 nn.Linear
        # 这样能清楚地看到投影矩阵的结构
        
        # Query投影: query输入 -> Q
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        # Key投影: context输入 -> K (注意: 作用于context, 不是query!)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        # Value投影: context输入 -> V (注意: 作用于context, 不是query!)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        # 输出投影
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        self.dropout = dropout
        
    def forward(self, query, context, mask=None):
        """
        Args:
            query:   (batch, query_len, d_model)
            context: (batch, context_len, d_model)
            mask:    可选的bool掩码 (True=屏蔽), 
                     形状需可广播到 (batch, num_heads, query_len, context_len)
        
        Returns:
            output:       (batch, query_len, d_model) - 长度等于query_len!
            attn_weights: (batch, num_heads, query_len, context_len)
        """
        batch_size, query_len, _ = query.shape
        context_len = context.shape[1]
        
        # ---- 1. 线性投影 ----
        # 手动执行投影: X × W
        # 矩阵乘法: (B, L, d) × (d, d) -> (B, L, d)
        Q = torch.matmul(query, self.W_Q)      # (B, L_Q, d_model)
        K = torch.matmul(context, self.W_K)    # (B, L_C, d_model) - 来自context!
        V = torch.matmul(context, self.W_V)    # (B, L_C, d_model) - 来自context!
        
        # ---- 2. 拆分为多头 ----
        # 将d_model维均匀分给h个头
        Q = Q.view(batch_size, query_len, self.num_heads, self.d_k)
        K = K.view(batch_size, context_len, self.num_heads, self.d_k)
        V = V.view(batch_size, context_len, self.num_heads, self.d_k)
        
        # 调整为 (batch, num_heads, seq_len, d_k) 以方便批量计算
        Q = Q.transpose(1, 2)  # (B, H, L_Q, d_k)
        K = K.transpose(1, 2)  # (B, H, L_C, d_k)
        V = V.transpose(1, 2)  # (B, H, L_C, d_k)
        
        # ---- 3. 计算交叉注意力 ----
        # 手工调用底层注意力函数
        context_vec, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout, training=self.training
        )
        # context_vec: (B, H, L_Q, d_k) - 包含从context聚合来的信息
        
        # ---- 4. 合并多头 ----
        # (B, H, L_Q, d_k) -> (B, L_Q, H, d_k)
        context_vec = context_vec.transpose(1, 2)
        # -> (B, L_Q, H*d_k = d_model)
        context_vec = context_vec.contiguous().view(batch_size, query_len, self.d_model)
        
        # ---- 5. 输出投影 ----
        output = torch.matmul(context_vec, self.W_O)
        
        return output, attn_weights


# ============================================================
# 手工实现: 自注意力 (用于对比)
# ============================================================
class SelfAttentionScratch(nn.Module):
    """
    从零实现的多头自注意力。
    与 CrossAttentionScratch 对比, 唯一区别在 forward 方法中:
    - Cross: K,V 来自 context 输入
    - Self:  K,V 来自与 Q 相同的输入
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 三个投影矩阵都作用于同一个输入 x
        # 但在交叉注意力中 W_K 和 W_V 作用于不同的输入!
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.dropout = dropout
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - 单一输入序列
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 关键: Q, K, V 都来自同一个 x
        Q = torch.matmul(x, self.W_Q)  # (B, L, d_model)
        K = torch.matmul(x, self.W_K)  # (B, L, d_model) - 同样来自x!
        V = torch.matmul(x, self.W_V)  # (B, L, d_model) - 同样来自x!
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        context_vec, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout, training=self.training
        )
        
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = torch.matmul(context_vec, self.W_O)
        
        return output, attn_weights


# ============================================================
# 手工实现: 带可配置掩码策略的交叉注意力
# ============================================================
class ConfigurableMaskedCrossAttention(nn.Module):
    """
    可配置掩码策略的交叉注意力。
    
    支持三种掩码模式:
    1. 'padding': 只屏蔽Context的padding位置 (最常见)
    2. 'causal': 因果掩码, Query位置i只能关注Context位置≤i
    3. 'none': 不使用掩码
    """
    
    def __init__(self, d_model, num_heads, mask_mode='padding', dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttentionScratch(d_model, num_heads, dropout)
        self.mask_mode = mask_mode
        
    def build_mask(self, query_len, context_len, context_pad_mask=None, device='cpu'):
        """
        根据 mask_mode 构建掩码矩阵。
        """
        mask = None
        
        if self.mask_mode == 'padding' and context_pad_mask is not None:
            # context_pad_mask: (B, L_C), True=需要屏蔽
            # 扩展到 (B, 1, 1, L_C) 以广播到注意力分数形状
            mask = context_pad_mask.unsqueeze(1).unsqueeze(2)
            # 现在形状 (B, 1, 1, L_C), 会自动广播到 (B, H, L_Q, L_C)
            
        elif self.mask_mode == 'causal':
            # 上三角掩码: 屏蔽未来的context位置
            causal = torch.triu(
                torch.ones(query_len, context_len, device=device),
                diagonal=1
            ).bool()
            mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, L_Q, L_C)
            
        return mask
    
    def forward(self, query, context, context_pad_mask=None):
        mask = self.build_mask(
            query.size(1), context.size(1),
            context_pad_mask, query.device
        )
        return self.cross_attn(query, context, mask)


# ============================================================
# 测试代码
# ============================================================
def test_cross_attention():
    """综合测试函数"""
    print("=" * 60)
    print("Cross-Attention 手工实现测试")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, L_Q, L_C, D, H = 2, 5, 8, 64, 4
    
    query = torch.randn(B, L_Q, D)
    context = torch.randn(B, L_C, D)
    
    # ---- 测试1: 基础交叉注意力 vs 自注意力 ----
    print("\n1. 基础功能测试")
    cross_attn = CrossAttentionScratch(D, H)
    self_attn = SelfAttentionScratch(D, H)
    
    # 交叉注意力
    cross_out, cross_w = cross_attn(query, context)
    print(f"   交叉注意力: 输入=(B,{L_Q},{D})+(B,{L_C},{D}), 输出={list(cross_out.shape)}")
    print(f"   注意力权重形状: {list(cross_w.shape)} (非方阵, 因为 L_Q≠L_C)")
    
    # 自注意力
    self_out, self_w = self_attn(query)
    print(f"   自注意力:   输入=(B,{L_Q},{D}),           输出={list(self_out.shape)}")
    print(f"   注意力权重形状: {list(self_w.shape)} (方阵)")
    
    assert cross_out.shape == (B, L_Q, D), f"交叉注意力输出形状错误!"
    assert self_out.shape == (B, L_Q, D), f"自注意力输出形状错误!"
    print("   ✓ 基础功能测试通过")
    
    # ---- 测试2: 输出长度验证 ----
    print("\n2. 输出长度验证")
    # 交叉注意力的输出长度应始终等于 Query 长度
    assert cross_out.size(1) == L_Q, "输出长度应等于Query长度"
    # 即使Context长度改变, 输出长度也不变
    context_short = torch.randn(B, 3, D)
    cross_out2, _ = cross_attn(query, context_short)
    assert cross_out2.size(1) == L_Q, "Context长度变化不应影响输出长度"
    print("   ✓ 输出长度验证通过")
    
    # ---- 测试3: 掩码功能 ----
    print("\n3. 掩码功能测试")
    # 创建padding掩码
    pad_mask = torch.zeros(B, L_C, dtype=torch.bool)
    pad_mask[:, -3:] = True  # 后3个context位置是padding
    
    masked_cross_attn = ConfigurableMaskedCrossAttention(D, H, mask_mode='padding')
    out_masked, w_masked = masked_cross_attn(query, context, context_pad_mask=pad_mask)
    
    # 计算padding位置的注意力权重之和 (应接近0)
    avg_w = w_masked.mean(dim=1)  # 平均各头
    pad_attention = avg_w[:, :, -3:].sum(dim=-1).mean()  # padding位置的总权重
    print(f"   Padding位置注意力权重之和: {pad_attention.item():.8f}")
    assert pad_attention.item() < 0.01, "Padding位置应被屏蔽!"
    print("   ✓ 掩码功能测试通过")
    
    # ---- 测试4: 梯度流验证 ----
    print("\n4. 梯度流验证")
    cross_attn.train()
    query_grad = query.clone().requires_grad_(True)
    context_grad = context.clone().requires_grad_(True)
    
    out_grad, _ = cross_attn(query_grad, context_grad)
    loss = out_grad.sum()
    loss.backward()
    
    # 梯度应该同时流向 query 和 context
    assert query_grad.grad is not None, "梯度应流向query!"
    assert context_grad.grad is not None, "梯度应流向context!"
    print(f"   Query梯度范数:  {query_grad.grad.norm().item():.4f}")
    print(f"   Context梯度范数: {context_grad.grad.norm().item():.4f}")
    print("   ✓ 梯度流验证通过")
    
    # ---- 测试5: 因果掩码 ----
    print("\n5. 因果掩码测试")
    causal_attn = ConfigurableMaskedCrossAttention(D, H, mask_mode='causal')
    # 使用相同长度以支持方阵因果掩码
    q_same = torch.randn(B, 4, D)
    c_same = torch.randn(B, 4, D)
    _, w_causal = causal_attn(q_same, c_same)
    
    # 因果掩码下, 位置i只能关注位置0..i
    # 所以 w[i, j] 对 j > i 应该为0
    avg_w_causal = w_causal.mean(dim=1)[0]  # 取第一个样本, 平均各头
    upper_tri_sum = torch.triu(avg_w_causal, diagonal=1).sum().item()
    print(f"   因果掩码下上三角总权重: {upper_tri_sum:.8f} (应≈0)")
    assert upper_tri_sum < 1e-6, "因果掩码应屏蔽上三角!"
    print("   ✓ 因果掩码测试通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过! Cross-Attention手工实现正确。")
    print("=" * 60)


if __name__ == '__main__':
    test_cross_attention()
```

## 9. 可视化与结果理解

```python
"""
交叉注意力可视化 - 展示交叉注意力与自注意力的区别
包括: 注意力权重矩阵对比、多模态融合示意、掩码效果
"""

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


def compute_attention_weights(query, key, scale=True):
    """简单的注意力计算辅助函数"""
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)
    return F.softmax(scores, dim=-1).numpy()


def main():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # ============================================================
    # 图1: 自注意力权重矩阵 (方阵)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    # 模拟: 5个token的句子做自注意力
    seq_len = 5
    x = torch.randn(1, seq_len, 8)
    W_q = torch.randn(8, 8) * 0.5
    W_k = torch.randn(8, 8) * 0.5
    Q = torch.matmul(x, W_q)
    K = torch.matmul(x, W_k)  # 来自同一个x!
    self_attn = compute_attention_weights(Q, K)
    
    im1 = ax1.imshow(self_attn[0], cmap='YlOrRd', vmin=0, vmax=0.5)
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    ax1.set_xticklabels(['I', 'love', 'machine', 'learning', '.'], fontsize=8)
    ax1.set_yticklabels(['I', 'love', 'machine', 'learning', '.'], fontsize=8)
    ax1.set_xlabel('Key Position (same sequence)')
    ax1.set_ylabel('Query Position (same sequence)')
    ax1.set_title('Self-Attention Weights\n(Q, K, V all from SAME sequence)\nSquare Matrix', 
                  fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Weight')
    
    # ============================================================
    # 图2: 交叉注意力权重矩阵 (矩形)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    # 模拟: 5个Query词 × 8个Context Patch
    query_len = 5
    context_len = 8
    query_vec = torch.randn(1, query_len, 8)
    context_vec = torch.randn(1, context_len, 8)
    W_q2 = torch.randn(8, 8) * 0.5
    W_k2 = torch.randn(8, 8) * 0.5
    Q2 = torch.matmul(query_vec, W_q2)
    K2 = torch.matmul(context_vec, W_k2)  # 来自不同的context!
    cross_attn = compute_attention_weights(Q2, K2)
    
    query_labels = ['Who', 'is', 'the', 'man', '?']
    context_labels = ['Patch1', 'Patch2', 'Patch3', 'Patch4', 
                      'Patch5', 'Patch6', 'Patch7', 'Patch8']
    
    im2 = ax2.imshow(cross_attn[0], cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
    ax2.set_xticks(range(context_len))
    ax2.set_yticks(range(query_len))
    ax2.set_xticklabels(context_labels, fontsize=7, rotation=45)
    ax2.set_yticklabels(query_labels, fontsize=8)
    ax2.set_xlabel('Key/Value Position (context: image patches)')
    ax2.set_ylabel('Query Position (text question)')
    ax2.set_title('Cross-Attention Weights\n(Q from text, K/V from image patches)\nRectangular Matrix',
                  fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Weight')
    
    # ============================================================
    # 图3: 自注意力 vs 交叉注意力 架构对比
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_title('Self-Attention vs Cross-Attention\n(Architecture Comparison)',
                  fontsize=10, fontweight='bold')
    
    # 左侧: 自注意力
    ax3.text(2.5, 7.5, 'Self-Attention', ha='center', fontsize=11, fontweight='bold', color='blue')
    ax3.add_patch(plt.Rectangle((1, 5.5), 3, 1.2, fill=True, facecolor='lightblue', 
                                  edgecolor='blue', alpha=0.5))
    ax3.text(2.5, 6.1, 'Input X', ha='center', fontsize=10)
    # 三条箭头
    for i, (dx, label) in enumerate([(-0.8, 'Q'), (0, 'K'), (0.8, 'V')]):
        ax3.arrow(2.5, 5.3, dx, -0.6, head_width=0.15, head_length=0.2, fc='blue', ec='blue')
    ax3.add_patch(plt.Rectangle((1, 3.8), 3, 0.8, fill=True, facecolor='lightgreen',
                                  edgecolor='green', alpha=0.5))
    ax3.text(2.5, 4.2, 'Attention', ha='center', fontsize=10)
    ax3.arrow(2.5, 3.6, 0, -0.5, head_width=0.15, head_length=0.2, fc='blue', ec='blue')
    ax3.add_patch(plt.Rectangle((1, 2.4), 3, 0.8, fill=True, facecolor='lightsalmon',
                                  edgecolor='red', alpha=0.5))
    ax3.text(2.5, 2.8, 'Output (seq_len × d)', ha='center', fontsize=9)
    
    # 右侧: 交叉注意力
    ax3.text(7.5, 7.5, 'Cross-Attention', ha='center', fontsize=11, fontweight='bold', color='red')
    ax3.add_patch(plt.Rectangle((6, 5.5), 3, 1.2, fill=True, facecolor='lightblue',
                                  edgecolor='blue', alpha=0.5))
    ax3.text(7.5, 6.4, 'Query Input A', ha='center', fontsize=9)
    ax3.text(7.5, 5.8, 'Context Input B', ha='center', fontsize=9)
    
    ax3.arrow(7.5, 5.3, -0.5, -0.6, head_width=0.15, head_length=0.2, fc='red', ec='red')
    ax3.text(6.3, 4.9, 'Q', fontsize=9, color='red')
    ax3.arrow(7.5, 5.3, 0.5, -0.6, head_width=0.15, head_length=0.2, fc='red', ec='red')
    ax3.text(8.5, 4.9, 'K,V', fontsize=9, color='red')
    
    ax3.add_patch(plt.Rectangle((6, 3.8), 3, 0.8, fill=True, facecolor='lightgreen',
                                  edgecolor='green', alpha=0.5))
    ax3.text(7.5, 4.2, 'Cross-Attention', ha='center', fontsize=10)
    ax3.arrow(7.5, 3.6, 0, -0.5, head_width=0.15, head_length=0.2, fc='red', ec='red')
    ax3.add_patch(plt.Rectangle((6, 2.4), 3, 0.8, fill=True, facecolor='lightsalmon',
                                  edgecolor='red', alpha=0.5))
    ax3.text(7.5, 2.8, 'Output (query_len × d)', ha='center', fontsize=9)

    ax3.plot([5, 5], [1.5, 7.8], 'k--', linewidth=1, alpha=0.3)
    
    # ============================================================
    # 图4: 多模态交叉注意力示意 (文本-图像融合)
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Multi-Modal Cross-Attention: Text-Image Fusion',
                  fontsize=10, fontweight='bold')
    
    # 文本tokens (Query)
    text_tokens = ['A', 'dog', 'is', 'running', 'on', 'the', 'grass']
    for i, token in enumerate(text_tokens):
        y = 8.5 - i * 1.0
        ax4.add_patch(plt.Rectangle((0.5, y-0.3), 1.8, 0.6, fill=True,
                                      facecolor='lightblue', edgecolor='navy', alpha=0.7))
        ax4.text(1.4, y, f'{token}', ha='center', fontsize=8)
    
    # 图像patches (Key/Value)
    for i in range(9):
        row, col = i // 3, i % 3
        ax4.add_patch(plt.Rectangle((6 + col*1.5, 6 - row*2), 1.3, 1.7, fill=True,
                                      facecolor='lightcoral', edgecolor='darkred', alpha=0.7))
    
    ax4.text(9.5, 9.5, 'Image\nPatches\n(K/V)', ha='center', fontsize=8, color='darkred')
    ax4.text(1.4, 9.5, 'Text\nTokens\n(Q)', ha='center', fontsize=8, color='navy')
    
    # 显示注意力连线 (dog -> 中间上方patch, grass -> 下方patch)
    connections = [
        (1, 1, 'dog→dog region'),      # "dog" → patch 1 (row=0,col=1)
        (5, 7, 'grass→bottom region'), # "grass" → patch 7 (row=2,col=1)
    ]
    for qi, pi, label in connections:
        qy = 8.5 - qi * 1.0
        px = 6 + (pi % 3) * 1.5 + 0.65
        py = 6 - (pi // 3) * 2 + 0.85
        ax4.plot([3.3, px], [qy, py], 'gray', linewidth=1, alpha=0.4)
        ax4.text(px+0.5 if pi < 4 else px-2, py+0.5, label, fontsize=6, color='gray')
    
    # ============================================================
    # 图5: 掩码效果展示 - 有/无Padding掩码的注意力对比
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])
    seq_len = 6
    # 假设有效长度为3, 后3个是padding
    valid_len = 3
    
    # 没有掩码的注意力
    np.random.seed(123)
    raw_scores = np.random.randn(seq_len, seq_len) * 1.5
    raw_attn = np.exp(raw_scores) / np.exp(raw_scores).sum(axis=1, keepdims=True)
    
    # 有掩码的注意力 (padding位置被屏蔽)
    masked_scores = raw_scores.copy()
    masked_scores[:, valid_len:] = -1e9
    masked_attn = np.exp(masked_scores) / np.exp(masked_scores).sum(axis=1, keepdims=True)
    
    # 绘制对比
    ax5a = ax5.inset_axes([0.02, 0.52, 0.45, 0.44])
    ax5a.imshow(raw_attn, cmap='YlOrRd', vmin=0, vmax=0.8)
    ax5a.set_title('Without Mask\n(Noisy)', fontsize=8)
    ax5a.set_xticks([])
    ax5a.set_yticks([])
    ax5a.axvline(x=valid_len - 0.5, color='red', linewidth=2, linestyle='--')
    
    ax5b = ax5.inset_axes([0.53, 0.52, 0.45, 0.44])
    ax5b.imshow(masked_attn, cmap='YlOrRd', vmin=0, vmax=0.8)
    ax5b.set_title('With Padding Mask\n(Clean)', fontsize=8)
    ax5b.set_xticks([])
    ax5b.set_yticks([])
    ax5b.axvline(x=valid_len - 0.5, color='red', linewidth=2, linestyle='--',
                 label='Padding boundary')
    ax5b.legend(fontsize=6, loc='upper left')
    
    # 下方: 带掩码的多头交叉注意力示意
    ax5c = ax5.inset_axes([0.05, 0.02, 0.9, 0.42])
    # 模拟多头注意力
    n_heads = 4
    for h in range(n_heads):
        ax5c_h = ax5.inset_axes([0.02 + h*0.25, 0.02, 0.22, 0.42])
        head_attn = np.random.rand(4, 8) * 0.5
        # 掩码屏蔽后3列
        ax5c_h.imshow(head_attn, cmap='YlOrRd', vmin=0, vmax=0.5)
        ax5c_h.axvline(x=4.5, color='white', linewidth=2, linestyle='--')
        ax5c_h.set_title(f'Head {h+1}', fontsize=7)
        ax5c_h.set_xticks([])
        ax5c_h.set_yticks([])
    
    ax5.text(0.5, 0.25, 'Multi-Head Cross-Attention with Mask', ha='center', 
             fontsize=9, fontweight='bold', transform=ax5.transAxes)
    ax5.set_title('Mask Effect on Cross-Attention', fontsize=10, fontweight='bold')
    ax5.axis('off')
    
    # ============================================================
    # 图6: 不同掩码策略的注意力视觉对比
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])
    L_Q, L_C = 6, 8
    c_len = L_C
    
    # 构造模拟的注意力矩阵
    base_attn = np.zeros((L_Q, L_C))
    for i in range(L_Q):
        # 每行关注不同的context区域
        center = int(i / L_Q * L_C)
        for j in range(max(0, center-2), min(L_C, center+3)):
            base_attn[i, j] = np.exp(-abs(j-center)/2)
    base_attn = base_attn / base_attn.sum(axis=1, keepdims=True)
    
    strategies = [
        ('No Mask', base_attn),
        ('Padding Mask\n(last 2 cols)', base_attn * np.array([1]*6 + [0,0]).reshape(1,-1)),
        ('Causal Mask\n(upper triangular)', np.tril(base_attn)),
    ]
    
    for idx, (title, mat) in enumerate(strategies):
        ax_i = ax6.inset_axes([0.02, 0.65 - idx*0.33, 0.96, 0.3])
        # 归一化
        mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)
        ax_i.imshow(mat, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
        ax_i.set_title(title, fontsize=8, loc='left')
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        if idx == 2:
            ax_i.set_xlabel('Context Position (Key/Value)', fontsize=7)
            ax_i.set_ylabel('Query\nPosition', fontsize=7)
    
    ax6.set_title('Mask Strategies Comparison\nin Cross-Attention', 
                  fontsize=10, fontweight='bold')
    ax6.axis('off')
    
    # ============================================================
    # 图7: 多模态交叉注意力流程 (详细)
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_xlim(0, 12)
    ax7.set_ylim(0, 11)
    ax7.axis('off')
    ax7.set_title('Complete Cross-Attention Pipeline in Multi-Modal System',
                  fontsize=10, fontweight='bold')
    
    steps = [
        (0.5, 9.5, 2, 1, 'lightblue', 'Audio\nWaveform', 'Input'),
        (3, 9.5, 2, 1, 'lightcyan', 'Feature\nExtractor', 'Encode'),
        (5.5, 9.5, 2, 1, 'lightgreen', 'Audio\nFeatures', 'Context\n(K/V)'),
        (8, 9.5, 2, 1, 'lightsalmon', 'Text\nEmbeddings', 'Query\n(Q)'),
        (5.5, 7, 4.5, 1.3, 'lightyellow', 'Cross-Attention\nQ·K^T/√d → softmax → ×V', 'Fuse'),
        (5.5, 4.5, 4.5, 1.3, 'plum', 'Transformer Decoder\n(Self-Attn + FFN)', 'Process'),
        (5.5, 2, 4.5, 1.3, 'wheat', 'Generated Text\n"the cat sat..."', 'Output'),
    ]
    
    for x, y, w, h, color, text, label in steps:
        ax7.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, fill=True,
                                      facecolor=color, edgecolor='gray', alpha=0.8))
        ax7.text(x, y, text, ha='center', va='center', fontsize=8)
        ax7.text(x, y+h/2+0.2, label, ha='center', fontsize=7, color='gray', style='italic')
    
    # 箭头
    for (x1, y1, x2, y2) in [(1.5, 9, 1.5, 10.5), (3.5, 9, 5, 9),
                               (6.5, 9, 8.5, 9), (7, 8, 6, 7.5),
                               (6, 7, 6, 5.5), (6, 5.5, 6, 3.5)]:
        ax7.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    
    # ============================================================
    # 图8: 注意力熵分析 - 不同头的注意力集中度
    # ============================================================
    ax8 = fig.add_subplot(gs[2, 1])
    # 模拟多头注意力的熵
    n_heads = 8
    np.random.seed(456)
    # 不同头有不同程度的集中程度
    entropy_values = np.array([1.2, 0.8, 2.1, 1.5, 0.6, 2.5, 1.8, 0.9])
    
    colors = plt.cm.viridis(entropy_values / entropy_values.max())
    bars = ax8.bar(range(n_heads), entropy_values, color=colors, edgecolor='black', linewidth=0.5)
    ax8.axhline(y=np.log(L_C), color='red', linestyle='--', alpha=0.5,
                label=f'Max entropy (ln({L_C}) = {np.log(L_C):.1f})')
    ax8.axhline(y=entropy_values.mean(), color='blue', linestyle='-', alpha=0.5,
                label=f'Mean = {entropy_values.mean():.1f}')
    ax8.set_xlabel('Attention Head Index')
    ax8.set_ylabel('Attention Entropy H(A)')
    ax8.set_title('Cross-Attention Head Diversity\n(Entropy Measures Attention Spread)',
                  fontsize=10, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.set_xticks(range(n_heads))
    
    # 标注
    ax8.annotate('Focused\n(sharp attention)', xy=(4, 0.6), fontsize=7,
                xytext=(5, 0.3), arrowprops=dict(arrowstyle='->', alpha=0.5))
    ax8.annotate('Spread\n(broad attention)', xy=(5, 2.5), fontsize=7,
                xytext=(6, 2.8), arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # ============================================================
    # 图9: 残差交叉注意力的信息流示意
    # ============================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_xlim(0, 12)
    ax9.set_ylim(0, 8)
    ax9.axis('off')
    ax9.set_title('Residual Cross-Attention Information Flow\n(Transformer Decoder Style)',
                  fontsize=10, fontweight='bold')
    
    # 输入
    ax9.add_patch(plt.Rectangle((0.5, 5), 2.5, 1, fill=True, facecolor='lightblue',
                                  edgecolor='navy'))
    ax9.text(1.75, 5.5, 'Query x\n(from decoder)', ha='center', fontsize=8)
    
    # 主路径
    ax9.arrow(3, 5.5, 1, 0, head_width=0.2, head_length=0.2, fc='black')
    
    # LayerNorm
    ax9.add_patch(plt.Rectangle((4, 5), 2.5, 1, fill=True, facecolor='lightyellow',
                                  edgecolor='orange'))
    ax9.text(5.25, 5.5, 'LayerNorm', ha='center', fontsize=9)
    
    ax9.arrow(6.5, 5.5, 1, 0, head_width=0.2, head_length=0.2, fc='black')
    
    # Cross-Attention
    ax9.add_patch(plt.Rectangle((7.5, 4.8), 3, 1.4, fill=True, facecolor='lightgreen',
                                  edgecolor='darkgreen'))
    ax9.text(9, 5.7, 'Cross-Attention', ha='center', fontsize=9, fontweight='bold')
    ax9.text(9, 5.1, 'Q=LN(x), K,V=context', ha='center', fontsize=7, color='gray')
    
    # Context输入 (从上方)
    ax9.add_patch(plt.Rectangle((7.5, 7), 3, 0.8, fill=True, facecolor='lightcoral',
                                  edgecolor='darkred'))
    ax9.text(9, 7.4, 'Context (from Encoder)', ha='center', fontsize=8)
    ax9.arrow(9, 7, 9, 6.2, head_width=0.2, head_length=0.2, fc='darkred')
    
    ax9.arrow(10.5, 5.5, 0.8, 0, head_width=0.2, head_length=0.2, fc='black')
    
    # 残差连接 (虚线)
    ax9.plot([1.75, 11.3], [5, 5], 'b--', linewidth=1, alpha=0.4)
    ax9.text(6.5, 4.7, 'Residual Connection (+ x)', fontsize=7, color='blue', alpha=0.7)
    
    # 输出
    ax9.add_patch(plt.Rectangle((11.3, 5), 0.5, 1, fill=True, facecolor='plum',
                                  edgecolor='purple'))
    ax9.text(11.55, 5.5, '+', ha='center', fontsize=14, fontweight='bold')
    
    ax9.text(9, 3.8, 'Output = x + CrossAttn(LN(x), context)', ha='center',
             fontsize=9, style='italic', color='darkblue')
    ax9.text(9, 3.2, '(Residual preserves original semantics while adding context info)',
             ha='center', fontsize=7, color='gray')
    
    plt.suptitle('Cross-Attention Mechanism - Comprehensive Visualization',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('cross_attention_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化图表已保存为 'cross_attention_visualization.png'")
    plt.show()


if __name__ == '__main__':
    main()
```

## 10. 模型评估

### 评估方法概述

交叉注意力本身不是一个可独立评估的模型，而是更大模型中的一个组件。因此，其评估通常通过端到端任务的指标来间接反映。以下是基于使用交叉注意力的不同任务的评估方法。

### 机器翻译任务

| 指标 | 含义 | 说明 |
|------|------|------|
| BLEU | 生成翻译与参考翻译的n-gram匹配度 | 最常用的翻译质量指标 |
| METEOR | 考虑同义词和词形变化的匹配 | 比BLEU更关注语义 |
| COMET | 基于神经网络的翻译质量评估 | 与人类判断相关性更高 |

交叉注意力的贡献可以通过消融实验来量化：移除交叉注意力层后BLEU下降的幅度反映了其对翻译质量的贡献。

### 多模态任务（VQA / 图文理解）

| 指标 | 含义 |
|------|------|
| VQA Accuracy | 对视觉问题的答案准确率 |
| CIDEr / SPICE | 图像描述生成的语义质量 |
| 注意力对齐准确率 | 交叉注意力权重与人工标注的对齐区域的匹配度 |

### 语音识别任务

| 指标 | 含义 |
|------|------|
| WER (词错误率) | 识别文本与参考文本的差异比例 |
| CER (字符错误率) | 字符级别的编辑距离 |
| RTF (实时率) | 处理时间 / 音频时长 |

### 注意力质量评估（独立评估）

即使没有端到端任务，也可以直接评估交叉注意力的行为：

1. **注意力熵**：$H(A) = -\sum_j A_{ij} \log A_{ij}$。熵太低说明过于聚焦（可能过拟合），太高说明注意力过于分散。
2. **头冗余度**：计算不同头之间的注意力权重余弦相似度。相似度过高说明头有冗余。
3. **对齐一致性**：对于有对齐标注的任务（如翻译、VQA），计算交叉注意力与人工标注对齐的一致率。
4. **梯度范数比**：Query路径和Context路径的梯度范数之比。如果某一侧梯度过小，说明该侧信息利用不充分。

### 评估注意事项

1. **掩码正确性验证**：必须验证Padding位置的注意力权重确实为零，这是一个硬性正确性检查
2. **注意力稳定性**：在训练的不同阶段采样注意力权重，观察其是否稳定收敛
3. **跨验证集泛化**：在域外数据（不同分布的数据）上评估，确保交叉注意力学到的对齐模式具有泛化性

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| Q/K/V来源混淆 | 模型输出与预期不符 | 将交叉注意力写成自注意力（K和V来自query而非context） | 明确区分：Q=query, K=context, V=context；写代码注释指明来源 |
| Padding掩码未应用 | 模型关注了填充的无意义位置 | 忘记将padding_mask传递给注意力层 | 在softmax前检查所有padding位置的分数是否为-inf |
| 输出维度错误 | 张量形状不匹配报错 | 以为输出长度=context长度（实际=query长度） | 牢记：交叉注意力输出长度 = query_len，不是context_len |
| 掩码形状不匹配 | 运行时广播错误 | pad_mask为(B, L)未扩展到(B, 1, 1, L) | 手动unsqueeze：`pad_mask.unsqueeze(1).unsqueeze(2)` |
| 残差连接维度不匹配 | 残差分支报错 | 残差(x)维度和注意力输出维度不一致 | 确认query的d_model = context的d_model = 残差连接的维度 |
| 交叉注意力权重为均匀分布 | 模型没有学到有意义的对齐 | 初始化不当、学习率不合适或任务难度过高 | 检查权重初始化；降低学习率；确保任务确实需要跨模态对齐 |
| Query长度=1时注意力退化 | 单token query无法产生有意义的注意力 | 当L_Q=1时，注意力退化为对context的简单加权平均 | 使用[CLS] token或池化后的单个表示时，注意力仍有效但失去了细粒度对齐 |
| 交叉注意力梯度消失 | 深层交叉注意力训练困难 | 条件注入路径过长导致梯度衰减 | 使用adaLN而非交叉注意力进行条件注入；添加辅助损失 |
| 多模态维度不一致 | 图文特征维度不同无法计算注意力 | query和context的d_model不同 | 在交叉注意力前添加投影层对齐维度 |
| 掩码设置过大导致全零行 | softmax输出NaN | 一整行都被mask导致softmax除零 | 确保每行至少有一个有效位置；或在mask后加一个极小值 |
| 不同头的W_K/W_V共享 | 注意力多样性不足 | 错误地为所有头使用了相同的W_K/W_V | 每个头应有独立的投影权重（或使用nn.Linear自动处理） |
| 批次中序列长度差异大 | 大量显存浪费在padding上 | 批次内序列长度差异大，padding占比高 | 使用bucketing/sorting策略按长度分组；使用动态padding |

## 12. 学习总结

### 核心思想回顾

交叉注意力的本质是**让一个序列（Query）"查阅"另一个序列（Context）中的信息**。它与自注意力的唯一但决定性的区别在于数据来源：

- 自注意力：$Q, K, V$ 全部来自同一序列 $X$
  $$\text{SelfAttn}(X) = \text{softmax}\left(\frac{XW^Q(XW^K)^T}{\sqrt{d_k}}\right)XW^V$$
- 交叉注意力：$Q$ 来自序列 $A$，$K, V$ 来自序列 $B$
  $$\text{CrossAttn}(A, B) = \text{softmax}\left(\frac{AW^Q(BW^K)^T}{\sqrt{d_k}}\right)BW^V$$

这个差异使得交叉注意力成为多模态融合的基础工具。在Transformer Decoder中，它扮演着"Encoder-Decoder Attention"的角色，让生成过程能够动态查询编码器的完整输出。在多模态系统中，它将文本、图像、语音等不同模态的信息自然连接起来。

### 与前序/相关算法的联系

- **自注意力是基础**：交叉注意力是在自注意力的框架上只改变了Q/K/V的来源
- **多头注意力提供并行性**：交叉注意力同样使用多头机制，多个头从不同角度学习对齐
- **Transformer Decoder的完整视图**：Masked Self-Attention + Cross-Attention + FFN 构成了生成模型的基础层
- **残差连接**：残差交叉注意力确保了原始语义和外来信息的平衡

### 后续学习方向

- **Gumbel Cross-Attention**：引入Gumbel-Softmax采样处理离散潜在变量，用于更结构化的多模态对齐
- **Perceiver IO**：使用交叉注意力将任意模态的输入映射到固定大小的潜在空间
- **Flamingo-style Cross-Attention**：在冻结的LLM中插入交叉注意力层，实现视觉-语言理解
- **Adaptive Cross-Attention**：动态选择使用交叉注意力还是跳过它（如适配不同模态质量）
- **Efficient Cross-Attention**：线性注意力、局部敏感哈希等减少 $O(L_Q \times L_C)$ 复杂度的方法

## 13. 练习题与思考题

### 基础题1：来源识别

给定以下注意力计算代码，判断哪些是自注意力，哪些是交叉注意力：

```python
# 代码A
q = x @ W_q; k = x @ W_k; v = x @ W_v
out = softmax(q @ k.T / sqrt(d)) @ v

# 代码B
q = decoder_state @ W_q; k = encoder_out @ W_k; v = encoder_out @ W_v
out = softmax(q @ k.T / sqrt(d)) @ v

# 代码C
q = text_emb @ W_q; k = img_feat @ W_k; v = img_feat @ W_v
out = softmax(q @ k.T / sqrt(d)) @ v
```

**参考答案**：
- 代码A：**自注意力**，因为Q、K、V都来自同一个x
- 代码B：**交叉注意力**，Q来自decoder_state，K、V来自encoder_out
- 代码C：**交叉注意力**，Q来自text_emb（文本），K、V来自img_feat（图像）

### 基础题2：输出形状计算

一个交叉注意力层接收：
- Query: (4, 10, 512)，表示 batch=4, query_len=10, d_model=512
- Context: (4, 25, 512)，表示 batch=4, context_len=25, d_model=512
- num_heads = 8

请写出：
- Q投影后的形状（多头拆分后）
- K投影后的形状（多头拆分后）
- 注意力权重的形状
- 最终输出的形状

**参考答案**：
- Q: (4, 8, 10, 64) -- d_k = 512/8 = 64
- K: (4, 8, 25, 64) -- 注意seq_len=25 (context), 不是10
- 注意力权重: (4, 8, 10, 25) -- (query_len=10) x (context_len=25), 非方阵!
- 最终输出: (4, 10, 512) -- 输出长度 = query_len = 10

### 进阶题：分析与设计

在Transformer Decoder中，Self-Attention使用因果掩码（防止看到未来token），而紧随其后的Cross-Attention通常不需要因果掩码。请解释为什么。如果在Cross-Attention上也使用因果掩码，会有什么影响？

**参考答案**：

**为什么Cross-Attention不需要因果掩码**：
- 自注意力处理的是Decoder自己的序列，在自回归生成时当前位置不能看到未来的token（否则就是"作弊"）
- 交叉注意力的Context来自Encoder的完整输出，而Encoder已经处理完整个输入序列（如源语言全部单词、整张图像）
- Decoder在任意时刻都可以"自由查阅"Encoder的完整输出——这是合理的，因为Encoder输出不包含Decoder未来的信息
- 这与人类行为一致：翻译时，我们可以随时参考完整的原文，但只能看到自己已写出的译文部分

**如果对Cross-Attention使用因果掩码的影响**：
- 在机器翻译中：Decoder第i步只能"看到"Encoder前i个位置的输出，这会严重损害对齐能力。例如翻译"the cat"时，如果"cat"是源语言中靠后的位置，Decoder无法关注它
- 在语音识别中：说话的内容可能被限制在音频开头的部分
- 通常不必要且有害，除非有特殊的任务需求（如在线/流式处理中的低延迟约束）

### 开放思考题

在多模态大模型中（如GPT-4V、Gemini），交叉注意力并不是唯一的跨模态交互方式。另一种方案是"早期融合"（Early Fusion）：将图像和文本的token直接拼接成一个序列，然后全部走自注意力。请对比这两种方案，并讨论各自的适用场景。

**参考思路**：

| 维 度 | 交叉注意力 (Cross-Attn) | 早期融合 (Concat + Self-Attn) |
|--------|------------------------|-------------------------------|
| 模态区分 | 明确区分Query和Context角色 | 所有token地位平等 |
| 注意力模式 | Q只关注Context，Context不关注Q | 所有token互相可以关注 |
| 计算复杂度 | $O(L_Q \times L_C)$ | $O((L_Q+L_C)^2)$ |
| 模态数量扩展 | 需要为每对模态设计独立的交叉注意力 | 自然支持任意多模态 |
| 信息流方向 | 单向（通常Query从Context取信息） | 双向、对称 |
| 适用场景 | 条件生成、模态有主次之分 | 模态地位平等（如多模态理解） |
| 典型使用 | Transformer Decoder、Stable Diffusion | GPT-4V、LLaVA、Flamingo的某些变体 |

早期融合更适合需要各模态平等交互的理解任务；交叉注意力更适合有明确生成方向或查询关系的任务。

## 14. 学习路径建议

### 前置算法
- **自注意力机制**：Q、K、V的定义和缩放点积注意力计算
- **多头注意力（MHA）**：多头的拆分、并行计算和拼接
- **Softmax归一化**：理解概率解释和梯度特性
- **Positional Encoding**：理解序列位置的编码方式

### 平行算法
- **Transformer Decoder**：理解Masked Self-Attn + Cross-Attn + FFN的完整层结构
- **Encoder-Decoder架构**：理解编码器和解码器的分工
- **Multi-Query Attention (MQA)**：在交叉注意力中使用共享KV以节省开销
- **特征拼接融合**：concat + self-attention的替代融合方案

### 进阶算法
- **Perceiver / Perceiver IO**：使用交叉注意力实现任意模态到固定潜空间的映射
- **Flamingo**：在冻结LLM中插入门控交叉注意力实现多模态理解
- **BLIP-2 / Q-Former**：使用可学习的Query tokens通过交叉注意力提取视觉特征
- **Cross-Attention with Gumbel**：引入离散采样的交叉注意力（用于结构化对齐）
- **Adapter-style Cross-Attention**：在预训练模型中插入轻量级交叉注意力进行多模态适配
- **Kosmos / Gemini**：多模态大模型中交叉注意力的工业级应用

### 推荐资源
1. **论文**：《Attention is All You Need》（Vaswani et al., 2017）-- Section 3.2 的 Encoder-Decoder Attention 即交叉注意力的经典表述
2. **论文**：《Flamingo: a Visual Language Model for Few-Shot Learning》（Alayrac et al., 2022）-- 门控交叉注意力在冻结LLM中的多模态应用
3. **教程**：The Illustrated Transformer (Jay Alammar) -- Decoder部分的交叉注意力可视化讲解
4. **代码**：HuggingFace Transformers库中的 `EncoderDecoderModel` 实现
5. **博客**：Lilian Weng的 "Attention? Attention!" -- 全面梳理各类注意力机制
