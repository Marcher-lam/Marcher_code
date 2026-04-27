# Self Attention Detail 自注意力机制详解 学习文档

> 自注意力（Self-Attention）是Transformer架构的核心组件，使模型能够捕捉序列内部的长距离依赖关系

---

## 1. 算法基础认知

### 1.1 一句话定义

**自注意力机制**是一种计算序列中每个位置与其他所有位置相关性的加权机制，通过动态计算注意力权重来聚合全局信息，使得序列中的每个元素都能直接关注到序列的其他元素。

### 1.2 直觉类比

想象你正在阅读一篇论文：当你遇到一个重要的概念时，你的眼睛会自动扫描整个页面，找到之前提到过的相关定义、例子或解释——你的大脑在做类似的事情。**自注意力**就是让模型的每个"单词"都能自动找到序列中最相关的其他"单词"，并根据相关性程度分配不同的注意力权重。

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 2016 | Neural Machine Translation in RNNs with Attention (Bahdanau Attention) |
| 2017 | Attention Is All You Need (Transformer) - 首次提出纯注意力机制 |
| 2018 | BERT - 预训练 + 自注意力 |
| 2020 | Vision Transformer (ViT) - 自注意力扩展到图像 |
| 2022-2024 | GPT系列、LLaMA等大模型的基础 |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 序列建模 / 特征聚合 |
| 核心 | Query-Key-Value 注意力计算 |
| 地位 | Transformer的核心组件 |
| 变体 | Multi-Head Attention, PQA, GQA等 |

### 1.5 前置知识

- 线性代数（矩阵乘法、内积）
- 概率论（Softmax概率分布）
- 深度学习基础（神经网络、前向传播）
- Python + NumPy + PyTorch

---

## 2. 核心原理

### 2.1 核心思想

自注意力的核心思想是：**让序列中的每个位置都能关注到序列的所有位置，并根据相关性动态分配权重**。

给定输入序列 $X = [x_1, x_2, ..., x_n]$，其中每个 $x_i \in \mathbb{R}^d$：

1. **线性投影**：将输入分别投影为Query、Key、Value三个向量
2. **相似度计算**：计算Query与每个Key的相似度（点积）
3. **权重归一化**：使用Softmax将相似度转换为概率分布
4. **信息聚合**：根据权重对Value进行加权求和

### 2.2 QKV机制详解

| 参数 | 角色 | 解释 |
|------|------|------|
| **Query (Q)** | 查询者 | 当前需要关注其他位置的位置 |
| **Key (K)** | 索引 | 每个位置的"索引标签"，用于被查询 |
| **Value (V)** | 内容 | 每个位置的实际信息内容 |

**类比理解**：
- 想象在图书馆找书：Query是你的搜索词，Key是书的索引标签，Value是书的内容
- 你用Query搜索匹配Key，找到最相关的Value

### 2.3 工作流程

```python
# 自注意力计算流程（简化伪代码）
def self_attention(Q, K, V):
    # Step 1: 计算相似度分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (n × n)
    
    # Step 2: 缩放（防止梯度消失）
    scores = scores / sqrt(d_k)
    
    # Step 3: Softmax归一化
    attn_weights = softmax(scores, dim=-1)
    
    # Step 4: 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

### 2.4 Multi-Head Attention

多头注意力的思想是：**并行执行多个注意力函数，每个头学习不同的注意力模式**。

```python
# 多头注意力示意
def multi_head_attention(x, num_heads=8):
    for head in range(num_heads):
        # 每个头有独立的W_Q, W_K, W_V
        head_output, head_attn = self_attention(
            x @ W_Q[head], 
            x @ W_K[head], 
            x @ W_V[head]
        )
        outputs.append(head_output)
    
    # 拼接所有头的输出
    concat = torch.cat(outputs, dim=-1)
    
    # 最终线性变换
    output = concat @ W_O
    
    return output
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $X$ | 输入序列 | $(batch, n, d_{model})$ |
| $Q$ | Query矩阵 | $(batch, n, d_k)$ |
| $K$ | Key矩阵 | $(batch, n, d_k)$ |
| $V$ | Value矩阵 | $(batch, n, d_v)$ |
| $W^Q$ | Query投影矩阵 | $(d_{model}, d_k)$ |
| $W^K$ | Key投影矩阵 | $(d_{model}, d_k)$ |
| $W^V$ | Value投影矩阵 | $(d_{model}, d_v)$ |
| $W^O$ | 输出投影矩阵 | $(h \cdot d_v, d_{model})$ |

### 3.2 单头注意力公式

**Query, Key, Value投影**：
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

**注意力分数计算（点积）**：
$$\text{scores} = QK^T$$

**缩放点积（防止数值过大）**：
$$\text{scores}_{scaled} = \frac{QK^T}{\sqrt{d_k}}$$

**Softmax归一化**：
$$\text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.3 多头注意力公式

**每个头的注意力计算**：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^KVW_i^V)$$

**多头输出拼接**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

### 3.4 详细推导

**为什么需要缩放因子 $\sqrt{d_k}$？**

设 $q_i, k_j \sim \mathcal{N}(0, 1)$ 为独立同分布，则：
$$E[q_i \cdot k_j] = 0, \quad \text{Var}(q_i \cdot k_j) = 1$$

点积的方差：
$$\text{Var}\left(\sum_i q_i k_i\right) = d_k$$

当 $d_k$ 较大时，方差也会很大，导致Softmax的梯度非常小（接近0/1的边界）。

**缩放后的方差**：
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1$$

这保证了梯度的稳定传播。

### 3.5 掩码机制（Masking）

**Padding Mask**：屏蔽.pad token
$$M_{pad}(i) = \begin{cases} 0 & \text{if } x_i = \text{pad} \ 1 & \text{otherwise} \end{cases}$$

**Causal Mask**：保证自回归生成
$$M_{causal}(i, j) = \begin{cases} 0 & \text{if } j > i \ 1 & \text{otherwise} \end{cases}$$

### 3.6 补充公式

**Scaled Dot-Product Attention**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**带掩码的注意力**：
$$\text{Attention}_{masked}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 通常是一个非常大的负数（$-infty$）用于屏蔽。

---

## 4. PyTorch实现

### 4.1 基础自注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """基础自注意力机制"""
    
    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) or (batch, seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 线性投影
        Q = self.W_Q(x).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)
        
        # 缩放点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 拼接和输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_O(context)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        多头注意力前向传播
        
        Args:
            x: (batch, seq_len, d_model)
            mask: optional mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 投影
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 转置以便多头计算
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 缩放点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 拼接多头输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.W_O(context)
        
        return output, attn_weights
```

### 4.2 带掩码的多头注意力

```python
def create_causal_mask(seq_len, device):
    """创建因果掩码（下三角掩码）"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def create_padding_mask(token_ids, pad_token_id=0):
    """创建padding掩码"""
    return (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)


class MaskedMultiHeadAttention(nn.Module):
    """带掩码的多头注意力"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MaskedMultiHeadAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(self, x, causal=True, padding=True, pad_token_id=0):
        masks = []
        
        if causal:
            mask = create_causal_mask(x.size(2), x.device)
            masks.append(mask)
        
        if padding:
            pad_mask = create_padding_mask(x, pad_token_id)
            masks.append(pad_mask)
        
        # 合并所有掩码
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask * m
        else:
            mask = None
        
        return self.attention(x, mask)
```

### 4.3 完整Transformer Encoder层

```python
class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder层（包含自注意力 + FFN）"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        
        # FFN + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights
```

---

## 5. 代码示例

### 5.1 完整自注意力训练示例

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def demo_self_attention():
    """演示自注意力机制的工作过程"""
    
    # 简单的序列数据
    seq_len = 5
    d_model = 8
    
    # 模拟输入：一个句子（5个词，每个词向量维度8）
    x = torch.randn(1, seq_len, d_model)
    
    print(f"输入序列形状: {x.shape}")
    print("=" * 50)
    
    # 创建自注意力层
    attention = SelfAttention(d_model)
    attention.eval()
    
    # 前向传播
    with torch.no_grad():
        output, attn_weights = attention(x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print()
    
    # 可视化注意力权重
    plt.figure(figsize=(10, 8))
    attn = attn_weights[0].numpy()
    
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(attn[i], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'Head {i+1}')
    
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    plt.close()
    
    print("注意力权重可视化已保存到 attention_weights.png")
    
    # 分析注意力模式
    print("\n注意力模式分析:")
    print(f"- 每个头的注意力分布形状: {attn[0].shape}")
    print(f"- 第一行（第一个词关注其他词）: {attn[0][0]}")
    print(f"- 对角线（自注意力）: {[attn[i][i] for i in range(min(5, seq_len))]}")
    
    return attn_weights


def demo_multi_head_patterns():
    """演示多头注意力的不同注意力模式"""
    
    # 创建多头注意力
    d_model = 16
    num_heads = 4
    seq_len = 6
    
    mha = MultiHeadAttention(d_model, num_heads)
    mha.eval()
    
    # 测试不同类型的输入
    test_cases = {
        "相同token": torch.ones(1, seq_len, d_model),
        "随机序列": torch.randn(1, seq_len, d_model),
        "梯度序列": torch.arange(seq_len, dtype=torch.float).unsqueeze(0).expand(-1, d_model).unsqueeze(0),
    }
    
    print("\n不同输入的注意力模式:")
    print("=" * 50)
    
    for name, x in test_cases.items():
        with torch.no_grad():
            _, attn = mha(x)
        
        print(f"\n{name}:")
        for h in range(num_heads):
            attn_head = attn[0, h].numpy()
            # 找到每个位置最关注的token
            most_attended = np.argmax(attn_head, axis=1)
            print(f"  Head {h}: {most_attended.tolist()}")


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    # 演示自注意力
    attn_weights = demo_self_attention()
    
    # 演示多头模式
    demo_multi_head_patterns()
```

---

## 6. 应用场景

### 6.1 NLP应用

| 应用 | 说明 |
|------|------|
| **机器翻译** | Transformer encoder-decoder架构 |
| **文本分类** | BERT, RoBERTa等预训练模型 |
| **命名实体识别** | Token-level分类 |
| **问答系统** | SQuAD等阅读理解任务 |
| **文本生成** | GPT系列自回归生成 |

### 6.2 多模态应用

| 应用 | 说明 |
|------|------|
| **视觉Transformer** | ViT将图像划分为patch序列 |
| **Video Understanding** | 时空注意力 |
| **多模态融合** | CLIP, BLIP等 |
| **语音识别** | Conformer |

### 6.3 代码实现

```python
# Vision Transformer (ViT) 中的自注意力
class ViTAttention(nn.Module):
    """ViT中的注意力机制"""
    
    def __init__(self, d_model, num_heads=16):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, N+1, D) - [CLS] + N个patches
        attn_output, _ = self.attention(x)
        x = self.norm(x + attn_output)
        return x
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **并行计算** | 相比RNN，自注意力可完全并行 |
| **长距离建模** | 直接计算任意位置间的关系 |
| **可解释性** | 注意力权重可可视化 |
| **表达能力强** | 多头可学习不同模式 |
| **位置感知** | 配合位置编码可处理位置信息 |

### 7.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| **$O(n^2)$复杂度** | 序列长度二次方 | Sparse Attention, Linear Attention |
| **内存占用大** | KV Cache大 | GQA, PQA |
| **顺序感知弱** | 无位置信息 | 位置编码 |
| **可能过平滑** | 忽略位置差异 | 相对位置编码 |

### 7.3 与其他机制对比

| 机制 | 复杂度 | 长距离建模 | 并行度 |
|------|--------|------------|--------|
| RNN | $O(n \cdot d)$ | $O(n)$ | 低 |
| CNN | $O(k \cdot n)$ | $O(\log n)$ | 高 |
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | 高 |

---

## 8. 常见问题与易错点

### 8.1 数值问题

**问题**：Softmax梯度消失

**原因**：当分数差异过大时，Softmax趋近于one-hot，梯度接近0

**解决方案**：
```python
# 缩放点积
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

# 或者使用数值稳定的Softmax
attn_weights = F.softmax(scores, dim=-1)  # 内部已实现数值稳定
```

### 8.2 掩码问题

**问题**：Padding token被关注

**解决方案**：
```python
def create_padding_mask(padding_token_id):
    mask = (token_ids != padding_token_id).unsqueeze(1).unsqueeze(2)
    return mask

# 应用掩码
scores = scores.masked_fill(~padding_mask, float('-inf'))
```

### 8.3 效率问题

**问题**：长序列时内存爆炸

**解决方案**：使用稀疏注意力或线性注意力
```python
# 局部注意力（只关注前后k个token）
def local_attention(x, window_size=5):
    seq_len = x.size(1)
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1
    return local_attention_fn(x, mask)
```

---

## 9. 学习总结

### 9.1 核心要点

1. **QKV机制**：Query找Key，Value承载信息
2. **缩放点积**：防止梯度消失的核心
3. **多头注意力**：并行学习多种模式
4. **掩码机制**：控制信息流动

### 9.2 关键公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

### 9.3 学习路径

自注意力 → 多头注意力 → Transformer → BERT/GPT → Vision Transformer

---

## 10. 练习题

### 10.1 基础题

1. **QKV维度计算**：设 $d_{model}=512$, $n_{head}=8$，求Q、K、V、O的维度

2. **注意力权重分析**：给定注意力权重矩阵，分析每个token最关注哪个位置

### 10.2 进阶题

3. **手动实现**：用NumPy实现基础自注意力

4. **可视化**：绘制不同头在不同层的注意力模式

### 10.3 答案

<details>
<summary>答案1</summary>

$Q, K, V$ 维度：$(seq, d_{model}) \times (d_{model}, d_{model}) = (seq, d_{model})$

或者从多头角度：每个头 $d_k = 512/8 = 64$，Q、K、V为 $(seq, 8, 64)$ → 转置后 $(8, seq, 64)$

</details>

<details>
<summary>答案2</summary>

取每行最大值位置：`np.argmax(attn_weights, axis=1)`

</details>

<details>
<summary>答案3</summary>

```python
import numpy as np

def self_attention_numpy(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    output = np.dot(attn_weights, V)
    return output, attn_weights

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段：掌握基础

1. 理解QKV机制
2. 理解缩放点积
3. 理解Softmax
4. 用PyTorch实现基础自注意力

**时间**：1周

### 11.2 第二阶段：深入理解

1. 理解多头注意力
2. 理解掩码机制
3. 理解位置编码
4. 分析不同任务的注意力模式

**时间**：1-2周

### 11.3 第三阶段：扩展学习

1. Sparse Attention
2. Linear Attention
3. Flash Attention
4. GQA/PQA

**时间**：2周

### 11.4 推荐资源

- Attention Is All You Need (2017)
- BERT: Pre-training of Deep Bidirectional Transformers
- Lilian Weng的博客：Attention and Transformers
- PyTorch官方教程：Transformer

---

## 12. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens=None):
    """可视化注意力权重"""
    
    if attention_weights.ndim == 4:
        # 多头：取平均
        attn = attention_weights.mean(dim=1)[0]
    else:
        attn = attention_weights[0]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap='viridis', annot=True if len(attn) < 10 else False)
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.title('Self-Attention Weights')
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.close()


def analyze_attention_patterns(attention_weights):
    """分析注意力模式"""
    
    attn = attention_weights[0]  # (num_heads, seq_len, seq_len)
    
    patterns = {
        'diagonal': [],  # 对角线（关注邻近）
        'vertical': [],  # 列（被关注最多）
        'horizontal': [],  # 行（关注最多）
        'uniform': [],  # 均匀分布
    }
    
    for h in range(attn.shape[0]):
        head_attn = attn[h]
        
        # 对角线
        diag = np.mean([head_attn[i, i] for i in range(min(head_attn.shape))])
        patterns['diagonal'].append(diag)
        
        # 被关注最多的位置
        vertical = np.mean(np.argmax(head_attn, axis=1))
        patterns['vertical'].append(vertical)
        
        # 熵（多样性）
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-9), axis=-1).mean()
        patterns['uniform'].append(entropy)
    
    return patterns
```

---

## 13. 模型评估

### 13.1 评估指标

| 指标 | 说明 |
|------|------|
| **Perplexity** | 语言模型困惑度 |
| **BLEU** | 机器翻译得分 |
| **ROUGE** | 摘要生成得分 |
| **准确率** | 分类任务 |

### 13.2 注意力质量评估

```python
def evaluate_attention_quality(attention_weights):
    """评估注意力质量"""
    
    attn = attention_weights[0]
    
    metrics = {}
    
    # 1. 稀疏性：越稀疏说明注意力越集中
    metrics['sparsity'] = (attn < 0.01).sum() / attn.numel()
    
    # 2. 多样性：不同头的差异
    metrics['head_diversity'] = attn.std(dim=1).mean()
    
    # 3. 对角线强度（对位置建模）
    diag_sum = sum(attn[i, :, i].mean() for i in range(min(attn.shape[1], 10)))
    metrics['diagonal_strength'] = diag_sum / min(attn.shape[1], 10)
    
    return metrics
```

---

## 14. 补充内容

### 14.1 注意力变体

| 变体 | 描述 |
|------|------|
| **Scaled Dot-Product** | 标准缩放点积 |
| **Additive** | 加性注意力 |
| **Multi-Query** | 多Query单KV |
| **Grouped Query** | 分组Query共享KV |
| **Linear** | 线性复杂度 |
| **Sparse** | 稀疏模式 |

### 14.2 Flash Attention原理

Flash Attention使用IO-aware算法，将 $O(N^2)$ 内存降到 $O(N)$：

1. 分块计算（tiling）
2. 在线Softmax
3. 熔断机制（recount）

### 14.3 实践建议

1. **短序列**：使用标准自注意力
2. **长序列**：使用Flash Attention或Sparse Attention
3. **内存受限**：使用GQA代替MHA
4. **可解释性**：分析注意力权重可视化

---

**文档结束**

*参考论文：Attention Is All You Need (Vaswani et al., 2017)*