# Transformer 学习文档

> "Attention Is All You Need"——改变AI的里程碑

---

## 1. 算法基础认知

### 1.1 什么是Transformer

**Transformer** 是2017年Google在论文"Attention Is All You Need"中提出的模型架构，完全基于注意力机制，抛弃了RNN的循环结构。

### 1.2 为什么Transformer是革命性的

| 维度 | RNN | Transformer |
|------|-----|-------------|
| 并行性 | 必须逐步计算 | 完全并行 |
| 长距离依赖 | 受限于梯度消失 | 直接关注任意位置 |
| 计算复杂度 | O(n·d²) | O(n²·d) |
| 可扩展性 | 难以扩展到大规模 | GPT/BERT已证明可扩展 |

### 1.3 在推荐系统中的地位

```
Transformer 在推荐系统中的直接应用:
- BERT4Rec: 用BERT做序列推荐
- SASRec:  用Self-Attention做序列推荐
- BST:     用Transformer建模用户行为序列
- DIEN:    兴趣演化（GRU变体，但思想类似）
- 大模型推荐: GPT/BERT直接用于推荐
```

---

## 2. 核心原理

### 2.1 整体架构

```
Transformer = Encoder + Decoder

Encoder: 输入 → N × [Multi-Head Attention → FFN] → 编码表示
Decoder: 目标 → N × [Masked MHA → Cross-Attention → FFN] → 输出

推荐系统通常只用Encoder部分
```

### 2.2 核心组件

```
┌────────────────────────────────────────┐
│           Transformer Block            │
│                                        │
│  Input → LayerNorm → Multi-Head Attn   │
│          ↑              ↓              │
│          └──── + (残差连接)             │
│                   ↓                    │
│          LayerNorm → FFN               │
│          ↑              ↓              │
│          └──── + (残差连接)             │
│                   ↓                    │
│                Output                  │
└────────────────────────────────────────┘
```

### 2.3 位置编码（Positional Encoding）

由于Transformer没有循环结构，需要额外注入位置信息：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

## 3. 数学公式与推导

### 3.1 Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.2 Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

通常 $d_{ff} = 4 \times d_{model}$（如 512 → 2048 → 512）

### 3.3 完整Encoder Block

$$\text{Output} = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))$$

$$\text{Final} = \text{LayerNorm}(\text{Output} + \text{FFN}(\text{Output}))$$

### 3.4 参数量分析

| 组件 | 参数量 (d=512, h=8) |
|------|-------------------|
| MHA (4个W矩阵) | 4 × 512² = 1,048,576 |
| FFN | 512×2048 + 2048×512 = 2,097,152 |
| LayerNorm | 2 × 512 = 1,024 |
| **单个Block** | **≈ 3.1M** |
| 6层Encoder | ≈ 18.6M |

---

## 4. 训练过程讲解

### 4.1 训练技巧

```
1. 学习率预热 (Warmup):
   lr = d_model^{-0.5} × min(step^{-0.5}, step × warmup_steps^{-1.5})
   先线性增加到峰值，再按step^{-0.5}衰减

2. Label Smoothing:
   真实标签从1.0变为1-ε=0.9，其余类0.1/(K-1)

3. Dropout:
   注意力权重 + Embedding + 子层输出

4. 梯度裁剪:
   clip_grad_norm = 1.0
```

---

## 5. 应用场景

### 5.1 SASRec (Self-Attentive Sequential Recommendation)

```python
# SASRec核心思想:
# 用户序列 → 位置编码 → N × Transformer Block → 预测下一个物品
# 不需要RNN，纯Self-Attention
```

### 5.2 BST (Behavior Sequence Transformer)

```
用户行为序列: [点击A, 浏览B, 购买C, ...]
                ↓
Embedding + Positional Encoding
                ↓
Transformer Encoder
                ↓
与其他特征拼接 → MLP → CTR预估
```

---

## 7. 调库实现

```python
"""
Transformer 完整实现 - PyTorch
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """单个Transformer Block"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention + 残差 + LayerNorm
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Transformer Encoder（推荐系统常用）"""
    
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation
    基于Transformer的序列推荐模型
    """
    
    def __init__(self, num_items, d_model=64, n_heads=2, n_layers=2, max_len=50):
        super().__init__()
        
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            n_layers=n_layers
        )
        self.output_layer = nn.Linear(d_model, num_items)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, item_seq):
        """
        item_seq: (batch, seq_len) 物品ID序列
        """
        # Embedding + Position
        x = self.item_embedding(item_seq)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Causal Mask（防止看到未来的行为）
        seq_len = item_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(item_seq.device)
        
        # Transformer Encoder
        x, _ = self.encoder(x, mask=None)  # SASRec不用mask也行
        
        # 预测
        logits = self.output_layer(x)  # (batch, seq_len, num_items)
        
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. 基本Transformer Encoder
    print("=== Transformer Encoder ===")
    d_model = 64
    encoder = TransformerEncoder(d_model=d_model, n_heads=4, d_ff=256, n_layers=2)
    x = torch.randn(2, 10, d_model)
    out, weights = encoder(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    
    # 2. SASRec 序列推荐
    print("\n=== SASRec 序列推荐 ===")
    model = SASRec(num_items=1000, d_model=64, n_heads=2, n_layers=2)
    item_seq = torch.randint(0, 1000, (4, 20))  # 4用户，20个历史行为
    logits = model(item_seq)
    print(f"行为序列: {item_seq.shape}")
    print(f"预测输出: {logits.shape}")  # (4, 20, 1000)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
```

---

## 8. 手工代码实现

```python
"""
Transformer 核心组件 —— 纯 NumPy
"""
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads):
    """
    多头自注意力
    X: (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
    
    Q = X @ W_Q  # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V
    
    # 分头: (seq_len, d_model) → (n_heads, seq_len, d_k)
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # 每个头独立计算注意力
    heads = []
    for i in range(n_heads):
        scores = Q[i] @ K[i].T / np.sqrt(d_k)
        weights = softmax(scores)
        head = weights @ V[i]
        heads.append(head)
    
    # 合并头
    concat = np.concatenate(heads, axis=-1)  # (seq_len, d_model)
    output = concat @ W_O
    
    return output

def transformer_block(X, params, n_heads):
    """单个Transformer Block"""
    # Self-Attention
    attn_out = multi_head_attention(
        X, params['W_Q'], params['W_K'], params['W_V'], params['W_O'], n_heads
    )
    # 残差 + LayerNorm
    X = layer_norm(X + attn_out)
    
    # FFN
    ffn_out = np.maximum(0, X @ params['W1']) @ params['W2']
    # 残差 + LayerNorm
    X = layer_norm(X + ffn_out)
    
    return X

if __name__ == "__main__":
    np.random.seed(42)
    
    d_model = 64
    n_heads = 4
    seq_len = 10
    d_ff = 256
    
    X = np.random.randn(seq_len, d_model)
    
    params = {
        'W_Q': np.random.randn(d_model, d_model) * 0.1,
        'W_K': np.random.randn(d_model, d_model) * 0.1,
        'W_V': np.random.randn(d_model, d_model) * 0.1,
        'W_O': np.random.randn(d_model, d_model) * 0.1,
        'W1': np.random.randn(d_model, d_ff) * 0.1,
        'W2': np.random.randn(d_ff, d_model) * 0.1,
    }
    
    output = transformer_block(X, params, n_heads)
    print(f"输入: {X.shape}")
    print(f"输出: {output.shape}")
    print("Transformer Block执行成功！")
```

---

## 12. 学习总结

1. **Transformer = Self-Attention + FFN + 残差 + LayerNorm**
2. **核心**：Multi-Head Attention并行计算序列内所有位置关系
3. **位置编码**：注入位置信息，弥补无循环结构的不足
4. **O(n²)复杂度**：序列长度二次方，长序列的开销大
5. **推荐应用**：SASRec, BERT4Rec, BST等经典模型

---

## 13. 练习题

**Q1**：为什么Transformer比RNN更适合做序列推荐？

<details>
<summary>答案</summary>

1. **并行计算**：RNN必须逐步处理，Transformer一次处理整个序列
2. **长距离依赖**：RNN受梯度消失限制，Transformer直接关注任意位置
3. **灵活的注意力**：用户行为序列中，不同历史行为对当前预测的重要性不同，Attention自动学习这种关系
4. **可扩展性**：容易通过增加层数和维度提升效果

</details>

---

## 14. 学习路径

```
Attention → [当前: Transformer] → BERT4Rec(序列推荐) → SASRec → GPT/BERT
```
