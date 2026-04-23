# Attention机制 学习文档

> 让模型学会"关注重点"——从Bahdanau到Self-Attention

---

## 1. 算法基础认知

### 1.1 什么是Attention

**注意力机制（Attention Mechanism）** 让模型在处理信息时能够"聚焦"于最相关的部分，而不是平均对待所有输入。

```
人类阅读: "今天天气很好，适合去公园散步"
          ↑↑↑↑↑↑↑↑          ↑↑↑↑↑↑↑↑
          关注"天气"          关注"活动"

注意力机制: 给每个词分配一个权重，重要的词权重高
```

### 1.2 为什么需要Attention

| 问题 | 无Attention | 有Attention |
|------|-----------|------------|
| 长序列 | 最后的向量"遗忘"早期信息 | 直接关注任意位置 |
| 相关性 | 所有输入平均对待 | 重要的输入权重高 |
| 可解释性 | 黑盒 | 可以查看注意力权重 |

### 1.3 在推荐系统中的核心地位

**Attention是现代推荐系统的基石！**

- **DIN（阿里）**：用Attention对用户历史行为加权
- **Transformer推荐**：Self-Attention建模序列
- **BST**：Behavior Sequence Transformer
- **MHA**：多头注意力捕捉多种兴趣

---

## 2. 核心原理

### 2.1 Query-Key-Value框架

Attention的核心是QKV框架：

```
类比图书管检索:
- Query(Q): 你想找什么（"推荐系统"相关）
- Key(K):   每本书的标签（"机器学习"、"推荐"、"深度学习"）
- Value(V): 每本书的内容

流程:
1. 用Q和所有K计算相似度 → 得到注意力分数
2. 用Softmax归一化 → 得到注意力权重
3. 用权重对V加权求和 → 得到最终结果
```

### 2.2 注意力计算步骤

```
Step 1: 计算注意力分数
  score_i = Q · Kᵢ  (Query与每个Key的点积)

Step 2: 归一化
  α_i = softmax(score_i / √d_k)  (除以√d_k防止梯度消失)

Step 3: 加权求和
  context = Σ α_i · V_i
```

### 2.3 注意力变体

| 类型 | Score函数 | 说明 |
|------|----------|------|
| **点积注意力** | Q·K | 最简单 |
| **缩放点积** | Q·K/√dₖ | Transformer使用 |
| **加性注意力** | vᵀtanh(W₁Q+W₂K) | Bahdanau提出 |
| **余弦注意力** | cos(Q,K) | 归一化相似度 |

---

## 3. 数学公式与推导

### 3.1 缩放点积注意力（Transformer使用）

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

展开：
- $Q \in \mathbb{R}^{n \times d_k}$（n个Query）
- $K \in \mathbb{R}^{m \times d_k}$（m个Key）
- $V \in \mathbb{R}^{m \times d_v}$（m个Value）

$$QK^T \in \mathbb{R}^{n \times m} \xrightarrow{/\sqrt{d_k}} \xrightarrow{\text{softmax}} \alpha \in \mathbb{R}^{n \times m}$$

$$\text{Output} = \alpha V \in \mathbb{R}^{n \times d_v}$$

### 3.2 为什么除以√dₖ

当 $d_k$ 很大时，$QK^T$ 的方差也变大，Softmax的梯度趋近于0（进入饱和区）。

$$\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q) \cdot \text{Var}(k)$$

除以 $\sqrt{d_k}$ 使方差回到1：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) \approx 1$$

### 3.3 自注意力（Self-Attention）

Q、K、V都来自同一个输入X：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{XW_Q(XW_K)^T}{\sqrt{d_k}}\right)XW_V$$

> 自注意力让序列中的每个位置都能直接关注其他所有位置。

### 3.4 多头注意力（Multi-Head Attention）

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

> 多头让模型同时关注不同类型的信息（如：有的头关注语法，有的关注语义）。

---

## 5. 应用场景

### 5.1 DIN中的注意力

```python
# DIN (Deep Interest Network) 的核心思想:
# 用户历史: [点击了衣服, 点击了鞋子, 点击了手机, 点击了包]
# 候选物品: 手机壳
# 
# Attention权重:
#   衣服 → 0.1 (不相关)
#   鞋子 → 0.1 (不相关)
#   手机 → 0.7 (非常相关！) ← 注意力机制自动发现这个关联
#   包   → 0.1 (不相关)
#
# 加权后的用户表征更关注与当前候选相关的历史行为
```

---

## 7. 调库实现

```python
"""
Attention 机制完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, n_heads, seq_q, d_k)
        K: (batch, n_heads, seq_k, d_k)
        V: (batch, n_heads, seq_k, d_v)
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换 + 分头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        context, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        # 输出变换
        output = self.W_O(context)
        
        return output, attn_weights


class DINAttention(nn.Module):
    """
    DIN注意力机制
    用于推荐系统中用户历史行为的加权
    """
    
    def __init__(self, embedding_dim, hidden_units):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_units),  # [item, target, item-target, item*target]
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
    
    def forward(self, item_embeds, target_embed):
        """
        item_embeds: (batch, seq_len, embed_dim) 用户历史行为
        target_embed: (batch, embed_dim) 候选物品
        """
        seq_len = item_embeds.size(1)
        
        # 扩展target到序列长度
        target_expanded = target_embed.unsqueeze(1).expand_as(item_embeds)
        
        # 拼接特征 [item, target, item-target, item*target]
        attention_input = torch.cat([
            item_embeds,
            target_expanded,
            item_embeds - target_expanded,
            item_embeds * target_expanded
        ], dim=-1)
        
        # 计算注意力权重
        attn_scores = self.attention_mlp(attention_input).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权求和
        weighted_embed = torch.bmm(attn_weights.unsqueeze(1), item_embeds).squeeze(1)
        
        return weighted_embed, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. 缩放点积注意力
    print("=== 缩放点积注意力 ===")
    Q = torch.randn(2, 8, 16)  # (batch, seq, dim)
    K = torch.randn(2, 8, 16)
    V = torch.randn(2, 8, 16)
    
    attention = ScaledDotProductAttention()
    context, weights = attention(Q, K, V)
    print(f"输入: Q{Q.shape}, K{K.shape}, V{V.shape}")
    print(f"输出: context{context.shape}, weights{weights.shape}")
    print(f"权重和: {weights[0, 0].sum().item():.4f}")  # 应该≈1.0
    
    # 2. 多头注意力
    print("\n=== 多头注意力 ===")
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 64)
    output, weights = mha(x, x, x)  # Self-Attention
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    
    # 3. DIN注意力
    print("\n=== DIN注意力 ===")
    din_attn = DINAttention(embedding_dim=32, hidden_units=16)
    history = torch.randn(4, 10, 32)  # 4用户，10个历史行为
    target = torch.randn(4, 32)       # 候选物品
    weighted, attn_w = din_attn(history, target)
    print(f"历史行为: {history.shape}")
    print(f"候选物品: {target.shape}")
    print(f"加权表征: {weighted.shape}")
    print(f"注意力权重: {attn_w.shape}")
    print(f"权重示例: {attn_w[0].detach().numpy().round(3)}")
```

---

## 8. 手工代码实现

```python
"""
缩放点积注意力 + 自注意力 —— 纯 NumPy
"""
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    缩放点积注意力
    Q: (n, d_k), K: (m, d_k), V: (m, d_v)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数
    scores = Q @ K.T / np.sqrt(d_k)  # (n, m)
    
    # 2. Softmax归一化
    attn_weights = softmax(scores)     # (n, m)
    
    # 3. 加权求和
    context = attn_weights @ V         # (n, d_v)
    
    return context, attn_weights

def self_attention(X, W_Q, W_K, W_V):
    """
    自注意力
    X: (seq_len, d_model)
    W_Q, W_K, W_V: (d_model, d_k)
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    return scaled_dot_product_attention(Q, K, V)

if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟序列: 5个词，每个词8维
    X = np.random.randn(5, 8)
    W_Q = np.random.randn(8, 4)
    W_K = np.random.randn(8, 4)
    W_V = np.random.randn(8, 4)
    
    context, weights = self_attention(X, W_Q, W_K, W_V)
    print(f"输入序列: {X.shape}")
    print(f"自注意力输出: {context.shape}")
    print(f"\n注意力权重矩阵:")
    print(np.round(weights, 3))
    print(f"\n每行和: {weights.sum(axis=1)}")  # 每行≈1.0
```

---

## 12. 学习总结

1. **Attention = QKV加权**：用Query和Key的相似度对Value加权
2. **缩放点积**：$\text{softmax}(QK^T/\sqrt{d_k})V$，除以√dₖ稳定训练
3. **Self-Attention**：Q=K=V来自同一输入，建模序列内部关系
4. **Multi-Head**：多个头并行计算，捕捉不同类型的关系
5. **推荐核心**：DIN用Attention对用户历史行为加权

---

## 14. 学习路径

```
RNN/LSTM → [当前: Attention] → Multi-Head Attention → Transformer
                                              ↓
                                    DIN/DIEN/BST/Transformer推荐
```
