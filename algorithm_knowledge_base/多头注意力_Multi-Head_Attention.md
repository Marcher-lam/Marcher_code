# 多头注意力（Multi-Head Attention）学习文档

> 并行运行多个注意力头，让模型能够关注不同类型的特征。

## 1. 算法基础认知

### 一句话定义

多头注意力通过并行运行多个独立的注意力机制，让模型能够在不同的表示子空间中捕获多种类型的依赖关系。

### 直觉类比

就像同时问三个人对同一问题的看法，每个人关注的重点可能不同。有的人关注语法，有的人关注情感，有的人关注事实。多头注意力正是让模型同时从多个角度"看"输入。

### 历史背景

- **2017年**：Vaswani等人在"Attention Is All You Need"中首次提出多头注意力，作为Transformer的核心组件
- **2018年**：BERT使用12~16个注意力头
- **2020年**：ViT等视觉模型也采用多头注意力
- **后续**：头数从8增加到128（如GPT-3），但性能收益递减

### 算法定位

多头注意力是**Transformer的核心组件**，是自注意力的增强版本。

## 2. 核心原理

### 2.1 核心思想

将输入投影到多个不同的"表示空间"，在每个空间中分别计算注意力，然后拼接。这让模型能够同时捕获句法关系、语义关系、位置关系等不同类型的依赖。

### 2.2 工作流程

```
输入X → 线性变换 Q,K,V → 分割h个头 → 并行的缩放点积注意力
→ 拼接所有头 → 线性变换 → 输出
```

### 2.3 为什么多个头比单头好？

- **单头注意力**：只能捕获一种类型的依赖关系
- **多头注意力**：每个头在不同的子空间中学习不同的关系模式。例如：
  - 头1：关注语法依赖（主语-动词）
  - 头2：关注语义依赖（实体-属性）
  - 头3：关注位置依赖（附近词）

## 3. 数学公式与推导

### 3.1 基本公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3.2 变换矩阵

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$：第$i$个头的Query投影
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：第$i$个头的Key投影
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$：第$i$个头的Value投影
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$：输出投影

### 3.3 维度关系

通常设置 $d_k = d_v = d_{\text{model}} / h$，使得总参数量和计算量不变：

$$\text{总参数量} = h \times \frac{d_{\text{model}}}{h} \times d_{\text{model}} \times 3 + d_{\text{model}} \times d_{\text{model}} = 4 \times d_{\text{model}}^2$$

### 3.4 为什么除以h？

保持计算量不变。如果 $h=8$, $d_k = d_{\text{model}}/8$，8个头的总计算量 = 1个完整单头注意力的计算量。

## 4. 训练过程

多头注意力随整个模型一起训练，不需要单独训练。关键点：
- 所有头的 $W_i^Q, W_i^K, W_i^V$ 是独立随机初始化的
- 训练过程中，不同的头自动分化——学习不同类型的注意力模式
- Dropout通常在注意力权重上应用（防止过拟合）

### 4.1 训练细节
- Dropout rate: 0.1（Transformer默认）
- 初始化：xavier uniform
- 每个头的 $d_k$ 越小，训练越稳定（高维点积容易饱和）

## 5. 应用场景

1. **Transformer核心**：所有Transformer变体（BERT、GPT、T5）使用多头注意力
2. **机器翻译**：捕获源语言和目标语言的多种关系
3. **文本摘要**：同时关注不同的文档区域
4. **视觉Transformer**：ViT、Swin Transformer等使用多头注意力

## 6. 优缺点分析

### 优点
1. **多视角**：捕获不同类型的依赖关系
2. **可解释**：可视化各头的注意力权重，分析模型行为
3. **稳定训练**：每个头维度小，避免softmax饱和
4. **即插即用**：可替换任何单头注意力

### 缺点
1. **参数增加**：$h$ 组投影矩阵带来额外参数
2. **计算量增加**：$h$ 倍的注意力计算
3. **头冗余**：有些头学习到的模式可能重复
4. **内存消耗**：需要保存 $h$ 个注意力矩阵

## 7. 调库实现

```python
"""
多头注意力的完整PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力

    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: Dropout概率
        bias: 是否使用偏置
    """

    def __init__(self, d_model, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Q, K, V投影
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

        # 输出投影
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, Q, K, V, mask=None, return_attention=False):
        """前向传播

        参数:
            Q: Query (B, Lq, d_model)
            K: Key (B, Lk, d_model)
            V: Value (B, Lv, d_model) — 通常Lv=Lk
            mask: 注意力掩码 (B, Lq, Lk)
            return_attention: 是否返回注意力权重

        返回:
            output: (B, Lq, d_model)
            attention_weights: (B, num_heads, Lq, Lk) 可选
        """
        batch_size, Lq, _ = Q.shape
        _, Lk, _ = K.shape
        _, Lv, _ = V.shape

        # 1. 线性投影并分割多头
        # Q: (B, Lq, d_model) -> (B, Lq, num_heads, d_k) -> (B, num_heads, Lq, d_k)
        Q = self.W_Q(Q).view(batch_size, Lq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, Lk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, Lv, self.num_heads, self.d_v).transpose(1, 2)

        # 2. 缩放点积注意力
        # scores: (B, num_heads, Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask: (B, Lq, Lk) -> (B, 1, Lq, Lk)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. 加权聚合
        # context: (B, num_heads, Lq, d_v)
        context = torch.matmul(attn_weights, V)

        # 4. 合并多头
        # context: (B, Lq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, Lq, self.d_model)

        # 5. 输出投影
        output = self.W_O(context)

        if return_attention:
            return output, attn_weights
        return output


def demo():
    """多头注意力演示"""
    batch_size, seq_len, d_model, num_heads = 4, 16, 512, 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # 自注意力（Q=K=V=x）
    out, attn = mha(x, x, x, return_attention=True)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"注意力: {attn.shape}")

    # 查看每个头的统计
    print(f"\n每个头的注意力统计:")
    for h in range(num_heads):
        head_attn = attn[0, h]
        entropy = -(head_attn * torch.log(head_attn + 1e-8)).sum(dim=-1).mean()
        print(f"  头 {h+1}: 熵={entropy:.4f}")

    # 参数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"理论值: 4 * {d_model}^2 = {4 * d_model**2:,}")


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""多头注意力核心手工实现"""
import numpy as np

def multihead_attention_numpy(Q, K, V, W_Q, W_K, W_V, W_O, num_heads):
    """手工多头注意力"""
    B, Lq, D = Q.shape
    _, Lk, _ = K.shape
    d_k = D // num_heads

    # 投影并分割
    def project_and_split(W, X):
        proj = X @ W.T  # (B, L, D)
        return proj.reshape(B, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_m = project_and_split(W_Q, Q)
    K_m = project_and_split(W_K, K)
    V_m = project_and_split(W_V, V)

    # 注意力
    scores = (Q_m @ K_m.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # 加权
    context = attn @ V_m
    context = context.transpose(0, 2, 1, 3).reshape(B, Lq, D)
    output = context @ W_O.T
    return output, attn

def test():
    np.random.seed(42)
    B, L, D, H = 2, 8, 64, 4
    X = np.random.randn(B, L, D)
    W = [np.random.randn(D, D) * 0.1 for _ in range(4)]
    out, attn = multihead_attention_numpy(X, X, X, *W, H)
    print(f"多头注意力手工: {X.shape} -> {out.shape}")
    print(f"注意力: {attn.shape}")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
"""多头注意力可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_multihead_attention(attention_weights, tokens, save_path='mha_vis.png'):
    """可视化多头注意力权重"""
    num_heads, Lq, Lk = attention_weights.shape
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(3*num_heads//2, 5))
    axes = axes.flatten()

    for h in range(num_heads):
        im = axes[h].imshow(attention_weights[h], cmap='Blues', aspect='auto')
        axes[h].set_title(f'Head {h+1}')
        axes[h].set_xlabel('Key')
        axes[h].set_ylabel('Query')
        if tokens is not None:
            axes[h].set_xticks(range(len(tokens)))
            axes[h].set_yticks(range(len(tokens)))
            axes[h].set_xticklabels(tokens, fontsize=6)
            axes[h].set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=axes[h], fraction=0.046)

    plt.suptitle('多头注意力: 每个头关注不同的依赖关系', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存到 {save_path}")

def demo():
    np.random.seed(42)
    tokens = ['我', '爱', '机器', '学习', '和', '深度', '学习']
    L = len(tokens)
    H = 8
    # 模拟不同类型的关系
    attn = np.random.rand(H, L, L)
    for h in range(H):
        attn[h] = attn[h] / attn[h].sum(axis=1, keepdims=True)
    visualize_multihead_attention(attn, tokens)

if __name__ == "__main__":
    demo()
```

## 10. 模型评估

```python
"""多头注意力分析"""

def analyze_head_behavior(model, x):
    """分析每个注意头的行为"""
    all_attentions = []
    _ = model(x, x, x, return_attention=True)
    return all_attentions

def head_entropy_analysis():
    """分析注意力头的熵（不确定性）"""
    mha = MultiHeadAttention(512, 8)
    x = torch.randn(4, 20, 512)
    _, attn = mha(x, x, x, return_attention=True)
    entropies = []
    for h in range(8):
        h_attn = attn[0, h]
        ent = -(h_attn * torch.log(h_attn + 1e-8)).sum(dim=-1).mean()
        entropies.append(ent.item())
    for h, e in enumerate(entropies):
        print(f"头 {h+1}: 熵={e:.4f}")
    print(f"平均熵: {np.mean(entropies):.4f}")

if __name__ == "__main__":
    import numpy as np
    head_entropy_analysis()
```

## 11. 常见问题与易错点

**Q1: 头数越多越好吗？**
不是。实验表明8~16个头时效果最好，更多的头会引入冗余（有些头学习重复模式）且增加计算量。

**Q2: 为什么每个头的维度要减小？**
保持总计算量不变。$h \times (d_{\text{model}}/h)^2 = d_{\text{model}}^2/h$，比单头的 $d_{\text{model}}^2$ 还小（当h>1时）。

**Q3: 如何判断哪些头是冗余的？**
通过分析注意力权重的熵——熵高的头（注意力分散）可能是冗余的，可以剪枝。

**Q4: Multi-Head和Grouped-Query Attention（GQA）的区别？**
GQA是分组查询注意力，多个Query头共享一组Key/Value头，用于减少KV缓存大小。MHA是每个Query有独立的Key/Value。

## 12. 学习总结

- 多头注意力是Transformer的核心——并行计算多个注意力
- 每个头学习不同类型的依赖关系（语法、语义、位置等）
- $d_k = d_{\text{model}}/h$ 保证参数量和计算量不变
- 头数通常8~16，更多不一定更好
- 可解释性强：可视化注意力权重可以理解模型行为

## 13. 练习题

**基础题：**

1. 多头注意力中每个头的 $d_k$ 和 $d_{\text{model}}$ 有什么关系？
> **答案：** $d_k = d_{\text{model}} / h$，保证总计算量不变。

2. 写出多头注意力公式，解释每个符号的含义。
> **答案：** $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$，$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

**进阶题：**

3. 如果所有注意力头的权重完全相同，会发生什么？
> **答案：** 多头退化成了单头。拼接h个相同的结果再投影，等价于一个线性变换，没有任何多视角的好处。

4. 在多头注意力中应用Dropout，应该在什么位置？
> **答案：** 通常在注意力权重（softmax后）应用dropout，防止过拟合特定位置的关系。也可在输出投影前应用。

## 14. 学习路径

**前置：** 自注意力机制、缩放点积注意力
**平行：** Transformer模型
**进阶：** Flash Attention（高效注意力）、GQA（分组查询注意力）、Multi-Query Attention