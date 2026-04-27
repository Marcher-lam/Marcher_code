# 自注意力机制（Self-Attention）学习文档

> 让序列中的每个位置都能直接关注序列中的所有位置，建立全局依赖关系。

## 1. 算法基础认知

### 一句话定义

自注意力是一种让序列内每个元素都"看"到其他所有元素的注意力机制，解决长距离依赖问题的核心技术。

### 直觉类比

读一段很长的文章时，你可能会回头看前文提到的某个概念来理解当前句子。自注意力让模型也能这样做——每个词都能直接"看到"句子中的其他所有词，并根据上下文决定应该关注哪些词。

### 历史背景

- **2016年**：Long Range Arena论文系统评估自注意力
- **2017年**：Transformer论文提出"缩放点积注意力"
- **2018年**：BERT使用双向自注意力
- **2020年**：Transformers在CV领域兴起

### 算法定位

自注意力是注意力机制的特例，Query、Key、Value都来自同一输入。属于**深度学习基础模块**。

### 前置知识

- 注意力机制基础
- 矩阵运算
- PyTorch/TensorFlow

---

## 2. 核心原理

### 核心思想

自注意力的核心是**全局建模**——每个位置的表示都是对所有位置信息的加权聚合，不依赖序列距离。

### 工作流程

1. 输入序列 $X = [x_1, x_2, ..., x_n]$
2. 三个线性变换：$Q=XW_Q$, $K=XW_K$, $V=XW_V$
3. 计算两两之间的相似度：$e_{ij} = q_i \cdot k_j$
4. Softmax归一化：$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$
5. 加权聚合：$y_i = \sum_j \alpha_{ij} v_j$

### 关键特点

- **并行计算**：所有位置同时计算，不依赖序列顺序
- **全局感受野**：每个位置都能关注所有其他位置
- **动态权重**：权重随输入内容变化

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $X$ | 输入序列 | $(n, d_{model})$ |
| $W_Q, W_K, W_V$ | 线性变换矩阵 | $(d_{model}, d_k)$ |
| $Q, K, V$ | 查询、键、值矩阵 | $(n, d_k)$ |

### 问题形式化

给定输入序列 $X \in \mathbb{R}^{n \times d_{model}}$，输出序列：
$$\text{Self-Attention}(X) = \text{softmax}\left(\frac{XW_QW_K^TX^T}{\sqrt{d_k}}\right)XV$$

### 详细推导

**Step 1: 生成Q、K、V**
$$Q = XW_Q \in \mathbb{R}^{n \times d_k}$$
$$K = XW_K \in \mathbb{R}^{n \times d_k}$$
$$V = XW_V \in \mathbb{R}^{n \times d_v}$$

**Step 2: 注意力分数矩阵**
$$E = QK^T \in \mathbb{R}^{n \times n}, \quad E_{ij} = q_i \cdot k_j$$

**Step 3: 缩放**
$$E_{scaled} = \frac{E}{\sqrt{d_k}}$$

**Step 4: 归一化**
$$A = \text{softmax}(E_{scaled}) \in \mathbb{R}^{n \times n}$$

**Step 5: 输出**
$$O = AV \in \mathbb{R}^{n \times d_v}$$

---

## 4. 训练过程讲解

### 参数初始化

- $W_Q, W_K, W_V$：Xavier初始化
- 偏置：初始化为0

### 超参数表

| 参数 | 作用 | 常见取值 |
|------|------|----------|
| $d_k$ | Q、K维度 | 64 |
| $d_v$ | V维度 | 64 |
| 缩放因子 | 防止梯度消失 | $\sqrt{d_k}$ |

---

## 5. 应用场景

1. **Transformer核心组件**：编码器和解码器的核心
2. **BERT**：双向自注意力建模
3. **NLP任务**：机器翻译、文本分类、问答
4. **CV领域**：Vision Transformer (ViT)
5. **多模态**：CLIP、DALL-E

---

## 6. 优缺点分析

### 优点

1. **长距离依赖**：直接建模任意距离的关系
2. **并行计算**：效率高于RNN
3. **可解释性**：注意力权重可可视化

### 缺点

1. **$O(n^2)$复杂度**：序列长度增加时计算量爆炸
2. **位置信息丢失**：需要额外位置编码

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """自注意力机制实现"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 统一的线性变换，同时生成Q、K、V
        self.W = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 一次性生成Q、K、V
        qkv = self.W(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        
        # 调整形状以应用多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权聚合
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context), attn_weights

# 测试
if __name__ == "__main__":
    x = torch.randn(32, 100, 512)
    attn = SelfAttention(512, 8)
    out, weights = attn(x)
    print(f"输出形状: {out.shape}")
    print(f"注意力形状: {weights.shape}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class NumPySelfAttention:
    """纯NumPy实现的自注意力"""
    
    def __init__(self, d_model, d_k=None):
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        
        # Xavier初始化
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))
        self.W = np.random.randn(self.d_model, self.d_k * 3) * scale
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        # 一次性计算Q、K、V
        qkv = np.dot(X, self.W)  # (batch, seq, 3*d_k)
        Q, K, V = np.split(qkv, 3, axis=-1)
        
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Softmax归一化
        attn_weights = self.softmax(scores)
        
        # 加权求和
        output = np.matmul(attn_weights, V)
        
        return output, attn_weights

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 10, 64)
    attn = NumPySelfAttention(64)
    out, weights = attn.forward(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_self_attention(sentence, save_path="self_attention.png"):
    """可视化自注意力矩阵"""
    words = sentence.split()
    n = len(words)
    
    # 模拟一个对角线加权的注意力矩阵
    np.random.seed(42)
    attn = np.random.rand(n, n) * 0.3
    for i in range(n):
        for j in range(n):
            attn[i, j] += 0.4 / (abs(i - j) + 1)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xticks(range(n), words, rotation=45, ha='right')
    plt.yticks(range(n), words)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_self_attention("The cat sat on the mat because it was tired")
```

---

## 10. 模型评估

```python
def evaluate_attention_pattern(attn_weights):
    """评估自注意力的特性"""
    # 注意力集中度
    max_attn = np.max(attn_weights, axis=-1)
    concentration = np.mean(max_attn)
    
    # 注意力均匀度
    uniform = 1.0 / attn_weights.shape[-1]
    deviation = np.mean(np.abs(attn_weights - uniform))
    
    return {"concentration": concentration, "uniform_deviation": deviation}
```

---

## 11. 常见问题与易错点

1. **Q、K、V维度不匹配**：确保$d_k$和$d_v$设置正确
2. **缩放因子遗漏**：不除以$\sqrt{d_k}$会导致训练不稳定
3. **Mask使用错误**：padding位置应mask为负无穷

---

## 12. 学习总结

自注意力是现代深度学习的基石，通过$O(n^2)$的计算代价换取全局建模能力。后续学习的Transformer、BERT都是基于此机制。

---

## 13. 练习题

1. **基础**：解释为什么自注意力需要位置编码？
2. **进阶**：如果序列长度从100增加到10000，计算量增加多少倍？

---

## 14. 学习路径建议

- 前置：注意力机制基础
- 平行：多头注意力
- 进阶：Transformer → BERT → ViT