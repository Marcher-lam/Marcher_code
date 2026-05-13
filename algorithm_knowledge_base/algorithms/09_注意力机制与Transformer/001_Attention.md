# Attention 注意力机制

> 注意力机制使模型能够根据当前上下文动态决定需要"关注"输入的哪些部分，是Transformer成功的核心。

## 1. 算法基础认知

### 1.1 什么是注意力机制

注意力机制（Attention Mechanism）是一种让模型学习"应该关注什么"的技术。在处理序列时，不是对所有信息一视同仁，而是根据当前任务动态地分配不同的权重。

### 1.2 直觉类比

想象你在一间嘈杂的咖啡馆里和朋友聊天。你的耳朵并不是平均接收所有声音，而是自动"关注"朋友的声音，过滤掉其他噪音。注意力机制做的事情类似——让神经网络学会关注重要的部分。

### 1.3 历史背景

- **2014年**：Bahdanau等人首次在神经机器翻译中提出"Additive Attention"
- **2015年**：Luong等人提出"Multiplicative Attention"
- **2017年**：Vaswani等人提出"Scaled Dot-Product Attention"，成为Transformer基础

### 1.4 算法定位

- **任务类型**：可用于各种序列到序列任务
- **所属类别**：深度学习/特征聚焦机制
- **前置知识**：神经网络、Softmax、矩阵运算

## 2. 核心原理

### 2.1 核心思想

注意力机制的核心是**软加权**——对不同位置的信息进行加权求和，权重反映了当前位置对其他位置的"关注程度"。

### 2.2 工作流程

1. **计算Query**：当前需要处理的内容表示为"查询"
2. **计算Key**：每个位置提供"键"作为匹配的标识
3. **计算Score**：Query与所有Key的相似度
4. **Softmax归一化**：将分数转换为概率分布（权重）
5. **Weighted Sum**：用权重对Value进行加权求和

### 2.3 Query-Key-Value解释

```
Query（查询）：我在找什么信息？
Key（键）：每个位置有什么特征？
Value（值）：每个位置的具体内容？

匹配过程：
Query → 与所有Key比较 → 得到关注权重 → 加权Value
```

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $Q$ | Query矩阵 (seq_len_q, d_k) |
| $K$ | Key矩阵 (seq_len_k, d_k) |
| $V$ | Value矩阵 (seq_len_v, d_v) |
| $d_k$ | Key/Query的维度 |
| $d_v$ | Value的维度 |

### 3.2 Scaled Dot-Product Attention

$$ Score = \frac{Q \cdot K^T}{\sqrt{d_k}} $$

$$ Attention(Q, K, V) = softmax(Score) \cdot V $$

### 3.3 Additive Attention（早期版本）

$$ Score = v^T \cdot \tanh(W_q \cdot Q + W_k \cdot K) $$

其中$W_q, W_k$是可学习的参数，$v$是输出向量。

## 4. 训练过程讲解

### 4.1 参数形式

注意力机制的可学习参数：
- Query投影矩阵 $W^Q$：将输入映射到Query空间
- Key投影矩阵 $W^K$：将输入映射到Key空间  
- Value投影矩阵 $W^V$：将输入映射到Value空间
- 输出投影矩阵 $W^O$：合并多头注意力结果

### 4.2 训练特点

- 无需额外的监督信号：注意力是有监督的
- 与主任务联合训练：端到端训练

## 5. 应用场景

### 5.1 典型应用

1. **神经机器翻译**：最初的应用
2. **文本摘要**：决定关注原文的哪些部分
3. **问答系统**：找到问题相关的答案片段
4. **图像描述**：生成描述时关注图像区域

### 5.2 适用场景

- 需要从大量信息中筛选关键内容
- 需要建模长距离依赖关系
- 需要可解释的决策过程

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 长距离依赖 | 直接建模任意距离的关系 |
| 可解释性 | 权重可可视化 |
| 并行化 | 计算可并行 |
| 通用性 | 适用于各种模态 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| O(n²)复杂度 | 序列长度平方 |
| 内存占用 | 需要存储注意力矩阵 |
| 有时不work | 对超参数敏感 |

## 7. 调库实现

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# PyTorch自带的MultiheadAttention
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout
        )
    
    def forward(self, query, key, value, mask=None):
        output, attn_weights = self.attention(
            query, key, value, 
            attn_mask=mask,
            key_padding_mask=mask
        )
        return output, attn_weights

# 测试
d_model = 512
num_heads = 8
batch = 2
seq_len = 10

model = AttentionLayer(d_model, num_heads)
x = torch.randn(seq_len, batch, d_model)

output, weights = model(x, x, x)
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
```

## 8. 手工代码实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        
        # 计算相似度分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 可选mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 归一化为权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class SelfAttention(nn.Module):
    """自注意力实现"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 三个投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 投影到Q, K, V
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 计算注意力
        output, weights = self.attention(q, k, v, mask)
        
        return output, weights


# 测试
d_model = 512
batch = 2
seq_len = 10

x = torch.randn(batch, seq_len, d_model)
attn = SelfAttention(d_model)

output, weights = attn(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"权重: {weights.shape}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(weights, words, save_path=None):
    """可视化注意力权重"""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights.numpy(), 
               xticklabels=words,
               yticklabels=words,
               cmap='YlOrRd')
    plt.title('注意力权重分布')
    plt.xlabel('Key (被关注)')
    plt.ylabel('Query (关注者)')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 示例
words = ['The', 'movie', 'was', 'not', 'bad']
weights = torch.rand(5, 5)
weights = F.softmax(weights, dim=-1)

visualize_attention(weights, words)
```

## 10. 模型评估

注意力机制通常与其他组件一起评估，可通过：

1. **注意力分布分析**：不同层/头的注意力模式
2. **可视化检查**：���否��注正确的位置
3. **消融实验**：移除注意力后的性能下降

## 11. 常见问题与易错点

| 问题 | 原因 | 解决 |
|------|------|------|
| 梯度消失 | Softmax饱和 | 缩放因子 |
| OOM | 序列太长 | 减少batch或截断 |
| 不收敛 | 学习率太大 | 降低学习率 |

## 12. 学习总结

注意力机制的核心是**软寻址**——通过学习权重来选择性地聚合信息。它让深度学习模型学会了"关注什么"，是Transformer革命的关键。

## 13. 练习题

**基础题**：解释为什么需要缩放因子？

**答案**：当$d_k$较大时，点积的方差会很大，导致Softmax函数进入饱和区，梯度变得非常小，影响训练。

## 14. 学习路径建议

- **前置**：神经网络基础
- **进阶**：Multi-Head Attention → Transformer
- **资源**：原始论文"Attention Is All You Need"