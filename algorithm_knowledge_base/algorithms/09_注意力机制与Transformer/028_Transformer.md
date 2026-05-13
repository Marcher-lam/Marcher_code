# Transformer 学习文档

> 完全基于注意力机制构建的编码器-解码器架构，奠定了大语言模型时代的基础。

## 1. 算法基础认知

### 一句话定义

Transformer是一种完全基于自注意力机制的网络架构，能够并行处理序列数据并有效捕获长距离依赖关系。

### 直觉类比

想象你阅读一本小说时，能够同时记住前面章节的所有关键信息来理解当前情节。Transformer正是让计算机具备这种"全局记忆"能力的架构。

### 历史背景

- **2017年**：Google Brain团队在论文《Attention Is All You Need》中提出
- **2018年**：BERT、GPT相继发布
- **2020年**：GPT-3展示大规模Transformer的威力
- **2022年**：ChatGPT让Transformer家喻户晓

### 算法定位

Transformer是**深度学习基础架构**，既可用于NLP（机器翻译、文本生成），也可用于CV（图像分类、目标检测）。

### 前置知识

- 神经网络基础
- 自注意力机制
- 编码器-解码器架构

---

## 2. 核心原理

### 核心思想

Transformer完全摒弃RNN和CNN，仅使用注意力机制来建模序列中的依赖关系。核心组件包括：

1. **编码器**：处理输入序列，提取特征
2. **解码器**：基于编码器输出生成目标序列
3. **位置编码**：注入序列位置信息

### 工作流程

1. 输入序列经过词嵌入和位置编码
2. 编码器：N个相同的层，每层包含自注意力+残差+LayerNorm+FFN
3. 解码器：N个相同的层，每层包含自注意力+残差+编码器-解码器注意力+FFN
4. 线性层+Softmax输出预测

### 架构图

```
输入 → 嵌入 + 位置编码 → 编码器 × N → 编码器输出
                                      ↓
                            解码器 × N → 输出概率
                                      ↓
输入(shifted) → 嵌入 + 位置编码 → (Masked自注意力)
```

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $d_{model}$ | 模型维度，通常512 |
| $d_{ff}$ | FFN隐藏层维度，通常2048 |
| $h$ | 注意力头数，通常8 |
| $N$ | 编码器/解码器层数，通常6 |

### 编码器层

**自注意力**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**前馈网络**：
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**带残差的LayerNorm**：
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 位置编码

$$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

---

## 4. 训练过程讲解

### 超参数表

| 参数 | 作用 | 常用值 |
|------|------|--------|
| $d_{model}$ | 模型宽度 | 512, 768, 1024 |
| $h$ | 注意力头数 | 8, 12, 16 |
| $N$ | 层数 | 6, 12, 24 |
| $d_{ff}$ | FFN宽度 | 2048, 3072, 4096 |
| dropout | 正则化 | 0.1 |
| 学习率 | 优化 | 3e-4 (warmup) |

### 训练技巧

1. **学习率预热**：前4000步线性增加，后按平方根衰减
2. **标签平滑**：0.1防止过拟合
3. **梯度裁剪**：防止梯度爆炸
4. **Adam优化器**：$\beta_1=0.9, \beta_2=0.98$

---

## 5. 应用场景

1. **机器翻译**：原始应用场景
2. **文本生成**：GPT系列
3. **文本理解**：BERT系列
4. **图像处理**：ViT、DETR
5. **语音识别**：Conformer
6. **多模态**：CLIP、DALL-E

---

## 6. 优缺点分析

### 优点

1. **并行计算**：比RNN高效
2. **长距离依赖**：直接建模全局关系
3. **通用性强**：NLP和CV都适用
4. **可扩展性**：越大效果越好

### 缺点

1. **$O(n^2)$复杂度**：长序列计算量大
2. **位置信息需要编码**：不如RNN直接
3. **内存占用大**：需要大量GPU显存

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 自注意力 + 残差 + LayerNorm
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        
        # FFN + 残差 + LayerNorm
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout2(src2))
        return src

class Transformer(nn.Module):
    """完整Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 解码器（简化版，实际需要更复杂实现）
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码
        src = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # 解码
        tgt = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask)
        
        return self.fc_out(tgt)

# 测试
if __name__ == "__main__":
    model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
    src = torch.randint(0, 10000, (32, 100))
    tgt = torch.randint(0, 10000, (32, 50))
    output = model(src, tgt)
    print(f"输出形状: {output.shape}")  # (32, 50, 10000)
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleTransformer:
    """简化版Transformer实现（仅编码器）"""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 嵌入层
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # 位置编码
        self.pos_encoding = self._positional_encoding(1000, d_model)
        
        # 简化的注意力权重
        self.W_Q = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_layers)]
        self.W_K = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_layers)]
        self.W_V = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_layers)]
        
    def _positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / np.power(10000, i / d_model))
                pe[pos, i+1] = np.cos(pos / np.power(10000, i / d_model))
        return pe
    
    def forward(self, x):
        # 嵌入 + 位置编码
        x = np.array([[self.embedding[t] for t in seq] for seq in x])
        x = x + self.pos_encoding[:x.shape[1]]
        
        # 简化的自注意力层
        for i in range(len(self.W_Q)):
            Q = np.dot(x, self.W_Q[i])
            K = np.dot(x, self.W_K[i])
            V = np.dot(x, self.W_V[i])
            
            scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
            attn = self._softmax(scores)
            x = np.matmul(attn, V)
        
        return x
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    model = SimpleTransformer(1000)
    x = np.random.randint(0, 1000, (2, 10))
    out = model.forward(x)
    print(f"输出形状: {out.shape}")  # (2, 10, 128)
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt

def visualize_transformer_architecture():
    """可视化Transformer架构"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制各组件
    components = [
        (5, 9, "输入序列", "lightblue"),
        (5, 8, "嵌入层+位置编码", "lightgreen"),
        (2, 7, "编码器×N", "lightyellow"),
        (5, 7, "自注意力", "lightcoral"),
        (5, 5, "编码器输出", "lightblue"),
        (8, 4, "解码器×N", "lightyellow"),
        (8, 2, "输出概率", "lightgray"),
    ]
    
    for x, y, text, color in components:
        rect = plt.Rectangle((x-1.5, y-0.4), 3, 0.8, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # 箭头
    ax.annotate('', xy=(5, 8.4), xytext=(5, 9.2), 
               arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.title("Transformer 架构图", fontsize=14)
    plt.tight_layout()
    plt.savefig("transformer_architecture.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_transformer_architecture()
```

---

## 10. 模型评估

```python
def count_parameters(model):
    """计算Transformer参数量"""
    return sum(p.numel() for p in model.parameters())

def calculate_flops(sequence_length, d_model, num_layers, num_heads):
    """估算Transformer FLOPs"""
    # 注意力: O(n^2 * d)
    attention_flops = sequence_length ** 2 * d_model * num_layers
    # FFN: O(n * d * d_ff)
    ffn_flops = sequence_length * d_model * 2048 * num_layers
    return attention_flops + ffn_flops
```

---

## 11. 常见问题

1. **序列太长内存不足**：使用稀疏注意力或FlashAttention
2. **训练不稳定**：检查学习率和梯度裁剪
3. **推理慢**：使用KV Cache加速

---

## 12. 学习总结

Transformer是深度学习里程碑式的架构，其核心思想是"注意力即全部"。从2017年至今，催生了BERT、GPT等重要模型，彻底改变了NLP领域。

---

## 13. 练习题

1. **基础**：Transformer相比RNN的核心优势是什么？
2. **进阶**：为什么需要位置编码而不是直接使用位置索引？
3. **开放**：Transformer的复杂度是$O(n^2)$，如何优化长序列处理？

---

## 14. 学习路径

- 前置：自注意力机制
- 平行：位置编码、多头注意力
- 进阶：BERT → GPT → T5 → ChatGPT