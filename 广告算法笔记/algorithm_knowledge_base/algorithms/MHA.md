# MHA 学习文档

## 1. 算法基础认知

多头注意力（Multi-Head Attention, MHA）是 Transformer 架构的核心组件，由 Vaswani 等人在 2017 年的 "Attention Is All You Need" 论文中提出。MHA 通过并行运行多个独立的注意力头（Attention Head），让模型能够同时关注不同位置的不同表征子空间中的信息。

如果说单头注意力是"用一只眼睛看世界"，那么多头注意力就是"用多只眼睛从不同角度看世界"。每个头学习不同的注意力模式——有的关注局部语法关系，有的捕获长距离语义依赖，最后将这些不同视角的信息融合为统一表示。

## 2. 核心原理

**缩放点积注意力（Scaled Dot-Product Attention）**：

MHA 的基础运算单元。给定查询（Query）、键（Key）、值（Value）三组矩阵，计算注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $d_k$ 是键向量的维度，除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。

**多头并行**：

将 $Q, K, V$ 分别通过 $h$ 组不同的线性投影，得到 $h$ 组 $(Q_i, K_i, V_i)$，独立计算注意力后拼接结果并做最终线性变换：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

## 3. 数学公式与推导

**输入**：$Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$

**单头注意力详细推导**：

1. 计算注意力得分矩阵：$S = QK^\top \in \mathbb{R}^{n \times m}$
2. 缩放：$S_{ij}$ 除以 $\sqrt{d_k}$，即 $S'_{ij} = S_{ij} / \sqrt{d_k}$
3. Softmax 归一化（对每行）：$\alpha_{ij} = \frac{\exp(S'_{ij})}{\sum_{l=1}^{m} \exp(S'_{il})}$
4. 加权求和：$Z = \alpha V \in \mathbb{R}^{n \times d_v}$

**为什么缩放**：当 $d_k$ 较大时，$QK^\top$ 的元素方差为 $d_k$（假设元素独立且方差为 1）。大的点积值使 softmax 趋向 one-hot，梯度接近于零。除以 $\sqrt{d_k}$ 使方差回到 1。

**多头参数量**：$h$ 个头，每个头的投影矩阵 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$，输出投影 $W^O \in \mathbb{R}^{hd_v \times d_{model}}$。通常 $d_k = d_v = d_{model} / h$。

## 4. 训练过程讲解

1. **线性投影**：输入序列经 $h$ 组独立的线性变换，分别投影到 $h$ 个 Q、K、V 子空间。
2. **并行注意力计算**：$h$ 个头独立计算缩放点积注意力。得益于矩阵运算，这可以高效并行化。
3. **拼接与融合**：将 $h$ 个头的输出沿特征维度拼接，通过输出投影矩阵 $W^O$ 融合为最终输出。
4. **残差连接与层归一化**：MHA 通常配合残差连接（$x + \text{MHA}(x)$）和 LayerNorm 使用。
5. **训练**：通过反向传播同时优化所有头的投影矩阵。不同头会自发学习不同的注意力模式。

## 5. 应用场景

- Transformer 的自注意力层（Q=K=V=输入序列）
- Transformer 的交叉注意力层（Q=解码器状态，K=V=编码器输出）
- BERT（仅编码器，自注意力）
- GPT（仅解码器，因果注意力/掩码注意力）
- 广告系统中的用户行为序列建模、多模态特征融合
- ViT（Vision Transformer，图像块序列的自注意力）

## 6. 优缺点分析

**优点：**
- 多头并行捕获多种注意力模式，表达能力强
- 计算可高度并行化，训练效率远超 RNN
- 灵活：通过掩码实现因果注意力、双向注意力等
- 可解释性：注意力权重可视化展示模型关注模式

**缺点：**
- 自注意力计算复杂度 $O(n^2)$，序列长度受限
- 位置信息需额外编码（位置编码）
- 内存消耗大（需存储 $n \times n$ 的注意力矩阵）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)

d_model, num_heads, seq_len, batch = 512, 8, 30, 16
mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch, seq_len, d_model)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
output = mha(x, x, x, mask)
print(f"Input: {x.shape}, Output: {output.shape}")

mha_lib = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
lib_out, _ = mha_lib(x, x, x)
print(f"PyTorch nn.MultiheadAttention output: {lib_out.shape}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return attn_weights @ V, attn_weights

def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads):
    d_model = X.shape[-1]
    d_k = d_model // num_heads
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Q_heads = Q.reshape(Q.shape[0], num_heads, d_k)
    K_heads = K.reshape(K.shape[0], num_heads, d_k)
    V_heads = V.reshape(V.shape[0], num_heads, d_k)
    head_outputs = []
    for i in range(num_heads):
        out, _ = scaled_dot_product_attention(Q_heads[:, i], K_heads[:, i], V_heads[:, i])
        head_outputs.append(out)
    concat = np.concatenate(head_outputs, axis=-1)
    return concat @ W_O

d_model, num_heads, seq_len = 64, 4, 10
X = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_model) * 0.02
W_K = np.random.randn(d_model, d_model) * 0.02
W_V = np.random.randn(d_model, d_model) * 0.02
W_O = np.random.randn(d_model, d_model) * 0.02
output = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads)
print(f"MHA output shape: {output.shape}")
```

## 9. 可视化与结果理解

- **注意力热力图**：将每个头的注意力权重矩阵绘制为热力图，不同头呈现不同的关注模式
- **多头多样性**：对比不同头的注意力分布，观察是否有头关注局部、有头关注全局
- **头的重要性分析**：通过剪枝实验（移除某些头），分析每个头对最终性能的贡献
- **注意力随训练变化**：观察训练初期和后期注意力模式的变化，通常从均匀分布逐渐变为稀疏聚焦

## 10. 模型评估

- **下游任务性能**：MHA 本身是组件而非完整模型，通过所在模型（BERT、GPT 等）的下游任务表现间接评估
- **注意力质量**：注意力权重是否与语言学知识（如指代关系、句法依赖）一致
- **头冗余度**：分析不同头之间的相似度，高冗余意味着可以剪枝减少计算量
- **计算效率**：$O(n^2 d)$ 的复杂度，在长序列上评估内存和速度瓶颈

## 11. 常见问题与易错点

- **维度不整除**：$d_{model}$ 必须能被 $num\_heads$ 整除，否则每个头的维度不是整数
- **Mask 形状**：掩码张量的形状必须与注意力得分矩阵匹配，广播维度容易出错
- **Batch 维度顺序**：PyTorch 的 `nn.MultiheadAttention` 默认输入格式为 `(seq_len, batch, d_model)`，设置 `batch_first=True` 可改为 `(batch, seq_len, d_model)`
- **因果掩码方向**：自回归解码时，掩码应遮住未来位置（上三角），方向搞反会导致信息泄露
- **Dropout 位置**：注意力 dropout 应在 softmax 之后、乘以 V 之前

## 12. 学习总结

多头注意力是现代深度学习最重要的组件之一。它通过并行多组 Q/K/V 投影和注意力计算，让模型从多个表征子空间同时捕获依赖关系。MHA 是 Transformer 的核心，也是 BERT、GPT、ViT 等模型的基础。理解 Q/K/V 的角色、缩放点积的原理以及多头的意义，是掌握现代深度学习架构的关键。

## 13. 练习题与思考题（含答案）

**Q1：为什么除以 $\sqrt{d_k}$ 而不是 $d_k$？**

A1：假设 $Q$ 和 $K$ 的元素独立且方差为 1，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$（独立随机变量乘积的方差之和）。除以 $\sqrt{d_k}$ 使方差回到 1，保持 softmax 输入的数值稳定。除以 $d_k$ 会过度缩小。

**Q2：8 个头的 MHA 与单头注意力，参数量谁更多？**

A2：参数量相同。8 头时每个头的投影矩阵维度为 $d_{model}/8$，总参数：$8 \times 3 \times d_{model} \times (d_{model}/8) + d_{model} \times d_{model} = 4 d_{model}^2$。单头：$3 \times d_{model} \times d_{model} + d_{model} \times d_{model} = 4 d_{model}^2$。

**Q3：Q、K、V 分别扮演什么角色？一个直觉类比是什么？**

A3：直觉类比图书馆找书：Q 是"我想要什么"（查询意图），K 是"这本书讲什么"（索引标签），V 是"书的内容"（实际信息）。通过 Q 和 K 的匹配度决定从每本书中提取多少 V 的内容。

## 14. 学习路径建议

1. 先理解单头缩放点积注意力的数学
2. 手动计算一个小型注意力示例（如 3×3 矩阵）
3. 实现多头注意力并验证与 PyTorch `nn.MultiheadAttention` 的一致性
4. 在简单任务（如文本分类）上训练 Transformer 块
5. 可视化不同头的注意力模式
6. 进阶：学习高效注意力变体（Linear Attention、Flash Attention、Sparse Attention）
