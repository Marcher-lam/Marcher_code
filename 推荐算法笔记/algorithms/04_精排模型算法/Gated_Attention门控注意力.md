# Gated Attention（NeurIPS 2025 最佳论文）

## 1. 算法基础认知

Gated Attention 是一种对标准缩放点积注意力（SDPA）的极简改进，在注意力输出后施加**头部专属、逐元素、Sigmoid 乘性门控**，仅增加约 1% 参数即可显著提升大语言模型的性能和长上下文外推能力。该工作获得 **NeurIPS 2025 最佳论文奖**，并已应用于 Qwen3-Next 系列。

## 2. 详细原理

### 2.1 标准注意力的线性瓶颈

标准 SDPA 的输出为：

$$Y = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V \cdot W_o$$

$W_o$ 是一个线性投影，因此整个注意力模块是 **Value 矩阵的线性变换**，表达能力受限。即使多头机制增加了子空间数量，每个头内仍然是线性的。

### 2.2 Gated Attention 核心公式

在 SDPA 输出后施加门控：

$$o_i = y_i \odot \sigma(W_g x_i)$$

其中：
- $y_i$ 是 SDPA 的输出（注意力层的输出）
- $x_i$ 是注意力层**之前**的隐藏状态（残差连接的输入）
- $W_g \in \mathbb{R}^{d \times d}$ 是可学习的门控投影矩阵
- $\sigma$ 是 Sigmoid 函数
- $\odot$ 是逐元素乘法

### 2.3 设计要点

1. **头部专属**：$W_g$ 对每个注意力头独立，不同头学不同的门控模式
2. **逐元素**：每个维度独立门控，而非整个头一个标量
3. **Sigmoid 乘性**：用 Sigmoid（非 Softmax）产生 $[0,1]$ 值域的门控信号
4. **输入来自层前**：门控信号 $x_i$ 取自注意力输入（而非输出），提供"原始信号"视角

### 2.4 为什么有效

**引入非线性**：

标准注意力是 $V$ 的线性变换。门控引入逐元素乘法，打破了线性瓶颈：

$$o_i = (softmax(\cdot) V W_o) \odot \sigma(W_g x_i)$$

这是两个非线性函数的乘积，表达能力大幅提升。

**查询依赖稀疏性**：

实验发现门控分数高度稀疏，均值仅 0.116（接近 90% 的维度被关闭）。这意味着模型学会主动过滤与当前查询无关的历史信息。

**消除 Attention Sink**：

Attention Sink 指首 token 吸引过多注意力（传统模型高达 46.7%）。门控机制使模型能够"遗忘"无关信息，无需依赖首 token 作为"垃圾桶"。首 token 注意力占比从 46.7% 降至 4.8%。

## 3. 数学分析

### 3.1 梯度特性

门控的梯度为：

$$\frac{\partial o_i}{\partial W_g} = y_i \odot \sigma'(W_g x_i) \odot x_i$$

由于 $\sigma'(z) = \sigma(z)(1-\sigma(z))$，当门控值接近 0 或 1 时梯度趋零，形成自然稀疏梯度。

### 3.2 信息论视角

门控实现了对注意力输出的**选择性过滤**：

$$I(O; X) \leq I(Y; X)$$

门控通过丢弃冗余维度，减少输出与输入的互信息，起到信息瓶颈的正规化作用。

## 4. 模型集成方式

Gated Attention 可无缝集成到任何使用 SDPA 的 Transformer 模型中：

```
标准 Transformer 层:
  x → SDPA → W_o → + x → FFN → + → 输出

Gated Attention 层:
  x → SDPA → W_o → ⊙ σ(W_g·x) → + x → FFN → + → 输出
                       ↑
                   门控（新增）
```

## 5. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import math

class GatedAttention(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.W_gate = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        B, S, D = x.shape
        residual = x
        
        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = torch.matmul(attn, V)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        y = self.W_o(y)
        
        gate_input = self.norm(residual)
        gate = torch.sigmoid(self.W_gate(gate_input))
        y = y * gate
        
        y = self.resid_dropout(y)
        return y + residual


class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, ffn_dim=4096, dropout=0.1):
        super().__init__()
        self.attn = GatedAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.attn(x, mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

## 6. 应用场景

| 场景 | 效果 |
|------|------|
| 大语言模型 | PPL 降低 0.2+，仅增 1% 参数 |
| 长上下文外推 | 128K 外推时性能衰减仅 6.89%（基线 41.56%） |
| 消除 Attention Sink | 首注意力从 46.7% 降至 4.8% |
| 推荐系统序列建模 | 更好地过滤无关历史行为 |

## 7. 优缺点分析

**优点**：
- 极简改动，兼容所有 Transformer 架构
- 参数增量极小（~1%）
- 显著提升长上下文外推能力
- 天然稀疏性降低计算和显存
- 消除 Attention Sink 无需特殊训练策略

**缺点**：
- 逐元素乘法引入额外计算
- 短序列（<4K）提升有限
- 门控稀疏性可能导致部分注意力头退化
- 与 FlashAttention 需要适配

## 8. 与相关方法对比

| 方法 | 增加参数 | 非线性 | 稀疏性 | Attention Sink |
|------|---------|--------|--------|---------------|
| 标准 SDPA | 0 | 无 | 无 | 严重 |
| SwiGLU | ~50% | 有 | 无 | 中等 |
| Gated Attention | ~1% | 有 | 强（90%关闭） | 消除 |
| StreamingLLM | 0 | 无 | 无 | 硬编码保留 |

## 9. 常见问题与易错点

1. **门控输入选错**：$x_i$ 必须取自注意力层**之前**的输入，不是注意力输出
2. **用 Softmax 替代 Sigmoid**：Sigmoid 是逐元素独立，Softmax 会引入竞争关系
3. **初始化问题**：$W_g$ 应初始化为较大正值（如 2.0），使初始门控接近 1（不干扰训练初期）
4. **与 RoPE 冲突**：门控作用在投影后，与 RoPE 不冲突，但需注意维度对齐

## 10. 学习总结

Gated Attention 以极简设计（一行公式）解决了 Transformer 的三个核心问题：线性瓶颈、冗余信息累积和 Attention Sink。其成功启示我们：好的改进不一定需要复杂的架构变革，关键在于找到正确的介入点和形式。门控信号取自层前输入而非层后输出这一设计选择尤其精妙。

## 11. 学习路径建议

- **前置知识**：Transformer 架构、缩放点积注意力、Attention Sink 现象
- **进阶方向**：SwiGLU、FlashAttention、长上下文外推技术
- **推荐论文**：Gated Attention (NeurIPS 2025)、Attention Sink (StreamingLLM 2023)
