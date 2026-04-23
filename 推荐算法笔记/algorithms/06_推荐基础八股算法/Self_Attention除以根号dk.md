# Self Attention 计算公式里为什么要除以 √dk

## 1. 核心原因

Self-Attention 的计算公式为：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

除以 $\sqrt{d_k}$ 的根本原因是：**防止点积结果过大，导致 Softmax 进入饱和区，引发梯度消失，从而阻碍训练收敛。** 同时，缩放操作使注意力分数的方差保持稳定，不受维度 $d_k$ 变化的影响。

## 2. 数学推导详解

### 2.1 问题建模

设查询向量 $q \in \mathbb{R}^{d_k}$，键向量 $k \in \mathbb{R}^{d_k}$，点积为：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

假设 $q_i$ 和 $k_i$ 相互独立，均值为 0，方差为 1：

$$E[q_i] = 0, \quad Var(q_i) = 1, \quad E[k_i] = 0, \quad Var(k_i) = 1$$

### 2.2 点积的期望

$$E[q \cdot k] = \sum_{i=1}^{d_k} E[q_i k_i] = \sum_{i=1}^{d_k} E[q_i] \cdot E[k_i] = 0$$

点积的期望为 0，说明没有偏移。

### 2.3 点积的方差

由于 $q_i$ 与 $k_i$ 独立：

$$Var(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = E[q_i^2] \cdot E[k_i^2] = Var(q_i) \cdot Var(k_i) = 1$$

因此点积的方差为：

$$Var(q \cdot k) = \sum_{i=1}^{d_k} Var(q_i k_i) = d_k$$

**关键结论：点积的方差随维度 $d_k$ 线性增长。** 当 $d_k = 64$ 时方差为 64，$d_k = 512$ 时方差为 512。

### 2.4 缩放后方差归一化

除以 $\sqrt{d_k}$ 后：

$$Var\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} Var(q \cdot k) = \frac{d_k}{d_k} = 1$$

缩放后，无论 $d_k$ 多大，注意力分数的方差恒为 1。

### 2.5 为什么方差稳定很重要

当方差为 $d_k$（比如 512）时，点积分量的标准差为 $\sqrt{512} \approx 22.6$。这意味着大部分值落在 $[-22.6, 22.6]$ 之间，Softmax 输入值极大或极小，几乎全部概率质量集中在一个位置。

## 3. Softmax 饱和与梯度消失

### 3.1 Softmax 函数

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

### 3.2 梯度推导

Softmax 关于 $z_i$ 的雅可比矩阵：

$$\frac{\partial \text{Softmax}(z_i)}{\partial z_j} = \text{Softmax}(z_i)(\delta_{ij} - \text{Softmax}(z_j))$$

当输入值很大时，Softmax 输出接近 one-hot 向量（某个位置接近 1，其余接近 0），此时：

- 对最大值位置：$\frac{\partial}{\partial z_{max}} \approx p_{max}(1 - p_{max}) \approx 0$
- 对非最大值位置：$\frac{\partial}{\partial z_j} \approx -p_{max} \cdot p_j \approx 0$

**梯度趋近于零，反向传播信号消失，模型无法有效学习。**

### 3.3 数值稳定性问题

当 $z$ 很大时，$e^z$ 可能溢出（float32 最大值约 $3.4 \times 10^{38}$，对应 $z \approx 88$）。缩放可将值控制在安全范围内。

## 4. 不缩放的数值示例

假设 $d_k = 64$，随机初始化的 $q$ 和 $k$：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
dk = 64
q = torch.randn(dk)
k = torch.randn(dk)

dot_product = torch.dot(q, k)
print(f"点积值: {dot_product:.2f}")
print(f"理论上标准差: {dk**0.5:.2f}")

scores = torch.randn(4, dk) @ torch.randn(dk, 4)
print(f"\n不缩放的 Softmax 输出:")
print(F.softmax(scores, dim=-1))
print(f"不缩放的最大概率: {F.softmax(scores, dim=-1).max().item():.6f}")

scaled_scores = scores / (dk ** 0.5)
print(f"\n缩放后的 Softmax 输出:")
print(F.softmax(scaled_scores, dim=-1))
print(f"缩放后最大概率: {F.softmax(scaled_scores, dim=-1).max().item():.6f}")
```

输出示例：
```
点积值: 7.68
理论上标准差: 8.00

不缩放的 Softmax 输出:
tensor([[0.9999, 0.0000, 0.0001, 0.0000],
        ...])
不缩放的最大概率: 0.999878

缩放后的 Softmax 输出:
tensor([[0.3521, 0.0823, 0.3145, 0.2511],
        ...])
缩放后最大概率: 0.421537
```

**结论：不缩放时 Softmax 近似 one-hot，梯度几乎为零；缩放后分布更加均匀，梯度信号正常。**

## 5. PyTorch 代码实现与对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class UnscaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


def compare_attention():
    batch_size = 2
    seq_len = 10
    d_k = 64
    d_model = 512

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_model)

    scaled_attn = ScaledDotProductAttention(d_k)
    unscaled_attn = UnscaledDotProductAttention()

    _, scaled_weights = scaled_attn(Q, K, V)
    _, unscaled_weights = unscaled_attn(Q, K, V)

    print("缩放注意力权重分布:")
    print(f"  最大值: {scaled_weights.max().item():.4f}")
    print(f"  最小值: {scaled_weights.min().item():.6f}")
    print(f"  熵: {-((scaled_weights + 1e-9).log() * scaled_weights).sum(-1).mean().item():.4f}")

    print("\n不缩放注意力权重分布:")
    print(f"  最大值: {unscaled_weights.max().item():.4f}")
    print(f"  最小值: {unscaled_weights.min().item():.6f}")
    print(f"  熵: {-((unscaled_weights + 1e-9).log() * unscaled_weights).sum(-1).mean().item():.4f}")


def gradient_comparison():
    d_k = 256
    Q = torch.randn(1, 4, d_k, requires_grad=True)
    K = torch.randn(1, 4, d_k)
    V = torch.randn(1, 4, d_k)

    scaled_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    unscaled_scores = Q @ K.transpose(-2, -1)

    scaled_out = F.softmax(scaled_scores, dim=-1).sum()
    unscaled_out = F.softmax(unscaled_scores, dim=-1).sum()

    scaled_out.backward()
    scaled_grad_norm = Q.grad.norm().item()
    Q.grad.zero_()

    unscaled_out.backward()
    unscaled_grad_norm = Q.grad.norm().item()

    print(f"dk = {d_k}")
    print(f"缩放后梯度范数: {scaled_grad_norm:.6f}")
    print(f"不缩放梯度范数: {unscaled_grad_norm:.6f}")
    print(f"梯度比: {scaled_grad_norm / (unscaled_grad_norm + 1e-10):.2f}x")


if __name__ == "__main__":
    print("=== 注意力权重对比 ===")
    compare_attention()
    print("\n=== 梯度对比 ===")
    gradient_comparison()
```

## 6. 与其他缩放方法的对比

| 方法 | 缩放因子 | 优点 | 缺点 |
|------|---------|------|------|
| 除以 $\sqrt{d_k}$ | $1/\sqrt{d_k}$ | 理论优雅，方差归一化 | 假设 Q/K 独立同分布 |
| 除以 $d_k$ | $1/d_k$ | 更激进地压缩 | 过度平滑，注意力区分度下降 |
| 不缩放 | 1 | 简单 | 高维时梯度消失 |
| 温度参数 $\tau$ | $1/\tau$（可学习） | 灵活自适应 | 增加参数量 |
| LayerNorm 后点积 | - | 自适应归一化 | 计算开销略大 |

**实际中 $\sqrt{d_k}$ 缩放是最优的折中方案**，既保证了理论合理性，又不引入额外参数。

## 7. 为什么不除以其他值

### 为什么不除以 $d_k$？

方差变为 $1/d_k$，过小的方差使 Softmax 输出过于均匀，注意力机制丧失选择性，无法区分重要和不重要的位置。

### 为什么不用 BatchNorm？

BatchNorm 依赖 batch 统计量，在变长序列和推理阶段需要额外处理，且引入额外参数。$\sqrt{d_k}$ 缩放无需额外参数和统计量。

### 为什么不用 ReLU 替代 Softmax？

ReLU 不保证注意力权重非负且归一化为 1，破坏了注意力的概率解释。

## 8. 不同维度下的影响

| $d_k$ | 点积标准差 | 不缩放 Softmax 最大值 | 缩放后 Softmax 最大值 |
|--------|-----------|---------------------|---------------------|
| 16 | 4.0 | ~0.85 | ~0.40 |
| 64 | 8.0 | ~0.99 | ~0.40 |
| 256 | 16.0 | ~1.00 | ~0.40 |
| 512 | 22.6 | ~1.00 | ~0.40 |

**维度越高，不缩放的问题越严重，而缩放后分布始终保持稳定。**

## 9. 常见面试问题与解答

### Q1：为什么是 $\sqrt{d_k}$ 而不是 $d_k$？

因为方差与 $d_k$ 成正比，标准差与 $\sqrt{d_k}$ 成正比。除以标准差才能将方差归一化为 1。除以 $d_k$ 会使方差变为 $1/d_k$，过度压缩。

### Q2：如果 Q 和 K 不是独立同分布呢？

实践中 Q 和 K 通过线性变换得到，初始化时近似独立同分布。训练过程中分布会变化，但缩放因子仍然有效，因为网络会自适应调整。

### Q3：Multi-Head Attention 中每个 head 都需要缩放吗？

是的。每个 head 独立计算注意力，维度为 $d_k = d_{model}/h$，每个 head 都需要除以对应的 $\sqrt{d_k}$。

### Q4：缩放因子可以设为可学习参数吗？

可以，但通常没有必要。固定的 $\sqrt{d_k}$ 已经足够好，且不增加参数。某些工作中使用可学习温度参数 $\tau$ 作为改进。

### Q5：在推理阶段缩放还有意义吗？

有意义。推理时 Q 和 K 的分布与训练时一致，如果不缩放会导致注意力分布异常，输出质量下降。

## 10. 学习总结

| 要点 | 内容 |
|------|------|
| 问题根源 | 高维点积方差大，Softmax 饱和 |
| 解决方案 | 除以 $\sqrt{d_k}$，方差归一化为 1 |
| 数学本质 | 将点积从 $\mathcal{N}(0, d_k)$ 映射到 $\mathcal{N}(0, 1)$ |
| 实际效果 | 防止梯度消失，加速收敛 |
| 适用范围 | 所有基于点积的注意力机制 |

## 11. 学习路径建议

1. **前置知识**：线性代数（点积、方差）、Softmax 函数、反向传播
2. **进阶阅读**：Transformer 原论文 "Attention Is All You Need" 第 3.2.1 节
3. **延伸主题**：Multi-Head Attention、相对位置编码、Flash Attention
4. **实践建议**：动手跑上面的代码，对比不同 $d_k$ 下缩放与不缩放的差异
