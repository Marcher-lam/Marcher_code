# 面试题：HSTU 和 Transformer 两种序列建模架构的对比

# 面试题：HSTU 和 Transformer 两种序列建模架构的对比

Transformer 与 HSTU（Hierarchical Sequential Transduction Unit）是序列建模中的重要架构，但它们的设计目标、技术实现和适用场景存在显著差异。

 HSTU 论文链接：https://arxiv.org/pdf/2402.17152  
 Transformer 论文链接：https://arxiv.org/pdf/1706.03762

<table><tr><td></td><td>Transformer</td><td>HSTU</td></tr><tr><td>设计目标</td><td>通用序列建模（最初为NLP设计），捕捉全局依赖关系</td><td>专为大规模推荐系统优化，处理高基数、非平稳的动态流式数据</td></tr><tr><td>注意力机制</td><td>基于Softmax的缩放点积注意力，输出是值向量的概率加权和</td><td>基于Pointwise聚合注意力，摒弃Softmax，直接加权求和，以保留用户偏好强度信息</td></tr><tr><td>位置/时间编码</td><td>通常使用正弦位置编码或可学习的位置嵌入，主要编码绝对或相对位置信息</td><td>引入相对注意力偏置（RAB），同时编码位置（p）和时间（t）信息，对推荐场景至关重要</td></tr><tr><td>前馈网络（FFN）</td><td>编码器-解码器每层都包含一个独立的FFN子层</td><td>通过门控机制U(X)等设计，省去了显式的FFN层，结构更简洁</td></tr><tr><td>计算效率与优化</td><td>面临长序列计算平方复杂度挑战，依赖如FlashAttention等优化</td><td>针对推荐数据长尾分布，采用随机长度（SL）策略提高稀疏性；有高度优化的内核，训练效率更高</td></tr><tr><td>主要应用场景</td><td>NLP（机器翻译、文本生成）、CV（图像分类、目标检测）、多模态任务</td><td>工业级生成式推荐系统，涵盖召回与排序任务</td></tr></table>

![](images/8d5a1a422e105452b5dc52a3fb7d576c3ad5de1a6a5138c0bb07c8833c5c1803.jpg)

![](images/f101c54098321393f8d8499c56527a093f9318a4356410ccb71540d28770e4b8.jpg)

# ① 算法原理对比

两者最核心的区别体现在注意力机制的计算公式上。

# Transformer 的注意力公式

Transformer 采用标准的缩放点积注意力，其核心公式如下：

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V
$$

 Q(Query)、K(Key)、V(Value)：分别是输入序列经过线性变换得到的矩阵。  
$d _ { k }$ ：是 K 的维度，用于缩放点积，防止内积过大导致 Softmax 梯度消失。  
 Softmax：对注意力得分进行归一化，使得所有注意力权重之和为 1，形成一个概率分布。这有助于稳定训练，但也会归一化掉原始的强度信号。

# HSTU的注意力公式

HSTU 的注意力机制，称为 Pointwise 聚合注意力，其公式可以简化为以下形式：

$$
U (X), Q (X), K (X), V (X) = \operatorname {S p l i t} \left(\phi_ {1} \left(f _ {1} (X)\right)\right)
$$

$$
A (X) = \phi_ {2} (Q (X) K (X) ^ {T} + r a b ^ {p, t})
$$

$$
Y (X) = f _ {2} (\operatorname {N o r m} (A (X) V (X)) \odot U (X))
$$

 U(X), Q(X), K(X), V(X)：通过一个共享的基础映射（如 MLP） $f _ { 1 } ( X )$ 和 Split 函数一次性得到。其中 $U ( X )$ 是一个专门的门控权重。

 $r a b ^ { p , t }$ ：相对注意力偏置，直接加在 $Q K ^ { T }$ 之后，同时注入相对位置差和相对时间差的信息。  
 $\phi _ { 2 }$ ：非线性激活函数（如 SiLU），取代了 Softmax。这使得注意力权重不再被归一化，可以保留绝对值大小所代表的用户交互强度（如点击时长、评论深度等）。  
 $\operatorname { N o r m } ( . . . ) \odot U ( X ) )$ ：对注意力池化后的结果进行归一化，再与门控权重 $U ( X )$ 进行点乘，这一设计使其不再需要独立的 FFN 层。

# ① 各自适用的场景

基于上述根本性差异，两者的适用场景有明确划分：

#  Transformer 更适合于以下场景：

 自然语言处理任务：如机器翻译、文本摘要、对话生成等，其中语言的语法结构使得概率分布的归一化具有天然合理性。  
 计算机视觉任务：当图像被处理为序列数据（如 ViT）时，Transformer 能有效捕捉全局信息。  
 多模态学习：作为统一骨干网络处理文本、图像、音频等多种模态信息。  
 科研探索与通用架构原型：由于其通用性和普适性，常作为新想法的基线模型。

#  HSTU 更专注于以下场景：

 工业级大规模推荐系统：这是其设计的根本目标。特别擅长处理包含数十亿动态变化物品ID、长用户行为序列的场景。  
 对用户偏好强度敏感的任务：例如，需要区分用户"短暂停留"与"深度阅读"行为差异的排序任务，HSTU 能更好地利用这种强度信号。  
 计算效率和推理延迟要求极高的在线服务：其 M-FALCON 等推理优化算法，能大幅降低复杂模型的服务成本。研究表明，即使是 SASRec 等传统推荐模型，在引入 RAB 和调整残差连接后也能获得一定的可扩展性。

# 核心差异深度解析

## Softmax vs SiLU：为什么推荐系统不应该用 Softmax？

Transformer 的 Softmax 将注意力权重归一化为概率分布 $\sum_j \alpha_{ij} = 1$。在 NLP 中这是合理的——翻译时每个词的"关注"是一种概率分配。

但在推荐系统中，用户的行为强度是绝对值而非相对值：
- 用户 A 观看视频 60 分钟（强偏好）
- 用户 B 观看视频 5 秒（弱偏好）

如果使用 Softmax，两个用户的注意力权重都被归一化为和为 1，强度差异被抹去。HSTU 使用 SiLU（或类似激活函数）保留了这种绝对强度信号。

数学上的区别：

```
Softmax: α_ij = exp(q_i · k_j) / Σ_j exp(q_i · k_j)    (归一化)
SiLU:    α_ij = silu(q_i · k_j + rab_{ij})               (保留绝对值)
```

## 相对注意力偏置（RAB）的设计

HSTU 的相对注意力偏置 $rab^{p,t}$ 同时编码了两种信息：

- **位置差** $p$：序列中两个行为的位置距离（第 1 个行为 vs 第 5 个行为）
- **时间差** $t$：两个行为的实际时间间隔（1 小时前 vs 1 天前）

这在推荐系统中至关重要：
- 用户 30 秒前浏览的商品对当前决策的影响，远大于 30 天前浏览的
- 即使时间很近，序列位置靠后的行为通常更相关

RAB 的参数化方式：

$$rab_{ij} = b_{p}(\Delta p_{ij}) + b_{t}(\Delta t_{ij})$$

其中 $\Delta p_{ij} = p_i - p_j$（位置差），$\Delta t_{ij} = t_i - t_j$（时间差），$b_p$ 和 $b_t$ 是可学习的嵌入表。

## 门控机制替代 FFN

标准 Transformer 的每层包含 Self-Attention + FFN 两个子层。HSTU 通过门控设计将两者融合：

$$Y(X) = f_2(\text{Norm}(A(X) \cdot V(X)) \odot U(X))$$

其中 $U(X)$ 起到了类似 FFN 中非线性变换的作用，但与注意力输出做了元素级乘法（而非串联）。这种设计的优势：
1. 参数更少（不需要独立的 FFN 权重矩阵）
2. 门控机制提供了更强的非线性表达能力
3. 注意力输出和门控的交互更加直接

## 随机长度（Stochastic Length, SL）策略

推荐系统的用户行为序列长度分布通常极度长尾（少数用户有数千条行为，大多数用户只有几十条）。HSTU 提出了 SL 策略：

- 训练时对每个样本随机截取序列长度，从均匀分布中采样
- 长序列被截断为较短的子序列，短序列保持不变
- 这使得模型对不同长度的输入都能良好适应
- 同时显著减少了训练时的计算量（平均序列长度变短）

## 代码实现对比

### Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout(self.W_o(attn_output)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

### HSTU Block

```python
class HSTUBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_pos=512, max_time_bins=256):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.base_proj = nn.Linear(d_model, d_model * 4)
        self.pos_bias = nn.Embedding(max_pos, n_heads)
        self.time_bias = nn.Embedding(max_time_bins, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, x, time_diffs=None, mask=None):
        batch_size, seq_len, d_model = x.shape
        proj = self.base_proj(x)
        U, Q, K, V = proj.chunk(4, dim=-1)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if time_diffs is not None:
            time_diffs_clamped = time_diffs.clamp(0, self.time_bias.num_embeddings - 1)
            rab = self.time_bias(time_diffs_clamped)
            rab = rab.permute(0, 3, 1, 2)
            attn_scores = attn_scores + rab
        attn_weights = F.silu(attn_scores)
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(1).unsqueeze(-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        normed = self.norm(attn_output)
        U = U.sigmoid()
        gated = normed * U
        output = self.out_proj(gated)
        return x + output
```

### 性能对比测试

```python
import time


def benchmark(model_cls, d_model, n_heads, seq_len, batch_size, n_iters=100, **kwargs):
    model = model_cls(d_model, n_heads, **kwargs)
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.time()
            _ = model(x)
            times.append(time.time() - start)
    return np.mean(times[10:])


d_model = 128
n_heads = 4
for seq_len in [64, 128, 256, 512]:
    tf_time = benchmark(TransformerBlock, d_model, n_heads, seq_len, 32, d_ff=512)
    hstu_time = benchmark(HSTUBlock, d_model, n_heads, seq_len, 32)
    print(f"Seq={seq_len:4d} | Transformer: {tf_time*1000:.2f}ms | HSTU: {hstu_time*1000:.2f}ms")
```

## 推荐场景选择指南

| 推荐任务 | 序列特征 | 推荐架构 | 原因 |
|---------|---------|---------|------|
| CTR 预估 | 短序列（<50） | Transformer | 序列短，Softmax 归一化损失可忽略 |
| CTCVR 预估 | 中长序列（50~500） | HSTU | 需要强度信号和 RAB |
| 生成式推荐 | 超长序列（>500） | HSTU | SL 策略 + 计算优化 |
| 召回 | 用户行为序列 | HSTU | 需要处理高基数物品 ID |
| 多模态推荐 | 图文+行为 | Transformer | 通用架构更适合多模态融合 |

## M-FALCON 推理优化

HSTU 配套的 M-FALCON（Millisecond-scale Falcon）推理算法，通过以下方式优化推理延迟：

1. **KV Cache 复用**：对于同一用户的多次请求，缓存历史的 Key/Value 表示
2. **自适应序列截断**：根据实时负载动态调整输入序列长度
3. **量化推理**：使用 INT8 量化减少内存和计算开销

这些优化使得 HSTU 在推荐场景的推理成本远低于同等参数量的 Transformer。

## 常见问题

1. **Q: HSTU 能否用于 NLP 任务？**
   A: 理论上可以，但不推荐。NLP 中的 Softmax 归一化有其语言学合理性（概率分配），HSTU 的设计是针对推荐场景优化的。

2. **Q: 能否在 Transformer 中引入 RAB？**
   A: 可以，ALiBi 等工作已经在 Transformer 中引入了类似的相对位置偏置。但 HSTU 的 RAB 同时编码了位置和时间，这是针对推荐场景的特殊设计。

3. **Q: HSTU 的 SiLU 激活会不会导致注意力值爆炸？**
   A: SiLU 的输出范围是 $(-0.28, \infty)$，不会像 ReLU 那样无上界增长。同时 HSTU 在注意力输出后接了 Norm 层，确保了数值稳定性。

## 学习总结

HSTU 和 Transformer 的核心差异在于：HSTU 是面向推荐系统"量身定制"的架构，每一个设计决策都针对推荐场景的特点（Softmax → SiLU 保留强度、FFN → 门控减少参数、位置编码 → RAB 编码时间信息）。Transformer 是通用架构，在推荐系统中也能工作但不是最优选择。理解这种"场景驱动设计"的思路，比记住具体公式更重要。

# 4.4 多任务&多场景建模
