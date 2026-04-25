# LLM架构详解

## 概述

本章详细解释Qwen3 LLM的架构组件。虽然本书的重点是推理方法，而不是LLM本身，但理解LLM的内部工作原理有助于更好地掌握推理技术的工作原理。

**注意：** 完全从零实现LLM需要一本单独的书，这正是《Build A Large Language Model (From Scratch)》的主题。但感兴趣的读者可以在这里看到Qwen3实现的完整代码。

## Qwen3 vs GPT-2架构对比

Qwen3（2025年发布）和GPT-2（2019年发布）整体上非常相似，因为它们都基于原始transformer架构的解码器子模块。

**主要区别：**
- 自2019年以来，一些设计选择已经演变
- 大多数在Qwen3中找到的设计选择不是Qwen3独有的，而是在许多其他当代LLM中也能找到

### 架构对比图

```
GPT-2 (2019):
输入 → 嵌入 → [Transformer块] × N → LayerNorm → 输出
         ↓
    绝对位置嵌入

Qwen3 (2025):
输入 → 嵌入 → [Transformer块] × N → RMSNorm → 输出
         ↓
    RoPE旋转位置嵌入
```

## 1. RMS归一化（RMS Normalization）

### LayerNorm vs RMSNorm

与使用标准LayerNorm的GPT-2不同，较新的Qwen3架构用均方根层归一化（RMSNorm）替换了它。这是近年来在模型架构中变得日益普遍的趋势。

### 功能比较

**LayerNorm（左）：**
- 减去均值并除以标准差
- 使层输出具有零均值和单位方差（方差为1，标准差为1）
- 这在梯度值方面产生有利于稳定训练的性质

**RMSNorm（右）：**
- 输入除以均方根
- 将激活缩放到可比较的大小，而不强制零均值或单位方差
- 均值和方差保持在训练稳定的合理范围内

### RMSNorm的优势

**计算更便宜：**
- RMSNorm减少昂贵的均值和方差计算为单个均方根操作
- 这将跨特征减少从两次降低到一次
- 降低GPU上的通信开销并略微提高训练效率

**参数更少：**
- RMSNorm默认不使用偏差（shift）项
- 减少可训练参数数量

**示例：**
```
输入: [2, 4, 6, 8]

LayerNorm:
均值 = 5
方差 = 5
输出 ≈ [-1.35, -0.45, 0.45, 1.35]  # 均值=0, 标准差=1

RMSNorm:
均方根 = sqrt(16) = 4
输出 = [0.5, 1.0, 1.5, 2.0]  # 均值=1.25, 方差=0.41
```

### 代码实现

```python
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
```

## 2. 前馈网络模块（Feed-Forward Module）

### 标准前馈 vs 门控线性单元（GLU）

前馈网络模块（一个小多层感知器）被门控线性单元（GLU）变体替换，这是在2020年的一篇论文中引入的。

在这种设计中，两个标准全连接层被三个替换，如图所示。

```
GPT-2（标准）:
输入 → [Linear1] → 激活 → [Linear2] → 输出

Qwen3（GLU变体）:
输入 → [Linear1] ┐
         → 激活 → 元素乘法 → [Linear3] → 输出
输入 → [Linear2] ┘
```

### Qwen3前馈模块实现

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(
            cfg["emb_dim"],
            cfg["hidden_dim"],
            dtype=cfg["dtype"],
            bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"],
            cfg["hidden_dim"],
            dtype=cfg["dtype"],
            bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"],
            cfg["emb_dim"],
            dtype=cfg["dtype"],
            bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
```

**说明：** 这里的非线性激活函数是SiLU函数，稍后讨论。

### 参数数量对比

虽然看起来GLU变体应该优于标准前馈变体，因为它添加了额外的线性层（三个而不是两个）并且看起来有更多参数，但这种直觉是误导的。

**示例：**
假设"Linear层1"的输入维度为1024（`cfg["emb_dim"]`）。

**GLU变体：**
```
fc1: 1024 × 3072 = 3,145,728
fc2: 1024 × 3072 = 3,145,728
fc3: 3072 × 1024 = 3,145,728
总计: 9,437,184 参数
```

**标准前馈：**
```
fc1: 1024 × 2×3072 = 6,291,456
fc2: 2×3072 × 1024 = 6,291,456
总计: 12,582,912 参数
```

**结论：** GLU变体的fc1和fc2层各是标准前馈模块中fc1层宽度的一半，实际上参数更少。

### GLU性能优势

GLU变体通常比常规前馈模块有更少的参数，但表现更好。改进来自于门控机制引入的额外乘法交互`activation(x_fc1) * x_fc2`，这增加了模型的表达能力。

这类似于在给定适当训练的情况下，更深、更窄的网络可以优于更浅、更宽的网络。

## 激活函数：SiLU（Swish）

### 历史演变

历史上，激活函数是辩论的热门话题，直到深度学习社区在十多年前大体上收敛于整流线性单元（ReLU）。

**ReLU：**
- 简单且计算便宜
- 但在零处有急剧的折点

这激励研究者探索更平滑的函数，如高斯误差线性单元（GELU）和sigmoid线性单元（SiLU）。

### GELU

GELU涉及高斯累积分布函数（CDF）。计算这个CDF很慢，因为它使用分段逻辑和指数，这使得编写融合、优化的GPU内核变得困难（虽然存在使用更便宜操作的tanh近似，运行更快且结果接近）。

**结论：** 虽然GELU产生平滑的激活曲线，但整体上比简单函数计算上更昂贵。

### SiLU（Swish）

更新的模型主要用SiLU（也称为Swish）函数替换了GELU，它：
- 平滑地将大负输入抑制到~0
- 对于大正输入近似线性

SiLU有类似的平滑度，但计算比GELU略便宜，并提供可比的建模性能。

**实际使用：**
- SiLU现在在大多数架构中使用
- GELU仅在部分模型中保留使用，如Google的Gemma开源权重LLM

在前馈模块实现中，通过`nn.functional.silu`调用这个SiLU函数。前馈模块也常被称为**SwiGLU**，这是从术语Swish和GLU派生的缩写。

## 3. 旋转位置嵌入（RoPE）

### 为什么需要位置编码？

在基于transformer的LLM中，位置编码是必要的，因为注意力机制。默认情况下，注意力将输入token视为没有顺序。

**原始GPT：** 绝对位置嵌入通过为序列中的每个位置添加一个学习的嵌入向量来解决这个问题，然后将其添加到token嵌入。

### RoPE的革新

RoPE（Rotary Position Embeddings）引入了不同的方法：
- 不是将位置信息作为单独嵌入添加
- 通过以一种依赖于每个token位置的方式旋转注意力机制中的查询和键向量来编码位置信息

### RoPE的实现形式

RoPE可以用两种数学上等效的方式实现：
1. **交错形式**：将相邻维度配对进行旋转
2. **两半形式**：为了方便将维度分为余弦和正弦两半

### RoPE代码实现

```python
def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    dtype=torch.float32
):
    assert head_dim % 2 == 0, "嵌入维度必须是偶数"

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "头维度必须是偶数"

    x1 = x[..., : head_dim // 2]  # 前半
    x2 = x[..., head_dim // 2:]  # 后半

    cos = cos[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    # 形状变为: (1, 1, seq_len, head_dim // 2)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)
```

## 4. 分组查询注意力（GQA）

### MHA vs GQA

分组查询注意力（GQA）已经成为原始多头注意力（MHA）机制的标准、更计算和参数高效的替代方案。

**MHA：** 每个头也有自己的一组键和值

**GQA：** 为了减少内存使用，多个头共享同一组键和值投影

### GQA的核心思想

减少键和值头的数量，在多个查询头之间共享它们。这（1）降低模型的参数数量和（2）减少推理期间键和值张量的内存带宽使用，因为需要从KV缓存中存储和检索的键和值更少。

### 性能比较

虽然GQA主要是MHA的计算效率解决方案，但消融研究（如在原始GQA论文中呈现）显示，它在LLM建模性能方面与标准MHA表现相当。

### GQA实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups,
                 head_dim=None, qk_norm=False, dtype=None):
        super().__init__()

        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(
            d_in,
            num_kv_groups * head_dim,
            bias=False,
            dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in,
            num_kv_groups * head_dim,
            bias=False,
            dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)  # 形状: (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # 形状: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # 形状: (b, num_tokens, num_kv_groups * head_dim)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache.get(layer_idx)
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys, values)
        else:
            start_pos = 0
            keys, values = keys_new, values_new
            next_cache = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        return self.out_proj(context), next_cache
```

### QKNorm

你可能注意到GQA机制中的实现还包括一个`qk_norm`参数。这不是标准GQA设计的一部分。当`qk_norm = True`时，对查询和键都应用额外的基于查询/键RMSNorm的归一化，称为QKNorm，这是Qwen3中使用的技术。如前在RMSNorm部分讨论的，QKNorm有助于提高训练稳定性。

## 5. Transformer块

### 结构

Transformer块是LLM的中心组件，它结合了本附录迄今为止涵盖的所有单个元素。如图所示，它重复多次；在Qwen3的0.6亿参数版本中，它重复28次。

### Transformer块实现

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )

        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x

        x = self.norm1(x)
        x, next_cache = self.att(
            x,
            mask,
            cos,
            sin,
            start_pos=start_pos,
            cache=cache
        )
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache
```

**观察：** 列表中的transformer块只是连接了我们在前面部分实现的各种元素。

## 6. 主模型代码

### Qwen3Model类

我们将定义在第2章中导入和使用的`Qwen3Model`类。

### 完整实现

```python
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 主模型参数
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
            dtype=cfg["dtype"]
        )

        # 可重用工具
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg = cfg
        self.current_pos = 0  # 跟踪KV缓存中的当前位置

    def forward(self, inidx, cache=None):
        tok_embedding = self.tok_emb(inidx)
        x = tok_embedding

        num_tokens = x.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end

            mask = torch.triu(
                torch.ones(
                    pos_end, pos_end,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )[pos_start: pos_end, pos_start: pos_end]
        else:
            pos_start = 0  # 不是严格必需但有助于torch.compile
            mask = torch.triu(
                torch.ones(
                    num_tokens, num_tokens,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )
            mask = mask[None, None, :, :]  # 形状 (1, 1, num_tokens, num_tokens)

        next_cache = []

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(
                x,
                mask,
                self.cos,
                self.sin,
                start_pos=pos_start,
                cache=blk_cache
            )
            if cache is not None:
                cache.update(i, new_blk_cache)
            next_cache.append(new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        return logits

    def reset_kv_cache(self):
        self.current_pos = 0
```

**说明：**
- 由于我们已经拥有所有主要成分，`Qwen3Model`类只是在transformer块周围添加了一些组件
- 即嵌入和输出层（包括另一个RMSNorm层）
- 代码可能显得有些复杂，这是由于KV缓存选项
- 如第2章讨论，KV缓存可以加快文本生成过程

## 总结

Qwen3架构整合了现代LLM的多个先进技术：

1. **RMSNorm**：比LayerNorm计算更高效的归一化
2. **GLU前馈**（SwiGLU）：比标准前馈网络更高效且表现更好
3. **SiLU激活**：比GELU更平滑且计算更便宜
4. **RoPE**：比绝对位置嵌入更优雅的位置编码
5. **GQA**：比MHA更高效的注意力机制
6. **QKNorm**：提高训练稳定性的额外归一化

这些组件一起构成了一个强大且高效的LLM架构，作为我们实现推理技术的基础。
