# 面试题：Transformer 包含哪两种 Mask 机制，各自如何作用的？

面试题：Transformer 包含哪两种 Mask 机制，各自如何作用的？

Transformer 模型中包含两种关键的 Mask 机制：Padding Mask 和 Sequence Mask，它们在注意力计算中分别承担不同的作用，具体如下：

# 1、Padding Mask（填充掩码）

 作用：处理不同批次输入序列长度不一致的问题。通过将短序列末尾填充 0对齐长度，但模型需忽略这些无意义的填充位置。其核心目标是防止注意力机制关注无效的填充区域。

#  实现方式：

 在填充位置（值为0）加上一个极大的负数（如负无穷），经过 Softmax后这些位置注意力权重趋近于 0。  
 具体实现时，生成一个布尔型张量（True 表示填充 Padding 位置），扩展为与注意力矩阵相同的维度。

 应用场景：所有层的注意力计算（包括 Encoder 和 Decoder）均需使用 Padding Mask。

# 2. Sequence Mask（序列掩码）

 作用：仅用于Decoder的Self-Attention 层， 防止模型在训练时"窥见"未来信息。例如，解码第t个词时，只能依赖前t-1个词的输出，避免数据标签泄漏。

#  实现方式：

 生成一个上三角矩阵（对角线以上元素为 1，其余为 0），作用于序列。在计算注意力时，将未来位置（t 时刻之后）的权重设为负无穷，从而屏蔽这些位置的影响。  
 具体代码中，可通过 torch.triu 函数生成该矩阵，并设置 diagonal=1 以排除当前时间步自身。

 应用场景：仅用于 Decoder 的 Self-Attention 层，与 Padding Mask 叠加后共同作用于注意力计算。

# 两种Mask的叠加使用

 在 Decoder 的 Self-Attention 中，需同时处理填充位置和未来信息屏蔽。具体实现方式是将 Padding Mask 和 SequenceMask 相加，形成一个综合的掩码矩阵，再作用于注意力权重。其他情况下（如 Encoder 的 Self-Attention 或 Encoder-Decoder Attention），仅需使用 Padding Mask。

总结对比  

<table><tr><td>Mask 类型</td><td>作用</td><td>应用场景</td><td>实现方法</td></tr><tr><td>Padding Mask</td><td>忽略无效填充位置</td><td>所有注意力层</td><td>填充位置加负无穷</td></tr><tr><td>Sequence Mask</td><td>防止解码看到未来信息</td><td>Decoder Self-Attention 层</td><td>生成上三角矩阵，屏蔽未来位置</td></tr></table>

# 3. Padding Mask 的数学原理

Padding Mask 的核心目标是让填充位置的注意力权重在 Softmax 后趋近于零。

注意力权重计算公式为：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是掩码矩阵。对于 Padding 位置 $j$：

$$M_{ij} = \begin{cases} 0 & \text{如果 } j \text{ 不是填充位置} \\ -\infty & \text{如果 } j \text{ 是填充位置} \end{cases}$$

经过 Softmax 后，填充位置的权重为：

$$\alpha_{ij} = \frac{e^{s_{ij} + M_{ij}}}{\sum_k e^{s_{ik} + M_{ik}}} = \frac{e^{-\infty}}{\sum_k e^{s_{ik} + M_{ik}}} \approx 0$$

在 PyTorch 中，通常用 $-10^9$ 或 `float('-inf')` 代替 $-\infty$。

# 4. Sequence Mask（Causal Mask）的数学原理

Sequence Mask 确保自回归性质：位置 $i$ 只能关注位置 $\leq i$ 的信息。

掩码矩阵为下三角矩阵：

$$M_{ij}^{\text{causal}} = \begin{cases} 0 & \text{如果 } j \leq i \\ -\infty & \text{如果 } j > i \end{cases}$$

这个矩阵等价于：

$$M^{\text{causal}} = (1 - \text{LowerTriangular}) \times (-\infty)$$

在训练中，Sequence Mask 使得模型通过一次前向传播就能并行计算所有位置的概率，而无需逐步自回归解码，极大提升了训练效率。

# 5. 两种 Mask 在 GPT 和 BERT 中的使用差异

## BERT（仅编码器）

- **使用 Padding Mask**：屏蔽 [PAD] token
- **不使用 Sequence Mask**：BERT 是双向注意力模型，每个位置可以关注所有位置
- 额外使用 **Attention Mask**：在 MLM 任务中屏蔽 [MASK] token 的注意力（非必须，属于任务相关）

## GPT（仅解码器）

- **使用 Padding Mask**：屏蔽 [PAD] token
- **使用 Sequence Mask（Causal Mask）**：确保自回归生成，当前 token 只能看到之前的 token
- 两种 Mask 叠加使用

# 6. 推荐系统中的 Causal Mask 应用

在推荐系统中，Causal Mask 有重要应用：

**序列推荐（SASRec、BST 等）**：用户行为序列建模中，预测第 $t+1$ 个交互时，只能使用前 $t$ 个行为。Causal Mask 确保模型不"偷看"未来的行为，与线上推理时的一致。

**多目标预估**：在多任务推荐模型中，有时需要防止信息从高优先级任务泄漏到低优先级任务。

# 7. 完整 PyTorch 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_padding_mask(seq, pad_idx=0):
    batch_size, seq_len = seq.shape
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def create_combined_mask(seq, pad_idx=0):
    batch_size, seq_len = seq.shape
    padding_mask = create_padding_mask(seq, pad_idx)
    causal_mask = create_causal_mask(seq_len)
    combined = padding_mask | causal_mask
    return combined

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-1e9'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output), attn_weights

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_out, attn_w = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_w

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        self_attn_out, self_attn_w = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self_attn_out)
        cross_attn_out, cross_attn_w = self.cross_attention(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + cross_attn_out)
        x = self.norm3(x + self.ffn(x))
        return x, self_attn_w, cross_attn_w


def demo_padding_mask():
    print("=" * 60)
    print("Padding Mask 演示")
    print("=" * 60)
    seq = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 0, 0],
        [9, 10, 0, 0, 0]
    ])
    print(f"输入序列:\n{seq}")
    mask = create_padding_mask(seq, pad_idx=0)
    print(f"\nPadding Mask 形状: {mask.shape}")
    print(f"Padding Mask (True=被遮蔽):\n{mask.squeeze().int()}")
    return seq, mask

def demo_causal_mask():
    print("\n" + "=" * 60)
    print("Causal Mask (Sequence Mask) 演示")
    print("=" * 60)
    seq_len = 5
    mask = create_causal_mask(seq_len)
    print(f"Causal Mask (True=被遮蔽, seq_len={seq_len}):\n{mask.squeeze().squeeze().int()}")
    print("\n解释:")
    print("  位置0只能看到位置0")
    print("  位置1能看到位置0,1")
    print("  位置2能看到位置0,1,2")
    print("  位置3能看到位置0,1,2,3")
    print("  位置4能看到位置0,1,2,3,4")

def demo_combined_mask():
    print("\n" + "=" * 60)
    print("组合 Mask 演示 (Padding + Causal)")
    print("=" * 60)
    seq = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 0, 0],
        [9, 10, 0, 0, 0]
    ])
    combined = create_combined_mask(seq, pad_idx=0)
    print(f"输入序列:\n{seq}")
    print(f"\n组合 Mask (True=被遮蔽):\n{combined.squeeze().int()}")

def demo_attention_with_masks():
    print("\n" + "=" * 60)
    print("带 Mask 的注意力计算演示")
    print("=" * 60)
    torch.manual_seed(42)

    d_model = 16
    n_heads = 2
    batch_size = 2
    seq_len = 4

    encoder = TransformerEncoderBlock(d_model, n_heads, d_ff=64)
    seq = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 0, 0]
    ])
    x = torch.randn(batch_size, seq_len, d_model)
    padding_mask = create_padding_mask(seq, pad_idx=0)

    enc_out, enc_attn = encoder(x, mask=padding_mask)
    print(f"Encoder 输入: {x.shape}")
    print(f"Encoder 输出: {enc_out.shape}")
    print(f"Encoder 注意力权重 (batch 0, head 0):")
    print(f"  {enc_attn[0, 0].detach().numpy().round(3)}")

    decoder = TransformerDecoderBlock(d_model, n_heads, d_ff=64)
    decoder_input = torch.randn(batch_size, seq_len, d_model)
    combined_mask = create_combined_mask(seq, pad_idx=0)

    dec_out, self_attn, cross_attn = decoder(
        decoder_input, enc_out,
        self_mask=combined_mask,
        cross_mask=padding_mask
    )
    print(f"\nDecoder 输出: {dec_out.shape}")
    print(f"Decoder Self-Attention 权重 (batch 0, head 0):")
    self_w = self_attn[0, 0].detach()
    print(f"  {self_w.numpy().round(3)}")
    print(f"  第2个样本位置2,3是否被遮蔽(padding): {self_w[1, 2, 2].item() == 0 and self_w[1, 2, 3].item() == 0}")

if __name__ == "__main__":
    demo_padding_mask()
    demo_causal_mask()
    demo_combined_mask()
    demo_attention_with_masks()
```

# 8. Mask 机制的直观理解

想象一个会议室对话场景：

**Padding Mask** = 会议桌上有空椅子。你在听别人讲话时，不会去听空椅子上"没有人"说的话。Padding Mask 告诉注意力机制："这些位置是空的，不要关注它们"。

**Sequence Mask** = 发言顺序规则。在第 3 个人发言之前，他只能参考第 1、2 个人的观点，不能偷看第 4、5 个人还没说的内容。Sequence Mask 告诉注意力机制："你还没到看未来信息的时候"。

在 Decoder 的 Self-Attention 中，两种规则同时生效：既不能看空椅子（Padding），也不能看还没发言的人（Causal）。

# 9. 常见问题与易错点

- **Mask 的值方向**：PyTorch 中 `masked_fill(mask, value)` 的 mask 为 True 的位置会被填充。容易混淆 True 是"保留"还是"遮蔽"。正确理解：True = 被遮蔽（不参与注意力计算）。
- **维度广播**：Padding Mask 的形状通常为 $(B, 1, 1, S)$，Causal Mask 为 $(1, 1, S, S)$。叠加时需要正确广播到 $(B, H, S, S)$。
- **Causal Mask 的 diagonal 参数**：`torch.triu` 的 `diagonal=1` 表示不包含对角线，即当前位置可以看到自己。如果设为 `diagonal=0`，则当前位置也会被遮蔽，这通常是错误的。
- **Encoder 不需要 Causal Mask**：BERT 类模型的双向注意力不需要因果约束。在 Encoder-Decoder 架构中，Encoder 的 Self-Attention 只需 Padding Mask。
- **Cross-Attention 的 Mask**：Decoder 的 Cross-Attention 只需 Padding Mask（来自 Encoder 的 Padding），不需要 Causal Mask，因为 Encoder 的输出已经是完整的编码。
- **推理时的 KV Cache 与 Mask**：自回归推理时，由于每次只生成一个 token，Causal Mask 退化为恒等（不需要遮蔽任何东西），但 Padding Mask 仍然需要。

# 10. 不同模型的 Mask 策略对比

## 10.1 GPT / BERT / T5 的 Mask 差异

| 模型 | Padding Mask | Causal Mask | 特殊 Mask | 训练效率 |
|------|-------------|-------------|----------|---------|
| GPT 系列 | 有（屏蔽 PAD） | 有（自回归生成） | 无 | 需要逐 token 自回归 |
| BERT | 有（屏蔽 PAD） | 无（双向注意力） | MLM Mask（随机屏蔽 15% token） | 一次前向传播完成 |
| T5（Encoder） | 有 | 无 | Span Mask（屏蔽连续片段） | 一次前向传播 |
| T5（Decoder） | 有 | 有 | Sentinel Token Mask | 自回归生成 |
| UniLM | 有 | 部分（根据任务切换） | 结合双向/单向/seq2seq | 灵活但实现复杂 |
| XLNet | 有 | 排列语言模型 Mask | Permutation Mask | 计算开销较大 |

## 10.2 Mask 策略的优缺点分析

### Padding Mask

**优点**：实现简单，所有框架原生支持；有效避免无效位置对注意力的干扰；支持变长 batch 处理。

**缺点**：填充比例过高时（如短序列 + 大 batch），有效计算比例低；某些实现中填充位置仍消耗计算资源（未优化前）。

**优化建议**：使用 `nn.utils.rnn.pack_padded_sequence` 压缩填充位置，或使用 Flash Attention 自动跳过填充。

### Causal Mask

**优点**：保证自回归性质，训练时可并行计算所有位置（一次前向传播完成）；实现简单（下三角矩阵）。

**缺点**：限制了当前位置对未来的信息获取，双向建模能力弱于 BERT；推理时需要 KV Cache 配合才能高效。

### MLM Mask（BERT 特有）

**优点**：利用了双向上下文信息，理解能力强；预训练与微调一致性好。

**缺点**：预训练与微调存在输入差异（[MASK] token 仅在预训练出现）；屏蔽比例为超参数（15% 是经验值）。

## 10.3 推荐系统中的 Mask 特殊应用

### 序列推荐中的 Causal Mask

在 SASRec、BST 等序列推荐模型中，Causal Mask 的使用需要特别注意：

```python
def create_rec_causal_mask(seq_len, behavior_types=None):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    if behavior_types is not None:
        for i in range(seq_len):
            for j in range(i):
                if behavior_types[j] == 'negative_feedback':
                    mask[i, j] = 0
    return mask.unsqueeze(0).unsqueeze(0)
```

**推荐场景的特殊考量**：

1. **负反馈屏蔽**：用户的不喜欢/跳过行为应被特殊处理，可选择屏蔽或降低权重
2. **多类型行为 Mask**：点击、购买、加购等不同行为类型可设计不同的 Mask 策略
3. **时间衰减 Mask**：基于时间间隔的软 Mask（非硬 0/1），越远的行为权重越低

### 列表推荐中的序列 Mask

推荐列表生成（如 List-wise 推荐）中，Causal Mask 确保第 $i$ 个推荐位只参考前 $i-1$ 个已推荐物品，避免信息泄露：

```python
def listwise_causal_mask(list_size, num_candidates):
    causal = torch.tril(torch.ones(list_size, list_size))
    return causal.unsqueeze(0).unsqueeze(0)
```

## 10.4 Mask 机制的选择指南

| 场景 | 推荐的 Mask 组合 | 理由 |
|------|----------------|------|
| 双向文本理解（分类/匹配） | 仅 Padding Mask | 双向注意力利用完整上下文 |
| 文本生成 | Padding + Causal Mask | 保证自回归性质 |
| 序列推荐 | Padding + Causal Mask + 时间衰减 | 防止未来信息泄露 + 远端衰减 |
| 列表推荐 | Padding + Causal Mask | 确保列表生成的因果性 |
| 多任务推荐 | Padding + 任务间隔离 Mask | 防止任务间信息泄露 |
| 对话推荐 | Padding + Causal Mask + 轮次 Mask | 区分对话轮次边界 |
