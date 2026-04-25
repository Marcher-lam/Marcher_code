# GQA（Grouped-Query Attention）分组查询注意力

## 背景

传统 MHA 每个头独立 KV，显存占用大；MQA 所有头共享一组 KV，表达能力损失。GQA 在两者间取得平衡。

## 核心原理

将 num_heads 个 Query 头分为 num_groups 组，每组共享一组 K 和 V：

$$Attention_g(Q_g, K_g, V_g) = softmax\left(\frac{Q_g K_g^T}{\sqrt{d_k}}\right) V_g$$

$$Output = Concat(Attention_1, \dots, Attention_G) W^O$$

## 数学公式推导

设输入序列 $X \in \mathbb{R}^{B \times T \times d_{model}}$，其中 $B$ 为批次大小，$T$ 为序列长度，$d_{model}$ 为模型维度。

**Query 投影：**

$$Q = X W^Q, \quad Q \in \mathbb{R}^{B \times T \times n_{heads} \cdot d_k}$$

**Key / Value 投影（共享）：**

$$K = X W^K, \quad K \in \mathbb{R}^{B \times T \times n_{groups} \cdot d_k}$$

$$V = X W^V, \quad V \in \mathbb{R}^{B \times T \times n_{groups} \cdot d_k}$$

**分组注意力计算：**

将 $Q$ 按 $n_{groups}$ 分组，每组包含 $h_{per\_group} = n_{heads} / n_{groups}$ 个头。对第 $g$ 组：

$$\text{Attn}_g = \text{softmax}\left(\frac{Q_g K_g^T}{\sqrt{d_k}}\right) V_g$$

其中 $Q_g \in \mathbb{R}^{B \times h_{per\_group} \times T \times d_k}$，$K_g, V_g \in \mathbb{R}^{B \times T \times d_k}$ 通过 `repeat_interleave` 扩展到 $h_{per\_group}$ 个头。

**KV Cache 机制：**

在自回归生成中，KV Cache 存储 $K_{\leq t}$ 和 $V_{\leq t}$：

$$K_{cache} = [K_1, K_2, \dots, K_t] \in \mathbb{R}^{B \times n_{groups} \times t \times d_k}$$

每个生成步骤只需计算 $K_t, V_t$ 并拼接到缓存中，避免重复计算历史 Token。相比 MHA，KV Cache 显存占用减少为 $\frac{n_{groups}}{n_{heads}}$。

## 对比

| 特性 | MHA | MQA | GQA |
|------|-----|-----|-----|
| KV 头数 | num_heads | 1 | num_groups |
| 计算效率 | 低 | 高 | 中高 |
| 模型质量 | 高 | 低 | 接近 MHA |
| 代表模型 | BERT, GPT-3 | PaLM | LLaMA-2, Qwen |
| KV Cache 显存 | 高 | 最低 | 中等 |
| 推理速度 | 慢 | 最快 | 较快 |

## 应用场景

- **大语言模型推理**：LLaMA-2/3、Qwen、Mistral 等主流 LLM 均采用 GQA，在长序列生成中 KV Cache 显存占用降低 4-8 倍
- **多模态模型**：LLaVA 等视觉语言模型使用 GQA 处理大量图像 Token
- **推荐系统精排**：基于 Transformer 的序列建模中，GQA 降低大规模用户行为序列的注意力计算开销
- **流式推理服务**：在线部署时 GQA 减少显存带宽瓶颈，提升吞吐量

## 优缺点分析

**优点：**
- 相比 MHA 显著降低 KV Cache 显存和带宽开销（降低比例为 $n_{heads}/n_{groups}$）
- 推理速度接近 MQA，模型质量接近 MHA
- 超参数 $n_{groups}$ 灵活调节效率与质量的平衡
- 与 FlashAttention 等 GPU 优化技术兼容性好

**缺点：**
- $n_{groups}$ 需要精心选择，过小会损失模型质量
- 实现复杂度高于 MHA 和 MQA
- 训练时 KV 共享可能导致不同 Query 头间的信息干扰

## 与 LLM 的集成方式

以 LLaMA-2 为例的 GQA 配置：
- 70B 模型：$n_{heads}=64$，$n_{groups}=8$（每组 8 个头共享 KV）
- KV Cache 显存降低为 MHA 的 $1/8$
- 配合 RoPE（旋转位置编码）和 SwiGLU 激活函数使用

## PyTorch 实现（含 KV Cache）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """GQA 分组查询注意力实现，支持 KV Cache"""

    def __init__(self, d_model, n_heads, n_groups, max_seq_len=2048):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.k_cache = nn.Parameter(
            torch.zeros(1, n_groups, max_seq_len, self.d_k), requires_grad=False
        )
        self.v_cache = nn.Parameter(
            torch.zeros(1, n_groups, max_seq_len, self.d_k), requires_grad=False
        )
        self.cache_pos = 0

    def forward(self, x, use_cache=False, past_kv=None):
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_groups, self.d_k).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        new_kv = (K, V) if use_cache else None

        K_expanded = K.repeat_interleave(self.heads_per_group, dim=1)
        V_expanded = V.repeat_interleave(self.heads_per_group, dim=1)

        attn = F.softmax(
            Q @ K_expanded.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1
        )
        out = (attn @ V_expanded).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), new_kv


class MultiHeadAttention(nn.Module):
    """标准 MHA 实现，用于对比"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = F.softmax(Q @ K.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


def benchmark_kv_cache():
    """对比 MHA 与 GQA 的 KV Cache 显存占用"""
    d_model = 4096
    n_heads = 32
    seq_len = 512
    batch_size = 4

    mha = MultiHeadAttention(d_model, n_heads)
    gqa = GroupedQueryAttention(d_model, n_heads, n_groups=4)

    x = torch.randn(batch_size, seq_len, d_model)

    mha_k_params = n_heads * seq_len * (d_model // n_heads)
    gqa_k_params = 4 * seq_len * (d_model // n_heads)
    print(f"MHA KV Cache 参数量: {mha_k_params * batch_size}")
    print(f"GQA KV Cache 参数量: {gqa_k_params * batch_size}")
    print(f"GQA 相比 MHA 显存降低: {(1 - gqa_k_params / mha_k_params) * 100:.1f}%")


if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    n_groups = 2
    gqa = GroupedQueryAttention(d_model, n_heads, n_groups)
    x = torch.randn(2, 10, d_model)
    out, kv = gqa(x, use_cache=True)
    print(f"输出形状: {out.shape}")
    out2, kv2 = gqa(torch.randn(2, 1, d_model), use_cache=True, past_kv=kv)
    print(f"增量输出形状: {out2.shape}")
    benchmark_kv_cache()
```

## 常见问题与易错点

1. **KV 扩展维度错误**：`repeat_interleave` 必须在 `dim=1`（头维度）上操作，而非序列维度
2. **$n_{groups}$ 整除性**：$n_{heads}$ 必须能被 $n_{groups}$ 整除，否则 `heads_per_group` 不是整数
3. **KV Cache 拼接顺序**：自回归推理时新 Token 的 K/V 应拼接到缓存末尾，顺序反了会导致因果性错误
4. **RoPE 兼容性**：使用旋转位置编码时，K 的位置索引需要与缓存位置对齐

## 学习总结

GQA 是 MHA 与 MQA 的优雅折中方案，通过分组共享 KV 在推理效率和模型质量间取得最佳平衡。其核心思想简洁：同组 Query 共享 KV 参数，减少 KV Cache 显存占用的同时保持足够的注意力多样性。在 LLaMA-2 之后已成为大模型注意力机制的事实标准。
