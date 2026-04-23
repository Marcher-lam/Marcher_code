# GQA（Grouped-Query Attention）分组查询注意力

## 背景

传统 MHA 每个头独立 KV，显存占用大；MQA 所有头共享一组 KV，表达能力损失。GQA 在两者间取得平衡。

## 核心原理

将 num_heads 个 Query 头分为 num_groups 组，每组共享一组 K 和 V：

$$Attention_g(Q_g, K_g, V_g) = softmax\left(\frac{Q_g K_g^T}{\sqrt{d_k}}\right) V_g$$

$$Output = Concat(Attention_1, \dots, Attention_G) W^O$$

## 对比

| 特性 | MHA | MQA | GQA |
|------|-----|-----|-----|
| KV 头数 | num_heads | 1 | num_groups |
| 计算效率 | 低 | 高 | 中高 |
| 模型质量 | 高 | 低 | 接近 MHA |
| 代表模型 | BERT, GPT-3 | PaLM | LLaMA-2, Qwen |

## PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_groups):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_groups, self.d_k).transpose(1, 2)

        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        attn = F.softmax(Q @ K.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```
