# MLA、GQA、DSA 注意力机制全面对比

## 1. 算法基础认知

现代大语言模型面临的核心挑战之一是 **KV Cache 的内存瓶颈**。标准多头注意力（MHA）每个注意力头都需缓存独立的 Key 和 Value，在长序列推理时显存占用巨大。GQA、MLA、DSA 分别从不同角度优化这一问题，是当前大模型推理优化的三大主流注意力架构。

## 2. 核心对比

| 维度 | MHA（基准） | GQA | MLA | DSA |
|------|------------|-----|-----|-----|
| 核心思路 | 每头独立 KV | 分组共享 KV 头 | 低秩压缩 KV 到潜空间 | 动态筛选 Top-K Token |
| 优化对象 | 无 | KV 头数量 | KV 特征维度 | 参与计算的 Token 数 |
| KV 缓存 | 1× | MHA 的 1/4~1/8 | MHA 的 6%~10% | 较 MLA 再降 75% |
| 计算复杂度 | O(n²·h·d) | O(n²·g·d) | O(n²·d_c) | O(n·k·d) |
| 精度 | 最高 | 接近 MHA | 持平甚至超越 MHA | 精度损失 <0.5% |
| 代表模型 | GPT-3, PaLM | Llama 2/3, Qwen | DeepSeek V2/V3 | DeepSeek V3.2 |

## 3. GQA（分组查询注意力）

### 3.1 详细原理

GQA 将 Q 的头分为 G 组，每组共享一组 KV。当 G=1 时退化为 MQA，G=H 时退化为 MHA。

设 Q 头数 H=32，KV 组数 G=4：
- MHA：32 组 KV（128 维 × 32 = 4096 维缓存/Token）
- GQA：4 组 KV（128 维 × 4 = 512 维缓存/Token），压缩比 8×

### 3.2 数学公式

$$Attention(Q_i, K_g, V_g) = softmax\left(\frac{Q_i K_g^T}{\sqrt{d_k}}\right)V_g$$

其中 $g = \lfloor i \cdot G / H \rfloor$ 为第 $i$ 个 Q 头所属的组。

### 3.3 PyTorch 代码

```python
import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    def __init__(self, d_model=1024, num_q_heads=32, num_kv_groups=4):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_groups = num_kv_groups
        self.heads_per_group = num_q_heads // num_kv_groups
        self.d_k = d_model // num_q_heads
        
        self.W_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(num_q_heads * self.d_k, d_model, bias=False)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_kv_groups, self.d_k).transpose(1, 2)
        
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        K = K.reshape(B, self.num_q_heads, S, self.d_k)
        V = V.reshape(B, self.num_q_heads, S, self.d_k)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

## 4. MLA（多头潜变量注意力）

### 4.1 详细原理

MLA 由 DeepSeek 提出，核心思想是**不减少 KV 头数**，而是将每组 KV 通过低秩投影压缩到低维潜空间。推理时只需缓存压缩后的潜变量，推理质量几乎无损。

KV 缓存仅为 MHA 的 **6%~10%**。

### 4.2 数学公式

压缩阶段：

$$c_t^{KV} = W_{DKV} \cdot h_t$$

推理缓存 $c_t^{KV}$（维度 $d_c \ll d_h$），解压时：

$$k_t = W_{UK} \cdot c_t^{KV}, \quad v_t = W_{UV} \cdot c_t^{KV}$$

同样对 Q 也做低秩压缩（仅训练时节省计算，不缓存）：

$$c_t^Q = W_{DQ} \cdot h_t, \quad q_t = W_{UQ} \cdot c_t^Q$$

### 4.3 PyTorch 代码

```python
class MLA(nn.Module):
    def __init__(self, d_model=1024, num_heads=32, d_compress=128):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_compress = d_compress
        
        self.W_dq = nn.Linear(d_model, d_compress, bias=False)
        self.W_uq = nn.Linear(d_compress, num_heads * self.d_k, bias=False)
        self.W_dkv = nn.Linear(d_model, d_compress, bias=False)
        self.W_uk = nn.Linear(d_compress, num_heads * self.d_k, bias=False)
        self.W_uv = nn.Linear(d_compress, num_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)
    
    def forward(self, x):
        B, S, _ = x.shape
        c_q = self.W_dq(x)
        Q = self.W_uq(c_q).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        c_kv = self.W_dkv(x)
        K = self.W_uk(c_kv).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_uv(c_kv).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out), c_kv
```

## 5. DSA（DeepSeek 稀疏注意力）

### 5.1 详细原理

DSA 在 MLA 基础上进一步优化超长上下文场景。核心是**动态稀疏选择**——每层只关注 Top-K 个最相关的 Token，将计算从 $O(n^2)$ 降至 $O(n \cdot k)$。

通过 Lightning Indexer 预索引，在推理时快速检索 Top-K 相关 Token。

### 5.2 工作流程

1. 当前 Token 生成压缩 Q
2. Lightning Indexer 在缓存中检索 Top-K 相关 Token 的索引
3. 只取对应位置的压缩 KV 参与注意力计算
4. 输出经过标准投影层

## 6. 应用场景

| 机制 | 适用场景 |
|------|---------|
| GQA | 通用 LLM 推理，平衡精度与效率 |
| MLA | 超长上下文推理（128K+），KV 缓存敏感场景 |
| DSA | 极长上下文（1M+），需要动态路由的场景 |

## 7. 优缺点分析

| 机制 | 优点 | 缺点 |
|------|------|------|
| GQA | 实现简单，兼容性好，社区支持广 | 压缩比有限，组数选择需调参 |
| MLA | 压缩比极高，精度无损，支持 RoPE 融合 | 实现复杂，不支持 FlashAttention 原生接口 |
| DSA | 极致压缩，突破长度限制 | 依赖索引结构，动态选择有信息损失 |

## 8. 常见问题与易错点

1. **GQA 组数选择**：过小（如 G=1 MQA）精度下降明显，建议 G=4~8
2. **MLA RoPE 兼容**：MLA 的 RoPE 需施加在 Q/K 的额外维度上，不能直接加在压缩向量上
3. **DSA 索引一致性**：训练和推理的索引策略必须一致，否则精度崩塌
4. **FlashAttention 兼容**：GQA 可直接用 FlashAttention 2；MLA 需要定制化适配

## 9. 学习总结

GQA、MLA、DSA 代表了注意力机制优化的三个方向：**减少头数、压缩维度、稀疏选择**。选择建议：通用场景用 GQA（最成熟），长上下文用 MLA（最优压缩），极端长序列考虑 DSA。实际部署中还需考虑 FlashAttention 兼容性和工程实现复杂度。

## 10. 学习路径建议

- **前置知识**：多头注意力机制、Transformer 架构、KV Cache 原理
- **进阶方向**：FlashAttention、RingAttention、MoE + 稀疏注意力
- **推荐论文**：GQA (EMNLP 2023)、DeepSeek V2 (2024)、FlashAttention 2 (2023)
