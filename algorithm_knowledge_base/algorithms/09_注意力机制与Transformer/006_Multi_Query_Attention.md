# Multi Query Attention 学习文档

## 1. 算法基础认知

### 1.1 定义

Multi-Query Attention（多查询注意力）是标准多头注意力的一种变体，由 Shazeer 等人在 2019 年提出。其核心创新是：**多个查询头共享同一组键（Key）和值（Value）**，从而显著减少内存和计算开销。

标准多头注意力：
- $N$ 个查询头：$Q_1, Q_2, ..., Q_N$
- $N$ 组键/值：$K_1, V_1, ..., K_N, V_N$

多查询注意力：
- $N$ 个查询头：$Q_1, Q_2, ..., Q_N$
- 1 组键/值：$K, V$（所有查询头共享）

### 1.2 直观类比

将 Multi-Query Attention 想象为**会议室讨论**：多个问题（查询）同时讨论，但只准备一份参考资料（键/值），只是从不同角度（投影）提出问题。

### 1.3 历史背景

- **Multi-Head Attention**（2017）：Transformer 论文提出
- **Multi-Query Attention**（2019）：为推理加速设计
- **Grouped-Query Attention**（2023）：LLaMA2 等使用

---

## 2. 核心原理

### 2.1 计算对比

| 方面 | MHA | MQA |
|------|-----|-----|
| Key/Value 头数 | N | 1 |
| 推理 KV 缓存 | $N \times L \times D$ | $L \times D$ |
| 参数量 | 较多 | 较少 |
| 效果 | 略好 | 可接受 |

### 2.2 数学形式

MHA：
$$
\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$
$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_N) W^O
$$

MQA：
$$
\text{head}_i = \text{Attention}(Q_i W_i^Q, K W^K, V W^V)
$$

### 2.3 内存节省

设序列长度为 $L$，隐藏维度为 $D$，头数为 $N$：
- **MHA**：KV 缓存需要 $2 \times N \times L \times D$ 元素
- **MQA**：KV 缓存只需 $2 \times L \times D$ 元素

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $Q$ | 查询矩阵 | $(B, N, L, D)$ |
| $K$ | 键矩阵（共享） | $(B, 1, L, D)$ |
| $V$ | 值矩阵（共享） | $(B, 1, L, D)$ |
| $N$ | 查询头数 | - |
| $D$ | 头维度 | - |

### 3.2 前向传播

```python
def multi_query_attention(Q, K, V):
    """多查询注意力
    
    Q: [B, N, L, D]
    K: [B, 1, L, D]
    V: [B, 1, L, D]
    """
    B, N, L, D = Q.shape
    
    # 缩放点积
    scores = torch.einsum('bnld,bld->bnl', Q, K.squeeze(1)) / np.sqrt(D)
    
    # Softmax
    attn = F.softmax(scores, dim=-1)
    
    # 加权
    out = torch.einsum('bnl,bld->bnld', attn, V.squeeze(1))
    
    return out
```

### 3.3 复杂度分析

| 操作 | MHA | MQA |
|------|-----|-----|
| QKV 投影 | $O(B L N D^2)$ | $O(B L N D^2 + B L D^2)$ |
| 注意力计算 | $O(B N L^2 D)$ | $O(B N L^2 D)$ |
| KV 缓存 | $O(N L D)$ | $O(L D)$ |

---

## 4. 训练过程讲解

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """多查询注意力"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 查询投影（多头）
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        
        # 键值投影（单头）
        self.W_k = nn.Linear(hidden_dim, self.head_dim)
        self.W_v = nn.Linear(hidden_dim, self.head_dim)
        
        # 输出投影
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, past_kv=None, use_cache=False):
        """
        x: [B, L, D]
        past_kv: (past_K, past_V) 缓存
        """
        B, L, D = x.shape
        
        # 查询投影
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [B, N, L, D]
        
        # 键值投影（共享）
        K = self.W_k(x).unsqueeze(1)  # [B, 1, L, D]
        V = self.W_v(x).unsqueeze(1)  # [B, 1, L, D]
        
        # 缓存
        if use_cache and past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)
        
        if use_cache:
            cache = (K, V)
        else:
            cache = None
        
        # 注意力
        scores = torch.einsum('bnld,bld->bnl', Q, K.squeeze(1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bnl,bld->bnld', attn, V.squeeze(1))
        
        # 输出
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)
        
        return out, cache
```

### 4.2 Transformer 层

```python
class TransformerBlock(nn.Module):
    """Transformer 块（使用 MQA）"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiQueryAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, past_kv=None):
        # 注意力
        attn_out, cache = self.attn(self.norm1(x), past_kv)
        x = x + attn_out
        
        # FFN
        x = x + self.ff(self.norm2(x))
        
        return x, cache

# 测试
block = TransformerBlock(hidden_dim=512, num_heads=8)
x = torch.randn(2, 100, 512)
out, cache = block(x)
print(f"输入: {x.shape}, 输出: {out.shape}")
print(f"KV 缓存: {cache[0].shape if cache else None}")
```

### 4.3 训练循环

```python
def train_with_mqa():
    """使用 MQA 训练"""
    
    # 模型
    model = nn.Sequential(
        *[TransformerBlock(256, 8) for _ in range(6)]
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    for epoch in range(10):
        for batch in dataloader:
            x = batch['input']
            
            optimizer.zero_grad()
            out, _ = model(x)
            
            loss = F.cross_entropy(out.view(-1, vocab_size), x.view(-1))
            loss.backward()
            optimizer.step()

train_with_mqa()
```

---

## 5. 应用场景

### 5.1 大语言模型推理

Multi-Query Attention 的主要应用：
- LLaMA2（使用 Grouped-Query Attention）
- Falcon
- PaLM 2

### 5.2 长序列生成

- 代码生成
- 文档续写
- 对话系统

### 5.3 内存受限场景

- 边缘设备部署
- 移动端推理

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 内存节省 | KV 缓存大幅减少 |
| 推理加速 | 吞吐量提升 |
| 效果可接受 | 性能损失小 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 表达能力降 | 键值共享限制 |
| 训练速度 | 与 MHA 相近 |
| 调参 | 需平衡头数 |

---

## 7. 调库实现

### 7.1 使用现有库

```python
# 使用 xformers
# pip install xformers
from xformers.ops import memory_efficient_attention

def use_xformers_mqa():
    """xformers 实现"""
    
    B, N, L, D = 2, 8, 100, 64
    
    Q = torch.randn(B, N, L, D)
    K = torch.randn(B, 1, L, D)
    V = torch.randn(B, 1, L, D)
    
    # 扩展为多头（重复）
    K = K.expand(-1, N, -1, -1)
    V = V.expand(-1, N, -1, -1)
    
    out = memory_efficient_attention(Q, K, V)
    print(f"输出: {out.shape}")

use_xformers_mqa()
```

### 7.2 Hugging Face 实现

```python
from transformers import AutoConfig, AutoModel

def use_huggingface():
    """使用 Hugging Face 配置"""
    
    # 配置 MQA
    config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_key_value_heads': 1,  # MQA
    }
    
    print(f"查询头: {config['num_attention_heads']}")
    print(f"键值头: {config['num_key_value_heads']}")
    
    return config

use_huggingface()
```

---

## 8. 手工代码实现

### 8.1 完整 MQA 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualMultiQueryAttention(nn.Module):
    """手动实现多查询注意力"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # 投影矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, self.head_dim)
        self.W_v = nn.Linear(hidden_dim, self.head_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # 查询：多头投影
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [B, N, L, D]
        
        # 键值：单头投影（广播）
        K = self.W_k(x).unsqueeze(1)  # [B, 1, L, D]
        V = self.W_v(x).unsqueeze(1)  # [B, 1, L, D]
        
        # 注意力分数
        scores = torch.einsum('bnld,bld->bnl', Q, K.squeeze(1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        
        # 加权
        out = torch.einsum('bnl,bld->bnld', attn, V.squeeze(1))
        
        # 输出
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)
        
        return out

# 验证
mqa = ManualMultiQueryAttention(256, 8)
x = torch.randn(2, 100, 256)
out = mqa(x)
print(f"输入: {x.shape}, 输出: {out.shape}")
```

### 8.2 增量推理版本

```python
class IncrementalMQA:
    """增量推理的 MQA"""
    
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 缓存
        self.K_cache = []
        self.V_cache = []
    
    def step(self, x_t, W_q, W_k, W_v):
        """单步推理
        
        x_t: [B, 1, D]
        """
        # 投影
        q_t = W_q(x_t).view(-1, self.num_heads, self.head_dim)  # [B, N, D]
        k_t = W_k(x_t).squeeze(1)  # [B, D]
        v_t = W_v(x_t).squeeze(1)  # [B, D]
        
        # 缓存
        self.K_cache.append(k_t)
        self.V_cache.append(v_t)
        
        K = torch.stack(self.K_cache, dim=1)  # [B, L, D]
        V = torch.stack(self.V_cache, dim=1)
        
        # 注意力
        scores = torch.einsum('bnd,bld->bnl', q_t, K) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.einsum('bnl,bld->bnd', attn, V)
        
        return out
    
    def reset(self):
        """重置缓存"""
        self.K_cache = []
        self.V_cache = []

# 测试
inc_mqa = IncrementalMQA(256, 8)

for step in range(5):
    x_t = torch.randn(1, 1, 256)
    # 模拟投影（实际应从模型获取）
    W_q = torch.randn(256, 256)
    W_k = torch.randn(256, 64)
    W_v = torch.randn(256, 64)
    
    out = inc_mqa.step(x_t, W_q, W_k, W_v)
    print(f"Step {step}: {out.shape}")
```

---

## 9. 可视化与结果理解

### 9.1 缓存大小对比

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_cache_comparison():
    """可视化 KV 缓存大小对比"""
    
    seq_lens = [512, 1024, 2048, 4096, 8192]
    hidden_dim = 512
    num_heads = 8
    
    # MHA
    mha_cache = [2 * num_heads * L * hidden_dim for L in seq_lens]
    
    # MQA
    mqa_cache = [2 * 1 * L * hidden_dim for L in seq_lens]
    
    plt.figure(figsize=(10, 5))
    plt.plot(seq_lens, mha_cache, 'r-', label='MHA', linewidth=2)
    plt.plot(seq_lens, mqa_cache, 'b-', label='MQA', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('KV Cache Size')
    plt.yscale('log')
    plt.legend()
    plt.title('KV Cache Size Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig('cache_comparison.png', dpi=150)
    plt.show()

plot_cache_comparison()
```

### 9.2 头注意力可视化

```python
def visualize_attention():
    """可视化注意力"""
    
    # 模拟注意力权重
    attn = np.random.rand(8, 10)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Head')
    plt.title('Multi-Query Attention Pattern')
    plt.savefig('attention_pattern.png', dpi=150)
    plt.show()

visualize_attention()
```

---

## 10. 模型评估

### 10.1 质量指标

```python
def evaluate_mqa():
    """评估 MQA 模型"""
    
    metrics = {
        'Perplexity': 15.2,
        'Accuracy': 0.75,
        'Throughput': 1.8,
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    return metrics

evaluate_mqa()
```

---

## 11. 常见问题与易错点

### 11.1 头数设置

**问题**：查询头和键值头的比例？

**解答**：MQA 时 $N_{kv} = 1$，GQA 时 $N_{kv} = N / 4$。

### 11.2 缓存管理

**问题**：长序列缓存过大？

**解答**：使用 KV 量化或缓存替换策略。

---

## 12. 学习总结

### 12.1 核心要点

1. **键值共享**：多个查询头共享一组 KV
2. **内存节省**：缓存从 $N \times L \times D$ 减到 $L \times D$
3. **推理加速**：吞吐量提升显著
4. **效果损失**：可接受范围内

### 12.2 变体

| 方法 | KV 头数 | 特点 |
|------|---------|------|
| MHA | N | 标准 |
| MQA | 1 | 最省内存 |
| GQA | N/4 | 平衡 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：实现 Multi-Query Attention。

### 13.2 思考题

**思考题**：MQA 何时效果下降明显？

---

## 14. 学习路径建议

### 14.1 第一阶段

1. 理解标准注意力
2. 理解 MQA 原理

### 14.2 第二阶段

1. 实现 MQA
2. 实现增量推理

### 14.3 第三阶段

1. 部署推理
2. 对比 GQA

### 14.4 推荐资源

- **论文**：《Fast Transformer Decoding》
- **代码**：LLaMA2

---

*Multi-Query Attention 是大模型推理加速的关键技术，它在效果和效率之间取得了很好的平衡。*