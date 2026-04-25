# Linear Attention 线性注意力 学习文档

## 1. 算法基础认知

### 1.1 定义

Linear Attention（线性注意力）是一种将标准 Softmax 注意力线性化的技术，其核心思想是使用**核函数近似**来将 $O(N^2)$ 的计算复杂度降为 $O(N)$。标准注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

线性注意力使用核函数 $\phi(\cdot)$ 近似：

$$
\text{LinearAttention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1}}
$$

### 1.2 直观类比

将 Linear Attention 想象为**用望远镜看星星**：传统方法需要比较所有星星对（$N^2$），线性方法用"望远镜"将星光汇聚后直接计算，减少比较次数。

### 1.3 历史背景

- **Linear Attention**（2020）：由 Katharopoulos 等人提出
- **Performer**（2020）：使用随机投影
- **Linear Transformers**（2020）：使用核函数

---

## 2. 核心原理

### 2.1 核函数近似

核心思想：将 softmax 函数用核函数近似：

$$
\text{softmax}(a_i) = \frac{e^{a_i}}{\sum_j e^{a_j}} = \frac{e^{a_i - \text{mlp}(a_j)}}{sum_j e^{a_j - \text{mlp}(a_j)}}
$$

使用 $\phi(x) = e^{frac{x}{2}}$ 近似：

$$
\text{softmax}(q \cdot k) \approx \phi(q)^T \phi(k)
$$

### 2.2 数学形式

$$
\text{Att}(Q, K, V)_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j}{\sum_j \phi(q_i)^T \phi(k_j)}
$$

重写为：

$$
\text{Att}(Q, K, V)_i = \frac{\phi(q_i)^T (sum_j \phi(k_j) v_j^T)}{\phi(q_i)^T (sum_j \phi(k_j))}
$$

### 2.3 特征映射

常用特征映射：

1. **指数映射**：$\phi(x) = \text{elu}(x) + 1$
2. **随机投影**：$\phi(x) = W x$（Performer）
3. **多项式核**：$\phi(x) = [1, x, x^2, ...]$

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $Q$ | 查询矩阵 | $(N, d)$ |
| $K$ | 键矩阵 | $(N, d)$ |
| $V$ | 值矩阵 | $(N, d)$ |
| $\phi$ | 特征映射 | $\mathbb{R}^d \to \mathbb{R}^m$ |

### 3.2 标准注意力

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

复杂度：$O(N^2 d)$

### 3.3 线性注意力

设特征映射 $\phi: \mathbb{R}^d \to \mathbb{R}^m$，则：

$$
S = \sum_j \phi(k_j) v_j^T \in \mathbb{R}^{m \times d}
$$

$$
z = \sum_j \phi(k_j) \in \mathbb{R}^m
$$

$$
\text{Output}_i = \frac{\phi(q_i)^T S}{\phi(q_i)^T z}
$$

复杂度：$O(N d m + N m d) = O(N d m)$

### 3.4 ELU 特征映射

使用 ELU 激活函数：

$$
\phi(x) = \text{ELU}(x) + 1 = \begin{cases} x + 1 & x > 0 \\ e^x - 1 + 1 = e^x & x \leq 0 \end{cases}
$$

---

## 4. 训练过程讲解

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """线性注意力层"""
    
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
    def forward(self, x):
        """
        x: [B, N, D]
        """
        B, N, D = x.shape
        
        # 线性投影
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv  # [B, N, inner_dim]
        
        # 特征映射（ELU + 1）
        phi_k = F.elu(k) + 1
        
        # 计算 S 和 z
        S = torch.einsum('bnd,bnf->bdf', phi_k, v)
        z = phi_k.sum(dim=1)
        
        # 特征映射 q
        phi_q = F.elu(q) + 1
        
        # 计算输出
        out = torch.einsum('bnd,bdf,bn->bnf', phi_q, S, 1.0 / (z + 1e-8))
        
        return out
```

### 4.2 因果掩码（自回归）

```python
class CausalLinearAttention(nn.Module):
    """带因果掩码的线性注意力"""
    
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 缓存
        self.S_cache = None
        self.z_cache = None
        
    def forward(self, x, cache=None):
        """增量推理时使用缓存"""
        B, N, D = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        phi_k = F.elu(k) + 1
        
        # 增量计算
        if cache is None:
            S = torch.einsum('bnd,bnf->bdf', phi_k, v)
            z = phi_k.sum(dim=1)
        else:
            S_old, z_old = cache
            S = S_old + torch.einsum('bnd,bnf->bdf', phi_k, v)
            z = z_old + phi_k
        
        phi_q = F.elu(q) + 1
        out = torch.einsum('bnd,bdf,bn->bnf', phi_q, S, 1.0 / (z + 1e-8))
        
        return out, (S, z)
```

### 4.3 完整 Transformer 块

```python
class LinearTransformerBlock(nn.Module):
    """线性 Transformer 块"""
    
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# 测试
block = LinearTransformerBlock(dim=256, heads=8)
x = torch.randn(2, 100, 256)
y = block(x)
print(f"输入: {x.shape}, 输出: {y.shape}")
```

---

## 5. 应用场景

### 5.1 长序列处理

Linear Attention 的主要优势：

- 处理超长序列（$N > 10000$）
- 降低内存占用
- 加速推理

### 5.2 流式推理

因果线性注意力支持：

- 自回归生成
- 在线学习
- 实时系统

### 5.3 Performer

Google 提出的 Performer 使用：

- **正交随机特征**：更稳定的近似
- **FASTER**：更快的注意力

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 线性复杂度 | $O(N)$ 而非 $O(N^2)$ |
| 内存高效 | 无需存储 $N^2$ 矩阵 |
| 增量推理 | 支持流式处理 |
| 近似保证 | 有理论误差界 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 近似误差 | 对某些模式可能不准 |
| 实现复杂度 | 比标准注意力复杂 |
| 超参数敏感 | 特征映射需调参 |

---

## 7. 调库实现

### 7.1 使用已有库

```python
# 方法1：使用 xformers（Meta 出品）
# pip install xformers
from xformers.ops import memory_efficient_attention

def use_xformers():
    """使用 xformers 的线性注意力"""
    q = torch.randn(2, 8, 100, 64)
    k = torch.randn(2, 8, 100, 64)
    v = torch.randn(2, 8, 100, 64)
    
    output = memory_efficient_attention(q, k, v)
    print(f"输出形状: {output.shape}")

use_xformers()

# 方法2：使用 flash-attn
# from flash_attn import flash_attn_func
# output = flash_attn_func(q, k, v)
```

### 7.2 自定义实现

```python
class EfficientAttention(nn.Module):
    """高效注意力（结合线性和 Flash）"""
    
    def __init__(self, dim, heads=8, dim_head=64, use_linear=True):
        super().__init__()
        self.use_linear = use_linear
        self.heads = heads
        self.dim_head = dim_head
        
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        if use_linear:
            self.feature_map = FeatureMap(dim_head)
        
    def forward(self, x, mask=None):
        B, N, D = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv  # [B, N, inner_dim]
        
        # 重排为多头
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        
        if self.use_linear:
            # 线性注意力
            out = linear_attention(q, k, v)
        else:
            # Flash 注意力
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        return out.transpose(1, 2).contiguous().view(B, N, -1)
```

---

## 8. 手工代码实现

### 8.1 完整线性注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttentionCore:
    """线性注意力核心计算"""
    
    @staticmethod
    def forward(q, k, v):
        """
        q: [B, H, N, D]
        k: [B, H, M, D]
        v: [B, H, M, D]
        """
        B, H, N, D = q.shape
        _, _, M, _ = k.shape
        
        # 特征映射（ELU + 1）
        phi_k = F.elu(k) + 1
        phi_q = F.elu(q) + 1
        
        # 计算上下文
        kv = torch.einsum('bhmd,bhmf->bhdf', phi_k, v)
        k_sum = phi_k.sum(dim=2)
        
        # 计算输出
        out = torch.einsum('bhnd,bhdf,bhm->bhnf', phi_q, kv, 1.0 / (k_sum + 1e-8))
        
        return out
    
    @staticmethod
    def forward_chunked(q, k, v, chunk_size=512):
        """分块处理（内存友好）"""
        B, H, N, D = q.shape
        _, _, M, _ = k.shape
        
        phi_q = F.elu(q) + 1
        out = torch.zeros_like(q)
        
        # 分块处理 k, v
        for i in range(0, M, chunk_size):
            k_chunk = k[:, :, i:i+chunk_size]
            v_chunk = v[:, :, i:i+chunk_size]
            
            phi_k = F.elu(k_chunk) + 1
            
            kv_chunk = torch.einsum('bhmd,bhmf->bhdf', phi_k, v_chunk)
            k_sum_chunk = phi_k.sum(dim=2)
            
            out += torch.einsum('bhnd,bhdf,bhm->bhnf', phi_q, kv_chunk, 1.0)
        
        # 归一化
        phi_k_full = F.elu(k) + 1
        k_sum_full = phi_k_full.sum(dim=2)
        out = out * (1.0 / (k_sum_full + 1e-8))
        
        return out

# 验证
q = torch.randn(2, 8, 100, 64)
k = torch.randn(2, 8, 100, 64)
v = torch.randn(2, 8, 100, 64)

out = LinearAttentionCore.forward(q, k, v)
print(f"输出形状: {out.shape}")
```

### 8.2 增量推理版本

```python
class IncrementalLinearAttention:
    """增量推理的线性注意力"""
    
    def __init__(self):
        self.S = None
        self.z = None
    
    def step(self, q_i, k_i, v_i):
        """单步推理
        
        q_i: [B, H, 1, D]
        k_i: [B, H, 1, D]
        v_i: [B, H, 1, D]
        """
        # 特征映射
        phi_k = F.elu(k_i) + 1
        phi_q = F.elu(q_i) + 1
        
        if self.S is None:
            self.S = phi_k.transpose(2, 3) @ v_i
            self.z = phi_k.sum(dim=2)
        else:
            self.S = self.S + phi_k.transpose(2, 3) @ v_i
            self.z = self.z + phi_k.squeeze(2)
        
        # 输出
        out_i = (phi_q @ self.S) / (self.z + 1e-8)
        
        return out_i
    
    def reset(self):
        """重置状态"""
        self.S = None
        self.z = None

# 测试增量推理
inc_attn = IncrementalLinearAttention()

for step in range(5):
    q = torch.randn(2, 8, 1, 64)
    k = torch.randn(2, 8, 1, 64)
    v = torch.randn(2, 8, 1, 64)
    
    out = inc_attn.step(q, k, v)
    print(f"Step {step}: output shape = {out.shape}")
```

---

## 9. 可视化与结果理解

### 9.1 复杂度对比

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_complexity():
    """可视化复杂度对比"""
    
    seq_lens = [100, 1000, 5000, 10000, 50000]
    
    # 标准注意力：O(N^2)
    standard = [n**2 for n in seq_lens]
    
    # 线性注意力：O(N)
    linear = [n for n in seq_lens]
    
    plt.figure(figsize=(10, 5))
    plt.plot(seq_lens, standard, 'r-', label='Standard (O(N²))', linewidth=2)
    plt.plot(seq_lens, linear, 'b-', label='Linear (O(N))', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Computations')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Attention Complexity Comparison')
    plt.savefig('complexity.png', dpi=150)
    plt.show()

plot_complexity()
```

### 9.2 质量对比

```python
def compare_quality():
    """对比注意力质量"""
    
    torch.manual_seed(42)
    
    # 创建测试数据
    d = 64
    n = 100
    
    q = torch.randn(1, n, d)
    k = torch.randn(1, n, d)
    v = torch.randn(1, n, d)
    
    # 标准注意力
    scores_std = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(d), dim=-1)
    out_std = scores_std @ v
    
    # 线性注意力
    phi_k = F.elu(k) + 1
    phi_q = F.elu(q) + 1
    S = phi_k.transpose(-2, -1) @ v
    z = phi_k.sum(dim=-2)
    out_lin = (phi_q @ S) / z
    
    diff = (out_std - out_lin).abs().mean()
    print(f"平均差异: {diff:.4f}")
    
    return diff

compare_quality()
```

---

## 10. 模型评估

### 10.1 质量指标

```python
def evaluate_linear_attention():
    """评估线性注意力"""
    
    test_cases = [
        {'name': '短序列', 'n': 100},
        {'name': '长序列', 'n': 5000},
        {'name': '超长序列', 'n': 50000},
    ]
    
    for tc in test_cases:
        print(f"处理 {tc['name']} (N={tc['n']})...")
        # 实际测试略
    
    print("评估就绪")

evaluate_linear_attention()
```

---

## 11. 常见问题与易错点

### 11.1 特征映射选择

**问题**：哪个特征映射最好？

**解答**：ELU + 1 是最常用的选择，有理论保证。

### 11.2 数值稳定性

**问题**：出现 nan？

**解答**：添加 epsilon，归一化时 $z + 1e-8$。

### 11.3 分块大小

**问题**：如何选择 chunk_size？

**解答**：根据显存调整，通常 512-1024。

---

## 12. 学习总结

### 12.1 核心要点

1. **核近似**：用 $\phi$ 近似 softmax
2. **复杂度**：$O(N^2) \to O(N)$
3. **特征映射**：ELU + 1
4. **增量推理**：缓存 S 和 z

### 12.2 与 Flash Attention 对比

| 方法 | 复杂度 | 近似 | 增量推理 |
|------|--------|------|----------|
| Standard | $O(N^2)$ | 无 | 否 |
| Linear | $O(N)$ | 核 | 是 |
| Flash | $O(N)$ | I/O | 否 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：推导线性注意力的时间复杂度。

**解答**：$O(N d^2)$ 或 $O(N)$（使用缓存）。

### 13.2 思考题

**思考题**：线性注意力何时失效？

**解答**：当注意力模式高度稀疏或��强��角倾向时。

---

## 14. 学习路径建议

### 14.1 第一阶段（1 天）

1. 理解标准注意力
2. 理解核近似思想

### 14.2 第二阶段（2 天）

1. 实现 Linear Attention
2. 理解增量推理

### 14.3 第三阶段（3 天）

1. 对比不同实现
2. 实际应用

### 14.4 推荐资源

- **论文**：《Linear Transformers Are Secretly Fast Transformers》
- **代码**：xformers

---

*Linear Attention 是处理长序列的重要技术，它将二次复杂度降为线性，使得处理超长序列成为可能。*