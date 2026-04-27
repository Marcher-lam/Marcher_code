# KV Cache 键值缓存 学习文档

> 缓存注意力计算中的中间结果，避免重复计算，将推理速度提升5-10倍。

> 来源线索：本节内容根据原书第2章2.8节关于KV caching的讲解整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
KV Cache是一种在自回归生成中存储并复用注意力键值对以避免冗余计算的技术。

### 直觉类比
你做一道很长的数学题，每一步都写下来。下次你不需要重新从头计算——直接查看之前写下的中间结果，只在最后一步上加新的计算。KV Cache就是"记下之前算好的中间结果"。

### 历史背景
KV Cache是随着Transformer自回归生成在2018-2019年被广泛意识到并采用的优化技术。它不是被单独提出的算法，而是Transformer架构在推理场景下的自然优化——当self-attention的输入在每一步只增加一个新token时，之前token的key和value不需要重新计算。

### 算法定位
- **类型**：推理优化 / 解码加速 / 缓存技术
- **性质**：推理时优化，不修改模型权重

### 前置知识
- 理解自回归文本生成的过程
- 了解Transformer的注意力机制基础（Q、K、V）
- 了解LLM内部的前向传播

## 2. 核心原理

### 核心思想
在自回归生成中，每次迭代只新增一个token。前一次迭代已经为所有"旧token"计算了key和value。KV Cache将这些key和value存储起来，下一次迭代时，只需：
1. 为新增的这一个token计算Q、K、V
2. 使用缓存中的旧K、V + 新计算的K、V进行注意力计算

这样避免了O(L²)的冗余计算（L为当前序列长度）。

### 工作流程
1. 第一次迭代：完整输入prompt，为所有token计算K和V，存入cache
2. 后续迭代：只输入上一个生成的新token
   - 只计算这一个token的Q、K、V
   - 从cache取出所有旧token的K、V
   - 用新token的Q对所有token的K、V做注意力
   - 新token的K、V追加到cache中
3. 直到生成结束

### 关键概念解释
- **K (Key)**：注意力机制中用于匹配的向量，决定了"哪些token与当前token相关"
- **V (Value)**：注意力机制中实际被聚合的向量，代表了"每个token携带的信息"
- **Cache命中**：直接从缓存获取先前计算结果，不需要重新计算
- **自注意力中的冗余**：没有KV cache时，每一步都要对全部token重新计算K和V

### 直观解释
```
没有KV Cache (浪费):
Step 1: 计算全部6个token的K,V → 生成token7
Step 2: 重新计算全部7个token的K,V → 生成token8  ← 前6个白算了!
Step 3: 重新计算全部8个token的K,V → 生成token9  ← 前7个白算了!

有KV Cache (高效):
Step 1: 计算全部6个token的K,V → 存入cache → 生成token7
Step 2: 只计算token7的K,V → 从cache取前6个 → 拼接→ 生成token8
Step 3: 只计算token8的K,V → 从cache取前7个 → 拼接→ 生成token9
  每次只算1个新token的K,V!
```

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $x_1, ..., x_n$ | 输入tokens的embedding |
| $Q_t, K_t, V_t$ | token t的query、key、value向量 |
| $W_Q, W_K, W_V$ | Q、K、V的投影矩阵 |
| $d_k$ | key的维度 |
| $\text{Cache}_K, \text{Cache}_V$ | 缓存的K、V |

### 标准注意力（无缓存）

对于序列 $X = [x_1, ..., x_L]$，对每个位置计算：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 带缓存的注意力

在第 $t$ 次迭代时（总长度=L）：

新token的嵌入：$x_L$

只计算新token:
$$K_{\text{new}} = x_L W_K, \quad V_{\text{new}} = x_L W_V$$
$$Q_{\text{new}} = x_L W_Q$$

拼接缓存和新值:
$$K = [\text{Cache}_K; K_{\text{new}}], \quad V = [\text{Cache}_V; V_{\text{new}}]$$

只在最后一个位置计算注意力输出（注意：Q只有一行）:

$$\text{Attention}_{\text{last}} = \text{softmax}\left(\frac{Q_{\text{new}} K^T}{\sqrt{d_k}}\right)V$$

### 计算复杂度对比

| 方法 | 每次迭代的K/V计算 | 注意力计算 | 总复杂度(每token) |
|------|--------------------|-----------|-------------------|
| 无Cache | O(L × d²) | O(L² × d) | O(L × d²) |
| 有Cache | O(1 × d²) | O(L × d) | O(d²) |

d = 模型维度, L = 当前序列长度

## 4. 训练过程讲解

KV Cache是纯推理优化，不需要训练。

### 使用条件
- 模型必须在自回归（因果）模式下使用
- 模型需要支持增量式的KV Cache传入（在代码中预留cache参数）
- 预填充阶段(prompt)需要完整前向传播

### 实现要点
1. Cache应该在第一次前向传播时初始化（空或None）
2. 预填充：所有prompt token一次性计算K、V并缓存
3. 解码：后续每次只输入1个新token
4. reset_kv_cache()用于开始新序列时清空缓存

### 内存分析
KV Cache 的内存占用随序列长度线性增长：每个transformer层都需要存储L个K、L个V向量（每个维度d_head × n_kv_heads）。这是长序列推理的主要内存瓶颈之一（比模型权重还大）。

## 5. 应用场景

所有涉及LLM自回归文本生成的场景都应使用KV Cache：对话、翻译、代码生成、推理模型输出、流式输出等。没有KV Cache的话，生成速度会慢5-50倍（取决于序列长度）。

## 6. 优缺点分析

### 优点
| 优点 | 说明 |
|------|------|
| 大幅加速 | 原书实测5→29 tokens/sec (CPU, 5.8倍) |
| 结果不变 | 输出与无缓存版本完全相同（无损优化） |
| 实现相对简单 | 只需要改推理过程中的K、V的计算方式 |

### 缺点
| 缺点 | 说明 |
|------|------|
| 内存增长 | 长序列时Cache内存可超过模型参数 |
| 批处理困难 | 不同请求长度不同时难以高效批处理 |
| 不能用于训练 | 训练时仍需要完整序列计算 |

## 7. 调库实现

```python
"""
KV Cache 的实用实现
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()


@torch.inference_mode()
def generate_with_kv_cache(prompt, max_new_tokens=100):
    """
    使用HuggingFace的KV cache生成本文

    use_cache=True 自动启用KV cache
    HuggingFace的generate已经内置了KV cache支持
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,       # 启用KV Cache（默认就是True）
        do_sample=False,      # 贪婪解码（更快）
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


prompt = "Explain what is KV cache in one sentence:"
import time
t0 = time.time()
result = generate_with_kv_cache(prompt, 50)
t = time.time() - t0
print(f"生成: {result[:100]}...")
print(f"耗时: {t:.2f}秒, 速度: {50/max(t,0.01):.0f} tokens/sec")
```

## 8. 手工代码实现

```python
"""
KV Cache的手工实现
展示内存重用的核心逻辑
"""

import torch
import torch.nn as nn


class KVCache:
    """手工KV Cache"""

    def __init__(self, n_layers, n_kv_heads, head_dim, max_seq_len, device="cpu"):
        """
        初始化KV Cache存储

        存储空间: n_layers × 2(K+V) × n_kv_heads × max_seq_len × head_dim
        """
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # 为每层分配 cache 内存（预先分配以提高效率）
        self.k_cache = torch.zeros(
            n_layers, max_seq_len, n_kv_heads, head_dim,
            device=device
        )
        self.v_cache = torch.zeros(
            n_layers, max_seq_len, n_kv_heads, head_dim,
            device=device
        )
        self.current_length = 0  # 当前缓存的有效长度

    def update(self, layer_idx, k_new, v_new):
        """
        将新token的K、V写入缓存

        参数:
            k_new, v_new: 形状 (1, n_kv_heads, head_dim)
        """
        pos = self.current_length
        self.k_cache[layer_idx, pos] = k_new[0]  # 去掉batch维度
        self.v_cache[layer_idx, pos] = v_new[0]

    def get_valid(self, layer_idx):
        """
        获取该层的所有有效的K、V（自第0位到current_length）

        返回: (K, V) 形状 (1, current_length, n_kv_heads, head_dim)
        """
        end = self.current_length
        k = self.k_cache[layer_idx, :end].unsqueeze(0)  # 加入batch维度
        v = self.v_cache[layer_idx, :end].unsqueeze(0)
        return k, v

    def advance(self):
        """推进缓存位置（新生成一个token后调用）"""
        self.current_length += 1

    def reset(self):
        """重置缓存"""
        self.current_length = 0


def demo_no_cache_vs_cache():
    """
    演示KV cache的实际效果

    模拟一个简化的自注意力计算：
    - 无缓存：每次都要计算全序列的K、V
    - 有缓存：只计算新token的K、V
    """
    print("=== KV Cache 效率演示 ===")
    print("模拟: embedding_dim=64, n_heads=4, 线性投影 64→64")

    # 参数
    emb_dim = 64
    num_heads = 4
    head_dim = 16
    n_kv_heads = 2  # GQA: 4个query heads共享2个KV heads
    n_layers = 28
    seq_len = 100  # 初始prompt长度
    num_new_tokens = 100  # 生成100个新token

    # OPS 估算（简化模型）
    ops_per_kv = emb_dim * n_kv_heads * head_dim  # 投影计算量
    ops_per_attn_per_pair = head_dim  # 注意力中每对K-V的计算量(简化)

    # 无缓存
    ops_no_cache = 0
    for step in range(num_new_tokens):
        L = seq_len + step + 1  # 当前总长度
        ops_no_cache += L * ops_per_kv  # 为全序列计算K、V
        ops_no_cache += L * head_dim

    # 有缓存
    ops_with_cache = 0
    for step in range(num_new_tokens):
        L = seq_len + step + 1
        ops_with_cache += 1 * ops_per_kv  # 只计算1个新token的K、V
        ops_with_cache += L * head_dim  # 注意力仍需全序列

    speedup = ops_no_cache / ops_with_cache
    print(f"无缓存计算量: {ops_no_cache:,} ops")
    print(f"有缓存计算量: {ops_with_cache:,} ops")
    print(f"加速比: {speedup:.1f}×")


demo_no_cache_vs_cache()
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

# 不同序列长度下KV Cache的加速效果
seq_lens = [50, 100, 200, 500, 1000, 2000, 4000]
speedups = [2.1, 3.5, 5.8, 11.2, 18.5, 28.0, 42.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(seq_lens, speedups, 'b-o', linewidth=2, markersize=10)
ax1.set_xlabel("序列长度", fontsize=12, fontweight="bold")
ax1.set_ylabel("加速比", fontsize=12, fontweight="bold")
ax1.set_title("KV Cache 加速比随序列长度增长\n序列越长优势越大", fontsize=13, fontweight="bold")
ax1.grid(alpha=0.3)

# 内存占用
ax2.plot(seq_lens, [s*28*2*8*128/1024/1024 for s in seq_lens], 'r-s', linewidth=2)
ax2.set_xlabel("序列长度", fontsize=12, fontweight="bold")
ax2.set_ylabel("KV Cache 内存 (MB)", fontsize=12, fontweight="bold")
ax2.set_title("KV Cache 内存与序列长度的关系\n(Qwen3 0.6B: 28层 × 8 KV头 × 128维)", fontsize=12, fontweight="bold")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
解读：
左图：加速比随序列长度近似线性增长——序列越长，无缓存的
      重复计算开销越大，缓存收益越高。
右图：内存也是线性增长——在长序列生成中可达数百MB。
      这是vLLM等框架引入PagedAttention来管理的核心原因。
""")
```

## 10. 模型评估

KV Cache本身不是模型，评估方法是比较有无缓存时的：
1. **生成速度**：tokens/sec的比值
2. **结果一致性**：有缓存和无缓存的输出是否一致（应完全一致）
3. **内存占用**：不同序列长度下的内存增长

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 缓存未重置 | 跨不同prompt时输出混乱 | 前一个对话的缓存残留 | 每次新对话前调用reset_kv_cache() |
| 缓存导致内存爆炸 | OOM in long generation | 序列太长，缓存内存过大 | 使用PagedAttention或限制max_length |
| 停用缓存后发现生成结果不同 | 怀疑缓存实现有bug | 缓存实现不正确（位置编码/掩码问题） | 用无缓存版本验证输出一致性 |

## 12. 学习总结

KV Cache是LLM推理的最重要的性能优化之一——通过缓存键值对来避免重复计算，在原书中将CPU上的生成速度从5 tokens/s提升到29 tokens/s（5.8×），配合torch.compile可达68 tokens/s。其核心机制是：在自回归生成中，每个旧token的K、V跨迭代不变，无需重新计算。

## 13. 练习题与思考题

**题1**：为什么KV Cache不能用于训练？

**参考答案**：训练时所有输入是已知的整个序列，可以并行计算所有位置的输出。不需要缓存中间状态——整个序列一次性送入模型。KV Cache专为推理中"每次只多一个token"的串行设计。

**题2**：为什么长序列生成中KV Cache内存会超过模型参数？

**参考答案**：以Qwen3 0.6B为例，模型参数约1.2GB(bf16)。在长度4096的序列上，KV Cache = 28层 × 2(K+V) × 8个KV头 × 128维 × 4096位置 × 2字节 = 约470MB。到了长度32768时，Cache ≈ 3.7GB，已远超模型参数。这在实际推理服务中是PagedAttention技术需要解决的核心问题。

## 14. 学习路径建议

- **前置**：自回归文本生成、Transformer注意力机制基础
- **进阶**：PagedAttention(vLLM)、Continuous Batching、Flash Attention
