# RoPE 旋转位置编码 学习文档

> 通过旋转向量编码位置信息，优雅地解决了Transformer中位置感知的问题。

> 来源线索：本节内容根据原书附录C C.3节关于RoPE的讲解整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
RoPE (Rotary Position Embedding) 通过旋转注意力中的 query 和 key 向量来编码 token 的相对位置。

### 直觉类比
想象一群人在一个圆圈上按顺序站好。每个人看别人的"角度"不同——站在第1位的人看第3位是前方两步，看第5位是前方四步。RoPE就是给每个位置的向量"旋转一个角度"，让不同位置的向量自然地区分开。

### 历史背景
RoPE由苏剑林等人于 2021 年提出。2023 年 Llama 模型采用了RoPE，使该方法成为现代LLM的标配位置编码。几乎所有最新的LLM（Llama、Mistral、Qwen、DeepSeek等）都使用 RoPE。

### 算法定位
- **类型**：位置编码 / 模型架构组件
- **性质**：模型注意力机制的一部分，训练和推理都使用

### 前置知识
- 了解为什么Transformer需要位置编码
- 了解向量旋转的几何直觉（cos、sin）
- 了解注意力机制中Q和K的作用

## 2. 核心原理

### 核心思想
RoPE的核心创新是：不在输入上添加位置信息，而是在注意力计算中对Q和K向量进行"位置相关的旋转"。这种旋转的数学性质导致了两个位置token的点积只依赖于它们的相对距离，与绝对位置无关——这是RoPE最优雅的性质。

### 关键数学性质
RoPE使得：$\langle \text{RoPE}(q_m, m), \text{RoPE}(k_n, n) \rangle = f(q_m, k_n, m-n)$

即两个位置m和n的Q、K内积只依赖其"相对距离" m-n，不依赖绝对位置m和n。

### 工作流程
1. 为每个位置计算旋转角度 $\theta_i = \text{base}^{-2i/d}$
2. 按位置将每个head_dim/2对维度旋转不同角度
3. 在注意力前对Q和K分别应用旋转
4. V不受影响（RoPE只旋转Q和K）

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $d$ | head_dim |
| $\theta_{\text{base}}$ | 旋转基频（通常10000或1000000） |
| $m, n$ | token的绝对位置 |
| $q_m, k_n$ | 位置m的query、位置n的key |

### 频率计算
$$\Theta = \left\{\theta_i = \theta_{\text{base}}^{-2i/d} \mid i = 0, 1, ..., d/2-1\right\}$$

Qwen3 0.6B 使用 $\theta_{\text{base}} = 1,000,000$（远大于标准10000，增强长距离依赖）。

### 旋转操作

对每一对维度 $(x_{2i}, x_{2i+1})$，按角度 $m \cdot \theta_i$ 旋转：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

### 实现方式

原书采用"两半式"实现（更易读）：
```python
x1, x2 = x[..., :d//2], x[..., d//2:]
rotated = torch.cat([-x2, x1], dim=-1)  # 90°旋转
x_rope = x * cos + rotated * sin
```

数学上等价于上述旋转矩阵，但实现更简单——将向量分为两半，一半做cos缩放，另一半做sin旋转。

## 4. 训练过程讲解

RoPE不需要单独训练（它没有可学习参数），关键配置是基频 $\theta_{\text{base}}$：

- $\theta_{\text{base}}$ 越大，高频旋转越少 → 长距离衰减越慢 → 更适合处理长序列
- Qwen3 0.6B: $\theta_{\text{base}} = 1,000,000$ → 支持 40960 的上下文
- 原始RoPE: $\theta_{\text{base}} = 10,000$

## 5. 应用场景

所有现代Transformer LLM中的注意力位置编码。对需要长上下文处理的应用尤其重要。

## 6. 优缺点分析

| 优点 | 说明 |
|------|------|
| 相对位置感知 | 内积依赖相对距离，符合自然语言直觉 |
| 外推性好 | 可以处理比训练时更长的序列 |
| 无额外参数 | 纯数学操作，不增加参数量 |
| 理论优雅 | 旋转群的性质自然导出相对位置 |

| 缺点 | 说明 |
|------|------|
| 数学复杂 | 比添加position embedding更难理解 |
| 基频需调参 | 不同模型大小/任务需要不同θ_base |
| V不参与旋转 | V向量的位置信息仅间接通过注意力传递 |

## 7. 调库实现
```python
# PyTorch 2.6+ 尚未内置RoPE，可用第三方库
# pip install rotary-embedding-torch
from rotary_embedding_torch import RotaryEmbedding
rope = RotaryEmbedding(dim=128)
```

## 8. 手工代码实现

```python
"""RoPE手工实现"""
import torch

def compute_rope_params(head_dim, theta_base=1000000, context_length=4096, dtype=torch.float32):
    """预计算cos和sin表"""
    assert head_dim % 2 == 0
    # 频率: 1/(theta^(2i/d))
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype)[:head_dim//2].float() / head_dim
    ))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]  # (L, d/2)
    angles = torch.cat([angles, angles], dim=1)       # (L, d): 每对维度共享角度
    cos, sin = torch.cos(angles), torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    """
    应用RoPE旋转: 两半式实现
    x: (batch, num_heads, seq_len, head_dim)
    """
    seq_len = x.shape[-2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    cos = cos[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)  # 几何: 90度旋转
    return (x * cos + rotated * sin).to(x.dtype)

# 测试
head_dim = 128
cos, sin = compute_rope_params(head_dim)
x = torch.randn(1, 4, 10, head_dim)
y = apply_rope(x, cos, sin)
print(f"RoPE: input {x.shape} → output {y.shape}")
```

## 9-14. 总结、问题、练习、路径

### 常见问题
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 长序列性能下降 | >4096 token后PPL上升 | θ_base太小 | 增大θ_base到1e6 |

### 学习总结
RoPE通过旋转Q和K向量编码相对位置——它的内积只依赖相对距离而非绝对位置，使模型自然地处理长序列和序列外推。

### 练习题
**题1**：为什么RoPE只旋转Q和K，不旋转V？

**参考答案**：注意力分数(Q·K)需要感知位置来判断"这个token与那个token的距离"。而V是注意力加权后的信息聚合——位置感知已经在softmax(QK^T)的权重中体现了，V只需要承载"内容信息"，不需要额外的位置旋转。旋转V还会破坏内容的语义方向。

### 学习路径
- **前置**：Transformer注意力机制、三角几何基础
- **进阶**：YaRN/Linear RoPE等用于超长序列的位置编码扩展
