# 面试题：旋转位置编码 RoPE 原理

# 面试题：旋转位置编码 RoPE 原理

旋转位置编码（RoPE）是一种巧妙的位置编码方法，它通过旋转向量的方式将位置信息注入到查询（Query）和键（Key）中，使得注意力机制能够天然地捕捉相对位置信息。

下表对比了 RoPE 与两种传统位置编码方式的区别。

<table><tr><td>特点</td><td>绝对位置编码（如正弦编码）</td><td>可学习位置向量</td><td>旋转位置编码（RoPE）</td></tr><tr><td>核心思想</td><td>为每个绝对位置生成一个固定的编码向量，与词嵌入相加</td><td>为每个位置分配一个可学习的参数向量，与词嵌入相加</td><td>通过旋转矩阵对Q、K向量进行变换，将位置信息表示为角度</td></tr><tr><td>位置信息类型</td><td>绝对位置</td><td>绝对位置</td><td>相对位置</td></tr><tr><td>长度外推能力</td><td>差，难以泛化到训练时长度的序列</td><td>差，固定最大长度</td><td>强，能更好地处理长序列</td></tr><tr><td>关键优势</td><td>简单，无需学习</td><td>可适应训练数据</td><td>内积结果只依赖于相对位置，数学优雅，计算高效</td></tr></table>

# 一、理论原理

# 1 位置编码的本质

自注意力机制本身无法感知位置顺序，需通过位置编码引入序列信息。传统绝对位置编码（如 Sinusoidal）直接与词向量相加，但经过线性变换后，位置信息的远程衰减特性易被破坏。

# 2 RoPE 的核心思想

将位置编码转化为复数域的旋转操作：对查询（Query）和键（Key）向量分别施加旋转矩阵，使得它们的相对位置差通过旋转角度自然体现。这种操作等价于将词向量在复数空间中旋转特定角度，从而计算注意力分数时保留相对位置关系。

# 3 几何意义与优势

旋转不变性：旋转操作不改变向量模长，保持模型稳定性。  
 相对位置编码：通过旋转角度差直接编码相对位置，无需显式设计相对位置参数。  
 外推性：旋转矩阵的连续性使得模型在训练长度外也能保持位置感知能力

# 二、数学公式推导

# 1. 二维情形推导

假设词向量为二维复数 ${ \bf x } _ { m } = x _ { 0 } + i x _ { 1 }$ ，RoPE 通过旋转角度 $m \theta$ （其中 $m$ 为位置索引）构造位置编码：

$$
\mathbf {x} _ {m} ^ {\prime} = \mathbf {x} _ {m} \cdot e ^ {i m \theta}, \text {展 开 为 实 数 形 式 即}:
$$

$$
\left[ \begin{array}{c} x _ {0} ^ {\prime} \\ x _ {1} ^ {\prime} \end{array} \right] = \left[ \begin{array}{c c} \cos m \theta & - \sin m \theta \\ \sin m \theta & \cos m \theta \end{array} \right] \left[ \begin{array}{c} x _ {0} \\ x _ {1} \end{array} \right]
$$

该操作等价于将二维向量逆时针旋转 弧度。

# 2. 高维推广

对于 维词向量，将其分为 $d / 2$ 组，每组两两应用二维旋转变换：

$$
\mathbf {x} _ {m} ^ {\prime} = \bigoplus_ {k = 1} ^ {d / 2} \left[ \begin{array}{c c} \cos m \theta_ {k} & - \sin m \theta_ {k} \\ \sin m \theta_ {k} & \cos m \theta_ {k} \end{array} \right] \mathbf {x} _ { [ 2 k: 2 k + 1 ]}
$$

其中 $\theta _ { k } = 1 0 0 0 0 ^ { - 2 k / d }$ ，通过指数衰减调节不同维度的旋转频率。

# 3. 注意力计算融合

在自注意力机制中，对query ${ \bf q } _ { m }$ 和 key $\mathbf { k } _ { n }$ 分别施加旋转后计算内积：

$$
\operatorname {A t t e n t i o n} (m, n) = \operatorname {R e} \left[ \left(\mathbf {q} _ {m} e ^ {i m \theta}\right) \left(\mathbf {k} _ {n} e ^ {i n \theta}\right) ^ {*} \right]
$$

展开后包含相对位置项 $( m - n ) \theta$ ，显式编码相对距离信息。

# 三、核心特性与优势

# 1、远程衰减性

内积结果随相对距离增大呈震荡衰减趋势，符合自然语言中邻近词关联更强的特性：

$$
\langle \mathbf {q} _ {m}, \mathbf {k} _ {n} \rangle \propto \sum_ {k = 1} ^ {d / 2} \cos ((m - n) \theta_ {k})
$$

随着 q 和 k 的相对距离的增加，它们之间的内积分数呈现出远程衰减的性质。

![](images/2528fccdac02e7245333396ffaea7f398365d9cf61ff8b155bc7a5b6a2c3bf0b.jpg)

# 2、外推能力

旋转操作的周期性允许模型处理超过训练长度的序列，如训练使用 4k 长度，推理可扩展至 $3 2 \mathsf { k } _ { \circ }$

# 3、正交性保持

旋转矩阵是正交矩阵，保持向量模长不变，增强模型训练稳定性。

# 1. 主要应用场景

长文本建模：如 LLaMA、ChatGLM 等千亿参数模型采用 ROPE 处理长文档生成；  
高效线性 Attention：与线性 Attention 兼容，降低长序列计算复杂度；

多模态扩展：在视频、语音序列中验证位置感知有效性。

# 四、核心代码

```python
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算旋转频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) [:(dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs_device) # 位置索引
    freqs = torch.outer(t, freqs) # 外积生成位置-频率矩阵
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # 转换为复数形式
    return freqs_cis
def apply_rotary_emb(xq: torch.Tensor,
xk: torchTensor,
freqs_cis: torch.Tensor,
):
    # 将向量转换为复数形式并旋转
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[: -1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[: -1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis). flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis). flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk) 
```

# 五、RoPE 的扩展与改进

| 改进方法 | 核心思想 | 优势 | 代表模型 |
|---------|---------|------|---------|
| NTK-aware Scaling | 调整 base 频率以适配更长序列 | 实现简单，无需微调 | Code Llama |
| YaRN | 结合温度缩放和注意力衰减 | 长文本外推效果好 | 多个开源模型 |
| Dynamic NTK | 动态调整 base 频率 | 无需预设目标长度 | Perplexity |
| Position Interpolation | 将位置索引线性插值到训练范围内 | 稳定性好 | LLaMA 长文本版本 |
| Combined RoPE | 融合多种缩放策略 | 综合效果最优 | Qwen2 等 |

# 六、完整代码实现：RoPE 注意力机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    seq_len = xq_.shape[1]
    freqs_cis_slice = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis_slice).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis_slice).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
class RotaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)
d_model, n_heads, max_seq_len = 512, 8, 2048
rope_attn = RotaryAttention(d_model, n_heads, max_seq_len)
x = torch.randn(2, 128, d_model)
causal_mask = torch.tril(torch.ones(128, 128)).unsqueeze(0).unsqueeze(0)
output = rope_attn(x, mask=causal_mask)
print(f"Input: {x.shape}, Output: {output.shape}")
```

# 七、常见问题与易错点

| 问题 | 说明 | 建议 |
|------|------|------|
| RoPE 只作用于 Q 和 K | V 向量不施加旋转，因为注意力分数只由 Q·K 决定 | 确保 apply_rotary_emb 只对 Q、K 操作 |
| 频率计算错误 | $\theta_k = 10000^{-2k/d}$ 容易实现为 $10000^{-k/d}$ | 注意公式中指数是 $2k/d$，不是 $k/d$ |
| 序列长度外推 | 直接使用超出训练长度的位置会退化 | 使用 NTK Scaling 或 Position Interpolation |
| 与 Flash Attention 兼容 | RoPE 需要在 QK 投影后、注意力计算前应用 | 确保 RoPE 作用位置正确 |
| 多头注意力的维度 | 每个头独立施加旋转，维度为 head_dim | 注意 reshape 时正确分组 |

# 八、学习总结

1. RoPE 通过旋转矩阵将位置信息编码为角度，使得注意力分数天然包含相对位置信息，数学上非常优雅
2. 核心公式：$q_m^T k_n = q^T R_{m-n} k$，注意力分数只依赖相对位置差 $m-n$
3. 远程衰减性保证了邻近 token 关联更强，符合语言直觉
4. 外推能力是 RoPE 的关键优势，通过 NTK Scaling 等方法可从 4K 扩展到 128K+
5. LLaMA、Qwen、Mistral 等主流大模型均采用 RoPE，已成为位置编码的事实标准

# 九、思考题

1. 为什么 RoPE 不对 Value 向量施加旋转？
2. 如果将 $\theta_k = 10000^{-2k/d}$ 中的 base 从 10000 改为其他值（如 500000），会带来什么影响？

**参考答案：**

1. 注意力分数由 $q_m^T k_n$ 计算，RoPE 的目标是让这个分数包含相对位置信息。Value 向量是在注意力权重确定后才参与的加权求和，对其旋转不会影响位置编码的效果，反而可能引入不必要的变换。

2. 增大 base 会使频率衰减更慢，即高维度的旋转频率更大。这会增强模型对长距离位置的区分能力，但可能牺牲短距离的精度。NTK-aware Scaling 正是通过增大 base 来实现长度外推的。
