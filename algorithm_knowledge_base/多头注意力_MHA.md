# 多头注意力 (MHA) 学习文档

> 来源线索：本节内容根据原书中关于"多头注意力机制"（第3章 3.1.5节）的相关章节整理、扩展与教学化改写。

> 并行多个注意力"视角"，让模型在不同子空间同时学习，再融合决策。

## 1. 算法基础认知

**一句话定义**：将多个自注意力并行计算后拼接，使模型从不同表示子空间抽取信息。

**直觉类比**：想象一个评审委员会评审一篇论文——每位评审从不同角度审视（有的看创新性，有的看实验严谨性，有的看写作质量），最后综合所有评审意见做出决定。多头注意力类似：每个"头"关注输入的不同方面，最后拼接融合。

**历史背景**：2017年Google在《Attention is All You Need》论文中提出多头注意力机制作为Transformer的核心组件。该论文证明了抛弃RNN和CNN、纯粹依赖多头注意力是完全可行的，开启了NLP领域的革命。

**算法定位**：深度学习 / 注意力机制。是对单头自注意力的直接扩展，属于神经网络架构组件。

**前置知识**：
- 自注意力机制（Scaled Dot-Product Attention）
- 线性投影（Linear Projection）和矩阵运算
- PyTorch张量的reshape和transpose操作
- 理解"特征子空间"的概念

## 2. 核心原理

### 核心思想

单头自注意力虽然强大，但有一个局限：它对所有位置的交互只能学习一种模式。例如在"她把苹果给了邻居"和"她用苹果手机拍照"中，"苹果"一词的含义不同，但单头注意力只能产生一种注意力分布，难以同时捕获"苹果→邻居"和"苹果→手机"两种不同的关联。

多头注意力的解决方案简单而优雅：**多做几遍，每遍用不同的投影矩阵**。通过h个并行的注意力"头"，每个头在各自的低维子空间中计算注意力，最后拼接融合。

### 工作流程

1. **多组投影**：输入X经过h组不同的 $W_i^Q, W_i^K, W_i^V$（每组维度为 $d_k = d_{model}/h$），得到h组 $(Q_i, K_i, V_i)$
2. **并行注意力**：每组独立计算缩放点积注意力，得到h个输出 $head_i$
3. **拼接**：将所有头的输出沿特征维度拼接：$\text{Concat}(head_1, ..., head_h)$
4. **最终投影**：通过一个额外的线性层 $W^O$ 将拼接结果映射回 $d_{model}$ 维

### 关键概念解释

- **头（Head）**：一个独立的注意力计算单元，有自己专属的 $W^Q, W^K, W^V$
- **为什么降低维度**：每个头的维度 $d_k = d_{model}/h$，这样h个头拼接后的总维度恰好等于 $d_{model}$，计算量与单头高维注意力持平
- **头的作用分化**：训练后不同头会自然学习到不同的注意力模式——有的头关注相邻词，有的关注远距离依赖，有的关注句法结构

### 几何/直观解释

```
         输入 X: (batch, seq, d_model=512)
                |
        ┌───────┼────────┬────────┐
        v       v        v        v
     头1      头2      头3  ...  头8
   (64维)   (64维)   (64维)    (64维)
        |       |        |        |
   Attention Attention Attention Attention
        |       |        |        |
        v       v        v        v
      head1   head2    head3 ... head8
        |       |        |        |
        └───────┴────────┴────────┘
                |
           拼接 (512维)
                |
             W^O 投影
                |
          输出 (512维)
```

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| $h$ | 注意力头数 | 标量 |
| $d_{model}$ | 模型总维度 | 标量 |
| $d_k = d_v$ | 每个头的维度 | $d_{model} / h$ |
| $W_i^Q$ | 第i个头的Query投影 | $(d_{model}, d_k)$ |
| $W_i^K$ | 第i个头的Key投影 | $(d_{model}, d_k)$ |
| $W_i^V$ | 第i个头的Value投影 | $(d_{model}, d_v)$ |
| $W^O$ | 输出投影矩阵 | $(h \cdot d_v, d_{model})$ |
| $head_i$ | 第i个头的注意力输出 | $(N, d_v)$ |

### 公式推导

**单个头的计算**（与自注意力完全相同）：

$$head_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

其中：
$$Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V$$

**多头拼接与投影**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, head_2, ..., head_h) \cdot W^O$$

由于每个 $head_i$ 维度为 $(N, d_v)$，拼接后为 $(N, h \cdot d_v) = (N, d_{model})$，乘以 $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ 后输出仍为 $(N, d_{model})$。

### 为什么多头优于单头？

设总计算量相同（总参数不变），多头将高维注意力分解为多个低维注意力：

1. **子空间多样性**：不同的投影矩阵使得各头关注数据的不同方面，类似集成学习
2. **降低方差**：平均来说，多个低维注意力的组合比单个高维注意力更加鲁棒
3. **计算等价性**：h个维度为 $d_k$ 的头的计算量 ≈ 1个维度为 $d_{model}$ 的单头注意力计算量（都涉及 $O(N^2 \cdot d_{model})$ 的矩阵乘法）

## 4. 训练过程讲解

### 数据预处理

- 输入序列需要padding到统一长度
- 通过Embedding层将token ID转换为词向量
- 添加位置编码后再送入多头注意力层

### 参数初始化

- $W_i^Q, W_i^K, W_i^V, W^O$ 使用Xavier均匀初始化
- 通常不设偏置项（bias=False），减少参数量
- 在标准Transformer中，$h=8$ 且 $d_{model}=512$，每个头 $d_k=64$

### 迭代过程

1. **前向传播**：输入 → h路并行注意力 → 拼接 → 输出投影
2. **反向传播**：梯度从输出投影回传到各个头，各个头独立更新自己的投影矩阵
3. 不同头通过梯度信号自然分化，学习不同的注意力模式

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| $h$ (头数) | 并行注意力的数量 | 4-16 | 8（需能被 $d_{model}$ 整除） |
| $d_k$ | 每头维度 | $d_{model}/h$ | 64 |
| dropout | 注意力权重和输出的dropout | 0.0-0.3 | 0.1 |

## 5. 应用场景

### 典型应用

1. **Transformer编码器**：BERT等模型在文本编码中使用多头注意力，不同头分别学习句法结构、语义关联、指代关系等。这是多头注意力最经典的用途。

2. **Transformer解码器**：GPT等生成模型中也使用多头注意力，结合因果掩码使生成过程只能看到已生成的内容。多头帮助解码器同时关注最近的词（局部流畅性）和远距离上下文（全局一致性）。

3. **Vision Transformer (ViT)**：将图像分为patch序列后，用多头注意力捕获不同图像区域之间的关系，替代卷积的局部感受野。研究表明ViT的不同头会关注图像的不同语义区域。

4. **跨模态理解**：在图文匹配模型（如CLIP、ViLT）中，多头注意力分别处理文本和图像的内部关系，不同头可能分别负责对象识别、空间关系、语义理解等。

### 适用数据特征

- 序列数据，需要捕获多样的、多维度的依赖关系
- 输入中蕴含多种类型的关联模式（语法、语义、上下文等）
- 中等长度序列（< 2048），超出后需考虑计算开销

### 不适用场景

- 模型维度较小时（$d_{model} < 64$），分头后每头维度太小效果差
- 超长序列（平方复杂度瓶颈）
- 严格需要单一解释的场景（多个头使可解释性降低）

## 6. 优缺点分析

### 优点

| 优点 | 成立条件 | 说明 |
|------|----------|------|
| 多角度特征提取 | h ≥ 4，数据足够多样 | 不同头学习不同注意力模式，组合后比单头更全面 |
| 计算效率 | $d_k = d_{model}/h$ | 分头后每头维度降低，计算量 ≈ 单头，但效果更好 |
| 训练稳定性 | 使用dropout正则化 | 多个头的"集成"效果降低了单个头的方差 |
| 灵活性强 | 可根据需求调整头数 | 简单的任务可以用更少的头，复杂任务用更多的头 |

### 缺点

| 缺点 | 何时出问题 | 缓解思路 |
|------|-----------|----------|
| 头冗余 | 头数过多时，部分头的注意力模式高度相似 | 使用attention head pruning或在训练中加入头多样性正则化 |
| 解释困难 | 多头的决策交织在一起，难以理解 | 逐头可视化分析，关注异常头 |
| 内存开销 | 每个头都存储 KV cache（推理时 | 使用MQA/GQA共享KV减少缓存 |
| 对头数敏感 | 头数没有选好时效果差异大 | 在验证集上调参，或使用网格搜索 |

### 与单头自注意力的对比

| 特性 | 多头注意力 | 单头自注意力 |
|------|------------|------------|
| 注意力模式 | 多种模式并存 | 单一模式 |
| 参数效率 | 高（多个低维头） | 中等 |
| 表达力 | 强（子空间学习） | 中等 |
| 计算量 | ≈ 单头（总维度相同） | 与头数无关 |
| 可解释性 | 较差（多路交织） | 较好 |

## 7. 调库实现

```python
"""使用PyTorch内置多头注意力"""
import torch
import torch.nn as nn

# ------ 方法1：nn.MultiheadAttention ------
batch_size, seq_len, d_model = 2, 6, 512
num_heads = 8

# 创建多头注意力层
mha = nn.MultiheadAttention(
    embed_dim=d_model,    # 输入/输出维度
    num_heads=num_heads,   # 头数
    dropout=0.1,           # dropout
    batch_first=True       # 使用 (batch, seq, feature) 格式
)

# 模拟输入
x = torch.randn(batch_size, seq_len, d_model)

# 自注意力：Q=K=V=x
output, attn_weights = mha(x, x, x)

print("=== PyTorch 多头自注意力 ===")
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attn_weights.shape} (batch, seq, seq)")
# 注意: nn.MultiheadAttention 的 attn_weights 是所有头的平均
# 如果想看每个头的权重，需要自己实现

# ------ 查看每头权重（需要设置 average_attn_weights=False）------
mha2 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
_, attn_per_head = mha2(x, x, x, average_attn_weights=False)
print(f"每个头的注意力权重形状: {attn_per_head.shape} (batch, num_heads, seq, seq)")
```

**运行结果示例**：
```
=== PyTorch 多头自注意力 ===
输入形状: torch.Size([2, 6, 512])
输出形状: torch.Size([2, 6, 512])
注意力权重形状: torch.Size([2, 6, 6]) (batch, seq, seq)
每个头的注意力权重形状: torch.Size([2, 8, 6, 6]) (batch, num_heads, seq, seq)
```

## 8. 手工代码实现

```python
"""从零手写多头注意力（使用PyTorch张量操作）"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """手工实现的标准多头注意力层
    
    完全从零实现，不使用 nn.MultiheadAttention。
    包含 Q、K、V 的各自投影和最终输出投影。
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        参数:
            d_model: 输入/输出维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q、K、V 的联合投影：用一个大的线性层将输入投影到 3*d_model 维
        # 然后拆分为 Q、K、V，这样效率更高
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入, (batch_size, seq_len, d_model)
            mask: 注意力掩码, (batch_size, seq_len, seq_len)
        返回:
            output: (batch_size, seq_len, d_model)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 步骤1：联合投影得到 Q、K、V
        # W_qkv 一次计算，然后拆分
        qkv = self.W_qkv(x)  # (batch, seq_len, 3*d_model)
        # 拆分为三份，每份 d_model 维
        Q, K, V = qkv.chunk(3, dim=-1)
        
        # 步骤2：重塑为多头形式
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 步骤3：计算缩放点积注意力
        # Q @ K^T: (batch, num_heads, seq_len, d_k) x (batch, num_heads, d_k, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, seq_len, seq_len)
        
        # 步骤4：应用掩码
        if mask is not None:
            # 扩展mask的维度以匹配多头形式
            # mask: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 步骤5：softmax得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 步骤6：加权求和
        # attn_weights @ V: (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)
        # context: (batch, num_heads, seq_len, d_k)
        
        # 步骤7：合并所有头
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 步骤8：输出投影
        output = self.W_o(context)
        
        return output, attn_weights


# ========== 测试代码 ==========
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建模型：512维，8个头，每头64维
    mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.0)
    print(f"总参数量: {sum(p.numel() for p in mha.parameters()):,}")
    
    # 创建测试数据
    x = torch.randn(2, 10, 512)  # batch=2, seq=10, dim=512
    print(f"\n=== 手写多头注意力测试 ===")
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output, weights = mha(x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 验证输出维度正确
    assert output.shape == x.shape, "输出和输入形状必须相同（残差连接的前提）"
    print("\n验证通过: 输出形状 == 输入形状 ✓")
    
    # 验证不同头的注意力模式不同
    head_sim = torch.corrcoef(weights[0].flatten(1))  # 8个头之间的相关性矩阵
    print(f"\n8个头之间的平均相关性: {head_sim.mean():.3f}")
    print("(初期相关性接近，训练后会分化)")
```

**预期输出**：
```
总参数量: 1,050,624

=== 手写多头注意力测试 ===
输入形状: torch.Size([2, 10, 512])
输出形状: torch.Size([2, 10, 512])
注意力权重形状: torch.Size([2, 8, 10, 10])

验证通过: 输出形状 == 输入形状 ✓

8个头之间的平均相关性: 0.XXX
(初期相关性接近，训练后会分化)
```

## 9. 可视化与结果理解

```python
"""多头注意力可视化：展示不同头的注意力模式"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(456)

# 模拟一个有意义的句子
words = ["[CLS]", "今天", "天气", "非常", "好", "适合", "出去", "玩", "[SEP]"]
seq_len = len(words)
d_model = 64
num_heads = 4

# 创建模拟的多头注意力权重（这里手动构造以展示）
# 通常训练后不同头会自然分化
head_names = [
    "头1: 局部关注",
    "头2: 全局关注",
    "头3: CLS关注",
    "头4: 句法关注"
]

# 手动构造注意力模式来演示
attn_patterns = np.zeros((num_heads, seq_len, seq_len))

# 头1：局部关注（每个词主要关注自己和邻居）
for i in range(seq_len):
    for j in range(max(0, i-1), min(seq_len, i+2)):
        attn_patterns[0, i, j] = 1.0
attn_patterns[0] /= attn_patterns[0].sum(axis=-1, keepdims=True)

# 头2：全局关注（均匀关注所有词）
attn_patterns[1] = 1.0 / seq_len

# 头3：CLS关注（所有词关注[CLS]）
attn_patterns[2, :, 0] = 0.8
for i in range(1, seq_len):
    attn_patterns[2, i, i] = 0.2
attn_patterns[2] /= attn_patterns[2].sum(axis=-1, keepdims=True)

# 头4：句法关注（关注特定句法位置）
attn_patterns[3, 4, :] = np.ones(seq_len) / seq_len  # 程度词"非常"的注意力

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx in range(num_heads):
    sns.heatmap(attn_patterns[idx], annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=words, yticklabels=words, ax=axes[idx],
                vmin=0, vmax=0.5)
    axes[idx].set_title(head_names[idx], fontsize=13)
    axes[idx].set_xlabel('Key（被关注方）')
    axes[idx].set_ylabel('Query（关注方）')

plt.suptitle('多头注意力中不同头的关注模式示意', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('multi_head_attention_viz.png', dpi=100, bbox_inches='tight')
plt.show()

print("可视化解读：")
print("- 头1展示局部模式：每个词只关注相邻词（类似CNN）")
print("- 头2展示全局模式：均匀关注（捕获全局语境）")
print("- 头3展示特殊模式：[CLS]标签承担了汇聚信息的角色")
print("- 头4展示任务相关模式：特定位置关注特定语法成分")
print("- 正是这些不同模式的叠加，使得多头注意力比单头强大")
```

**结果解读**：
不同头的注意力模式揭示了多头注意力的核心价值——**多样性**。在实际训练好的模型中，这些模式是自动涌现的，不需要手动设计。有的头可能学会关注局部n-gram模式（类似卷积），有的可能关注长距离依赖，还有的可能学会特殊的语法和语义角色。将多路信息拼接后，模型获得了比任何单一模式都更丰富的表示。

## 10. 模型评估

### 针对多头注意力的评估方法

1. **头间相似度**：计算不同头注意力权重之间的相关性。相似度过高说明头冗余。
2. **头重要性分析**：逐个mask掉头，观察对下游任务的影响，识别关键头和冗余头。

```python
"""多头注意力质量评估"""
def evaluate_mha_heads(attn_weights_per_head):
    """
    attn_weights_per_head: (batch, num_heads, seq_len, seq_len)
    """
    num_heads = attn_weights_per_head.shape[1]
    
    # 1. 头间相似度
    # 将每个头的注意力展平
    flat_heads = attn_weights_per_head[0].reshape(num_heads, -1)
    # 计算余弦相似度
    flat_heads_norm = flat_heads / (flat_heads.norm(dim=-1, keepdim=True) + 1e-9)
    sim_matrix = flat_heads_norm @ flat_heads_norm.T
    
    # 排除对角线（自己和自己的相似度为1）
    mask = ~torch.eye(num_heads, dtype=bool)
    avg_inter_head_sim = sim_matrix[mask].mean().item()
    
    # 2. 每个头的熵（集中度）
    eps = 1e-9
    entropies = -(attn_weights_per_head[0] * torch.log(attn_weights_per_head[0] + eps)).sum(dim=-1).mean(dim=-1)
    
    print("=== 多头注意力质量评估 ===")
    print(f"头间平均相似度: {avg_inter_head_sim:.3f}")
    print(f"  (越接近0越好 -> 头越多样化)")
    print(f"\n各头平均熵:")
    for i in range(num_heads):
        print(f"  头{i}: 熵={entropies[i].item():.3f}")
    print(f"  熵的方差: {entropies.var().item():.4f} (越大说明头越多样化)")
    
    # 判断
    if avg_inter_head_sim > 0.8:
        print("\n⚠ 头间相似度过高，可能存在头冗余，考虑减少头数")

# 假设有一组真实的注意力权重
# evaluate_mha_heads(real_attn_weights)
```

**结果解读**：
- 头间相似度低 → 头功能多样化，多头价值充分体现
- 头间相似度高 → 存在头冗余，可考虑pruning或减少头数
- 熵的方差大 → 头之间有明显的功能分化（好）

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| padding未mask | 模型关注了填充位置 | 多个头并行时忘记传递mask | 确保mask形状正确扩展，能广播到多头维度 |
| 序列长度被头数限制 | 短序列时注意力退化 | 序列长度小于头数时某些头没有足够的上下文 | 对短序列任务减少头数或增加上下文窗口 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| $d_{model}$不能被h整除 | 报错或维度不匹配 | 头数选择不当 | 确保 $d_{model} \bmod h = 0$ |
| reshape/transpose错误 | 张量形状不对 | 多头重组时维度顺序搞混 | 画出每一步的张量形状，逐行验证 |
| 头塌缩 | 所有头学到的模式相同 | 头数太多或训练不充分 | 减少头数、加入熵正则化 |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 头数过多 | 训练慢、显存占用大、效果不提升 | 头冗余 | 基础Transformer用8头，小模型用4头 |
| 头数过少 | 表达能力不足 | 没有足够的子空间 | 至少4头，通常8-16头 |

## 12. 学习总结

### 核心思想回顾

多头注意力的本质是用"集成学习"的思想提升单头注意力——多组独立的注意力并行提取特征，最后融合。关键公式：
$$\text{MultiHead}(X) = \text{Concat}(head_1, ..., head_h) W^O$$
$$head_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

### 与前序/相关算法的联系

- **自注意力是基础**：MHA就是h个并行的自注意力
- **MQA/GQA是优化**：通过共享KV减少缓存开销
- **MLA是创新**：DeepSeek的低秩KV压缩进一步优化
- 多头思想也可用于CNN（分组卷积）等其他架构

### 后续学习方向

- 注意力优化变体（MQA、GQA、MLA）
- 多头注意力的剪枝和压缩
- Cross-Attention（跨序列的注意力）
- Vision Transformer中多头注意力的特殊设计

## 13. 练习题与思考题

### 基础题1：参数计算

一个多头注意力层，$d_{model}=768$，$h=12$。请问：
- 每个头的维度 $d_k$ 是多少？
- Q、K、V投影矩阵的总参数量是多少（不含偏置）？
- 输出投影 $W^O$ 的参数量是多少？

**参考答案**：
- $d_k = 768 / 12 = 64$
- Q、K、V各有独立的投影矩阵 $W \in \mathbb{R}^{768 \times 64}$，共 $3 \times 12 \times 768 \times 64 = 1,769,472$ 个参数
- 或采用联合投影 $W_{qkv} \in \mathbb{R}^{768 \times 2304}$，参数量 $768 \times 2304 = 1,769,472$（相同）
- $W^O \in \mathbb{R}^{768 \times 768}$，参数量 $768 \times 768 = 589,824$
- 总参数量：$1,769,472 + 589,824 = 2,359,296$

### 基础题2：代码补全

写一个函数，给定d_model和num_heads，检查是否合法（d_model能否被num_heads整除），并返回每头维度。

**参考答案**：
```python
def validate_mha_config(d_model, num_heads):
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) 不能被 num_heads ({num_heads}) 整除。"
            f"建议: 使用 num_heads = {d_model // 8} 或类似值"
        )
    d_k = d_model // num_heads
    print(f"配置: d_model={d_model}, h={num_heads}, d_k={d_k}")
    return d_k
```

### 进阶题：分析与设计

在推理时，解码器的多头注意力需要存储KV cache。设序列长度=N，头数=h，$d_k=64$，batch=1。如果用半精度（float16），存储整个KV cache需要多少显存？如果序列增长到10K，又需要多少？从工程角度如何解决10K序列的内存问题？

**参考答案**：
- N=100时：$1 \times h \times N \times d_k \times 2（K和V） \times 2$字节 = $1 \times h \times 100 \times 64 \times 2 \times 2 = 25,600h$ 字节 ≈ $25.6h$ KB
- N=10K时：$\approx 2.56h$ MB
- 对h=32：约81.92 MB——看起来不大，但这只是一层！Transformer通常有几十层，累积起来就很大
- 工程方案：
  1. **MQA/GQA**：共享KV，本质上是减少h
  2. **MLA**（DeepSeek方案）：低秩KV压缩，大幅降低缓存
  3. **Sliding Window Attention**：只保留最近K个token的缓存
  4. **PagedAttention**（vLLM方案）：分页管理KV cache，提高利用率

### 开放思考题

如果让你设计一个新的注意力变体，在多头注意力的拼接阶段，不是简单地将所有头等权拼接，而是学习一个"头的重要性权重"，你认为可能有什么优缺点？

**参考思路**：
- **优点**：允许模型动态调整对不同注意力模式的依赖程度，对某些任务可能有利
- **缺点**：
  1. 增加了参数量和计算量
  2. 权重可能退化到只依赖少数头（与多头的初衷矛盾）
  3. 如果某些头权重为0，这些头的参数实际上被浪费了
  4. 本质上与门控机制（Gating）类似，可能引入额外的优化难度
- **改进方向**：可以使用Sparsity约束（如Top-K选择）或L2正则化来防止头权重的极端分布

## 14. 学习路径建议

### 前置算法
- **自注意力机制**：理解Scaling和softmax的作用
- **词嵌入**：理解连续向量表示
- **线性变换**：理解投影矩阵的作用

### 平行算法
- **MQA（多查询注意力）**：KV共享的极致版
- **GQA（分组查询注意力）**：MHA和MQA的折中
- **分组卷积（Grouped Convolution）**：CNN中的类似思想

### 进阶算法
- **MLA（多头潜在注意力）**：DeepSeek核心创新，低秩压缩KV
- **差分注意力**：引入噪声过滤的多头注意力
- **Cross-Attention**：跨序列的多头注意力
- **Flash Attention**：工程优化，减少显存和加速计算

### 推荐资源
1. **论文**：《Attention is All You Need》（Vaswani et al., 2017）Section 3.2 Multi-Head Attention
2. **博客**：The Annotated Transformer (Harvard NLP) — 逐行代码注释
3. **视频**：Andrej Karpathy的"Let's build GPT from scratch" — 包含MHA的完整实现讲解
