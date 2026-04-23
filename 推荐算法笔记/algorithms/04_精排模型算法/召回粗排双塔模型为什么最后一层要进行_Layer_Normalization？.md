# 面试题：召回粗排双塔模型为什么最后一层要进行 Layer Normalization？

# 面试题：召回粗排双塔模型为什么最后一层要进行 Layer Normalization？

# 回答总结：

在推荐系统的召回粗排双塔模型中，最后一层应用 Layer Normalization (LayerNorm) 是一项关键优化。

Layer Normalization 在双塔模型中的主要作用有如下四点：

<table><tr><td>作用方面</td><td>具体说明</td></tr><tr><td>保持训练稳定</td><td>归一化层输入，缓解内部协变量偏移，加速收敛。</td></tr><tr><td>相似度计算一致性</td><td>使点积等价于余弦相似度，并与向量检索引擎兼容。</td></tr><tr><td>防止模型坍塌</td><td>约束模长，鼓励模型学习均匀分布的表示，提升泛化能力。</td></tr><tr><td>与温度系数协同</td><td>将相似度得分缩放至合适范围，使损失函数能有效关注困难负样本。</td></tr></table>

# 1. 稳定训练与加速收敛

LayerNorm 通过对每个样本的特征维度进行归一化，使神经网络各层的输入分布保持稳定，从而缓解内部协变量偏移问题。具体来说，对于输入向量 x（即双塔最后一层的输出），LayerNorm 的计算步骤如下：

 计算均值和方差：

$$
\mu = \frac {1}{d} \sum_ {i = 1} ^ {d} x _ {i}, \quad \sigma^ {2} = \frac {1}{d} \sum_ {i = 1} ^ {d} (x _ {i} - \mu) ^ {2} \quad ,   \text {其 中} d \text {是 嵌 入 向 量 的 维 度}.
$$

 归一化：

$$
\hat {x} _ {i} = \frac {x _ {i} - \mu}{\sqrt {\sigma^ {2} + \epsilon}} \text {, 这 里} \epsilon \text {是 一 个 很 小 的 常 数 (例 如 1 e - 1 2) , 用 于 防 止 除 以 零 。}
$$

 缩放和平移：

$y _ { i } = \gamma \hat { x } _ { i } + \beta$ ，其中 γ 和 $\beta$ 是可学习的参数，用于恢复模型的表现力。

这种操作使得每个特征维度的数值分布更加稳定，有利于梯度在反向传播时更平稳地流动，从而加速模型收敛并提高训练稳定性。

**BatchNorm vs LayerNorm 在双塔中的对比**：

BatchNorm 对同一个特征维度跨Batch做归一化，公式为 $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$，其中 $\mu_B$ 和 $\sigma_B^2$ 是当前mini-batch的统计量。在双塔模型中不选择BatchNorm的原因：

1. **Batch size波动**：召回模型的训练数据通常来自流式数据，batch size波动较大，导致BatchNorm的统计量不稳定
2. **推理不一致**：推理时使用训练阶段累积的滑动平均统计量，与训练时的batch统计量存在差异
3. **序列特征问题**：用户行为序列等变长输入不适合跨样本归一化

LayerNorm 是对单个样本的特征维度做归一化，不依赖batch统计量，因此在推理时行为完全一致，更适合双塔模型。

**梯度稳定性分析**：假设损失函数 $L$ 对归一化后向量 $\hat{x}$ 的梯度为 $\frac{\partial L}{\partial \hat{x}}$，则对原始输入 $x$ 的梯度为：

$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sigma} \left( \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{d}\sum_{j=1}^{d}\frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \cdot \frac{1}{d}\sum_{j=1}^{d}\hat{x}_j \frac{\partial L}{\partial \hat{x}_j} \right)
$$

LayerNorm 将梯度缩放至 $\frac{1}{\sigma}$ 量级，防止了梯度爆炸或消失。

# 2. 统一向量尺度与相似度计算

在双塔模型中，User Embedding 和 Item Embedding 的相似度通常通过点积或余弦相似度计算。LayerNorm 通过 L2Norm 将向量投影到单位超球面上，带来关键好处：

 点积与余弦相似度等价：对向量 u 和 v 进行 L2 归一化后，点积等价于余弦相似度：

$$
\operatorname {c o s i n e} (u, v) = \frac {u \cdot v}{| | u | | \cdot | | v | |} = \hat {u} \cdot \hat {v}
$$

其中 $\hat { u }$ 和 $\hat { v }$ 是归一化后的向量。

 与向量检索引擎兼容：主流的向量检索引擎（如 FAISS）通常支持内积或欧氏距离作为度量。归一化后，点积计算更高效，且欧氏距离与余弦相似度可以相互转化（因为当向量模长为 1 时，欧氏距离与余弦相似度存在单调关系）。这确保了训练与推理阶段的一致性。

**向量检索引擎的兼容性详解**：

以 FAISS 为例，常用的索引类型包括：
- `IndexFlatIP`：内积索引，归一化向量后等价于余弦相似度
- `IndexIVFFlat`：倒排索引，需要配合内积或L2距离
- `IndexHNSWFlat`：基于图的方法，支持内积度量

训练阶段使用归一化向量 + 点积，部署阶段直接用 `IndexFlatIP` 做最近邻检索，保证了线上线下一致性。如果不做归一化，需要在线上额外做余弦相似度的归一化计算，增加延迟。

**欧氏距离与余弦相似度的关系推导**：

$$
\|u - v\|^2 = \|u\|^2 + \|v\|^2 - 2u \cdot v
$$

当 $\|u\| = \|v\| = 1$ 时，$\|u - v\|^2 = 2 - 2 \cos(u, v)$，即欧氏距离与余弦相似度存在严格单调关系。

# 3. 防止模型坍塌与提升表示质量

在对比学习框架下，LayerNorm 有助于防止"模型坍塌"（即所有样本的嵌入坍塌到同一个点）。一个好的对比学习系统应兼顾：

 Alignment：正样本对在投影空间中距离应尽可能接近。  
 Uniformity：所有样本在投影空间中的分布应尽可能均匀，以保留个性化信息。

LayerNorm 通过约束向量的模长，迫使模型更专注于学习向量间的角度差异，而非依靠增大向量模长来简单降低损失。这有助于模型学习到更均匀分布的表示，避免坍塌。

如果没有归一化，模型容易"走捷径"：频繁出现的物品（如热门商品）其嵌入向量的模长会被学习得很大，以简单扩大点积值，但这会损害模型对细粒度语义信息的学习能力。

**Uniformity 指标的数学定义**：

$$
\mathcal{W}(f) = \log \mathbb{E}_{x, y \sim p_{data}} \left[ e^{-t \|f(x) - f(y)\|^2} \right]
$$

其中 $t$ 是温度超参数。该指标衡量嵌入向量在超球面上的均匀程度，值越小表示分布越均匀。实验表明，加入LayerNorm后，Uniformity指标显著改善。

**模长爆炸的数值分析**：假设不使用归一化，热门物品 $i$ 出现频率为 $p_i$，其嵌入向量模长的期望增长为：

$$
\|e_i\| \propto \sqrt{\text{梯度累积次数}} \propto \sqrt{N \cdot p_i}
$$

其中 $N$ 为总训练步数。高频物品的模长会远大于低频物品，导致检索结果严重偏向热门物品。

# 4. 与温度系数协同工作

在对比损失（如 InfoNCE Loss）中，温度系数 $\tau$ 与 LayerNorm 协同工作，对模型效果至关重要：

 温度系数的作用：损失函数公式为：

$$
L o s s = - \log \frac {\exp (\sin (u , v _ {+}) / \tau)}{\sum_ {v \in \{v _ {+} \cup V _ {-} \}} \exp (\sin (u , v) / \tau)}
$$

$\tau$ 调节对困难负样本的关注程度。较小的 $\tau$ 会使损失更关注那些与正样本相似度较高的困难负样本。

 与 LayerNorm 的协同：LayerNorm 将相似度得分限制在 [−1,1]范围内。若不使用温度系数进行缩放，Softmax 函数的响应会不够敏感，模型难以有效学习。温度系数（通常取 0.01 到 0.1 之间）将相似度得分放大回一个适合 Softmax 函数敏感区间的范围，使梯度更新更具区分性。

**温度系数的理论分析**：

Softmax函数的梯度为 $\frac{\partial L}{\partial s_i} = p_i - \mathbb{1}[i = y]$，其中 $s_i = \text{sim}(u, v_i) / \tau$ 是logit。当 $\tau$ 很小时，logit被放大，分布变得更加尖锐，梯度集中在困难样本上；当 $\tau$ 很大时，分布趋于均匀，梯度被均摊到所有样本。

典型的温度系数选择：
- $\tau = 0.05$：强关注困难负样本，适合负样本质量高的场景
- $\tau = 0.1$：平衡选择，工业界最常用
- $\tau = 0.5$：弱关注困难负样本，适合负样本噪声大的场景

# 5. Python 代码验证

```python
import torch
import torch.nn as nn
import numpy as np

d = 64
batch_size = 128
num_items = 10000

user_embeddings = torch.randn(batch_size, d)
item_embeddings = torch.randn(num_items, d)

popular_items = torch.randn(100, d) * 3.0
item_embeddings[:100] = popular_items

def cosine_sim(u, v):
    u_norm = u / u.norm(dim=-1, keepdim=True)
    v_norm = v / v.norm(dim=-1, keepdim=True)
    return torch.mm(u_norm, v_norm.t())

def dot_product_sim(u, v):
    return torch.mm(u, v.t())

sims_dot = dot_product_sim(user_embeddings[:5], item_embeddings)
topk_dot = sims_dot.topk(10, dim=-1).indices
print("=== 不归一化 Top-10 倾向热门物品 ===")
print(f"Top-10 索引: {topk_dot}")
print(f"热门物品占比: {(topk_dot < 100).float().mean():.2%}")

user_norm = user_embeddings / user_embeddings.norm(dim=-1, keepdim=True)
item_norm = item_embeddings / item_embeddings.norm(dim=-1, keepdim=True)
sims_cos = torch.mm(user_norm[:5], item_norm.t())
topk_cos = sims_cos.topk(10, dim=-1).indices
print("\n=== 归一化后 Top-10 分布更均匀 ===")
print(f"Top-10 索引: {topk_cos}")
print(f"热门物品占比: {(topk_cos < 100).float().mean():.2%}")

layer_norm = nn.LayerNorm(d)
user_ln = layer_norm(user_embeddings)
item_ln = layer_norm(item_embeddings)
sims_ln = torch.mm(user_ln[:5], item_ln.t())
topk_ln = sims_ln.topk(10, dim=-1).indices
print("\n=== LayerNorm 后 Top-10 ===")
print(f"Top-10 索引: {topk_ln}")
print(f"热门物品占比: {(topk_ln < 100).float().mean():.2%}")

print("\n=== 温度系数效果对比 ===")
tau_values = [0.01, 0.05, 0.1, 0.5]
for tau in tau_values:
    logits = sims_cos[:1] / tau
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_prob = probs.max().item()
    print(f"tau={tau}: entropy={entropy:.2f}, max_prob={max_prob:.4f}")
```

# 6. 常见问题与易错点

1. **LayerNorm 放在最后一层还是倒数第二层**：应放在最后一层（输出层之前），确保输出向量模长可控。放在中间层主要起训练稳定作用，无法约束最终输出的模长。
2. **是否需要可学习参数 $\gamma$ 和 $\beta$**：在双塔最后一层，通常设 $\gamma=1, \beta=0$（即不学习），纯粹做归一化。保留可学习参数可能让模型"学回去"，弱化归一化效果。
3. **L2Norm vs LayerNorm**：L2Norm 直接将向量投影到单位球面，LayerNorm 先中心化再缩放。在实践中两者效果接近，但LayerNorm与向量检索引擎的兼容性更好。
4. **忘记在推理阶段做归一化**：如果训练时最后一层有LayerNorm，线上推理时也必须包含，否则相似度计算不一致。使用ONNX或TensorRT部署时需确保归一化层被正确导出。

# 第四章：精排模型算法

# 4.1 特征交叉结构
