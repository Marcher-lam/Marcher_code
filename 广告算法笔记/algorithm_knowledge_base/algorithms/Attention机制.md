# Attention机制 学习文档

## 1. 算法基础认知

Attention 机制是深度学习中的加权聚合方法，核心思想是"聚焦于重要的部分"。通过 Query-Key-Value 三元组计算相关性权重，对 Value 加权求和。在广告系统中，Attention 是 CTR/CVR 模型的核心组件，从 DIN 的 Target Attention 到 Transformer 的 Self-Attention。

## 2. 核心原理

Attention 的本质是一个可微分的"软寻址"机制。Query 发出查询请求，Key 是检索索引，Value 是实际内容。通过 Q 和 K 的相似度计算注意力权重，再对 V 加权求和。缩放因子 $\sqrt{d_k}$ 防止点积过大导致 softmax 饱和。

## 3. 数学公式与推导

**缩放点积注意力（Scaled Dot-Product Attention）**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

展开形式：

$$
\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}
$$

$$
\text{output}_i = \sum_j \alpha_{ij} v_j
$$

**DIN 中的 Target Attention**：

$$
\alpha_i = \frac{\exp(\mathbf{e}_i^\top \mathbf{W} \mathbf{e}_{ad})}{\sum_j \exp(\mathbf{e}_j^\top \mathbf{W} \mathbf{e}_{ad})}
$$

$$
\mathbf{v}_u = \sum_i \alpha_i \mathbf{e}_i
$$

其中 $\mathbf{e}_{ad}$ 是候选广告 embedding，$\mathbf{e}_i$ 是历史行为 embedding。

## 4. 训练过程讲解

1. 将输入映射为 Q、K、V（通过线性变换或直接使用 embedding）
2. 计算注意力分数：$S = QK^T / \sqrt{d_k}$
3. Softmax 归一化得到权重 $\alpha$
4. 加权求和：$O = \alpha \cdot V$
5. 输出参与后续网络计算，端到端反向传播
6. 梯度通过 softmax 和矩阵乘法回传到 Q、K、V

## 5. 应用场景

- **Target Attention**：DIN/DIEN 中对候选广告相关的历史行为加权
- **Self-Attention**：BST/DSIN 中序列内依赖建模
- **Cross-Attention**：InterFormer 中 User-Ad 交叉匹配
- 多模态特征融合（图文广告）
- 重排上下文建模（PRM）

## 6. 优缺点分析

**优点：**
- 灵活的加权聚合，可解释性强
- 直接建模任意距离的依赖关系
- 支持并行计算
- 端到端可训练

**缺点：**
- 计算复杂度 $O(n^2)$（序列长度）
- 内存占用随序列长度二次增长
- 注意力权重可能过于平滑（退化为例均）
- 需要足够数据学习有意义的权重

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

d_k = 64
attn = ScaledDotProductAttention(d_k)
Q = torch.randn(2, 8, 10, d_k)
K = torch.randn(2, 8, 10, d_k)
V = torch.randn(2, 8, 10, d_k)
out, weights = attn(Q, K, V)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, d_k):
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

Q = torch.randn(1, 10, 64)
K = torch.randn(1, 10, 64)
V = torch.randn(1, 10, 64)
output, attn_weights = attention(Q, K, V, d_k=64)
```

## 9. 可视化与结果理解

- 注意力权重热力图：哪些历史行为与当前候选广告最相关
- DIN 中不同候选广告激活不同历史行为的对比
- 缩放因子 $\sqrt{d_k}$ 前后 softmax 分布的变化
- Head-level 可视化：不同注意力头关注不同模式

## 10. 模型评估

- 广告 CTR：AUC + LogLoss
- 消融实验：有 Attention vs 无 Attention（均值池化）的 AUC 差异
- 注意力分布熵：过低的熵表示过度聚焦，过高的熵表示退化为例均

## 11. 常见问题与易错点

- **缩放因子必要性**：$d_k$ 较大时点积方差为 $d_k$，不缩放 softmax 会进入饱和区
- **Mask 使用**：因果注意力中 mask 掉未来位置，padding 位置也需 mask
- **DIN 注意力激活函数**：使用外积+FC 而非简单点积，增强表达能力
- **注意力退化**：序列过长时注意力可能退化为近似均匀分布

## 12. 学习总结

Attention 机制的核心贡献在于提出了 Query-Key-Value 这一通用加权聚合框架，使模型能够根据任务需求动态聚焦于最相关的信息，而非对输入做简单的均值或最大值池化。缩放因子 $\sqrt{d_k}$ 的引入解决了高维空间中 softmax 饱和的数值问题，是工程与理论结合的典范。

Attention 的关键优势是灵活性强、可解释性好（注意力权重直接反映重要性），且能建模任意距离的依赖关系。在广告推荐场景中，Target Attention 能自动从用户历史行为中筛选出与当前候选广告最相关的部分，是解决"用户兴趣多样性"问题的利器。但标准 Attention 的 $O(n^2)$ 复杂度限制了超长序列的应用。

在知识体系中，Attention 是本库中 Transformer、BERT、GPT 等模型的共同基础组件，同时也是广告领域 DIN、DIEN、BST 等核心模型的设计灵感来源。从 Target Attention → Self-Attention → Multi-Head Attention → Cross-Attention 构成了一条清晰的技术演进路线。

工业实践中，广告系统使用 Attention 时需注意序列长度对线上延迟的影响，常通过截断历史行为长度或使用近似注意力（如 Linformer）来控制推理耗时。DIN 中的 Attention 使用外积+FC 而非简单点积，表达能力更强但计算量也更大。

## 13. 练习题与思考题（含答案）

**Q1：为什么要除以 $\sqrt{d_k}$？**
A1：Q 和 K 独立分布时点积方差为 $d_k \cdot \sigma^2$，除以 $\sqrt{d_k}$ 使方差稳定为 $\sigma^2$，防止 softmax 饱和。

**Q2：DIN 的 Target Attention 和 Self-Attention 的区别？**
A2：Target Attention 中 Q 来自候选广告，K/V 来自历史行为；Self-Attention 中 Q/K/V 都来自同一序列。

**Q3：Attention 复杂度如何？如何优化？**
A3：标准复杂度 $O(n^2 d)$，可通过 Linear Attention、Flash Attention、稀疏注意力优化。

**Q4：为什么 DIN 用 Attention 而不是简单平均池化？**
A4：用户历史行为中只有部分与当前候选广告相关，Attention 能自适应地聚焦于相关行为。

## 14. 学习路径建议

1. 理解 QKV 框架和缩放点积注意力
2. 学习 DIN 论文，理解 Target Attention 在广告中的应用
3. 学习 Multi-Head Attention（Transformer 核心）
4. 学习 Self-Attention 在 BST 中的应用
5. 进阶 Cross-Attention 和 Flash Attention
