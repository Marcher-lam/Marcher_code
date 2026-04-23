# 面试题：常见的对比学习 Loss 有哪些？

面试题：常见的对比学习 Loss 有哪些？

在推荐系统中，对比学习通过构建正负样本对的相似性关系来优化特征表示，以下是常见的对比学习损失函数及其详细表达式：

1. InfoNCE Loss（噪声对比估计损失，最流行）

核心思想：最大化正样本对的相似度，同时最小化正样本与多个负样本的相似度。

表达式： $L _ { \mathrm { I n f o N C E } } = - \log { \frac { \exp ( s ( x , x ^ { + } ) / \tau ) } { \exp ( s ( x , x ^ { + } ) / \tau ) + \sum _ { x ^ { - } \in X ^ { - } } \exp ( s ( x , x ^ { - } ) / \tau ) } }$

$s ( x , y )$ ：样本 $x$ 和 $y$ 的相似度（如余弦相似度）；  
$x ^ { + }$ ：正样本，$x ^ { - }$ ：负样本集合；  
 $\tau$ ：温度参数，控制分布尖锐程度。

应用场景：推荐系统的用户-物品交互建模，如序列推荐中的正样本（点击）与负样本（未点击）对比。

# 2. Triplet Loss（三元组损失）

核心思想：通过锚点（Anchor）、正样本（Positive）、负样本（Negative）的相对距离优化表示。

表达式： $L _ { \mathrm { T r i p l e t } } = \operatorname* { m a x } ( 0 , d ( a , p ) - d ( a , n ) + \operatorname* { m a r g i n } )$

 ：锚点与正样本的距离（如欧氏距离）； $d ( a , p )$   
：锚点与负样本的距离； $d ( a , n )$   
margin：最小间隔阈值，确保区分性。

应用场景：推荐系统中的个性化排序，例如用户历史行为（正样本）与未交互物品（负样本）的对比。

# 3. Contrastive Loss（对比损失）

核心思想：直接区分正负样本对的相似性关系。

表达式：

$$
L _ {\text {C o n t r a s t i v e}} = y \cdot d \left(x _ {1}, x _ {2}\right) + (1 - y) \cdot \max  (\operatorname {m a r g i n} - d \left(x _ {1}, x _ {2}\right), 0)
$$

 y=1：正样本对（相似），需最小化距离 $d ( x _ { 1 } , x _ { 2 } )$ ；  
 $y = 0$ ：负样本对（不相似），若距离小于 margin 则施加惩罚。

应用场景：用户兴趣建模，如社交推荐中用户相似关系与不相关关系的区分。

# 4. N-Pair Loss（多负样本对比损失）

核心思想：Triplet Loss 的扩展，支持单正样本对多个负样本的对比。

表达式：

$$
L _ {\mathrm{N} - \text {P a i r}} = \log \left(1 + \sum_ {i = 1} ^ {N} \exp \left(s \left(x, x _ {i} ^ {-}\right) - s \left(x, x ^ {+}\right)\right)\right)
$$

$_ x$ ：锚点样本；  
$x ^ { + }$ ：正样本，$\boldsymbol { x } _ { i } ^ { - }$ ：第 $j$ 个负样本。

应用场景：大规模推荐场景，如电商中用户点击商品与海量未曝光商品的对比。

# 5. NCE Loss（噪声对比估计损失）

核心思想：通过采样负样本近似全量分布，降低计算复杂度。

表达式：

$$
L _ {\mathrm{N C E}} = - \log \frac {\exp (s (x , x ^ {+}))}{\exp (s (x , x ^ {+})) + \sum_ {x ^ {-} \in X ^ {-}} \exp (s (x , x ^ {-}))}
$$

与 InfoNCE 的区别：NCE Loss 不包含温度参数 τ，常用于语言模型和长尾推荐中的负采样优化。

总结与对比  

<table><tr><td>损失函数</td><td>核心特点</td><td>适用场景</td></tr><tr><td>InfoNCE</td><td>引入温度参数，支持多负样本对比</td><td>大规模推荐、跨模态对齐（如CLIP）</td></tr><tr><td>Triplet Loss</td><td>强调相对距离，需手动设定间隔阈值</td><td>精细化排序、用户兴趣建模</td></tr><tr><td>Contrastive Loss</td><td>显式控制正负样本距离，需预设间隔参数</td><td>有监督推荐</td></tr><tr><td>N-Pair Loss</td><td>单正样本对多负样本，提升训练效率</td><td>电商、广告推荐中的长尾物品处理</td></tr><tr><td>NCE Loss</td><td>简化采样复杂度，适合长尾分布数据</td><td>语言模型、点击率预测</td></tr></table>

---

# 六、数学推导补充

## 1. InfoNCE 与互信息的关系

InfoNCE 损失实际上是对互信息的一个下界的负估计。对于正样本对 $(x, x^+)$：

$$
I(x; x^+) \geq \log(N) - L_{\text{InfoNCE}}
$$

其中 $N$ 为负样本数量。这意味着最小化 InfoNCE 损失等价于最大化正样本对之间的互信息下界。

## 2. 温度参数 τ 的影响分析

当 $\tau \to 0$ 时，Softmax 分布趋向于 one-hot，只关注最难的负样本；当 $\tau \to \infty$ 时，分布趋向于均匀，所有负样本权重相同。

梯度对温度参数的敏感度：

$$
\frac{\partial L_{\text{InfoNCE}}}{\partial s(x, x^+)} = -\frac{1}{\tau}\left(1 - \frac{\exp(s/\tau)}{\sum \exp(s/\tau)}\right)
$$

实践中 $\tau = 0.05 \sim 0.1$ 效果较好。

## 3. Triplet Loss 的困难样本挖掘

标准 Triplet Loss 对所有三元组一视同仁，但简单三元组（负样本远离锚点）贡献的梯度为零。困难样本挖掘策略：

- **Semi-hard mining**：选择 $d(a, p) < d(a, n) < d(a, p) + \text{margin}$ 的三元组
- **Hard mining**：选择 $d(a, n) < d(a, p)$ 的最难三元组

## 4. NCE 的概率解释

NCE 将多分类问题转化为二分类问题：区分真实样本（正类）和噪声样本（负类）。

$$
P(C=1|x, x^+) = \frac{p(x|x^+)}{p(x|x^+) + k \cdot p_n(x)}
$$

其中 $k$ 为负样本数，$p_n(x)$ 为噪声分布。当负样本数足够多时，NCE 趋近于 Softmax。

# 七、各损失函数的计算复杂度对比

| 损失函数 | 每步计算复杂度 | 内存占用 | 负样本数要求 | 分布式友好度 |
|---------|-------------|---------|------------|------------|
| InfoNCE | $O(N \cdot d)$ | 中 | 建议 ≥256 | 高（支持跨卡） |
| Triplet Loss | $O(d)$ | 低 | 1 | 高 |
| Contrastive Loss | $O(d)$ | 低 | 1 | 高 |
| N-Pair Loss | $O(N \cdot d)$ | 中 | 建议 ≥16 | 中 |
| NCE Loss | $O(N \cdot d)$ | 中 | 建议 ≥100 | 高 |

# 八、应用场景

**序列推荐**：用户历史行为序列作为锚点，下一个交互物品为正样本，随机未交互物品为负样本，使用 InfoNCE 优化。

**跨模态检索**：图文对齐（如 CLIP），图像为锚点，对应文本为正样本，其他文本为负样本。

**社交推荐**：用户社交关系构建正样本对（好友），非好友构建负样本对，使用 Contrastive Loss。

**商品去重**：相似商品（同款不同卖家）作为正样本对，不同商品为负样本对，使用 Triplet Loss。

**冷启动推荐**：利用辅助信息（文本描述、图像）构建对比学习任务，缓解交互稀疏问题。

# 九、优缺点分析

## 优点

- 无需人工标注，利用数据自身结构构建监督信号
- 学习到的表示具有良好的聚类和区分性质
- 可以与有监督学习联合训练，互相增强
- 对长尾物品友好，不依赖丰富的交互数据

## 缺点

- 负样本质量直接影响效果，假负样本（False Negative）会损害性能
- 温度参数、margin 等超参数需要调优
- 大规模场景下负样本存储和计算开销大
- 正样本构造策略（数据增强）的选择需要领域知识

# 十、Python 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    neg_sims = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / temperature
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def contrastive_loss(x1, x2, label, margin=1.0):
    dist = F.pairwise_distance(x1, x2, p=2)
    loss = label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)
    return loss.mean()


def n_pair_loss(anchor, positive, negatives):
    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True)
    neg_sims = torch.matmul(negatives, anchor.unsqueeze(-1))
    logits = torch.cat([pos_sim, neg_sims], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def nce_loss(anchor, positive, negatives):
    pos_sim = torch.sum(anchor * positive, dim=-1)
    neg_sims = torch.sum(negatives * anchor.unsqueeze(1), dim=-1)
    logits_pos = torch.exp(pos_sim)
    logits_neg = torch.exp(neg_sims).sum(dim=-1)
    loss = -torch.log(logits_pos / (logits_pos + logits_neg + 1e-8))
    return loss.mean()


torch.manual_seed(42)
batch_size = 8
embed_dim = 64

anchor = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
positive = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
negatives = F.normalize(torch.randn(batch_size, 32, embed_dim), dim=-1)
negative = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
labels = torch.randint(0, 2, (batch_size,)).float()

ce = info_nce_loss(anchor, positive, negatives, temperature=0.07)
tri = triplet_loss(anchor, positive, negative, margin=0.5)
con = contrastive_loss(anchor, positive, labels, margin=1.0)
npair = n_pair_loss(anchor, positive, negatives)
nce = nce_loss(anchor, positive, negatives)

print(f"InfoNCE Loss:      {ce.item():.4f}")
print(f"Triplet Loss:      {tri.item():.4f}")
print(f"Contrastive Loss:  {con.item():.4f}")
print(f"N-Pair Loss:       {npair.item():.4f}")
print(f"NCE Loss:          {nce.item():.4f}")


class ContrastiveRecommender(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, temperature=0.07):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.temperature = temperature
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        u = F.normalize(self.user_embed(user_ids), dim=-1)
        pos_i = F.normalize(self.item_embed(pos_item_ids), dim=-1)
        neg_i = F.normalize(self.item_embed(neg_item_ids), dim=-1)
        return info_nce_loss(u, pos_i, neg_i, self.temperature)


n_users, n_items = 1000, 5000
model = ContrastiveRecommender(n_users, n_items, embed_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(100):
    user_ids = torch.randint(0, n_users, (64,))
    pos_ids = torch.randint(0, n_items, (64,))
    neg_ids = torch.randint(0, n_items, (64, 16))

    u = F.normalize(model.user_embed(user_ids), dim=-1)
    pos = F.normalize(model.item_embed(pos_ids), dim=-1)
    neg = F.normalize(model.item_embed(neg_ids.view(-1, 1).expand(-1, 16).reshape(-1)).view(64, 16, -1), dim=-1)

    loss = info_nce_loss(u, pos, neg, temperature=0.07)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

# 十一、常见问题与易错点

## 1. 负样本数量选择

负样本过少（如<16）导致对比信号弱，模型区分度不够；过多（如>4096）增加计算开销且收益递减。推荐范围内 64-1024。

## 2. 假负样本（False Negative）

随机采样的负样本中可能包含用户实际感兴趣但未曝光的物品。处理方法：使用曝光未点击样本、或 CL4CVR 的 FNE 策略。

## 3. 温度参数调节

τ 过小导致只关注最难负样本，训练不稳定；τ 过大导致所有负样本权重相近，区分度不够。建议从 0.05-0.1 开始搜索。

## 4. 特征归一化的必要性

使用余弦相似度时，必须先对嵌入做 L2 归一化（`F.normalize`），否则相似度范围不固定，影响温度参数的效果。

# 十二、学习路径建议

1. **基础**：理解对比学习的核心思想（拉近正样本、推开负样本）
2. **核心**：掌握 InfoNCE 的推导及其与互信息的关系
3. **进阶**：学习数据增强策略（Dropout、Mask、Mixup）对正样本构造的影响
4. **拓展**：研究推荐系统中的自监督学习（SGL、CL4CVR、CoSeRec）
