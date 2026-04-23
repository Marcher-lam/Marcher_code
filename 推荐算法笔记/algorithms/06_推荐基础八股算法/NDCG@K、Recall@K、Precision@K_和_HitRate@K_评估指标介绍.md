# 面试题：NDCG@K、Recall@K、Precision@K 和 HitRate@K 评估指标介绍

面试题：NDCG@K、Recall@K、Precision@K 和 HitRate@K 评估指标介绍

推荐系统中，NDCG@K、Recall@K、Precision@K 和 HitRate@K 这些指标能帮助我们从不同角度评估推荐列表的质量。下面这个表格汇总了它们的核心特点，方便快速了解：

| 评估指标 | 核心关注点 | 计算方式概述 | 适用场景 |
|---------|----------|------------|---------|
| Precision@K | 推荐准确性 | 前K个推荐中用户喜欢的物品比例 | 注重推荐结果准确性的场景，如电商商品推荐 |
| Recall@K | 兴趣覆盖度 | 用户喜欢的物品中被成功推荐的比例 | 注重挖掘用户潜在兴趣的场景，如内容发现 |
| HitRate@K | 简单命中率 | 推荐列表是否至少命中一个用户喜欢的物品 | 快速A/B测试、初步效果对比 |
| NDCG@K | 排名质量 | 考虑物品相关性和位置折扣的加权收益 | 对排序位置敏感的场景，如搜索引擎、列表页推荐 |

# 1. Precision@K：推荐准确性

- 原理：Precision@K衡量的是在推荐系统给出的前 K个结果中，有多少是用户真正喜欢的（即相关的）。它只关注推荐列表本身，计算的是精度。
- 公式：

$$
Precision@K = \frac{\text{前} K \text{个推荐中用户喜欢的物品数量}}{K}
$$

对所有用户计算后，通常再取平均得到整体的 Precision@K。

**数学推导示例：** 假设推荐列表为 $[A, B, C, D, E]$（K=5），用户实际喜欢的物品集合为 $\{A, C, F\}$。则命中数为 2（A 和 C），Precision@5 = 2/5 = 0.4。

- 场景：适用于高度重视推荐结果准确性的场景，例如电商商品推荐、付费广告推荐，希望用户点击或购买的物品尽可能都是他们感兴趣的。

**Precision@K 的局限性：** 它不考虑推荐列表的排序顺序，只关心"前K个中有多少是对的"。这意味着无论相关物品排在第1位还是第K位，对 Precision@K 的贡献相同。

# 2. Recall@K：兴趣覆盖度

- 原理：Recall@K 衡量的是用户喜欢的物品中，有多少被推荐系统成功发掘并放在了前 K 个推荐里。它关注的是系统覆盖用户兴趣范围的能力。
- 公式：

$$
Recall@K = \frac{\text{前} K \text{个推荐中用户喜欢的物品数量}}{\text{用户喜欢的物品总数}}
$$

同样需要对所有用户平均。

**与 Precision@K 的关系：** Recall@K 和 Precision@K 的分子相同，但分母不同。Recall 的分母是用户喜欢的物品总数（固定值），Precision 的分母是 K（固定值）。当 K 增大时，Recall 通常会增大（覆盖更多兴趣），Precision 可能增大或减小。

- 场景：适用于希望尽可能挖掘用户潜在兴趣、避免信息茧房的场景，例如新闻推荐、内容发现平台，旨在帮助用户发现更多他们可能感兴趣的新内容。

**Recall@K 的局限性：** 当用户喜欢的物品总数很多时，即使推荐了很多相关物品，Recall 也可能较低。此外，它也不考虑排序位置。

# 3. HitRate@K：简单命中率

- 原理：HitRate@K 是一个非常直观的指标。它只关心推荐的前 K个物品中，是否至少有一个是用户喜欢的（即"命中"）。它计算的是命中发生的用户比例。

公式：

$$
HitRate@K = \frac{\text{前} K \text{个推荐至少命中一个喜欢物品的用户数}}{\text{总用户数}}
$$

**数学理解：** 对每个用户 $u$，定义命中指示函数：

$$
hit_u = \begin{cases} 1, & \text{if } |R_u \cap G_u| \geq 1 \\ 0, & \text{otherwise} \end{cases}$$

其中 $R_u$ 为推荐列表，$G_u$ 为用户喜欢的物品集合。则 $HitRate@K = \frac{1}{N}\sum_{u=1}^{N} hit_u$。

- 场景：常用于快速的 A/B测试初期，或作为一项简单直观的指标向非技术背景的伙伴解释模型效果。因为它无法区分命中一个和命中多个的差异，所以在深度评估中通常会结合其他指标。

**HitRate@K 的局限与改进：** 它是二元指标（命中/未命中），无法反映命中程度。在评估时通常与 Recall@K、NDCG@K 配合使用，形成完整的评估体系。

# 4. NDCG@K：排名质量

- 原理：NDCG@K (Normalized Discounted Cumulative Gain) 不仅考虑推荐物品是否相关，还考虑了相关物品在推荐列表中的位置。它认为排名越靠前的相关物品，其价值越高，因此会赋予更高的权重；排名越靠后，价值会因折损而降低。最后会通过归一化处理，使结果落在[0, 1]区间，便于比较。

- 计算过程：

**CG@K (Cumulative Gain)：** 简单累加前 K 个物品的相关性分数：

$$
CG@K = \sum_{i=1}^{K} rel_i
$$

其中 $rel_i$ 为第 $i$ 个位置的物品相关性分数（如喜欢为1，不喜欢为0）。

**DCG@K (Discounted Cumulative Gain)：** 在 CG 的基础上，对每个物品的增益除以一个与其排名位置有关的折损因子：

$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

折损因子 $\log_2(i+1)$ 使得排名靠前的物品贡献更大。位置 1 的折损为 $\log_2(2)=1$，位置 2 的折损为 $\log_2(3)\approx1.585$，位置 10 的折损为 $\log_2(11)\approx3.459$。

**IDCG@K (Ideal DCG)：** 将所有物品按照真实相关性降序排列，计算前 K 个物品的 DCG，这是理论上可能达到的最大值：

$$
IDCG@K = \sum_{i=1}^{K} \frac{rel_i^{ideal}}{\log_2(i+1)}
$$

**NDCG@K (Normalized DCG)：** 将 DCG 与 IDCG 相比进行归一化，消除列表长度等因素的影响：

$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

NDCG@K 的值域为 $[0, 1]$，1 表示完美排序，0 表示完全无序。

**完整计算示例：** 假设推荐列表 $[A, B, C, D, E]$（K=5），相关性标签为 $[3, 2, 0, 1, 2]$（分数越高越相关）。

- $DCG@5 = \frac{3}{\log_2 2} + \frac{2}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} + \frac{2}{\log_2 6} = 3 + 1.26 + 0 + 0.43 + 0.77 = 5.46$
- 理想排序为 $[3, 2, 2, 1, 0]$，$IDCG@5 = 3 + 1.26 + 0.77 + 0.43 + 0 = 5.46$
- $NDCG@5 = 5.46 / 5.46 = 1.0$（恰好是最优排序）

- 场景：适用于对排序位置非常敏感的场景，例如搜索引擎的结果列表、流媒体音乐或视频应用的主页推荐，这些场景中排在最前面的几个结果至关重要

# 如何选择评估指标

选择哪些指标，主要取决于你的推荐目标和业务场景：

- 追求极致准确：优先看 Precision@K
- 希望全面覆盖用户兴趣：重点关注 Recall@K
- 排序位置至关重要：NDCG@K 是最佳选择之一
- 快速验证和直观展示：可以先用 HitRate@K

# Python 代码实现

```python
import numpy as np

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0.0

def hit_rate_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return 1.0 if len(set(recommended_k) & set(relevant)) > 0 else 0.0

def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommendations(all_recommended, all_relevant, k):
    precisions = [precision_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    recalls = [recall_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    hit_rates = [hit_rate_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    ndcgs = [ndcg_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        f'HitRate@{k}': np.mean(hit_rates),
        f'NDCG@{k}': np.mean(ndcgs)
    }

all_rec = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 2, 3, 4, 5]]
all_rel = [[1, 2, 3], [3, 5, 7], [5, 6, 7, 8]]
results = evaluate_recommendations(all_rec, all_rel, k=5)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

# 常见误区与注意事项

1. **NDCG 的 IDCG 为 0 的情况：** 当用户没有喜欢的物品时，IDCG 为 0，此时 NDCG 无定义。实践中需要过滤掉这类用户或单独处理。
2. **相关性标签的设定：** NDCG 支持多级相关性（如 0/1/2/3），而不仅仅是二值的"喜欢/不喜欢"。多级相关性在搜索场景中更常见。
3. **指标间的权衡：** 提高 Precision 往往会降低 Recall，反之亦然。不同业务场景对两者的重视程度不同。
4. **离线指标与线上效果的相关性：** 离线评估指标（如 NDCG）与线上业务指标（如 CTR、转化率）之间存在 gap，需要结合 A/B 测试验证。

# 第六章：推荐基础八股算法

# 6.1 树模型面试题
