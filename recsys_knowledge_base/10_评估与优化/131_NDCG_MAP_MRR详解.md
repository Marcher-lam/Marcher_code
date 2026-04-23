# NDCG、MAP、MRR 详解 学习文档

## 1. 基础认知

### 1.1 为什么需要排序指标？

推荐系统不仅要判断用户**是否喜欢**，还要考虑**喜欢程度**和**排序位置**。排序指标能够评估推荐结果的质量。

### 1.2 常见排序指标

| 指标 | 关注点 | 适用场景 |
|------|--------|----------|
| MRR | 第一个正确位置 | 搜索、问答 |
| MAP | 所有正确位置的平均 | 信息检索 |
| NDCG | 位置+相关性级别 | 推荐系统 |
| Hit@K | Top-K 命中 | 快速评估 |

## 2. MRR (Mean Reciprocal Rank)

### 2.1 定义

MRR 关注**第一个正确答案**出现的位置。

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

其中：
- $|Q|$：查询数量
- $rank_i$：第 i 个查询的第一个正确答案的排名

### 2.2 实现

```python
import numpy as np
from typing import List, Dict


def compute_mrr(recommendations: List[List],
               ground_truth: List[List]) -> float:
    """
    计算 MRR

    参数:
        recommendations: 每个用户的推荐列表 [[item1, item2, ...], ...]
        ground_truth: 每个用户的真实交互列表 [[item1, item2, ...], ...]

    返回:
        MRR 分数
    """
    rr_sum = 0.0

    for recs, truth in zip(recommendations, ground_truth):
        truth_set = set(truth)

        # 找到第一个命中的位置
        rr = 0.0
        for i, item in enumerate(recs):
            if item in truth_set:
                rr = 1.0 / (i + 1)
                break

        rr_sum += rr

    return rr_sum / len(recommendations) if recommendations else 0.0


class MRREvaluator:
    """
    MRR 评估器
    """

    def __init__(self):
        self.rrs = []

    def add(self, recommendations: List, ground_truth: List):
        """添加一个用户的评估结果"""
        truth_set = set(ground_truth)

        rr = 0.0
        for i, item in enumerate(recommendations):
            if item in truth_set:
                rr = 1.0 / (i + 1)
                break

        self.rrs.append(rr)

    def compute(self) -> float:
        """计算 MRR"""
        return np.mean(self.rrs) if self.rrs else 0.0
```

### 2.3 示例

```python
def example_mrr():
    """MRR 示例"""
    # 三个用户的推荐和真实结果
    recommendations = [
        ['item1', 'item2', 'item3', 'item4'],  # 用户1
        ['item5', 'item6', 'item1', 'item2'],  # 用户2
        ['item7', 'item8', 'item9', 'item1'],  # 用户3
    ]

    ground_truth = [
        ['item2'],  # 用户1的真实点击
        ['item1'],  # 用户2的真实点击
        ['item1'],  # 用户3的真实点击
    ]

    # 计算每个用户的 RR
    # 用户1: item2 在位置2, RR = 1/2 = 0.5
    # 用户2: item1 在位置3, RR = 1/3 = 0.333
    # 用户3: item1 在位置4, RR = 1/4 = 0.25

    mrr = compute_mrr(recommendations, ground_truth)
    print(f"MRR: {mrr:.4f}")  # (0.5 + 0.333 + 0.25) / 3 ≈ 0.361

    # 手动验证
    print(f"验证: {(1/2 + 1/3 + 1/4) / 3:.4f}")


if __name__ == "__main__":
    example_mrr()
```

## 3. MAP (Mean Average Precision)

### 3.1 定义

MAP 关注**所有正确答案**的位置，计算每个正确答案出现时的精确率。

$$AP = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot rel(k)$$

$$MAP = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP_i$$

其中：
- $|R|$：相关物品数量
- $P(k)$：前 k 个位置的精确率
- $rel(k)$：第 k 位置是否相关（0 或 1）

### 3.2 实现

```python
def compute_ap(recommendations: List, ground_truth: List) -> float:
    """
    计算单个用户的 AP (Average Precision)

    参数:
        recommendations: 推荐列表
        ground_truth: 真实列表

    返回:
        AP 分数
    """
    truth_set = set(ground_truth)
    n_relevant = len(truth_set)

    if n_relevant == 0:
        return 0.0

    ap_sum = 0.0
    hits = 0

    for i, item in enumerate(recommendations):
        if item in truth_set:
            hits += 1
            # 精确率 @ position
            precision_at_k = hits / (i + 1)
            ap_sum += precision_at_k

    return ap_sum / n_relevant


def compute_map(recommendations: List[List],
               ground_truth: List[List]) -> float:
    """
    计算 MAP (Mean Average Precision)

    参数:
        recommendations: 每个用户的推荐列表
        ground_truth: 每个用户的真实列表

    返回:
        MAP 分数
    """
    aps = []

    for recs, truth in zip(recommendations, ground_truth):
        ap = compute_ap(recs, truth)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


class MAPEvaluator:
    """
    MAP 评估器
    """

    def __init__(self):
        self.aps = []

    def add(self, recommendations: List, ground_truth: List):
        """添加一个用户的评估结果"""
        ap = compute_ap(recommendations, ground_truth)
        self.aps.append(ap)

    def compute(self) -> float:
        """计算 MAP"""
        return np.mean(self.aps) if self.aps else 0.0
```

### 3.3 示例

```python
def example_map():
    """MAP 示例"""
    recommendations = [
        ['item1', 'item2', 'item3', 'item4', 'item5'],
        ['item6', 'item7', 'item1', 'item2', 'item3'],
    ]

    ground_truth = [
        ['item1', 'item3', 'item5'],  # 用户1有3个相关物品
        ['item1', 'item3'],           # 用户2有2个相关物品
    ]

    # 用户1分析:
    # position 1: item1 命中, P@1 = 1/1 = 1.0
    # position 3: item3 命中, P@3 = 2/3 = 0.667
    # position 5: item5 命中, P@5 = 3/5 = 0.6
    # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756

    # 用户2分析:
    # position 3: item1 命中, P@3 = 1/3 = 0.333
    # position 5: item3 命中, P@5 = 2/5 = 0.4
    # AP = (0.333 + 0.4) / 2 = 0.367

    map_score = compute_map(recommendations, ground_truth)
    print(f"MAP: {map_score:.4f}")

    # 手动验证
    ap1 = (1/1 + 2/3 + 3/5) / 3
    ap2 = (1/3 + 2/5) / 2
    print(f"验证: AP1={ap1:.4f}, AP2={ap2:.4f}")
    print(f"验证: MAP={(ap1 + ap2) / 2:.4f}")


if __name__ == "__main__":
    example_map()
```

## 4. NDCG (Normalized Discounted Cumulative Gain)

### 4.1 定义

NDCG 考虑**相关性级别**和**位置衰减**，是最常用的排序指标。

**CG (Cumulative Gain):**
$$CG@k = \sum_{i=1}^{k} rel_i$$

**DCG (Discounted CG):**
$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

**IDCG (Ideal DCG):**
$$IDCG@k = DCG@k \text{ of ideal ranking}$$

**NDCG (Normalized DCG):**
$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

### 4.2 实现

```python
import math
from typing import List, Dict


def compute_dcg(relevance_scores: List[float], k: int = None) -> float:
    """
    计算 DCG

    参数:
        relevance_scores: 相关性分数列表（按推荐顺序）
        k: 截断位置

    返回:
        DCG 分数
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        # 位置从1开始
        position = i + 1
        # 折扣因子
        discount = math.log2(position + 1)
        # 增益
        gain = (2 ** rel - 1) / discount
        dcg += gain

    return dcg


def compute_ndcg(recommendations: List, relevance_dict: Dict,
                k: int = None) -> float:
    """
    计算 NDCG

    参数:
        recommendations: 推荐物品列表
        relevance_dict: {item: relevance_score}
        k: 截断位置

    返回:
        NDCG 分数
    """
    if k is not None:
        recommendations = recommendations[:k]

    # 获取推荐物品的相关性分数
    rec_relevance = [relevance_dict.get(item, 0) for item in recommendations]

    # 计算 DCG
    dcg = compute_dcg(rec_relevance)

    # 计算理想情况下的 DCG
    ideal_relevance = sorted(relevance_dict.values(), reverse=True)
    if k is not None:
        ideal_relevance = ideal_relevance[:k]
    idcg = compute_dcg(ideal_relevance)

    # 计算 NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_ndcg_binary(recommendations: List, ground_truth: List,
                       k: int = None) -> float:
    """
    计算二值相关性 NDCG（简化版本）

    参数:
        recommendations: 推荐列表
        ground_truth: 真实列表
        k: 截断位置

    返回:
        NDCG 分数
    """
    truth_set = set(ground_truth)
    relevance_dict = {item: 1 for item in truth_set}

    return compute_ndcg(recommendations, relevance_dict, k)


class NDCGEvaluator:
    """
    NDCG 评估器
    """

    def __init__(self, k_list: List[int] = [5, 10, 20]):
        """
        参数:
            k_list: 要计算的 K 值列表
        """
        self.k_list = k_list
        self.ndcg_sums = {k: 0.0 for k in k_list}
        self.n_users = 0

    def add(self, recommendations: List, relevance_dict: Dict):
        """添加一个用户的评估结果"""
        for k in self.k_list:
            ndcg = compute_ndcg(recommendations, relevance_dict, k)
            self.ndcg_sums[k] += ndcg

        self.n_users += 1

    def add_binary(self, recommendations: List, ground_truth: List):
        """添加二值相关性的评估结果"""
        truth_set = set(ground_truth)
        relevance_dict = {item: 1 for item in truth_set}
        self.add(recommendations, relevance_dict)

    def compute(self) -> Dict[int, float]:
        """计算各 K 的 NDCG"""
        if self.n_users == 0:
            return {k: 0.0 for k in self.k_list}

        return {k: self.ndcg_sums[k] / self.n_users for k in self.k_list}
```

### 4.3 示例

```python
def example_ndcg():
    """NDCG 示例"""
    # 推荐列表
    recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']

    # 相关性分数（0-5分制）
    relevance = {
        'item1': 3,  # 高度相关
        'item2': 0,  # 不相关
        'item3': 2,  # 中度相关
        'item4': 0,  # 不相关
        'item5': 1,  # 低度相关
    }

    # 计算不同 K 的 NDCG
    for k in [1, 3, 5]:
        ndcg = compute_ndcg(recommendations, relevance, k)
        print(f"NDCG@{k}: {ndcg:.4f}")

    # 分析:
    # DCG@5 = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4) + ...
    #       = 7/1 + 0 + 3/2 + 0 + 1/log2(6)
    #       = 7 + 1.5 + 0.387 = 8.887

    # 理想排序: [3, 2, 1, 0, 0]
    # IDCG@5 = 7/1 + 3/2 + 1/log2(4) = 7 + 1.5 + 0.5 = 9

    # NDCG@5 = 8.887 / 9 = 0.987

    # 验证
    dcg = compute_dcg([3, 0, 2, 0, 1])
    idcg = compute_dcg([3, 2, 1, 0, 0])
    print(f"\n验证: DCG={dcg:.4f}, IDCG={idcg:.4f}, NDCG={dcg/idcg:.4f}")


if __name__ == "__main__":
    example_ndcg()
```

## 5. 综合评估器

### 5.1 统一评估接口

```python
from typing import List, Dict, Tuple
import numpy as np


class RankingMetricsEvaluator:
    """
    排序指标综合评估器
    """

    def __init__(self, k_list: List[int] = [1, 5, 10, 20, 50]):
        """
        参数:
            k_list: 要评估的 K 值列表
        """
        self.k_list = k_list

    def evaluate(self, recommendations: List[List],
                ground_truth: List[List],
                relevance: List[Dict] = None) -> Dict:
        """
        综合评估

        参数:
            recommendations: 每个用户的推荐列表
            ground_truth: 每个用户的真实列表
            relevance: 可选的相关性分数字典列表

        返回:
            指标字典
        """
        metrics = {}

        # MRR
        mrr = compute_mrr(recommendations, ground_truth)
        metrics['MRR'] = mrr

        # MAP
        map_score = compute_map(recommendations, ground_truth)
        metrics['MAP'] = map_score

        # NDCG@K, Precision@K, Recall@K, Hit@K
        for k in self.k_list:
            ndcgs = []
            precisions = []
            recalls = []
            hits = []

            for i, (recs, truth) in enumerate(zip(recommendations, ground_truth)):
                recs_k = recs[:k]
                truth_set = set(truth)

                # Hit@K
                hit = 1 if set(recs_k) & truth_set else 0
                hits.append(hit)

                # Precision@K
                n_hit = len(set(recs_k) & truth_set)
                precisions.append(n_hit / k)

                # Recall@K
                if len(truth) > 0:
                    recalls.append(n_hit / len(truth))
                else:
                    recalls.append(0)

                # NDCG@K
                if relevance and i < len(relevance):
                    ndcg = compute_ndcg(recs, relevance[i], k)
                else:
                    ndcg = compute_ndcg_binary(recs, truth, k)
                ndcgs.append(ndcg)

            metrics[f'NDCG@{k}'] = np.mean(ndcgs)
            metrics[f'Precision@{k}'] = np.mean(precisions)
            metrics[f'Recall@{k}'] = np.mean(recalls)
            metrics[f'Hit@{k}'] = np.mean(hits)

        return metrics

    def print_metrics(self, metrics: Dict):
        """打印指标"""
        print("\n" + "=" * 50)
        print("排序指标评估结果")
        print("=" * 50)

        print(f"\nMRR: {metrics['MRR']:.4f}")
        print(f"MAP: {metrics['MAP']:.4f}")

        print("\n按位置 K 的指标:")
        print("-" * 50)
        print(f"{'K':<8} {'NDCG':<12} {'Precision':<12} {'Recall':<12} {'Hit':<8}")
        print("-" * 50)

        for k in self.k_list:
            print(f"{k:<8} "
                  f"{metrics[f'NDCG@{k}']:<12.4f} "
                  f"{metrics[f'Precision@{k}']:<12.4f} "
                  f"{metrics[f'Recall@{k}']:<12.4f} "
                  f"{metrics[f'Hit@{k}']:<8.4f}")


# 使用示例
def demo_evaluation():
    """综合评估示例"""
    np.random.seed(42)

    # 生成模拟数据
    n_users = 100
    n_items = 1000

    recommendations = []
    ground_truth = []

    for _ in range(n_users):
        # 推荐列表
        recs = list(np.random.choice(n_items, 20, replace=False))
        recommendations.append([f'item{i}' for i in recs])

        # 真实交互（部分与推荐重叠）
        truth_size = np.random.randint(1, 5)
        truth = list(np.random.choice(n_items, truth_size, replace=False))
        ground_truth.append([f'item{i}' for i in truth])

    # 评估
    evaluator = RankingMetricsEvaluator(k_list=[1, 5, 10, 20])
    metrics = evaluator.evaluate(recommendations, ground_truth)
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    demo_evaluation()
```

## 6. 指标选择指南

### 6.1 各指标特点

| 指标 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| MRR | 简单、直观 | 只关注第一个 | 搜索、问答 |
| MAP | 考虑所有位置 | 假设二元相关 | 信息检索 |
| NDCG | 支持分级相关 | 计算复杂 | 推荐、广告 |

### 6.2 选择建议

```python
def select_metric(task_type: str) -> List[str]:
    """
    根据任务类型选择指标

    参数:
        task_type: 任务类型

    返回:
        推荐的指标列表
    """
    guide = {
        'search': ['MRR', 'NDCG@10', 'Precision@10'],
        'recommendation': ['NDCG@10', 'NDCG@20', 'Recall@10', 'Hit@10'],
        'qa': ['MRR', 'Accuracy@1'],
        'ad_ranking': ['NDCG', 'AUC', 'GAUC'],
    }

    return guide.get(task_type, ['NDCG@10', 'MRR'])
```

## 7. 学习总结

### 7.1 核心要点

1. **MRR**：关注第一个正确位置
2. **MAP**：关注所有正确位置的平均精确率
3. **NDCG**：考虑相关性和位置衰减

### 7.2 记忆技巧

- MRR = 第一个命中位置的倒数
- MAP = 命中时的精确率平均
- NDCG = 带位置折扣的增益 / 理想情况

## 8. 练习题

1. 手动计算一个推荐列表的 MRR、MAP、NDCG@5。

2. 比较不同排序策略对指标的影响。

3. 实现一个支持多级相关性的完整评估框架。
