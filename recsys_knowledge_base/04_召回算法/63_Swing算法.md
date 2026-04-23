# Swing算法 学习文档

## 1. Swing算法概述

### 1.1 什么是Swing算法？

```
Swing算法:

- 阿里提出的经典召回算法
- 基于物品共现 + 用户权重
- 解决传统ItemCF的流行度偏差

核心思想:
- 两个物品被同一用户点击，说明有相关性
- 如果被多个用户点击，相关性更强
- 但如果这些用户本身很活跃，相关性要打折

与其他方法对比:
- ItemCF: 只看物品共现，不考虑用户权重
- Swing: 共现 + 用户活跃度惩罚
```

### 1.2 算法原理

```python
"""
Swing 公式:

Swing(i, j) = Σ_{u ∈ U_i ∩ U_j} Σ_{v ∈ U_i ∩ U_j, v > u} 1 / (α + |I_u ∩ I_v|)

其中:
- U_i: 点击过物品 i 的用户集合
- U_j: 点击过物品 j 的用户集合
- I_u: 用户 u 点击过的物品集合
- α: 平滑参数

解释:
- 分子: 两用户同时点击 i 和 j
- 分母: 两用户的公共物品数（活跃度惩罚）
- 公共物品越多，说明用户很活跃，贡献打折
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from itertools import combinations


class SwingAlgorithm:
    """
    Swing 算法实现
    """

    def __init__(self, alpha: float = 1.0, top_k: int = 100):
        """
        参数:
            alpha: 平滑参数
            top_k: 每个物品保留的相似物品数
        """
        self.alpha = alpha
        self.top_k = top_k

        # 物品相似度矩阵
        self.item_similarities: Dict[int, Dict[int, float]] = {}

    def fit(self, interactions: List[Tuple[int, int]]):
        """
        训练

        interactions: [(user_id, item_id), ...]
        """
        # 构建用户-物品倒排索引
        user_items: Dict[int, Set[int]] = defaultdict(set)
        item_users: Dict[int, Set[int]] = defaultdict(set)

        for user_id, item_id in interactions:
            user_items[user_id].add(item_id)
            item_users[item_id].add(user_id)

        # 计算 Swing 分数
        item_swing: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        items = list(item_users.keys())
        n_items = len(items)

        print(f"Computing Swing for {n_items} items...")

        for i, item_i in enumerate(items):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{n_items}")

            users_i = item_users[item_i]

            for item_j in items:
                if item_j <= item_i:  # 只计算上三角
                    continue

                users_j = item_users[item_j]

                # 共同用户
                common_users = users_i & users_j

                if len(common_users) < 2:
                    continue

                # 计算两两用户组合的 Swing
                swing_score = 0.0

                for u, v in combinations(common_users, 2):
                    # 用户 u 和 v 的公共物品数
                    common_items = len(user_items[u] & user_items[v])

                    # Swing 分数
                    swing_score += 1.0 / (self.alpha + common_items)

                if swing_score > 0:
                    item_swing[item_i][item_j] = swing_score
                    item_swing[item_j][item_i] = swing_score

        # 保留 Top-K
        print("Selecting top-k similar items...")
        for item_id in item_swing:
            sims = item_swing[item_id]
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            self.item_similarities[item_id] = dict(sorted_sims[:self.top_k])

        print("Swing computation completed!")

        return self

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取相似物品

        返回: [(item_id, similarity), ...]
        """
        if item_id not in self.item_similarities:
            return []

        sims = self.item_similarities[item_id]
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        return sorted_sims[:top_k]

    def recall(self, user_history: List[int], top_k: int = 50
              ) -> List[Tuple[int, float]]:
        """
        召回

        基于用户历史物品召回相似物品
        """
        candidate_scores: Dict[int, float] = defaultdict(float)

        for item_id in user_history:
            if item_id not in self.item_similarities:
                continue

            for sim_item, score in self.item_similarities[item_id].items():
                if sim_item not in user_history:  # 排除已交互物品
                    candidate_scores[sim_item] += score

        # 排序
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_candidates[:top_k]
```

## 2. 优化实现

### 2.1 并行计算优化

```python
from multiprocessing import Pool
from functools import partial


class ParallelSwing:
    """
    并行 Swing 算法
    """

    def __init__(self, alpha: float = 1.0, top_k: int = 100,
                 n_workers: int = 4):
        self.alpha = alpha
        self.top_k = top_k
        self.n_workers = n_workers
        self.item_similarities = {}

    def _compute_item_swing(self, item_i: int, item_users: Dict[int, Set[int]],
                           user_items: Dict[int, Set[int]],
                           all_items: List[int]) -> Dict[int, float]:
        """计算单个物品的 Swing"""
        sims = {}
        users_i = item_users[item_i]

        for item_j in all_items:
            if item_j <= item_i:
                continue

            users_j = item_users[item_j]
            common_users = users_i & users_j

            if len(common_users) < 2:
                continue

            swing_score = 0.0
            for u, v in combinations(common_users, 2):
                common_items = len(user_items[u] & user_items[v])
                swing_score += 1.0 / (self.alpha + common_items)

            if swing_score > 0:
                sims[item_j] = swing_score

        return sims

    def fit(self, interactions: List[Tuple[int, int]]):
        """并行训练"""
        # 构建索引
        user_items = defaultdict(set)
        item_users = defaultdict(set)

        for user_id, item_id in interactions:
            user_items[user_id].add(item_id)
            item_users[item_id].add(user_id)

        all_items = list(item_users.keys())

        # 并行计算
        with Pool(self.n_workers) as pool:
            func = partial(
                self._compute_item_swing,
                item_users=item_users,
                user_items=user_items,
                all_items=all_items
            )

            results = pool.map(func, all_items)

        # 汇总结果
        for item_id, sims in zip(all_items, results):
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            self.item_similarities[item_id] = dict(sorted_sims[:self.top_k])

            # 同时更新对称项
            for sim_item, score in sims.items():
                if sim_item not in self.item_similarities:
                    self.item_similarities[sim_item] = {}
                self.item_similarities[sim_item][item_id] = score

        return self
```

### 2.2 稀疏矩阵优化

```python
from scipy.sparse import csr_matrix, lil_matrix


class SparseSwing:
    """
    基于稀疏矩阵的 Swing 实现
    """

    def __init__(self, alpha: float = 1.0, top_k: int = 100):
        self.alpha = alpha
        self.top_k = top_k
        self.similarity_matrix = None
        self.item_mapping = {}
        self.reverse_mapping = {}

    def fit(self, interactions: List[Tuple[int, int]],
            n_users: int, n_items: int):
        """
        训练

        使用稀疏矩阵优化存储和计算
        """
        # 构建稀疏交互矩阵
        rows = []
        cols = []
        for user_id, item_id in interactions:
            rows.append(user_id)
            cols.append(item_id)

        # 用户-物品矩阵
        ui_matrix = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_users, n_items)
        )

        # 物品-物品共现矩阵
        # II = UI.T @ UI
        ii_matrix = ui_matrix.T @ ui_matrix

        # 计算用户权重矩阵
        # 用户活跃度 = 每个用户的交互物品数
        user_activity = np.array(ui_matrix.sum(axis=1)).flatten()

        # 计算每个物品对的 Swing
        print("Computing Swing scores...")

        similarity = lil_matrix((n_items, n_items))

        # 只计算非零共现的物品对
        ii_coo = ii_matrix.tocoo()

        for i, j, count in zip(ii_coo.row, ii_coo.col, ii_coo.data):
            if i >= j:
                continue

            # 获取共同用户
            users_i = ui_matrix[:, i].nonzero()[0]
            users_j = ui_matrix[:, j].nonzero()[0]
            common_users = set(users_i) & set(users_j)

            if len(common_users) < 2:
                continue

            # 计算 Swing
            swing = 0.0
            for u, v in combinations(common_users, 2):
                # 用户 u 和 v 的公共物品数
                items_u = set(ui_matrix[u].nonzero()[1])
                items_v = set(ui_matrix[v].nonzero()[1])
                common = len(items_u & items_v)

                swing += 1.0 / (self.alpha + common)

            similarity[i, j] = swing
            similarity[j, i] = swing

        # 转为 CSR 格式
        self.similarity_matrix = similarity.tocsr()

        print("Swing computation completed!")

        return self

    def get_similar_items(self, item_id: int, top_k: int = 10
                         ) -> List[Tuple[int, float]]:
        """获取相似物品"""
        if self.similarity_matrix is None:
            return []

        row = self.similarity_matrix[item_id].toarray().flatten()
        top_indices = np.argsort(row)[::-1][:top_k]

        return [(int(idx), float(row[idx])) for idx in top_indices if row[idx] > 0]
```

## 3. Swing 变体

### 3.1 Time-aware Swing

```python
class TimeAwareSwing:
    """
    时间感知 Swing

    考虑行为的时间衰减
    """

    def __init__(self, alpha: float = 1.0, time_decay: float = 0.1,
                 top_k: int = 100):
        self.alpha = alpha
        self.time_decay = time_decay
        self.top_k = top_k
        self.item_similarities = {}

    def fit(self, interactions: List[Tuple[int, int, float]]):
        """
        训练

        interactions: [(user_id, item_id, timestamp), ...]
        """
        # 按用户组织数据
        user_data = defaultdict(list)
        for user_id, item_id, timestamp in interactions:
            user_data[user_id].append((item_id, timestamp))

        # 排序，获取最近时间
        max_time = max(t for _, _, t in interactions)

        # 构建索引
        item_users = defaultdict(set)
        user_items = defaultdict(set)

        for user_id, items in user_data.items():
            for item_id, _ in items:
                item_users[item_id].add(user_id)
                user_items[user_id].add(item_id)

        # 计算带时间衰减的 Swing
        item_swing = defaultdict(lambda: defaultdict(float))
        items = list(item_users.keys())

        for item_i in items:
            users_i = item_users[item_i]

            for item_j in items:
                if item_j <= item_i:
                    continue

                users_j = item_users[item_j]
                common_users = users_i & users_j

                if len(common_users) < 2:
                    continue

                swing = 0.0
                for u, v in combinations(common_users, 2):
                    # 用户公共物品数
                    common_items = len(user_items[u] & user_items[v])

                    # 时间衰减
                    # 获取用户对这两个物品的交互时间
                    times_u = {item: t for item, t in user_data[u]}
                    times_v = {item: t for item, t in user_data[v]}

                    time_u = max(times_u.get(item_i, 0), times_u.get(item_j, 0))
                    time_v = max(times_v.get(item_i, 0), times_v.get(item_j, 0))

                    decay_u = np.exp(-self.time_decay * (max_time - time_u))
                    decay_v = np.exp(-self.time_decay * (max_time - time_v))

                    swing += (decay_u + decay_v) / 2 / (self.alpha + common_items)

                if swing > 0:
                    item_swing[item_i][item_j] = swing
                    item_swing[item_j][item_i] = swing

        # 保留 Top-K
        for item_id in item_swing:
            sims = item_swing[item_id]
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            self.item_similarities[item_id] = dict(sorted_sims[:self.top_k])

        return self
```

### 3.2 Category-aware Swing

```python
class CategoryAwareSwing:
    """
    类目感知 Swing

    同一类目的物品权重更高
    """

    def __init__(self, alpha: float = 1.0, category_boost: float = 2.0,
                 top_k: int = 100):
        self.alpha = alpha
        self.category_boost = category_boost
        self.top_k = top_k
        self.item_similarities = {}
        self.item_categories = {}

    def set_categories(self, item_categories: Dict[int, int]):
        """设置物品类目"""
        self.item_categories = item_categories

    def fit(self, interactions: List[Tuple[int, int]]):
        """训练"""
        user_items = defaultdict(set)
        item_users = defaultdict(set)

        for user_id, item_id in interactions:
            user_items[user_id].add(item_id)
            item_users[item_id].add(user_id)

        item_swing = defaultdict(lambda: defaultdict(float))
        items = list(item_users.keys())

        for item_i in items:
            users_i = item_users[item_i]
            cat_i = self.item_categories.get(item_i, -1)

            for item_j in items:
                if item_j <= item_i:
                    continue

                users_j = item_users[item_j]
                cat_j = self.item_categories.get(item_j, -1)
                common_users = users_i & users_j

                if len(common_users) < 2:
                    continue

                swing = 0.0
                for u, v in combinations(common_users, 2):
                    common_items = len(user_items[u] & user_items[v])
                    swing += 1.0 / (self.alpha + common_items)

                # 类目加权
                if cat_i == cat_j and cat_i != -1:
                    swing *= self.category_boost

                if swing > 0:
                    item_swing[item_i][item_j] = swing
                    item_swing[item_j][item_i] = swing

        for item_id in item_swing:
            sims = item_swing[item_id]
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            self.item_similarities[item_id] = dict(sorted_sims[:self.top_k])

        return self
```

## 4. 实际应用

### 4.1 完整召回流程

```python
class SwingRecallPipeline:
    """
    Swing 召回流水线
    """

    def __init__(self, alpha: float = 1.0, top_k: int = 200):
        self.swing = SwingAlgorithm(alpha=alpha, top_k=top_k)
        self.user_history = defaultdict(list)

    def fit(self, interactions: List[Tuple[int, int]]):
        """训练"""
        # 记录用户历史
        for user_id, item_id in interactions:
            self.user_history[user_id].append(item_id)

        # 训练 Swing
        self.swing.fit(interactions)

        return self

    def recall(self, user_id: int, n: int = 100,
               exclude_history: bool = True) -> List[Tuple[int, float]]:
        """
        召回
        """
        history = self.user_history.get(user_id, [])
        candidates = self.swing.recall(history, top_k=n * 2)

        if exclude_history:
            history_set = set(history)
            candidates = [(i, s) for i, s in candidates if i not in history_set]

        return candidates[:n]

    def batch_recall(self, user_ids: List[int], n: int = 100
                    ) -> Dict[int, List[Tuple[int, float]]]:
        """批量召回"""
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recall(user_id, n)
        return results


def evaluate_swing(model: SwingRecallPipeline, test_interactions: List[Tuple[int, int]],
                   k_list: List[int] = [10, 50, 100]) -> Dict[str, float]:
    """
    评估 Swing 召回效果
    """
    # 构建测试数据
    test_user_items = defaultdict(set)
    for user_id, item_id in test_interactions:
        test_user_items[user_id].add(item_id)

    metrics = {f'Recall@{k}': [] for k in k_list}

    for user_id, true_items in test_user_items.items():
        recalled = model.recall(user_id, n=max(k_list))
        recalled_items = set(i for i, _ in recalled)

        for k in k_list:
            top_k = set(i for i, _ in recalled[:k])
            hit = len(top_k & true_items)
            recall = hit / len(true_items) if true_items else 0
            metrics[f'Recall@{k}'].append(recall)

    # 平均
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics
```

## 5. 学习总结

### 5.1 核心要点

```
1. Swing = 共现 + 用户活跃度惩罚
2. 活跃用户的贡献要打折
3. 适合电商等流行度偏差严重的场景
4. 计算复杂度较高，需要优化
```

### 5.2 与 ItemCF 对比

```
特性          ItemCF          Swing
──────────────────────────────────────
共现计算      简单计数        用户组合
用户权重      无              活跃度惩罚
流行度偏差    有              较好缓解
计算复杂度    O(n²)           O(n² * u²)
召回效果      一般            更准确
```

### 5.3 最佳实践

```
1. α 参数: 通常设为 1-5
2. top_k: 每个物品保留 100-200 个相似物品
3. 过滤: 移除共现次数太少的物品对
4. 类目: 可结合类目过滤提升准确性
5. 时间: 加入时间衰减处理时效性
```
