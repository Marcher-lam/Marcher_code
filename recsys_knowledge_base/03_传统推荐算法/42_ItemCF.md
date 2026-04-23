# ItemCF - 基于物品的协同过滤

> 电商推荐系统的经典选择

---

## 1. 算法基础认知

### 1.1 什么是ItemCF

**ItemCF（Item-based Collaborative Filtering，基于物品的协同过滤）的核心思想是：**

> "推荐和你之前喜欢的物品相似的物品"

### 1.2 直观理解

```
┌────────────────────────────────────────────────────────────┐
│                    ItemCF 直觉理解                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  物品关系：                                                │
│  ┌─────────────────────────────────────────────┐          │
│  │  iPhone 12 ←──相似──→ iPhone 13            │          │
│  │     ↑                      ↑               │          │
│  │   相似                   相似              │          │
│  │     ↓                      ↓               │          │
│  │  iPhone 11 ←──相似──→ iPhone 14            │          │
│  └─────────────────────────────────────────────┘          │
│                                                            │
│  用户行为：                                                │
│  - 用户A 买过 iPhone 12                                    │
│  - iPhone 12 和 iPhone 13 相似                            │
│  - 向用户A 推荐 iPhone 13                                  │
│                                                            │
│  推荐理由："你买过iPhone 12，可能也喜欢iPhone 13"         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.3 UserCF vs ItemCF

```
┌────────────────────────────────────────────────────────────┐
│                  UserCF vs ItemCF                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  UserCF: 找和你相似的人，推荐他们喜欢的                    │
│  ┌─────────────────────────────────────────────┐          │
│  │  你 ──相似──→ 用户B                         │          │
│  │               ↓                             │          │
│  │            喜欢物品X ──→ 推荐给你           │          │
│  └─────────────────────────────────────────────┘          │
│                                                            │
│  ItemCF: 找和你喜欢的物品相似的物品                        │
│  ┌─────────────────────────────────────────────┐          │
│  │  你喜欢物品A                                │          │
│  │       ↓                                     │          │
│  │  物品A ──相似──→ 物品B ──→ 推荐给你         │          │
│  └─────────────────────────────────────────────┘          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.4 在推荐系统中的位置

```
ItemCF 应用场景：

✅ 电商平台（淘宝、京东、亚马逊）
   - 物品相对稳定
   - 用户行为丰富
   - 推荐可解释性强

✅ 视频网站（Netflix、爱奇艺）
   - 视频内容相对稳定
   - 用户看相似视频

✅ 音乐平台（网易云、Spotify）
   - 歌曲相对稳定
   - 用户听相似歌曲
```

---

## 2. 核心原理

### 2.1 算法流程

```
ItemCF 算法步骤：

1. 构建用户-物品交互矩阵
2. 计算物品之间的相似度
3. 对于目标用户，找到他交互过的物品
4. 根据这些物品的相似物品，预测用户可能感兴趣的物品
5. 推荐预测评分最高的N个物品
```

### 2.2 物品相似度计算

**核心思想：两个物品被同一群用户喜欢/交互，则它们相似**

```
物品相似度公式（余弦相似度）：

sim(i, j) = cos(i, j) = (|U_i ∩ U_j|) / (√|U_i| × √|U_j|)

其中：
- U_i：喜欢物品i的用户集合
- U_j：喜欢物品j的用户集合
- |U_i ∩ U_j|：同时喜欢物品i和j的用户数

更常用的形式（考虑用户活跃度惩罚）：

sim(i, j) = (Σ_u∈U_i∩U_j 1/log(1+|I_u|)) / (√Σ_u∈U_i 1/log(1+|I_u|) × √Σ_u∈U_j 1/log(1+|I_u|))

其中：
- |I_u|：用户u交互过的物品数（活跃度）
- 活跃用户对相似度的贡献应该降低
```

### 2.3 为什么惩罚活跃用户

```
问题场景：

用户X是个"刷子"，点击了10000个商品
这会导致很多商品因为用户X的共同点击而被认为相似
但实际上这种相似是虚假的

解决方案：
- 活跃用户（交互多）的权重降低
- 使用 1/log(1+|I_u|) 作为权重
- 用户越活跃，权重越低
```

### 2.4 评分预测

```
预测用户u对物品i的评分：

p(u, i) = (Σ_j∈I_u sim(i, j) × r_u,j) / Σ_j∈I_u |sim(i, j)|

其中：
- I_u：用户u交互过的物品集合
- sim(i, j)：物品i和j的相似度
- r_u,j：用户u对物品j的评分

简单理解：
- 找用户交互过的所有物品
- 计算这些物品与目标物品的相似度
- 加权平均得到预测分数
```

---

## 3. 数学公式与推导

### 3.1 相似度计算详解

```python
# 示例：计算物品相似度

用户-物品交互矩阵（点击行为）：

           商品A  商品B  商品C  商品D  商品E
用户1        1      1      0      1      0
用户2        1      1      1      0      0
用户3        0      1      1      0      1
用户4        1      0      1      1      0

计算商品A和B的相似度：

U_A = {用户1, 用户2, 用户4}  # 点击过商品A的用户
U_B = {用户1, 用户2, 用户3}  # 点击过商品B的用户

|U_A| = 3
|U_B| = 3
|U_A ∩ U_B| = |{用户1, 用户2}| = 2

简单余弦相似度：
sim(A, B) = 2 / (√3 × √3) = 2/3 ≈ 0.667

带活跃度惩罚的相似度：
用户1：交互3个物品 → 权重 1/log(1+3) = 1/log(4) ≈ 0.72
用户2：交互3个物品 → 权重 1/log(4) ≈ 0.72

分子：0.72 + 0.72 = 1.44
分母：√(3×0.72) × √(3×0.72) = 1.47

sim(A, B) = 1.44 / 1.44 ≈ 1.0
```

### 3.2 推荐分数计算

```python
# 示例：为用户3推荐商品

用户3的交互历史：商品B、商品C、商品E

假设物品相似度矩阵：
        商品A  商品B  商品C  商品D  商品E
商品A   1.0    0.67   0.33   0.50   0.00
商品B   0.67   1.0    0.67   0.33   0.50
商品C   0.33   0.67   1.0    0.33   0.50
商品D   0.50   0.33   0.33   1.0    0.00
商品E   0.00   0.50   0.50   0.00   1.0

预测用户3对商品A的评分（用户3没交互过A）：

p(用户3, 商品A) =
    (sim(A,B)×r_B + sim(A,C)×r_C + sim(A,E)×r_E) /
    (sim(A,B) + sim(A,C) + sim(A,E))

假设r都是1（点击）：
= (0.67×1 + 0.33×1 + 0.00×1) / (0.67 + 0.33 + 0.00)
= 1.0 / 1.0
= 1.0

预测用户3对商品D的评分：

p(用户3, 商品D) =
    (sim(D,B)×1 + sim(D,C)×1 + sim(D,E)×1) /
    (sim(D,B) + sim(D,C) + sim(D,E))
= (0.33 + 0.33 + 0.00) / (0.33 + 0.33 + 0.00)
= 0.66 / 0.66
= 1.0
```

---

## 4. 完整算法实现步骤

### 4.1 离线计算流程

```
┌────────────────────────────────────────────────────────────┐
│                    ItemCF 离线流程                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Step 1: 数据准备                                          │
│  ├── 读取用户行为日志                                      │
│  ├── 构建用户-物品交互矩阵                                 │
│  └── 统计物品热度、用户活跃度                              │
│                                                            │
│  Step 2: 计算物品共现                                      │
│  ├── 遍历每个用户的行为序列                               │
│  ├── 统计物品两两共现次数                                 │
│  └── 存储：co_occur[i][j] = 共现次数                      │
│                                                            │
│  Step 3: 计算物品相似度                                    │
│  ├── 对共现矩阵归一化                                     │
│  ├── 应用活跃度惩罚                                       │
│  └── 存储：sim[i][j] = 相似度                             │
│                                                            │
│  Step 4: 构建物品相似物品列表                              │
│  ├── 对每个物品，取Top-K最相似物品                        │
│  └── 存储：similar_items[i] = [(j, sim), ...]            │
│                                                            │
│  输出：物品相似度表（用于在线服务）                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.2 在线推荐流程

```
┌────────────────────────────────────────────────────────────┐
│                    ItemCF 在线流程                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  用户请求推荐                                              │
│       │                                                    │
│       ▼                                                    │
│  Step 1: 获取用户历史行为                                  │
│  ├── 查询用户最近点击/购买/收藏的物品                     │
│  └── 取最近N个物品作为种子                                │
│                                                            │
│  Step 2: 召回相似物品                                      │
│  ├── 对每个种子物品，查找相似物品                         │
│  ├── 合并去重                                             │
│  └── 过滤用户已交互的物品                                 │
│                                                            │
│  Step 3: 计算推荐分数                                      │
│  ├── 对每个候选物品，计算推荐分数                         │
│  │   score = Σ sim(candidate, history_item)              │
│  └── 按分数排序                                           │
│                                                            │
│  Step 4: 返回Top-N                                         │
│  └── 返回推荐列表                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 5. 应用场景

### 5.1 最佳适用场景

| 场景 | 适用性 | 原因 |
|-----|-------|-----|
| **电商** | ⭐⭐⭐⭐⭐ | 物品稳定，用户买相似商品 |
| **视频** | ⭐⭐⭐⭐ | 看相似视频 |
| **音乐** | ⭐⭐⭐⭐ | 听相似歌曲 |
| **新闻** | ⭐⭐ | 新闻时效性强，物品不稳定 |
| **社交** | ⭐⭐ | 内容更新快 |

### 5.2 ItemCF在电商中的应用

```
淘宝/京东 商品详情页推荐：

┌────────────────────────────────────────────────────────────┐
│  iPhone 15 详情页                                         │
├────────────────────────────────────────────────────────────┤
│  [商品图片]                                               │
│  iPhone 15  ¥5999                                         │
│                                                            │
│  ─────────────────────────────────────────────────        │
│  看了又看                                                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                        │
│  │手机 │ │手机 │ │手机 │ │手机 │  ← 基于ItemCF          │
│  │壳  │ │膜  │ │充  │ │耳  │                           │
│  └─────┘ └─────┘ └─────┘ └─────┘                        │
│                                                            │
│  相似商品                                                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                        │
│  │iP15 │ │iP15 │ │iP14 │ │iP14 │  ← 基于ItemCF          │
│  │Pro │ │Plus│ │Pro │ │    │                           │
│  └─────┘ └─────┘ └─────┘ └─────┘                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|-----|------|
| **可解释性强** | "因为你买过X，推荐Y" |
| **稳定性好** | 物品相似度相对稳定 |
| **计算效率高** | 物品数通常小于用户数 |
| **适合电商** | 物品属性稳定，用户行为丰富 |
| **实时性好** | 相似度可预计算 |

### 6.2 缺点

| 缺点 | 说明 |
|-----|------|
| **冷启动** | 新物品没有交互，无法计算相似度 |
| **流行度偏差** | 热门物品容易与很多物品相似 |
| **多样性差** | 倾向于推荐相似物品，缺乏惊喜 |

### 6.3 UserCF vs ItemCF 复杂度对比

```
UserCF:
- 相似度矩阵：O(m²)，m = 用户数
- 适合：用户数 < 物品数

ItemCF:
- 相似度矩阵：O(n²)，n = 物品数
- 适合：物品数 < 用户数（电商场景）

举例：
- 淘宝：用户数 8亿，商品数 10亿
  - 但单个类目下，商品数远小于用户数
  - 按类目分别计算ItemCF更高效
```

---

## 7. 调库实现

### 7.1 基础实现

```python
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

class ItemCF:
    """
    基于物品的协同过滤
    """

    def __init__(self, k=10, sim_threshold=0.0):
        """
        参数:
        k: 每个物品保留的最相似物品数
        sim_threshold: 相似度阈值
        """
        self.k = k
        self.sim_threshold = sim_threshold
        self.item_similarity = defaultdict(dict)  # 物品相似度矩阵
        self.user_items = defaultdict(set)        # 用户交互过的物品
        self.item_users = defaultdict(set)        # 物品被哪些用户交互

    def fit(self, interactions):
        """
        训练模型

        参数:
        interactions: list of (user_id, item_id, rating) 或 DataFrame
        """
        # 构建用户-物品关系
        if isinstance(interactions, pd.DataFrame):
            for _, row in interactions.iterrows():
                user, item = row['user_id'], row['item_id']
                rating = row.get('rating', 1)
                self.user_items[user].add((item, rating))
                self.item_users[item].add(user)
        else:
            for user, item, rating in interactions:
                self.user_items[user].add((item, rating))
                self.item_users[item].add(user)

        # 计算物品共现矩阵
        item_pairs = defaultdict(float)  # (i, j) -> 共现次数（带权重）

        for user, items in self.user_items.items():
            items_list = list(items)
            # 用户活跃度惩罚
            user_weight = 1.0 / np.log(1 + len(items_list))

            # 两两组合
            for (item1, r1), (item2, r2) in combinations(items_list, 2):
                weight = user_weight
                item_pairs[(item1, item2)] += weight
                item_pairs[(item2, item1)] += weight

        # 计算物品相似度
        all_items = list(self.item_users.keys())

        for i in all_items:
            for j in all_items:
                if i >= j:
                    continue

                # 余弦相似度
                if (i, j) in item_pairs and (j, i) in item_pairs:
                    co_ij = item_pairs[(i, j)]
                    co_ji = item_pairs[(j, i)]

                    # |U_i| 和 |U_j|
                    n_i = len(self.item_users[i])
                    n_j = len(self.item_users[j])

                    # 带活跃度惩罚的分母
                    norm_i = np.sqrt(sum(
                        1.0/np.log(1 + len(self.user_items[u]))
                        for u in self.item_users[i]
                    ))
                    norm_j = np.sqrt(sum(
                        1.0/np.log(1 + len(self.user_items[u]))
                        for u in self.item_users[j]
                    ))

                    if norm_i > 0 and norm_j > 0:
                        sim = co_ij / (norm_i * norm_j)

                        if sim > self.sim_threshold:
                            self.item_similarity[i][j] = sim
                            self.item_similarity[j][i] = sim

        # 每个物品只保留Top-K相似物品
        for item in self.item_similarity:
            sims = self.item_similarity[item]
            top_k = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:self.k]
            self.item_similarity[item] = dict(top_k)

        return self

    def predict(self, user_id, item_id):
        """
        预测用户对物品的兴趣分数
        """
        if user_id not in self.user_items:
            return 0.0

        if item_id not in self.item_similarity:
            return 0.0

        user_history = {item: rating for item, rating in self.user_items[user_id]}
        similar_items = self.item_similarity[item_id]

        score = 0.0
        total_sim = 0.0

        for hist_item, rating in user_history.items():
            if hist_item in similar_items:
                sim = similar_items[hist_item]
                score += sim * rating
                total_sim += sim

        if total_sim > 0:
            return score / total_sim
        return 0.0

    def recommend(self, user_id, n=10, exclude_rated=True):
        """
        为用户推荐Top-N物品
        """
        if user_id not in self.user_items:
            # 冷启动：返回热门物品
            return self._get_popular_items(n)

        user_history = {item for item, _ in self.user_items[user_id]}

        # 召回候选物品
        candidates = defaultdict(float)

        for hist_item, rating in self.user_items[user_id]:
            if hist_item in self.item_similarity:
                for sim_item, sim in self.item_similarity[hist_item].items():
                    if not exclude_rated or sim_item not in user_history:
                        candidates[sim_item] += sim * rating

        # 排序
        sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def _get_popular_items(self, n):
        """获取热门物品"""
        item_counts = {item: len(users) for item, users in self.item_users.items()}
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [(item, 0) for item, _ in sorted_items[:n]]

    def get_similar_items(self, item_id, n=10):
        """获取最相似的N个物品"""
        if item_id not in self.item_similarity:
            return []
        sims = self.item_similarity[item_id]
        return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:n]

    def explain(self, user_id, item_id):
        """解释推荐原因"""
        user_history = {item: rating for item, rating in self.user_items[user_id]}
        similar_items = self.item_similarity.get(item_id, {})

        reasons = []
        for hist_item, rating in user_history.items():
            if hist_item in similar_items:
                sim = similar_items[hist_item]
                reasons.append({
                    'history_item': hist_item,
                    'similarity': sim,
                    'rating': rating
                })

        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons


# ==================== 测试 ====================
# 创建示例数据
interactions = pd.DataFrame([
    ('user1', 'item1', 5),
    ('user1', 'item2', 4),
    ('user1', 'item3', 3),
    ('user2', 'item1', 4),
    ('user2', 'item2', 5),
    ('user2', 'item4', 2),
    ('user3', 'item2', 3),
    ('user3', 'item3', 4),
    ('user3', 'item4', 5),
    ('user4', 'item1', 5),
    ('user4', 'item3', 4),
    ('user4', 'item5', 3),
], columns=['user_id', 'item_id', 'rating'])

print("交互数据：")
print(interactions)

# 训练模型
model = ItemCF(k=5, sim_threshold=0.1)
model.fit(interactions)

# 查看物品相似度
print("\n物品相似度（部分）：")
for item in ['item1', 'item2', 'item3']:
    similar = model.get_similar_items(item, n=3)
    print(f"{item} 的相似物品：")
    for sim_item, sim in similar:
        print(f"  {sim_item}: {sim:.3f}")

# 推荐
print("\n为 user1 推荐：")
recs = model.recommend('user1', n=5)
for item, score in recs:
    print(f"  {item}: 预测分数 {score:.2f}")

# 解释推荐
print("\n解释为什么向 user1 推荐 item4：")
reasons = model.explain('user1', 'item4')
for r in reasons:
    print(f"  因为用户交互过 {r['history_item']} (评分={r['rating']})，")
    print(f"  而 {r['history_item']} 与 item4 相似度为 {r['similarity']:.3f}")
```

### 7.2 大规模数据处理

```python
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

class ItemCFLargeScale:
    """
    大规模ItemCF实现
    支持增量更新
    """

    def __init__(self, k=100):
        self.k = k
        self.item_similarity = {}
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        self.item_counts = defaultdict(int)

    def fit(self, interactions_df, batch_size=100000):
        """
        批量训练
        """
        # 统计共现
        item_pairs = defaultdict(float)

        # 按用户分组
        for user_id, group in interactions_df.groupby('user_id'):
            items = group['item_id'].tolist()
            n_items = len(items)

            if n_items < 2:
                continue

            # 用户权重（活跃度惩罚）
            user_weight = 1.0 / np.log(1 + n_items)

            # 更新统计
            for item in items:
                self.user_items[user_id].add(item)
                self.item_users[item].add(user)
                self.item_counts[item] += 1

            # 计算共现
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    item_i, item_j = items[i], items[j]
                    item_pairs[(item_i, item_j)] += user_weight
                    item_pairs[(item_j, item_i)] += user_weight

        # 计算相似度
        print("计算物品相似度...")

        # 收集每个物品的共现信息
        item_cooccurs = defaultdict(dict)
        for (i, j), count in item_pairs.items():
            item_cooccurs[i][j] = count

        # 计算相似度并保留Top-K
        for item_i, cooccur in item_cooccurs.items():
            n_i = self.item_counts[item_i]
            sims = {}

            for item_j, co_count in cooccur.items():
                n_j = self.item_counts[item_j]

                # Jaccard相似度（更快）
                sim = co_count / (n_i + n_j - co_count)

                if sim > 0.001:  # 阈值过滤
                    sims[item_j] = sim

            # Top-K
            top_k = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:self.k]
            self.item_similarity[item_i] = dict(top_k)

        print(f"训练完成，共 {len(self.item_similarity)} 个物品有相似物品")
        return self

    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'item_similarity': self.item_similarity,
                'item_counts': self.item_counts
            }, f)

    def load(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.item_similarity = data['item_similarity']
            self.item_counts = data['item_counts']
        return self

    def recommend_fast(self, user_history_items, n=10):
        """
        快速推荐（用于在线服务）
        """
        candidates = defaultdict(float)

        for item in user_history_items:
            if item in self.item_similarity:
                for sim_item, sim in self.item_similarity[item].items():
                    candidates[sim_item] += sim

        # 过滤已交互的
        for item in user_history_items:
            candidates.pop(item, None)

        # 排序
        sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
```

---

## 8. 手工代码实现

### 8.1 简洁版实现

```python
from collections import defaultdict
import math

def train_itemcf(interactions):
    """
    训练ItemCF模型

    参数:
    interactions: [(user_id, item_id), ...]

    返回:
    item_similarity: {item: {other_item: similarity}}
    """
    # 1. 构建用户-物品和物品-用户字典
    user_items = defaultdict(set)
    item_users = defaultdict(set)

    for user, item in interactions:
        user_items[user].add(item)
        item_users[item].add(user)

    # 2. 计算物品共现
    # co_occur[i][j] = 同时交互过物品i和j的用户加权贡献
    co_occur = defaultdict(lambda: defaultdict(float))

    for user, items in user_items.items():
        # 用户权重（活跃度惩罚）
        weight = 1.0 / math.log(1 + len(items))

        items_list = list(items)
        for i in range(len(items_list)):
            for j in range(i+1, len(items_list)):
                item_i, item_j = items_list[i], items_list[j]
                co_occur[item_i][item_j] += weight
                co_occur[item_j][item_i] += weight

    # 3. 计算相似度
    item_similarity = defaultdict(dict)

    all_items = list(item_users.keys())

    for i in all_items:
        for j in all_items:
            if i >= j:
                continue

            if j not in co_occur[i]:
                continue

            # 计算分母（带权重的物品热度）
            def item_norm(item):
                return math.sqrt(sum(
                    1.0 / math.log(1 + len(user_items[u]))
                    for u in item_users[item]
                ))

            norm_i = item_norm(i)
            norm_j = item_norm(j)

            if norm_i > 0 and norm_j > 0:
                sim = co_occur[i][j] / (norm_i * norm_j)
                if sim > 0.01:  # 阈值
                    item_similarity[i][j] = sim
                    item_similarity[j][i] = sim

    return item_similarity, user_items


def recommend(item_similarity, user_items, user_id, n=10):
    """
    为用户推荐
    """
    if user_id not in user_items:
        return []

    history = user_items[user_id]
    scores = defaultdict(float)

    for item in history:
        if item in item_similarity:
            for sim_item, sim in item_similarity[item].items():
                if sim_item not in history:
                    scores[sim_item] += sim

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]


# ==================== 测试 ====================
interactions = [
    ('u1', 'a'), ('u1', 'b'), ('u1', 'c'),
    ('u2', 'a'), ('u2', 'b'), ('u2', 'd'),
    ('u3', 'b'), ('u3', 'c'), ('u3', 'd'),
    ('u4', 'a'), ('u4', 'c'), ('u4', 'e'),
]

sim, user_items = train_itemcf(interactions)

print("物品a的相似物品：")
for item, s in sorted(sim['a'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {item}: {s:.3f}")

print("\n为u1推荐：")
recs = recommend(sim, user_items, 'u1', 5)
for item, score in recs:
    print(f"  {item}: {score:.3f}")
```

---

## 9. 可视化与结果理解

### 9.1 物品相似度可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_item_similarity(item_similarity, items=None):
    """绘制物品相似度热力图"""
    if items is None:
        items = list(item_similarity.keys())[:10]  # 取前10个

    # 构建矩阵
    n = len(items)
    matrix = np.zeros((n, n))

    for i, item_i in enumerate(items):
        matrix[i][i] = 1.0
        for j, item_j in enumerate(items):
            if item_j in item_similarity.get(item_i, {}):
                matrix[i][j] = item_similarity[item_i][item_j]

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        xticklabels=items,
        yticklabels=items,
        cmap='RdYlBu_r'
    )
    plt.title('物品相似度矩阵')
    plt.tight_layout()
    plt.show()

# 使用示例
plot_item_similarity(model.item_similarity)
```

### 9.2 推荐路径可视化

```python
def visualize_recommendation_path(model, user_id, recommended_item):
    """可视化推荐路径"""
    reasons = model.explain(user_id, recommended_item)

    if not reasons:
        print("无法解释此推荐")
        return

    print(f"\n推荐 {recommended_item} 给 {user_id} 的原因：")
    print("=" * 50)

    for i, r in enumerate(reasons[:5], 1):
        print(f"{i}. 用户曾交互过 {r['history_item']} (评分={r['rating']})")
        print(f"   └─ {r['history_item']} ↔ {recommended_item} 相似度 = {r['similarity']:.3f}")
        print()

# 使用示例
visualize_recommendation_path(model, 'user1', 'item4')
```

---

## 10. 模型评估

### 10.1 离线评估

```python
def evaluate_itemcf(model, test_df, k=10):
    """
    评估ItemCF模型

    指标：
    - Precision@K
    - Recall@K
    - Coverage
    """
    # 构建测试集真实标签
    test_user_items = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

    precisions = []
    recalls = []
    recommended_items = set()
    all_items = set(model.item_similarity.keys())

    for user_id, actual_items in test_user_items.items():
        # 推荐
        recs = model.recommend(user_id, n=k)
        rec_items = set(item for item, _ in recs)

        # 统计推荐物品
        recommended_items.update(rec_items)

        # 命中数
        hits = len(rec_items & actual_items)

        # Precision
        precision = hits / k if k > 0 else 0
        precisions.append(precision)

        # Recall
        recall = hits / len(actual_items) if actual_items else 0
        recalls.append(recall)

    # 覆盖率
    coverage = len(recommended_items) / len(all_items) if all_items else 0

    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        'Coverage': coverage
    }

# 使用示例
metrics = evaluate_itemcf(model, test_df, k=10)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### 10.2 在线A/B测试

```
A/B测试设计：

实验组：使用ItemCF推荐
对照组：使用热门推荐

关键指标：
- CTR（点击率）
- 转化率
- 人均浏览物品数
- 停留时长

分析维度：
- 不同用户群体
- 不同物品类目
- 不同时间段
```

---

## 11. 常见问题与优化

### 11.1 冷启动问题

```python
# 新物品冷启动解决方案

def recommend_with_cold_start(model, user_id, new_items, n=10):
    """
    处理新物品冷启动

    策略：
    1. 新物品使用基于内容的相似度
    2. 与ItemCF结果融合
    """
    # ItemCF推荐
    cf_recs = model.recommend(user_id, n=n//2)

    # 新物品推荐（基于内容相似度）
    # 假设已有 content_similarity
    content_recs = []
    # ... 根据内容特征计算相似度

    # 融合
    final_recs = cf_recs + content_recs
    final_recs.sort(key=lambda x: x[1], reverse=True)

    return final_recs[:n]
```

### 11.2 多样性问题

```python
def recommend_with_diversity(model, user_id, n=10, lambda_div=0.3):
    """
    增加推荐多样性

    使用MMR算法
    """
    # 获取用户历史
    history = [item for item, _ in model.user_items[user_id]]

    # 候选物品
    candidates = defaultdict(float)
    for item in history:
        if item in model.item_similarity:
            for sim_item, sim in model.item_similarity[item].items():
                if sim_item not in history:
                    candidates[sim_item] += sim

    # MMR选择
    selected = []
    remaining = list(candidates.keys())

    while len(selected) < n and remaining:
        best_score = -float('inf')
        best_item = None

        for item in remaining:
            # 相关性
            relevance = candidates[item]

            # 与已选物品的最大相似度
            max_sim = 0
            for sel in selected:
                if sel in model.item_similarity.get(item, {}):
                    max_sim = max(max_sim, model.item_similarity[item][sel])

            # MMR分数
            mmr_score = (1 - lambda_div) * relevance - lambda_div * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        if best_item:
            selected.append(best_item)
            remaining.remove(best_item)

    return [(item, candidates[item]) for item in selected]
```

### 11.3 实时更新

```python
class ItemCFOnline:
    """
    支持实时更新的ItemCF
    """

    def __init__(self, k=100):
        self.k = k
        self.item_similarity = {}
        self.item_counts = defaultdict(int)
        self.item_pair_counts = defaultdict(lambda: defaultdict(int))

    def add_interaction(self, user_id, item_id, user_history):
        """
        增量更新：用户产生了新的交互
        """
        # 更新物品计数
        self.item_counts[item_id] += 1

        # 更新共现
        for hist_item in user_history:
            if hist_item != item_id:
                self.item_pair_counts[item_id][hist_item] += 1
                self.item_pair_counts[hist_item][item_id] += 1

                # 重新计算相似度
                self._update_similarity(item_id, hist_item)

    def _update_similarity(self, item_i, item_j):
        """更新两个物品的相似度"""
        co = self.item_pair_counts[item_i][item_j]
        n_i = self.item_counts[item_i]
        n_j = self.item_counts[item_j]

        # Jaccard相似度
        sim = co / (n_i + n_j - co) if (n_i + n_j - co) > 0 else 0

        # 更新
        if sim > 0.001:
            if item_i not in self.item_similarity:
                self.item_similarity[item_i] = {}
            if item_j not in self.item_similarity:
                self.item_similarity[item_j] = {}

            self.item_similarity[item_i][item_j] = sim
            self.item_similarity[item_j][item_i] = sim
```

---

## 12. 学习总结

### 12.1 核心要点

1. **核心思想**：推荐与用户历史物品相似的物品
2. **相似度计算**：基于共现，考虑活跃度惩罚
3. **适用场景**：物品相对稳定，如电商、视频
4. **优势**：可解释性强、计算效率高、实时性好

### 12.2 记忆口诀

```
ItemCF找相似物品，推荐和历史像的
共现次数算相似，活跃用户要惩罚
可解释性特别强，电商场景最适用
新物品有冷启动，内容特征来帮忙
```

### 12.3 UserCF vs ItemCF 选择

```
选择依据：

1. 用户数 vs 物品数
   - 用户少、物品多 → UserCF
   - 物品少、用户多 → ItemCF

2. 稳定性
   - 用户兴趣稳定 → ItemCF
   - 物品属性稳定 → ItemCF

3. 实时性
   - 需要快速响应 → ItemCF（可预计算）

4. 可解释性
   - 需要解释推荐原因 → ItemCF

5. 多样性
   - 需要惊喜 → UserCF
```

---

## 13. 练习题与思考题

### 13.1 基础题

1. ItemCF的核心思想是什么？
2. 为什么计算物品相似度时要惩罚活跃用户？
3. ItemCF和UserCF的主要区别是什么？

### 13.2 进阶题

4. 如何解决新物品的冷启动问题？
5. 如何提高ItemCF推荐的多样性？
6. 为什么电商场景更适合ItemCF？

### 13.3 参考答案

```
2. 惩罚活跃用户的原因：
   - 活跃用户点击很多东西，但未必真的都感兴趣
   - 他们的行为会产生很多虚假的物品共现
   - 降低他们的权重，使相似度更准确

5. 提高多样性的方法：
   - MMR算法
   - 类目打散
   - 与其他召回源混合
   - 降低相似度阈值

6. 电商适合ItemCF的原因：
   - 物品数通常少于用户数，计算效率高
   - 物品属性相对稳定，相似度可预计算
   - 用户倾向买相似商品
   - 推荐可解释性强（"你买过X"）
```

---

## 14. 学习路径建议

```
学完ItemCF后，建议学习：

1. 矩阵分解（MF）
   - 解决稀疏性问题
   ↓
2. FM
   - 自动特征交叉
   ↓
3. 协同过滤+深度学习
   - NCF、DeepFM
```

**下一章**：[44_矩阵分解基础](./44_矩阵分解基础.md) - 理解隐因子模型
