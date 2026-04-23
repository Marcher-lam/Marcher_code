# UserCF - 基于用户的协同过滤

> 推荐系统最经典算法之一

---

## 1. 算法基础认知

### 1.1 什么是UserCF

**UserCF（User-based Collaborative Filtering，基于用户的协同过滤）的核心思想是：**

> "物以类聚，人以群分" —— 找到和你兴趣相似的用户，推荐他们喜欢但你还没看过的物品。

### 1.2 直观理解

```
┌────────────────────────────────────────────────────────────┐
│                    UserCF 直觉理解                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  用户A：⭐️ 电影1  ⭐️ 电影2  ⭐️ 电影3                     │
│  用户B：⭐️ 电影1  ⭐️ 电影2  ❓ 电影4                     │
│  用户C：⭐️ 电影1           ⭐️ 电影5                      │
│                                                            │
│  分析：                                                    │
│  - 用户A和B都看过电影1、2，兴趣相似                       │
│  - 用户A还看过电影3                                        │
│  - 推测用户B可能也喜欢电影3                                │
│                                                            │
│  结论：向用户B推荐电影3                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.3 在推荐系统中的位置

```
推荐系统架构：
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   召回层 ────────────────────────────────────────────────  │
│   ├── 协同过滤召回 ← UserCF在这里                          │
│   │   ├── UserCF                                          │
│   │   └── ItemCF                                          │
│   ├── 向量召回                                            │
│   └── 热门召回                                            │
│                                                            │
│   排序层 ────────────────────────────────────────────────  │
│   └── 精排模型                                            │
│                                                            │
└────────────────────────────────────────────────────────────┘

UserCF 主要用于：召回阶段
```

---

## 2. 核心原理

### 2.1 算法流程

```
UserCF 算法步骤：

1. 构建用户-物品评分矩阵
2. 计算用户之间的相似度
3. 找到目标用户的K个最相似用户
4. 根据相似用户的行为，预测目标用户对物品的评分
5. 推荐预测评分最高的N个物品
```

### 2.2 用户-物品矩阵

```
用户-物品评分矩阵示例：

           电影1  电影2  电影3  电影4  电影5
用户A        5      4      ?      3      ?
用户B        4      5      4      ?      2
用户C        ?      3      5      4      ?
用户D        5      ?      4      3      5

说明：
- 5/4/3/2/1 表示评分
- ? 表示未看过/未评分
- 目标：预测 ? 的值
```

### 2.3 相似度计算

**核心问题：如何衡量两个用户之间的相似度？**

最常用的方法是**余弦相似度**：

```
余弦相似度公式：

sim(u, v) = cos(u, v) = (u · v) / (||u|| × ||v||)

其中：
- u, v：两个用户的评分向量
- u · v：向量点积
- ||u||：向量的模（范数）

展开：
sim(u, v) = Σ(rᵤᵢ × rᵥᵢ) / (√Σrᵤᵢ² × √Σrᵥᵢ²)

只考虑两个用户都评过分的物品 i
```

---

## 3. 数学公式与推导

### 3.1 相似度计算详解

```python
# 用户A的评分向量
user_a = [5, 4, 0, 3, 0]  # 0表示未评分

# 用户B的评分向量
user_b = [4, 5, 4, 0, 2]

# 计算余弦相似度
# 只考虑两人都评过分的物品：电影1、电影2、电影5

# 共同评分的物品
common_items = [0, 1, 4]  # 索引

# 用户A在共同物品上的评分
a_common = [5, 4, 0]  # 注意：用户A没评电影5，这里用0

# 实际上应该只用都评过的：电影1、电影2
a_ratings = [5, 4]
b_ratings = [4, 5]

# 点积
dot_product = 5*4 + 4*5 = 40

# 向量模
norm_a = sqrt(5² + 4²) = sqrt(41)
norm_b = sqrt(4² + 5²) = sqrt(41)

# 余弦相似度
similarity = 40 / (sqrt(41) * sqrt(41)) = 40/41 ≈ 0.976
```

### 3.2 评分预测公式

找到最相似的K个用户后，预测目标用户对物品i的评分：

```
方法1：加权平均

p(u, i) = (Σᵥ∈N(u) sim(u, v) × rᵥᵢ) / Σᵥ∈N(u) |sim(u, v)|

其中：
- N(u)：用户u最相似的K个用户
- sim(u, v)：用户u和v的相似度
- rᵥᵢ：用户v对物品i的评分

方法2：考虑用户评分偏差（更精确）

p(u, i) = r̄ᵤ + (Σᵥ sim(u, v) × (rᵥᵢ - r̄ᵥ)) / Σᵥ |sim(u, v)|

其中：
- r̄ᵤ：用户u的平均评分
- r̄ᵥ：用户v的平均评分
```

### 3.3 为什么考虑偏差

```
场景：用户A评分普遍高，用户B评分普遍低

用户A：平均评分 4.5
用户B：平均评分 2.5

如果用户B给某电影3分（对他来说算高分）
直接计算会低估这个信号

改进方法：使用相对评分（减去各自均值）
用户B相对评分：3 - 2.5 = 0.5（正向偏好）
```

---

## 4. 算法实现步骤

### 4.1 完整流程

```
┌────────────────────────────────────────────────────────────┐
│                    UserCF 完整流程                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  输入：用户-物品交互数据                                   │
│                                                            │
│  Step 1: 构建用户-物品矩阵                                 │
│          ┌──────────────────────────┐                      │
│          │  user_item_matrix        │                      │
│          │  rows = users            │                      │
│          │  cols = items            │                      │
│          └──────────────────────────┘                      │
│                         ↓                                  │
│  Step 2: 计算用户相似度矩阵                               │
│          ┌──────────────────────────┐                      │
│          │  user_similarity_matrix  │                      │
│          │  sim[u][v]               │                      │
│          └──────────────────────────┘                      │
│                         ↓                                  │
│  Step 3: 为目标用户找K个最相似用户                        │
│          ┌──────────────────────────┐                      │
│          │  neighbors[u] = top K    │                      │
│          └──────────────────────────┘                      │
│                         ↓                                  │
│  Step 4: 预测用户对未交互物品的评分                       │
│          ┌──────────────────────────┐                      │
│          │  predict(u, i)           │                      │
│          └──────────────────────────┘                      │
│                         ↓                                  │
│  Step 5: 推荐Top-N物品                                    │
│          ┌──────────────────────────┐                      │
│          │  recommendations[u]      │                      │
│          └──────────────────────────┘                      │
│                                                            │
│  输出：推荐列表                                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 5. 应用场景

### 5.1 适用场景

| 场景 | 适用性 | 说明 |
|-----|-------|-----|
| **新闻推荐** | ✅ 高 | 用户兴趣相似，新闻更新快 |
| **社交媒体** | ✅ 高 | 关注相似用户的内容 |
| **音乐推荐** | ✅ 中 | 基于口味相似推荐 |
| **电商** | ⚠️ 中 | 用户数远大于商品数时效率低 |
| **视频** | ⚠️ 中 | 需要结合其他方法 |

### 5.2 在推荐系统中的作用

```
现代推荐系统中的UserCF：

┌────────────────────────────────────────────────────────────┐
│                    多路召回架构                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   用户请求                                                 │
│      │                                                     │
│      ├──→ [UserCF召回] ─────┐                             │
│      │                      │                              │
│      ├──→ [ItemCF召回] ─────┤                             │
│      │                      │                              │
│      ├──→ [向量召回] ───────┼──→ 合并 ──→ 候选集          │
│      │                      │                              │
│      ├──→ [热门召回] ───────┤                             │
│      │                      │                              │
│      └──→ [规则召回] ───────┘                             │
│                                                            │
│   UserCF 作为召回通道之一                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|-----|------|
| **原理简单** | 容易理解和实现 |
| **可解释性强** | "和你相似的用户也喜欢..." |
| **发现惊喜** | 能推荐意想不到的内容 |
| **不需要物品特征** | 只依赖用户行为 |

### 6.2 缺点

| 缺点 | 说明 |
|-----|------|
| **冷启动** | 新用户无行为，无法计算相似度 |
| **稀疏性** | 用户-物品矩阵稀疏，相似度计算不准 |
| **可扩展性差** | 用户数大时，计算复杂度高 |
| **流行度偏差** | 容易推荐热门物品 |

### 6.3 复杂度分析

```
时间复杂度：
- 相似度计算：O(m² × n)，m=用户数，n=物品数
- 找邻居：O(m × k)
- 预测：O(k × n)

空间复杂度：
- 用户相似度矩阵：O(m²)

当用户数 m 很大时（百万级），计算和存储都是问题！
```

---

## 7. 调库实现

### 7.1 基础实现

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class UserCF:
    """
    基于用户的协同过滤
    """

    def __init__(self, k=10):
        """
        参数:
        k: 最近邻用户数量
        """
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_mean = None

    def fit(self, ratings_df):
        """
        训练模型

        参数:
        ratings_df: DataFrame, 列为 ['user_id', 'item_id', 'rating']
        """
        # 构建用户-物品矩阵
        self.ratings_df = ratings_df
        self.user_item_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # 计算用户平均评分
        self.user_mean = self.user_item_matrix.mean(axis=1)

        # 计算用户相似度矩阵
        # 方法1：直接用cosine_similarity（未考虑均值）
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        return self

    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分

        参数:
        user_id: 用户ID
        item_id: 物品ID

        返回:
        预测评分
        """
        # 检查物品是否存在
        if item_id not in self.user_item_matrix.columns:
            return self.user_mean.get(user_id, 0)

        # 找到对该物品评过分的用户
        rated_users = self.user_item_matrix[
            self.user_item_matrix[item_id] > 0
        ].index

        if len(rated_users) == 0:
            return self.user_mean.get(user_id, 0)

        # 获取与目标用户最相似的K个用户（在评过该物品的用户中）
        sim_scores = self.user_similarity.loc[user_id, rated_users]
        top_k_users = sim_scores.nlargest(self.k)

        # 加权预测
        numerator = 0
        denominator = 0

        for neighbor, sim in top_k_users.items():
            if sim > 0:  # 只考虑正相似度
                neighbor_rating = self.user_item_matrix.loc[neighbor, item_id]
                neighbor_mean = self.user_mean[neighbor]

                # 使用均值中心化的方法
                numerator += sim * (neighbor_rating - neighbor_mean)
                denominator += abs(sim)

        if denominator == 0:
            return self.user_mean.get(user_id, 0)

        prediction = self.user_mean[user_id] + numerator / denominator
        return prediction

    def recommend(self, user_id, n=10):
        """
        为用户推荐Top-N物品

        参数:
        user_id: 用户ID
        n: 推荐数量

        返回:
        推荐物品列表 [(item_id, predicted_rating), ...]
        """
        # 获取用户已交互的物品
        user_rated = self.user_item_matrix.loc[user_id]
        rated_items = set(user_rated[user_rated > 0].index)

        # 预测未交互物品的评分
        predictions = []
        for item_id in self.user_item_matrix.columns:
            if item_id not in rated_items:
                pred = self.predict(user_id, item_id)
                predictions.append((item_id, pred))

        # 排序返回Top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def get_similar_users(self, user_id, n=10):
        """
        获取最相似的N个用户

        参数:
        user_id: 用户ID
        n: 返回数量

        返回:
        相似用户列表 [(user_id, similarity), ...]
        """
        sim_scores = self.user_similarity.loc[user_id]
        sim_scores = sim_scores.drop(user_id)  # 排除自己
        return list(sim_scores.nlargest(n).items())


# ==================== 测试 ====================
# 创建示例数据
data = {
    'user_id': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'D'],
    'item_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4, 5],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 5, 4, 3, 4]
}
ratings_df = pd.DataFrame(data)

print("评分数据：")
print(ratings_df)
print()

# 训练模型
model = UserCF(k=2)
model.fit(ratings_df)

print("\n用户-物品矩阵：")
print(model.user_item_matrix)
print()

# 查看用户相似度
print("\n用户相似度矩阵：")
print(model.user_similarity.round(3))
print()

# 预测评分
print("预测用户A对物品4的评分：", model.predict('A', 4))

# 推荐物品
print("\n为用户A推荐物品：")
recs = model.recommend('A', n=3)
for item_id, score in recs:
    print(f"  物品{item_id}: 预测评分 {score:.2f}")

# 查看相似用户
print("\n与用户A最相似的用户：")
similar = model.get_similar_users('A', n=3)
for user, sim in similar:
    print(f"  用户{user}: 相似度 {sim:.3f}")
```

### 7.2 MovieLens数据集实战

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 加载MovieLens数据 ====================
# 如果没有数据，可以使用以下方式模拟
def create_sample_movielens():
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 2000

    users = np.random.randint(1, n_users + 1, n_ratings)
    items = np.random.randint(1, n_items + 1, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)

    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings
    })
    # 去重
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    return df

ratings_df = create_sample_movielens()
print(f"数据规模: {len(ratings_df)} 条评分")
print(f"用户数: {ratings_df['user_id'].nunique()}")
print(f"物品数: {ratings_df['item_id'].nunique()}")

# ==================== 划分训练集和测试集 ====================
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# ==================== 训练UserCF ====================
model = UserCF(k=10)
model.fit(train_df)

# ==================== 评估 ====================
def evaluate(model, test_df):
    """评估模型"""
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual = row['rating']

        try:
            pred = model.predict(user_id, item_id)
            predictions.append(pred)
            actuals.append(actual)
        except:
            continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    return {'RMSE': rmse, 'MAE': mae}

metrics = evaluate(model, test_df)
print(f"\n评估结果：")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
```

---

## 8. 手工代码实现

### 8.1 完整手工实现

```python
import numpy as np
from collections import defaultdict

class UserCFManual:
    """
    手工实现的UserCF
    不依赖sklearn，从头实现
    """

    def __init__(self, k=10):
        self.k = k
        self.user_items = defaultdict(dict)  # user -> {item: rating}
        self.item_users = defaultdict(dict)  # item -> {user: rating}
        self.user_mean = {}
        self.user_similarity = defaultdict(dict)

    def fit(self, ratings):
        """
        训练

        参数:
        ratings: list of (user_id, item_id, rating)
        """
        # 构建用户-物品和物品-用户字典
        for user, item, rating in ratings:
            self.user_items[user][item] = rating
            self.item_users[item][user] = rating

        # 计算用户平均评分
        for user in self.user_items:
            ratings = list(self.user_items[user].values())
            self.user_mean[user] = np.mean(ratings)

        # 计算用户相似度
        users = list(self.user_items.keys())
        for i, u1 in enumerate(users):
            for u2 in users[i+1:]:
                sim = self._cosine_similarity(u1, u2)
                if sim > 0:
                    self.user_similarity[u1][u2] = sim
                    self.user_similarity[u2][u1] = sim

    def _cosine_similarity(self, u1, u2):
        """计算两个用户的余弦相似度"""
        # 找共同评过分的物品
        common_items = set(self.user_items[u1].keys()) & set(self.user_items[u2].keys())

        if not common_items:
            return 0.0

        # 使用均值中心化
        numerator = 0
        norm1 = 0
        norm2 = 0

        for item in common_items:
            r1 = self.user_items[u1][item] - self.user_mean[u1]
            r2 = self.user_items[u2][item] - self.user_mean[u2]

            numerator += r1 * r2
            norm1 += r1 ** 2
            norm2 += r2 ** 2

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return numerator / (np.sqrt(norm1) * np.sqrt(norm2))

    def predict(self, user, item):
        """预测用户对物品的评分"""
        # 如果用户不存在，返回全局平均
        if user not in self.user_mean:
            return 3.0  # 默认评分

        # 找到对物品评过分的用户中，与目标用户最相似的K个
        if item not in self.item_users:
            return self.user_mean[user]

        candidate_users = list(self.item_users[item].keys())

        # 获取相似度
        sim_users = []
        for candidate in candidate_users:
            if candidate != user and candidate in self.user_similarity[user]:
                sim_users.append((candidate, self.user_similarity[user][candidate]))

        # 取Top-K
        sim_users.sort(key=lambda x: x[1], reverse=True)
        sim_users = sim_users[:self.k]

        if not sim_users:
            return self.user_mean[user]

        # 加权预测
        numerator = 0
        denominator = 0

        for neighbor, sim in sim_users:
            neighbor_rating = self.user_items[neighbor][item]
            neighbor_mean = self.user_mean[neighbor]

            numerator += sim * (neighbor_rating - neighbor_mean)
            denominator += abs(sim)

        if denominator == 0:
            return self.user_mean[user]

        return self.user_mean[user] + numerator / denominator

    def recommend(self, user, n=10):
        """推荐物品"""
        if user not in self.user_items:
            return []

        # 用户已交互的物品
        rated_items = set(self.user_items[user].keys())

        # 预测所有未交互物品
        all_items = set(self.item_users.keys())
        unrated_items = all_items - rated_items

        predictions = []
        for item in unrated_items:
            pred = self.predict(user, item)
            predictions.append((item, pred))

        # 排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


# ==================== 测试手工实现 ====================
ratings = [
    ('A', 1, 5), ('A', 2, 4), ('A', 3, 3),
    ('B', 1, 4), ('B', 2, 5), ('B', 4, 2),
    ('C', 2, 3), ('C', 3, 4), ('C', 4, 5),
    ('D', 1, 5), ('D', 3, 4), ('D', 4, 3), ('D', 5, 4)
]

model = UserCFManual(k=2)
model.fit(ratings)

print("预测用户A对物品4的评分：", model.predict('A', 4))
print("\n为用户A推荐：")
for item, score in model.recommend('A', 3):
    print(f"  物品{item}: {score:.2f}")
```

---

## 9. 可视化与结果理解

### 9.1 用户相似度热力图

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_user_similarity(user_similarity_df):
    """绘制用户相似度热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        user_similarity_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0
    )
    plt.title('用户相似度矩阵')
    plt.xlabel('用户')
    plt.ylabel('用户')
    plt.tight_layout()
    plt.show()

# 使用前面训练的模型
plot_user_similarity(model.user_similarity.round(3))
```

### 9.2 推荐结果解读

```python
def explain_recommendation(model, user_id, item_id):
    """
    解释为什么推荐这个物品
    """
    # 找相似用户
    similar_users = model.get_similar_users(user_id, n=5)

    # 找对该物品评过分的相似用户
    explanations = []

    for neighbor, sim in similar_users:
        if item_id in model.user_item_matrix.columns:
            rating = model.user_item_matrix.loc[neighbor, item_id]
            if rating > 0:
                explanations.append({
                    'user': neighbor,
                    'similarity': sim,
                    'rating': rating
                })

    print(f"推荐物品 {item_id} 给用户 {user_id} 的原因：")
    print("-" * 50)

    if explanations:
        print("相似用户对该物品的评价：")
        for exp in explanations:
            print(f"  用户{exp['user']}: 相似度 {exp['similarity']:.3f}, 评分 {exp['rating']}")
    else:
        print("  无相似用户评价过该物品")

    pred = model.predict(user_id, item_id)
    print(f"\n预测评分: {pred:.2f}")

# 使用示例
explain_recommendation(model, 'A', 4)
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_usercf(model, test_df):
    """
    评估UserCF模型

    指标：
    - RMSE: 均方根误差
    - MAE: 平均绝对误差
    - Coverage: 覆盖率
    """
    predictions = []
    actuals = []

    # 评分预测评估
    for _, row in test_df.iterrows():
        try:
            pred = model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])
        except:
            continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # MAE
    mae = mean_absolute_error(actuals, predictions)

    # 覆盖率
    all_items = set(model.user_item_matrix.columns)
    recommended_items = set()

    for user_id in model.user_item_matrix.index[:100]:  # 采样100个用户
        recs = model.recommend(user_id, n=10)
        for item_id, _ in recs:
            recommended_items.add(item_id)

    coverage = len(recommended_items) / len(all_items)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'Coverage': coverage
    }
```

### 10.2 Top-N推荐评估

```python
def precision_recall_at_k(model, test_df, k=10):
    """
    计算Top-K推荐的精确率和召回率
    """
    # 构建测试集的真实偏好
    test_user_items = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

    precisions = []
    recalls = []

    for user_id, actual_items in test_user_items.items():
        try:
            # 获取推荐列表
            recommendations = model.recommend(user_id, n=k)
            recommended_items = set([item for item, _ in recommendations])

            # 计算交集
            hits = len(recommended_items & actual_items)

            # 精确率
            precision = hits / k if k > 0 else 0
            precisions.append(precision)

            # 召回率
            recall = hits / len(actual_items) if actual_items else 0
            recalls.append(recall)

        except:
            continue

    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls)
    }
```

---

## 11. 常见问题与易错点

### 11.1 数据稀疏问题

```python
# 问题：用户-物品矩阵非常稀疏
# 解决方案：

# 1. 使用皮尔逊相关系数代替余弦相似度（考虑评分偏差）
def pearson_similarity(u, v):
    """皮尔逊相关系数"""
    common = u.keys() & v.keys()
    if len(common) < 2:
        return 0

    u_mean = np.mean([u[i] for i in common])
    v_mean = np.mean([v[i] for i in common])

    numerator = sum((u[i]-u_mean)*(v[i]-v_mean) for i in common)
    denominator = np.sqrt(sum((u[i]-u_mean)**2 for i in common)) * \
                  np.sqrt(sum((v[i]-v_mean)**2 for i in common))

    return numerator / denominator if denominator > 0 else 0

# 2. 设置最小公共物品数阈值
def similarity_with_threshold(u, v, min_common=2):
    common = u.keys() & v.keys()
    if len(common) < min_common:
        return 0
    return cosine_similarity(u, v)
```

### 11.2 冷启动问题

```python
# 问题：新用户没有行为数据
# 解决方案：

# 1. 使用热门推荐作为fallback
def recommend_with_cold_start(model, user_id, n=10, top_items=None):
    try:
        recs = model.recommend(user_id, n)
        if recs:
            return recs
    except:
        pass

    # 冷启动：返回热门物品
    if top_items:
        return [(item, 0) for item in top_items[:n]]
    return []

# 2. 使用用户属性找相似用户
def find_similar_by_attributes(new_user_profile, all_users_profiles):
    """基于用户属性（年龄、性别等）找相似用户"""
    # 计算属性相似度
    # 返回相似用户
    pass
```

### 11.3 计算效率问题

```python
# 问题：用户数大时，相似度计算慢
# 解决方案：

# 1. 只计算部分相似度（稀疏存储）
# 2. 使用近似算法（LSH）
# 3. 分布式计算

# 使用稀疏矩阵优化
from scipy.sparse import csr_matrix

def build_sparse_matrix(ratings_df, n_users, n_items):
    """构建稀疏矩阵"""
    row = ratings_df['user_id'].astype('category').cat.codes
    col = ratings_df['item_id'].astype('category').cat.codes
    data = ratings_df['rating']

    return csr_matrix((data, (row, col)), shape=(n_users, n_items))
```

---

## 12. 学习总结

### 12.1 核心要点

1. **核心思想**：找相似用户，推荐他们喜欢的物品
2. **关键步骤**：计算用户相似度 → 找邻居 → 预测评分 → 排序推荐
3. **相似度计算**：余弦相似度、皮尔逊相关系数
4. **主要问题**：冷启动、稀疏性、可扩展性

### 12.2 记忆口诀

```
UserCF找人像，相似用户做参考
余弦相似算距离，K个邻居来帮忙
加权平均预测分，高分物品往上排
冷启动是大问题，热门推荐来兜底
```

### 12.3 与ItemCF对比

| 维度 | UserCF | ItemCF |
|-----|--------|--------|
| **核心** | 找相似用户 | 找相似物品 |
| **适用场景** | 用户少、用户兴趣稳定 | 物品少、物品属性稳定 |
| **推荐理由** | "和你相似的人喜欢" | "你喜欢的物品相似" |
| **更新频率** | 用户行为变化需更新 | 物品被更多人评过更准 |

---

## 13. 练习题与思考题

### 13.1 基础题

1. UserCF的核心思想是什么？
2. 如何计算两个用户之间的相似度？
3. UserCF适用于什么场景？

### 13.2 进阶题

4. UserCF如何处理冷启动问题？
5. 为什么UserCF在用户数远大于物品数时效率较低？
6. UserCF和ItemCF的区别是什么？各自适用于什么场景？

### 13.3 参考答案

```
1. 核心思想：找到与目标用户兴趣相似的其他用户，
   推荐这些相似用户喜欢但目标用户未交互的物品。

2. 相似度计算方法：
   - 余弦相似度：向量夹角的余弦值
   - 皮尔逊相关系数：考虑均值偏差的相关性
   - Jaccard相似度：交集/并集（隐式反馈）

3. 适用场景：
   - 用户兴趣相对稳定
   - 用户数量不是特别大
   - 新闻、社交媒体等

5. 效率问题：
   - 相似度矩阵大小 O(m²)，m为用户数
   - 用户数大时，存储和计算开销大
   - ItemCF的物品数通常远小于用户数，所以更高效
```

---

## 14. 学习路径建议

```
学完UserCF后，建议学习：

1. ItemCF
   - 基于物品的协同过滤
   ↓
2. 矩阵分解
   - 解决稀疏性问题
   ↓
3. FM
   - 特征交叉
```

**下一章**：[42_ItemCF](./42_ItemCF.md) - 基于物品的协同过滤
