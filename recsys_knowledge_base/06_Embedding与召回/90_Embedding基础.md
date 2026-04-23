# Embedding 基础 学习文档

## 1. 基础认知

### 1.1 什么是 Embedding？

Embedding（嵌入）是将离散的、高维的、稀疏的表示映射到连续的、低维的、稠密的向量空间的技术。

**简单理解：**
- 把一个 ID（如用户 ID "12345"）变成一个向量 [0.1, 0.3, -0.2, ...]
- 这个向量能捕捉对象的语义信息
- 相似的对象有相似的 Embedding

### 1.2 为什么需要 Embedding？

**问题：One-Hot 编码**

```python
# 假设有 100 万个用户
user_id = "user_12345"

# One-Hot 编码
one_hot = [0, 0, 0, ..., 1, ..., 0]  # 100 万维，只有 1 个位置是 1

# 问题：
# 1. 维度爆炸：100 万维
# 2. 稀疏：只有 1 个非零元素
# 3. 无法表达相似性：任意两个用户向量正交
```

**解决方案：Embedding**

```python
# 将用户 ID 映射到 64 维向量
user_embedding = [0.12, -0.34, 0.56, ...]  # 64 维稠密向量

# 优势：
# 1. 维度低：64 维
# 2. 稠密：所有元素都有值
# 3. 语义信息：相似用户有相似向量
```

### 1.3 Embedding 的作用

| 作用 | 说明 |
|------|------|
| 降维 | 从百万维降到几十维 |
| 密度化 | 从稀疏变成稠密 |
| 语义表示 | 相似对象距离近 |
| 泛化能力 | 可以处理未见过的相似对象 |
| 计算高效 | 向量运算快 |

## 2. 核心概念

### 2.1 Embedding 的数学表示

给定一个对象 $o$，Embedding 函数 $f$ 将其映射为向量：

$$e = f(o) \in \mathbb{R}^d$$

其中 $d$ 是 Embedding 维度。

**实现方式：查表**

```python
# Embedding 本质是一个查找表
# 假设有 V 个对象，每个对象映射到 d 维向量
embedding_table = np.random.randn(V, d)

# 获取某个对象的 Embedding
object_id = 12345
embedding = embedding_table[object_id]  # shape: (d,)
```

### 2.2 相似度计算

Embedding 向量可以计算相似度：

**余弦相似度：**
$$sim(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}$$

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**欧氏距离：**
$$dist(a, b) = ||a - b||_2$$

```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
```

**点积：**
$$score(a, b) = a \cdot b$$

```python
def dot_product(a, b):
    return np.dot(a, b)
```

### 2.3 Embedding 的学习

Embedding 通过训练学习得到，而不是人工设计。

**学习方式：**

1. **端到端学习**：Embedding 作为模型参数，随模型一起训练
2. **预训练**：先用无监督方法预训练，再用于下游任务
3. **矩阵分解**：通过矩阵分解得到 Embedding

## 3. Embedding 在推荐系统中的应用

### 3.1 用户 Embedding

```python
# 用户 Embedding 表示用户的兴趣、偏好
user_embedding = model.get_user_embedding(user_id)

# 用途：
# 1. 用户画像：分析用户兴趣
# 2. 相似用户：找兴趣相似的用户
# 3. 个性化推荐：与物品 Embedding 匹配
```

### 3.2 物品 Embedding

```python
# 物品 Embedding 表示物品的特征、属性
item_embedding = model.get_item_embedding(item_id)

# 用途：
# 1. 物品画像：分析物品特征
# 2. 相似物品：推荐相似物品
# 3. 物品聚类：物品分组
```

### 3.3 召回阶段

```python
# Embedding 召回
def embedding_recall(user_embedding, item_embeddings, top_k=100):
    """
    基于 Embedding 的召回

    参数:
        user_embedding: 用户 Embedding (d,)
        item_embeddings: 物品 Embedding 矩阵 (n_items, d)
        top_k: 返回 top-k 个物品

    返回:
        推荐物品 ID 列表
    """
    # 计算用户与所有物品的相似度
    scores = np.dot(item_embeddings, user_embedding)

    # 返回 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices, scores[top_indices]
```

## 4. Embedding 学习方法

### 4.1 矩阵分解

```python
import numpy as np

def learn_embedding_by_mf(rating_matrix, k=64, epochs=100, lr=0.01, reg=0.01):
    """
    通过矩阵分解学习 Embedding
    """
    n_users, n_items = rating_matrix.shape

    # 初始化 Embedding
    user_emb = np.random.normal(0, 0.1, (n_users, k))
    item_emb = np.random.normal(0, 0.1, (n_items, k))

    # 获取非零评分
    users, items = rating_matrix.nonzero()

    for epoch in range(epochs):
        for u, i in zip(users, items):
            # 预测
            pred = np.dot(user_emb[u], item_emb[i])
            error = rating_matrix[u, i] - pred

            # 更新
            user_emb[u] += lr * (error * item_emb[i] - reg * user_emb[u])
            item_emb[i] += lr * (error * user_emb[u] - reg * item_emb[i])

    return user_emb, item_emb
```

### 4.2 Word2Vec 思想

Word2Vec 的 Skip-gram 模型可以应用到推荐：

```python
# 用户行为序列类似句子，物品类似单词
# sequence = [item1, item2, item3, item4, item5]
# 用 item3 预测上下文 [item1, item2, item4, item5]

import torch
import torch.nn as nn

class Item2Vec(nn.Module):
    """Item2Vec: 将 Word2Vec 应用于物品序列"""

    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.center_embedding = nn.Embedding(num_items, embed_dim)
        self.context_embedding = nn.Embedding(num_items, embed_dim)

    def forward(self, center, context, negative):
        """
        参数:
            center: 中心物品 (batch,)
            context: 上下文物品 (batch,)
            negative: 负样本物品 (batch, num_neg)
        """
        # 正样本
        center_emb = self.center_embedding(center)  # (batch, dim)
        context_emb = self.context_embedding(context)  # (batch, dim)
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # (batch,)

        # 负样本
        neg_emb = self.context_embedding(negative)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # (batch, num_neg)

        # 损失
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        return (pos_loss + neg_loss).mean()
```

### 4.3 双塔模型

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    """
    双塔模型：用户塔和物品塔分别输出 Embedding
    """

    def __init__(self, user_feature_dims, item_feature_dims, embed_dim):
        super().__init__()

        # 用户塔
        self.user_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in user_feature_dims.items()
        })
        self.user_mlp = nn.Sequential(
            nn.Linear(len(user_feature_dims) * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 物品塔
        self.item_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in item_feature_dims.items()
        })
        self.item_mlp = nn.Sequential(
            nn.Linear(len(item_feature_dims) * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def get_user_embedding(self, user_features):
        """获取用户 Embedding"""
        embs = []
        for name, ids in user_features.items():
            embs.append(self.user_embeddings[name](ids))
        concat = torch.cat(embs, dim=-1)
        return self.user_mlp(concat)

    def get_item_embedding(self, item_features):
        """获取物品 Embedding"""
        embs = []
        for name, ids in item_features.items():
            embs.append(self.item_embeddings[name](ids))
        concat = torch.cat(embs, dim=-1)
        return self.item_mlp(concat)

    def forward(self, user_features, item_features):
        """计算用户-物品得分"""
        user_emb = self.get_user_embedding(user_features)
        item_emb = self.get_item_embedding(item_features)

        # 点积
        score = torch.sum(user_emb * item_emb, dim=-1)
        return torch.sigmoid(score)
```

## 5. Embedding 的性质

### 5.1 语义相似性

```python
# 好的 Embedding：相似对象的向量距离近
king = embedding['king']
queen = embedding['queen']
man = embedding['man']
woman = embedding['woman']

# 著名的例子：king - man + woman ≈ queen
result = king - man + woman
# result 应该和 queen 相似
```

### 5.2 线性关系

```python
# Embedding 空间中的线性关系
# 例如：用户向量 + 物品向量 ≈ 交互向量

# 或者：相似用户聚类
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
user_clusters = kmeans.fit_predict(user_embeddings)
```

### 5.3 可视化

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels=None):
    """可视化 Embedding"""
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                         c=labels if labels is not None else None,
                         alpha=0.6, cmap='tab10')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Embedding Visualization')
    if labels is not None:
        plt.colorbar(scatter)
    plt.show()
```

## 6. 工程实践

### 6.1 Embedding 存储

```python
import numpy as np
import faiss

class EmbeddingStore:
    """Embedding 存储与检索"""

    def __init__(self, dim):
        self.dim = dim
        self.embeddings = None
        self.id_map = {}
        self.index = None

    def add(self, item_id, embedding):
        """添加 Embedding"""
        self.id_map[item_id] = len(self.id_map)

        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

    def build_index(self):
        """构建索引"""
        self.index = faiss.IndexFlatIP(self.dim)  # 内积索引
        faiss.normalize_L2(self.embeddings)  # 归一化
        self.index.add(self.embeddings.astype('float32'))

    def search(self, query_embedding, top_k=10):
        """相似度搜索"""
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item_id = list(self.id_map.keys())[idx]
            results.append((item_id, score))

        return results
```

### 6.2 Embedding 更新

```python
class EmbeddingUpdater:
    """Embedding 增量更新"""

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.lr = learning_rate

    def update_user_embedding(self, user_id, positive_item_ids, negative_item_ids=None):
        """
        基于用户行为更新用户 Embedding
        """
        user_emb = self.model.get_user_embedding(user_id)

        # 正样本：拉近用户和正物品的距离
        for item_id in positive_item_ids:
            item_emb = self.model.get_item_embedding(item_id)
            # 向正物品方向移动
            user_emb += self.lr * (item_emb - user_emb)

        # 负样本：推远用户和负物品的距离
        if negative_item_ids:
            for item_id in negative_item_ids:
                item_emb = self.model.get_item_embedding(item_id)
                user_emb -= self.lr * (item_emb - user_emb) * 0.5

        self.model.set_user_embedding(user_id, user_emb)
```

### 6.3 冷启动处理

```python
class ColdStartHandler:
    """Embedding 冷启动处理"""

    def __init__(self, model):
        self.model = model

    def get_new_user_embedding(self, user_features):
        """
        新用户 Embedding
        方法：基于用户特征加权平均
        """
        # 获取用户的人口统计学特征
        age = user_features['age']
        gender = user_features['gender']
        city = user_features['city']

        # 加权平均已有用户的 Embedding
        similar_users = self.find_similar_users(age, gender, city)

        if similar_users:
            weights = np.array([u['similarity'] for u in similar_users])
            weights = weights / weights.sum()

            emb = np.zeros(self.model.embed_dim)
            for user, w in zip(similar_users, weights):
                emb += w * self.model.get_user_embedding(user['id'])
            return emb
        else:
            # 返回平均 Embedding
            return self.model.get_average_user_embedding()

    def get_new_item_embedding(self, item_features):
        """
        新物品 Embedding
        方法：基于物品内容特征
        """
        category = item_features['category']
        brand = item_features['brand']

        # 同类目物品的平均 Embedding
        same_cat_items = self.find_items_by_category(category)

        if same_cat_items:
            embs = [self.model.get_item_embedding(i) for i in same_cat_items]
            return np.mean(embs, axis=0)
        else:
            return self.model.get_average_item_embedding()
```

## 7. 评估 Embedding 质量

### 7.1 内在评估

```python
def evaluate_embedding_intrinsic(embeddings, labels):
    """
    内在评估：评估 Embedding 本身的质量
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import kneighbors_graph

    # 聚类质量
    silhouette = silhouette_score(embeddings, labels)

    # k-NN 准确率
    knn_graph = kneighbors_graph(embeddings, n_neighbors=5)
    # ... 计算 k-NN 准确率

    return {
        'silhouette_score': silhouette,
        # ...
    }
```

### 7.2 外在评估

```python
def evaluate_embedding_extrinsic(model, test_data):
    """
    外在评估：评估 Embedding 在下游任务上的效果
    """
    from sklearn.metrics import roc_auc_score

    predictions = []
    labels = []

    for user_id, item_id, label in test_data:
        user_emb = model.get_user_embedding(user_id)
        item_emb = model.get_item_embedding(item_id)
        score = np.dot(user_emb, item_emb)
        predictions.append(score)
        labels.append(label)

    auc = roc_auc_score(labels, predictions)

    return {
        'AUC': auc
    }
```

## 8. 常见问题与易错点

### 8.1 常见问题

**Q1：Embedding 维度如何选择？**

A：
- 通常在 16-256 之间
- 维度越高，表达能力越强，但计算量越大
- 可以通过实验选择最佳维度

**Q2：如何评估 Embedding 质量？**

A：
- 内在评估：聚类质量、相似度分布
- 外在评估：下游任务效果（如 CTR AUC）

**Q3：Embedding 需要归一化吗？**

A：
- 用于余弦相似度：需要归一化
- 用于点积：可以不归一化
- 通常归一化更稳定

### 8.2 易错点

1. **维度选择不当**：太小欠拟合，太大过拟合
2. **初始化问题**：初始化值太大会导致训练困难
3. **未处理冷启动**：新用户/物品没有 Embedding
4. **未定期更新**：Embedding 需要随着数据变化更新

## 9. 学习总结

### 9.1 核心要点

1. **Embedding 是稠密向量表示**：将离散 ID 映射到连续向量
2. **相似性编码**：相似对象的 Embedding 距离近
3. **通过训练学习**：端到端学习或预训练
4. **广泛应用**：召回、排序、用户画像等

### 9.2 知识图谱

```
Embedding
├── 基础概念
│   ├── 定义
│   ├── 相似度计算
│   └── 学习方法
├── 学习方法
│   ├── 矩阵分解
│   ├── Word2Vec
│   ├── 双塔模型
│   └── Graph Embedding
├── 应用
│   ├── 召回
│   ├── 排序
│   └── 用户画像
└── 工程实践
    ├── 存储
    ├── 更新
    └── 冷启动
```

## 10. 练习题

### 10.1 基础题

1. Embedding 相比 One-Hot 编码有什么优势？

2. 常用的相似度计算方法有哪些？

3. Embedding 是如何学习的？

### 10.2 进阶题

4. 实现一个基于矩阵分解的 Embedding 学习。

5. 比较不同 Embedding 维度对推荐效果的影响。

### 10.3 思考题

6. 如何设计一个 Embedding 系统，支持实时更新？

7. Embedding 在冷启动场景下如何处理？

## 11. 学习路径建议

### 11.1 前置知识

- [ ] 线性代数（向量、矩阵）
- [ ] 深度学习基础
- [ ] 推荐系统基础

### 11.2 学习顺序

1. 理解概念 → 什么是 Embedding
2. 学习矩阵分解 → MF 产生 Embedding
3. 学习 Word2Vec → 序列 Embedding
4. 学习双塔模型 → 工业 Embedding
5. 工程实践 → 存储、更新、召回

### 11.3 下一步学习

- **Item2Vec**：物品序列 Embedding
- **双塔模型**：DSSM、YouTube DNN
- **Graph Embedding**：DeepWalk、Node2Vec
- **Faiss**：向量检索
