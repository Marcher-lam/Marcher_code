# 向量召回与ANN检索 学习文档

## 1. 向量召回概述

### 1.1 什么是向量召回？

向量召回是将用户和物品映射到同一向量空间，通过向量相似度检索召回候选物品。

### 1.2 核心优势

```
1. 语义相似性: 捕捉深层语义关系
2. 泛化能力: 解决稀疏性问题
3. 高效检索: ANN 加速
4. 多模态融合: 可融合文本、图像等
```

## 2. 向量化方法

### 2.1 基于矩阵分解

```python
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class MatrixFactorizationEmbedding:
    """
    基于矩阵分解的嵌入
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        # 初始化嵌入
        self.user_embeddings = np.random.normal(0, 0.1, (n_users, embed_dim))
        self.item_embeddings = np.random.normal(0, 0.1, (n_items, embed_dim))

        # 偏置
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0

    def fit(self, interactions: List[Tuple[int, int, float]],
            n_epochs: int = 20, lr: float = 0.01, reg: float = 0.02):
        """
        训练

        参数:
            interactions: [(user_id, item_id, rating), ...]
        """
        for epoch in range(n_epochs):
            np.random.shuffle(interactions)
            total_loss = 0

            for user_id, item_id, rating in interactions:
                # 预测
                pred = (self.global_bias +
                       self.user_bias[user_id] +
                       self.item_bias[item_id] +
                       np.dot(self.user_embeddings[user_id],
                              self.item_embeddings[item_id]))

                # 误差
                error = rating - pred
                total_loss += error ** 2

                # 更新偏置
                self.user_bias[user_id] += lr * (error - reg * self.user_bias[user_id])
                self.item_bias[item_id] += lr * (error - reg * self.item_bias[item_id])

                # 更新嵌入
                user_embed = self.user_embeddings[user_id].copy()
                self.user_embeddings[user_id] += lr * (
                    error * self.item_embeddings[item_id] -
                    reg * self.user_embeddings[user_id]
                )
                self.item_embeddings[item_id] += lr * (
                    error * user_embed -
                    reg * self.item_embeddings[item_id]
                )

            rmse = np.sqrt(total_loss / len(interactions))
            print(f"Epoch {epoch+1}: RMSE = {rmse:.4f}")

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取相似物品"""
        item_vec = self.item_embeddings[item_id]
        scores = np.dot(self.item_embeddings, item_vec)
        top_indices = np.argsort(scores)[::-1][1:top_k+1]  # 排除自己
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class Item2VecEmbedding:
    """
    Item2Vec 嵌入

    基于 Word2Vec 的物品嵌入
    """

    def __init__(self, embed_dim: int = 64, window_size: int = 5):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.item_embeddings = {}

    def train(self, item_sequences: List[List[int]], n_items: int,
              n_epochs: int = 10, lr: float = 0.025):
        """
        Skip-gram 训练

        参数:
            item_sequences: 用户行为序列 [[item1, item2, ...], ...]
        """
        # 初始化嵌入
        self.item_embeddings = {
            i: np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, self.embed_dim)
            for i in range(n_items)
        }

        # 生成训练样本
        samples = []
        for seq in item_sequences:
            for i, center in enumerate(seq):
                # 上下文窗口
                start = max(0, i - self.window_size)
                end = min(len(seq), i + self.window_size + 1)

                for j in range(start, end):
                    if j != i:
                        samples.append((center, seq[j]))

        # 负采样参数
        n_negative = 5
        item_counts = defaultdict(int)
        for seq in item_sequences:
            for item in seq:
                item_counts[item] += 1

        # 构建负采样表（按频率的3/4次方）
        total = sum(count ** 0.75 for count in item_counts.values())
        neg_table = []
        for item, count in item_counts.items():
            prob = (count ** 0.75) / total
            neg_table.extend([item] * int(prob * 1e6))

        # 训练
        for epoch in range(n_epochs):
            np.random.shuffle(samples)
            total_loss = 0

            for center, context in samples:
                # 负采样
                negatives = []
                while len(negatives) < n_negative:
                    neg = neg_table[np.random.randint(len(neg_table))]
                    if neg != context:
                        negatives.append(neg)

                # 前向传播
                center_vec = self.item_embeddings[center]
                context_vec = self.item_embeddings[context]

                # 正样本: sigmoid(dot) → 1
                pos_score = np.dot(center_vec, context_vec)
                pos_sigmoid = 1 / (1 + np.exp(-np.clip(pos_score, -10, 10)))
                pos_grad = (pos_sigmoid - 1)

                # 负样本: sigmoid(dot) → 0
                for neg in negatives:
                    neg_vec = self.item_embeddings[neg]
                    neg_score = np.dot(center_vec, neg_vec)
                    neg_sigmoid = 1 / (1 + np.exp(-np.clip(neg_score, -10, 10)))
                    neg_grad = neg_sigmoid

                    # 更新负样本
                    self.item_embeddings[neg] -= lr * neg_grad * center_vec
                    center_vec -= lr * neg_grad * neg_vec

                # 更新正样本
                self.item_embeddings[context] -= lr * pos_grad * center_vec
                self.item_embeddings[center] -= lr * pos_grad * context_vec

            print(f"Epoch {epoch+1} complete")
```

### 2.2 双塔模型嵌入

```python
import torch
import torch.nn as nn


class TwoTowerEmbedding(nn.Module):
    """
    双塔模型嵌入
    """

    def __init__(self, user_feature_dim: int, item_feature_dim: int,
                 hidden_dim: int = 256, embed_dim: int = 64):
        super().__init__()

        # 用户塔
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim)
        )

        # 物品塔
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim)
        )

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """编码用户"""
        embed = self.user_tower(user_features)
        return nn.functional.normalize(embed, p=2, dim=-1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """编码物品"""
        embed = self.item_tower(item_features)
        return nn.functional.normalize(embed, p=2, dim=-1)

    def forward(self, user_features: torch.Tensor,
               item_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        user_embed = self.encode_user(user_features)
        item_embed = self.encode_item(item_features)
        return torch.sum(user_embed * item_embed, dim=-1)


class TwoTowerTrainer:
    """
    双塔模型训练器
    """

    def __init__(self, model: TwoTowerEmbedding, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def contrastive_loss(self, user_embeds: torch.Tensor,
                        item_embeds: torch.Tensor) -> torch.Tensor:
        """
        对比学习损失（InfoNCE）

        同一批次内的正负样本对比
        """
        batch_size = user_embeds.size(0)

        # 相似度矩阵
        sim_matrix = torch.matmul(user_embeds, item_embeds.T) / self.temperature

        # 对角线为正样本
        labels = torch.arange(batch_size, device=user_embeds.device)

        # 交叉熵损失
        loss = nn.functional.cross_entropy(sim_matrix, labels)

        return loss

    def train_batch(self, user_features: torch.Tensor,
                   item_features: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        """训练一个批次"""
        self.model.train()
        optimizer.zero_grad()

        user_embeds = self.model.encode_user(user_features)
        item_embeds = self.model.encode_item(item_features)

        loss = self.contrastive_loss(user_embeds, item_embeds)

        loss.backward()
        optimizer.step()

        return loss.item()
```

## 3. ANN 近似检索

### 3.1 暴力搜索基线

```python
class BruteForceSearch:
    """
    暴力搜索（基线）
    """

    def __init__(self, embeddings: np.ndarray):
        """
        参数:
            embeddings: 物品嵌入矩阵 (n_items, embed_dim)
        """
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / (norms + 1e-10)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索

        复杂度: O(n × d)
        """
        # 归一化查询
        query = query / (np.linalg.norm(query) + 1e-10)

        # 计算相似度
        scores = np.dot(self.embeddings, query)

        # Top-K
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class BruteForceSearchBatch:
    """
    批量暴力搜索
    """

    def __init__(self, embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / (norms + 1e-10)

    def search_batch(self, queries: np.ndarray,
                    top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """批量搜索"""
        # 归一化
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-10)

        # 批量计算
        scores = np.dot(queries, self.embeddings.T)

        results = []
        for i in range(len(queries)):
            top_indices = np.argpartition(scores[i], -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[i][top_indices])[::-1]]
            results.append([(int(idx), float(scores[i, idx])) for idx in top_indices])

        return results
```

### 3.2 局部敏感哈希 (LSH)

```python
class LocalitySensitiveHashing:
    """
    局部敏感哈希 (LSH)

    使用随机投影将相似向量映射到相同桶
    """

    def __init__(self, embed_dim: int, n_tables: int = 10, n_hashes: int = 12):
        """
        参数:
            embed_dim: 向量维度
            n_tables: 哈希表数量
            n_hashes: 每个表的哈希函数数量
        """
        self.embed_dim = embed_dim
        self.n_tables = n_tables
        self.n_hashes = n_hashes

        # 随机投影矩阵
        self.projections = [
            np.random.randn(n_hashes, embed_dim)
            for _ in range(n_tables)
        ]

        # 哈希表
        self.hash_tables = [defaultdict(list) for _ in range(n_tables)]

        self.item_embeddings = None

    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        """计算哈希值"""
        projection = self.projections[table_idx]
        bits = (np.dot(projection, vector) > 0).astype(int)
        return ''.join(map(str, bits))

    def index(self, embeddings: np.ndarray):
        """构建索引"""
        self.item_embeddings = embeddings
        n_items = len(embeddings)

        for table_idx in range(self.n_tables):
            for item_id in range(n_items):
                hash_key = self._hash(embeddings[item_id], table_idx)
                self.hash_tables[table_idx][hash_key].append(item_id)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索"""
        # 收集候选
        candidates = set()
        for table_idx in range(self.n_tables):
            hash_key = self._hash(query, table_idx)
            candidates.update(self.hash_tables[table_idx][hash_key])

        if not candidates:
            # 没有候选，返回随机结果
            candidates = set(range(min(top_k * 10, len(self.item_embeddings))))

        # 精确计算
        candidates = list(candidates)
        candidate_embeddings = self.item_embeddings[candidates]

        # 归一化
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        cand_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )

        scores = np.dot(cand_norms, query_norm)

        # Top-K
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        return [(candidates[idx], float(scores[idx])) for idx in top_k_indices]


class MultiProbeLSH(LocalitySensitiveHashing):
    """
    Multi-Probe LSH

    探测多个相近的桶以提高召回率
    """

    def __init__(self, embed_dim: int, n_tables: int = 10,
                 n_hashes: int = 12, n_probes: int = 3):
        super().__init__(embed_dim, n_tables, n_hashes)
        self.n_probes = n_probes

    def _get_nearby_hashes(self, base_hash: str, table_idx: int,
                          vector: np.ndarray) -> List[str]:
        """获取附近的哈希桶"""
        hashes = [base_hash]

        projection = self.projections[table_idx]
        margins = np.abs(np.dot(projection, vector))

        # 找到最接近边界的位
        bit_scores = [(i, margins[i]) for i in range(len(margins))]
        bit_scores.sort(key=lambda x: x[1])

        # 翻转最接近边界的几位
        for i in range(min(self.n_probes, len(bit_scores))):
            bit_idx = bit_scores[i][0]
            new_hash = list(base_hash)
            new_hash[bit_idx] = '1' if new_hash[bit_idx] == '0' else '0'
            hashes.append(''.join(new_hash))

        return hashes

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索（多探测）"""
        candidates = set()

        for table_idx in range(self.n_tables):
            base_hash = self._hash(query, table_idx)
            nearby_hashes = self._get_nearby_hashes(base_hash, table_idx, query)

            for h in nearby_hashes:
                candidates.update(self.hash_tables[table_idx].get(h, []))

        if not candidates:
            candidates = set(range(min(top_k * 10, len(self.item_embeddings))))

        # 精确计算
        candidates = list(candidates)
        candidate_embeddings = self.item_embeddings[candidates]

        query_norm = query / (np.linalg.norm(query) + 1e-10)
        cand_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )

        scores = np.dot(cand_norms, query_norm)
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        return [(candidates[idx], float(scores[idx])) for idx in top_k_indices]
```

### 3.3 乘积量化 (PQ)

```python
class ProductQuantization:
    """
    乘积量化 (Product Quantization)

    将向量分割成子空间，分别量化
    """

    def __init__(self, embed_dim: int, n_subspaces: int = 8, n_centroids: int = 256):
        """
        参数:
            embed_dim: 向量维度
            n_subspaces: 子空间数量
            n_centroids: 每个子空间的聚类中心数（通常为256）
        """
        assert embed_dim % n_subspaces == 0, "embed_dim 必须能被 n_subspaces 整除"

        self.embed_dim = embed_dim
        self.n_subspaces = n_subspaces
        self.subspace_dim = embed_dim // n_subspaces
        self.n_centroids = n_centroids

        # 每个子空间的聚类中心
        self.centroids = np.zeros((n_subspaces, n_centroids, self.subspace_dim))

        # 量化后的编码
        self.codes = None
        self.item_embeddings = None

    def train(self, embeddings: np.ndarray, n_iter: int = 20):
        """
        训练量化器（K-Means）
        """
        n_items = len(embeddings)

        for m in range(self.n_subspaces):
            # 提取子空间数据
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            sub_vectors = embeddings[:, start:end]

            # K-Means
            centroids = self._kmeans(sub_vectors, self.n_centroids, n_iter)
            self.centroids[m] = centroids

    def _kmeans(self, data: np.ndarray, k: int, n_iter: int) -> np.ndarray:
        """K-Means 实现"""
        n = len(data)

        # 随机初始化
        idx = np.random.choice(n, k, replace=False)
        centroids = data[idx].copy()

        for _ in range(n_iter):
            # 分配
            distances = np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)

            # 更新
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            centroids = new_centroids

        return centroids

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """编码"""
        n_items = len(embeddings)
        codes = np.zeros((n_items, self.n_subspaces), dtype=np.uint8)

        for m in range(self.n_subspaces):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            sub_vectors = embeddings[:, start:end]

            # 找最近的聚类中心
            distances = np.sum(
                (sub_vectors[:, np.newaxis] - self.centroids[m]) ** 2,
                axis=2
            )
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def index(self, embeddings: np.ndarray):
        """构建索引"""
        self.item_embeddings = embeddings
        self.train(embeddings)
        self.codes = self.encode(embeddings)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索"""
        # 预计算查询到各聚类中心的距离
        # 距离表: (n_subspaces, n_centroids)
        distance_table = np.zeros((self.n_subspaces, self.n_centroids))

        for m in range(self.n_subspaces):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            query_sub = query[start:end]

            # 到所有聚类中心的距离
            distance_table[m] = np.sum(
                (self.centroids[m] - query_sub) ** 2, axis=1
            )

        # 非对称距离计算 (ADC)
        n_items = len(self.codes)
        distances = np.zeros(n_items)

        for i in range(n_items):
            for m in range(self.n_subspaces):
                distances[i] += distance_table[m, self.codes[i, m]]

        # Top-K（最小距离）
        top_indices = np.argpartition(distances, top_k)[:top_k]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        return [(int(idx), float(-distances[idx])) for idx in top_indices]


class OptimizedProductQuantization(ProductQuantization):
    """
    优化乘积量化 (OPQ)

    在 PQ 之前先对向量空间进行旋转优化
    """

    def __init__(self, embed_dim: int, n_subspaces: int = 8, n_centroids: int = 256):
        super().__init__(embed_dim, n_subspaces, n_centroids)
        self.rotation = np.eye(embed_dim)  # 旋转矩阵

    def train(self, embeddings: np.ndarray, n_iter: int = 20):
        """训练（带旋转优化）"""
        # 初始化旋转为单位矩阵
        self.rotation = np.eye(self.embed_dim)

        # 交替优化
        for it in range(n_iter):
            # 应用旋转
            rotated = np.dot(embeddings, self.rotation.T)

            # 更新聚类中心
            for m in range(self.n_subspaces):
                start = m * self.subspace_dim
                end = start + self.subspace_dim
                sub_vectors = rotated[:, start:end]

                centroids = self._kmeans(sub_vectors, self.n_centroids, 5)
                self.centroids[m] = centroids

            # 更新旋转矩阵（简化：使用 PCA）
            # 实际应使用更复杂的优化方法
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.embed_dim)
            pca.fit(rotated)
            self.rotation = pca.components_.T

    def index(self, embeddings: np.ndarray):
        """构建索引"""
        self.item_embeddings = embeddings
        self.train(embeddings)
        rotated = np.dot(embeddings, self.rotation.T)
        self.codes = self.encode(rotated)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索"""
        # 旋转查询
        rotated_query = np.dot(query, self.rotation.T)
        return super().search(rotated_query, top_k)
```

## 4. Faiss 实战

### 4.1 Faiss 基础使用

```python
class FaissIndex:
    """
    Faiss 索引封装
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.index = None
        self.id_map = {}

    def build_flat_index(self, embeddings: np.ndarray, ids: List[int] = None):
        """
        构建暴力索引（精确）

        IndexFlatIP: 内积
        IndexFlatL2: L2 距离
        """
        import faiss

        n_items, dim = embeddings.shape
        assert dim == self.embed_dim

        # 归一化后使用内积等价于余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(normalized.astype('float32'))

        if ids:
            self.id_map = {i: ids[i] for i in range(len(ids))}

        print(f"索引构建完成: {n_items} 个向量")

    def build_ivf_index(self, embeddings: np.ndarray, n_clusters: int = 100,
                       n_probe: int = 10):
        """
        构建 IVF 索引（倒排索引）
        """
        import faiss

        n_items, dim = embeddings.shape

        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # 量化器
        quantizer = faiss.IndexFlatIP(dim)

        # IVF 索引
        self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters,
                                        faiss.METRIC_INNER_PRODUCT)

        # 训练
        self.index.train(normalized.astype('float32'))
        self.index.add(normalized.astype('float32'))

        # 设置搜索时探测的聚类数
        self.index.nprobe = n_probe

        print(f"IVF 索引构建完成: {n_clusters} 个聚类")

    def build_hnsw_index(self, embeddings: np.ndarray, m: int = 32,
                        ef_construction: int = 40):
        """
        构建 HNSW 索引（图索引）
        """
        import faiss

        n_items, dim = embeddings.shape

        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # HNSW 索引
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction

        self.index.add(normalized.astype('float32'))

        print(f"HNSW 索引构建完成")

    def build_pq_index(self, embeddings: np.ndarray, n_subquantizers: int = 8,
                       n_bits: int = 8):
        """
        构建 PQ 索引
        """
        import faiss

        n_items, dim = embeddings.shape

        self.index = faiss.IndexPQ(dim, n_subquantizers, n_bits)

        # 训练
        self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))

        print(f"PQ 索引构建完成")

    def build_ivf_pq_index(self, embeddings: np.ndarray, n_clusters: int = 1000,
                          n_subquantizers: int = 8, n_bits: int = 8):
        """
        构建 IVF-PQ 索引（混合索引）
        """
        import faiss

        n_items, dim = embeddings.shape

        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, n_clusters,
                                      n_subquantizers, n_bits)

        self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))

        print(f"IVF-PQ 索引构建完成")

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索"""
        if self.index is None:
            return []

        # 归一化
        query = query.reshape(1, -1)
        norm = np.linalg.norm(query)
        query = query / (norm + 1e-10)

        distances, indices = self.index.search(query.astype('float32'), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                original_id = self.id_map.get(idx, idx)
                results.append((int(original_id), float(dist)))

        return results

    def search_batch(self, queries: np.ndarray,
                    top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """批量搜索"""
        if self.index is None:
            return []

        # 归一化
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-10)

        distances, indices = self.index.search(queries.astype('float32'), top_k)

        results = []
        for i in range(len(queries)):
            batch_result = []
            for idx, dist in zip(indices[i], distances[i]):
                if idx >= 0:
                    original_id = self.id_map.get(idx, idx)
                    batch_result.append((int(original_id), float(dist)))
            results.append(batch_result)

        return results

    def save(self, path: str):
        """保存索引"""
        import faiss
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """加载索引"""
        import faiss
        self.index = faiss.read_index(path)
```

## 5. 召回系统集成

### 5.1 完整向量召回流程

```python
class VectorRecallPipeline:
    """
    向量召回完整流程
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.user_encoder = None
        self.item_encoder = None
        self.faiss_index = None

    def train_embeddings(self, interactions: List[Tuple],
                        user_features: Dict, item_features: Dict,
                        n_epochs: int = 20):
        """训练嵌入"""
        # 构建双塔模型
        user_dim = len(next(iter(user_features.values())))
        item_dim = len(next(iter(item_features.values())))

        model = TwoTowerEmbedding(user_dim, item_dim, 256, self.embed_dim)
        trainer = TwoTowerTrainer(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练
        for epoch in range(n_epochs):
            np.random.shuffle(interactions)
            total_loss = 0

            for user_id, item_id, _ in interactions[:10000]:
                u_feat = torch.FloatTensor(user_features[user_id]).unsqueeze(0)
                i_feat = torch.FloatTensor(item_features[item_id]).unsqueeze(0)

                loss = trainer.train_batch(u_feat, i_feat, optimizer)
                total_loss += loss

            print(f"Epoch {epoch+1}: Loss = {total_loss/10000:.4f}")

        self.user_encoder = model
        self.item_encoder = model

    def build_item_index(self, item_features: Dict):
        """构建物品索引"""
        item_ids = list(item_features.keys())
        item_matrix = np.array([item_features[i] for i in item_ids])

        # 编码物品
        with torch.no_grad():
            item_tensor = torch.FloatTensor(item_matrix)
            item_embeds = self.item_encoder.encode_item(item_tensor).numpy()

        # 构建 Faiss 索引
        self.faiss_index = FaissIndex(self.embed_dim)
        self.faiss_index.build_ivf_index(item_embeds, n_clusters=100)

        return item_embeds

    def recall(self, user_id: int, user_features: Dict,
              top_k: int = 100) -> List[Tuple[int, float]]:
        """召回"""
        # 编码用户
        user_vec = torch.FloatTensor(user_features[user_id]).unsqueeze(0)
        with torch.no_grad():
            user_embed = self.user_encoder.encode_user(user_vec).numpy()[0]

        # 搜索
        return self.faiss_index.search(user_embed, top_k)
```

## 6. 学习总结

### 6.1 核心要点

1. **向量召回**: 语义相似性，泛化能力强
2. **嵌入方法**: 矩阵分解、双塔模型、对比学习
3. **ANN 算法**: LSH、PQ、HNSW 各有优劣
4. **Faiss**: 工业界首选的向量检索库

### 6.2 算法选择指南

```
数据规模          推荐索引
─────────────────────────────
< 100万          IndexFlatIP (精确)
100万-1000万     IndexIVFFlat
> 1000万         IndexIVFPQ / HNSW
超大规模         分布式索引
```

### 6.3 最佳实践

```
1. 归一化: 使用内积前先归一化
2. 维度选择: 64-256 之间权衡
3. 索引更新: 增量更新 vs 全量重建
4. 召回率监控: 定期评估 ANN 召回率
```

