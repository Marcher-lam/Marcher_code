# Faiss 向量检索 学习文档

## 1. 基础认知

### 1.1 什么是 Faiss？

Faiss（Facebook AI Similarity Search）是 Facebook 开源的高效向量相似度搜索库，专门用于大规模向量检索。

### 1.2 为什么需要 Faiss？

**问题场景：**

```python
# 假设有 1000 万物品，每个物品 128 维向量
item_vectors = np.random.randn(10000000, 128).astype('float32')

# 用户向量
user_vector = np.random.randn(1, 128).astype('float32')

# 暴力搜索：计算用户与所有物品的相似度
# 时间复杂度：O(n × d)
scores = np.dot(item_vectors, user_vector.T)  # 非常慢！
```

**Faiss 的解决方案：**
- 使用索引结构加速搜索
- 支持近似搜索（牺牲少量精度换取速度）
- 支持GPU加速
- 可处理十亿级向量

### 1.3 核心概念

| 概念 | 说明 |
|------|------|
| 向量（Vector） | 高维特征表示 |
| 索引（Index） | 数据结构，加速检索 |
| 距离度量 | Inner Product（内积）、L2（欧氏距离） |
| 召回率（Recall） | 正确结果被找到的比例 |
| 近似最近邻（ANN） | 可能不是精确最优，但足够接近 |

## 2. 索引类型

### 2.1 暴力搜索 IndexFlatIP / IndexFlatL2

最精确但最慢的索引。

```python
import faiss
import numpy as np

# 准备数据
d = 128  # 向量维度
nb = 100000  # 向量数量
vectors = np.random.randn(nb, d).astype('float32')

# 创建内积索引（需要先归一化）
faiss.normalize_L2(vectors)  # 归一化
index = faiss.IndexFlatIP(d)

# 添加向量
index.add(vectors)
print(f"索引包含 {index.ntotal} 个向量")

# 搜索
k = 10  # 返回 top-k
query = np.random.randn(1, d).astype('float32')
faiss.normalize_L2(query)

distances, indices = index.search(query, k)
print(f"Top-{k} 索引: {indices}")
print(f"Top-{k} 距离: {distances}")

# L2 距离索引
index_l2 = faiss.IndexFlatL2(d)
index_l2.add(vectors)
distances, indices = index_l2.search(query, k)
```

### 2.2 IVF 索引（倒排索引）

将向量聚类到多个桶（Voronoi cell），搜索时只搜索最近的几个桶。

```python
import faiss
import numpy as np

d = 128
nb = 100000
vectors = np.random.randn(nb, d).astype('float32')
faiss.normalize_L2(vectors)

# 定义量化器
nlist = 100  # 聚类中心数量
quantizer = faiss.IndexFlatIP(d)

# 创建 IVF 索引
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# 训练（聚类）
index.train(vectors[:10000])  # 使用部分数据训练
print(f"训练完成: {index.is_trained}")

# 添加向量
index.add(vectors)

# 搜索
k = 10
nprobe = 10  # 搜索的桶数量
index.nprobe = nprobe

query = np.random.randn(1, d).astype('float32')
faiss.normalize_L2(query)

distances, indices = index.search(query, k)
```

### 2.3 IVF + PQ（乘积量化）

使用乘积量化压缩向量，减少内存占用。

```python
import faiss
import numpy as np

d = 128
nb = 1000000  # 100万向量
vectors = np.random.randn(nb, d).astype('float32')

# IVF + PQ 参数
nlist = 1000   # 聚类中心数
m = 8          # 子向量数量
nbits = 8      # 每个子向量的比特数

# 创建索引
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# 训练
index.train(vectors[:50000])

# 添加
index.add(vectors)

# 搜索
index.nprobe = 32  # 搜索更多桶提高精度
k = 10
query = np.random.randn(1, d).astype('float32')

distances, indices = index.search(query, k)

# 内存占用对比
print(f"原始向量内存: {nb * d * 4 / 1024 / 1024:.2f} MB")
print(f"PQ 压缩后: {nb * m * nbits / 8 / 1024 / 1024:.2f} MB")
```

### 2.4 HNSW（层次化可导航小世界图）

基于图的索引，搜索速度快。

```python
import faiss

d = 128
nb = 100000
vectors = np.random.randn(nb, d).astype('float32')

# HNSW 参数
M = 32  # 每个节点的连接数
index = faiss.IndexHNSWFlat(d, M)

# 添加（HNSW 不需要训练）
index.add(vectors)

# 搜索
k = 10
query = np.random.randn(1, d).astype('float32')

distances, indices = index.search(query, k)
```

## 3. 索引选择指南

### 3.1 索引对比

| 索引 | 搜索速度 | 内存占用 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| IndexFlatIP | 慢 | 大 | 100% | 小数据、精确搜索 |
| IndexIVFFlat | 中 | 大 | 高 | 中等数据 |
| IndexIVFPQ | 快 | 小 | 中 | 大数据、内存受限 |
| IndexHNSW | 很快 | 大 | 高 | 实时搜索 |
| IndexIVFSQ | 很快 | 很小 | 低 | 超大数据 |

### 3.2 选择流程

```python
def select_index(num_vectors, dim, memory_limit_mb, latency_req_ms):
    """
    根据需求选择索引

    参数:
        num_vectors: 向量数量
        dim: 向量维度
        memory_limit_mb: 内存限制（MB）
        latency_req_ms: 延迟要求（毫秒）
    """
    # 估算原始向量内存
    raw_memory_mb = num_vectors * dim * 4 / 1024 / 1024

    if num_vectors < 100000:
        # 小数据：暴力搜索
        return "IndexFlatIP"

    if memory_limit_mb < raw_memory_mb * 0.3:
        # 内存受限：PQ 压缩
        return "IndexIVFPQ"

    if latency_req_ms < 10:
        # 低延迟要求：HNSW
        return "IndexHNSW"

    # 默认：IVF
    return "IndexIVFFlat"
```

## 4. 完整封装

### 4.1 向量检索服务

```python
import faiss
import numpy as np
import os
from typing import List, Tuple, Optional


class VectorSearchEngine:
    """
    向量检索引擎封装
    """

    def __init__(self, dim: int, index_type: str = 'ivf',
                 nlist: int = 100, nprobe: int = 10,
                 use_gpu: bool = False):
        """
        参数:
            dim: 向量维度
            index_type: 索引类型 ('flat', 'ivf', 'ivfpq', 'hnsw')
            nlist: IVF 聚类中心数量
            nprobe: 搜索时探测的聚类数
            use_gpu: 是否使用 GPU
        """
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self.index = None
        self.id_map = {}  # 向量索引到原始 ID 的映射
        self.is_trained = False

    def build_index(self, vectors: np.ndarray, ids: List = None):
        """
        构建索引

        参数:
            vectors: (n, dim) 向量矩阵
            ids: 可选的 ID 列表
        """
        vectors = vectors.astype('float32')
        n = vectors.shape[0]

        # 归一化（用于内积搜索）
        faiss.normalize_L2(vectors)

        # 创建索引
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dim)

        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.nprobe

        elif self.index_type == 'ivfpq':
            m = 8  # 子向量数
            nbits = 8
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dim, self.nlist, m, nbits
            )
            self.index.nprobe = self.nprobe

        elif self.index_type == 'hnsw':
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dim, M)

        # GPU 加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # 训练（如果需要）
        if hasattr(self.index, 'train'):
            train_size = min(n, self.nlist * 100)
            self.index.train(vectors[:train_size])

        # 添加向量
        self.index.add(vectors)

        # ID 映射
        if ids is not None:
            self.id_map = {i: id_ for i, id_ in enumerate(ids)}
        else:
            self.id_map = {i: i for i in range(n)}

        self.is_trained = True
        print(f"索引构建完成，共 {self.index.ntotal} 个向量")

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[List, List]:
        """
        搜索

        参数:
            query: (dim,) 或 (batch, dim) 查询向量
            top_k: 返回数量

        返回:
            ids: 原始 ID 列表
            scores: 相似度列表
        """
        if not self.is_trained:
            raise ValueError("索引未训练")

        # 预处理
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype('float32')
        faiss.normalize_L2(query)

        # 搜索
        scores, indices = self.index.search(query, top_k)

        # 转换为原始 ID
        results_ids = []
        results_scores = []

        for i in range(len(query)):
            ids = [self.id_map.get(idx, -1) for idx in indices[i]]
            results_ids.append(ids)
            results_scores.append(scores[i].tolist())

        if len(results_ids) == 1:
            return results_ids[0], results_scores[0]
        return results_ids, results_scores

    def add_vectors(self, vectors: np.ndarray, ids: List):
        """
        增量添加向量

        注意：某些索引类型不支持增量添加
        """
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)

        start_idx = self.index.ntotal
        self.index.add(vectors)

        for i, id_ in enumerate(ids):
            self.id_map[start_idx + i] = id_

    def save(self, path: str):
        """保存索引"""
        # GPU 索引需要先转到 CPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)

        # 保存 ID 映射
        import json
        with open(path + '.map', 'w') as f:
            json.dump({str(k): v for k, v in self.id_map.items()}, f)

    def load(self, path: str):
        """加载索引"""
        self.index = faiss.read_index(path)

        # GPU 加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # 加载 ID 映射
        import json
        with open(path + '.map', 'r') as f:
            self.id_map = {int(k): v for k, v in json.load(f).items()}

        self.is_trained = True


# 使用示例
if __name__ == "__main__":
    # 创建引擎
    engine = VectorSearchEngine(
        dim=128,
        index_type='ivf',
        nlist=100,
        nprobe=10
    )

    # 构建索引
    num_items = 10000
    item_vectors = np.random.randn(num_items, 128)
    item_ids = [f"item_{i}" for i in range(num_items)]

    engine.build_index(item_vectors, item_ids)

    # 搜索
    query = np.random.randn(128)
    ids, scores = engine.search(query, top_k=10)

    print("搜索结果:")
    for id_, score in zip(ids, scores):
        print(f"  {id_}: {score:.4f}")

    # 保存
    engine.save("item_index.faiss")

    # 加载
    engine2 = VectorSearchEngine(dim=128)
    engine2.load("item_index.faiss")
```

## 5. 性能优化

### 5.1 参数调优

```python
def tune_ivf_index(vectors, ground_truth, nlist_range, nprobe_range):
    """
    调优 IVF 索引参数
    """
    results = []

    for nlist in nlist_range:
        for nprobe in nprobe_range:
            # 构建索引
            quantizer = faiss.IndexFlatIP(vectors.shape[1])
            index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist)
            index.train(vectors[:nlist * 100])
            index.add(vectors)
            index.nprobe = nprobe

            # 搜索
            k = 10
            _, indices = index.search(vectors[:100], k)

            # 计算召回率
            recall = 0
            for i in range(100):
                recall += len(set(indices[i]) & set(ground_truth[i][:k])) / k
            recall /= 100

            results.append({
                'nlist': nlist,
                'nprobe': nprobe,
                'recall': recall
            })

    return results
```

### 5.2 批量搜索

```python
def batch_search(index, queries, batch_size=1000, top_k=10):
    """
    批量搜索，避免内存溢出
    """
    all_ids = []
    all_scores = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        scores, indices = index.search(batch, top_k)
        all_ids.append(indices)
        all_scores.append(scores)

    return np.vstack(all_ids), np.vstack(all_scores)
```

## 6. 推荐系统应用

```python
class RecSysVectorRetrieval:
    """
    推荐系统的向量召回
    """

    def __init__(self, dim=128):
        self.engine = VectorSearchEngine(dim=dim, index_type='ivf')
        self.user_vectors = {}
        self.item_vectors = {}

    def index_items(self, item_embeddings: dict):
        """
        索引物品向量

        参数:
            item_embeddings: {item_id: embedding}
        """
        item_ids = list(item_embeddings.keys())
        vectors = np.array([item_embeddings[id_] for id_ in item_ids])

        self.engine.build_index(vectors, item_ids)
        self.item_vectors = item_embeddings

    def update_user_vector(self, user_id: str, embedding: np.ndarray):
        """更新用户向量"""
        self.user_vectors[user_id] = embedding

    def recall(self, user_id: str, top_k: int = 100,
               exclude_items: list = None) -> list:
        """
        召回

        参数:
            user_id: 用户 ID
            top_k: 返回数量
            exclude_items: 排除的物品列表

        返回:
            [(item_id, score), ...]
        """
        if user_id not in self.user_vectors:
            return []

        query = self.user_vectors[user_id]
        ids, scores = self.engine.search(query, top_k * 2)  # 多取一些

        # 排除已交互物品
        results = []
        exclude_set = set(exclude_items or [])

        for id_, score in zip(ids, scores):
            if id_ not in exclude_set:
                results.append((id_, score))
                if len(results) >= top_k:
                    break

        return results
```

## 7. 常见问题

### 7.1 索引选择

**Q：什么时候用 HNSW，什么时候用 IVF？**

A：
- HNSW：低延迟、高精度、内存充足
- IVF：大数据、内存受限、可接受近似

### 7.2 性能调优

**Q：如何提高召回率？**

A：
- 增加 nprobe（IVF）
- 增加 M（HNSW）
- 减少量化压缩（PQ）

## 8. 学习总结

### 8.1 核心要点

1. **Faiss 是高效的向量检索库**
2. **索引类型选择取决于场景**
3. **近似搜索牺牲精度换速度**
4. **参数调优很重要**

### 8.2 知识图谱

```
Faiss
├── 索引类型
│   ├── IndexFlat（暴力）
│   ├── IndexIVF（倒排）
│   ├── IndexPQ（量化）
│   └── IndexHNSW（图）
├── 距离度量
│   ├── Inner Product
│   └── L2 Distance
└── 应用
    ├── 召回
    ├── 相似物品
    └── 去重
```

## 9. 练习题

1. 实现一个支持增量更新的向量检索服务。

2. 比较不同索引类型的搜索速度和召回率。

3. 实现一个基于 Faiss 的推荐召回模块。
