# 面试题：快手 UniDex 介绍，一种基于语义 ID 的新型倒排索引技术

- 参考论文：https://arxiv.org/pdf/2509.24632
- 参考文章：https://mp.weixin.qq.com/s/e0-2svkQ2IaWT1u8LkzRDg
- 概要：快手提出的UniDex是一项对搜索引擎核心机制— 倒排索引— 进行彻底革新的技术。
- 核心思想：不再使用传统的关键词作为索引和检索的基本单位，而是利用大模型生成的语义 ID 来构建索引，让搜索系统能真正理解用户的意图，而不再只是进行字面匹配。
- 下表对比了传统搜索与 UniDex 的核心差异。

<table><tr><td>对比维度</td><td>传统关键词搜索</td><td>快手的UniDex</td></tr><tr><td>核心单位</td><td>词汇（Term）</td><td>语义ID</td></tr><tr><td>理解能力</td><td>依赖字面匹配，无法理解同义词、近义词</td><td>深度语义理解，能跨越词汇鸿沟</td></tr><tr><td>系统链路</td><td>复杂，依赖多路召回、同义词扩展等大量人工规则</td><td>简洁，统一由模型处理，大幅简化</td></tr><tr><td>资源消耗</td><td>高（存储、计算）</td><td>显著降低（响应速度提升25%，节省大量CPU和内存）</td></tr><tr><td>长尾查询处理</td><td>弱，依赖现有词表</td><td>强，基于语义泛化，效果显著改善</td></tr></table>

![](images/98ff5338a35cefdf0e6a685b0c775a63c491474df357467f82505947b94f0c95.jpg)

# 一、核心架构：两大模块的密切协作

UniDex 的成功关键在于其内部两个精密协作的核心模块：UniTouch（负责召回）和 UniRank（负责排序）。

## 1.1 UniTouch：语义召回

- UniTouch 的任务是将用户的查询（Query）和视频文档（Doc）映射到同一个语义空间中。它通过一个共享的编码器，为 Query 和 Doc 生成一组稠密的语义向量，然后通过创新的有限标量化（FSQ）技术，将这些连续向量离散化成一个个具体的、整数形式的语义 ID。例如，一个关于"猫咪"的视频和查询"可爱的猫"，即使字面不同，也可能被赋予相同或相似的一组语义ID。
- 在检索时，UniTouch采用 "Max-Max"匹配策略：只要用户Query产生的语义ID集合与视频 Doc 的语义ID集合中有一个能匹配上，该视频就会被召回。这很好地应对了用户查询意图的多样性。

### 1.1.1 FSQ（Finite Scalar Quantization）代码实现

FSQ 将连续向量离散化为有限整数集合，是 UniDex 的核心创新之一：

```python
import torch
import torch.nn as nn
import numpy as np

class FSQ(nn.Module):
    def __init__(self, levels):
        """
        levels: 每个维度的离散化级别数
        例如 levels=[5, 5, 8, 8] 表示4个维度，分别有5、5、8、8个离散值
        """
        super().__init__()
        self.levels = nn.Parameter(torch.tensor(levels, dtype=torch.float32), requires_grad=False)
        self.dim = len(levels)
        self.codebook_size = int(np.prod(levels))

    def bound(self, z):
        half_levels = (self.levels - 1) / 2
        return z / (half_levels + 1e-8)

    def quantize(self, z):
        half_levels = (self.levels - 1) / 2
        z_norm = self.bound(z)
        z_norm = torch.tanh(z_norm)
        z_quant = torch.round(z_norm * half_levels) / half_levels
        return z_quant

    def get_codes(self, z):
        z_quant = self.quantize(z)
        half_levels = (self.levels - 1) / 2
        z_norm = torch.tanh(self.bound(z))
        indices = torch.round(z_norm * half_levels).long()
        indices = torch.clamp(indices, min=0)
        for i in range(self.dim):
            indices[:, i] = torch.clamp(indices[:, i], max=int(self.levels[i].item()) - 1)
        return indices

    def forward(self, z):
        z_quant = self.quantize(z)
        codes = self.get_codes(z)
        z_ste = z + (z_quant - z).detach()
        return z_ste, codes

class UniTouchEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, fsq_levels=None):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [5, 5, 8, 8, 5, 5]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(fsq_levels)),
        )
        self.fsq = FSQ(fsq_levels)

    def forward(self, x):
        z = self.encoder(x)
        z_quant, semantic_ids = self.fsq(z)
        return z_quant, semantic_ids

encoder = UniTouchEncoder(input_dim=768, fsq_levels=[5, 5, 8, 8, 5, 5])
sample_input = torch.randn(4, 768)
z_quant, semantic_ids = encoder(sample_input)
print(f"量化向量 shape: {z_quant.shape}")
print(f"语义 ID:\n{semantic_ids}")
print(f"码本大小: {encoder.fsq.codebook_size}")
```

## 1.2 UniRank：精排

- 在 UniTouch 完成初步筛选后，UniRank 负责对召回的结果进行更精细的语义重排。它与 UniTouch 共享同一套语义编码框架，保证了两个阶段语义理解的一致性。
- UniRank 的核心创新在于 Token 级别的细粒度交互。它会让 Query 的每一个语义 Token 都与视频的所有语义 Token 进行深度交互和匹配计算，最后综合得出一个更精确的相关性分数。这种方式比简单地计算整体向量的相似度能更好地捕捉复杂的语义关联。

### 1.2.1 Token 级别交互实现

```python
class UniRanker(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True, dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, query_tokens, doc_tokens):
        query_len = query_tokens.shape[1]
        combined = torch.cat([query_tokens, doc_tokens], dim=1)
        cross_attn_out = self.transformer(combined)
        query_repr = cross_attn_out[:, :query_len, :]
        score = self.score_head(query_repr).squeeze(-1)
        relevance_score = score.mean(dim=-1)
        return relevance_score

ranker = UniRanker(hidden_dim=64, n_heads=4, n_layers=2)
query_tok = torch.randn(2, 3, 64)
doc_tok = torch.randn(2, 8, 64)
scores = ranker(query_tok, doc_tok)
print(f"相关性分数: {scores}")
```

# 二、关键创新：语义离散化

UniDex 最根本的突破在于语义离散化思路。它通过 FSQ 技术，将深度学习模型输出的连续、不易直接索引的语义向量，转换成了离散的语义 ID。这种做法的优势在于：

- 兼容成熟生态：离散的语义 ID 可以直接接入工业界非常成熟、高效的倒排索引基础设施，享受其久经考验的性能和稳定性红利，避免了向量检索常面临的高成本和延迟问题。
- 可解释性：每个语义 ID 可以看作一个"语义格子"，为理解模型的匹配逻辑提供了一定线索。
- 灵活性：可以为简短、语义集中的 Query 分配较少的语义 ID（如 3 个），为内容丰富的视频分配更多的语义 ID（如 8 个），实现弹性的、与信息密度相匹配的语义表示。

## 2.1 倒排索引构建代码

```python
from collections import defaultdict
import numpy as np

class SemanticInvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_metadata = {}

    def add_document(self, doc_id, semantic_ids, metadata=None):
        self.doc_metadata[doc_id] = metadata or {}
        for token_id in semantic_ids:
            token_key = tuple(token_id.tolist()) if hasattr(token_id, 'tolist') else tuple(token_id)
            self.index[token_key].append(doc_id)

    def search(self, query_semantic_ids, top_k=None):
        doc_scores = defaultdict(int)
        for token_id in query_semantic_ids:
            token_key = tuple(token_id.tolist()) if hasattr(token_id, 'tolist') else tuple(token_id)
            for doc_id in self.index.get(token_key, []):
                doc_scores[doc_id] += 1
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked

index = SemanticInvertedIndex()
np.random.seed(42)
fsq_levels = [5, 5, 8, 8]
for doc_id in range(10000):
    n_tokens = np.random.randint(3, 9)
    semantic_ids = [np.random.randint(0, l, size=n_tokens) for l in fsq_levels]
    semantic_ids = np.array(semantic_ids).T
    metadata = {"title": f"视频_{doc_id}", "category": np.random.choice(["搞笑", "科技", "美食"])}
    index.add_document(doc_id, semantic_ids, metadata)

query_ids = np.array([[2, 3, 5, 6], [1, 2, 4, 5], [3, 4, 7, 3]])
results = index.search(query_ids, top_k=10)
print(f"查询结果 (Top 10):")
for doc_id, score in results:
    print(f"  doc_id={doc_id}, 匹配分数={score}, metadata={index.doc_metadata[doc_id]}")
print(f"\n索引统计:")
print(f"  文档数: {len(index.doc_metadata)}")
print(f"  语义token数: {len(index.index)}")
print(f"  平均每个token对应文档数: {np.mean([len(v) for v in index.index.values()]):.1f}")
```

# 三、与传统稠密检索对比

## 3.1 方法对比

| 维度 | 稠密检索（ANN） | 传统倒排索引 | UniDex |
|------|---------------|------------|--------|
| 索引结构 | 向量索引（HNSW/IVF） | 倒排索引（Term→Doc） | 倒排索引（语义ID→Doc） |
| 查询方式 | 向量近似最近邻 | 关键词精确匹配 | 语义ID精确匹配 |
| 语义理解 | 有（向量空间） | 无 | 有（语义空间离散化） |
| 索引更新 | 成本高（需重建） | 低（增量插入） | 低（增量插入） |
| 内存效率 | 高（压缩向量） | 高 | 高 |
| 推理延迟 | 中等（ANN搜索） | 低 | 低 |

## 3.2 性能对比实验代码

```python
import time
import numpy as np

class DenseRetriever:
    def __init__(self, dim=64):
        self.doc_embeddings = None
        self.dim = dim

    def build_index(self, n_docs=10000):
        self.doc_embeddings = np.random.randn(n_docs, self.dim)
        self.doc_embeddings /= np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)

    def search(self, query_emb, top_k=10):
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self.doc_embeddings @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, scores[top_indices]

class UniDexRetriever:
    def __init__(self, fsq_levels=None):
        self.index = defaultdict(list)
        self.fsq_levels = fsq_levels or [5, 5, 8, 8]
        self.doc_embeddings = None

    def build_index(self, n_docs=10000):
        self.doc_embeddings = np.random.randn(n_docs, 64)
        for doc_id in range(n_docs):
            n_tokens = np.random.randint(3, 9)
            for _ in range(n_tokens):
                token = tuple(np.random.randint(0, l) for l in self.fsq_levels)
                self.index[token].append(doc_id)

    def search(self, query_tokens, top_k=10):
        doc_scores = defaultdict(int)
        for token in query_tokens:
            token_key = tuple(token)
            for doc_id in self.index.get(token_key, []):
                doc_scores[doc_id] += 1
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [d[0] for d in ranked], [d[1] for d in ranked]

n_docs = 100000
dense = DenseRetriever(dim=64)
dense.build_index(n_docs)
start = time.time()
for _ in range(1000):
    query = np.random.randn(64)
    dense.search(query, top_k=10)
dense_time = (time.time() - start) / 1000

unidex = UniDexRetriever()
unidex.build_index(n_docs)
start = time.time()
for _ in range(1000):
    query_tokens = [np.random.randint(0, l) for l in unidex.fsq_levels]
    query_tokens = [query_tokens] * 3
    unidex.search(query_tokens, top_k=10)
unidex_time = (time.time() - start) / 1000

print(f"稠密检索平均耗时: {dense_time*1000:.3f}ms")
print(f"UniDex检索平均耗时: {unidex_time*1000:.3f}ms")
print(f"速度比: {dense_time/unidex_time:.2f}x")
```

# 四、实际效果

根据快手公开的实践数据，UniDex 在落地后取得了显著的效果：

- 指标提升：UniDex 在 RS 数据集上，Recall@300 较基线 Sparse 模型提升 $14.18\%$，MRR@10 提升 $10.02\%$。
- 效率优化：系统响应时间降低了 $25\%$，同时节省 2 万 CPU-Core 和 37TB 内存使用，实现了效果与效率的双赢。

<table><tr><td rowspan="2">UniDex</td><td>Sat.</td><td>CTR ↑ +0.185%</td><td>VPD ↑ +0.287%</td><td>LPC ↑ +0.352%</td><td>MRS ↑ +0.346%</td></tr><tr><td>Cost</td><td>Core ↓ -20550</td><td>Memory ↓ -37TB</td><td>Latency ↓ -25%</td><td></td></tr></table>

## 4.1 评估指标说明

| 指标 | 含义 | 说明 |
|------|------|------|
| Recall@K | 前K个结果中相关文档的召回率 | 衡量召回能力 |
| MRR@K | 前K个结果中第一个相关文档排名倒数的均值 | 衡量排序质量 |
| CTR | 点击率 | 用户行为指标 |
| VPD | 视频播放深度 | 用户观看深度 |
| LPC | 长尾查询覆盖率 | 长尾场景下的改善 |

# 五、部署注意事项

## 5.1 工程部署要点

```python
class UniDexPipeline:
    def __init__(self, encoder, fsq, index, ranker=None):
        self.encoder = encoder
        self.fsq = fsq
        self.index = index
        self.ranker = ranker

    def index_document(self, doc_id, doc_features):
        with torch.no_grad():
            z = self.encoder(doc_features.unsqueeze(0))
            z_quant, semantic_ids = self.fsq(z)
        self.index.add_document(doc_id, semantic_ids[0])
        return semantic_ids[0]

    def search(self, query_features, top_k_recall=100, top_k_final=10):
        with torch.no_grad():
            z = self.encoder(query_features.unsqueeze(0))
            z_quant, semantic_ids = self.fsq(z)
        candidates = self.index.search(semantic_ids[0], top_k=top_k_recall)
        if self.ranker is not None and len(candidates) > 0:
            doc_ids = [c[0] for c in candidates]
            scores = [c[1] for c in candidates]
            return list(zip(doc_ids[:top_k_final], scores[:top_k_final]))
        return candidates[:top_k_final]

    def batch_index(self, doc_ids, doc_features_batch):
        for doc_id, features in zip(doc_ids, doc_features_batch):
            self.index_document(doc_id, features)
        print(f"批量索引完成: {len(doc_ids)} 个文档")
```

## 5.2 关键部署考虑

| 考虑因素 | 建议方案 |
|---------|---------|
| FSQ 码本设计 | 根据语料规模选择级别数，推荐总码本 $\geq$ 文档数 / 100 |
| 语义ID数量 | Query 3-5 个，Doc 5-10 个 |
| 索引更新策略 | 增量更新，定期全量重建 |
| 模型更新 | 共享编码器热更新，注意语义ID兼容性 |
| 在线延迟 | 倒排检索 < 5ms，精排 < 20ms |

# 六、常见问题与易错点

1. **FSQ 级别数设置不当**：级别数过少导致码本太小，大量文档映射到同一语义ID，区分度差。级别数过多导致码本过大，稀疏匹配效率低。
2. **忽略语义ID的稳定性**：模型更新后语义编码可能变化，导致索引失效。需要版本管理和兼容策略。
3. **Query 和 Doc 的 Token 数量不对称**：Query 通常语义集中（3-5个），Doc 语义丰富（5-10个），需要灵活处理。
4. **长尾查询处理**：虽然 UniDex 对长尾有优势，但仍需关注极低频查询的兜底策略。

# 七、学习路径建议

1. 理解倒排索引和向量检索的基本原理
2. 学习向量量化技术：PQ、OPQ、FSQ、RVQ
3. 阅读论文：UniDex（快手 2025）、Semantic Token相关研究
4. 实践：搭建一个基于语义ID的小型检索系统
5. 关注前沿：大模型 + 检索、多模态语义索引
