# 完整推荐系统Demo 学习文档

## 1. 项目概述

### 1.1 系统架构

```
完整推荐系统架构:

用户请求 → [网关] → [召回服务] → [排序服务] → [重排服务] → 返回结果
              │           │            │            │
              ↓           ↓            ↓            ↓
          [特征服务]  [向量检索]   [模型推理]   [多样性策略]
              │           │            │            │
              ↓           ↓            ↓            ↓
           [Redis]    [Faiss]      [ONNX]       [规则引擎]
```

### 1.2 项目结构

```python
"""
推荐系统Demo项目结构:

recsys_demo/
├── config/
│   ├── recall.yaml
│   ├── ranking.yaml
│   └── rerank.yaml
├── data/
│   ├── items.csv
│   ├── users.csv
│   └── interactions.csv
├── models/
│   ├── recall/
│   ├── ranking/
│   └── rerank/
├── services/
│   ├── recall_service.py
│   ├── ranking_service.py
│   ├── rerank_service.py
│   └── feature_service.py
├── api/
│   ├── gateway.py
│   └── recommend_api.py
├── utils/
│   ├── feature_utils.py
│   └── metrics.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── README.md
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time
from collections import defaultdict
```

## 2. 召回模块

### 2.1 多路召回

```python
@dataclass
class RecallConfig:
    """召回配置"""
    user_cf_topk: int = 200
    item_cf_topk: int = 200
    swing_topk: int = 200
    hot_topk: int = 100
    embed_topk: int = 200
    total_topk: int = 500


class UserCFRecall:
    """
    UserCF 召回
    """

    def __init__(self, user_item_matrix: Dict[int, List[int]],
                 item_users: Dict[int, List[int]]):
        self.user_item_matrix = user_item_matrix
        self.item_users = item_users
        self.user_similarity = {}

    def compute_similarity(self, user_id: int, top_k: int = 100):
        """计算用户相似度"""
        if user_id not in self.user_item_matrix:
            return {}

        user_items = set(self.user_item_matrix[user_id])

        # 找到有共同物品的用户
        candidate_users = set()
        for item in user_items:
            candidate_users.update(self.item_users.get(item, []))

        # 计算相似度
        similarity = {}
        for candidate in candidate_users:
            if candidate == user_id:
                continue

            candidate_items = set(self.user_item_matrix.get(candidate, []))
            intersection = len(user_items & candidate_items)
            union = len(user_items | candidate_items)

            if union > 0:
                similarity[candidate] = intersection / union

        # Top-K
        sorted_sim = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_sim[:top_k])

    def recall(self, user_id: int, top_k: int = 200) -> List[Tuple[int, float]]:
        """召回"""
        if user_id not in self.user_similarity:
            self.user_similarity[user_id] = self.compute_similarity(user_id)

        sim_users = self.user_similarity[user_id]

        # 用户已交互物品
        user_items = set(self.user_item_matrix.get(user_id, []))

        # 聚合相似用户的物品
        item_scores = defaultdict(float)
        for sim_user, sim in sim_users.items():
            for item in self.user_item_matrix.get(sim_user, []):
                if item not in user_items:
                    item_scores[item] += sim

        # 排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


class ItemCFRecall:
    """
    ItemCF 召回
    """

    def __init__(self, item_similarity: Dict[int, Dict[int, float]]):
        self.item_similarity = item_similarity

    def recall(self, user_history: List[int], top_k: int = 200
              ) -> List[Tuple[int, float]]:
        """召回"""
        item_scores = defaultdict(float)
        history_set = set(user_history)

        for item in user_history:
            if item not in self.item_similarity:
                continue

            for sim_item, sim in self.item_similarity[item].items():
                if sim_item not in history_set:
                    item_scores[sim_item] += sim

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


class EmbeddingRecall:
    """
    向量召回
    """

    def __init__(self, item_embeddings: np.ndarray,
                 item_ids: List[int]):
        self.item_embeddings = item_embeddings
        self.item_ids = item_ids
        self.index = None

    def build_index(self, n_clusters: int = 100):
        """构建索引"""
        import faiss

        d = self.item_embeddings.shape[1]
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(d), d, n_clusters
        )

        # 训练
        self.index.train(self.item_embeddings)
        self.index.add(self.item_embeddings)

    def recall(self, user_embedding: np.ndarray, top_k: int = 200
              ) -> List[Tuple[int, float]]:
        """召回"""
        if self.index is None:
            self.build_index()

        # 搜索
        query = user_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.item_ids):
                results.append((self.item_ids[idx], float(dist)))

        return results


class HotRecall:
    """
    热门召回
    """

    def __init__(self, item_popularity: Dict[int, float]):
        self.hot_items = sorted(
            item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def recall(self, exclude_items: set, top_k: int = 100
              ) -> List[Tuple[int, float]]:
        """召回"""
        results = [
            (item, score) for item, score in self.hot_items
            if item not in exclude_items
        ]
        return results[:top_k]


class MultiChannelRecall:
    """
    多路召回融合
    """

    def __init__(self, config: RecallConfig):
        self.config = config

        # 各路召回器
        self.user_cf = None
        self.item_cf = None
        self.swing = None
        self.embed = None
        self.hot = None

    def recall(self, user_id: int, user_history: List[int],
              user_embedding: np.ndarray = None) -> List[Tuple[int, float]]:
        """
        多路召回

        返回融合后的候选物品
        """
        all_candidates = defaultdict(float)
        history_set = set(user_history)

        # 1. UserCF
        if self.user_cf:
            user_cf_results = self.user_cf.recall(user_id, self.config.user_cf_topk)
            for item, score in user_cf_results:
                if item not in history_set:
                    all_candidates[item] += score * 1.0

        # 2. ItemCF
        if self.item_cf:
            item_cf_results = self.item_cf.recall(user_history, self.config.item_cf_topk)
            for item, score in item_cf_results:
                if item not in history_set:
                    all_candidates[item] += score * 1.2

        # 3. Swing
        if self.swing:
            swing_results = self.swing.recall(user_history, self.config.swing_topk)
            for item, score in swing_results:
                if item not in history_set:
                    all_candidates[item] += score * 1.5

        # 4. 向量召回
        if self.embed and user_embedding is not None:
            embed_results = self.embed.recall(user_embedding, self.config.embed_topk)
            for item, score in embed_results:
                if item not in history_set:
                    all_candidates[item] += score * 1.0

        # 5. 热门召回 (补充)
        if self.hot:
            hot_results = self.hot.recall(history_set, self.config.hot_topk)
            for item, score in hot_results:
                all_candidates[item] += score * 0.5

        # 排序返回
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_candidates[:self.config.total_topk]
```

## 3. 排序模块

### 3.1 排序模型

```python
class RankingModel:
    """
    排序模型

    预测用户对物品的点击概率
    """

    def __init__(self, model_path: str):
        # 加载模型 (简化示例)
        self.model = None
        self.feature_names = []
        self.load_model(model_path)

    def load_model(self, path: str):
        """加载模型"""
        # 实际实现中加载 ONNX 或 PyTorch 模型
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        # 简化: 返回随机分数
        return np.random.random(len(features))


class FeatureExtractor:
    """
    特征提取器
    """

    def __init__(self):
        self.user_features = {}    # 用户特征缓存
        self.item_features = {}    # 物品特征缓存

    def extract_features(self,
                        user_id: int,
                        item_ids: List[int],
                        context: Dict) -> np.ndarray:
        """
        提取特征

        返回: (n_items, n_features) 特征矩阵
        """
        n_items = len(item_ids)
        features = []

        # 用户特征
        user_feat = self.user_features.get(user_id, {})

        for item_id in item_ids:
            # 物品特征
            item_feat = self.item_features.get(item_id, {})

            # 组合特征
            feat = []

            # 用户特征
            feat.append(user_feat.get('click_count_7d', 0))
            feat.append(user_feat.get('avg_price', 0))

            # 物品特征
            feat.append(item_feat.get('popularity', 0))
            feat.append(item_feat.get('price', 0))
            feat.append(item_feat.get('ctr', 0))

            # 交叉特征
            feat.append(user_feat.get('preferred_category') == item_feat.get('category'))

            # 上下文特征
            feat.append(context.get('hour', 12))
            feat.append(context.get('day_of_week', 0))

            features.append(feat)

        return np.array(features)


class RankingService:
    """
    排序服务
    """

    def __init__(self, model_path: str):
        self.model = RankingModel(model_path)
        self.feature_extractor = FeatureExtractor()

    def rank(self,
            user_id: int,
            candidates: List[Tuple[int, float]],
            context: Dict,
            top_k: int = 100) -> List[Tuple[int, float]]:
        """
        排序

        candidates: 召回候选 [(item_id, recall_score), ...]
        返回: [(item_id, ranking_score), ...]
        """
        if not candidates:
            return []

        item_ids = [item for item, _ in candidates]

        # 提取特征
        features = self.feature_extractor.extract_features(user_id, item_ids, context)

        # 预测
        scores = self.model.predict(features)

        # 组合结果
        results = list(zip(item_ids, scores))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
```

## 4. 重排模块

### 4.1 多样性重排

```python
class DiversityReranker:
    """
    多样性重排

    增加推荐结果的多样性
    """

    def __init__(self, item_categories: Dict[int, int],
                 diversity_weight: float = 0.3):
        self.item_categories = item_categories
        self.diversity_weight = diversity_weight

    def rerank(self,
              ranked_list: List[Tuple[int, float]],
              top_k: int = 50,
              max_same_category: int = 3) -> List[Tuple[int, float]]:
        """
        多样性重排

        限制同类目物品数量
        """
        result = []
        category_count = defaultdict(int)

        for item, score in ranked_list:
            category = self.item_categories.get(item, 0)

            # 检查类目限制
            if category_count[category] >= max_same_category:
                continue

            result.append((item, score))
            category_count[category] += 1

            if len(result) >= top_k:
                break

        return result


class MMRDiversity:
    """
    MMR (Maximal Marginal Relevance) 多样性

    平衡相关性和多样性
    """

    def __init__(self, item_embeddings: np.ndarray,
                 item_ids: List[int],
                 lambda_param: float = 0.5):
        self.item_embeddings = item_embeddings
        self.item_id_to_idx = {id_: i for i, id_ in enumerate(item_ids)}
        self.lambda_param = lambda_param

    def rerank(self,
              ranked_list: List[Tuple[int, float]],
              top_k: int = 50) -> List[Tuple[int, float]]:
        """
        MMR 重排
        """
        if not ranked_list:
            return []

        result = []
        remaining = list(ranked_list)
        selected_embeddings = []

        while len(result) < top_k and remaining:
            best_item = None
            best_score = float('-inf')

            for item, relevance in remaining:
                if item not in self.item_id_to_idx:
                    continue

                item_idx = self.item_id_to_idx[item]
                item_embed = self.item_embeddings[item_idx]

                # 相关性
                relevance_score = relevance

                # 多样性 (与已选物品的最大相似度)
                if selected_embeddings:
                    similarities = [
                        np.dot(item_embed, sel_embed) /
                        (np.linalg.norm(item_embed) * np.linalg.norm(sel_embed) + 1e-10)
                        for sel_embed in selected_embeddings
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0

                # MMR 分数
                mmr_score = (
                    self.lambda_param * relevance_score -
                    (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (item, relevance)

            if best_item:
                result.append(best_item)
                remaining.remove(best_item)

                item_idx = self.item_id_to_idx[best_item[0]]
                selected_embeddings.append(self.item_embeddings[item_idx])

        return result


class PositionReranker:
    """
    位置重排

    考虑位置偏差
    """

    def __init__(self, position_weights: List[float] = None):
        if position_weights is None:
            # 默认位置权重 (前面的位置更重要)
            self.position_weights = [1.0 / (i + 1) ** 0.5 for i in range(100)]
        else:
            self.position_weights = position_weights

    def rerank(self,
              ranked_list: List[Tuple[int, float]],
              top_k: int = 50) -> List[Tuple[int, float]]:
        """
        位置加权重排
        """
        weighted_scores = []

        for i, (item, score) in enumerate(ranked_list[:top_k]):
            position_weight = self.position_weights[i] if i < len(self.position_weights) else 0.1
            weighted_score = score * position_weight
            weighted_scores.append((item, weighted_score, score))

        # 按加权分数排序
        weighted_scores.sort(key=lambda x: x[1], reverse=True)

        return [(item, original_score) for item, _, original_score in weighted_scores]


class RerankService:
    """
    重排服务
    """

    def __init__(self, config: Dict):
        self.diversity_reranker = None
        self.mmr_reranker = None
        self.position_reranker = PositionReranker()

        self.enable_diversity = config.get('enable_diversity', True)
        self.enable_mmr = config.get('enable_mmr', False)
        self.top_k = config.get('top_k', 50)

    def set_diversity_reranker(self, reranker: DiversityReranker):
        self.diversity_reranker = reranker

    def set_mmr_reranker(self, reranker: MMRDiversity):
        self.mmr_reranker = reranker

    def rerank(self,
              ranked_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        执行重排
        """
        result = ranked_list

        # 1. 多样性重排
        if self.enable_diversity and self.diversity_reranker:
            result = self.diversity_reranker.rerank(result, self.top_k)

        # 2. MMR 重排
        if self.enable_mmr and self.mmr_reranker:
            result = self.mmr_reranker.rerank(result, self.top_k)

        return result[:self.top_k]
```

## 5. 完整推荐流程

### 5.1 推荐引擎

```python
@dataclass
class RecommendRequest:
    """推荐请求"""
    user_id: int
    context: Dict
    top_k: int = 50


@dataclass
class RecommendResponse:
    """推荐响应"""
    user_id: int
    items: List[int]
    scores: List[float]
    recall_latency_ms: float
    ranking_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float


class RecommendationEngine:
    """
    推荐引擎

    整合召回、排序、重排
    """

    def __init__(self,
                 recall_service: MultiChannelRecall,
                 ranking_service: RankingService,
                 rerank_service: RerankService):
        self.recall = recall_service
        self.ranking = ranking_service
        self.rerank = rerank_service

        # 用户数据
        self.user_history = {}
        self.user_embeddings = {}

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """
        执行推荐
        """
        start_time = time.time()

        user_id = request.user_id
        context = request.context
        top_k = request.top_k

        # 获取用户数据
        user_history = self.user_history.get(user_id, [])
        user_embedding = self.user_embeddings.get(user_id)

        # 1. 召回
        recall_start = time.time()
        candidates = self.recall.recall(user_id, user_history, user_embedding)
        recall_latency = (time.time() - recall_start) * 1000

        # 2. 排序
        ranking_start = time.time()
        ranked = self.ranking.rank(user_id, candidates, context, top_k * 2)
        ranking_latency = (time.time() - ranking_start) * 1000

        # 3. 重排
        rerank_start = time.time()
        final = self.rerank.rerank(ranked)
        rerank_latency = (time.time() - rerank_start) * 1000

        total_latency = (time.time() - start_time) * 1000

        # 返回结果
        items = [item for item, _ in final[:top_k]]
        scores = [score for _, score in final[:top_k]]

        return RecommendResponse(
            user_id=user_id,
            items=items,
            scores=scores,
            recall_latency_ms=recall_latency,
            ranking_latency_ms=ranking_latency,
            rerank_latency_ms=rerank_latency,
            total_latency_ms=total_latency
        )

    def update_user_data(self, user_id: int, history: List[int],
                        embedding: np.ndarray = None):
        """更新用户数据"""
        self.user_history[user_id] = history
        if embedding is not None:
            self.user_embeddings[user_id] = embedding
```

### 5.2 API 网关

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class RecommendAPI:
    """
    推荐 API
    """

    def __init__(self, engine: RecommendationEngine):
        self.engine = engine
        self.app = FastAPI(title="Recommendation API")

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/recommend")
        async def recommend(request: RecommendRequest):
            response = self.engine.recommend(request)
            return {
                "user_id": response.user_id,
                "items": response.items,
                "scores": response.scores,
                "latency": {
                    "recall_ms": response.recall_latency_ms,
                    "ranking_ms": response.ranking_latency_ms,
                    "rerank_ms": response.rerank_latency_ms,
                    "total_ms": response.total_latency_ms
                }
            }

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)


def create_demo_engine():
    """创建演示引擎"""
    # 配置
    recall_config = RecallConfig()

    # 召回
    recall_service = MultiChannelRecall(recall_config)

    # 排序
    ranking_service = RankingService("models/ranking/model.onnx")

    # 重排
    rerank_service = RerankService({
        'enable_diversity': True,
        'top_k': 50
    })

    # 引擎
    engine = RecommendationEngine(
        recall_service,
        ranking_service,
        rerank_service
    )

    return engine


if __name__ == "__main__":
    engine = create_demo_engine()
    api = RecommendAPI(engine)
    api.run()
```

## 6. 学习总结

### 6.1 系统架构要点

```
1. 召回: 多路召回、向量检索
2. 排序: 精排模型、特征工程
3. 重排: 多样性、位置偏差
4. 服务: 高可用、低延迟
```

### 6.2 性能优化

```
1. 召回: Faiss 加速向量检索
2. 排序: ONNX 模型推理
3. 缓存: Redis 缓存热点数据
4. 并行: 异步处理多路召回
```

### 6.3 监控指标

```
1. 延迟: P50/P99 推理延迟
2. 召回量: 各路召回数量
3. 穿透率: 缓存命中率
4. 业务指标: CTR、CVR
```
