# DSSM双塔召回 学习文档

## 1. DSSM概述

### 1.1 什么是DSSM？

```
DSSM (Deep Structured Semantic Models):

- 微软提出的深度语义匹配模型
- 最初用于搜索排序
- 后广泛应用于推荐召回

核心思想:
- 将 Query 和 Document 映射到同一语义空间
- 使用余弦相似度计算相关性
- 双塔结构：用户塔 + 物品塔

在召回中的应用:
- 用户-物品匹配
- Query-Item 匹配
- 向量召回的基础架构
```

### 1.2 模型架构

```python
"""
DSSM 架构:

        用户特征                  物品特征
           │                        │
        [嵌入层]                 [嵌入层]
           │                        │
        [MLP层]                  [MLP层]
           │                        │
        用户向量                  物品向量
           │                        │
           └────────┬───────────────┘
                    │
              [余弦相似度]
                    │
                预测分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class DSSM(nn.Module):
    """
    DSSM 双塔模型

    基础版本
    """

    def __init__(self,
                 user_feature_dims: Dict[str, int],
                 item_feature_dims: Dict[str, int],
                 embed_dim: int = 64,
                 hidden_dims: List[int] = [128, 64]):
        """
        参数:
            user_feature_dims: 用户各特征的维度
            item_feature_dims: 物品各特征的维度
            embed_dim: 嵌入维度
            hidden_dims: MLP 隐藏层维度
        """
        super().__init__()

        self.embed_dim = embed_dim

        # 用户塔
        self.user_tower = UserTower(
            feature_dims=user_feature_dims,
            hidden_dims=hidden_dims,
            output_dim=embed_dim
        )

        # 物品塔
        self.item_tower = ItemTower(
            feature_dims=item_feature_dims,
            hidden_dims=hidden_dims,
            output_dim=embed_dim
        )

    def forward(self, user_features: Dict[str, torch.Tensor],
                item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        返回: (batch,) 相似度分数
        """
        # 用户向量
        user_embed = self.user_tower(user_features)  # (batch, embed_dim)

        # 物品向量
        item_embed = self.item_tower(item_features)  # (batch, embed_dim)

        # 余弦相似度
        similarity = F.cosine_similarity(user_embed, item_embed, dim=-1)

        return similarity

    def get_user_embedding(self, user_features: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        """获取用户嵌入"""
        return self.user_tower(user_features)

    def get_item_embedding(self, item_features: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        """获取物品嵌入"""
        return self.item_tower(item_features)


class UserTower(nn.Module):
    """
    用户塔

    将用户特征编码为向量
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dims: List[int],
                 output_dim: int):
        super().__init__()

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, output_dim)
            for name, dim in feature_dims.items()
        })

        # MLP
        input_dim = len(feature_dims) * output_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        embeds = []
        for name, values in features.items():
            embed = self.embeddings[name](values)  # (batch, output_dim)
            embeds.append(embed)

        # 拼接
        concat = torch.cat(embeds, dim=-1)  # (batch, n_features * output_dim)

        # MLP
        output = self.mlp(concat)  # (batch, output_dim)

        # L2 归一化
        output = F.normalize(output, p=2, dim=-1)

        return output


class ItemTower(nn.Module):
    """
    物品塔

    将物品特征编码为向量
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dims: List[int],
                 output_dim: int):
        super().__init__()

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, output_dim)
            for name, dim in feature_dims.items()
        })

        # MLP
        input_dim = len(feature_dims) * output_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        embeds = []
        for name, values in features.items():
            embed = self.embeddings[name](values)
            embeds.append(embed)

        # 拼接
        concat = torch.cat(embeds, dim=-1)

        # MLP
        output = self.mlp(concat)

        # L2 归一化
        output = F.normalize(output, p=2, dim=-1)

        return output
```

## 2. 训练方法

### 2.1 负采样训练

```python
class DSSMTrainer:
    """
    DSSM 训练器
    """

    def __init__(self,
                 model: DSSM,
                 n_negatives: int = 4,
                 temperature: float = 0.1,
                 lr: float = 0.001):
        """
        参数:
            n_negatives: 每个正样本的负样本数
            temperature: 温度参数
            lr: 学习率
        """
        self.model = model
        self.n_negatives = n_negatives
        self.temperature = temperature

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self,
                   user_features: Dict[str, torch.Tensor],
                   pos_item_features: Dict[str, torch.Tensor],
                   neg_item_features: Dict[str, torch.Tensor]) -> float:
        """
        训练一步

        user_features: 用户特征
        pos_item_features: 正样本物品特征
        neg_item_features: 负样本物品特征 (batch, n_neg, ...)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 用户嵌入
        user_embed = self.model.get_user_embedding(user_features)  # (batch, embed_dim)

        # 正样本嵌入
        pos_item_embed = self.model.get_item_embedding(pos_item_features)  # (batch, embed_dim)

        # 负样本嵌入
        batch_size = user_features[list(user_features.keys())[0]].size(0)

        # 重塑负样本特征
        neg_embeds = []
        for i in range(self.n_negatives):
            neg_feat_i = {
                name: neg_item_features[name][:, i, :]
                if len(neg_item_features[name].shape) > 2
                else neg_item_features[name][:, i]
                for name in neg_item_features
            }
            neg_embed = self.model.get_item_embedding(neg_feat_i)
            neg_embeds.append(neg_embed)

        neg_embeds = torch.stack(neg_embeds, dim=1)  # (batch, n_neg, embed_dim)

        # 计算分数
        pos_score = (user_embed * pos_item_embed).sum(dim=-1) / self.temperature  # (batch,)

        neg_scores = torch.matmul(
            user_embed.unsqueeze(1),  # (batch, 1, embed_dim)
            neg_embeds.transpose(1, 2)  # (batch, embed_dim, n_neg)
        ).squeeze(1) / self.temperature  # (batch, n_neg)

        # Softmax 损失 (正样本应该得分最高)
        all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # (batch, 1+n_neg)
        labels = torch.zeros(batch_size, dtype=torch.long, device=all_scores.device)

        loss = F.cross_entropy(all_scores, labels)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()


class InBatchNegativesTrainer:
    """
    批内负采样训练器

    更高效，使用同批次内其他样本作为负样本
    """

    def __init__(self,
                 model: DSSM,
                 temperature: float = 0.05,
                 lr: float = 0.001):
        self.model = model
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self,
                   user_features: Dict[str, torch.Tensor],
                   item_features: Dict[str, torch.Tensor]) -> float:
        """
        批内负采样训练

        所有正样本对的物品同时作为其他样本的负样本
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_size = user_features[list(user_features.keys())[0]].size(0)

        # 获取嵌入
        user_embed = self.model.get_user_embedding(user_features)  # (batch, embed_dim)
        item_embed = self.model.get_item_embedding(item_features)  # (batch, embed_dim)

        # 计算相似度矩阵
        # (batch, embed_dim) @ (embed_dim, batch) = (batch, batch)
        similarity_matrix = torch.matmul(user_embed, item_embed.T) / self.temperature

        # 对角线是正样本
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        # 交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## 3. DSSM 变体

### 3.1 Multi-View DSSM

```python
class MultiViewDSSM(nn.Module):
    """
    多视图 DSSM

    用户和物品可以有多个视图/模态
    """

    def __init__(self,
                 user_feature_configs: Dict[str, Dict],
                 item_feature_configs: Dict[str, Dict],
                 embed_dim: int = 64,
                 hidden_dims: List[int] = [128, 64]):
        """
        参数:
            user_feature_configs: {
                'sparse': {'name': dim, ...},
                'dense': ['feature1', ...]
            }
        """
        super().__init__()

        self.embed_dim = embed_dim

        # 用户多视图塔
        self.user_towers = nn.ModuleDict()
        for view_name, config in user_feature_configs.items():
            self.user_towers[view_name] = self._build_tower(
                config, hidden_dims, embed_dim
            )

        # 物品多视图塔
        self.item_towers = nn.ModuleDict()
        for view_name, config in item_feature_configs.items():
            self.item_towers[view_name] = self._build_tower(
                config, hidden_dims, embed_dim
            )

        # 视图融合
        self.user_fusion = nn.Linear(
            len(user_feature_configs) * embed_dim,
            embed_dim
        )
        self.item_fusion = nn.Linear(
            len(item_feature_configs) * embed_dim,
            embed_dim
        )

    def _build_tower(self, config: Dict, hidden_dims: List[int],
                     output_dim: int) -> nn.Module:
        """构建单塔"""
        # 简化实现
        return nn.Sequential(
            nn.Linear(64, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim)
        )

    def forward(self, user_features: Dict[str, Dict],
                item_features: Dict[str, Dict]) -> torch.Tensor:
        """前向传播"""
        # 用户各视图嵌入
        user_view_embeds = []
        for view_name, features in user_features.items():
            view_embed = self.user_towers[view_name](features)
            user_view_embeds.append(view_embed)

        # 融合
        user_embed = self.user_fusion(torch.cat(user_view_embeds, dim=-1))
        user_embed = F.normalize(user_embed, p=2, dim=-1)

        # 物品各视图嵌入
        item_view_embeds = []
        for view_name, features in item_features.items():
            view_embed = self.item_towers[view_name](features)
            item_view_embeds.append(view_embed)

        item_embed = self.item_fusion(torch.cat(item_view_embeds, dim=-1))
        item_embed = F.normalize(item_embed, p=2, dim=-1)

        # 相似度
        return F.cosine_similarity(user_embed, item_embed, dim=-1)
```

### 3.2 YouTube DNN

```python
class YouTubeDNN(nn.Module):
    """
    YouTube DNN 召回模型

    DSSM 在视频推荐中的应用
    """

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 embed_dim: int = 256,
                 hidden_dims: List[int] = [1024, 512, 256]):
        super().__init__()

        # 用户嵌入
        self.user_embedding = nn.Embedding(n_users, embed_dim)

        # 用户历史序列
        self.history_embedding = nn.Embedding(n_items, embed_dim)

        # 用户塔
        user_input_dim = embed_dim * 2  # 用户ID + 历史平均

        layers = []
        input_dim = user_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        self.user_tower = nn.Sequential(*layers)

        # 物品塔 (简化为嵌入)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

    def encode_user(self, user_id: torch.Tensor,
                   history_ids: torch.Tensor,
                   history_mask: torch.Tensor = None) -> torch.Tensor:
        """编码用户"""
        # 用户 ID 嵌入
        user_embed = self.user_embedding(user_id)

        # 历史物品平均
        history_embeds = self.history_embedding(history_ids)

        if history_mask is not None:
            history_mask = history_mask.unsqueeze(-1)
            history_embeds = history_embeds * history_mask
            history_avg = history_embeds.sum(dim=1) / (history_mask.sum(dim=1) + 1e-10)
        else:
            history_avg = history_embeds.mean(dim=1)

        # 拼接
        user_input = torch.cat([user_embed, history_avg], dim=-1)

        # 用户塔
        user_vec = self.user_tower(user_input)
        user_vec = F.normalize(user_vec, p=2, dim=-1)

        return user_vec

    def forward(self, user_id: torch.Tensor,
               history_ids: torch.Tensor,
               item_id: torch.Tensor,
               history_mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        # 用户向量
        user_vec = self.encode_user(user_id, history_ids, history_mask)

        # 物品向量
        item_vec = self.item_embedding(item_id)
        item_vec = F.normalize(item_vec, p=2, dim=-1)

        # 点积
        score = (user_vec * item_vec).sum(dim=-1)

        return score
```

## 4. 向量召回部署

### 4.1 向量索引构建

```python
class DSSMVectorIndex:
    """
    DSSM 向量索引

    用于在线召回
    """

    def __init__(self, model: DSSM, embed_dim: int = 64):
        self.model = model
        self.embed_dim = embed_dim

        # 物品向量
        self.item_vectors = None
        self.item_ids = None

        # Faiss 索引
        self.index = None

    def build_index(self,
                   item_features: Dict[str, torch.Tensor],
                   item_ids: List[int],
                   n_clusters: int = 100):
        """
        构建向量索引
        """
        import faiss

        # 计算物品向量
        self.model.eval()
        with torch.no_grad():
            self.item_vectors = self.model.get_item_embedding(item_features)
            self.item_vectors = self.item_vectors.cpu().numpy()

        self.item_ids = np.array(item_ids)
        n_items = len(item_ids)

        # 构建 Faiss 索引
        quantizer = faiss.IndexFlatL2(self.embed_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.embed_dim, n_clusters
        )

        # 训练
        self.index.train(self.item_vectors)

        # 添加
        self.index.add(self.item_vectors)

        print(f"Built index with {n_items} items")

    def search(self, user_features: Dict[str, torch.Tensor],
              top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索相似物品
        """
        # 计算用户向量
        self.model.eval()
        with torch.no_grad():
            user_vector = self.model.get_user_embedding(user_features)
            user_vector = user_vector.cpu().numpy()

        # 搜索
        distances, indices = self.index.search(user_vector, top_k)

        # 转换为物品ID
        item_ids = self.item_ids[indices]

        return item_ids, distances

    def save(self, path: str):
        """保存索引"""
        import faiss

        faiss.write_index(self.index, f"{path}.index")
        np.save(f"{path}_ids.npy", self.item_ids)

    def load(self, path: str):
        """加载索引"""
        import faiss

        self.index = faiss.read_index(f"{path}.index")
        self.item_ids = np.load(f"{path}_ids.npy")
```

## 5. 学习总结

### 5.1 核心要点

```
1. 双塔结构: 用户塔和物品塔独立
2. 向量归一化: 使用 L2 归一化
3. 相似度计算: 余弦相似度或点积
4. 负采样: 批内负采样更高效
5. 温度参数: 控制分布的平滑程度
```

### 5.2 优势与劣势

```
优势:
- 计算高效 (物品向量可预计算)
- 适合大规模召回
- 灵活的特征组合

劣势:
- 无法使用交叉特征
- 信息在融合时可能丢失
- 需要大量负样本
```

### 5.3 最佳实践

```
1. 归一化: 所有输出向量都 L2 归一化
2. 温度: 0.01-0.1 之间
3. 负样本: 批内负采样 + 随机负采样
4. 嵌入维度: 64-256
5. 离线预计算: 物品向量提前计算
```
