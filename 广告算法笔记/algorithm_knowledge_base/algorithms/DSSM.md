# DSSM（Deep Structured Semantic Models）学习文档

## 1. 算法基础认知

DSSM（Deep Structured Semantic Models，深度结构化语义模型）是 Microsoft 2013 年提出的双塔模型，又称 Two-Tower 模型。它将用户和物品分别通过独立的深度网络编码为向量，再通过向量内积计算相似度。

## 2. 核心原理

### 模型结构

```
User Features → User Tower → u (向量)
                                  ↘
                                    cos(u, v) → 相似度
                                  ↗
Item Features → Item Tower → v (向量)
```

### 相似度计算

$$
\text{similarity}(u, v) = \cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}
$$

或使用点积：

$$
\hat{y} = \sigma(\mathbf{u}^T \mathbf{v})
$$

### 关键特性

- 用户塔和物品塔完全独立，可以离线预计算物品向量
- 适合大规模召回场景，延迟极低
- 无法建模用户-物品交叉特征

## 3. 应用场景

- 广告召回阶段（向量召回）
- 推荐系统召回
- 语义相似度计算
- 多模态广告召回（多模态双塔 + FAISS/Milvus）

## 4. 代码实现

```python
import torch
import torch.nn as nn

class DSSM(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, user_features, item_features):
        u = self.user_tower(user_features)
        v = self.item_tower(item_features)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.sum(u * v, dim=-1, keepdim=True)
```

## 5. 学习总结

DSSM（双塔模型）是广告召回阶段的基础架构，通过分离用户和物品编码实现高效检索。后续的 COLD、FSCD 等模型都在双塔基础上增加轻量级交叉。
