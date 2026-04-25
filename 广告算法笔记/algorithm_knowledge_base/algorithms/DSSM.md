# DSSM（Deep Structured Semantic Models）学习文档

## 1. 算法基础认知

DSSM（Deep Structured Semantic Models，深度结构化语义模型）是 Microsoft 2013 年提出的双塔模型，又称 Two-Tower 模型。它将用户和物品分别通过独立的深度网络编码为向量，再通过向量点积或余弦相似度计算匹配分数。是广告召回阶段的基础架构。

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

余弦相似度：

$$
\text{similarity}(u, v) = \cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}
$$

或使用点积：

$$
\hat{y} = \sigma(\mathbf{u}^T \mathbf{v})
$$

### 关键特性

- 用户塔和物品塔完全独立，可以离线预计算物品向量
- 适合大规模召回场景，延迟极低（FAISS 向量检索）
- 无法建模用户-物品交叉特征（表达能力受限）

## 3. 数学公式与推导

**用户塔编码**：

$$
\mathbf{u} = f_{user}(x_{user}) = W_2^{(u)} \cdot \text{ReLU}(W_1^{(u)} x_{user} + b_1^{(u)}) + b_2^{(u)}
$$

**物品塔编码**：

$$
\mathbf{v} = f_{item}(x_{item}) = W_2^{(v)} \cdot \text{ReLU}(W_1^{(v)} x_{item} + b_1^{(v)}) + b_2^{(v)}
$$

**训练损失**（ softmax 负采样）：

$$
L = -\log \frac{e^{\gamma \cos(u, v^+)}}{\sum_{j} e^{\gamma \cos(u, v_j)}}
$$

其中 $v^+$ 为正样本，$v_j$ 包含正样本和多个负样本，$\gamma$ 为温度系数。

## 4. 训练过程讲解

1. 用户侧特征（历史行为、画像）输入用户塔，输出用户向量 $\mathbf{u}$
2. 物品侧特征（属性、内容）输入物品塔，输出物品向量 $\mathbf{v}$
3. 计算正样本对 $(u, v^+)$ 的相似度，以及与 $K$ 个负样本的相似度
4. 通过 softmax 交叉熵损失优化，拉近正样本、推远负样本
5. 向量归一化后可离线建索引（FAISS），在线用 ANN 检索

## 5. 应用场景

- 广告召回阶段（向量召回）
- 推荐系统召回（百万级候选集筛选）
- 语义相似度计算（搜索查询-文档匹配）
- 多模态广告召回（多模态双塔 + FAISS/Milvus）

## 6. 优缺点分析

**优点**：
- 两塔独立，物品向量可离线预计算，线上只需向量检索
- 延迟极低（FAISS 毫秒级检索）
- 适合超大规模候选集（百万→千级筛选）

**缺点**：
- 无法建模用户-物品交叉特征，表达能力受限
- 训练时的负采样策略对效果影响很大
- 塔的深度和宽度需要仔细调优

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.temperature = 0.1

    def forward(self, user_features, item_features):
        u = self.user_tower(user_features)
        v = self.item_tower(item_features)
        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)
        return torch.sum(u * v, dim=-1, keepdim=True)

    def contrastive_loss(self, user_features, pos_items, neg_items):
        u = F.normalize(self.user_tower(user_features), dim=-1)
        v_pos = F.normalize(self.item_tower(pos_items), dim=-1)
        v_neg = F.normalize(self.item_tower(neg_items), dim=-1)
        pos_sim = torch.sum(u * v_pos, dim=-1) / self.temperature
        neg_sim = torch.sum(u.unsqueeze(1) * v_neg, dim=-1) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

model = DSSM(user_dim=50, item_dim=50)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
u_feat = torch.randn(32, 50)
pos_feat = torch.randn(32, 50)
neg_feat = torch.randn(32, 4, 50)
for epoch in range(10):
    loss = model.contrastive_loss(u_feat, pos_feat, neg_feat)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

class DSSMNumpy:
    def __init__(self, user_dim, item_dim, hidden=32, embed=16):
        s = 0.01
        self.Wu1 = np.random.randn(user_dim, hidden) * s
        self.Wu2 = np.random.randn(hidden, embed) * s
        self.Wv1 = np.random.randn(item_dim, hidden) * s
        self.Wv2 = np.random.randn(hidden, embed) * s

    def encode_user(self, x):
        return normalize(relu(relu(x @ self.Wu1) @ self.Wu2))

    def encode_item(self, x):
        return normalize(relu(relu(x @ self.Wv1) @ self.Wv2))

    def predict(self, user_feat, item_feat):
        u = self.encode_user(user_feat)
        v = self.encode_item(item_feat)
        return np.sum(u * v, axis=-1, keepdims=True)
```

## 9. 可视化与结果理解

- t-SNE 可视化用户和物品向量空间，观察聚类结构
- 绘制正负样本对的相似度分布对比
- 绘制 Recall@K 随 Embedding 维度的变化曲线

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(42)

n_user = 200
n_item = 300
embed_dim = 32

user_centroids = np.array([
    [1.0, 1.0], [-1.0, 1.0], [0.0, -1.5], [1.5, -0.5]
])
user_labels = np.random.randint(0, 4, n_user)
user_embeddings_high = np.random.randn(n_user, embed_dim) * 0.3
for i in range(n_user):
    user_embeddings_high[i, :2] += user_centroids[user_labels[i]]

item_centroids = np.array([
    [1.2, 1.2], [-1.2, 1.2], [0.2, -1.3], [1.3, -0.3]
])
item_labels = np.random.randint(0, 4, n_item)
item_embeddings_high = np.random.randn(n_item, embed_dim) * 0.3
for i in range(n_item):
    item_embeddings_high[i, :2] += item_centroids[item_labels[i]]

all_embeddings = np.vstack([user_embeddings_high, item_embeddings_high])
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

user_2d = embeddings_2d[:n_user]
item_2d = embeddings_2d[n_user:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

user_colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
user_cmap = [user_colors[l] for l in user_labels]
axes[0].scatter(user_2d[:, 0], user_2d[:, 1], c=user_cmap, alpha=0.6, s=30, edgecolors='gray', linewidth=0.5)
axes[0].set_title('User Embeddings (t-SNE)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('t-SNE Dim 1', fontsize=11)
axes[0].set_ylabel('t-SNE Dim 2', fontsize=11)
for c_idx, color in enumerate(user_colors):
    axes[0].scatter([], [], c=color, label=f'User Group {c_idx + 1}', s=50)
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.2)

item_colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
item_cmap = [item_colors[l] for l in item_labels]
axes[1].scatter(item_2d[:, 0], item_2d[:, 1], c=item_cmap, alpha=0.6, s=30, edgecolors='gray', linewidth=0.5, marker='D')
axes[1].set_title('Item Embeddings (t-SNE)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('t-SNE Dim 1', fontsize=11)
axes[1].set_ylabel('t-SNE Dim 2', fontsize=11)
for c_idx, color in enumerate(item_colors):
    axes[1].scatter([], [], c=color, label=f'Item Category {c_idx + 1}', s=50, marker='D')
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.2)

plt.suptitle('DSSM 双塔 Embedding t-SNE 可视化', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dssm_tsne_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **召回指标**：Recall@K、Hit Rate@K、NDCG@K
- **向量检索指标**：检索延迟、QPS
- **对比基线**：与协同过滤、FM 召回对比

## 11. 常见问题与易错点

- 负采样策略对效果影响很大，随机负采样往往不够好，需引入 hard negative
- 向量归一化是必要的，否则模长会干扰相似度计算
- 温度系数 $\gamma$ 需要调优，过大会导致梯度平滑，过小会过度集中
- 双塔无法利用交叉特征，如需交叉能力需升级为 COLD 等模型

## 12. 学习总结

DSSM（双塔模型）是广告召回阶段的基础架构，通过分离用户和物品编码实现高效检索。后续的 COLD、FSCD 等模型都在双塔基础上增加轻量级交叉，但双塔仍是大规模召回的首选方案。

## 13. 练习题与思考题（含答案）

**Q1**: 为什么双塔模型适合召回阶段？
> A1: 两塔独立，物品向量可离线预计算建索引，线上只需 ANN 检索，延迟极低，适合百万级候选集。

**Q2**: 双塔模型无法建模什么？
> A2: 无法建模用户-物品交叉特征，因为交叉只能在最终的内积操作时发生。

**Q3**: 为什么需要负采样？
> A3: softmax 分母需要对所有物品求和，计算不可行。负采样近似全量 softmax，使训练可行。

## 14. 学习路径建议

1. 先学习 Word2Vec 理解 Embedding 概念
2. 学习对比学习和 softmax 损失
3. 学习 DSSM 论文（Microsoft 2013）
4. 进阶：学习 COLD（双塔+SE）、MIND（多兴趣召回）、SASRec（序列召回）
