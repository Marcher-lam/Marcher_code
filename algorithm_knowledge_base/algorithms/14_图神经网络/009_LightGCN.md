# LightGCN 学习文档

> 图卷积网络的轻量化变体，移除特征变换，仅保留邻居聚合，在推荐系统中表现优异

---

## 1. 算法基础认知

**一句话定义**：LightGCN是GCN的轻量化变体，通过移除GCN中的特征变换矩阵W，只保留关键的邻居聚合操作，大幅减少参数量和计算量，同时在推荐系统任务中取得显著更好的效果。

**直觉类比**：LightGCN就像一个"纯粹的社交网络分析"。想象你在分析一个社交网络——传统GCN不仅分析"你和谁在联系"，还要分析每个人的"社交能力"、"表达能力"等（特征变换）。LightGCN认为这些额外信息不重要且容易引入噪音，关键是"你和谁在联系"。所以它只保留邻居聚合这一步，去掉了所有不必要的特征变换，就像删掉社交分析中的"废话"，只保留最核心的关系。这让它更高效、更专注、效果更好。

**历史背景**：
- 2020年，He et al. 在论文"LightGCN: Light Graph Convolutional Network for Recommender System"中提出
- 在MovieLens、Gowalla等数据集上，LightGCN相比GCN性能提升超过20%
- 同时参数量减少80%
- 后续成为推荐系统领域的基线方法

**算法定位**：
- 类型：图神经网络 → 推荐系统
- 输出：用户/物品嵌入
- 模型类型：轻量级GCN

**前置知识**：
- [必备]：GCN基础（卷积、聚合、邻接矩阵）
- [必备]：推荐系统基础（协同过滤、矩阵分解）
- [推荐]：图论基础

---

## 2. 核心原理

### 2.1 GCN的问题

标准GCN的层更新公式：

$$H^{(k+1)} = \sigma\left(D^{-1/2} A D^{-1/2} H^{(k)} W^{(k)}\right)$$

这个公式包含两个核心操作：
1. **邻居聚合**：$D^{-1/2} A D^{-1/2} H^{(k)}$ - 聚合邻居信息
2. **特征变换**：$H^{(k)} W^{(k)}$ - 线性变换矩阵W

**问题分析**：

| 操作 | 作用 | 在推荐中的问题 |
|------|------|-----------|
| 邻居聚合 | 捕捉协同过滤模式 | 核心，必须保留 |
| 特征变换 | 增加表达能力 | 引入噪音，性能下降 |

在推荐系统中：
- 用户和物品的ID嵌入是最重要的特征
- 额外的特征变换反而会破坏嵌入的语义
- 增加了过拟合风险

### 2.2 LightGCN的核心创新

**核心创新**：移除特征变换，只保留邻居聚合！

$$\text{LightGCN公式}: H^{(k+1)} = (D+I)^{-1/2} (A+I) (D+I)^{-1/2} H^{(k)}$$

或者写得更清晰：

$$H^{(k+1)} = \tilde{A} H^{(k)}$$

其中 $\tilde{A} = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}$ 是归一化邻接矩阵。

### 2.3 为什么LightGCN更好？

**理论解释**：

1. **保持嵌入空间语义**：在推荐系统中，用户/物品嵌入的度量空间是有意义的。特征变换会改变这个空间，破坏度量学习的有效性。

2. **减少过拟合**：移除W矩阵后，参数量大幅减少，泛化能力提升。

3. **更纯粹的协同过滤**：邻居聚合本质上就是协同过滤——"相似的用户会喜欢相似的物品"。LightGCN更纯粹地实现了这个思想。

**直观理解**：

| 方法 | 核心思想 | 类比 |
|------|----------|------|
| GCN | 聚合 + 变换 | 看朋友 + 分析朋友能力 |
| LightGCN | 仅聚合 | 只看朋友是谁 |
| 比喻 | 社交达人分析 | 简单关系网络 |

### 2.4 层叠加与嵌入组合

LightGCN使用**多层叠加**来捕捉高阶邻域：

```
初始嵌入 H^(0)
    │
    ▼ 1层聚合
H^(1) = Ã H^(0)
    │
    ▼ 2层聚合
H^(2) = Ã H^(1)
    │
    ▼ 3层聚合
H^(3) = Ã H^(2)
    │
    ▼ 平均
H = (H^(0) + H^(1) + H^(2) + H^(3)) / 4
```

**最终嵌入**是所有层的平均值：

$$H = \frac{1}{K+1} \sum_{k=0}^{K} H^{(k)}$$

---

## 3. 数学公式与推导

### 3.1 邻接矩阵构建

**用户-物品交互图**：

| 节点类型 | 数量 | 说明 |
|----------|------|------|
| 用户 | $N_u$ | 评分的用户 |
| 物品 | $N_i$ | 被评分的物品 |

**邻接矩阵** $A \in \mathbb{R}^{(N_u+N_i) \times (N_u+N_i)}$：

$$
A = \begin{bmatrix} 
0 & R \\\
R^T & 0 
\end{bmatrix}
$$

其中 $R$ 是评分/交互矩阵。

**带自环的归一化矩阵**：

$$\tilde{A} = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}$$

### 3.2 LightGCN层更新

**第k层**：

$$H^{(k+1)} = \tilde{A} H^{(k)}$$

**展开形式**：

$$h_u^{(k+1)} = \frac{1}{\sqrt{N_u^{(u)}}\sqrt{N_i^{(u)}}} \sum_{i \in \mathcal{N}_u} h_i^{(k)}$$

其中 $\mathcal{N}_u$ 是用户u交互过的物品集合。

### 3.3 嵌入组合

**最终用户嵌入**：

$$\bar{h}_u = \frac{1}{K+1} \sum_{k=0}^{K} h_u^{(k)}$$

**最终物品嵌入**：

$$\bar{h}_i = \frac{1}{K+1} \sum_{k=0}^{K} h_i^{(k)}$$

### 3.4 预测

**评分预测**（使用BPR或-dot product）：

$$\hat{y}_{ui} = \bar{h}_u^T \bar{h}_i$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       构建用户-物品图
           │
           ▼
    ┌───────────────┐
    │ 初始化嵌入   │ ← 随机或预训练
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  K层聚合   │ ← 每层邻居聚合
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  层平均    │ ← 组合所有层
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  计算损失   │ ← BPR / CE
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  反向传播    │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  更新嵌入   │
    └───────────────┘
```

### 4.2 损失函数

**BPR损失**（推荐系统常用）：

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

其中：
- $(u,i,j)$：用户u对物品i正/物品j负
- $\sigma$：sigmoid函数

**带L2正则**：

$$\mathcal{L} = \mathcal{L}_{BPR} + \lambda \|E\|_F^2$$

### 4.3 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| emb_dim | 64-256 | 嵌入维度 |
| num_layers | 2-4 | GCN层数 |
| lr | 0.001 | 学习率 |
| batch_size | 1024-4096 | 批次 |
| reg | 1e-4 | 正则化 |

### 4.4 训练技巧

| 技巧 | 说明 |
|------|------|
| 邻接矩阵稀疏化 | 减少内存 |
| 采样负样本 | 加速训练 |
| 早停 | 防止过拟合 |

---

## 5. 应用场景

### 5.1 推荐系统

这是LightGCN最核心的应用！

**场景**：
- 电影推荐（MovieLens）
- 音乐推荐（Spotify）
- 商品推荐（Amazon）
- 书籍推荐（Goodreads）

**优势**：
- 效果好（超越GCN 20%+）
- 速度快（参数量少）
- 可解释（高阶邻域）

### 5.2 链接预测

**任务**：预测用户和物品之间是否会有交互

**方法**：
- 计算相似度 $\bar{h}_u^T \bar{h}_i$
- 高相似度 → 可能交互

### 5.3 社交网络

**任务**：推荐新朋友

**方法**：学习用户嵌入，相似用户推荐

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **效果好** | 在推荐系统超越GCN 20%+ |
| **参数少** | 减少80%参数 |
| **训练快** | 更快收敛 |
| **泛化好** | 不易过拟合 |
| **简单** | 实现简洁 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 依赖图结构 | 需要构建图 |
| 需要交互数据 | 冷启动问题 |
| 长尾效果差 | 交互少的用户 |

### 6.3 改进方向

| 方向 | 方法 |
|------|------|
| 缓解冷启动 | 加入辅助特征 |
| 长尾优化 | 加入正则 |
| 多行为 | 多类型交互 |

---

## 7. 调库实现

### 7.1 使用RecBole（推荐）

```python
# 安装
# pip install recbole

from recbole.model.general_recommender import LightGCN
from recbole.data import dataset

# 使用内置数据集
from recbole.config import Config
from recbole.data import create_dataset

# 配置
config_dict = {
    'model': 'LightGCN',
    'dataset': 'ml-100k',
    'embedding_size': 64,
    'n_layers': 3,
    'learning_rate': 0.001,
    'num_epochs': 100,
}

# 创建模型
model = LightGCN(config_dict, dataset)

# 训练
model.train()
```

### 7.2 使用PyTorch Geometric

```python
# 安装
# pip install torch torch_geometric

import torch
import torch.nn as nn
from torch_geometric.nn import LightGCNConv


class LightGCN(nn.Module):
    """LightGCN模型"""
    
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        self.convs = nn.ModuleList([
            LightGCNConv(embedding_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, edge_index):
        x = self.embedding.weight
        
        all_embeddings = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)
        
        # 平均所有层
        embeddings = torch.stack(all_embeddings).mean(dim=0)
        
        return embeddings


# 训练
if __name__ == "__main__":
    num_nodes = 1000
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    
    model = LightGCN(num_nodes=4, embedding_dim=64)
    out = model(edge_index)
    print(out.shape)
```

### 7.3 完整推荐系统实现

```python
import torch
import torch.nn as nn
import numpy as np


class LightGCNRecommender(nn.Module):
    """LightGCN推荐系统"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, adj_matrix):
        """
        Args:
            adj_matrix: 归一化邻接矩阵 [N, N]
        Returns:
            用户和物品嵌入
        """
        
        # 分离用户和物品嵌入
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        
        # 拼接
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        
        # 多层聚合
        all_embeddings = [all_emb]
        
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            all_embeddings.append(all_emb)
        
        # 层平均
        final_emb = torch.stack(all_embeddings).mean(dim=0)
        
        # 分离
        final_users = final_emb[:self.num_users]
        final_items = final_emb[self.num_users:]
        
        return final_users, final_items
    
    def predict(self, adj_matrix, user_ids, item_ids):
        """预测评分"""
        
        users_emb, items_emb = self.forward(adj_matrix)
        
        users = users_emb[user_ids]
        items = items_emb[item_ids]
        
        # 点积
        predictions = (users * items).sum(dim=-1)
        
        return torch.sigmoid(predictions)


def build_adj_matrix(interactions, num_users, num_items):
    """构建归一化邻接矩阵"""
    
    # 构建稀疏矩阵
    rows = []
    cols = []
    data = []
    
    for user_id, item_id in interactions:
        rows.append(user_id)
        cols.append(num_items + item_id)
        data.append(1)
        
        rows.append(num_items + item_id)
        cols.append(user_id)
        data.append(1)
    
    # 转为稀疏矩阵
    adj = torch.sparse.FloatTensor(
        torch.tensor([rows, cols]),
        torch.tensor(data, dtype=torch.float32),
        torch.Size([num_users + num_items, num_users + num_items])
    )
    
    # 度归一化
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    degrees = torch.pow(degrees, -0.5)
    degrees[degrees == float('inf')] = 0
    
    # 归一化
    d_indices = torch.arange(num_users + num_items)
    d_matrix = torch.sparse.FloatTensor(
        torch.stack([d_indices, d_indices]),
        degrees,
        torch.Size([num_users + num_items, num_users + num_items])
    )
    
    adj = torch.sparse.mm(d_matrix, torch.sparse.mm(adj, d_matrix))
    
    return adj


def bpr_loss(pos_scores, neg_scores):
    """BPR损失"""
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))
    return loss.mean()


def train_lightgcn(interactions, num_users, num_items, epochs=100):
    """训练LightGCN"""
    
    # 构建邻接矩阵
    adj = build_adj_matrix(interactions, num_users, num_items)
    
    # 模型
    model = LightGCNRecommender(num_users, num_items, embedding_dim=64, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        
        # 采样正负样本
        user_ids = torch.randint(0, num_users, (1024,))
        pos_items = torch.randint(0, num_items, (1024,))
        neg_items = torch.randint(0, num_items, (1024,))
        
        # 预测
        pos_scores = model.predict(adj, user_ids, pos_items)
        neg_scores = model.predict(adj, user_ids, neg_items)
        
        # 损失
        loss = bpr_loss(pos_scores, neg_scores)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    # 示例
    num_users = 100
    num_items = 50
    interactions = [(i % num_users, i % num_items) for i in range(500)]
    
    model = train_lightgcn(interactions, num_users, num_items)
    print("训练完成！")
```

---

## 8. 手工代码实现

### 8.1 核心LightGCN层

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse


class LightGCNLayer(nn.Module):
    """LightGCN单层"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, adj):
        """
        Args:
            x: 节点嵌入 [N, embed_dim]
            adj: 稀疏归一化邻接矩阵
        Returns:
            更新后的嵌入
        """
        
        # 稀疏矩阵乘法
        out = torch.sparse.mm(adj, x)
        
        return out


class LightGCNModel(nn.Module):
    """完整LightGCN模型"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 可学习嵌入
        self.embedding = nn.Embedding(
            num_users + num_items, 
            embedding_dim
        )
        
        # LightGCN层（无参数！）
        self.lightgcn = LightGCNLayer()
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        
    def forward(self, adj):
        """
        Args:
            adj: 归一化邻接矩阵
        Returns:
            用户和物品嵌入
        """
        
        x = self.embedding.weight
        
        all_embeddings = [x]
        
        # K层聚合
        for _ in range(self.num_layers):
            x = self.lightgcn(x, adj)
            all_embeddings.append(x)
        
        # 平均
        final_emb = torch.stack(all_embeddings).mean(dim=0)
        
        # 分离
        users_emb = final_emb[:self.num_users]
        items_emb = final_emb[self.num_users:]
        
        return users_emb, items_emb
    
    def get_recommendations(self, adj, user_id, top_k=10):
        """获取推荐"""
        
        users_emb, items_emb = self.forward(adj)
        
        # 用户嵌入
        user_emb = users_emb[user_id]
        
        # 计算相似度
        scores = torch.mm(user_emb.unsqueeze(0), items_emb.t()).squeeze()
        
        # Top-K
        _, top_items = torch.topk(scores, top_k)
        
        return top_items


def build_normalized_adj(user_item_interactions, num_users, num_items):
    """构建归一化邻接矩阵"""
    
    n_nodes = num_users + num_items
    
    # 构建边
    edges = []
    weights = []
    
    for (user, item) in user_item_interactions:
        edges.append((user, num_users + item))
        edges.append((num_users + item, user))
        weights.append(1)
        weights.append(1)
    
    # 转稀疏
    adj = sparse.csr_matrix(
        (weights, (np.array(edges)[:, 0], np.array(edges)[:, 1])),
        shape=(n_nodes, n_nodes)
    )
    
    # 度
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees = np.power(degrees, -0.5)
    degrees[degrees == np.inf] = 0
    
    # 归一化
    d = sparse.diags(degrees)
    adj = d @ adj @ d
    
    # 转PyTorch
    adj = torch.sparse.FloatTensor(
        torch.from_numpy(adj.tocoo().row).long(),
        torch.from_numpy(adj.tocoo().col).long(),
        torch.from_numpy(adj.tocoo().data).float(),
        size=(n_nodes, n_nodes)
    )
    
    return adj
```

### 8.2 训练循环

```python
import torch.optim as optim


def train(model, adj, train_data, epochs=100, lr=0.001):
    """训练"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        
        for user, pos_item, neg_item in train_data:
            users_emb, items_emb = model.forward(adj)
            
            # 正样本分数
            pos_score = torch.sum(users_emb[user] * items_emb[pos_item])
            
            # 负样本分数
            neg_score = torch.sum(users_emb[user] * items_emb[neg_item])
            
            # BPR损失
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_data):.4f}")


def evaluate(model, adj, test_data):
    """评估"""
    
    model.eval()
    
    users_emb, items_emb = model.forward(adj)
    
    hits = 0
    total = 0
    
    with torch.no_grad():
        for user, pos_item in test_data:
            # 用户嵌入
            user_emb = users_emb[user]
            
            # 所有物品分数
            scores = torch.mm(user_emb.unsqueeze(0), items_emb.t()).squeeze()
            
            # Top-K
            _, top_items = torch.topk(scores, 10)
            
            if pos_item in top_items:
                hits += 1
            
            total += 1
    
    hit_rate = hits / total
    print(f"Hit Rate @ 10: {hit_rate:.4f}")
    
    return hit_rate
```

---

## 9. 可视化与结果理解

### 9.1 嵌入可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings(users_emb, items_emb, user_labels=None):
    """可视化嵌入（t-SNE）"""
    
    # 合并
    all_emb = torch.cat([users_emb, items_emb], dim=0)
    
    # t-SNE
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(all_emb.cpu().numpy())
    
    # 绘制
    plt.figure(figsize=(10, 10))
    
    # 用户
    n_users = users_emb.size(0)
    plt.scatter(emb_2d[:n_users, 0], emb_2d[:n_users, 1], 
               c='blue', marker='o', label='Users')
    
    # 物品
    plt.scatter(emb_2d[n_users:, 0], emb_2d[n_users:, 1], 
               c='red', marker='x', label='Items')
    
    plt.legend()
    plt.title("LightGCN Embeddings")
    plt.show()
```

### 9.2 层贡献分析

```python
def analyze_layer_contribution(model, adj):
    """分析各层的贡献"""
    
    x = model.embedding.weight
    layer_embeddings = []
    
    for _ in range(model.num_layers):
        x = torch.sparse.mm(adj, x)
        layer_embeddings.append(x)
    
    # 计算各层的重要性
    importances = []
    for i, emb in enumerate(layer_embeddings):
        imp = torch.norm(emb - layer_embeddings[0])
        importances.append(imp.item())
    
    # 绘制
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Layer")
    plt.ylabel("Contribution")
    plt.title("Layer Contribution")
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 计算 |
|------|------|------|
| Recall@K | 命中率 | 命中的正样本 / 总正样本 |
| NDCG@K | NDCG | 考虑排序的DCG |
| Hit@K | Hit Rate | 有命中=1，否则=0 |

### 10.2 基准数据集

| 数据集 | 用户 | 物品 | 交互 |
|--------|------|------|------|
| MovieLens-100K | 943 | 1682 | 100K |
| MovieLens-1M | 6040 | 3706 | 1M |
| Gowalla | 29K | 31K | 1M |

### 10.3 GCN vs LightGCN

| 方法 | Recall@20 | 参数量 |
|------|-----------|--------|
| GCN | 0.14 | 256K |
| LightGCN | 0.18 | 50K |
| 改进 | +28% | -80% |

---

## 11. 常见问题与易错点

### 11.1 稀疏图

**问题**：图太稀疏导致效果差

**解决**：
- 边采样增强
- 加入辅助特征

### 11.2 过度平滑

**问题**：层数太多，嵌入趋同

**解决**：
- 限制层数（2-4层）
- 层组合权重

### 11.3 冷启动

**问题**：新用户没有交互

**解决**：
- 加入内容特征
- 混合方法

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | GCN去除特征变换 |
| 核心 | H = ÃH |
| 优势 | 效果好 + 参数少 |
| 应用 | 推荐系统 |

### 12.2 公式记忆

$$H^{(k+1)} = \tilde{A} H^{(k)}$$
$$\bar{H} = \frac{1}{K+1} \sum_{k=0}^{K} H^{(k)}$$

### 12.3 扩展阅读

| 论文 | 年份 | 贡献 |
|------|------|------|
| LightGCN | 2020 | 原始论文 |
| NGCF | 2019 | 图协同过滤 |
| GAT | 2019 | 注意力GNN |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：为什么LightGCN比GCN效果好？

**答案**：在推荐系统中，特征变换矩阵W会破坏嵌入空间的度量语义，而邻居聚合本质就是协同过滤——只保留这个核心操作，避免了过拟合和噪声。

**练习2**：LightGCN的参数量有多少？

**答案**：只有嵌入矩阵的参数量N×d，而GCN还有额外的W矩阵。

**练习3**：层平均的作用是什么？

**答案**：融合不同阶的邻域信息，1层捕获1阶邻居，2层捕获2阶...

### 13.2 进阶思考

**思考1**：如何处理冷启动用户？

**提示**：可以加入辅助内容特征或使用混合方法。

**思考2**：如何选择层数？

**提示**：通常2-4层，需要根据数据规模和稀疏程度调参。

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | GCN基础 | 理解卷积 |
| 3-4 | 推荐系统 | 协同过滤 |
| 5-6 | LightGCN | 原理理解 |
| 7 | 代码 | 跑通demo |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 实现 | 完整代码 |
| 2 | 优化 | 调参与改进 |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| LightGCN论文 | https://arxiv.org/abs/2003.00919 |
| RecBole | https://recbole.cn/ |

### B. 代码资源

```python
# 推荐项目
# 1. RecBole
# 2. PyG
# 3. NGCF / LightGCN官方实现
```

---

**文档结束**