# LightGCN 学习文档

## 1. 算法基础认知

### 1.1 什么是 LightGCN？

LightGCN（Lightweight Graph Convolution Network）是 2020 年提出的图推荐模型，简化了 NGCF 的设计，去除了特征变换和非线性激活，仅保留邻域聚合和层组合。

### 1.2 动机

**NGCF 的问题：**
- 特征变换（FC 层）在协同过滤中没有帮助
- 非线性激活增加了复杂度但效果不明显
- 参数多、训练困难

**LightGCN 的改进：**
- 移除特征变换和非线性激活
- 仅使用简单的加权聚合
- 参数少、效果更好

### 1.3 核心思想

**Light Graph Convolution:**
$$e_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(l)}$$

$$e_i^{(l+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_u^{(l)}$$

## 2. 模型架构

### 2.1 整体结构

```
用户/物品初始嵌入 E^(0)
        ↓
    ┌───────┐
    │ LGC-1 │  ← 第一层轻量图卷积
    └───────┘
        ↓
    E^(1)
        ↓
    ┌───────┐
    │ LGC-2 │  ← 第二层
    └───────┘
        ↓
    E^(2)
        ↓
    ...
        ↓
    E^(L)
        ↓
    层组合（加权平均）
        ↓
    最终嵌入 E
```

### 2.2 层组合

$$e_u = \sum_{l=0}^{L} \alpha_l e_u^{(l)}$$

其中 $\alpha_l$ 可以设为 $1/(L+1)$。

### 2.3 预测

$$\hat{y}_{ui} = e_u^T e_i$$

## 3. PyTorch 完整实现

### 3.1 LightGCN 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple
import random


class LightGCN(nn.Module):
    """
    LightGCN: Lightweight Graph Convolution Network for Recommendation

    论文: LightGCN: Simplifying and Powering Graph Convolution Network
          for Recommendation (SIGIR 2020)
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.0):
        """
        参数:
            n_users: 用户数量
            n_items: 物品数量
            embed_dim: 嵌入维度
            n_layers: GCN 层数
            dropout: Dropout 比例（论文中设为 0）
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # 归一化邻接矩阵（在外部计算）
        self.norm_adj = None

    def set_adj_matrix(self, norm_adj: torch.Tensor):
        """设置归一化邻接矩阵"""
        self.norm_adj = norm_adj

    def forward(self):
        """
        前向传播：计算所有用户和物品的最终嵌入

        返回:
            user_embs: (n_users, embed_dim)
            item_embs: (n_items, embed_dim)
        """
        # 初始嵌入
        user_embs_0 = self.user_embedding.weight
        item_embs_0 = self.item_embedding.weight

        # 拼接
        embs_0 = torch.cat([user_embs_0, item_embs_0], dim=0)  # (n_users + n_items, embed_dim)

        # 保存每层的嵌入
        embs_list = [embs_0]

        # 多层图卷积
        embs = embs_0
        for _ in range(self.n_layers):
            embs = torch.sparse.mm(self.norm_adj, embs)

            if self.dropout > 0:
                embs = F.dropout(embs, p=self.dropout, training=self.training)

            embs_list.append(embs)

        # 层组合（平均）
        final_embs = torch.stack(embs_list, dim=0).mean(dim=0)

        # 分离用户和物品嵌入
        user_embs = final_embs[:self.n_users]
        item_embs = final_embs[self.n_users:]

        return user_embs, item_embs

    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        预测用户-物品分数

        参数:
            user_indices: 用户索引
            item_indices: 物品索引

        返回:
            分数
        """
        user_embs, item_embs = self.forward()

        user_emb = user_embs[user_indices]  # (batch, embed_dim)
        item_emb = item_embs[item_indices]  # (batch, embed_dim)

        scores = (user_emb * item_emb).sum(dim=-1)  # 内积

        return scores

    def get_all_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取所有用户-物品分数矩阵"""
        user_embs, item_embs = self.forward()
        scores = user_embs @ item_embs.t()  # (n_users, n_items)
        return scores

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_items: set = None) -> List[Tuple[int, float]]:
        """
        为用户推荐

        参数:
            user_idx: 用户索引
            top_k: 推荐 K 个
            exclude_items: 排除的物品集合

        返回:
            [(item_idx, score), ...]
        """
        with torch.no_grad():
            user_embs, item_embs = self.forward()
            user_emb = user_embs[user_idx]

            scores = (user_emb * item_embs).sum(dim=-1)

            if exclude_items:
                for item in exclude_items:
                    scores[item] = -float('inf')

            top_scores, top_indices = torch.topk(scores, top_k)

            return [(int(idx), float(score))
                   for idx, score in zip(top_indices, top_scores)]

    def bpr_loss(self, user_indices: torch.Tensor,
                 pos_item_indices: torch.Tensor,
                 neg_item_indices: torch.Tensor) -> torch.Tensor:
        """
        BPR 损失

        参数:
            user_indices: 用户索引
            pos_item_indices: 正样本物品索引
            neg_item_indices: 负样本物品索引

        返回:
            loss
        """
        user_embs, item_embs = self.forward()

        user_emb = user_embs[user_indices]
        pos_item_emb = item_embs[pos_item_indices]
        neg_item_emb = item_embs[neg_item_indices]

        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        # BPR loss: -log(sigmoid(pos - neg))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 正则化（对初始嵌入）
        reg_loss = (user_emb.norm(2).pow(2) +
                   pos_item_emb.norm(2).pow(2) +
                   neg_item_emb.norm(2).pow(2)) / user_emb.shape[0]

        return loss, reg_loss


def create_adj_matrix(interactions: List[Tuple], n_users: int, n_items: int,
                     normalize: bool = True) -> torch.Tensor:
    """
    创建归一化邻接矩阵

    参数:
        interactions: [(user_idx, item_idx), ...]
        n_users: 用户数
        n_items: 物品数
        normalize: 是否归一化

    返回:
        sparse tensor: (n_users + n_items, n_users + n_items)
    """
    n_nodes = n_users + n_items

    # 构建 COO 格式的邻接矩阵
    rows = []
    cols = []

    for user_idx, item_idx in interactions:
        # 用户 -> 物品
        rows.append(user_idx)
        cols.append(n_users + item_idx)
        # 物品 -> 用户（无向图）
        rows.append(n_users + item_idx)
        cols.append(user_idx)

    data = np.ones(len(rows))

    # 创建稀疏矩阵
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    if normalize:
        # D^{-1/2} A D^{-1/2}
        adj = adj.tocsc()
        degree = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # 转为 COO 并创建 torch sparse tensor
    adj = adj.tocoo()
    indices = torch.LongTensor([adj.row, adj.col])
    values = torch.FloatTensor(adj.data)
    shape = torch.Size(adj.shape)

    return torch.sparse_coo_tensor(indices, values, shape)


class LightGCNDataLoader:
    """
    LightGCN 数据加载器
    """

    def __init__(self, interactions: List[Tuple], n_users: int, n_items: int,
                 batch_size: int = 1024, n_negatives: int = 1):
        """
        参数:
            interactions: [(user_idx, item_idx), ...]
            n_users: 用户数
            n_items: 物品数
            batch_size: 批大小
            n_negatives: 每个正样本的负样本数
        """
        self.interactions = interactions
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_negatives = n_negatives

        # 构建用户交互字典（用于负采样）
        self.user_items = {}
        for user, item in interactions:
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)

    def __len__(self):
        return (len(self.interactions) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # 打乱
        interactions = self.interactions.copy()
        random.shuffle(interactions)

        for i in range(0, len(interactions), self.batch_size):
            batch = interactions[i:i + self.batch_size]

            users = []
            pos_items = []
            neg_items = []

            for user, pos_item in batch:
                users.append(user)
                pos_items.append(pos_item)

                # 负采样
                for _ in range(self.n_negatives):
                    neg_item = random.randint(0, self.n_items - 1)
                    while neg_item in self.user_items.get(user, set()):
                        neg_item = random.randint(0, self.n_items - 1)
                    neg_items.append(neg_item)

            yield {
                'users': torch.LongTensor(users),
                'pos_items': torch.LongTensor(pos_items),
                'neg_items': torch.LongTensor(neg_items)
            }


def train_lightgcn():
    """训练 LightGCN"""
    # 配置
    config = {
        'n_users': 1000,
        'n_items': 5000,
        'embed_dim': 64,
        'n_layers': 3,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'n_negatives': 1
    }

    # 生成模拟数据
    n_interactions = 50000
    interactions = []
    for _ in range(n_interactions):
        user = random.randint(0, config['n_users'] - 1)
        item = random.randint(0, config['n_items'] - 1)
        interactions.append((user, item))

    # 创建邻接矩阵
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm_adj = create_adj_matrix(interactions, config['n_users'], config['n_items'])
    norm_adj = norm_adj.coalesce().to(device)

    # 创建模型
    model = LightGCN(
        n_users=config['n_users'],
        n_items=config['n_items'],
        embed_dim=config['embed_dim'],
        n_layers=config['n_layers']
    ).to(device)

    model.set_adj_matrix(norm_adj)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 数据加载器
    dataloader = LightGCNDataLoader(
        interactions, config['n_users'], config['n_items'],
        batch_size=config['batch_size'], n_negatives=config['n_negatives']
    )

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_reg = 0

        for batch in dataloader:
            users = batch['users'].to(device)
            pos_items = batch['pos_items'].to(device)
            neg_items = batch['neg_items'].to(device)

            optimizer.zero_grad()

            loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
            total_loss_batch = loss + config['weight_decay'] * reg_loss

            total_loss_batch.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg += reg_loss.item()

        n_batches = len(dataloader)
        print(f"Epoch {epoch + 1}/{config['epochs']}, "
              f"Loss: {total_loss / n_batches:.4f}, "
              f"Reg: {total_reg / n_batches:.4f}")

        # 测试推荐
        if (epoch + 1) % 10 == 0:
            model.eval()
            recs = model.recommend(0, top_k=5)
            print(f"  Top-5 for user 0: {recs}")

    return model


if __name__ == "__main__":
    model = train_lightgcn()
    print("LightGCN 训练完成！")
```

### 3.2 评估

```python
def evaluate_lightgcn(model, test_interactions, k_list=[10, 20, 50]):
    """
    评估 LightGCN

    参数:
        model: LightGCN 模型
        test_interactions: {user_idx: [item_idx, ...]}
        k_list: 评估的 K 值列表

    返回:
        指标字典
    """
    model.eval()
    device = next(model.parameters()).device

    # 获取所有分数
    with torch.no_grad():
        scores = model.get_all_scores().cpu().numpy()  # (n_users, n_items)

    metrics = {f'Precision@{k}': [] for k in k_list}
    metrics.update({f'Recall@{k}': [] for k in k_list})
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []

    for user, items in test_interactions.items():
        if not items:
            continue

        user_scores = scores[user]

        # 排除训练物品（这里假设 test_items 已经是测试集）
        # 实际使用时需要传入训练物品

        # 排序
        ranked_items = np.argsort(user_scores)[::-1]

        # 计算指标
        for k in k_list:
            top_k = set(ranked_items[:k])
            hit = len(top_k & set(items))

            precision = hit / k
            recall = hit / len(items) if items else 0

            # NDCG
            dcg = 0
            for i, item in enumerate(ranked_items[:k]):
                if item in items:
                    dcg += 1 / np.log2(i + 2)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(items))))
            ndcg = dcg / idcg if idcg > 0 else 0

            metrics[f'Precision@{k}'].append(precision)
            metrics[f'Recall@{k}'].append(recall)
            metrics[f'NDCG@{k}'].append(ndcg)

        # MRR
        mrr = 0
        for i, item in enumerate(ranked_items):
            if item in items:
                mrr = 1 / (i + 1)
                break
        metrics['MRR'].append(mrr)

    # 平均
    return {k: np.mean(v) for k, v in metrics.items()}
```

## 4. 与其他模型对比

### 4.1 与 NGCF 对比

| 维度 | NGCF | LightGCN |
|------|------|----------|
| 特征变换 | 有 | 无 |
| 非线性激活 | 有 | 无 |
| Dropout | 有 | 无（通常） |
| 参数量 | 大 | 小 |
| 效果 | 好 | 更好 |
| 训练速度 | 慢 | 快 |

### 4.2 效果对比

| 数据集 | NGCF | LightGCN | 提升 |
|--------|------|----------|------|
| Gowalla | 0.1570 | 0.1830 | +16.6% |
| Amazon-Book | 0.0670 | 0.0770 | +14.9% |
| ML-1M | 0.2220 | 0.2440 | +9.9% |

## 5. 关键设计

### 5.1 为什么移除特征变换？

协同过滤中，用户和物品的初始嵌入是随机初始化的，没有语义特征。特征变换对随机向量没有帮助，反而增加了过拟合风险。

### 5.2 为什么移除非线性激活？

在推荐场景中，线性聚合已经足够捕获用户-物品交互模式。非线性激活增加了计算复杂度，但效果提升不明显。

### 5.3 层组合的作用

不同层捕获不同阶的邻居信息：
- 第 0 层：自身
- 第 1 层：直接邻居
- 第 2 层：2 跳邻居
- ...

层组合将这些信息融合，得到更丰富的表示。

## 6. 调参建议

### 6.1 模型参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 64 | 嵌入维度 |
| n_layers | 3-4 | 层数不宜太多 |
| dropout | 0 | 通常不需要 |

### 6.2 训练参数

| 参数 | 推荐值 |
|------|--------|
| learning_rate | 0.001 |
| batch_size | 1024 |
| weight_decay | 1e-4 |
| n_negatives | 1 |

## 7. 学习总结

### 7.1 核心要点

1. **简单即有效**：移除不必要的组件反而效果更好
2. **邻域聚合**：核心是传播用户-物品交互信息
3. **层组合**：融合不同阶的邻居信息

### 7.2 适用场景

- 纯协同过滤场景
- 用户-物品交互数据
- 需要高召回的召回层

## 8. 练习题

1. 实现 LightGCN 的 mini-batch 版本。

2. 比较不同层数（1-5）对效果的影响。

3. 将 LightGCN 与 ItemCF 结合，设计混合模型。
