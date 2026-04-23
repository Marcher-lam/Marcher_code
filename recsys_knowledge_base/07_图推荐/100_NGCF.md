# NGCF (Neural Graph Collaborative Filtering) 学习文档

## 1. 算法基础认知

### 1.1 什么是 NGCF？

NGCF 是一种基于图神经网络的协同过滤方法，通过在用户-物品交互图上传播嵌入来捕捉高阶协同信号。

### 1.2 核心创新

```
传统 CF: 仅使用一阶交互（用户-物品直接交互）
NGCF:   显式建模高阶交互（用户-物品-用户-物品...）
```

### 1.3 模型架构

```
User/Item Embeddings
       ↓
Message Passing (GNN Layers)
       ↓
Embedding Aggregation
       ↓
Prediction
```

## 2. 核心原理

### 2.1 交互图构建

```
二部图结构:
- 节点: 用户和物品
- 边: 交互关系（点击、购买等）

邻接矩阵:
A = [ 0   R ]
    [ R^T 0 ]

其中 R 是用户-物品交互矩阵
```

### 2.2 消息传播

对于用户 u 和其邻居物品 i:

$$m_{u \leftarrow i} = \frac{1}{\sqrt{|N_u||N_i|}} (W_1 e_i + W_2 (e_i \odot e_u))$$

其中:
- $e_u, e_i$: 用户和物品的嵌入
- $W_1, W_2$: 可学习的权重矩阵
- $\odot$: 逐元素乘积

### 2.3 嵌入更新

$$e_u^{(l+1)} = \text{LeakyReLU}(m_{u \leftarrow u} + \sum_{i \in N_u} m_{u \leftarrow i})$$

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import scipy.sparse as sp


class NGCFLayer(nn.Module):
    """
    NGCF 单层
    """

    def __init__(self, embed_dim: int, mess_dropout: float = 0.1):
        super().__init__()

        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(mess_dropout)

    def forward(self, user_embed: torch.Tensor, item_embed: torch.Tensor,
                adj_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单层消息传播

        参数:
            user_embed: (n_users, embed_dim)
            item_embed: (n_items, embed_dim)
            adj_norm: 归一化邻接矩阵

        返回:
            new_user_embed, new_item_embed
        """
        n_users = user_embed.size(0)

        # 拼接用户和物品嵌入
        all_embed = torch.cat([user_embed, item_embed], dim=0)

        # 线性变换
        W1_embed = self.W1(all_embed)
        W2_embed = self.W2(all_embed)

        # 交互项
        # 对于每个用户，计算与邻居物品的交互
        # 这里简化实现，使用邻接矩阵乘法

        # 消息聚合
        agg_embed = torch.sparse.mm(adj_norm, all_embed)
        agg_W1 = torch.sparse.mm(adj_norm, W1_embed)

        # 逐元素交互（简化版本）
        interaction = torch.sparse.mm(adj_norm, W2_embed * all_embed)

        # 合并
        output = agg_W1 + interaction
        output = self.leaky_relu(output)
        output = self.dropout(output)

        # 分离用户和物品
        new_user_embed = output[:n_users]
        new_item_embed = output[n_users:]

        return new_user_embed, new_item_embed


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64,
                 n_layers: int = 4,
                 mess_dropout: float = 0.1,
                 node_dropout: float = 0.0):
        """
        参数:
            n_users: 用户数量
            n_items: 物品数量
            embed_dim: 嵌入维度
            n_layers: GNN 层数
            mess_dropout: 消息 dropout
            node_dropout: 节点 dropout
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.node_dropout = node_dropout

        # 初始嵌入
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # NGCF 层
        self.ngcf_layers = nn.ModuleList([
            NGCFLayer(embed_dim, mess_dropout)
            for _ in range(n_layers)
        ])

        # 预测层（可选）
        self.use_predictor = False

    def create_adj_matrix(self, interaction_matrix: sp.csr_matrix) -> torch.Tensor:
        """
        创建归一化邻接矩阵

        参数:
            interaction_matrix: 用户-物品交互矩阵 (n_users, n_items)

        返回:
            归一化邻接矩阵 (COO 格式)
        """
        n_users, n_items = interaction_matrix.shape

        # 创建二部图邻接矩阵
        # [ 0   R ]
        # [ R^T 0 ]
        R = interaction_matrix.tocoo()
        n_nodes = n_users + n_items

        # 构建对称邻接矩阵的索引
        row = np.concatenate([R.row, R.col + n_users])
        col = np.concatenate([R.col + n_users, R.row])
        data = np.ones(len(row))

        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        # 添加自环
        adj = adj + sp.eye(n_nodes)

        # 对称归一化: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

        # 转换为 PyTorch sparse tensor
        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = torch.Size(adj_norm.shape)

        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            adj_norm: 归一化邻接矩阵

        返回:
            user_final_embed, item_final_embed
        """
        # 初始嵌入
        user_embed = self.user_embedding.weight
        item_embed = self.item_embedding.weight

        # 存储各层嵌入用于最终聚合
        user_embeds = [user_embed]
        item_embeds = [item_embed]

        # 逐层传播
        for ngcf_layer in self.ngcf_layers:
            user_embed, item_embed = ngcf_layer(user_embed, item_embed, adj_norm)
            user_embeds.append(user_embed)
            item_embeds.append(item_embed)

        # 聚合各层嵌入
        user_final = torch.mean(torch.stack(user_embeds, dim=1), dim=1)
        item_final = torch.mean(torch.stack(item_embeds, dim=1), dim=1)

        return user_final, item_final

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                user_embed: torch.Tensor, item_embed: torch.Tensor) -> torch.Tensor:
        """
        预测用户-物品分数
        """
        user_e = user_embed[user_ids]
        item_e = item_embed[item_ids]

        return torch.sum(user_e * item_e, dim=-1)

    def get_all_scores(self, user_embed: torch.Tensor,
                       item_embed: torch.Tensor) -> torch.Tensor:
        """
        计算所有用户对所有物品的分数
        """
        return torch.matmul(user_embed, item_embed.T)


class BPRLoss(nn.Module):
    """
    BPR 损失函数
    """

    def __init__(self, reg_weight: float = 1e-5):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor,
                reg_loss: torch.Tensor = None) -> torch.Tensor:
        """
        BPR 损失
        """
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10))

        if reg_loss is not None:
            loss += self.reg_weight * reg_loss

        return loss


class NGCFTrainer:
    """
    NGCF 训练器
    """

    def __init__(self, model: NGCF, adj_norm: torch.Tensor,
                 learning_rate: float = 0.001, reg_weight: float = 1e-5):
        self.model = model
        self.adj_norm = adj_norm
        self.reg_weight = reg_weight

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = BPRLoss(reg_weight)

    def sample_negatives(self, pos_items: torch.Tensor, n_items: int) -> torch.Tensor:
        """采样负样本"""
        neg_items = torch.randint(0, n_items, pos_items.shape)
        return neg_items

    def train_step(self, user_ids: torch.Tensor, pos_items: torch.Tensor) -> float:
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()

        # 前向传播
        user_embed, item_embed = self.model(self.adj_norm)

        # 采样负样本
        neg_items = self.sample_negatives(pos_items, self.model.n_items)

        # 计算分数
        pos_score = self.model.predict(user_ids, pos_items, user_embed, item_embed)
        neg_score = self.model.predict(user_ids, neg_items, user_embed, item_embed)

        # 正则化
        reg_loss = (user_embed[user_ids] ** 2).sum() + \
                   (item_embed[pos_items] ** 2).sum() + \
                   (item_embed[neg_items] ** 2).sum()

        # 计算损失
        loss = self.loss_fn(pos_score, neg_score, reg_loss)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, test_dict: Dict[int, List[int]], k_list: List[int] = [10, 20]) -> Dict:
        """评估"""
        self.model.eval()

        with torch.no_grad():
            user_embed, item_embed = self.model(self.adj_norm)
            all_scores = self.model.get_all_scores(user_embed, item_embed)

        # 计算指标
        metrics = {f'recall@{k}': [] for k in k_list}
        metrics.update({f'ndcg@{k}': [] for k in k_list})

        for user_id, pos_items in test_dict.items():
            scores = all_scores[user_id].cpu().numpy()

            # 排除训练集中的物品（这里简化）

            # 排序
            top_k_items = np.argsort(scores)[::-1]

            for k in k_list:
                # Recall@K
                top_k_set = set(top_k_items[:k])
                pos_set = set(pos_items)
                hits = len(top_k_set & pos_set)
                recall = hits / len(pos_set) if pos_set else 0
                metrics[f'recall@{k}'].append(recall)

                # NDCG@K
                dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(top_k_items[:k])
                         if item in pos_set)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(pos_set))))
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'ndcg@{k}'].append(ndcg)

        return {k: np.mean(v) for k, v in metrics.items()}


def demo_ngcf():
    """NGCF 示例"""
    # 配置
    n_users = 1000
    n_items = 500
    n_interactions = 10000

    # 创建模拟交互矩阵
    rows = np.random.randint(0, n_users, n_interactions)
    cols = np.random.randint(0, n_items, n_interactions)
    data = np.ones(n_interactions)

    interaction_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # 创建模型
    model = NGCF(
        n_users=n_users,
        n_items=n_items,
        embed_dim=64,
        n_layers=3,
        mess_dropout=0.1
    )

    # 创建邻接矩阵
    adj_norm = model.create_adj_matrix(interaction_matrix)

    # 前向传播
    user_embed, item_embed = model(adj_norm)

    print(f"用户嵌入形状: {user_embed.shape}")
    print(f"物品嵌入形状: {item_embed.shape}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    demo_ngcf()
```

## 4. 与 LightGCN 对比

### 4.1 主要区别

| 特性 | NGCF | LightGCN |
|------|------|----------|
| 消息函数 | W1*e + W2*(e⊙e) | 仅聚合 |
| 非线性 | LeakyReLU | 无 |
| 参数量 | 大 | 小 |
| 性能 | 好 | 更好 |

### 4.2 实验结果

```
MovieLens-1M:
- NGCF:  Recall@20 ≈ 0.158, NDCG@20 ≈ 0.128
- LightGCN: Recall@20 ≈ 0.164, NDCG@20 ≈ 0.134

LightGCN 在大多数数据集上优于 NGCF
```

## 5. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 64 | 嵌入维度 |
| n_layers | 3-4 | GNN 层数 |
| mess_dropout | 0.1 | 消息 dropout |
| learning_rate | 0.001 | 学习率 |
| reg_weight | 1e-5 | L2 正则化 |

## 6. 学习总结

### 6.1 核心要点

1. **图结构**: 将交互建模为二部图
2. **高阶交互**: 多层传播捕捉高阶关系
3. **嵌入聚合**: 聚合各层嵌入作为最终表示

### 6.2 优缺点

**优点:**
- 显式建模高阶协同信号
- 端到端学习

**缺点:**
- 参数量较大
- 训练相对慢

## 7. 练习题

1. 比较不同层数对效果的影响。

2. 实现 NGCF 的 mini-batch 训练。

3. 添加边特征（如时间）到 NGCF。
