# LINE（大规模信息网络嵌入）学习文档

> 针对大规模网络的图嵌入方法，同时保留一阶和二阶相似度

---

## 1. 算法基础认知

**一句话定义**：LINE（Large-scale Network Embedding）是由微软亚洲研究院的Tang等人于2015年提出的图嵌入算法，专门针对大规模网络设计，通过同时保留**一阶相似度**（直接连接的节点相似）和**二阶相似度**（邻居结构相似的节点相似）来学习节点的低维向量表示。

**直觉类比**：LINE就像社交网络中的"人以类聚"。想象你在社交网络中：你的好朋友（直接关联的人）和你一阶相似——这是"一阶相似度"；而那些和你有多少共同好友的人，和你也可能相似——这是"二阶相似度"。LINE同时考虑这两种相似性，既认识你的直接朋友，也认识和你"朋友的朋友"相似的人，这样就能更好地理解你在网络中的位置。

**历史背景**：
- 2015年，Microsoft research的Tang等人在论文"LINE: Large-scale Network Embedding"中提出
- 解决了传统图嵌入无法处理大规模网络的问题
- 成为图嵌入领域的里程碑方法

**算法定位**：
- 类型：图嵌入 → 无监督学习
- 输出：节点的低维向量表示
- 模型类型：大规模网络嵌入

**前置知识**：
- [必备]：图论基础（节点、边、邻接矩阵）
- [必备]：Skip-gram模型（因为LINE本质是Skip-gram的推广）

---

## 2. 核心原理

### 2.1 传统方法的局限

之前的方法如Isomap、MDS需要计算所有节点对距离，无法处理大规模网络。

DeepWalk使用随机游走，但只考虑了二阶相似度，没有保留直接连接的信息。

### 2.2 LINE的核心创新

同时保留两种相似性！

| 相似度 | 定义 | 对应方法 |
|--------|------|----------|
| 一阶相似度 | 直接连接的节点相似 | 共现概率 |
| 二阶相似度 | 共享相似邻居的节点相似 | Skip-gram |

```
一阶相似度：        二阶相似度：
A —— B          A —— B —— C —— D
（直接连接）         （邻居结构相似）

保留边 (A,B)       保留 B 的邻居 = {C,D}
```

### 2.3 整体流程

```
            输入大规模网络
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │           一阶相似度                │
    │    P(e) ∝ sim(u,v) = σ(·)         │
    │    优化：KL散度或负采样          │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │           二阶相似度                │
    │    P(v|c) ∝ Sim(c,v)            │
    │    优化：Skip-gram loss           │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │        拼接双重嵌入                  │
    │    [u] = [u₁ || u₂]              │
    └─────────────────┬─────────────────┘
                       │
                       ▼
                   节点嵌入向量
```

---

## 3. 数学公式与推导

### 3.1 一阶相似度

**目标**：直接相连的节点有更高的���似度

**定义边概率**：

$$P(e_{ij}) = \frac{1}{1 + \exp(-\vec{u_i}^T \cdot \vec{u_j})}$$

其中 $\vec{u_i}$ 是节点 i 的嵌入向量。

**损失函数**：

对于无向边集合 E：

$$L_1 = - \sum_{(i,j) \in E} \log P(e_{ij})$$

使用负采样简化：

$$L_1 = - \sum_{(i,j) \in E} \left( \log \sigma(\vec{u_i}^T \cdot \vec{u_j}) + \sum_{k=1}^{K} \log \sigma(-\vec{u_i}^T \cdot \vec{u_k}) \right)$$

其中 K 是负样本数量。

### 3.2 二阶相似度

**核心思想**：如果两个节点的邻居相似，则它们相似。

用条件概率定义：$P(v_j | u_i)$ 表示节点 i 的上下文 j 的概率。

**定义**：

$$P(v_j | u_i) = \frac{\exp(\vec{u_j}^T \cdot \vec{c_i})}{\sum_{k=1}^{|V|} \exp(\vec{u_k}^T \cdot \vec{c_i})}$$

其中 $\vec{c_i}$ 是节点的"上下文"向量（类似Skip-gram的中心词和上下文）。

**损失函数**：

对于每个节点及其邻居：

$$L_2 = \sum_{i \in V} \lambda_i \cdot KL(P(\cdot | u_i) || Q(\cdot | u_i))$$

简化为Skip-gram形式：

$$L_2 = \sum_{(i,j) \in E} \log \sigma(\vec{c_j}^T \cdot \vec{u_i}) + \sum_{k=1}^{K} \log \sigma(-\vec{c_k}^T \cdot \vec{u_i})$$

### 3.3 双重嵌入结合

**最终目标函数**：

$$L = L_1 + \alpha \cdot L_2$$

其中 $\alpha$ 是平衡两个目标的权重（通常取0.01-0.1）。

**最终嵌入**：

$$\vec{z_i} = [\vec{u_i} || \vec{c_i}]$$

拼接一阶和二阶的嵌入！

### 3.4 边的采样策略

由于网络边数可能非常多，LINE使用**边采样**来加速：

- 按边权分布采样（有权图）
- 按度分布采样（无权图）

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       输入网络
           │
           ▼
    ┌───────────────┐
    │ 采样边       │ ← 边采样策略
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 一阶优化    │ ← 优化L1
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 二阶优化    │ ← 优化L2
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 拼接嵌入    │ ← 最终向量
    └───────────────┘
```

### 4.2 负采样

LINE使用的负采样：

```python
# 伪代码
# 对于每个正样本，采样K个负样本
for (i,j) in positive_edges:
    # 正样本
    positive_score = sigmoid(u_i · u_j)
    
    # 负样本
    for _ in range(K):
        k = sample_negative_vertex()
        negative_score = sigmoid(u_i · u_k)
```

### 4.3 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| embedding_size | 128-256 | 嵌入维度 |
| order | 1, 2, 或两者 | 保留的阶数 |
| negative_samples | 5 | 负采样数 |
| batch_size | 1000 | 批大小 |
| learning_rate | 0.025 | 学习率 |

### 4.4 训练技巧

| 技巧 | 说明 |
|------|------|
| 边采样 | 按度分布采样 |
| 学习率衰减 | 防止震荡 |
| 并行训练 | 多线程处理边 |

---

## 5. 应用场景

### 5.1 节点分类

```python
# 节点分类任务
# 1. 学习嵌入
# 2. 用嵌入特征训练分类器
```

### 5.2 链接预测

```python
# 链接预测
# 预测边的概率 = sigmoid(z_u · z_v)
```

### 5.3 可视化

```
# t-SNE可视化
# 高维嵌入 -> 2维可视化
```

### 5.4 推荐系统

- 用户-商品交互网络嵌入
- 协同过滤的向量化

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **可扩展** | 处理大规模网络 |
| **效率高** | 边采样+负采样 |
| **两种相似度** | 综合保留结构信息 |
| **无需随机游走** | 更稳定 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **无权图不完美** | 需要改进处理 |
| **只考虑直接邻居** | 丢失远距离信息 |
| **需要大内存** | 存储嵌入 |

### 6.3 改进方案

| 改进 | 方法 |
|------|------|
| 高阶近似 | 使用DeepWalk的随机游走 |
| 异构图 | Node2Vec的bFS/DFS |
| 动态图 | Time-varying网络 |

---

## 7. 调库实现

### 7.1 GEM实现

```python
# 安装
# pip install gem

from gem.embedding import LINE

# 创建模型
model = LINE(
    d=128,                 # 嵌入维度
    order=2,               # 1: 一阶, 2: 二阶, 3: 两者
    max_iter=50,            # 最大迭代
    learning_rate=0.025,   # 学习率
    negative_samples=5,    # 负采样
    verbose=True
)

# 训练（输入networkx图）
import networkx as nx
G = nx.fast_gnp_random_graph(1000, 0.01)  # 随机图
model.fit(G)

# 获取嵌入
embeddings = model.get_embedding()
print(embeddings.shape)  # (1000, 128)
```

### 7.2 PyTorch实现

```python
import torch
import torch.nn as nn
import numpy as np


class LINEModel(nn.Module):
    """LINE模型"""
    
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # 一阶嵌入
        self.first_order = nn.Embedding(num_nodes, embedding_dim)
        
        # 二阶嵌入（中心+上下文）
        self.center = nn.Embedding(num_nodes, embedding_dim)
        self.context = nn.Embedding(num_nodes, embedding_dim)
        
    def forward_first_order(self, edge_idx):
        """一阶损失"""
        node_u = edge_idx[:, 0]
        node_v = edge_idx[:, 1]
        
        u_embed = self.first_order(node_u)
        v_embed = self.first_order(node_v)
        
        return torch.sum(u_embed * v_embed, dim=1)
        
    def forward_second_order(self, center_idx, context_idx):
        """二阶损失"""
        center_embed = self.center(center_idx)
        context_embed = self.context(context_idx)
        
        return torch.sum(center_embed * context_embed, dim=1)


class LINE Trainer:
    """LINE训练器"""
    
    def __init__(self, num_nodes, embedding_dim=128, order=2):
        self.model = LINEModel(num_nodes, embedding_dim)
        self.order = order
        
    def train(self, edges, epochs=10, lr=0.025, negatives=5, batch_size=1000):
        """训练"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            # 随机打乱边
            np.random.shuffle(edges)
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]
                
                if len(batch) == 0:
                    break
                
                edge_idx = torch.tensor(batch, dtype=torch.long)
                
                # 正样本
                if self.order in [1, 3]:
                    pos_scores = self.model.forward_first_order(edge_idx)
                    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pos_scores, torch.ones_like(pos_scores)
                    )
                    loss = pos_loss
                    
                if self.order in [2, 3]:
                    # 二阶
                    center = edge_idx[:, 0]
                    context = edge_idx[:, 1]
                    pos_scores = self.model.forward_second_order(center, context)
                    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pos_scores, torch.ones_like(pos_scores)
                    )
                    loss = loss + pos_loss if self.order == 3 else pos_loss
                
                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        return self.model
    
    def get_embedding(self):
        """获取嵌入"""
        if self.order == 1:
            return self.model.first_order.weight.detach().numpy()
        elif self.order == 2:
            return self.model.center.weight.detach().numpy()
        else:
            # 拼接
            return torch.cat([
                self.model.first_order.weight,
                self.model.center.weight
            ], dim=1).detach().numpy()


def demo():
    """演示"""
    import networkx as nx
    
    # 创建图
    G = nx.fast_gnp_random_graph(100, 0.1, seed=42)
    edges = list(G.edges())
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    # 训练
    trainer = LINE Trainer(100, embedding_dim=32, order=3)
    trainer.train(edges, epochs=5)
    
    # 嵌入
    emb = trainer.get_embedding()
    print(f"嵌入形状: {emb.shape}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

### 8.1 完整LINE实现

```python
import numpy as np
from collections import defaultdict
import random


class LINE:
    """完整LINE实现"""
    
    def __init__(self, num_nodes, embedding_dim=128, order=2, 
                 learning_rate=0.025, negative_samples=5):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order
        self.lr = learning_rate
        self.neg = negative_samples
        
        # 初始化嵌入
        self.first_order_emb = (np.random.randn(num_nodes, embedding_dim) * 0.1).astype(np.float32)
        self.second_order_center = (np.random.randn(num_nodes, embedding_dim) * 0.1).astype(np.float32)
        self.second_order_context = (np.random.randn(num_nodes, embedding_dim) * 0.1).astype(np.float32)
        
        # 节点度（用于采样）
        self.degrees = np.zeros(num_nodes)
        
    def fit(self, edges, epochs=5):
        """训练LINE"""
        
        # 统计度
        for u, v in edges:
            self.degrees[u] += 1
            self.degrees[v] += 1
            
        # 计算边权（用于采样）
        edge_weights = np.array([self.degrees[u] + self.degrees[v] for u, v in edges])
        edge_weights = edge_weights / edge_weights.sum()
        
        # 训练
        for epoch in range(epochs):
            total_loss = 0
            
            for _ in range(len(edges)):
                # 采样边
                idx = np.random.choice(len(edges), p=edge_weights)
                u, v = edges[idx]
                
                # 一阶
                if self.order in [1, 3]:
                    loss = self._train_first_order(u, v)
                    total_loss += loss
                    
                # 二阶
                if self.order in [2, 3]:
                    loss = self._train_second_order(u, v)
                    total_loss += loss
            
            print(f"Epoch {epoch}, Loss: {total_loss / len(edges):.4f}")
        
        return self
    
    def _train_first_order(self, u, v):
        """一阶训练"""
        # 正样本
        score = np.dot(self.first_order_emb[u], self.first_order_emb[v])
        sigmoid = 1 / (1 + np.exp(-score))
        loss = -np.log(sigmoid + 1e-10)
        
        # 更新
        grad = (sigmoid - 1) * self.lr
        self.first_order_emb[u] -= grad * self.first_order_emb[v]
        self.first_order_emb[v] -= grad * self.first_order_emb[u]
        
        # 负采样
        for _ in range(self.neg):
            k = np.random.randint(0, self.num_nodes)
            score = np.dot(self.first_order_emb[u], self.first_order_emb[k])
            sigmoid = 1 / (1 + np.exp(-score))
            loss -= np.log(1 - sigmoid + 1e-10)
            
            grad = sigmoid * self.lr
            self.first_order_emb[u] -= grad * self.first_order_emb[k]
        
        return loss
    
    def _train_second_order(self, u, v):
        """二阶训练"""
        # u是中心，v是上下文
        score = np.dot(self.second_order_center[u], self.second_order_context[v])
        sigmoid = 1 / (1 + np.exp(-score))
        loss = -np.log(sigmoid + 1e-10)
        
        # 更新
        grad = (sigmoid - 1) * self.lr
        self.second_order_center[u] -= grad * self.second_order_context[v]
        self.second_order_context[v] -= grad * self.second_order_center[u]
        
        # 负采样
        for _ in range(self.neg):
            k = np.random.randint(0, self.num_nodes)
            score = np.dot(self.second_order_center[u], self.second_order_context[k])
            sigmoid = 1 / (1 + np.exp(-score))
            loss -= np.log(1 - sigmoid + 1e-10)
            
            grad = sigmoid * self.lr
            self.second_order_center[u] -= grad * self.second_order_context[k]
        
        return loss
    
    def get_embedding(self):
        """获取嵌入"""
        if self.order == 1:
            return self.first_order_emb
        elif self.order == 2:
            return self.second_order_center
        else:
            return np.concatenate([self.first_order_emb, self.second_order_center], axis=1)


def line_demo():
    """演示"""
    import networkx as nx
    
    # 创建图
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    edges = list(G.edges())
    
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    
    # 训练
    model = LINE(100, embedding_dim=32, order=3)
    model.fit(edges, epochs=5)
    
    # 嵌入
    emb = model.get_embedding()
    print(f"嵌入形状: {emb.shape}")


if __name__ == "__main__":
    line_demo()
```

---

## 9. 可视化与结果理解

### 9.1 嵌入可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embedding(embeddings, labels):
    """可视化嵌入"""
    
    # t-SNE降维
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10')
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估任务

| 任务 | 指标 |
|------|------|
| 节点分类 | Accuracy, F1 |
| 链接预测 | AUC, AP |
| 可视化 | t-SNE |

### 10.2 对比方法

| 方法 | 说明 |
|------|------|
| DeepWalk | 随机游走+Skip-gram |
| Node2Vec | 有偏随机游走 |
| LINE | 直接优化一阶+二阶 |

---

## 11. 常见问题与易错点

### 11.1 只用一阶

**问题**：只用一阶会丢失远距离信息

**解决**：使用order=3，同时用一阶和二阶

### 11.2 负采样数

**问题**：负采样的选择影响结果

**解决**：通常5-10个负样本足够

### 11.3 嵌入维度

**问题**：太大/太小

**解决**：128是常用的默认值

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 保���一阶+二阶相似度 |
| 一阶 | 直接连接的边 |
| 二阶 | 相同邻居的结构 |
| 优化 | 负采样Skip-gram |

### 12.2 公式记忆

**一阶损失**：
$$L_1 = \sum_{(i,j) \in E} \log \sigma(\vec{u_i}^T \vec{u_j})$$

**二阶损失**：
$$L_2 = \sum_{(i,j) \in E} \log \sigma(\vec{c_j}^T \vec{u_i})$$

### 12.3 扩展阅读

| 方法 | 年份 | 贡献 |
|------|------|------|
| DeepWalk | 2014 | 随机游走 |
| Node2Vec | 2016 | 有偏游走 |
| LINE | 2015 | 一阶+二阶 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：什么是一阶相似度？

**答案**：直接相连的节点在嵌入空间中应该相似。保留边的信息。

**练习2**：为什么需要二阶相似度？

**答案**：有些节点虽未直接相连，但可能有相似的邻居结构，二阶可以捕获这种相似性。

### 13.2 进阶思考

**思考1**：LINE和DeepWalk的区别？

**答案**：DeepWalk使用随机游走定义"上下文"，LINE直接用邻居定义。可以认为是随机游走的极限情况。

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 图嵌入基础 | 理解表示学习 |
| 3-4 | Skip-gram | 理解Word2Vec |
| 5-6 | LINE原理 | 理解一阶+二阶 |
| 7 | 代码 | 跑通demo |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 完整实现 | LINE代码 |
| 2 | 对比实验 | vs DeepWalk |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| LINE原始论文 | https://arxiv.org/abs/1503.03578 |
| GEM库 | https://github.com/palash1992/GEM |

### B. 参数速查

| 参数 | 默认值 |
|------|--------|
| embedding_size | 128 |
| order | 2 |
| negative_samples | 5 |

---

**文档结束**