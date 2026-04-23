# LightGCN 学习文档

> 图卷积网络的轻量化变体，移除特征变换，仅保留邻居聚合

---

## 1. 算法基础认知

**一句话定义**：LightGCN是简化版的GCN，通过移除GCN中的特征变换（线性变换），只保留关键的邻居聚合操作，大幅减少参数和计算量。

**直觉类比**：LightGCN就像简化版的"社交网络分析"——传统GCN就像在分析社交网络时，不仅看谁和朋友在互动，还要分析每个人的"社交能力"（特征变换）。LightGCN认为这不重要，重要的是"你和谁在联系"，所以它只保留邻居聚合这一步。

**历史背景**：2020年，He等人提出LightGCN，在推荐系统（矩阵分解任务）中大幅超越GCN，同时参数减少80%。

---

## 2. 核心原理

### 2.1 核心创新

移除GCN中的特征变换矩阵W，只保留：

$$H^{(k+1)} = D^{-1/2} A D^{-1/2} H^{(k)}$$

简化后公式：
$$H^{(k+1)} = (D+I)^{-1/2} (A+I) (D+I)^{-1/2} H^{(k)}$$

### 2.2 为什么work

- 推荐系统中，用户交互模式比用户特征更重要
- 去掉noise参数，提升泛化

---

## 3. 实现

```python
import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_users + num_items, emb_dim)
        
    def forward(self, adj):
        embeddings = self.embedding.weight
        
        all_embeddings = [embeddings]
        
        for _ in range(self.num_layers):
            embeddings = adj @ embeddings
            all_embeddings.append(embeddings)
        
        return torch.stack(all_embeddings).mean(0)
```

---

## 4. 应用

- 推荐系统
- 链接预测

---

## 5. 练习

**问题**：GCN和LightGCN的区别？

答案：GCN有特征变换矩阵W，LightGCN没有。

---

**文档结束**