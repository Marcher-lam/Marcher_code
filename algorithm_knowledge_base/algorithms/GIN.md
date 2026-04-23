# GIN（图同构网络）学习文档

> 图神经网络中表达能力最强的变体，通过可学习的聚合函数实现完美的图同构测试

---

## 1. 算法基础认知

**一句话定义**：GIN是图同构网络(Graph Isomorphism Network)，通过使用Sum聚合和超参数ε实现比WL测试更强的图同构区分能力。

**直觉类比**：GIN就像一个"顶级鉴定师"——它不仅能看出两张图是不是完全一样的，还能分辨出"双胞胎兄弟"和"三胞胎兄弟"之间的细微差别。传统GNN可能把它们都看成是一样的，GIN则能精确区分。

**历史背景**：2019年，Keyulu Xu等人在论文"How Powerful are Graph Neural Networks?"中证明WL测试是GNN的表达上界，并提出GIN作为首个突破这个限制的GNN变体。

**算法定位**：
- 类型：图神经网络 → 图分类/节点分类
- 输出：图/节点嵌入
- 模型类型：消息传递神经网络

**前置知识**：
- [必备]：图论基础（节点、边、邻接矩阵）
- [必备]：神经网络基础
- [扩展]：GCN、GAT、MpNN

---

## 2. 核心原理

### 2.1 核心思想

GIN的核心创新是**可学习的聚合函数**：

$$h_v^{(k)} = \text{MLP}^{(k)}\left((1+\varepsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)$$

关键在于使用Sum而不是Mean/Pooling，这使得GIN具有最强的表达能力。

核心思想可以概括为：**让神经网络自己学会最合适的聚合方式**。

### 2.2 工作流程

1. **初始化**：每个节点的特征
2. **消息传递**：k层邻居聚合
3. **READOUT**：图级别池化
4. **分类/回归**

### 2.3 关键概念

- **Weisfeiler-Lehman (WL)测试**：图同构的经典检验算法
- **Sum聚合**：区分多重集（multiset）
- **可学习参数ε**：控制自环重要性

---

## 3. 数学公式

### 3.1 节点更新

$$h_v^{(k)} = \text{MLP}^{(k)}\left((1+\varepsilon) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)$$

### 3.2 图嵌入

$$h_G = \text{READOUT}(\{h_v^{(K)} | v \in G\})$$

### 3.3 WL测试

迭代更新着色：$c_v^{(t+1)} = \text{HASH}(c_v^{(t)}, \{\{c_u^{(t)} | u \in N(v)}\}})$如果两图的WL着色序列不同，则不同构。

---

## 4. 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    """GIN卷积层"""
    
    def __init__(self, in_channels, out_channels, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index):
        # x: (num_nodes, in_channels)
        # edge_index: (2, num_edges)
        
        # 邻居聚合
        src, dst = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, src, x[dst])
        
        # GIN更新
        x = (1 + self.eps) * x + aggregated
        x = self.mlp(x)
        
        return x


class GIN(nn.Module):
    """GIN模型"""
    
    def __init__(self, num_features, num_classes, num_layers=3, hidden=64):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = num_features if i == 0 else hidden
            self.convs.append(GINConv(in_ch, hidden))
        
        self.fc = nn.Linear(hidden, num_classes)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # 简单READOUT：求和
        graph_embedding = x.sum(dim=0, keepdim=True)
        
        return self.fc(graph_embedding)
```

---

## 5. 应用

### 5.1 应用领域

- 化学分子属性预测
- 社交网络分析
- 蛋白质相互作用
- 代码bug检测

### 5.2 同类对比

| 方法 | 表达能力 | 计算复杂度 |
|------|----------|------------|
| GIN | 最强(WL+) | 中 |
| GCN | WL | 低 |
| GAT | < WL | 中 |

---

## 6. 练习

**问题**：为什么Sum比Mean更适合区分图？

答案：Sum能保留节点度的信息，Mean会丢失。

---

## 7. 学习路径

### 7.1 前置

- [ ] 图论
- [ ] GNN基础

### 7.2 进阶

- [ ] WL测试
- [ ] 更多的GNN变体

---

## 附录

### A. 代码

见第4节。

### B. 参考文献

1. Xu et al., "How Powerful are Graph Neural Networks?", 2019

---

**文档结束**