# NGCF（神经图协同过滤）学习文档

> 基于图神经网络的推荐系统，将用户-项目交互建模为图结构

---

## 1. 算法基础认知

**一句话定义**：NGCF（Neural Graph Collaborative Filtering，神经图协同过滤）是由Wang等人于2019年提出的推荐系统模型，将用户-项目交互建模为二部图，利用图神经网络在交互图上传播嵌入，捕捉高阶协同过滤信号。

**直觉类比**：NGCF就像在社交网络中"口口相传"。假设你想找餐厅，你的五个朋友都推荐了同一家店——这家店可能真的很好。NGCF做的就是这个：它把用户和项目（比如餐厅）都看成图上的节点，你和朋友的连接表示你们"交互过"。通过在图上传播信息，朋友的偏好会影响你的推荐。这就是"协同过滤"的图神经网络实现。

**历史背景**：
- 2019年，Wang等人在WWW会议发表"NGCF: Neural Graph Collaborative Filtering"
- 将GCN引入推荐系统
- 后续演化出LightGCN、GRAFR等

**核心定位**：
- 类型：推荐系统 → 图神经网络
- 输出：用户-项目交互预测
- 模型类型：GCN + 嵌入

**前置知识**：
- [必备]：GCN（图卷积网络）
- [必备]：协同过滤
- [推荐]：矩阵分解

---

## 2. 核心原理

### 2.1 传统协同过滤的问题

| 方法 | 问题 |
|------|------|
| 矩阵分解 | 无法捕捉高阶关系 |
| SVD++ | 稀疏数据上效果差 |
| NeuMF | 只用直接邻居 |

**核心局限**：只考虑直接交互，忽略"朋友的朋友"关系。

### 2.2 NGCF核心思想

**突破**：用GCN在交互图上传播嵌入！

```
用户-项目交互图
    Alice ──▶ 电影A
    │           
    ▼           
    Bob ──▶ 电影B
    │           
    ▼           
    Carol ──▶ 电影C
    
传统方法：A和C没直接关系
NGCF：通过Bob传播，A和C建立联系！
```

### 2.3 架构

```
    用户嵌入矩阵 U   项目嵌入矩阵 V
         │                │
         ▼                ▼
    ┌────────┐       ┌────────┐
    │ 初始化 │       │ 初始化 │
    └────┬───┘       └────┬───┘
         │                │
         ▼                ▼
    ┌────────────────────────────────────┐
    │      图卷积层 (3层)                  │
    │  1. 消息构建                       │
    │  2. 消息聚合                      │
    │  3. 嵌入更新                      │
    └────────────┬─────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │      输出层                         │
    │  拼接 + MLP / 内积                   │
    └────────────┬─────────────────────┘
                 │
                 ▼
            预测分数
```

---

## 3. 数学公式与推导

### 3.1 图构建

交互矩阵 R ∈ ℝ^(M×N)，M用户N项目：

$$R_{ui} = 1 \quad 如果用户u交互过项目i$$

邻接矩阵 A ∈ ℝ^(M+N)：

$$A = \begin{pmatrix} 0 & R \\ R^T & 0 \end{pmatrix}$$

### 3.2 消息构建

对边(u,v)，消息m为：

$$m_{u \leftarrow v} = \frac{1}{\sqrt{d_u d_v}} \cdot (W_1 \cdot e_v + W_2 \cdot (e_v \odot e_u))$$

其中：
- $e_u, e_v$：节点嵌入
- $d_u$：度
- $W_1, W_2$：可学习权重
- $\odot$：元素积（product）

### 3.3 消息聚合

$$e_u^{(l+1)} = \text{LeakyReLU}( \sum_{v \in \mathcal{N}_u} m_{u \leftarrow v})$$

使用LeakyReLU保持负信号。

### 3.4 多层传播

$$E^{(l+1)} = \bar{A} \cdot E^{(l)} \cdot W^{(l)}$$

其中 $\bar{A}$ 是归一化邻接矩阵。

---

## 4. 训练过程

### 4.1 训练流程

```
    构建交互图
         │
         ▼
    初始化嵌入
         │
         ▼
    ┌───────────────┐
    │ 图卷积传播   │ ← 3层
    └───────┬───────┘
         │
         ▼
    ┌───────────────┐
    │ 输出预测    │
    └───────┬───────┘
         │
         ▼
    ┌───────────────┐
    │ BPR损失     │ ← 优化
    └───────────────┘
```

### 4.2 损失函数

BPR（贝叶斯个性化排序）损失：

$$L_{BPR} = \sum_{(u,i,j)}- \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

$$\hat{y}_{ui} = f(e_u, e_i) = e_u^T \cdot e_i$$

加L2正则：

$$L_{total} = L_{BPR} + \lambda \|E\|^2$$

### 4.3 参数

| 参数 | 典型值 |
|------|--------|
| 嵌入维度 | 64 |
| 层数 | 3 |
| 学习率 | 0.001 |
| batch | 1024 |
| 正则化 | 1e-5 |

---

## 5. 应用场景

### 5.1 推荐系统

- 电影推荐（MovieLens）
- 商品推荐（Amazon）
- 音乐推荐（Spotify）

### 5.2 链接预测

- 朋友推荐
- 兴趣小组

---

## 6. 优缺点

### 6.1 优点

| 优点 |
|------|
| 高阶关系捕捉 |
| 稀疏数据友好 |
| 端到端 |

### 6.2 缺点

| 缺点 |
|------|
| 计算重 |
| 内存大 |
| 层数敏感 |

### 6.3 改进

- LightGCN
- NGCF+
- GRAFR

---

## 7. 调库实现

### 7.1 PyTorch Geometric

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class NGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 计算度
        row, col = edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
        
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class NGCF(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers=3):
        super().__init__()
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([
            NGCFConv(embedding_dim, embedding_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index):
        xs = [self.embedding(x)]
        
        for conv in self.convs:
            xs.append(conv(xs[-1], edge_index))
        
        # 拼接所有层
        return torch.cat(xs, dim=-1)
```

### 7.2 推荐库的实现

```python
# 使用RecBole
from recbole.model.sequential_recommender import NGCF

model = NGCF(
    config_dict={
        'embedding_size': 64,
        'n_layers': 3,
        'learning_rate': 0.001
    }
)
```

---

## 8. 手工实现

### 8.1 核心实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCFModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # 初始化嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # GCN层
        self.gcns = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(embed_dim * num_layers, 1)
        
    def forward(self, user_ids, item_ids, adj):
        # 嵌入
        u_embed = self.user_embedding(user_ids)
        i_embed = self.item_embedding(item_ids)
        
        # GCN传播
        all_embeddings = [u_embed, i_embed]
        
        for gcn in self.gcns:
            # 消息传递（简化）
            new_embed = gcn(adj)
            all_embeddings.append(new_embed)
        
        # 最后一层
        u_final = all_embeddings[-1]
        i_final = all_embeddings[-1]
        
        # 预测
        score = (u_final * i_final).sum(dim=-1)
        
        return torch.sigmoid(score)


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
```

### 8.2 训练示例

```python
def train_ngcf():
    """训练NGCF"""
    # 参数
    num_users = 1000
    num_items = 500
    embed_dim = 64
    
    model = NGCFModel(num_users, num_items, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟数据
    user_ids = torch.randint(0, num_users, (256,))
    item_pos = torch.randint(0, num_items, (256,))
    item_neg = torch.randint(0, num_items, (256,))
    
    adj = torch.randn(num_users + num_items, embed_dim) * 0.1
    
    # 训练
    for epoch in range(10):
        pos_score = model(user_ids, item_pos, adj)
        neg_score = model(user_ids, item_neg, adj)
        
        loss = bpr_loss(pos_score, neg_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: {loss.item():.4f}")


if __name__ == "__main__":
    train_ngcf()
```

---

## 9. 可视化与评估

### 9.1 嵌入可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels)
    plt.show()
```

### 9.2 评估指标

| 指标 | 说明 |
|------|------|
| Recall@K | Top-K召回 |
| NDCG@K | 排序质量 |
| AUC | 二分类 |

---

## 10. 常见问题与技巧

### 10.1 过平滑

问题：层数过多导致嵌入趋同

解决：3层足够

### 10.2 稀疏图

解决：邻居采样

---

## 11. 学习总结

### 11.1 核心要点

- GCN + 协同过滤
- 高阶关系
- 端到端学习

### 11.2 扩展

- LightGCN
- DeepGCN

---

## 12. 练习题

1. NGCF和GCN的区别？
2. 为什么需要多层？

---

## 13. 学习路径

1. GCN基础
2. 协同过滤
3. NGCF
4. 实战

---

## 附录

### 参考

- 论文：Wang et al., WWW 2019
- 库：PyG, RecBole

---

**文档结束**

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class NGCFNet(nn.Module):
    def __init__(self, dim_in=20, dim_h=64, dim_out=2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim_in, dim_h), nn.Linear(dim_h, dim_out)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

def train(model, X, y, epochs=100, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"Ep{ep+1} loss={loss.item():.4f}")

m = NGCFNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：NGCF与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('NGCF Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估

