# GAT 图注意力网络 学习文档

> 图注意力网络（Graph Attention Network），将注意力机制引入图神经网络，自适应学习邻居重要性。

---

## 1. 算法基础认知

### 1.1 什么是GAT？

GAT是一种将注意力机制（Attention Mechanism）融入图神经网络的新型架构。与GCN中对所有邻居使用固定且相同的聚合权重不同，GAT能够为每个邻居动态分配不同的注意力权重，从而自适应地捕获节点间的重要性差异。这使得模型能够区分"重要"和"不重要"的邻居，在处理异构图或带权图时表现更优。

### 1.2 历史背景与发展

- **2017年**：Veličković等人提出GAT，发表于ICLR 2018
- **核心创新**：将Transformer中的自注意力机制推广到图结构数据
- **与Transformer的关系**：GAT中的"多头注意力"直接启发了后续的Transformer架构
- **后续发展**：GATv2、Transformer for Graphs等

### 1.3 核心定位

GAT是第一个将注意力机制成功应用于图数据的模型，属于空域方法：
- vs GCN：GAT能自适应学习邻居权重，GCN权重固定
- vs GraphSAGE：GAT通过注意力而非采样处理邻居
- vs Transformer：都是注意力机制，但处理结构不同

---

## 2. 核心原理

### 2.1 注意力机制概述

GAT的核心是计算节点与其邻居之间的注意力系数：
$$e_{ij} = a(Wh_i, Wh_j)$$

其中：
- h_i：节点i的特征向量（维度为F'）
- W：可训练的线性变换矩阵（F×F'）
- a：单层感知机（attention function）
- e_ij：节点j对节点i的注意力得分

### 2.2 LeakyReLU激活

为保证注意力有正有负（便于学习不重要的邻居），使用LeakyReLU：
$$LeakyReLU(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}$$

负斜率设置为0.01（默认），也称为α=0.01。

### 2.3 注意力系数归一化

使用Softmax归一化得到最终注意力权重：
$$\alpha_{ij} = \frac{exp(LeakyReLU(e_{ij}))}{\sum_{k \in \mathcal{N}(i)} exp(LeakyReLU(e_{ik}))}$$

其中$\mathcal{N}(i)$包括节点i本身（可选）。

### 2.4 邻居聚合

使用注意力权重对邻居特征进行加权聚合：
$$h_i' = \sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} Wh_j)$$

其中σ是激活函数（通常为ELU）。

### 2.5 多头注意力

为稳定训练过程并捕获多方面的特征，GAT使用多头注意力（Multi-Head Attention）：
$$\hat{h}_i = \|_{k=1}^{K} \sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j)$$

其中K是注意力头数，$||$表示拼接。

最终输出可以使用拼接或平均：
- 中间层：使用拼接$hconcat$
- 输出层：使用平均$hmean$

---

## 3. 数学公式与推导

### 3.1 单头注意力推导

**步骤1：特征变换**
假设输入特征h_i维度为F，经过W变换后维度为F'：
$$h_i' = Wh_i \quad W \in \mathbb{R}^{F' \times F}$$

**步骤2：计算原始注意力得分**
通过单层感知机计算：
$$e_{ij} = a(Wh_i, Wh_j) = LeakyReLU(w^T [Wh_i || Wh_j])$$

其中w是维度为2F'的可学习向量，||表示拼接。

**步骤3：Softmax归一化**
$$\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} exp(e_{ik})} = softmax_j(e_{ij})$$

**步骤4：加权聚合**
$$h_i^{new} = \sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} Wh_j)$$

### 3.2 多头注意力

假设有K个注意力头，每个头有独立的参数：
- $W^{(k)}$：第k个头的变换矩阵
- $a^{(k)}$：第k个头的注意力函数
- $\alpha_{ij}^{(k)}$：第k个头计算的关注权重

**K头拼接（中间层）：**
$$h_i^{new} = \|{k=1}^{K} \sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j)$$

输出维度：K × F'

**K头平均（输出层）：**
$$h_i^{new} = \sigma(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j)$$

输出维度：F'

### 3.4 扩展公式补充

**注意力机制的对称性分析**
令$e_{ij} = a(Wh_i, Wh_j)$和$e_{ji} = a(Wh_j, Wh_i)$。

如果$a$是对称的（$a(x,y) = a(y,x)$），则：
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})} = \frac{\exp(e_{ji})}{\sum_k \exp(e_{ik})} = \alpha_{ji}$$

即$\alpha_{ij} = \alpha_{ji}$（对无权图）。

GAT默认使用不对称$a$（$a(Wh_i, Wh_j) \neq a(Wh_j, Wh_i)$），这适合有向图。

**与GCN的关系**
当注意力权重统一为$\alpha_{ij} = 1/\sqrt{d_i d_j}$时，GAT退化为GCN：
$$h_i' = \sigma\left(\sum_{j \in N(i)} \frac{1}{\sqrt{d_i d_j}} Wh_j\right)$$

这说明GCN是GAT的特例（固定注意力）。

**注意力权重的正则化**
为防止注意力权重过于sharp（集中在一个邻居），可使用温度调节：
$$\alpha_{ij} = \frac{\exp(e_{ij}/T)}{\sum_k \exp(e_{ik}/T)}$$

$T > 1$使分布更平滑，$T \to 0$使分布更sharp。

**GATv2的定义**
GATv2改进了原始GAT的注意力计算：
$$e_{ij} = w^T \tanh(W h_i + W h_j)$$

使用可学习的线性变换和双曲正切激活，提供更强的表达能力。

### 3.3 与GCN的对比

| 方面 | GCN | GAT |
|------|-----|-----|
| 邻居权重 | 固定（归一化度倒数） | 自适应学习 |
| 邻居重要性 | 假设相同 | 区分不同 |
| 计算复杂度 | O(N×E×F) | O(N×E×F + N×E) |
| 参数量 | 较少 | 较多（K个w） |

数学上，GCN可以视为GAT的特例：
- GCN: $\alpha_{ij} = d_i^{-1/2} d_j^{-1/2}$
- GAT: $\alpha_{ij}$通过学习获得

### 3.4 ELU激活函数

GAT默认使用ELU（Exponential Linear Unit）：
$$ELU(x) = \begin{cases} x & x > 0 \\ e^x - 1 & x \leq 0 \end{cases}$$

相比ReLU，ELU在负区间也有输出，有助于输出分布接近0均值。

---

## 4. 训练过程讲解

### 4.1 训练流程

GAT的训练流程与GCN基本相同，核心区别在于注意力权重的计算：

**数据准备**
```python
# 使用PyG加载数据
from torch_geometric.datasets import Cora
dataset = Cora()
data = dataset[0]
```

**模型定义**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 heads=8, dropout=0.5):
        super(GAT, self).__init__()
        # 第一层GATConv：多头注意力，输出拼接
        self.gat1 = GATConv(in_channels, hidden_channels, 
                           heads=heads, dropout=dropout)
        # 第二层GATConv：单头，输出类别数
        self.gat2 = GATConv(hidden_channels * heads, out_channels,
                           heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # 第一层 + 激活
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层（无激活，由loss处理）
        x = self.gat2(x, edge_index)
        return x
```

**损失函数**
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(pred[train_mask], labels[train_mask])
```

**优化器**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
```

### 4.2 关键超参数

**注意力头数（heads）**
- 推荐值：4-8
- 过多：计算量大，容易过拟合
- 过少：注意力表达能力不足

**隐藏维度**
- 单头输出维度：hidden_channels
- 多头拼接后：heads × hidden_channels

**Dropout**
- 默认0.5，防止过拟合
- 注意力权重也应用dropout

**学习率**
- GAT通常比GCN需要更小的学习率
- 推荐：0.001-0.01

### 4.3 训练技巧

**学习率调度**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=50, factor=0.95
)
```

**早停**
```python
best_val_acc = 0
patience = 100
counter = 0

for epoch in range(500):
    # 训练...
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"早停于第{epoch}轮")
            break
```

---

## 5. 应用场景

### 5.1 节点分类

**引文网络分类**
- Cora、CiteSeer论文分类
- GAT通常比GCN精度提升1-3%

**社交网络**
- 用户画像/兴趣分类
- 欺诈检测

### 5.2 图分类

**分子性质预测**
- 药物分子属性
- 蛋白质功能预测

### 5.3 其他应用

- **推荐系统**：用户-商品交互图
- **交通预测**：道路网络
- **知识图谱**：实体��类

---

## 6. 优缺点分析

### 6.1 优点

**自适应权重**
- 自动学习邻居重要性
- 无需手动设计聚合权重

**并行效率高**
- 注意力计算可并行
- 比需要特征分解的GCN更快

**表达能力更强**
- 可处理异构图
- 可处理带权图

**可解释性**
- 注意力权重可解释
- 可可视化节点关系

### 6.2 缺点

**计算开销**
- 需要计算所有边的注意力
- O(N×E×K) vs GCN的O(N×E)

**参数量大**
- K个注意力头有K组参数
- 比GCN参数量大K倍

**训练不稳定**
- 注意力值可能极值化
- 需要更小的学习率

**只处理无向图**
- 默认处理无向图
- 有向图需特殊处理

### 6.3 改进方向

| 问题 | 改进 |
|------|------|
| 计算开销 | GATv2（高效注意力） |
| 训练不稳定 | 归一化注意力 |
| 有向图 | 简化GAT |

---

## 7. 调库实现（PyTorch Geometric）

### 7.1 环境配置

```bash
pip install torch torch_geometric torch_sparse
```

### 7.2 完整代码实现

```python
"""
GAT在Cora引文网络上的节点分类
使用PyTorch Geometric实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import numpy as np

# ====================数据加载===================
print("正在加载Cora数据集...")
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")

# ====================模型定义===================
class GAT(nn.Module):
    """
    两层GAT模型用于节点分类
    
    架构:
    Input -> GATConv(8头) -> ELU -> Dropout -> GATConv(1头) -> Output
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.5):
        super(GAT, self).__init__()
        
        # 第一层GATConv：8头注意力，输出拼接
        # 输入: F维 -> 输出: heads * hidden_channels维
        self.gat1 = GATConv(
            in_channels, 
            hidden_channels,
            heads=heads,
            dropout=dropout
        )
        
        # 第二层GATConv：1头（输出层），不拼接
        # 输入: heads * hidden_channels维 -> 输出: num_classes维
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        
        返回:
            logits: 预测 logits [num_nodes, out_channels]
        """
        # 第一层：8头GAT卷积 + ELU激活
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # GAT使用ELU而非ReLU
        
        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层：单头GAT卷积（输出层）
        x = self.gat2(x, edge_index)
        
        return x

# ====================训练函数===================
def train():
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    out = model(data.x, data.edge_index)
    
    # 训练损失
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 计算准确率
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    
    train_acc = train_correct.float().mean().item()
    val_acc = val_correct.float().mean().item()
    test_acc = test_correct.float().mean().item()
    
    return train_acc, val_acc, test_acc

# ====================主训练流程===================
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化模型
# 注意：对于Cora数据集，hidden_channels=8（单头）效果较好
model = GAT(
    in_channels=dataset.num_features,   # 1433
    hidden_channels=8,               # 每头8维，8头共64维
    out_channels=dataset.num_classes, # 7
    heads=8,                        # 8个注意力头
    dropout=0.5                    # dropout比例
).to(device)

# 数据移到设备
data = data.to(device)

# 优化器
# GAT通常使用较小的学习率
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,  # 比GCN小
    weight_decay=5e-4
)

# 训练循环
epochs = 500
best_val_acc = 0
best_test_acc = 0
train_losses = []
val_accs = []
test_accs = []

print("\n开始训练...")
for epoch in range(1, epochs + 1):
    loss = train()
    train_losses.append(loss)
    
    train_acc, val_acc, test_acc = test()
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
              f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

print(f'\n最终结果:')
print(f'最佳验证准确率: {best_val_acc:.4f}')
print(f'最佳测试准确率: {best_test_acc:.4f}')

# ====================可视化===================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_accs, 'g-', linewidth=2, label='Validation')
plt.plot(test_accs, 'r-', linewidth=2, label='Test')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gat_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n训练完成！曲线已保存为 gat_training_curves.png")

# ====================注意力权重可视化===================
@torch.no_grad()
def visualize_attention():
    """可视化学习到的注意力权重"""
    model.eval()
    
    # 获取第一层的注意力权重
    # GATConv在forward中会保存attention weights
    attentions = []
    
    # 重新前向传播以捕获注意力
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # 手动计算第一层注意力
    x = F.dropout(x, p=0.5, training=True)
    x = model.gat1(x, edge_index)
    
    # 这里只是示例，实际注意力权重存储在内部
    print("\n注意: GAT的注意力权重可通过 model.gat1.edge_index 查看")
    
visualize_attention()

# ====================不同配置对比===================
print("\n" + "="*50)
print("GAT不同配置性能对比")
print("="*50)

configs = [
    {"heads": 4, "hidden": 16, "lr": 0.005},
    {"heads": 8, "hidden": 8, "lr": 0.005},
    {"heads": 8, "hidden": 16, "lr": 0.003},
]

for i, config in enumerate(configs):
    print(f"\n配置{i+1}: heads={config['heads']}, hidden={config['hidden']}, lr={config['lr']}")
    # 实际训练代码略
```

### 7.3 代码说明

**GATConv参数：**
- `heads`：注意力头数
- `concat`：True（中间层）输出拼接，False（输出层）输出平均
- `dropout`：注意力权重也应用dropout

**关键区别于GCN：**
- 使用ELU而非ReLU
- 多头输出需要拼接
- 最后一层不拼接改为输出类别

---

## 8. 手工代码实现（PyTorch）

### 8.1 核心注意力计算

```python
"""
GAT手工实现 - 完整版
不依赖PyG，实现核心注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import numpy as np

# ====================注意力层实现===================
class GraphAttentionLayer(nn.Module):
    """
    单层GAT注意力
    
    公式:
    h_i' = ELU(sigma_j a(Wh_i, Wh_j) * Wh_j)
    
    其中:
    a(Wh_i, Wh_j) = LeakyReLU(w^T [Wh_i || Wh_j])
    """
    def __init__(self, in_features, out_features, heads=8, 
                 alpha=0.2, dropout=0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # 每头的输出维度
        self.heads = heads
        self.alpha = alpha  # LeakyReLU负斜率
        self.concat = concat  # 是否拼接多头输出
        
        # 可训练权重矩阵 W (共享于所有头)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * heads)))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        
        # 注意力向量 a (每个头独立)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features, heads)))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, in_features]
            adj: 邻接矩阵 [num_nodes, num_nodes] (sparse or dense)
        
        返回:
            h: 更新后的特征 [num_nodes, heads*out_features] 或 [num_nodes, out_features]
        """
        N = x.size(0)
        
        # 特征变换: h = Wh
        h = torch.matmul(x, self.W)  # [N, out_features * heads]
        h = h.view(N, self.heads, self.out_features)  # [N, heads, out_features]
        
        # 获取邻居索引（假设adj是稀疏邻接矩阵）
        if isinstance(adj, np.ndarray):
            adj = torch.FloatTensor(adj)
        edge_index = adj.nonzero(as_tuple=False).T
        if edge_index.size(0) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # 计算每个头的注意力
        outputs = []
        for head in range(self.heads):
            h_head = h[:, head, :]  # [N, out_features]
            
            # 获取该头对应的attention vector
            a_head = self.a[:, :, head]  # [1, 2*out_features]
            
            # 对每条边计算注意力
            e_ij = self._compute_attention(h_head, a_head, edge_index)
            alpha_ij = F.softmax(e_ij, dim=1)
            alpha_ij = self.dropout(alpha_ij)
            
            # 加权聚合
            h_new = self._aggregate(h_head, alpha_ij, edge_index, N)
            outputs.append(h_new)
        
        if self.concat:
            # 拼接多头输出
            h = torch.cat(outputs, dim=1)  # [N, heads*out_features]
        else:
            # 多头平均
            h = torch.mean(torch.stack(outputs), dim=0)  # [N, out_features]
        
        return h
    
    def _compute_attention(self, h, a, edge_index):
        """计算边上的注意力分数"""
        # 源节点和目标节点特征
        src = h[edge_index[0]]  # [E, out_features]
        dst = h[edge_index[1]]  # [E, out_features]
        
        # 拼接 + LeakyReLU
        edge_h = torch.cat([src, dst], dim=1)  # [E, 2*out_features]
        e = torch.matmul(edge_h, a.squeeze(0))  # [E, 1]
        e = self.leakyrelu(e.squeeze(1))  # [E]
        
        return e
    
    def _aggregate(self, h, weights, edge_index, N):
        """加权聚合邻居特征"""
        h_new = torch.zeros(N, self.out_features, device=h.device)
        
        # 按权重累加
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # 简单实现：按目的节点分组求和
        for i in range(N):
            mask = dst_idx == i
            if mask.sum() > 0:
                neighbor_h = h[src_idx[mask]]
                neighbor_w = weights[mask]
                h_new[i] = (neighbor_h * neighbor_w.unsqueeze(1)).sum(0)
        
        # 加上ELU激活
        h_new = F.elu(h_new)
        
        return h_new

# ====================完整GAT模型===================
class GAT(nn.Module):
    """两层GAT模型"""
    def __init__(self, n_features, n_hidden, n_classes, 
                 heads=8, dropout=0.5):
        super(GAT, self).__init__()
        
        # 第一层：多头GAT
        self.gat1 = GraphAttentionLayer(
            n_features, n_hidden, heads=heads,
            dropout=dropout, concat=True
        )
        
        # 第二层：单头GAT（输出层）
        self.gat2 = GraphAttentionLayer(
            n_hidden * heads, n_classes, heads=1,
            dropout=dropout, concat=False
        )
        
        self.dropout = dropout
    
    def forward(self, x, adj):
        # 第一层
        x = self.gat1(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层
        x = self.gat2(x, adj)
        
        return x

# ====================训练函数===================
def load_cora():
    """加载简化Cora数据"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    
    features = torch.FloatTensor(np.random.rand(num_nodes, num_features))
    labels = torch.LongTensor(np.random.randint(0, num_classes, num_nodes))
    
    # 随机稀疏邻接矩阵
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj > 0.96).astype(float)
    adj = adj + np.eye(num_nodes)
    adj = csr_matrix(adj)
    
    train_mask = torch.BoolTensor([i < 140 for i in range(num_nodes)])
    val_mask = torch.BoolTensor([140 <= i < 640 for i in range(num_nodes)])
    test_mask = torch.BoolTensor([640 <= i < 1640 for i in range(num_nodes)])
    
    return features, labels, adj, train_mask, val_mask, test_mask

def train_gat():
    """训练GAT模型"""
    # 加载数据
    features, labels, adj, train_mask, val_mask, test_mask = load_cora()
    
    # 模型
    model = GAT(
        n_features=features.shape[1],
        n_hidden=8,
        n_classes=7,
        heads=8,
        dropout=0.5
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # 训练
    epochs = 200
    criterion = nn.CrossEntropyLoss()
    
    print("开始训练GAT...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features, adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            model.eval()
            pred = logits.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            print(f'Epoch {epoch+1}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}')

if __name__ == '__main__':
    train_gat()
```

### 8.2 关键实现细节

**注意力计算：**
```python
# 公式: e_ij = LeakyReLU(w^T [Wh_i || Wh_j])
edge_h = torch.cat([Wh_src, Wh_dst], dim=1)
e = leakyrelu(torch.matmul(edge_h, w))
```

**Softmax归一化：**
```python
# 按目标节点分组softmax
alpha_ij = softmax(e_ij, dim=dst_node)
```

**ELU激活：**
```python
# GAT使用ELU而非ReLU
h_new = F.elu(h_new)
```

---

## 9. 可视化与结果理解

### 9.1 注意力权重可视化

```python
"""
可视化学习到的注意力权重
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_weights(model, data, num_nodes=50):
    """可视化节点注意力分布"""
    model.eval()
    
    with torch.no_grad():
        # 获取注意力权重（需要从模型中提取）
        # 实际使用需修改模型结构保存注意力
        
        # 随机展示
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, ax in enumerate(axes):
            # 随机选一个节点
            node_id = np.random.randint(0, num_nodes)
            
            # 假设我们获取到了注意力
            # attention = attention_weights[:, node_id].cpu().numpy()
            # 这里用随机数据模拟
            attention = np.random.rand(30)
            attention = attention / attention.sum()
            
            ax.bar(range(len(attention)), attention)
            ax.set_xlabel('Neighbor Index')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'Node {node_id} Attention')
        
        plt.tight_layout()
        plt.savefig('attention_weights.png')
        plt.show()

# 调用
visualize_attention_weights(model, data)
```

### 9.2 节点嵌入可视化

```python
"""
GAT节点嵌入可视化
"""
from sklearn.manifold import TSNE

def visualize_embeddings(model, data):
    """可视化GAT学到的节点嵌入"""
    model.eval()
    
    with torch.no_grad():
        # 获取第一层输出（ELU后的表征）
        x = data.x
        edge_index = data.edge_index
        
        # 第一层输出
        h = model.gat1(x, edge_index)
        h = F.elu(h)
        
        # t-SNE降维
        h_np = h.cpu().numpy()
        h_2d = TSNE(n_components=2, random_state=42).fit_transform(h_np)
        
        # 绘制
        plt.figure(figsize=(10, 8))
        labels = data.y.cpu().numpy()
        
        plt.scatter(h_2d[:, 0], h_2d[:, 1], c=labels, 
                   cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(label='Class')
        plt.title('GAT Node Embeddings (t-SNE)')
        plt.savefig('gat_embeddings.png')
        plt.show()

visualize_embeddings(model, data)
```

### 9.3 训练曲线解读

**典型GAT训练曲线：**

- Loss下降更平滑（得益于多头注意力稳定训练）
- Val acc通常比GCN高1-3%
- 过拟合风险略高（需要更多正则化）

---

## 10. 模型评估

### 10.1 与GCN对比

```python
"""
GAT vs GCN 性能对比
"""
results = {
    'GCN': {'train': 0.98, 'val': 0.76, 'test': 0.81},
    'GAT(4头)': {'train': 0.99, 'val': 0.78, 'test': 0.83},
    'GAT(8头)': {'train': 0.99, 'val': 0.79, 'test': 0.84},
    'GraphSAGE': {'train': 0.97, 'val': 0.77, 'test': 0.82}
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.2

for i, (model_name, metrics) in enumerate(results.items()):
    ax.bar(x + i*width, [metrics['train'], metrics['val'], metrics['test']], 
          width, label=model_name)

ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy')
ax.set_title('GAT vs Other Models')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Train', 'Val', 'Test'])
ax.legend()
plt.tight_layout()
plt.savefig('gat_vs_others.png')
plt.show()
```

### 10.2 评估指标

```python
"""
GAT详细评估
"""
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_gat(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[mask].cpu()
        y_pred = pred[mask].cpu()
        
        print(classification_report(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred)
        
    return cm

cm = evaluate_gat(model, data, data.test_mask)
```

---

## 11. 常见问题与易错点

### 11.1 实现问题

**问题1：多头输出维度错误**
```
错误: dimension mismatch
原因: 第二层输入维度应该是 heads * hidden_channels
解决: GATConv(hidden_channels * heads, out_channels)
```

**问题2：注意力不稳定**
```
问题: 注意力值趋于0或1
原因: 学习率过大
解决: 减小学习率(0.001-0.005)
```

**问题3：内存爆炸**
```
原因: 注意力计算O(E*K)
解决: 减少头数或使用采样
```

### 11.2 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Loss不下降 | 学习率过小 | 增大学习率 |
| 过拟合 | dropout太小 | 增大dropout |
| 梯度爆炸 | 特征未归一化 | 标准化输入 |

---

## 12. 学习总结

### 12.1 核心要点

**GAT核心创新：**
- 将Transformer的注意力机制引入图数据
- 自适应学习每个邻居的重要性
- 多头注意力稳定训练

**关键公式：**
$$\alpha_{ij} = softmax(LeakyReLU(a \cdot [Wh_i || Wh_j]))$$
$$h_i' = ELU(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} Wh_j)$$

### 12.2 与GCN对比

| 方面 | GCN | GAT |
|------|-----|-----|
| 邻居权重 | 固定(度倒数) | 自适应学习 |
| 计算 | O(E) | O(E×K) |
| 表达能力 | 较弱 | 较强 |

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题

**1. GAT中的LeakyReLU负斜率是？**
A. 0.1
B. 0.01
C. 0.2
D. 0.01
**答案：C** 默认为0.2

**2. GAT中间层使用什么激活函数？**
A. ReLU
B. LeakyReLU
C. ELU
D. Tanh
**答案：C** 使用ELU

**3. GAT的多头注意力输出是？**
A. 求和
B. 平均
C. 拼接
D. 池化
**答案：C** 中间层拼接，输出层平均

### 13.2 简答题

**1. 为什么GAT使用LeakyReLU？**
答：LeakyReLU允许负值存在，使注意力可以是负的（不关注该邻居）。如果使用ReLU，负注意力会被置为0，会导致梯度无法回传，无法学习"忽略"某个邻居。

**2. GAT比GCN的优势在哪里？**
答：
- 自适应权重：学习邻居重要性
- 可解释性：注意力权重可解释节点关系
- 灵活性：可处理异构图
- 效果：通常精度更高

---

## 14. 学习路径建议建议

### 14.1 入门路径（2周）

1. **第1周**：理解注意力机制 → GAT论文精读
2. **第2周**：PyG实现 → Cora实验

### 14.2 进阶路径（2周）

1. **第3周**：实现GAT vs GCN对比实验
2. **第4周**：可视化注意力权重 → 项目实战

### 14.3 进一步学习

- GATv2（高效实现）
- Transformer for Graphs
- 异构图神经网络

---

**学习建议**：先理解Transformer的注意力机制，再学习GAT如何将其推广到图结构。GAT是理解后续Graph Transformer的基础。