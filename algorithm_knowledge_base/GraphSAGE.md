# GraphSAGE 学习文档

> GraphSAGE（Sampling and Aggregation），归纳式图神经网络，可处理未见节点。

---

## 1. 算法基础认知

### 1.1 什么是GraphSAGE？

GraphSAGE是2017年由Hamilton等人提出的 inductive learning（图归纳学习）框架。与GCN等直推式（transductive）方法要求训练时必须知道所有节点不同，GraphSAGE能够对训练过程中从未出现的新节点进行预测，这使其特别适合动态图和大规模图的场景。

### 1.2 核心创新

- **归纳学习**：可处理未见节点（inductive capability）
- **邻居采样**：通过采样控制计算复杂度
- **聚合函数**：通用聚合框架（Mean/LSTM/Pool）

### 1.3 与GCN对比

| 方面 | GCN（直推式） | GraphSAGE（归纳式） |
|------|---------------|-------------------|
| 新节点 | 不可处理 | 可处理 |
| 计算量 | O(N×E) | O(N×S×K) |
| 训练方式 | 全图训练 | 小批量训练 |
| 内存 | 大图OOM | 可扩展 |

### 1.4 历史发展

- **2017**：GraphSAGE原论文（Hamilton et al.）
- **后续发展**：无监督GraphSAGE、GeniePath、PinSage等

---

## 2. 核心原理

### 2.1 归纳学习 vs 直推学习

**直推学习（Transductive）**：
- 训练时已知整个图结构
- 测试时只能预测已知节点
- GCN属于此类

**归纳学习（Inductive）**：
- 训练时学习图的"生成模式"
- 测试时可预测新节点
- GraphSAGE属于此类

### 2.2 采样策略

为控制计算复杂度，GraphSAGE对邻居进行采样：

**K-hop采样**：
- 第1层：采样S1个邻居
- 第2层：在第1层邻居中再采样S2个
- 总采样数：S1 × S2 × ... × SK

**默认配置**：
```python
# sample_size = (20, 10) 表示
# 第1层采样20个邻居，第2层采样10个
```

### 2.3 聚合函数

GraphSAGE支持三种聚合方式：

**1. Mean聚合（均值）**：
$$h_v^k = \sigma(W \cdot MEAN(\{h_v^{k-1}\} \cup \{h_u^{k-1}: u \in \mathcal{N}(v)\}))$$

**2. LSTM聚合**：
- 将邻居随机排序后输入LSTM
- 捕获邻居的顺序依赖
- 计算量大但效果好

**3. Pool聚合**：
$$h_v^k = \max(\{\sigma(W \cdot h_u^{k-1}): u \in \mathcal{N}(v)\})$$

- 排列不变性（permutation invariant）
- 效果好于Mean

### 3.6 扩展公式补充

**GraphSAGE的归纳学习能力证明**

设训练图中节点的特征为$\{h_v^0\}_{v \in V_{train}}$，GraphSAGE学习聚合函数：
$$AGG(\{h_u: u \in N(v)\})$$

对于新节点$u \notin V_{train}$，只需获取其邻居特征（可能已知），即可计算表示：
$$h_u = AGG(\{h_u^0\} \cup \{h_v^0: v \in N(u)\})$$

这是GCN等直推式方法无法实现的。

**采样复杂度的数学分析**

设采样大小$S = (S_1, ..., S_K)$，总采样数：
$$M = \prod_{i=1}^{K} S_i$$

对于度分布$P(d)$，实际采样数为：
$$\mathbb{E}[M] = \prod_{i=1}^{K} S_i \cdot \mathbb{E}[d]^{K-1}$$

默认配置$(25,10)$适用于大规模图（度~20-50）。

**聚合函数的Permutation Invariance**

Mean聚合：
$$\text{AGG}(\{x_1, ..., x_k\}) = \frac{1}{k}\sum_i x_i$$

满足交换律：$\text{AGG}(\{x,y\}) = \text{AGG}(\{y,x\})$

Pool聚合：
$$\text{AGG}(\{x_1, ..., x_k\}) = \max(\sigma(Wx_i))$$

也满足交换律。

LSTM聚合不满足（依赖顺序），需要随机排序。

**无监督GraphSAGE的损失**

使用对比损失：
$$J(z_u) = -\log(\sigma(z_u^T z_v)) - \log(\sigma(-z_u^T z_{v'}))$$

其中$z_v$是真实邻居的表示，$z_{v'}$是负样本。

### 2.4 前向传播

完整的K层前向传播：

```
输入: 节点特征 h^(0)_v, 采样邻居 N(v)
for k in 1 to K:
    # 采样
    N'(v) = SAMPLE(N(v), S_k)
    
    # 聚合邻居表征
    h_N = AGGREGATE_k({h^(k-1)_u: u in N'(v)})
    
    # 拼接与变换
    h_concat = CONCAT(h^(k-1)_v, h_N)
    
    # 归一化
    h_concat = L2_NORM(h_concat)
    
    # 非线性变换
    h^(k)_v = σ(W^(k) · h_concat)
```

---

## 3. 数学公式与推导

### 3.1 原始GraphSAGE

设第k层聚合函数为AGGREGATE_k，权重矩阵为W^k：

**聚合阶段**：
$$h_{\mathcal{N}(v)}^k = AGGREGATE_k(\{h_u^{k-1}: u \in \mathcal{N}(v)\})$$

**更新阶段**：
$$h_v^k = \sigma(W^k \cdot CONCAT(h_v^{k-1}, h_{\mathcal{N}(v)}^k))$$

### 3.2 Batch前向传播

对于小批量训练，设批量节点集合为B：

```
输入:  batch节点B
输出:  批量表征 h^k_B

for k in K down to 1:
    for v in B:
        # 聚合子图
        B^k = B ∪ {邻居}
        
        # 检查已计算
        if h^k_v 已计算:
            continue
            
        # 计算v
        h^k_v = σ(W^k · CONCAT(h^{k-1}_v, AGGREGATE(h^{k-1}_N)))
```

### 3.3 复杂度分析

设batch size = B，每层采样S个邻居：

**时间复杂度**：
$$O(B \times S^K \times F^2)$$

其中：
- B：批量大小
- S：采样数
- K：层数
- F：特征维度

**空间复杂度**：
$$O(B \times S^K + E_{sampled})$$

### 3.4 损失函数

**有监督学习**：
$$\mathcal{L}_{sup} = -\sum_{v \in \mathcal{V}_{train}} \log(Z_v[y_v])$$

**无监督学习**：
$$\mathcal{L}_{unsup} = -\log \sigma(z_u^T z_v) - \sum_{u_n \in NEG(v)} \log \sigma(-z_u^T z_{u_n})$$

其中鼓励相邻节点表征相似，不同节点表征不同。

---

## 4. 训练过程讲解

### 4.1 满批量训练

```python
"""
GraphSAGE 满批量训练
适用于中小规模图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt

# 加载数据
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 模型定义
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 堆叠多层SAGEConv
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_dim, out_dim, normalize=True))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    num_layers=2
).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}')
```

### 4.2 小批量训练（推荐）

```python
"""
GraphSAGE 小批量训练
使用NeighborLoader采样
"""
from torch_geometric.loader import NeighborLoader

# 邻居采样器
train_loader = NeighborLoader(
    data,
    num_neighbors=[20, 10],  # 每层采样数
    batch_size=256,
    input_nodes=data.train_mask
)

# 或者使用full neighbor loader
full_loader = NeighborLoader(
    data,
    num_neighbors=[-1, -1],  # 全量邻居
    batch_size=None,
    shuffle=True
)

# 小批量训练
def train_minibatch():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 获取批量数据
        batch_x = batch.x.to(device)
        batch_edge_index = batch.edge_index.to(device)
        batch_y = batch.y[:batch.batch_size].to(device)
        
        # 前向传播
        out = model(batch_x, batch_edge_index)
        
        # 只计算批量节点的损失
        out = out[:batch.batch_size]
        
        loss = F.cross_entropy(out, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

for epoch in range(epochs):
    loss = train_minibatch()
    print(f'Epoch {epoch}: Loss={loss:.4f}')
```

### 4.3 无监督训练

```python
"""
GraphSAGE 无监督训练
使用对比损失
"""
class UnsupervisedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def unsupervised_loss(z, edge_index, neg_edge_index):
    """对比损失"""
    pos_loss = F.logsigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)).mean()
    neg_loss = F.logsigmoid(-(z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)).mean()
    return -pos_loss - neg_loss
```

---

## 5. 应用场景

### 5.1 动态图学习

- **新用户/商品推荐**：用户/商品不断加入系统
- **社交网络**：新用户注册
- **欺诈检测**：新账号识别

### 5.2 大规模图学习

- **PinTerest推荐**：数十亿节点
- **蛋白质相互作用网络**
- **药物发现**

### 5.3 节点分类/链接预测

- 属性预测
- 关系预测

---

## 6. 优缺点分析

### 6.1 优点

**归纳能力**
- 可处理新节点
- 适合动态图

**可扩展性**
- 邻居采样控制复杂度
- 可处理大规模图

**灵活性**
- 支持多种聚合函数
- 可自定义聚合

### 6.2 缺点

**采样可能丢失信息**
- 随机采样可能遗漏重要邻居
- 需要足够采样数

**额外超参数**
- 采样数S需要调优
- 聚合函数需要选择

### 6.3 改进方向

- PinSage：加权采样
- GraphSaint：层级采样
- VRDC：方差约简

---

## 7. 调库实现（PyTorch Geometric）

### 7.1 完整代码

```python
"""
GraphSAGE 完整实现
Cora节点分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv, GraphSAGE
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
import numpy as np

# ====================数据加载===================
print("加载Cora数据集...")
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")

# ====================使用PyG内置GraphSAGE===================
"""
PyG提供两种使用方式:
1. GraphSAGE类（一行代码）
2. SAGEConv层（自定义）
"""
# 方式1: 使用内置类
model1 = GraphSAGE(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    num_layers=2,
    dropout=0.5
)

# ====================自定义SAGEConv实现===================
"""
使用SAGEConv自定义模型
"""
class SAGENet(nn.Module):
    """
    两层GraphSAGE
    
    聚合方式: mean
    架构: Input -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> Output
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(SAGENet, self).__init__()
        
        # 第一层SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        
        # 第二层SAGEConv
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=True)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
        
        返回:
            logits: 预测 [N, out_channels]
        """
        # 第一层: 卷积 + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层: 卷积
        x = self.conv2(x, edge_index)
        
        return x

# ====================训练函数===================
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    
    return train_acc, val_acc, test_acc

# ====================训练流程===================
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型
model = SAGENet(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    dropout=0.5
).to(device)

data = data.to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练
epochs = 200
best_val = 0
best_test = 0
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
    
    if val_acc > best_val:
        best_val = val_acc
        best_test = test_acc
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
              f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

print(f'\n最终结果:')
print(f'最佳验证准确率: {best_val:.4f}')
print(f'最佳测试准确率: {best_test:.4f}')

# ====================可视化===================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_accs, 'g-', linewidth=2, label='Validation')
plt.plot(test_accs, 'r-', linewidth=2, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphsage_training.png')
plt.show()

# ====================不同聚合方式对比===================
"""
PyG支持不同聚合方式
"""
from torch_geometric.nn import SAGEConv

# Mean聚合（默认）
conv_mean = SAGEConv(16, 7, aggr='mean')

# Max聚合
conv_max = SAGEConv(16, 7, aggr='max')

# Mean+Max组合
conv_add_mean = SAGEConv(16, 7, aggr=['mean', 'max', 'softmax'])
```

### 7.2 小批量版本

```python
"""
使用NeighborLoader进行小批量训练
"""
from torch_geometric.loader import NeighborLoader

# 创建邻居采样器
train_loader = NeighborLoader(
    data,
    num_neighbors=[20, 10],  # 第1层20个，第2层10个
    batch_size=256,
    input_nodes=data.train_mask
)

# 小批量训练
def train_minibatch():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # 只对有标签节点计算loss
        out = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]
        
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    
    return total_loss / len(train_loader)

# 推理（对新节点）
@torch.no_grad()
def predict_new_node(new_x, new_edge_index, existing_model):
    """预测新节点"""
    existing_model.eval()
    with torch.no_grad():
        return existing_model(new_x, new_edge_index).argmax(dim=1)
```

---

## 8. 手工代码实现（PyTorch）

### 8.1 核心聚合函数

```python
"""
GraphSAGE 手工实现
不依赖PyG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ====================Mean聚合===================
class MeanAggregator(nn.Module):
    """Mean聚合器"""
    def __init__(self, in_channels, out_channels):
        super(MeanAggregator, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        参数:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
        
        返回:
            聚合后特征 [N, out_channels]
        """
        # 构建聚合表
        row, col = edge_index
        target = row
        source = col
        
        # 聚合：按目标节点分组求和
        out = torch.zeros(x.size(0), x.size(1), device=x.device)
        
        # 简单实现：遍历
        for i in range(x.size(0)):
            mask = target == i
            if mask.sum() > 0:
                neighbors = source[mask]
                out[i] = x[neighbors].mean(dim=0)
        
        # 线性变换
        out = self.linear(out)
        
        return out

# ====================Max聚合===================
class MaxAggregator(nn.Module):
    """Max聚合器"""
    def __init__(self, in_channels, out_channels):
        super(MaxAggregator, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        out = torch.zeros_like(x)
        
        for i in range(x.size(0)):
            mask = row == i
            if mask.sum() > 0:
                neighbors = col[mask]
                out[i] = x[neighbors].max(dim=0)[0]
        
        out = self.linear(out)
        return out

# ====================完整GraphSAGE层===================
class GraphSAGELayer(nn.Module):
    """完整GraphSAGE层"""
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super(GraphSAGELayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.zeros(in_channels * 2, out_channels))
        nn.init.xavier_uniform_(self.weight)
        
        # 聚合器
        if aggr == 'mean':
            self.aggregate = MeanAggregator(in_channels, in_channels)
        elif aggr == 'max':
            self.aggregate = MaxAggregator(in_channels, in_channels)
        else:
            raise ValueError(f"Unknown aggregator: {aggr}")
    
    def forward(self, x, edge_index):
        # 聚合邻居
        h_N = self.aggregate(x, edge_index)
        
        # 拼接自身特征和邻居聚合
        h_concat = torch.cat([x, h_N], dim=1)
        
        # 线性变换
        h_out = torch.matmul(h_concat, self.weight)
        
        # 归一化
        h_out = F.normalize(h_out, p=2, dim=1)
        
        return h_out

# ====================完整模型===================
class GraphSAGEModel(nn.Module):
    """两层GraphSAGE"""
    def __init__(self, n_features, n_hidden, n_classes):
        super(GraphSAGEModel, self).__init__()
        
        self.layer1 = GraphSAGELayer(n_features, n_hidden, aggr='mean')
        self.layer2 = GraphSAGELayer(n_hidden, n_classes, aggr='mean')
    
    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, edge_index)
        return x
```

### 8.2 对新节点的归纳推理

```python
"""
展示GraphSAGE的归纳能力
对新节点进行预测
"""
def inductive_inference(model, existing_x, existing_edge_index, new_node_features, new_edge_index):
    """
    对新节点进行归纳推理
    
    参数:
        model: 训练好的GraphSAGE模型
        existing_x: 已有节点特征
        existing_edge_index: 已有边
        new_node_features: 新节点特征
        new_edge_index: 新节点与已有节点的边
    """
    model.eval()
    
    with torch.no_grad():
        # 合并已有节点和新节点
        all_x = torch.cat([existing_x, new_node_features], dim=0)
        
        # 调整新边索引（考虑已有节点数）
        num_existing = existing_x.size(0)
        new_edge_index_adj = new_edge_index + num_existing
        
        # 合并边
        all_edge_index = torch.cat([existing_edge_index, new_edge_index_adj], dim=1)
        
        # 推理
        out = model(all_x, all_edge_index)
        
        # 只取新节点预测
        new_pred = out[num_existing:].argmax(dim=1)
        
    return new_pred
```

---

## 9. 可视化与结果理解

### 9.1 节点嵌入可视化

```python
"""
GraphSAGE节点嵌入可视化
"""
from sklearn.manifold import TSNE

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        h = out.cpu().numpy()
        
        # t-SNE降维
        h_2d = TSNE(n_components=2, random_state=42).fit_transform(h)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(h_2d[:, 0], h_2d[:, 1], c=data.y, cmap='tab10', alpha=0.6)
        plt.colorbar(label='Class')
        plt.title('GraphSAGE Embeddings')
        plt.savefig('graphsage_embeddings.png')
        plt.show()

visualize_embeddings(model, data)
```

### 9.2 训练曲线

- 训练速度：比GCN慢（需要采样）
- 准确率：与GCN相当或略高
- 泛化能力：更优（归纳学习）

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import classification_report

model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)

print(classification_report(data.y[data.test_mask], pred[data.test_mask]))
```

### 10.2 与GCN对比

| 模型 | Cora测试准确率 |
|------|-------------|
| GCN | ~81% |
| GraphSAGE | ~82% |
| GAT | ~83% |

---

## 11. 常见问题与易错点

### 11.1 实现问题

- 采样数设置不当
- 聚合函数选择不明智

### 11.2 训练问题

- 学习率过大
- Dropout过小

---

## 12. 学习总结

### 12.1 核心要点

- 归纳学习：处理新节点
- 邻居采样：控制复杂度
- 聚合函数：通用框架

### 12.2 关键公式

$$h_v^k = \sigma(W^k \cdot CONCAT(h_v^{k-1}, AGGREGATE(\{h_u^{k-1}: u \in \mathcal{N}(v)\})))$$

---

## 13. 练习题与思考题与思考题

### 13.1 选择题

**1. GraphSAGE属于什么学习范式？**
A. 直推学习
B. 归纳学习
C. 无监督学习
D. 增强学习
**答案：B** 归纳学习

**2. GraphSAGE不支持哪种聚合？**
A. Mean
B. LSTM
C. Max
D. Conv
**答案：D**

**3. GraphSAGE的优势是？**
A. ���算快
B. 可处理新节点
C. 准确率高
D. 内存小
**答案：B**

---

## 14. 学习路径建议建议

1. 理解GCN → 理解GraphSAGE
2. 实现各种聚合方式
3. 对比实验
4. 项目实战

---

**学习建议**：GraphSAGE是理解归纳学习的关键，与GCN对比学习能更好理解两种学习范式的差异。