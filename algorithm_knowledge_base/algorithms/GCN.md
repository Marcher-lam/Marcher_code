# GCN 图卷积网络 学习文档

> 图卷积网络（Graph Convolutional Network），谱域图卷积的经典实现，2017年Kipf等人提出。

---

## 1. 算法基础认知

### 1.1 什么是GCN？

图卷积网络（GCN）是第一个在图结构数据上实现有效卷积操作的深度学习模型。它将传统CNN中的卷积概念推广到非欧几里得空间，能够直接处理图拓扑结构中节点的特征信息和连接关系。GCN的核心思想是利用图的拉普拉斯矩阵进行谱域卷积，通过频谱理论在图上进行信号的滤波操作。

### 1.2 历史背景与发展

- **2013年**：Bruna等人首次提出基于频谱图理论的卷积方法（SCNN）
- **2015年**：Defferrard等人提出ChebNet，使用切比雪夫多项式近似加速卷积
- **2017年**：Kipf和Welling提出一阶ChebNet简化版，即今天广泛使用的GCN
- **核心贡献**：将复杂度从O(K×N²)降低到O(K×E)，使大规模图计算成为可能

### 1.3 核心定位

GCN是图神经网络的奠基之作，它建立了谱域方法与消息传递框架的联系。相对于传统方法：
- vs 节点嵌入方法（DeepWalk、Node2Vec）：GCN是监督学习方法，可端到端训练
- vs 谱域方法：GCN简化了计算复杂度，更易实现
- vs 空域方法：GCN提供了理论基础，两年后才出现GraphSAGE等空域方法

---

## 2. 核心原理

### 2.1 图的表示

图G = (V, E) 由节点集V和边集E组成：
- N = |V|：节点数量
- A ∈ R^{N×N}：邻接矩阵，A_ij = 1表示节点i和j相连
- D：度矩阵，D_ii = Σ_j A_ij
- X ∈ R^{N×F}：节点特征矩阵，F为特征维度

### 2.2 拉普拉斯矩阵

图拉普拉斯矩阵是GCN的核心工具，定义为：L = D - A

归一化形式：
$$\tilde{L} = I_N - D^{-1/2} A D^{-1/2} = U \Lambda U^T$$

其中U是特征向量矩阵，Λ是特征值对角矩阵。拉普拉斯矩阵的特性：
- 是半正定矩阵，特征值非负
- 特征值对应图的"频率"，特征向量对应"基波模式"
- 类似于离散拉普拉斯算子在连续空间的推广

### 2.3 谱域卷积

在频谱域，图信号x与滤波器g的卷积定义为：
$$g * x = U ((U^T g) \odot (U^T x))$$

其中⊙表示逐元素乘积。如果滤波器g可参数化，训练难度很高。

### 2.4 切比雪夫多项式近似

使用K阶切比雪夫多项式近似滤波器：
$$g_{\theta}(\Lambda) = \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

其中$\tilde{\Lambda} = 2\Lambda/\lambda_{max} - I$用于特征值归一化。切比雪夫多项式递归定义：
$$T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$$
$$T_0(x) = 1, T_1(x) = x$$

### 2.5 一阶GCN卷积

当K=1（只取k=0和k=1两项）时：
$$g * x \approx \theta_0 x + \theta_1 (D^{-1/2} A D^{-1/2}) x$$

设θ_0 = -θ_1 = 1（简化约束），得到：
$$g * x = (I + D^{-1/2} A D^{-1/2}) x$$

添加自环（self-loop）后：
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

其中：
- $\tilde{A} = A + I_N$：带自环的邻接矩阵
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$：对应的度矩阵
- W^{(l)}：第l层的可训练权重矩阵
- σ：非线性激活函数（通常用ReLU）

### 2.6 消息传递视角

从消息传递神经网络（MPNN）的角度看，GCN的单层卷积可写为：
$$h_v^{(l+1)} = \sigma(W^{(l)} \cdot MEAN(\{h_u^{(l)}, \forall u \in N(v) \cup \{v\}\}))$$

每个节点的新特征是其自身和邻居节点特征的加权平均，再经过线性变换和非线性激活。

---

## 3. 数学公式与推导

### 3.1 从原始卷积到简化形式

**步骤1：谱域卷积定义**
任意图信号x与滤波器g的谱卷积：
$$(g * x)_i = \sum_j \hat{g}_{ij} x_j = U ((\hat{g} \odot \hat{x})_j)$$

其中$\hat{x} = U^T x$是x的图傅里叶变换。

**步骤2：参数化滤波器**
设$\hat{g}(\lambda) = \sum_{k=0}^{K-1} \theta_k \lambda^k$，则：
$$g * x = U \sum_{k=0}^{K-1} \theta_k \Lambda^k U^T x = \sum_{k=0}^{K-1} \theta_k (U \Lambda^k U^T)^k x$$

注意这里用了$(U \Lambda^k U^T)^k \neq L^k$，所以不直接用L进行多项式展开。

**步骤3：切比雪夫多项式**
使用归一化拉普拉斯$\tilde{L} = 2L/\lambda_{max} - I$：
$$T_0(\tilde{L}) = I$$
$$T_1(\tilde{L}) = \tilde{L}$$
$$T_2(\tilde{L}) = 2\tilde{L}^2 - I$$
$$g * x = \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x$$

### 3.6 扩展公式补充

**从谱域到空域的数学桥梁**

设$h^{(l)}_v$为第$l$层节点$v$的特征，GCN的逐节点更新为：
$$h^{(l+1)}_v = \sigma\left(\sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{d_u d_v}} h^{(l)}_u W^{(l)}\right)$$

其中$d_v = \deg(v)$是节点度数。

**消息传递框架**
GCN是消息传递神经网络（MPNN）的特例：
$$h^{(l+1)}_v = \gamma^{(l)}\left(h^{(l)}_v, \square_{u \in N(v)} \mu^{(l)}(h^{(l)}_u, h^{(l)}_v)\right)$$

其中：
- 消息函数：$\mu(x,y) = \frac{1}{\sqrt{d_u d_v}} x W$
- 聚合函数：$\gamma = \sigma(\sum)$

**拉普拉斯矩阵的谱分析**
特征值$\lambda_i$表示图的"频率"：
- $\lambda_1 = 0$：对应均匀特征向量，常数信号无变化
- $\lambda_i \in [0,2]$：归一化后的范围

低通滤波器（$\lambda \approx 0$）保留平滑信号，高通滤波器（$\lambda \approx 2$）增强变化。

**图卷积的频率响应**
设滤波器$g_\theta$的频率响应为$g(\lambda)$：
$$g * x = U g(\Lambda) U^T x$$

- $g(\lambda) \approx 1$：低通，特征传播
- $g(\lambda) \approx \lambda$：高通，边缘检测
- $g(\lambda) = 1-\lambda$：图残差连接

**步骤4：一阶近似（K=1）**
当K=1时只取两项：
$$g * x = \theta_0 T_0(\tilde{L})x + \theta_1 T_1(\tilde{L})x = \theta_0 x + \theta_1 \tilde{L}x$$

**步骤5：对称归一化**
使用对称归一化拉普拉斯$\hat{L} = I - D^{-1/2}AD^{-1/2}$：
$$\tilde{L} = \hat{L} = I - D^{-1/2}AD^{-1/2}$$

所以：
$$g * x = \theta_0 x + \theta_1 (I - D^{-1/2}AD^{-1/2})x = \theta_0 x + \theta_1 x - \theta_1 D^{-1/2}AD^{-1/2}x$$

**步骤6：添加自环与简化**
添加自环：$\tilde{A} = A + I$，$\tilde{D} = D + I$：
$$g * x = (I + \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2})x$$

设$\theta_0 = \theta_1 = 1$，并添加激活函数：
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

### 3.2 维度分析

假设：
- 输入：N个节点，每个节点特征维度为F
- 隐藏层维度：H
- 权重矩阵：W^{(l)} ∈ R^{F×H}

前向传播计算：
- 输入H^{(l)}：N × F
- 邻接变换：N × N
- 输出H^{(l+1)}：N × H
- 计算复杂度：O(E × F × H)

### 3.3 多层堆叠

k层GCN的完整前向传播：
```python
H^{(0)} = X  # 初始节点特征
for i in range(1, k+1):
    H^{(i)} = σ(Ã~(-1/2)ÃÃ~(-1/2) H^{(i-1)} W^{(i-1)})
```

经过k层卷积，每个节点能够聚合k-hop邻居的信息。

---

## 4. 训练过程讲解

### 4.1 半监督节点分类

GCN常用于半监督节点分类，训练流程：

**数据准备**
```python
# Cora数据集
# 节点数：2708，包含7类论文
# 边数：5429（引文关系）
# 特征：1433维词向量
# 训练集：每类20个节点，共140个
# 测试集：1000个节点
# 验证集：500个节点
```

**模型定义**
```python
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj)
        return h
```

**损失函数**
```python
# 交叉熵损失，只对有标签节点计算
criterion = nn.CrossEntropyLoss()
loss = criterion(logits[train_mask], labels[train_mask])
```

**优化器**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
```

### 4.2 训练技巧

**归一化策略**
- 对特征进行标准化：X' = (X - μ) / σ
- 或使用批量归一化：GraphNorm

**正则化**
- Dropout：在每层GCN后添加，通常p=0.5
- 权重衰减：Adam优化器中设置weight_decay=5e-4
- 早停：监控验证集准确率

**学习率调度**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=10, factor=0.5
)
```

### 4.3 全监督模式

用于图分类时，需要对节点嵌入进行池化：
```python
# 全局平均池化
h_graph = torch.mean(h_nodes, dim=0)
# 全局最大池化
h_graph = torch.max(h_nodes, dim=0)[0]
# 全局求和池化
h_graph = torch.sum(h_nodes, dim=0)
```

---

## 5. 应用场景

### 5.1 节点分类

**引文网络**
- Cora、CiteSeer、PubMed引文网络
- 论文主题分类
- 未知论文可分类

**社交网络**
- 用户兴趣标签预测
- 用户属性推断
- 欺诈检测

### 5.2 图分类

**分子性质预测**
- 药物分子属性预测
- 蛋白质功能预测
- 材料性质预测

**知识图谱补全**
- 实体类型预测
- 关系预测

### 5.3 其他应用

- 交通预测：道路网络流量预测
- 推荐系统：用户-商品交互图
- 异常检测：图结构异常节点识别

---

## 6. 优缺点分析

### 6.1 优点

**理论基础扎实**
- 基于谱图理论，有严格的数学证明
- 拉普拉斯矩阵提供了几何解释
- 特征值对应图的"频率"特性

**实现简洁**
- 一行代码即可实现单层卷积
- 计算效率高，适合大规模图
- 与现有深度学习框架无缝集成

**效果稳定**
- 在引文网络等标准数据集上表现优异
- 对图结构变化有一定的鲁棒性
- 训练过程相对稳定

### 6.2 缺点

**感受野固定**
- 只能捕获k-hop邻居信息（k=层数）
- 无法自适应调整邻居聚合权重
- 对所有邻居平等对待，不区分重要性

**局限于转导学习**
- 无法处理新节点或新边（训练时不出现）
- 需要整个图结构参与计算
- 大规模图上内存消耗大

**过于平滑**
- 多次卷积后特征趋于一致
- 难以捕获局部结构差异
- 可能丢失节点自身特征

**无法捕获有向图信息**
- 归一化拉普拉斯只适用于无向图
- 有向图需要特殊处理

### 6.3 改进方向

| 问题 | 改进方法 |
|------|---------|
| 不能区分邻居重要性 | GAT（注意力机制） |
| 不能处理新节点 | GraphSAGE（归纳学习） |
| 没有边特征 | EdgeConv、RGCN |
| 过于平滑 | JKNet、DeepGCN |

---

## 7. 调库实现（PyTorch Geometric）

### 7.1 环境配置

```bash
pip install torch torch_geometric torch_sparse
```

### 7.2 完整代码实现

```python
"""
GCN在Cora引文网络上的节点分类
使用PyTorch Geometric实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt

# ====================数据加载===================
print("正在加载Cora数据集...")
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")

# ====================模型定义===================
class GCN(nn.Module):
    """
    两层GCN用于节点分类
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        # 第一个GCN卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        # 第二个GCN卷积层
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        
        ��回:
            logits: 预测 logits [num_nodes, out_channels]
        """
        # 第一层卷积 + ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积（输出层）
        x = self.conv2(x, edge_index)
        
        return x

# ====================训练函数===================
def train():
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    out = model(data.x, data.edge_index)
    
    # 只对训练集节点计算损失
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
    
    # 计算各数据集准确率
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    
    train_acc = train_correct.float().mean().item()
    val_acc = val_correct.float().mean().item()
    test_acc = test_correct.float().mean().item()
    
    return train_acc, val_acc, test_acc

# ====================主训练流程===================
# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化模型
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    dropout=0.5
).to(device)

# 加载数据到设备
data = data.to(device)

# 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

# 训练循环
epochs = 200
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
    
    if epoch % 20 == 0:
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
plt.savefig('gcn_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n训练完成！曲线已保存为 gcn_training_curves.png")

# ====================Spektral实现===================
"""
使用Spektral库的GCN实现（备选方案）
"""
try:
    from spektral.layers import GraphConv
    from spektral.models import GCN as SpektralGCN
    
    # 使用Spektral构建模型
    class SpektralGCNModel:
        def __init__(self, in_dim, hidden_dim, out_dim, n_classes):
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.n_classes = n_classes
            self.model = None
        
        def build_model(self):
            # 构建GCN模型
            X_in = Input(shape=(self.in_dim,))
            A_in = Input(shape=(None,), sparse=True)
            
            # 第一层GraphConv
            x = GraphConv(self.hidden_dim, activation='relu')([X_in, A_in])
            x = Dropout(0.5)(x)
            
            # 第二层GraphConv
            x = GraphConv(self.n_classes)([x, A_in])
            
            model = Model(inputs=[X_in, A_in], outputs=x)
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.model = model
            return model
    
    print("\n[备选] Spektral版本也已可用")

except ImportError:
    print("\n注意: 如需使用Spektral，请运行 pip install spektral")
```

### 7.3 代码说明

**核心组件：**
- `GCNConv`：PyG内置的GCN卷积层，实现核心卷积运算
- `Planetoid`：数据加载器，自动下载Cora等数据集
- `train_test_split_edges`：边索引的训练/测试/验证分割

**关键参数：**
- `cached=False`：每次前向传播重新计算归一化矩阵（内存换时间）
- `weight_decay=5e-4`：L2正则化系数
- `dropout=0.5`：随机丢弃比例

**运行结果（典型值）：**
- 训练准确率：~100%
- 验证准确率：~76%
- 测试准确率：~81%

---

## 8. 手工代码实现（PyTorch）

### 8.1 核心GCN卷积层实现

```python
"""
GCN手工实现 - 不依赖PyG
包含完整的图卷积层、前向传播、归一化拉普拉斯计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import numpy as np

# ====================图拉普拉斯计算===================
def compute_laplacian(adjacency, normalize=True):
    """
    计算归一化拉普拉斯矩阵
    
    参数:
        adjacency: 邻接矩阵 (numpy array or scipy sparse)
        normalize: 是否使用对称归一化
    
    返回:
        laplacian: 归一化拉普拉斯矩阵 (torch sparse tensor)
    """
    # 转为稀疏矩阵
    if not isinstance(adjacency, csr_matrix):
        adjacency = csr_matrix(adjacency)
    
    # 添加自环
    adjacency = adjacency + np.eye(adjacency.shape[0])
    
    # 度矩阵
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degrees, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    
    # 对称归一化: D^(-1/2) @ A @ D^(-1/2)
    d_inv_sqrt = np.diag(degree_inv_sqrt)
    laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt
    
    # 转为torch张量
    laplacian = torch.FloatTensor(laplacian)
    
    return laplacian

# ====================GCN卷积层===================
class GraphConvolution(nn.Module):
    """
    单层GCN卷积
    
    公式: H' = σ(Ð^(-1/2)ÃÐ^(-1/2) H W)
    
    其中:
    - Ã = A + I (带自环的邻接矩阵)
    - Ð = D + I (带自环的度矩阵)
    - H: 节点特征
    - W: 可训练权重
    - σ: 激活函数
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 可训练权重
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 权重初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        """
        前向传播
        
        参数:
            input: 节点特征 [num_nodes, in_features]
            adj: 归一化拉普拉斯矩阵 [num_nodes, num_nodes]
        
        返回:
            output: 更新后的节点特征 [num_nodes, out_features]
        """
        # 线性变换
        support = torch.matmul(input, self.weight)
        
        # 图卷积: @ 表示矩阵乘法
        output = torch.matmul(adj, support)
        
        # 偏置
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'

# ====================完整GCN模型===================
class GCN(nn.Module):
    """
    两层GCN模型
    
    结构:
    Input -> GCN(1433->16) -> ReLU -> Dropout -> GCN(16->7) -> Output
    """
    def __init__(self, n_features, n_hidden, n_classes, dropout=0.5):
        super(GCN, self).__init__()
        
        # 第一个卷积层
        self.gc1 = GraphConvolution(n_features, n_hidden)
        
        # 第二个卷积层（输出层）
        self.gc2 = GraphConvolution(n_hidden, n_classes)
        
        self.dropout = dropout
        self.relu = nn.ReLU()
    
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, n_features]
            adj: 归一化拉普拉斯矩阵 [num_nodes, num_nodes]
        
        返回:
            logits: 预测logits [num_nodes, n_classes]
        """
        # 第一层卷积 + ReLU
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积（无激活，等softmax交叉熵损失）
        x = self.gc2(x, adj)
        
        return x

# ====================数据预处理===================
def load_cora():
    """
    加载Cora引文网络数据
    
    返回:
        features: 节点特征矩阵 [2708, 1433]
        labels: 节点标签 [2708]
        adjacency: 邻接矩阵 [2708, 2708]
        train_mask, val_mask, test_mask: 分割掩码
    """
    # 读取数据（简化版，实际使用torch_geometric）
    # 这里使用随机生成数据演示
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    
    # 随机生成特征（真实数据需读取文件）
    features = torch.FloatTensor(np.random.rand(num_nodes, num_features))
    
    # 随机生成标签
    labels = torch.LongTensor(np.random.randint(0, num_classes, num_nodes))
    
    # 随机生成稀疏邻接矩阵
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj > 0.95).astype(float)  # 稀疏化
    adj = adj + np.eye(num_nodes)  # 添加自环
    adj = csr_matrix(adj)
    
    # 分割掩码
    train_mask = torch.BoolTensor([i < 140 for i in range(num_nodes)])
    val_mask = torch.BoolTensor([140 <= i < 640 for i in range(num_nodes)])
    test_mask = torch.BoolTensor([640 <= i < 1640 for i in range(num_nodes)])
    
    return features, labels, adj, train_mask, val_mask, test_mask

# ====================训练循环===================
def train_gcn():
    """训练GCN模型"""
    # 加载数据
    features, labels, adj_sparse, train_mask, val_mask, test_mask = load_cora()
    
    # 计算归一化拉普拉斯
    adj = compute_laplacian(adj_sparse)
    
    # 初始化模型
    model = GCN(
        n_features=features.shape[1],
        n_hidden=16,
        n_classes=7,
        dropout=0.5
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    epochs = 200
    best_val_acc = 0
    best_test_acc = 0
    
    print("开始训练手工实现的GCN...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(features, adj)
        
        # 计算训练损失
        loss = criterion(logits[train_mask], labels[train_mask])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if (epoch + 1) % 40 == 0:
            print(f'Epoch {epoch+1:03d} | Loss: {loss:.4f} | '
                  f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')
    
    print(f'\n最终结果:')
    print(f'最佳验证准确率: {best_val_acc:.4f}')
    print(f'最佳测试准确率: {best_test_acc:.4f}')
    
    return model

# ====================主函数===================
if __name__ == '__main__':
    model = train_gcn()
```

### 8.2 关键实现细节

**归一化拉普拉斯计算：**
```python
# 公式：Ã~ = D~^(-1/2) Ã D~^(-1/2)
# 其中 Ã = A + I（带自环）
# D~ 是带自环的度矩阵
```

**权重初始化：**
- 使用Xavier均匀分布初始化
- 确保信号在层间传播时方差稳定

**前向传播：**
- 第一层：特征变换 → 图聚合 → ReLU激活 → Dropout
- 第二层：特征变换 → 图聚合 → 输出logits

---

## 9. 可视化与结果理解

### 9.1 节点嵌入可视化

```python
"""
GCN节点嵌入可视化
使用t-SNE降维后在二维空间展示
"""
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(model, data, device):
    """可视化学到的节点嵌入"""
    model.eval()
    
    with torch.no_grad():
        # 获取第一层后的嵌入（ReLU后）
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # 第一层卷积
        h = model.gc1(x, edge_index)
        h = torch.relu(h)
        
        # t-SNE降维
        h_np = h.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        h_2d = tsne.fit_transform(h_np)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        labels_np = data.y.cpu().numpy()
        
        scatter = plt.scatter(h_2d[:, 0], h_2d[:, 1], 
                           c=labels_np, cmap='tab10', 
                           alpha=0.6, s=20)
        plt.colorbar(scatter, label='Class')
        plt.title('GCN Node Embeddings (t-SNE)', fontsize=14)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig('gcn_embeddings.png', dpi=150)
        plt.show()
        
        # 计算轮廓系数
        silhouette = silhouette_score(h_2d, labels_np)
        print(f'轮廓系数: {silhouette:.4f}')

# 调用
visualize_embeddings(model, data, device)
```

### 9.2 训练曲线解读

**典型训练曲线：**

![训练曲线](gcn_training_curves.png)

**Loss曲线特征：**
- 快速下降期（前20 epochs）：模型快速学习粗粒度特征
- 平稳期（20-100 epochs）：损失缓慢下降
- 轻微过拟合（100+ epochs）：验证准确率可能下降

**Accuracy曲线特���：**
- Train接近100%：可能过拟合
- Val/Test稳定：泛化能力评估
- Gap大：过拟合严重

### 9.3 特征重要性分析

```python
"""
分析GCN中学到的特征重要性
"""
def analyze_feature_importance(model, data):
    """分析各特征维度的贡献"""
    # 第一个卷积层的权重
    w = model.gc1.weight.detach().numpy()
    
    # 每维特征的重要性（权重的L2范数）
    importance = np.linalg.norm(w, axis=1)
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(importance)), importance, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (L2 Norm)')
    plt.title('Feature Importance in GCN')
    plt.savefig('feature_importance.png')
    plt.show()
    
    # 前10个最重要特征
    top_k = 10
    top_indices = np.argsort(importance)[-top_k:][::-1]
    print(f'前{top_k}个重要特征索引: {top_indices}')
    
analyze_feature_importance(model, data)
```

---

## 10. 模型评估

### 10.1 评估指标

```python
"""
GCN模型评估指标计算
"""
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

def evaluate_model(model, data, mask):
    """评估模型"""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 分类报告
        report = classification_report(y_true, y_pred)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

# 各数据集评估
train_metrics = evaluate_model(model, data, data.train_mask)
val_metrics = evaluate_model(model, data, data.val_mask)
test_metrics = evaluate_model(model, data, data.test_mask)

print("测试集评估结果:")
print(f"准确率: {test_metrics['accuracy']:.4f}")
print("\n分类报告:")
print(test_metrics['report'])
```

### 10.2 过拟合分析

**判断标准：**
- Train acc ≈ 100%，Val acc < 70%：严重过拟合
- Train acc > Val acc > Test acc：正常过拟合
- Train ≈ Val ≈ Test：欠拟合

**解决方案：**

| 策略 | 方法 |
|------|------|
| 增加正则化 | Dropout从0.5增到0.8 |
| 减少模型容量 | 隐藏层维度从16减到8 |
| 数据增强 | 使用节点特征扰动 |
| 早停 | 监控Val acc |

### 10.3 对比实验

```python
"""
GCN vs 其他方法对比
"""
results = {
    'GCN': 0.815,
    'DeepWalk': 0.670,
    'Node2Vec': 0.680,
    'Link Prediction': 0.750,
    'Label Propagation': 0.720
}

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='steelblue')
plt.ylim(0.5, 0.9)
plt.ylabel('Accuracy')
plt.title('GCN vs Other Methods on Cora')
plt.axhline(y=0.815, color='r', linestyle='--', label='GCN')
plt.legend()
plt.savefig('comparison.png')
plt.show()
```

---

## 11. 常见问题与易错点

### 11.1 实现问题

**问题1：邻接矩阵维度不匹配**
```
错误: matmul: dim1 must equal dim2 of other.matrix
原因: A和特征矩阵维度不一致
解决: 检查 A.shape[0] == X.shape[0]
```

**问题2：梯度消失/爆炸**
```
原因: 层数过深或学习率过大
解决: 
- 减小学习率（lr <= 0.01）
- 减少层数（<= 2层）
- 初始化权重使用Xavier
```

**问题3：邻接矩阵未归一化**
```
问题: 特征值随层数指数级增长
解决: 必须使用归一化拉普拉斯
```

### 11.2 训��问题

**问题1：训练不收敛**
```
检查:
1. 学习率是否过大（尝试0.001）
2. 权重初始化（Xavier）
3. 特征是否标准化
4. 是否有NaN/Inf
```

**问题2：准确率很低**
```
检查:
1. 标签是否正确
2. 图结构是否正确
3. 数据是否标准化
4. 数据量是否足够
```

### 11.3 内存问题

**问题：大规模图OOM**
```
解决方案:
1. 使用采样邻居（NeighborSampler）
2. 使用稀疏矩阵操作
3. 分批次处理（Mini-batch）
4. 减少隐藏层维度
```

### 11.4 其他问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 推理时需要原始图 | 转导学习限制 | 使用GraphSAGE |
| 只能处理无向图 | 拉普拉斯不对称 | 有向图预处理 |
| 邻居被平等对待 | GCN假设 | 改用GAT |

---

## 12. 学习总结

### 12.1 核心要点

**GCN的核心思想：**
- 将传统卷积推广到图结构数据
- 利用拉普拉斯矩阵进行谱域卷积
- 通过邻域聚合实现信息传递

**关键公式：**
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

**理解要点：**
- 拉普拉斯矩阵是核心工具，特征值对应频率
- 归一化保证数值稳定
- 切比雪夫多项式近似加速计算

### 12.2 重要概念

| 概念 | 含义 |
|------|------|
| 拉普拉斯矩阵 | L = D - A，图的离散拉普拉斯算子 |
| 归一化拉普拉斯 | 对称归一化形式 |
| 谱域卷积 | 在图频域进行卷积 |
| 感受野 | k层能捕获k-hop邻居 |

### 12.3 进一步学习方向

- 深入谱理论：理解特征值的几何意义
- 对比学习：GCN vs GAT vs GraphSAGE
- 实战项目：社交网络分析、推荐系统
- 前沿模型：GAT、Transformer for Graphs

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题

**1. GCN的核心操作是？**
A. 循环神经网络
B. 图卷积操作
C. 注意力机制
D. 池化操作
**答案：B** 图卷积操作

**2. GCN使用的归一化是？**
A. L1归一化
B. L2归一化
C. 对称归一化（$D^{-1/2}AD^{-1/2}$）
D. 批量归一化
**答案：C** 对称归一化

**3. GCN属于哪种学习范式？**
A. 归纳学习
B. 转导学习
C. 监督学习
D. 无监督学习
**答案：B** 转导学习（需要整个图结构）

### 13.2 简答题

**1. 为什么GCN需要添加自环？**
答：添加自环（$\tilde{A}=A+I$）是为了让每个节点在更新时能够保留自身特征，而不是只聚合邻居信息。如果不加自环，节点特征会逐渐被邻居特征"稀释"，丢失自身信息。

**2. 为什么GCN使用归一化拉普拉斯？**
答：使用归一化拉普拉斯$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$有两个原因：
（1）保证特征值在[0,1]范围，避免梯度爆炸
（2）归一化后节点的权重与其度成反比，大度节点不会主导聚合结果

**3. GCN和CNN的核心区别是什么？**
答：CNN在欧几里得空间（规则网格）上进行卷积，卷积核在图像上滑动感受野。GCN在非欧几里得空间（图结构）上进行卷积，邻居数量不固定，需要通过拉普拉斯矩阵实现聚合。

### 13.3 编程题

**题目1：实现三层GCN**
```python
# 参考答案
class ThreeLayerGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hid_dim)
        self.gc2 = GCNConv(hid_dim, hid_dim)
        self.gc3 = GCNConv(hid_dim, out_dim)
    
    def forward(self, x, edge_index):
        h = F.relu(self.gc1(x, edge_index))
        h = F.relu(self.gc2(h, edge_index))
        h = self.gc3(h, edge_index)
        return h
```

**题目2：比较有/无自环的效果**
```python
# 参考答案
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

# 无自环
out1 = model(x, edge_index)

# 有自环
edge_index_with_loop, _ = add_self_loops(edge_index)
out2 = model(x, edge_index_with_loop)

# 观察差异
print(f"无自环: {out1.mean().item():.4f}")
print(f"有自环: {out2.mean().item():.4f}")
```

### 13.4 思考题

**1. GCN能否处理异构图？**
答：默认GCN只能处理同构图（节点和边类型单一）。异构图需要使用RGCN（Relation GCN）或Hu等人提出的HetGNN等专门处理。

**2. GCN的层数如何选择？**
答：一般选择1-3层。层数过多会导致：
- 过度平滑（节点特征趋于一致）
- 感受野过大（计算量大）
- 过拟合风险增加

实际选择需根据图的直径（节点间最短路径的最大值）和任务需求。

**3. GCN vs GAT的区别？**
答：
- GCN：使用固定的归一化权重，对所有邻居平等对待
- GAT：使用注意力机制自适应学习邻居权重，能区分重要/不重要的邻居
- 结果：GAT通常效果更好，但计算成本稍高

---

## 14. 学习路径建议建议

### 14.1 入门路径（2周）

**第1周：基础概念**
1. 图论基础：邻接矩阵、度、拉普拉斯
2. 谱图理论简介：图傅里叶变换
3. GCN论文精读：Kipf & Welling 2017

**第2周：实现与实践**
1. PyG环境配置
2. GCN代码复现
3. Cora数据集实验

### 14.2 进阶路径（3周）

**第3周：变体学习**
1. GAT（注意力机制）- 1周
2. GraphSAGE（空域方法）- 1周
3. 对比三种方法异同 - 1周

**第4-5周：实战项目**
1. 社交网络分析
2. 推荐系统
3. 分子性质预测

### 14.3 高级路径（持续）

**第6周及以后：**
1. GNN最新论文（Transformer for Graph）
2. 异构图神经网络
3. 图生成模型
4. 可解释GNN

### 14.4 推荐资源

| 资源 | 链接 |
|------|------|
| GCN原论文 | [arxiv:1609.02907](https://arxiv.org/abs/1609.02907) |
| PyG文档 | [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/) |
| GNN论文集 | [Papers with Code](https://paperswithcode.com/task/graph-classification) |
| 图学习课程 | [Stanford CS224W](http://web.stanford.edu/class/cs224w/) |

---

**学习建议**：先理解谱图理论基础，再通过代码实现加深理解，最后在实际项目中应用。GCN是图神经网络的入门基础，掌握后再学习其他变体会更加顺畅。