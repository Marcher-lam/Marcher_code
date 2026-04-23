# Node2Vec 学习文档

> Node2Vec，混合游走策略的图节点嵌入算法，结合BFS和DFS特性。

---

## 1. 算法基础认知

### 1.1 什么是Node2Vec？

Node2Vec是2016年由Grover等人提出的图节点嵌入算法。它通过改进随机游走策略，在DeepWalk的随机游走基础上引入了两个可调参数p和q，分别控制游走的"返回性"和"广度"，从而能够同时捕获网络的"同质性"和"结构性"。Node2Vec将节点嵌入问题转化为序列化概率问题，使用Skip-gram模型训练。

### 1.2 核心创新

- **混合游走策略**：通过参数p和q平衡BFS和DFS特性
- **有向游走**：比DeepWalk更灵活的控制能力
- **负采样**：高效训练Skip-gram模型

### 1.3 与DeepWalk对比

| 方面 | DeepWalk | Node2Vec |
|------|---------|---------|
| 游走策略 | 纯随机 | 混合控制 |
| 参数 | 无 | p（返回）、q（广度） |
| 捕获特性 | 主要是同质性 | 同质性+结构性 |
| 控制能力 | 无 | 可调q |

### 1.4 历史发展

- **2014**：DeepWalk - 首次将词向量方法用于图节点嵌入
- **2016**：Node2Vec - 改进的混合游走策略
- **后续**：LINE、struc2vec、APP等

---

## 2. 核心原理

### 2.1 游走策略

Node2Vec的核心是改进的随机游走。给定当前游走位置v，下一步转移到x的概率为：

$$P(c_i=x|v) = \begin{cases} \frac{1}{p} & if \ (v,x) \in E_d \\ 1 & if \ (v,x) \in E_s \\ \frac{1}{q} & otherwise \end{cases}$$

其中：
- $E_s$：边(v,x)在上一跳中
- $E_d$：非$E_s$的边（远方节点）
- p：返回参数，控制返回上一节点的概率
- q：出入参数，控制游走向外还是向内

### 2.2 参数解释

**p（返回参数）**：
- p小（<1）：倾向于返回，局部探索，类似BFS
- p大（>1）：倾向于前进，全局探索，类似DFS

**q（广度参数）**：
- q小（<1）：倾向于访问距离远的节点（DFS特性）
- q大（>1）：倾向于访问近邻节点（BFS特性）

### 2.3 组合特性

- **p小 + q小**：高度局部，邻居导向
- **p大 + q大**：全局，随机游走
- **p小 + q大**：同质性优先
- **p大 + q小**：结构性优先

### 2.4 Skip-gram模型

与Word2Vec类似，Node2Vec将游走序列视为"句子"，节点视为"单词"，使用Skip-gram训练：

$$\max_{\theta} \sum_{v \in V} \log P(N_{\theta}(v) | v)$$

其中$N_{\theta}(v)$是softmax定义的邻域：

$$P(u|v) = \frac{exp(\theta_u \cdot \theta_v)}{\sum_{w \in V} exp(\theta_w \cdot \theta_v)}$$

---

## 3. 数学公式与推导

### 3.1 游走概率计算

对于边(v, x)，设其上一节点为t，定义：

- **d_tx = 1**：x是t的直连邻居（返回）
- **d_tx = 0**：x = t（返回起点）
- **d_tx = 2**：其他（远方）

转移概率：
$$w_{vx} = \begin{cases} 1/p & d_{tx} = 0 \\ 1 & d_{tx} = 1 \\ 1/q & d_{tx} = 2 \end{cases}$$

归一化：
$$P(x|v) = \frac{w_{vx}}{Z}$$

其中Z是归一化常数。

### 3.2 代价函数

使用负采样近似：

$$\mathcal{L} = \sum_{v \in V} \sum_{u \in N_R(v)} \log \sigma(\theta_u \cdot \theta_v) + \sum_{u \in N_N(v)} \log \sigma(-\theta_u \cdot \theta_v)$$

其中：
- $N_R(v)$：正样本（游走中相邻）
- $N_N(v)$：负样本（随机采样）
- σ：sigmoid函数

### 3.3 复杂度分析

设：
- r：游走数量
- l：游走长度
- k：负样本数

时间复杂度：O(r × l × (d + kd))

---

## 4. 训练过程讲解

### 4.1 整体流程

```
1. 构建图 G = (V, E)
2. 预计算转移概率
3. 游走生成 r × l 条序列
4. 使用Skip-gram训练嵌入
5. 输出节点嵌入 θ
```

### 4.2 参数配置

```python
# 关键超参数
d = 128          # 嵌入维度
r = 10           # 从每个节点开始的游走数
l = 80           # 游走长度
k = 5            # 负样本数
p = 1.0          # 返回参数
q = 1.0          # 广度参数
window = 10       # Skip-gram窗口大小
workers = 4       # 并行线程数
```

### 4.3 训练代码结构

```python
"""
Node2Vec训练流程
"""
# 1. 构建图
G = nx.Graph()
# 添加边...

# 2. 前处理（计算转移概率）
import node2vec
model = node2vec.Node2Vec(G, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=workers)

# 3. 训练
model.fit(window=window, workers=workers, sg=1)

# 4. 获取嵌入
embeddings = model.wv
```

---

## 5. 应用场景

### 5.1 节点分类

- 社交网络用户分类
- 引文网络论文分类

### 5.2 链接预测

- 朋友推荐
- 商品推荐

### 5.3 其他应用

- 可视化
- 异常检测

---

## 6. 优缺点分析

### 6.1 优点

- **灵活性**：通过p和q控制游走特性
- **可解释性**：参数物理意义明确
- **效果优**：通常优于DeepWalk

### 6.2 缺点

- **参数敏感性**：p和q需要调优
- **非监督**：可能缺少监督信号

---

## 7. 调库实现

### 7.1 使用node2vec库

```bash
pip install node2vec
```

```python
"""
Node2Vec 完整实现
"""
import networkx as nx
import node2vec
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt

# ====================构建图===================
"""
构建示例图：Karate Club网络
或使用真实网络
"""
# 方式1：使用Karate Club
G = nx.karate_club_graph()

# 方式2：自定义图
# G = nx.Graph()
# G.add_edges_from([(0,1), (1,2), ...])

print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")

# ====================Node2Vec模型===================
# 参数配置
dimensions = 64        # 嵌入维度
walk_length = 30        # 游走长度
num_walks = 10         # 每个节点的游走数
workers = 4            # 并行数
p = 1.0               # 返回参数
q = 1.0              # 广度参数

# 构建模型
node2vec_model = Node2Vec(
    G,
    dimensions=dimensions,
    walk_length=walk_length,
    num_walks=num_walks,
    workers=workers,
    p=p,
    q=q,
    quiet=False
)

# ====================训练===================
# 训练方法
# sg=1: Skip-gram (默认)
# sg=0: CBOW
model = node2vec_model.fit(
    window=10,       # 上下文窗口
    min_count=1,     # 最小词频
    sg=1             # 使用Skip-gram
)

print("\n训练完成！")

# ====================获取嵌入===================
# 获取所有节点嵌入
embeddings = {}
for node in G.nodes():
    embeddings[node] = model.wv[str(node)]

print(f"\n嵌入维度: {embeddings[0].shape}")

# ====================节点相似度===================
# 计算节点相似度
similar = model.wv.most_similar('0', topn=5)
print(f"\n与节点0最相似的5个节点:")
for node, sim in similar:
    print(f"  {node}: {sim:.4f}")

# ====================链接预测===================
# 预测未连接的边
def link_prediction(G, model, node1, node2):
    """使用嵌入相似度预测链接"""
    vec1 = model.wv[str(node1)]
    vec2 = model.wv[str(node2)]
    
    # 余弦相似度
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim

# ====================节点分类===================
# 示例：对Karate Club进行社区分类
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 获取真实标签
true_labels = []
for node in G.nodes():
    club = G.nodes[node]['club']
    true_labels.append(0 if club == 'Mr. Hi' else 1)

# K-Means聚类
node_list = list(G.nodes())
X = np.array([model.wv[str(n)] for n in node_list])
kmeans = KMeans(n_clusters=2, random_state=42)
pred_labels = kmeans.fit_predict(X)

# 评估
ari = adjusted_rand_score(true_labels, pred_labels)
print(f"\n聚类ARI分数: {ari:.4f}")

# ====================可视化===================
from sklearn.manifold import TSNE

# 降维
X_all = np.array([model.wv[str(n)] for n in G.nodes()])
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_all)

# 绘制
plt.figure(figsize=(10, 8))
colors = ['blue' if club == 'Mr. Hi' else 'red' for club in nx.get_node_attributes(G, 'club').values()]
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, s=100)
plt.title(f'Node2Vec Embeddings (ARI={ari:.4f})')
plt.savefig('node2vec_embeddings.png')
plt.show()

# ====================不同参数对比===================
"""
对比不同p, q配置
"""
configs = [
    {'p': 1.0, 'q': 0.5, 'name': 'DFS (q<1)'},
    {'p': 1.0, 'q': 2.0, 'name': 'BFS (q>1)'},
    {'p': 0.5, 'q': 1.0, 'name': 'Return (p<1)'},
    {'p': 2.0, 'q': 1.0, 'name': 'No Return (p>1)'},
]

results = {}
for config in configs:
    print(f"\n训练: {config['name']}")
    model = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, 
                     p=config['p'], q=config['q']).fit(window=10)
    
    X = np.array([model.wv[str(n)] for n in G.nodes()])
    pred = KMeans(n_clusters=2, random_state=42).fit_predict(X)
    ari = adjusted_rand_score(true_labels, pred)
    results[config['name']] = ari
    print(f"  ARI: {ari:.4f}")

# 结果对比
print("\n结果对比:")
for name, ari in results.items():
    print(f"  {name}: {ari:.4f}")
```

### 7.2 使用gensim实现

```python
"""
使用gensim实现Node2Vec
"""
from gensim.models import Word2Vec

# 游走生成
def generate_walks(G, num_walks, walk_length, p, q):
    """生成游走序列"""
    walks = []
    
    for node in G.nodes():
        for _ in range(num_walks):
            walk = [node]
            current = node
            
            for _ in range(walk_length - 1):
                # 获取邻居
                neighbors = list(G.neighbors(current))
                
                if not neighbors:
                    break
                
                # 计算概率
                probs = []
                for neighbor in neighbors:
                    # 简化：随机或根据p,q
                    if neighbor == walk[-1]:
                        probs.append(1/p)
                    else:
                        probs.append(1/q)
                
                # 归一化
                probs = np.array(probs)
                probs = probs / probs.sum()
                
                # 采样
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
                current = next_node
            
            walks.append(walk)
    
    return walks

# 生成游走
walks = generate_walks(G, num_walks=10, walk_length=30, p=1.0, q=1.0)

# Word2Vec训练
model = Word2Vec(
    sentences=walks,
    vector_size=64,
    window=10,
    min_count=1,
    sg=1,
    workers=4
)

# 获取嵌入
embeddings = {node: model.wv[str(node)] for node in G.nodes()}
```

---

## 8. 手工代码实现

### 8.1 游走实现

```python
"""
Node2Vec 手工实现
核心：混合游走策略 + Skip-gram训练
"""
import numpy as np
import random
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

# ====================混合游走===================
class Node2VecWalk:
    """混合游走策略实现"""
    def __init__(self, G, p=1.0, q=1.0):
        self.G = G
        self.p = p
        self.q = q
        
        # 预计算转移概率
        self._precompute()
    
    def _precompute(self):
        """预计算边权重"""
        self.alias_nodes = {}
        self.alias_edges = {}
        
        for node in self.G.nodes():
            self.alias_nodes[node] = self._get_transition_probs(node)
    
    def _get_transition_probs(self, node):
        """计算从node出发的转移概率"""
        G = self.G
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            return None
        
        weights = []
        for neighbor in neighbors:
            # 默认权重1
            weight = 1.0
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        
        return (neighbors, weights)
    
    def walk(self, start_node, walk_length):
        """生成单条游走"""
        walk = [start_node]
        current = start_node
        
        for _ in range(walk_length - 1):
            neighbors, probs = self.alias_nodes[current]
            
            if not neighbors:
                break
            
            # 采样下一个节点
            next_idx = np.random.choice(len(neighbors), p=probs)
            next_node = neighbors[next_idx]
            
            walk.append(next_node)
            current = next_node
        
        return walk
    
    def generate_walks(self, num_walks, walk_length, workers=1):
        """生成所有游走"""
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.walk(node, walk_length)
                walks.append(walk)
        
        return walks

# ====================Skip-gram模型===================
class SkipGram(nn.Module):
    """Skip-gram模型"""
    def __init__(self, num_nodes, embedding_dim):
        super(SkipGram, self).__init__()
        
        self.target_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # 初始化
        nn.init.uniform_(self.target_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.zeros_(self.context_embeddings.weight)
    
    def forward(self, target, context):
        """前向传播"""
        target_emb = self.target_embeddings(target)
        context_emb = self.context_embeddings(context)
        
        out = torch.matmul(target_emb, context_emb.T)
        return out

# ====================完整Node2Vec===================
class Node2Vec:
    """完整Node2Vec实现"""
    def __init__(self, G, dimensions=64, walk_length=30, num_walks=10, 
                 p=1.0, q=1.0, window=10, workers=4):
        self.G = G
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.workers = workers
        
        self.node2idx = {n: i for i, n in enumerate(G.nodes())}
        self.idx2node = {i: n for n, i in self.node2idx.items()}
    
    def train(self, epochs=5, lr=0.025):
        """训练"""
        # 生成游走
        walker = Node2VecWalk(self.G, self.p, self.q)
        walks = walker.generate_walks(self.num_walks, self.walk_length)
        
        # 准备训练数据
        train_data = self._create_training_data(walks)
        
        # 模型
        num_nodes = self.G.number_of_nodes()
        self.model = SkipGram(num_nodes, self.dimensions)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练
        for epoch in range(epochs):
            total_loss = 0
            for target, contexts in train_data:
                target_tensor = torch.LongTensor([target])
                context_tensor = torch.LongTensor(contexts)
                
                optimizer.zero_grad()
                out = self.model(target_tensor, context_tensor)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out, 
                    torch.ones_like(out)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_data):.4f}")
    
    def _create_training_data(self, walks):
        """创建训练数据"""
        train_data = []
        
        for walk in walks:
            for i, center in enumerate(walk):
                # 窗口内的节点
                start = max(0, i - self.window)
                end = min(len(walk), i + self.window + 1)
                
                context = [walk[j] for j in range(start, end) if j != i]
                
                if context:
                    train_data.append((center, context))
        
        return train_data
    
    def get_embedding(self, node):
        """获取节点嵌入"""
        idx = self.node2idx[node]
        return self.model.target_embeddings.weight[idx].detach().numpy()
```

---

## 9. 可视化与结果理解

### 9.1 嵌入可视化

```python
"""
Node2Vec嵌入可视化
"""
from sklearn.manifold import TSNE

def visualize(model, G):
    # 获取所有嵌入
    nodes = list(G.nodes())
    embeddings = np.array([model.get_embedding(n) for n in nodes])
    
    # t-SNE降维
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    colors = [G.nodes[n].get('club', 'unknown') for n in nodes]
    color_map = {'Mr. Hi': 'blue', 'Officer': 'red'}
    c = [color_map.get(club, 'gray') for club in colors]
    
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=c)
    plt.title('Node2Vec Embeddings')
    plt.savefig('node2vec_tsne.png')
    plt.show()
```

### 9.2 参数影响可视化

- p值小：局部结构
- q值小：全局结构

---

## 10. 模型评估

### 10.1 多类别分类

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

X = embeddings
y = labels

clf = LogisticRegression()
scores = cross_val_score(clf, X, y, cv=5)
print(f"交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 11. 常见问题与易错点

### 11.1 参数设置

- p、q需要根据任务调整
- 通常p=q=1.0是好的起点

### 11.2 计算问题

- 大规模图需要采样
- 游走数量影响效果

---

## 12. 学习总结

### 12.1 核心要点

- 混合游走：平衡BFS和DFS
- Skip-gram训练
- p控制返回，q控制广度

### 12.2 公式

$$P(x|v) = \frac{1}{Z} \begin{cases} 1/p & d=0 \\ 1 & d=1 \\ 1/q & d=2 \end{cases}$$

---

## 13. 练习题与思考题

### 13.1 选择题

**1. Node2Vec的q参数控制？**
A. 返回程度
B. 广度
C. 游走长度
D. 嵌入维度
**答案：B** 广度参数

**2. 当q<1时倾向于？**
A. BFS
B. DFS
C. 随机
D. 本地
**答案：B** DFS特性

---

## 14. 学习路径建议建议

1. 理解Word2Vec
2. 学习DeepWalk
3. 对比Node2Vec改进
4. 项目实战

---

**学习建议**：Node2Vec将Word2Vec的成功经验引入图学习，理解Skip-gram是关键。