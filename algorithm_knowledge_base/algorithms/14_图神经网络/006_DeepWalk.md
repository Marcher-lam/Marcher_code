# DeepWalk 学习文档

> DeepWalk，将词向量方法（Skip-gram）应用于图节点嵌入的先驱算法。

---

## 1. 算法基础认知

### 1.1 什么是DeepWalk？

DeepWalk是2014年由Perozzi等人提出的图节点嵌入算法。它首次将自然语言处理中成功的词向量方法（Skip-gram with Hierarchical Softmax）应用于图结构数据，开创了图嵌入学习的新范式。DeepWalk的核心思想是将图中的随机游走序列视为"句子"，节点视为"单词"，然后使用Word2Vec的方法训练节点嵌入。

### 1.2 核心创新

- **随机游走生成**：将图结构转化为序列
- **Skip-gram训练**：使用词向量方法学习嵌入
- **层次softmax**：高效计算softmax

### 1.3 与传统方法对比

| 方面 | 传统方法 | DeepWalk |
|------|---------|---------|
| 矩阵分解 | 需要SVD/特征分解 | 无需 |
| 时间复杂度 | O(N³) | O(N log N) |
| 可扩展性 | 差 | 好 |

### 1.4 历史背景

- **2013年**：Word2Vec提出（BOW, Skip-gram）
- **2014年**：DeepWalk将Word2Vec应用于图
- **2016年**：Node2Vec改进游走策略

---

## 2. 核心原理

### 2.1 随机游走

从每个节点开始，进行固定长度的随机游走：

```python
def random_walk(G, start, length):
    walk = [start]
    current = start
    
    for _ in range(length - 1):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        walk.append(current)
    
    return walk
```

### 2.2 Skip-gram模型

给定中心词，预测上下文词：

$$\max_{\theta} \sum_{c \in C_w} \log P(c|w;\theta)$$

其中P(c|w) = softmax(θ_c · θ_w)

### 2.3 层次 softmax

使用Huffman树加速softmax计算：

- 时间复杂度：O(log N) vs O(N)
- 路径上的二分类：左/右

---

## 3. 数学公式与推导

### 3.1 目标函数

$$\mathcal{L} = \sum_{v \in V} \sum_{u \in N_w(v)} \log P(u|v)$$

其中N_w(v)是游走序列中v的邻居窗口。

### 3.2 条件概率

$$P(u|v) = \frac{exp(\theta_u \cdot \theta_v)}{\sum_{k \in V} exp(\theta_k \cdot \theta_v)}$$

### 3.3 简化计算

使用负采样近似：

$$\log \sigma(\theta_u \cdot \theta_v) + \sum_{i=1}^{k} \mathbb{E}_{k \sim P_n} [\log \sigma(-\theta_k \cdot \theta_v)]$$

---

## 4. 训练过程讲解

### 4.1 整体流程

```
1. 构建图G = (V, E)
2. 随机游走生成序列
3. Skip-gram训练
4. 输出嵌入
```

### 4.2 参数配置

```python
# 关键参数
dimensions = 64      # 嵌入维度
walk_length = 80      # 游走长度
num_walks = 10       # 每个节点游走数
window = 10          # 上下文窗口
min_count = 1        # 最小计数
workers = 4          # 并行数
```

---

## 5. 应用场景

### 5.1 节点分类

- 社交网络分析
- 引文网络

### 5.2 链接预测

---

## 6. 优缺点分析

### 6.1 优点

- 简单有效
- 可扩展
- 无监督

### 6.2 缺点

- 随机游走可能不够高效
- 参数少，控制能力弱

---

## 7. 调库实现

### 7.1 使用gensim

```python
"""
DeepWalk 使用gensim实现
"""
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import random

# ====================构建图===================
# Karate Club
G = nx.karate_club_graph()

print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")

# ====================随机游走生成===================
def generate_random_walks(G, num_walks, walk_length, random_seed=42):
    """
    生成随机游走序列
    
    参数:
        G: 图
        num_walks: 每个节点游走次数
        walk_length: 每次游走长度
        random_seed: 随机种子
    
    返回:
        walks: 游走序列列表
    """
    random.seed(random_seed)
    walks = []
    nodes = list(G.nodes())
    
    for _ in range(num_walks):
        # 打乱节点顺序
        random.shuffle(nodes)
        
        for start_node in nodes:
            walk = [start_node]
            current = start_node
            
            for _ in range(walk_length - 1):
                # 获取邻居
                neighbors = list(G.neighbors(current))
                
                if not neighbors:
                    break
                
                # 随机选择下一个节点
                current = random.choice(neighbors)
                walk.append(current)
            
            walks.append(walk)
    
    return walks

# 生成游走
walks = generate_random_walks(G, num_walks=10, walk_length=80)
print(f"生成的游走序列数: {len(walks)}")
print(f"示例游走: {walks[0][:10]}...")

# ====================使用gensim训练===================
from gensim.models import Word2Vec

# 将节点编号转为字符串（gensim要求）
walks_str = [[str(n) for n in walk] for walk in walks]

# 训练Word2Vec模型
model = Word2Vec(
    sentences=walks_str,
    vector_size=64,       # 嵌入维度
    window=10,           # 上下文窗口
    min_count=1,         # 最小词频
    sg=1,               # Skip-gram
    workers=4,          # 并行数
    epochs=5             # 训练轮数
)

print(f"\n训练完成！")
print(f"词汇表大小: {len(model.wv)}")

# ====================获取嵌入===================
# 方式1: 通过模型获取
embedding_1 = model.wv['0']
print(f"节点嵌入维度: {embedding_1.shape}")

# 方式2: 获取所有节点嵌入
embeddings = np.array([model.wv[str(n)] for n in G.nodes())
embeddings = np.array(embeddings)
print(f"嵌入矩阵形状: {embeddings.shape}")

# ====================相似节点查询===================
similar_nodes = model.wv.most_similar('0', topn=5)
print(f"\n与节点0最相似的5个节点:")
for node, sim in similar_nodes:
    print(f"  节点{node}: {sim:.4f}")

# ====================链接预测===================
def link_prediction(model, node1, node2):
    """
    基于嵌入的链接预测
    使用余弦相似度
    """
    vec1 = model.wv[str(node1)]
    vec2 = model.wv[str(node2)]
    
    # 余弦相似度
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    cosine_sim = dot / (norm1 * norm2)
    return cosine_sim

# 计算示例
sim_0_1 = link_prediction(model, 0, 1)
print(f"\n节点0与节点1的相似度: {sim_0_1:.4f}")

# ====================节点分类（聚类）===================
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
# t-SNE降维
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10, 8))

# 颜色
colors = []
for node in G.nodes():
    club = G.nodes[node]['club']
    colors.append('blue' if club == 'Mr. Hi' else 'red')

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100, alpha=0.6)

# 标注节点号
for i, node in enumerate(node_list):
    plt.annotate(str(node), (X_2d[i, 0], X_2d[i, 1]), fontsize=9)

plt.title(f'DeepWalk Embeddings (Karate Club, ARI={ari:.4f})')
plt.savefig('deepwalk_embeddings.png')
plt.show()

# ====================不同参数对比===================
print("\n" + "="*50)
print("different parameter configurations:")
print("="*50)

# 参数配置
configs = [
    {'vector_size': 32, 'window': 5, 'epochs': 3},
    {'vector_size': 64, 'window': 10, 'epochs': 5},
    {'vector_size': 128, 'window': 10, 'epochs': 5},
]

results = []
for config in configs:
    print(f"\nConfig: {config}")
    
    # 重新训练
    model = Word2Vec(
        sentences=walks_str,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=1,
        sg=1,
        workers=4,
        epochs=config['epochs']
    )
    
    # 嵌入
    X = np.array([model.wv[str(n)] for n in node_list])
    
    # 聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    pred = kmeans.fit_predict(X)
    ari = adjusted_rand_score(true_labels, pred)
    
    results.append((config, ari))
    print(f"  ARI: {ari:.4f}")

best_config, best_ari = max(results, key=lambda x: x[1])
print(f"\nBest config: {best_config}, ARI: {best_ari:.4f}")
```

### 7.2 使用 karate-community 库

```python
# alternative: 使用专门的库
try:
    from karateclub import DeepWalk
    
    model = DeepWalk(dimensions=64, walk_length=80, num_walks=10)
    model.fit(G)
    
    embeddings = model.get_embedding()
except ImportError:
    print("Install karateclub: pip install karateclub")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
"""
DeepWalk 手工实现
不依赖gensim
"""
import numpy as np
import random
from collections import defaultdict

class SimpleDeepWalk:
    """简化的DeepWalk实现"""
    
    def __init__(self, dimensions=64, walk_length=80, num_walks=10, window=10):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
    
    def fit(self, G):
        """训练"""
        self.G = G
        self.node2idx = {n: i for i, n in enumerate(G.nodes())}
        
        # 游走
        walks = self._generate_walks()
        
        # 训练（简化：使用共现矩阵）
        self.embeddings = self._train_with_cooccurrence(walks)
        
        return self
    
    def _generate_walks(self):
        """生成随机游走"""
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                current = node
                
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.G.neighbors(current))
                    if not neighbors:
                        break
                    current = random.choice(neighbors)
                    walk.append(current)
                
                walks.append(walk)
        
        return walks
    
    def _train_with_cooccurrence(self, walks):
        """使用共现矩阵训练"""
        n = len(self.node2idx)
        
        # 构建共现矩阵
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for walk in walks:
            for i, center in enumerate(walk):
                start = max(0, i - self.window)
                end = min(len(walk), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = walk[j]
                        cooccurrence[center][context] += 1
        
        # SVD降维
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds
        
        # 构建稀疏矩阵
        mat = lil_matrix((n, n))
        for i, neighbors in cooccurrence.items():
            for j, count in neighbors.items():
                mat[self.node2idx[i], self.node2idx[j]] = count
        
        mat = mat.tocsr()
        
        # SVD
        U, S, V = svds(mat, k=self.dimensions)
        
        # 嵌入 = U * sqrt(S)
        embeddings = U * np.sqrt(S)
        
        return embeddings
    
    def get_embedding(self, node):
        """获取嵌入"""
        idx = self.node2idx[node]
        return self.embeddings[idx]

# ====================使用示例===================
# model = SimpleDeepWalk(dimensions=64)
# model.fit(G)
# embedding = model.get_embedding(0)
```

---

## 9. 可视化与结果理解

### 9.1 嵌入可视化

```python
# Already shown in section 7.1
```

### 9.2 结果解读

- ARI > 0.8: 很好的社区发现
- 嵌入空间分离明显

---

## 10. 模型评估

### 10.1 评估指标

```python
# 链接预测
from sklearn.metrics import roc_auc_score

# 节点分类
from sklearn.model_selection import cross_val_score
```

---

## 11. 常见问题与易错点

### 11.1 参数设置

- walk_length: 80-100常见
- num_walks: 10-20常见
- dimensions: 64-128常见

### 11.2 图类型

- 无向图效果更好
- 稀疏图需要更多游走

---

## 12. 学习总结

### 12.1 核心要点

- 随机游走 + Skip-gram
- 无监督学习
- 可扩展性强

### 12.2 关键公式

$$\mathcal{L} = \sum_{v \in V} \sum_{u \in N_w(v)} \log P(u|v)$$

---

## 13. 练习题与思考题

### 13.1 选择题

**1. DeepWalk的核心思想来自？**
A. CNN
B. RNN
C. Word2Vec
D. Transformer
**答案：C** Word2Vec

**2. DeepWalk的游走是？**
A. BFS
B. DFS
C. 随机游走
D. 确定性游走
**答案：C**

---

## 14. 学习路径建议建议

1. 学习Word2Vec
2. 理解Skip-gram
3. 对比DeepWalk和Node2Vec
4. 项目实战

---

**学习建议**：DeepWalk���图���入的里程碑，理解Word2Vec是理解DeepWalk的关键。