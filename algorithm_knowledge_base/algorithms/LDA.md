# LDA 学习文档

> Latent Dirichlet Allocation，潜在狄利克雷分配，用于主题建模的概率生成模型。

---

## 1. 算法基础认知

### 1.1 一句话定义

LDA（Latent Dirichlet Allocation）是2003年Blei等人提出的三层贝叶斯概率模型，用于对离散数据（如文档集合）进行主题建模，将文档表示为潜在主题的混合分布。

### 1.2 直觉类比

将LDA想象为**图书馆的图书分类**：每本书记载了不同主题的内容（词汇分布），每个书架代表一个主题（主题分布），而图书管理员（贝叶斯推理）需要同时推断书架的排列方式和每本书的主题分配。

### 1.3 历史背景

- **1999年**：Probabilistic LSI提出
- **2003年**：LDA正式发表（Blei, Ng, Jordan）
- **2006年**：Correlated Topic Models
- **2010s**：Online LDA、Stream LDA
- **现在**：NLP、推荐系统广泛使用

### 1.4 算法定位

- **类型**：概率生成模型 -> 主题建模
- **输出**：文档-主题分布、主题-词分布
- **模型类型**：无监督学习/生成模型
- **核心创新**：狄利克雷先验

### 1.5 前置知识

- 概率论基础：贝叶斯定理、分布
- 概率图模型：有向图、因子图
- 变分推断：EM算法基础
- 文本处理：词袋模型

---

## 2. 核心原理

### 2.1 核心思想

LDA的核心思想是将文档表示为主题的混合，同时每个主题是词汇的分布。

**生成过程**（文档d）：
1. 从主题分布 $\theta_d \sim \text{Dir}(\alpha)$ 采样主题混合
2. 对每个词 $w_n$：
   - 从 $z_n \sim \text{Multinomial}(theta_d)$ 采样主题
   - 从 $w_n \sim \text{Multinomial}(\beta_{z_n})$ 采样词汇

### 2.2 模型参数

| 参数 | 说明 | 维度 |
|------|------|------|
| $\alpha$ | 主题分布的狄利克雷先验 | K |
| $\beta$ | 主题-词分布 | K×V |
| $\theta$ | 文档-主题分布 | D×K |
| $\phi$ | 主题-词分布（变量） | K×V |
| $z$ | 主题分配 | D×N |
| $w$ | 观测词汇 | D×N |

### 2.3 概率图模型

LDA的概率图模型（三层）：
```
α → θ_d → z_dn → w_dn
          ↑
β → φ_z_dn
```

### 2.4 与其他模型对比

| 模型 | 先验 | 特点 |
|------|------|------|
| LSI | 高斯 | 线性、连续 |
| pLSA | 无先验 | 不完整生成 |
| LDA | 狄利克雷 | 完整贝叶斯 |

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| D | 文档数 |
| K | 主题数 |
| V | 词汇数 |
| N_d | 文档d的长度 |
| $\alpha$ | 超参数（主题先验） |
| $\beta$ | 超参数（词汇先验） |
| $\theta_d$ | 文档d的主题分布 |
| $\phi_k$ | 主题k的词汇分布 |
| $z_{d,n}$ | 文档d第n词的 topic |
| $w_{d,n}$ | 文档d第n词 |

### 3.2 概率定义

**联合概率**：
$$
p(w, z, \theta, \phi; \alpha, \beta) = \prod_{k=1}^K p(\phi_k; \beta) \prod_{d=1}^D p(\theta_d; \alpha) \prod_{n=1}^{N_d} p(z_{d,n}|\theta_d) p(w_{d,n}|z_{d,n}, \phi)
$$

**似然函数**：
$$
p(w; \alpha, \beta) = \int \int \sum_z p(w, z, \theta, \phi; \alpha, \beta) d\theta d\phi
$$

### 3.3 变分推断

**变分分布**（近似后验）：
$$
q(z, \theta, \phi) = \prod_k q(\phi_k) \prod_d q(\theta_d) \prod_n q(z_{d,n})
$$

**变分目标**：
$$
\log p(w) = \log \int \int \sum_z p(w, z, \theta, \phi) d\theta d\phi
$$

使用ELBO：
$$
\mathcal{L} = \mathbb{E}_q[\log p(w, z, \theta, \phi)] - \mathbb{E}_q[\log q(z, \theta, \phi)]
$$

### 3.4 变分参数更新

**gamma更新**（文档-主题参数）：
$$
\gamma_{d,k} \propto \exp(\psi(\alpha_k) + \sum_n \phi_{w_{d,n},k})
$$

**phi更新**（词-主题参数）：
$$
\phi_{d,n,k} \propto \exp(\psi(\gamma_{d,k}) + \psi(\beta_k) - \psi(\sum_j \beta_j))
$$

### 3.5 EM算法

**E步**：固定 $\alpha, \beta$，更新 $\gamma, \phi$

**M步**：最大化期望，更新 $\alpha, \beta$

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import numpy as np
from collections import Counter

def preprocess_documents(documents):
    """文档预处理"""
    
    # 词汇表构建
    vocab = set()
    for doc in documents:
        vocab.update(doc.split())
    vocab = list(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    # 词索引化
    docs = [[word2idx[w] for doc in documents for w in doc.split()]
    
    return docs, vocab, word2idx


def compute_doc_term_matrix(documents, word2idx):
    """文档-词矩阵"""
    
    D = len(documents)
    V = len(word2idx)
    
    doc_term = np.zeros((D, V))
    
    for d, doc in enumerate(documents):
        for w in doc.split():
            if w in word2idx:
                doc_term[d, word2idx[w]] += 1
    
    return doc_term
```

### 4.2 LDA模型实现

```python
import numpy as np
from scipy.special import gammaln, digamma

class LDA:
    """LDA模型实现"""
    
    def __init__(self, K=10, alpha=0.1, beta=0.1, max_iter=100):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
    
    def _init_params(self, D, V):
        """初始化参数"""
        
        # 变分参数
        self.gamma = np.random.gamma(100, 1./100, (D, self.K))
        self.phi = np.random.dirichlet(np.ones(V) * self.beta, (D, self.K))
        
        # 模型参数
        self.theta = np.zeros((D, self.K))
        self.beta_mat = np.random.dirichlet(np.ones(V) * self.beta, (self.K, V))
    
    def fit(self, documents):
        """训练"""
        
        D = len(documents)
        V = max(max(doc) for doc in documents) + 1
        
        self._init_params(D, V)
        
        # 更新gamma和phi
        for iteration in range(self.max_iter):
            # E步：更新变分参数
            self._update_gamma()
            self._update_phi(documents, V)
            
            # M步：更新模型参数
            self._update_beta(documents, D, V)
            
            # 检查收敛
            if iteration % 10 == 0:
                perplexity = self._compute_perplexity(documents)
                print(f"Iter {iteration}, Perplexity: {perplexity:.2f}")
    
    def _update_gamma(self):
        """更新gamma（文档-主题参数）"""
        
        self.theta = self.gamma / self.gamma.sum(1, keepdims=True)
        
        for d in range(len(self.gamma)):
            for k in range(self.K):
                self.gamma[d, k] = self.alpha + self.phi[d, :, k].sum()
    
    def _update_phi(self, documents, V):
        """更新phi（词-主题参数）"""
        
        for d, doc in enumerate(documents):
            word_counts = Counter(doc)
            
            for k in range(self.K):
                log_phi = np.log(self.theta[d, k) + np.log(self.beta_mat[k, :])
                log_phi = np.exp(log_phi - log_phi.max())
                log_phi = log_phi / log_phi.sum()
                
                for w, c in word_counts.items():
                    self.phi[d, w, k] = log_phi[w] * c
    
    def _update_beta(self, documents, D, V):
        """更新beta（主题-词分布）"""
        
        for k in range(self.K):
            for v in range(V):
                self.beta_mat[k, v] = self.beta
                for d, doc in enumerate(documents):
                    self.beta_mat[k, v] += self.phi[d, v, k]
            
            self.beta_mat[k, :] = self.beta_mat[k, :] / self.beta_mat[k, :].sum()
    
    def _compute_perplexity(self, documents):
        """计算困惑度"""
        
        ll = 0
        total_words = 0
        
        for d, doc in enumerate(documents):
            word_counts = Counter(doc)
            
            for w, c in word_counts.items():
                p_w = np.dot(self.phi[d, w, :], self.theta[d, :])
                ll += c * np.log(p_w + 1e-10)
                total_words += c
        
        return np.exp(-ll / total_words)
    
    def get_topics(self, n_top_words=10):
        """获取主题"""
        
        topics = []
        
        for k in range(self.K):
            top_indices = self.beta_mat[k, :].argsort()[::-1][:n_top_words]
            topics.append(top_indices)
        
        return topics
```

### 4.3 吉布斯采样实现

```python
class LDAGibbs:
    """使用吉布斯采样的LDA"""
    
    def __init__(self, K=10, alpha=0.1, beta=0.01, max_iter=100):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
    
    def fit(self, documents):
        """训练"""
        
        D = len(documents)
        V = max(max(doc) for doc in documents) + 1
        
        # 初始化主题分配
        self.z = [[np.random.randint(self.K) for _ in doc] for doc in documents]
        
        # 计数统计
        self.n_dk = np.zeros((D, self.K)) + self.alpha
        self.n_kv = np.zeros((self.K, V)) + self.beta
        self.n_k = np.zeros(self.K) + V * self.beta
        
        for iteration in range(self.max_iter):
            # 采样主题
            for d, doc in enumerate(documents):
                for n, w in enumerate(doc):
                    k_old = self.z[d][n]
                    self.n_dk[d, k_old] -= 1
                    self.n_kv[k_old, w] -= 1
                    self.n_k[k_old] -= 1
                    
                    # 采样新主题
                    probs = self.n_dk[d] * self.n_kv[:, w] / self.n_k
                    probs = probs / probs.sum()
                    k_new = np.random.choice(self.K, p=probs)
                    
                    self.z[d][n] = k_new
                    self.n_dk[d, k_new] += 1
                    self.n_kv[k_new, w] += 1
                    self.n_k[k_new] += 1
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}")
    
    def get_topics(self, n_words=10):
        """获取主题"""
        
        beta_normalized = self.n_kv / self.n_k[:, np.newaxis]
        
        topics = []
        
        for k in range(self.K):
            top_indices = beta_normalized[k, :].argsort()[::-1][:n_words]
            topics.append(top_indices)
        
        return topics
```

### 4.4 参数推荐

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| K | 主题数 | 10-100 |
| alpha | 主题先验 | 0.1-1.0 |
| beta | 词先验 | 0.01-0.1 |
| max_iter | 最大迭代 | 50-200 |

---

## 5. 应用场景

### 5.1 典型应用

- **主题建模**：文档主题发现
- **文本分类**：无监督分类
- **推荐系统**：文档推荐
- **信息检索**：主题相似度

### 5.2 适用数据

- 文档集合
- 短文本（微博、评论）
- 多文档关联

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| ���解��� | 主题语义清晰 |
| 可扩展 | 词汇可扩展 |
| 生成式 | 可生成新文档 |
| 灵活性 | 可加约束 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算 | 推理慢 |
| 收敛 | 易局部最优 |
| 选择 | K需调参 |
| 稀疏 | 对短文本效果差 |

---

## 7. 调库实现

### 7.1 scikit-learn

```python
from sklearn.decomposition import LatentDirichletAllocation

def use_sklearn_lda():
    """使用sklearn的LDA"""
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    documents = [
        "This is a document about machine learning",
        "Machine learning is a type of artificial intelligence",
        "Natural language processing is related to machine learning",
        "Deep learning is a subset of machine learning",
    ]
    
    # 向量化
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(documents)
    
    # LDA
    lda = LatentDirichletAllocation(
        n_components=2,
        random_state=42,
        max_iter=10
    )
    
    lda.fit(X)
    
    # 主题
    feature_names = vectorizer.get_feature_names()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    return lda, vectorizer
```

### 7.2 Gensim

```python
def use_gensim_lda():
    """使用Gensim的LDA"""
    
    from gensim import corpora
    from gensim.models import LdaModel
    
    documents = [
        ["This", "is", "a", "document", "about", "machine", "learning"],
        ["Machine", "learning", "is", "a", "type", "of", "artificial", "intelligence"],
        ["Natural", "language", "processing", "is", "related", "to", "machine", "learning"],
        ["Deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ]
    
    # 字典
    dictionary = corpora.Dictionary(documents)
    
    # 语料库
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    # LDA
    lda = LdaModel(corpus, id2word=dictionary, num_topics=2)
    
    # 主题
    for topic in lda.print_topics():
        print(topic)
    
    return lda
```

---

## 8. 手工代码实现

### 8.1 简化实现

```python
import numpy as np

class SimpleLDA:
    """简化LDA（基于NMF近似）"""
    
    def __init__(self, K=10):
        self.K = K
        self.components_ = None
    
    def fit(self, X):
        """使用NMF近似"""
        
        from sklearn.decomposition import NMF
        
        nmf = NMF(n_components=self.K, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        # theta = W, phi = H
        self.theta = W / W.sum(1, keepdims=True)
        self.phi = H / H.sum(1, keepdims=True)
        
        return self
    
    def transform(self, X):
        """推断主题"""
        
        from sklearn.decomposition import NMF
        
        W = np.linalg.lstsq(self.phi.T, X.T, rcond=None)[0].T
        W = np.maximum(W, 0)
        W = W / W.sum(1, keepdims=True)
        
        return W
```

### 8.2 可视化主题

```python
import matplotlib.pyplot as plt

def visualize_topics(lda, vocab, n_words=10):
    """主题可视化"""
    
    topics = lda.components_
    
    fig, axes = plt.subplots(2, topics.shape[0]//2, figsize=(12, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < topics.shape[0]:
            indices = topics[i, :].argsort()[::-1][:n_words]
            words = [vocab[j] for j in indices]
            probs = topics[i, indices]
            
            ax.barh(range(n_words), probs)
            ax.set_yticks(range(n_words))
            ax.set_yticklabels(words)
            ax.set_title(f"Topic {i+1}")
    
    plt.tight_layout()
    plt.savefig('topics.png', dpi=150)
    plt.show()
```

---

## 9. 可视化与结果理解

### 9.1 主题分布

```python
def plot_topic_distribution(lda):
    """主题-词分布"""
    
    topics = lda.components_
    K = topics.shape[0]
    
    for k in range(K):
        top_k = topics[k, :].argsort()[::-1][:10]
        
        print(f"Topic {k}:")
        for idx in top_k:
            print(f"  {idx}: {topics[k, idx]:.4f}")
```

### 9.2 文档-主题分布

```python
def plot_doc_topic(lda, X):
    """文档-主题分布"""
    
    doc_topic = lda.transform(X)
    
    plt.figure(figsize=(10, 6))
    
    plt.imshow(doc_topic.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Document')
    plt.ylabel('Topic')
    plt.title('Document-Topic Distribution')
    plt.savefig('doc_topic.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 困惑度

```python
def compute_perplexity(lda, X):
    """困惑度"""
    
    ll = 0
    total = 0
    
    for d in range(X.shape[0]):
        probs = lda.transform(X[d:d+1])
        for w in range(X.shape[1]):
            if X[d, w] > 0:
                ll += X[d, w] * np.log(probs[0, :].dot(lda.components_[:, w]) + 1e-10)
                total += X[d, w]
    
    return np.exp(-ll / total)
```

### 10.2 一致性

```python
def compute_coherence(lda, texts, word2idx):
    """主题一致性"""
    
    topics = lda.components_
    
    coherences = []
    
    for k in range(topics.shape[0]):
        top_words = topics[k, :].argsort()[::-1][:10]
        
        coherence = 0
        for i, w1 in enumerate(top_words):
            for w2 in top_words[i+1:]:
                coherence += np.log(1 + texts[:, w1].dot(texts[:, w2]))
        
        coherences.append(coherence)
    
    return np.mean(coherences)
```

---

## 11. 常见问题与易错点

### 11.1 主题数选择

**方法**：困惑度 + 主题一致性

### 11.2 稀疏文档

**方法**：添加先验或使用动态主题数

---

## 12. 学习总结

### 12.1 核心要点

1. **三层模型**：文档-主题-词
2. **狄利克雷**：先验分布
3. **变分**：推断方法
4. **生成**：完整模型

### 12.2 进阶方向

- **CTM**：关联主题模型
- **DTM**：动态主题模型

---

## 13. 练习题与思考题

### 练习题

**练习1**：LDA与NMF的区别

<details>
<summary>答案</summary>

LDA是概率生成模型，NMF是矩阵分解。LDA更灵活，可加入先验。

</details>

### 思考题

**思考题1**：如何选择K？

<details>
<summary>答案</summary>

使用困惑度+一致性曲线，或领域知识。

</details>

---

## 14. 学习路径建议

### 第一阶段

1. 概率基础
2. 文本处理
3. LDA原理

### 第二阶段

1. 实现变分推断
2. 吉布斯采样
3. 对比实验

### 第三阶段

1. 实际应用
2. 调参优化

### 推荐资源

- **论文**：《Latent Dirichlet Allocation》
- **代码**：Gensim
- **项目**：主题建模

---

*LDA是文本主题建模的基础模型，在NLP领域广泛应用。*
