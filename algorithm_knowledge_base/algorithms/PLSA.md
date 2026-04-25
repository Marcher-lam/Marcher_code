# PLSA 学习文档

> Probabilistic Latent Semantic Analysis (PLSA) 是一种基于概率图模型的文本主题挖掘算法，通过引入潜在变量建模文档-词语的共现关系，是LDA的前身，在主题模型发展史上具有里程碑意义。

---

## 1. 算法基础认知

### 一句话定义

PLSA是一种利用概率潜在语义索引进行文档主题挖掘的无监督学习方法，通过引入潜在主题变量 $z$ 建立文档 $d$ 、词语 $w$ 之间的条件概率关系，实现语义空间的降维表示。

### 直觉类比

想象你在图书馆整理书籍：虽然每本书（文档）包含很多词汇，但你可以通过阅读理解将每本书归纳为几个主题（如历史、科学、文学）。PLSA做的事情类似——它不知道有哪些主题，需要从数据中自动发现这些"隐藏的主题"。一个文档可以同时涉及多个主题，一篇文章可能30%讲历史、50%讲科学、20%讲文学。

### 历史背景

- **1999年**：Thomas Hofmann 提出 PLSA（Probabilistic Latent Semantic Analysis）
- **2003年**：David Blei 在 PLSA 基础上提出 LDA（Latent Dirichlet Allocation），加入先验分布
- PLSA 是主题模型的开创性工作，为后续 LDA、NMF 等方法奠定基础

### 算法定位

- **类型**：概率生成模型 / 主题模型
- **输出**：文档-主题分布 $\theta_d$、主题-词分布 $\phi_z$、潜在主题表示
- **模型类型**：生成式模型（与判别式模型如 SVM 对应）
- **所属类别**：无监督学习 → 主题挖掘

### 前置知识

学习 PLSA 需要具备以下基础：
- 条件概率与贝叶斯定理：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- 最大似然估计（MLE）：参数估计的经典方法
- 期望最大化算法（EM）：PLSA 的核心优化方法
- 矩阵分解的基本思想：理解 SVD/PCA 的直观含义

---

## 2. 核心原理

### 2.1 核心思想

PLSA 的核心思想是引入**潜在变量（latent variable）** $z$ 表示"隐藏的主题"，建模文档和词汇之间的桥梁。传统的方法直接建模 $P(w|d)$，但这种建模方式无法解释"为什么词语和文档有关联"。PLSA 假设：

$$P(w|d) = \sum_{z=1}^{K} P(w|z)P(z|d)$$

即：文档 $d$ 产生词语 $w$ 的过程是，先选择一个主题 $z$，再从该主题中生成词语。

### 2.2 工作流程

1. **数据输入**：文档-词语共现矩阵 $N(d,w)$
2. **参数初始化**：随机初始化 $P(z|d)$ 和 $P(w|z)$
3. **E步**：根据当前参数估计隐变量 $z$ 的后验分布
4. **M步**：根据 E 步结果更新模型参数
5. **迭代**：重复 E-M 步直到收敛
6. **输出**：主题-词分布 $\phi$、文档-主题分布 $\theta$

### 2.3 关键概念解释

| 概念 | 符号 | 含义 |
|------|------|------|
| 文档 | $d$ | 文本语料库中的第 $d$ 篇文档 |
| 词语 | $w$ | 词汇表中的第 $w$ 个词 |
| 潜在主题 | $z$ | 隐藏的语义主题，共 $K$ 个 |
| 文档-主题分布 | $\theta_d = P(z|d)$ | 文档 $d$ 包含主题 $z$ 的概率 |
| 主题-词分布 | $\phi_z = P(w|z)$ | 主题 $z$ 中产生词语 $w$ 的概率 |
| 共现计数 | $n(d,w)$ | 词语 $w$ 在文档 $d$ 中出现的次数 |

### 2.4 几何/直观解释

从几何角度理解 PLSA：
- 原始文档空间：高维（词汇表大小 $V$ 维），稀疏
- 潜在主题空间：低维（主题数 $K$ 维），稠密
- PLSA 实现的功能：$R^{|V|} \rightarrow R^{K}$ 的降维映射

每个文档 $d$ 可以表示为 $K$ 维的主题向量，这相当于在"主题坐标系"下的新表示。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $D$ | 文档总数 | 标量 |
| $W$ | 词汇表大小 | 标量 |
| $K$ | 潜在主题数 | 标量 |
| $n(d,w)$ | 词语 $w$ 在文档 $d$ 中的词频 | 标量 |
| $P(w|z)$ | 主题 $z$ 生成词 $w$ 的概率 | $|W| \times K$ |
| $P(z|d)$ | 文档 $d$ 包含主题 $z$ 的概率 | $K \times D$ |

### 3.2 问题形式化

**生成过程**（PLSA 的概率图模型）：

1. 对于每篇文档 $d$：
   - 以概率 $P(z|d)$ 选择主题 $z$
2. 给定主题 $z$：
   - 以概率 $P(w|z)$ 生成词语 $w$

**联合概率**：
$$P(d, w) = P(d) \sum_{z} P(w|z)P(z|d)$$

**目标**：最大化观测数据的对数似然：
$$\mathcal{L} = \sum_{d} \sum_{w} n(d,w) \log \sum_{z} P(w|z)P(z|d)$$

### 3.3 目标函数/损失函数

**对数似然函数**：
$$\mathcal{L}(\Phi, \Theta) = \sum_{d=1}^{D} \sum_{w=1}^{W} n(d,w) \log\left(\sum_{z=1}^{K} \phi_{zw} \theta_{dz}\right)$$

其中：
- $\phi_{zw} = P(w|z)$，满足 $\sum_{w} \phi_{zw} = 1$
- $\theta_{dz} = P(z|d)$，满足 $\sum_{z} \theta_{dz} = 1$

### 3.4 推导过程

**问题**：直接优化 $\mathcal{L}$ 有困难，因为 $\log$ 里有求和。

**EM 算法推导**：

**E步：计算隐变量的后验概率**

利用贝叶斯定理：
$$P(z|d,w) = \frac{P(w|z)P(z|d)}{\sum_{z'} P(w|z')P(z'|d)}$$

展开写：
$$P(z|d,w) = \frac{\phi_{zw} \theta_{dz}}{\sum_{z'=1}^{K} \phi_{z'w} \theta_{dz'}}$$

**M步：更新参数**

通过最大化期望完全数据的对数似然：

对 $\theta_{dz}$：
$$\theta_{dz} \leftarrow \frac{\sum_{w} n(d,w) P(z|d,w)}{\sum_{w} n(d,w) \sum_{z'} P(z'|d,w)}$$

对 $\phi_{zw}$：
$$\phi_{zw} \leftarrow \frac{\sum_{d} n(d,w) P(z|d,w)}{\sum_{d'} n(d',w) \sum_{z''} P(z''|d',w)}$$

**推导细节**：

完全数据的对数似然（Q 函数）：
$$Q = \sum_{d} \sum_{w} n(d,w) \sum_{z} P(z|d,w) [\log \phi_{zw} + \log \theta_{dz}]$$

使用拉格朗日乘数法，约束 $\sum_{w} \phi_{zw} = 1$ 和 $\sum_{z} \theta_{dz} = 1$：

对 $\phi_{zw}$：
$$\frac{\partial}{\partial \phi_{zw}} \left[ Q + \lambda_\phi (\sum_w \phi_{zw} - 1) \right] = 0$$

$$\frac{\sum_{d} n(d,w) P(z|d,w)}{\phi_{zw}} + \lambda_\phi = 0$$

$$\phi_{zw} = -\frac{\sum_{d} n(d,w) P(z|d,w)}{\lambda_\phi}$$

利用约束 $\sum_w \phi_{zw} = 1$ 求得 $\lambda_\phi$：
$$-\sum_w \frac{\sum_{d} n(d,w) P(z|d,w)}{\lambda_\phi} = 1$$

$$\lambda_\phi = -\sum_w \sum_d n(d,w) P(z|d,w)$$

最终得到：
$$\phi_{zw} = \frac{\sum_{d} n(d,w) P(z|d,w)}{\sum_{w'} \sum_{d} n(d,w') P(z|d,w')}$$

同理可得 $\theta_{dz}$ 的更新公式。

### 3.5 最终解/算法步骤

**EM 算法伪代码**：

```
输入：文档-词频矩阵 n(d,w)，主题数 K
输出：P(w|z), P(z|d)

1. 初始化：
   - 随机初始化 P(w|z) > 0，满足行和为1
   - 随机初始化 P(z|d) > 0，满足行和为1

2. 迭代（直到收敛）：
   // E步：计算隐变量后验
   for each d, w:
       P(z|d,w) = P(w|z)P(z|d) / Σ_z' P(w|z')P(z'|d)
   
   // M步：更新参数
   for each z, w:
       P(w|z) = Σ_d n(d,w) P(z|d,w) / Σ_w' Σ_d n(d,w') P(z|d,w')
   
   for each d, z:
       P(z|d) = Σ_w n(d,w) P(z|d,w) / Σ_z' Σ_w n(d,w) P(z'|d,w)

3. 返回 P(w|z), P(z|d)
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

1. **文本清洗**：去除停用词、标点、数字
2. **分词**：中文使用 jieba，英文使用空格
3. **构建词汇表**：过滤低频词（出现次数 < 5）
4. **构建词频矩阵**：
   $$n(d,w) = \text{文档 } d \text{ 中词 } w \text{ 出现的次数}$$

### 4.2 参数初始化建议

常用的初始化方法：
- **随机初始���**：均匀分布 + 归一化
- **SVD 初始化**：对文档-词矩阵做截断 SVD，用结果初始化（收敛更快）
- **共轭初始化**：$\phi_{zw} \propto \sum_d n(d,w)$，$\theta_{dz} \propto 1/K$

### 4.3 迭代过程代码（Python + NumPy）

```python
import numpy as np
from scipy.sparse import csr_matrix

class PLSA:
    """
    Probabilistic Latent Semantic Analysis (PLSA)
    
    使用 EM 算法进行参数学习，实现文档-主题挖掘。
    """
    
    def __init__(self, n_topics=10, max_iter=100, tol=1e-6):
        """
        初始化 PLSA 模型
        
        Args:
            n_topics: 潜在主题数 K
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.phi = None  # P(w|z): (vocab_size, n_topics)
        self.theta = None  # P(z|d): (n_topics, n_docs)
        
    def fit(self, doc_term_matrix):
        """
        训练 PLSA 模型
        
        Args:
            doc_term_matrix: 文档-词频矩阵，shape (n_docs, vocab_size)
                            可以是 dense array 或 sparse matrix
        """
        n_docs, vocab_size = doc_term_matrix.shape
        
        # 初始化参数 (使用均匀分布 + 噪声)
        self.phi = np.random.rand(vocab_size, self.n_topics)
        self.phi = self.phi / self.phi.sum(axis=0, keepdims=True)
        
        self.theta = np.random.rand(self.n_topics, n_docs)
        self.theta = self.theta / self.theta.sum(axis=0, keepdims=True)
        
        # 迭代优化
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            # E步：计算后验概率 P(z|d,w)
            # P(z|d,w) ∝ P(w|z) * P(z|d)
            log_posterior = np.log(self.phi[:, :, None] * self.theta[None, :, :])
            # shape: (vocab_size, n_topics, n_docs)
            
            # 使用 log-sum-exp trick 避免数值溢出
            log_posterior_max = log_posterior.max(axis=1, keepdims=True)
            log_posterior_normalized = log_posterior - log_posterior_max
            posterior = np.exp(log_posterior_normalized)
            posterior = posterior / posterior.sum(axis=1, keepdims=True)
            
            # 计算对数似然
            ll = 0.0
            for d in range(n_docs):
                for w in range(vocab_size):
                    if doc_term_matrix[d, w] > 0:
                        ll += doc_term_matrix[d, w] * np.log(
                            (self.phi[w, :] @ self.theta[:, d]) + 1e-10
                        )
            
            # 检查收敛
            if ll - prev_ll < self.tol:
                print(f"收敛于第 {iteration} 次迭代，对数似然: {ll:.4f}")
                break
            prev_ll = ll
            
            # M步：更新参数
            # 更新 phi: P(w|z)
            phi_new = np.zeros((vocab_size, self.n_topics))
            for w in range(vocab_size):
                for z in range(self.n_topics):
                    phi_new[w, z] = 0.0
                    for d in range(n_docs):
                        phi_new[w, z] += doc_term_matrix[d, w] * posterior[w, z, d]
            
            # 归一化
            phi_sum = phi_new.sum(axis=0, keepdims=True)
            phi_sum[phi_sum == 0] = 1  # 避免除零
            self.phi = phi_new / phi_sum
            
            # 更新 theta: P(z|d)
            theta_new = np.zeros((self.n_topics, n_docs))
            for d in range(n_docs):
                for z in range(self.n_topics):
                    theta_new[z, d] = 0.0
                    for w in range(vocab_size):
                        theta_new[z, d] += doc_term_matrix[d, w] * posterior[w, z, d]
            
            # ���一化
            theta_sum = theta_new.sum(axis=0, keepdims=True)
            theta_sum[theta_sum == 0] = 1
            self.theta = theta_new / theta_sum
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 对数似然: {ll:.4f}")
        
        return self
    
    def get_top_words(self, topic_id, n_words=10):
        """
        获取指定主题的 top-n 关键词
        
        Args:
            topic_id: 主题索引
            n_words: 返回的词语数量
            
        Returns:
            top_words: 关键词列表及概率
        """
        word_probs = self.phi[:, topic_id]
        top_indices = np.argsort(word_probs)[::-1][:n_words]
        return [(idx, word_probs[idx]) for idx in top_indices]
    
    def get_document_topics(self, doc_id):
        """
        获取文档的主题分布
        
        Args:
            doc_id: 文档索引
            
        Returns:
            topic_dist: 主题概率分布
        """
        return self.theta[:, doc_id]
    
    def transform(self, doc_term_matrix):
        """
        获取文档的主题表示
        
        Args:
            doc_term_matrix: 文档-词频矩阵
            
        Returns:
            doc_topics: 文档-主题分布
        """
        # 使用当前的 phi 推断文档主题
        doc_topics = self.phi.T @ doc_term_matrix.T
        doc_topics = doc_topics / (doc_topics.sum(axis=0, keepdims=True) + 1e-10)
        return doc_topics.T


def demo_plsa():
    """演示 PLSA 的使用"""
    np.random.seed(42)
    
    # 创建模拟数据：3个主题，每个主题生成20篇文档
    n_docs = 60
    vocab_size = 100
    n_topics = 3
    
    # 定义主题-词分布
    topic_word_probs = np.random.rand(vocab_size, n_topics)
    topic_word_probs = topic_word_probs / topic_word_probs.sum(axis=0, keepdims=True)
    
    # 生成文档
    doc_term_matrix = np.zeros((n_docs, vocab_size))
    
    for d in range(n_docs):
        # 每个文档对应一个主题（简化）
        topic_id = d % n_topics
        # 生成 50-150 个词
        n_words = np.random.randint(50, 151)
        for _ in range(n_words):
            word_id = np.random.choice(vocab_size, p=topic_word_probs[:, topic_id])
            doc_term_matrix[d, word_id] += 1
    
    print("=" * 60)
    print("PLSA 主题模型演示")
    print("=" * 60)
    print(f"文档数: {n_docs}, 词汇表大小: {vocab_size}, 主题数: {n_topics}")
    
    # 训练 PLSA
    plsa = PLSA(n_topics=n_topics, max_iter=50)
    plsa.fit(doc_term_matrix)
    
    # 展示每个主题的关键词
    print("\n" + "=" * 60)
    print("发现的主题：")
    print("=" * 60)
    
    for topic_id in range(n_topics):
        print(f"\n主题 {topic_id}:")
        # 简化的词语索引
        words = plsa.get_top_words(topic_id, n_words=10)
        for idx, prob in words:
            print(f"  词{idx}: {prob:.4f}")
    
    # 展示文档的主题分布
    print("\n" + "=" * 60)
    print("文档的主题分布示例：")
    print("=" * 60)
    
    for doc_id in [0, 20, 40]:
        topic_dist = plsa.get_document_topics(doc_id)
        print(f"文档 {doc_id}: {topic_dist}")


if __name__ == "__main__":
    demo_plsa()
```

### 4.4 收敛条件

常用的收敛判断方法：
1. **对数似然变化**：$|\mathcal{L}_{t} - \mathcal{L}_{t-1}| < \epsilon$
2. **参数变化**：$\max |\phi_t - \phi_{t-1}| < \epsilon$
3. **最大迭代次数**：防止无限循环

通常使用组合条件：任意一个满足即停止。

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| K（主题数） | 控制主题粒度 | 5-100，根据文档数调整 | 10 |
| max_iter | 防止无限迭代 | 50-200 | 100 |
| tol | 收敛精度 | 1e-4 ~ 1e-8 | 1e-6 |
| 初始化方法 | 影响收敛速度 | random/svd/uniform | random |

---

## 5. 应用场景

### 5.1 典型应用

1. **文档主题挖掘**：
   - 对新闻文章进行主题分类
   - 提取论文摘要的核心主题
   - 分析用户评论/反馈的主题分布

2. **文本表示学习**：
   - 将文档映射到低维主题空间
   - 作为下游分类器的输入特征

3. **推荐系统**：
   - 基于主题的文档相似度计算
   - 用户兴趣建模

4. **信息检索**：
   - 语义空间的信息匹配
   - 查询扩展

### 5.2 适用数据特征

- 文本数据：需要一定规模的语料（建议 > 100 篇文档）
- 词汇表：过滤后大小在 1000-50000 较为合适
- 主题可解释性：数据中存在明显的主题结构时效果更好

### 5.3 不适用场景

- 短文本（微博、评论）：数据稀疏，容易过拟合
- 主题不明确：没有清晰的潜在结构
- 大规模语料：计算复杂度较高（$O(D \times W \times K)$）

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 可解释性强 | 主题-词分布可直观理解 | 主题数适中 |
| 无监督 | 无需人工标注 | 数据存在潜在结构 |
| 生成式 | 可用于文本生成 | 参数收敛 |
| 理论基础 | 概率框架，理论完备 | - |
| 降维能力 | 高维 → 低维语义空间 | K << W |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 参数随文档数线性增长 | $P(z\|d)$ 参数量 D×K | 使用 LDA（加入先验） |
| 容易过拟合 | 缺乏正则化 | 早停、调整 K |
| 局部最优 | EM 只能找到局部最优 | 多次初始化 |
| 计算复杂度 | $O(I \times D \times W \times K)$ | 采样近似、并行化 |
| 无法泛化到新��档 | 需要重新训练 | 使用 LDA |

---

## 7. 调库实现（Python + 完整代码 + 注释）

虽然 scikit-learn 没有直接实现 PLSA，但可以使用 `sklearn.decomposition.NMF` 或 `LatentDirichletAllocation` 作为替代。以下展示使用 gensim 库实现主题模型：

```python
import numpy as np
from gensim import corpora
from gensim.models import LdaModel

class PLSAWrapper:
    """
    使用 gensim 的 LDA 作为 PLSA 的近似实现
    
    LDA 可以看作是加入狄利克雷先验的 PLSA，
    当先验参数很大时，两者行为相近。
    """
    
    def __init__(self, n_topics=10, max_iter=50, alpha='auto'):
        """
        初始化
        
        Args:
            n_topics: 主题数 K
            max_iter: 最大迭代次数
            alpha: 文档-主题先验，'auto' 表示自动选择
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.model = None
        self.dictionary = None
        
    def fit(self, texts):
        """
        训练模型
        
        Args:
            texts: 文本列表，每项是分词后的词语列表
                   例如: [['word1', 'word2'], ['word3', 'word4']]
        """
        # 构建词典
        self.dictionary = corpora.Dictionary(texts)
        
        # 过滤极端词频的词
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # 创建词袋模型
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # 训练 LDA（可看作是带先验的 PLSA）
        self.model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            iterations=self.max_iter,
            alpha=self.alpha,
            passes=1
        )
        
        return self
    
    def get_topics(self, num_words=10):
        """
        获取每个主题的 top-n 词语
        
        Args:
            num_words: 每个主题返回的词语数
            
        Returns:
            topics: 主题列表，每个主题是 (词, 概率) 对的列表
        """
        topics = []
        for topic_id in range(self.n_topics):
            words = self.model.show_topic(topic_id, topn=num_words)
            topics.append(words)
        return topics
    
    def get_document_topics(self, text):
        """
        获取文档的主题分布
        
        Args:
            text: 分词后的文本列表
            
        Returns:
            topic_dist: 主题-概率分布
        """
        bow = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(bow)
        return dict(topics)


def demo_plsa_with_gensim():
    """演示使用 gensim 实现主题模型"""
    
    # 示例文档
    documents = [
        ['bank', 'account', 'money', 'loan', 'interest', 'credit'],
        ['bank', 'river', 'water', 'bridge', 'flow'],
        ['government', 'political', 'policy', 'election', 'vote'],
        ['economy', 'market', 'stock', 'investment', 'trade'],
        ['river', 'fish', 'ocean', 'sea', 'water'],
    ]
    
    print("=" * 60)
    print("使用 gensim 实现主题模型 (LDA/PLSA)")
    print("=" * 60)
    
    # 训练模型
    model = PLSAWrapper(n_topics=2, max_iter=100)
    model.fit(documents)
    
    # 展示主题
    print("\n发现的主题：")
    topics = model.get_topics(num_words=5)
    for i, topic in enumerate(topics):
        print(f"\n主题 {i}:")
        for word, prob in topic:
            print(f"  {word}: {prob:.4f}")
    
    # 查询新文档的主题
    print("\n新文档的主题分布：")
    new_doc = ['bank', 'money', 'loan']
    topic_dist = model.get_document_topics(new_doc)
    for topic_id, prob in topic_dist.items():
        print(f"  主题 {topic_id}: {prob:.4f}")


if __name__ == "__main__":
    demo_plsa_with_gensim()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def plsa_EM(doc_term_matrix, n_topics, max_iter=100, tol=1e-6, verbose=True):
    """
    PLSA 的 EM 算法实现
    
    参数:
        doc_term_matrix: 文档-词频矩阵，shape (n_docs, vocab_size)
        n_topics: 潜在主题数 K
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印训练过程
        
    返回:
        phi: P(w|z), shape (vocab_size, n_topics)
        theta: P(z|d), shape (n_topics, n_docs)
    """
    n_docs, vocab_size = doc_term_matrix.shape
    
    # ==================== 步骤1：参数初始化 ====================
    # 使用均匀分布 + 小随机噪声，避免零概率
    phi = np.random.rand(vocab_size, n_topics) + 0.01
    phi = phi / phi.sum(axis=0, keepdims=True)
    
    theta = np.random.rand(n_topics, n_docs) + 0.01
    theta = theta / theta.sum(axis=0, keepdims=True)
    
    prev_ll = -np.inf
    
    for iteration in range(max_iter):
        # ==================== E步：计算后验概率 ====================
        # P(z|d,w) ∝ P(w|z) * P(z|d)
        # shape: (vocab_size, n_topics, n_docs)
        joint = phi[:, :, None] * theta[None, :, :]
        
        # 归一化得到后验概率
        posterior = joint / (joint.sum(axis=1, keepdims=True) + 1e-10)
        
        # ==================== 计算对数似然 ====================
        ll = 0.0
        for d in range(n_docs):
            for w in range(vocab_size):
                n = doc_term_matrix[d, w]
                if n > 0:
                    prob = (phi[w, :] @ theta[:, d]) + 1e-10
                    ll += n * np.log(prob)
        
        # ==================== 检查收敛 ====================
        if verbose and iteration % 10 == 0:
            print(f"迭代 {iteration}: 对数似然 = {ll:.4f}")
        
        if abs(ll - prev_ll) < tol:
            if verbose:
                print(f"第 {iteration} 次迭代收敛")
            break
        prev_ll = ll
        
        # ==================== M步：更新参数 ====================
        
        # 更新 phi: P(w|z)
        # phi(z,w) ∝ Σ_d n(d,w) * P(z|d,w)
        phi_new = np.zeros((vocab_size, n_topics))
        for w in range(vocab_size):
            for z in range(n_topics):
                phi_new[w, z] = 0.0
                for d in range(n_docs):
                    phi_new[w, z] += doc_term_matrix[d, w] * posterior[w, z, d]
        
        # 归一化：保证 Σ_w P(w|z) = 1
        phi_sum = phi_new.sum(axis=0, keepdims=True)
        phi_sum[phi_sum == 0] = 1
        phi = phi_new / phi_sum
        
        # 更新 theta: P(z|d)
        # theta(z,d) ∝ Σ_w n(d,w) * P(z|d,w)
        theta_new = np.zeros((n_topics, n_docs))
        for d in range(n_docs):
            for z in range(n_topics):
                theta_new[z, d] = 0.0
                for w in range(vocab_size):
                    theta_new[z, d] += doc_term_matrix[d, w] * posterior[w, z, d]
        
        # 归一化：保证 Σ_z P(z|d) = 1
        theta_sum = theta_new.sum(axis=0, keepdims=True)
        theta_sum[theta_sum == 0] = 1
        theta = theta_new / theta_sum
    
    return phi, theta


# ==================== 演示代码 ====================
if __name__ == "__main__":
    np.random.seed(42)
    
    # 创建模拟数据
    n_docs = 50
    vocab_size = 80
    n_topics = 3
    
    # 定义三个主题的词分布
    topic_word_dist = np.random.rand(vocab_size, n_topics)
    topic_word_dist = topic_word_dist / topic_word_dist.sum(axis=0, keepdims=True)
    
    # 生成文档
    doc_term_matrix = np.zeros((n_docs, vocab_size))
    for d in range(n_docs):
        topic_id = d % n_topics
        n_words = np.random.randint(30, 80)
        for _ in range(n_words):
            word_id = np.random.choice(vocab_size, p=topic_word_dist[:, topic_id])
            doc_term_matrix[d, word_id] += 1
    
    print("=" * 60)
    print("PLSA 手工实现演示")
    print("=" * 60)
    print(f"文档数: {n_docs}, 词汇表大小: {vocab_size}, 主题数: {n_topics}")
    
    # 训练 PLSA
    phi, theta = plsa_EM(doc_term_matrix, n_topics, max_iter=50)
    
    # 展示结果
    print("\n" + "=" * 60)
    print("各主题的关键词（top-5）:")
    print("=" * 60)
    for z in range(n_topics):
        top_words = np.argsort(phi[:, z])[::-1][:5]
        probs = phi[top_words, z]
        print(f"主题 {z}: {list(zip(top_words, probs.round(4)))}")
    
    print("\n文档的主题分布:")
    print(theta[:, :5].round(4))
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_plsa_results():
    """可视化 PLSA 的结果"""
    
    np.random.seed(42)
    
    # 模拟数据
    n_docs = 30
    vocab_size = 50
    n_topics = 3
    
    # 生成数据
    topic_word = np.random.rand(vocab_size, n_topics)
    topic_word = topic_word / topic_word.sum(axis=0, keepdims=True)
    
    doc_term = np.zeros((n_docs, vocab_size))
    for d in range(n_docs):
        tid = d % n_topics
        for _ in range(50):
            wid = np.random.choice(vocab_size, p=topic_word[:, tid])
            doc_term[d, wid] += 1
    
    # 训练（简化）
    # 实际使用上面的 plsa_EM 函数
    print("训练 PLSA...")
    phi, theta = plsa_EM(doc_term, n_topics, max_iter=30, verbose=False)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 主题-词分布热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(phi, aspect='auto', cmap='Blues')
    ax1.set_xlabel('主题')
    ax1.set_ylabel('词语')
    ax1.set_title('主题-词分布 P(w|z)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 文档-主题分布（前10个文档）
    ax2 = axes[0, 1]
    im2 = ax2.imshow(theta[:, :10], aspect='auto', cmap='Oranges')
    ax2.set_xlabel('文档')
    ax2.set_ylabel('主题')
    ax2.set_title('文档-主题分布 P(z|d)（前10个）')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 各文档的主要主题
    ax3 = axes[1, 0]
    main_topics = theta.argmax(axis=0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for z in range(n_topics):
        mask = main_topics == z
        ax3.scatter(np.where(mask)[0], np.full(mask.sum(), z), 
                   c=colors[z], label=f'主题{z}', alpha=0.7)
    ax3.set_xlabel('文档ID')
    ax3.set_ylabel('主要主题')
    ax3.set_title('各文档的主要主题')
    ax3.legend()
    ax3.set_yticks([0, 1, 2])
    
    # 4. 主题-词概率分布（直方图）
    ax4 = axes[1, 1]
    for z in range(n_topics):
        ax4.hist(phi[:, z], bins=20, alpha=0.5, label=f'主题{z}')
    ax4.set_xlabel('P(w|z)')
    ax4.set_ylabel('词语数量')
    ax4.set_title('主题-词概率分布')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('plsa_visualization.png', dpi=150)
    plt.show()
    print("可视化结果已保存到 plsa_visualization.png")


if __name__ == "__main__":
    visualize_plsa_results()
```

---

## 10. 模型评估

### 10.1 困惑度（Perplexity）

PLSA 常用的评估指标是**困惑度（Perplexity）**：
$$\text{Perplexity} = \exp\left(-\frac{1}{\sum_d n_d} \sum_d \sum_w n(d,w) \log P(w|d)\right)$$

其中 $n_d = \sum_w n(d,w)$ 是文档 $d$ 的总词数。

```python
def compute_perplexity(doc_term_matrix, phi, theta):
    """计算困惑度"""
    n_docs, vocab_size = doc_term_matrix.shape
    total_words = doc_term_matrix.sum()
    
    log_likelihood = 0.0
    for d in range(n_docs):
        for w in range(vocab_size):
            n = doc_term_matrix[d, w]
            if n > 0:
                prob = phi[w, :] @ theta[:, d]
                log_likelihood += n * np.log(prob + 1e-10)
    
    perplexity = np.exp(-log_likelihood / total_words)
    return perplexity

# 使用示例
# perplexity = compute_perplexity(doc_term_matrix, phi, theta)
# print(f"Perplexity: {perplexity:.2f}")
```

### 10.2 其他指标

1. **主题连贯性（Topic Coherence）**：
   - 衡量主题内词语的语义相关性
   - 常用 PMI（点互信息）计算

2. **Log-Likelihood**：
   - 训练过程中的对数似然
   - 越高越好（需考虑模型复杂度）

---

## 11. 常见问题与易错点

### 11.1 问题1：参数初始化不当导致不收敛
**原因**：
- 如果初始化的概率值为 0，EM 迭代后仍为 0（陷入零解）
- 如果初始值差距过大，可能陷入局部最优

**解决方案**：
```python
# 正确初始化：保证所有值 > 0
phi = np.random.rand(vocab_size, n_topics) + 0.01
phi = phi / phi.sum(axis=0, keepdims=True)

# 多次初始化取最优
best_ll = -np.inf
for _ in range(5):
    phi_init = np.random.rand(vocab_size, n_topics) + 0.01
    phi_init = phi_init / phi_init.sum(axis=0, keepdims=True)
    # ... 训练
    if ll > best_ll:
        best_ll = ll
        best_phi, best_theta = phi, theta
```

### 11.2 问题2：数值溢出
**原因**：
- 直接计算 $\phi_{zw} \theta_{dz}$ 可能非常���，���致下溢
- 计算 $\log$ 时可能出现 $\log(0)$

**解决方案**：
```python
# 使用 log-space 计算
log_phi = np.log(phi + 1e-10)
log_theta = np.log(theta + 1e-10)
logJoint = log_phi[:, :, None] + log_theta[None, :, :]

# log-sum-exp trick
logJoint_max = logJoint.max(axis=1, keepdims=True)
logJoint_normalized = logJoint - logJoint_max
Joint = np.exp(logJoint_normalized)
posterior = Joint / (Joint.sum(axis=1, keepdims=True) + 1e-10)
```

### 11.3 问题3：主题数选择困难
**原因**：
- K 太小：主题过于粗糙，不同主题混在一起
- K 太大：主题过于细碎，可能出现噪音主题

**解决方案**：
- 使用验证集选择最优 K
- 观察主题的可解释性
- 使用 Coherence Score 辅助选择

---

## 12. 学习总结

### 核心要点回顾：
1. **核心思想**：引入潜在变量 $z$ 建模文档-词的语义关系
2. **优化方法**：EM 算法迭代更新参数
3. **输出**：主题-词分布 $\phi$、文档-主题分布 $\theta$
4. **应用**：主题挖掘、文本表示、信息检索

### 从 PLSA 到其他算法：
- **PLSA → LDA**：加入狄利克雷先验，解决参数泛化问题
- **PLSA → NMF**：将非负矩阵分解应用于主题发现
- **PLSA → SVD/LSA**：基于矩阵分解的主题模型（EM→SVD）

### 实践建议：
1. **数据预处理**：做好分词、去停用词、过滤低频词
2. **主题数选择**：从 5-15 开始，逐步增加
3. **结果解释**：关注主题内的词是否有语义一致性
4. **结合下游任务**：作为分类/聚类的特征输入

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**
问题：假设有 2 个主题（z1, z2）、2 个词语（w1, w2）、1 篇文档，词频为 n(d,w1)=5, n(d,w2)=3，已知参数 $\phi_{w1,z1}=0.8$, $\phi_{w1,z2}=0.2$, $\phi_{w2,z1}=0.2$, $\phi_{w2,z2}=0.8$，$\theta_{z1,d}=0.6$, $\theta_{z2,d}=0.4$，计算 E 步的后验概率 $P(z|d,w1)$。

<details>
<summary>答案</summary>

**解答**：

根据贝叶斯定理：
$$P(z|d,w) = \frac{P(w|z)P(z|d)}{\sum_{z'} P(w|z')$$

对于 $w1, z1$：
$$P(z1|d,w1) = \frac{0.8 \times 0.6}{0.8 \times 0.6 + 0.2 \times 0.4} = \frac{0.48}{0.48 + 0.08} = \frac{0.48}{0.56} \approx 0.857$$

对于 $w1, z2$：
$$P(z2|d,w1) = \frac{0.2 \times 0.4}{0.56} = \frac{0.08}{0.56} \approx 0.143$$

验证：$0.857 + 0.143 = 1.0$ ✓

</details>

**习题2：编程实践**
问题：在 20 newsgroups 数据集上使用 PLSA，提取 5 个主题，查看每个主题的关键词。

<details>
<summary>答案</summary>

**代码示例**：
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 加载数据（简化版）
newsgroups = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.autos'])
documents = newsgroups.data[:500]  # 取前500篇

# 分词
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
doc_term = vectorizer.fit_transform(documents).toarray()

# 训练 PLSA
phi, theta = plsa_EM(doc_term, n_topics=5, max_iter=50)

# 展示主题
vocab = vectorizer.get_feature_names()
for z in range(5):
    top_indices = np.argsort(phi[:, z])[::-1][:10]
    print(f"主题 {z}: {[vocab[i] for i in top_indices]}")
```

</details>

**习题3：理论推导**
问题：推导 M 步中 $\phi_{zw}$ 的更新公式，说明为何使用归一化。

<details>
<summary>答案</summary>

**推导过程**：

完全数据（观测数据 + 隐变量）的对数似然：
$$Q = \sum_{d,w} n(d,w) \sum_z P(z|d,w) \log \phi_{zw}$$

约束条件：$\sum_w \phi_{zw} = 1$（每个主题的词分布需归一化）

使用拉格朗日乘数法：
$$\mathcal{L} = Q + \sum_z \lambda_z \left(\sum_w \phi_{zw} - 1\right)$$

对 $\phi_{zw}$ 求偏导并设为 0：
$$\frac{\partial \mathcal{L}}{\partial \phi_{zw}} = \frac{\sum_d n(d,w) P(z|d,w)}{\phi_{zw}} + \lambda_z = 0$$

$$\phi_{zw} = -\frac{\sum_d n(d,w) P(z|d,w)}{\lambda_z}$$

由约束条件：
$$-\sum_w \frac{\sum_d n(d,w) P(z|d,w)}{\lambda_z} = 1$$

$$\lambda_z = -\sum_w \sum_d n(d,w) P(z|d,w)$$

代回得到：
$$\phi_{zw} = \frac{\sum_d n(d,w) P(z|d,w)}{\sum_{w'} \sum_d n(d,w') P(z|d,w')}$$

这正是我们使用的更新公式。归一化的原因是 $\phi$ 是概率分布，必须满足 $\sum_w P(w|z) = 1$。

</details>

### 思考题

**思考题1**：PLSA 和 LDA 的主要区别是什么？为什么 LDA 可以避免 PLSA 的过拟合问题？

<details>
<summary>答案</summary>

**解答**：

主要区别：
1. **参数估计方式**：
   - PLSA：点估计（maximum likelihood）
   - LDA：贝叶斯估计（variational inference）

2. **参数数量**：
   - PLSA：$P(z|d)$ 随文档数线性增长
   - LDA：通过先验分布共享参数

3. **正则化**：
   - PLSA：无正则化，容易过拟合
   - LDA：狄利克雷先验起正则化作用

为什么 LDA 可以避免过拟合：
- LDA 对参数引入先验分布（狄利克雷分布），相当于 L1/L2 正则化
- 当数据量较小时，先验占主导，防止过拟合
- 参数是随机变量而非固定值，具有不确定性

</details>

**思考题2**：如果要在生产环境中使用 PLSA，需要考虑哪些工程优化？

<details>
<summary>答案</summary>

**解答**：

1. **计算效率优化**：
   - 稀疏矩阵表示：减少存储和计算
   - 分布式计算：MapReduce / Spark
   - 采样近似：Gibbs Sampling 替代 EM

2. **工程挑战**：
   - 新文档处理：PLSA 需重新训练，可用 LDA 解决
   - 增量学习：Online EM 算法
   - 模型压缩：剪枝、量化

3. **系统设计**：
   - 预处理流水线：分词、去噪、构建词表
   - 结果缓存：主题分布的快速查询
   - 监控与告警：困惑度、收敛情况

4. **替代方案**：
   - 现代场景可考虑：BERTopic（基于 BERT）、tomotopy（高效 LDA）

</details>

---

## 14. 学习路径建议

### 初级阶段（掌握 PLSA 基础）
1. 理解条件概率与贝叶斯定理
2. 掌握 EM 算法的思想
3. 手动实现 PLSA 的 E-M 步骤
4. 使用库实现主题挖掘

**学习时间**：1-2 周

### 中级阶段（理解原理和扩展）
1. 深入理解 PLSA 与 LDA 的关系
2. 学习变分推断（Variational Inference）
3. 掌握主题 coherence 评估
4. 在真实数据集上实践

**学习时间**：2-3 周

### 高级阶段（扩展到其他算法）
1. 学习 Hierarchical LDA
2. 研究 Neural Topic Models
3. 探索 BERTopic、DinamicaTM 等前沿方法

**学习时间**：3-4 周

### 实践项目建议
1. **基础项目**：对 20 newsgroups 进行主题分析
2. **进阶项目**：构建基于主题的文档推荐系统
3. **挑战项目**：实现分布式 PLSA/LDA 训练系统

### 推荐资源
- **书籍**：《Machine Learning for Text》《Pattern Recognition and Machine Learning》
- **论文**：Hofmann 1999 PLSA 原始论文，Blei 2003 LDA 论文
- **课程**：CS224N（NLP with Deep Learning）
- **代码**：gensim、tomotopy
- **实践**：Kaggle 文本分类比赛

---

*PLSA 是主题模型的基石，理解 PLSA 为学习 LDA 和现代主题模型打下坚实基础。*