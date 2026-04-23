# LSA 学习文档

## 1. 算法基础认知

潜在语义分析（Latent Semantic Analysis, LSA）是一种基于线性代数的话题分析方法。其核心思想是：通过对"词-文档"矩阵进行截断奇异值分解（SVD），将高维稀疏的词频空间映射到低维稠密的潜在语义空间，从而揭示词与词、文档与文档之间的隐含语义关系。

LSA 由 Deerwester 等人于 1990 年提出，是信息检索和自然语言处理中的经典无监督方法。

## 2. 核心原理

1. **构建词-文档矩阵 $X$**：行代表词项（term），列代表文档（document），元素 $X_{ij}$ 为词 $i$ 在文档 $j$ 中的出现频次或 TF-IDF 权重
2. **截断 SVD 分解**：$X \approx U_k \Sigma_k V_k^T$，其中 $k$ 为保留的主题数
3. **语义空间映射**：
   - $U_k$：词项在潜在语义空间中的坐标
   - $V_k$：文档在潜在语义空间中的坐标
   - $\Sigma_k$：每个潜在维度的"重要性"

通过降维，同义词被映射到相近位置，从而解决词汇不匹配问题。

## 3. 数学公式与推导

给定 $m \times n$ 的词-文档矩阵 $X$，其 SVD 分解为：

$$X = U \Sigma V^T$$

其中 $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, $V \in \mathbb{R}^{n \times n}$。

截断至 $k$ 维：

$$X_k = U_k \Sigma_k V_k^T$$

- $U_k \in \mathbb{R}^{m \times k}$：词项-主题矩阵
- $\Sigma_k \in \mathbb{R}^{k \times k}$：奇异值对角阵
- $V_k \in \mathbb{R}^{n \times k}$：文档-主题矩阵

**查询处理**：对新查询向量 $q$，映射到语义空间：$\hat{q} = q^T U_k \Sigma_k^{-1}$，再与 $V_k$ 计算余弦相似度。

## 4. 训练过程讲解

1. **预处理**：分词、去停用词、构建词-文档矩阵
2. **加权**：通常使用 TF-IDF 替代原始词频，降低高频常用词的影响
3. **SVD 分解**：对加权矩阵进行截断 SVD，选择主题数 $k$
4. **降维表示**：用 $U_k \Sigma_k$（词向量）和 $\Sigma_k V_k^T$（文档向量）作为低维表示
5. **相似度计算**：通过余弦相似度进行文档检索或聚类

## 5. 应用场景

- 文档检索与相似度计算
- 文本分类与聚类
- 跨语言信息检索
- 广告关键词语义匹配与投放优化
- 推荐系统中的物品语义特征提取

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 实现简单，基于成熟的线性代数 | 主题缺乏概率解释 |
| 能捕获同义关系 | SVD 计算代价高（大规模矩阵） |
| 降维降噪效果好 | $k$ 的选择依赖经验 |
| 无需标注数据 | 无法处理多义词（一个词只有一个向量） |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

docs = [
    "深度学习在广告推荐系统中广泛应用",
    "机器学习算法可以优化广告投放策略",
    "自然语言处理是人工智能的重要方向",
    "文本分类和情感分析属于自然语言处理",
    "推荐系统利用协同过滤和深度学习方法",
]

vectorizer = TfidfVectorizer(tokenizer=list)
X = vectorizer.fit_transform([list(d.replace(" ", "")) for d in docs])

svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X)

terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    top_idx = comp.argsort()[-5:][::-1]
    print(f"主题 {i}: {[terms[j] for j in top_idx]}")

print(f"\n原始维度: {X.shape[1]}, 降维后: {X_reduced.shape[1]}")
print(f"解释方差比: {svd.explained_variance_ratio_.sum():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from collections import Counter

def build_term_doc_matrix(docs):
    vocab = sorted(set(w for d in docs for w in d))
    word2idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(docs):
        for w in doc:
            X[word2idx[w], j] += 1
    tf = X / (X.sum(axis=0, keepdims=True) + 1e-10)
    idf = np.log(len(docs) / (1 + (X > 0).sum(axis=1))).reshape(-1, 1)
    return tf * idf, vocab

def lsa(X, k):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], np.diag(S[:k]), Vt[:k, :]

docs = [
    list("深度学习广告推荐"),
    list("机器学习广告投放"),
    list("自然语言处理分析"),
]

X, vocab = build_term_doc_matrix(docs)
U_k, S_k, Vt_k = lsa(X, k=2)

for i, w in enumerate(vocab):
    print(f"{w}: [{U_k[i,0]:.3f}, {U_k[i,1]:.3f}]")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
for i, doc in enumerate(docs):
    plt.annotate(f"doc{i+1}", (X_reduced[i, 0], X_reduced[i, 1]))
plt.xlabel("LSA Dimension 1")
plt.ylabel("LSA Dimension 2")
plt.title("LSA Document Space")
plt.savefig("lsa_visual.png", dpi=150)
plt.close()
```

在二维空间中，语义相近的文档应聚在一起，相距较远的文档主题差异大。

## 10. 模型评估

- **重构误差**：$\|X - X_k\|_F / \|X\|_F$，越小越好
- **解释方差比**：$\sum_{i=1}^{k}\sigma_i^2 / \sum_{i=1}^{r}\sigma_i^2$
- **下游任务评估**：在检索任务中用 Precision@K 或 MAP 评估
- **余弦相似度合理性**：人工检验语义相近文档的相似度是否更高

## 11. 常见问题与易错点

- **未做 TF-IDF 加权**：直接用原始词频会导致高频无意义词主导结果
- **主题数 $k$ 选择不当**：$k$ 太小丢失信息，太大则引入噪声且计算量大
- **忽略奇异值衰减分析**：应通过奇异值衰减曲线（Scree Plot）辅助确定 $k$
- **中文未正确分词**：中文需先分词，否则逐字符处理效果差

## 12. 学习总结

LSA 通过 SVD 降维将词-文档矩阵映射到低维语义空间，是文本挖掘的经典方法。它简单高效，但缺乏概率解释。后续的 PLSA 和 LDA 在概率框架下对 LSA 进行了扩展和改进。

## 13. 练习题与思考题（含答案）

**Q1**：LSA 为什么能处理同义词问题？

> A1：同义词经常出现在相似的文档中，SVD 降维后它们在潜在语义空间中被映射到相近的坐标，从而在相似度计算中被视为相关。

**Q2**：截断 SVD 和 PCA 有什么关系？

> A2：对去中心化后的矩阵做 SVD 等价于 PCA。LSA 通常不做去中心化，因此严格来说是截断 SVD 而非 PCA，但数学结构相同。

**Q3**：LSA 主题数 $k$ 如何选择？

> A3：常用方法包括：(1) 观察奇异值衰减曲线的拐点；(2) 在验证集上通过下游任务性能调参；(3) 经验法则取 $k \in [100, 300]$ 用于大规模语料。

## 14. 学习路径建议

- **前置知识**：线性代数（SVD）、信息检索基础、TF-IDF
- **进阶方向**：PLSA → LDA → Word2Vec → BERT 语义表示
- **推荐实践**：在新闻数据集上对比 LSA 和 LDA 的主题提取效果
