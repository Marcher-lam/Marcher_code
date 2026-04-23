# TF-IDF 学习文档

## 1. 算法基础认知

TF-IDF（Term Frequency - Inverse Document Frequency，词频-逆文档频率）是一种经典的文本特征提取方法，由 Jones 于 1972 年提出。它通过衡量一个词在文档中的重要程度来实现文本表示，是信息检索和文本挖掘的基石。

核心直觉：一个词在某文档中出现次数越多（TF 高），且在整个语料中出现越少（IDF 高），则该词对该文档越重要。

## 2. 核心原理

TF-IDF 由两部分组成：

- **TF（词频）**：词在当前文档中出现的频率，衡量词的局部重要性
- **IDF（逆文档频率）**：词在语料中的稀有程度，衡量词的全局区分能力

一个高频出现在所有文档中的词（如"的"、"是"）区分能力弱，应降低权重；而只在少数文档中出现的词区分能力强，应提高权重。

## 3. 数学公式与推导

**词频 TF**（多种变体）：

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

其中 $f_{t,d}$ 是词 $t$ 在文档 $d$ 中的出现次数。

**逆文档频率 IDF**：

$$\text{IDF}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}| + 1}$$

其中 $N$ 是文档总数，分母是包含词 $t$ 的文档数。加 1 防止除零。

**TF-IDF**：

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

sklearn 中使用的标准形式（带平滑）：

$$\text{IDF}(t) = \log \frac{1 + N}{1 + \text{df}(t)} + 1$$

## 4. 训练过程讲解

1. **构建词汇表**：遍历所有文档，收集唯一词，建立词到索引的映射
2. **计算 TF 矩阵**：对每个文档，统计每个词的出现频率
3. **计算 IDF 向量**：统计每个词出现在多少个文档中（df），计算 IDF 值
4. **计算 TF-IDF**：将 TF 矩阵与 IDF 向量逐元素相乘
5. **L2 归一化**（可选）：对每行向量进行归一化

## 5. 应用场景

- 搜索引擎中的文档相关性排序
- 文本分类任务的特征提取
- 关键词提取
- 文档相似度计算
- 推荐系统中用户画像的文本特征

## 6. 优缺点分析

**优点**：
- 原理简单，计算效率高
- 无需训练，可直接计算
- 有效降低常见停用词的权重
- 在信息检索任务中效果稳定

**缺点**：
- 无法捕获词序和语义信息
- 无法处理同义词（"电脑"和"计算机"完全独立）
- 对罕见词可能赋予过高权重
- 稀疏表示，维度等于词汇表大小

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "机器学习是人工智能的重要分支",
    "深度学习是机器学习的子领域",
    "自然语言处理使用深度学习技术",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

print("词汇表:", vectorizer.vocabulary_)
print("IDF 值:", vectorizer.idf_)
print("TF-IDF 矩阵形状:", tfidf_matrix.shape)

sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print("文档0与其余文档的余弦相似度:", sim)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
import math
from collections import Counter

class TfidfManual:
    def __init__(self):
        self.vocab = {}
        self.idf = None

    def fit(self, docs):
        word_set = set()
        for doc in docs:
            word_set.update(doc.split())
        self.vocab = {w: i for i, w in enumerate(sorted(word_set))}
        V = len(self.vocab)
        N = len(docs)
        df = np.zeros(V)
        for doc in docs:
            seen = set(doc.split())
            for w in seen:
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + N) / (1 + df)) + 1

    def transform(self, docs):
        V = len(self.vocab)
        result = []
        for doc in docs:
            tf = np.zeros(V)
            words = doc.split()
            counts = Counter(words)
            total = len(words)
            for w, c in counts.items():
                if w in self.vocab:
                    tf[self.vocab[w]] = c / total
            tfidf = tf * self.idf
            norm = np.linalg.norm(tfidf)
            if norm > 0:
                tfidf /= norm
            result.append(tfidf)
        return np.array(result)

docs = ["机器 学习 是 人工 智能", "深度 学习 是 机器 学习 子领域"]
m = TfidfManual()
m.fit(docs)
print("TF-IDF:\n", m.transform(docs))
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "机器学习是人工智能的重要分支",
    "深度学习是机器学习的子领域",
    "自然语言处理使用深度学习技术",
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(docs).toarray()
words = vectorizer.get_feature_names_out()

plt.figure(figsize=(10, 4))
sns.heatmap(matrix, xticklabels=words, yticklabels=["D1", "D2", "D3"],
            cmap="YlOrRd", annot=True, fmt=".2f")
plt.title("TF-IDF 热力图")
plt.tight_layout()
plt.savefig("tfidf_heatmap.png", dpi=150)
plt.show()
```

热力图中颜色越深表示 TF-IDF 值越高，说明该词对对应文档越重要。

## 10. 模型评估

- **检索质量**：使用 Precision@K、MAP、NDCG 评估检索排序效果
- **分类效果**：TF-IDF + 朴素贝叶斯/SVM 作为文本分类 baseline
- **特征质量**：通过下游任务（分类 F1 值）间接评估特征表示质量

## 11. 常见问题与易错点

- **未做分词**：中文需要先分词（如 jieba），否则以字为单位效果很差
- **忽略归一化**：长文档 TF 值天然偏高，需要归一化处理
- **IDF 溢出**：log 里分母加 1 是必须的，否则 df=0 时除零
- **停用词处理**：常见停用词 IDF 值低但非零，建议显式过滤

## 12. 学习总结

TF-IDF 通过 TF 衡量局部频率、IDF 衡量全局稀有度，实现了对文档中关键词的自动识别。它是文本表示的经典基线方法，简单高效，但无法捕获语义。理解 TF-IDF 后，应进一步学习稠密词嵌入方法（Word2Vec、GloVe）来弥补语义缺失。

## 13. 练习题与思考题（含答案）

**Q1**：词 "the" 在 1000 篇英文文档的每篇中都出现，其 IDF 值为多少？

**A1**：$\text{IDF} = \log\frac{1000}{1000} = \log 1 = 0$，因此 TF-IDF = 0，完美过滤了该停用词。

**Q2**：为什么 TF-IDF 无法区分"电脑"和"计算机"这两个同义词？

**A2**：TF-IDF 基于词袋模型，将每个词视为独立维度，不同词的向量正交，无法建模语义关系。

**Q3**：对比 One-Hot，TF-IDF 在哪些方面做了改进？

**A3**：TF-IDF 引入了词频加权和全局稀有度加权，使得重要词的权重更高，停用词权重趋近于 0，比 One-Hot 的均匀表示更有区分力。

## 14. 学习路径建议

1. 掌握 One-Hot 编码 → 2. 理解 TF-IDF 原理与实现 → 3. 学习 Word2Vec（稠密词嵌入）→ 4. 学习 GloVe（全局统计嵌入）→ 5. 学习 BERT（上下文嵌入）
