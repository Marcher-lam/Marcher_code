# TF-IDF 学习文档

> 信息检索的经典方法——词的重要性不仅看频率

---

## 1. 算法基础认知

### 1.1 什么是TF-IDF

**TF-IDF（Term Frequency - Inverse Document Frequency）** 是一种统计方法，用于评估一个词对于文档集中某篇文档的重要程度。

```
核心思想:
- 一个词在一篇文章中出现次数越多(TF高) → 越重要
- 一个词在很多文章都出现(IDF低) → 越不重要（如"的"、"是"）

TF-IDF = TF × IDF
```

### 1.2 在推荐系统中的应用

| 应用 | 说明 |
|------|------|
| **内容推荐** | 计算文章/商品的TF-IDF向量，找相似内容 |
| **关键词提取** | 提取文档的关键词作为特征 |
| **文本相似度** | 用TF-IDF向量计算文档间的余弦相似度 |
| **冷启动** | 新物品用TF-IDF提取内容特征 |

---

## 2. 核心原理

### 2.1 TF（词频）

$$TF(t, d) = \frac{t在文档d中出现的次数}{文档d的总词数}$$

### 2.2 IDF（逆文档频率）

$$IDF(t, D) = \log\frac{|D|}{|\{d \in D : t \in d\}| + 1}$$

- $|D|$：文档总数
- 分母：包含词 $t$ 的文档数

### 2.3 TF-IDF

$$\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)$$

**直觉**：
- "推荐系统"在某文章中出现很多次（高TF），且只在少数文章中出现（高IDF）→ TF-IDF高 → 是这篇文章的关键词
- "的"在很多文章中都出现（低IDF）→ TF-IDF低 → 不是关键词

---

## 7. 调库实现

```python
"""
TF-IDF 完整实现
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 文档集合（模拟推荐场景中的物品描述）
documents = [
    "推荐系统算法工程师需要掌握机器学习",
    "深度学习在推荐系统中广泛应用",
    "协同过滤是推荐系统的经典算法",
    "深度学习神经网络用于图像识别",
    "机器学习算法包括监督学习和无监督学习"
]

# TF-IDF向量化
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"词汇表: {vectorizer.get_feature_names_out()}")
print(f"TF-IDF矩阵: {tfidf_matrix.shape}")

# 内容推荐: 找与query最相似的文档
query = ["推荐系统算法"]
query_vec = vectorizer.transform(query)
similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

print("\n内容推荐结果:")
for idx in similarities.argsort()[::-1]:
    print(f"  [{similarities[idx]:.4f}] {documents[idx]}")
```

---

## 8. 手工代码实现

```python
"""
TF-IDF 纯手工实现
"""
import math
from collections import Counter

def compute_tf(word, doc_words):
    """计算词频"""
    return doc_words.count(word) / len(doc_words)

def compute_idf(word, all_docs):
    """计算逆文档频率"""
    n_containing = sum(1 for doc in all_docs if word in doc)
    return math.log(len(all_docs) / (1 + n_containing))

def compute_tfidf(doc, all_docs):
    """计算TF-IDF"""
    words = doc.split()
    word_counts = Counter(words)
    tfidf = {}
    for word in word_counts:
        tfidf[word] = compute_tf(word, words) * compute_idf(word, [d.split() for d in all_docs])
    return tfidf

# 示例
docs = [
    "推荐 系统 算法",
    "深度 学习 推荐系统",
    "协同 过滤 推荐 系统"
]

for i, doc in enumerate(docs):
    scores = compute_tfidf(doc, docs)
    print(f"文档{i+1}: {dict(sorted(scores.items(), key=lambda x: -x[1]))}")
```

---

## 12. 学习总结

1. **TF-IDF = 词频 × 逆文档频率**：衡量词对文档的区分度
2. **简单有效**：不需要训练，直接统计
3. **内容推荐基础**：基于TF-IDF的文本相似度计算
4. **局限性**：无法捕捉语义（"电脑"和"计算机"的TF-IDF向量正交）

---

## 14. 学习路径

```
One-Hot → [当前: TF-IDF] → Word2Vec → GloVe → BERT
```
