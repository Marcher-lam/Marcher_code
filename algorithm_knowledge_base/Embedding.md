# 嵌入学习文档

> 嵌入是将离散的词转换为连续的向量表示，使得语义相似的词在向量空间中距离相近。

## 1. 算法基础认知

### 1.1 什么是嵌入

嵌入（Embedding）将高维稀疏的词表示转换为低维稠密的向量，使得语义相关的词在向量空间中距离更近。

### 1.2 直觉类比

想象每个词都有一个位置，语义相近的词（如"国王"和"王后"）的位置更近，语义不同的词（如"石头"）距离更远。

### 1.3 历史背景

- **2013年**：Word2Vec发布（Tomas Mikolov）
- **2018年**：BERT发布
- **影响**：奠定了现代NLP的基础

### 1.4 算法定位

- **任务类型**：特征工程/表示学习
- **所属类别**：无监督学习

## 2. 核心原理

### 2.1 核心思想

通过神经网络学习词的向量表示，使得出现在相似上下文中的词具有相似的向量。

### 2.2 Word2Vec两种方法

1. **CBOW**：用周围词预测中心词
2. **Skip-gram**：用中心词预测周围词

## 3. 数学公式

### 3.1 Skip-gram目标

$$\mathcal{L} = \sum_{(c,w) \in D} \log P(w|c)$$

其中 $P(w|c) = softmax(v_c \cdot v_w^T)$

### 3.2 经典类比

$$vec("king") - vec("man") + vec("woman") \approx vec("queen")$$

## 4. 调库实现

```python
from gensim.models import Word2Vec
import nltk

# 准备数据
sentences = [["hello", "world"], ["machine", "learning"]]

# 训练模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 获取词向量
vector = model.wv["hello"]
print(f"向量形状: {vector.shape}")

# 找相似词
similar = model.wv.most_similar("hello", topn=3)
print(f"相似词: {similar}")
```

## 5. 可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ["king", "queen", "man", "woman", "prince", "princess"]
vectors = [model.wv[w] for w in words]

# PCA降维
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 绘制
plt.figure(figsize=(8, 6))
for i, w in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.annotate(w, (result[i, 0], result[i, 1]))
plt.show()
```

## 6. 优缺点

| 优点 | 缺点 |
|------|------|
| 低维稠密表示 | 无法处理未见过的词 |
| 语义相似性 | 无法处理多义词 |
| 高效计算 | 需要大量语料 |

## 7. 学习总结

嵌入是现代NLP的基石，能够将词转换为机器可处理的连续向量表示，是Transformer等模型的基础组件。

## 8. 练习题

**题目**：为什么Word2Vec能学到词的语义相似性？

**答案**：Word2Vec基于分布假说——"词由其上下文决定"。出现在相似上下文中的词会有相似的向量表示。