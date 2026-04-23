# GloVe 学习文档

> 结合全局统计与局部上下文的词向量方法

---

## 1. 算法基础认知

### 1.1 什么是GloVe

**GloVe（Global Vectors for Word Representation）** 是斯坦福大学2014年提出的词嵌入方法，结合了**全局矩阵分解**（如LSA）和**局部上下文窗口**（如Word2Vec）的优点。

### 1.2 核心思想

```
Word2Vec: 只看局部上下文窗口
LSA/PCA:  看全局共现矩阵但忽略了局部信息
GloVe:    利用全局共现矩阵的统计信息，学习词向量

关键洞察: 词向量的点积应该等于共现概率的对数
```

---

## 3. 数学公式

### 3.1 共现矩阵

$X_{ij}$ 表示词 $j$ 出现在词 $i$ 上下文中的次数。

### 3.2 目标函数

$$J = \sum_{i=1}^{V}\sum_{j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

其中加权函数：

$$f(x) = \begin{cases} (x/x_{max})^\alpha & x < x_{max} \\ 1 & x \geq x_{max} \end{cases}$$

> $x_{max}$ 通常设为100，$\alpha$ 通常设为0.75。

---

## 7. 调库实现

```python
"""
GloVe 词向量使用
"""
import numpy as np

# 加载预训练GloVe向量
def load_glove_embeddings(glove_file):
    """加载GloVe预训练向量"""
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 使用gensim加载
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# 如果有预训练文件:
# glove2word2vec('glove.6B.100d.txt', 'glove_w2v.txt')
# model = KeyedVectors.load_word2vec_format('glove_w2v.txt')

# 演示GloVe的词向量性质
print("GloVe词向量的经典性质:")
print("  vec('king') - vec('man') + vec('woman') ≈ vec('queen')")
print("  vec('paris') - vec('france') + vec('italy') ≈ vec('rome')")
```

---

## 12. 学习总结

1. **GloVe = 全局共现统计 + 向量学习**
2. **vs Word2Vec**：GloVe利用全局统计信息，Word2Vec只用局部窗口
3. **vs LSA**：GloVe效果更好，向量维度更低
4. **预训练模型**：常用GloVe.6B（60亿词训练），推荐用100d或300d

---

## 14. 学习路径

```
Word2Vec → [当前: GloVe] → FastText → ELMo → BERT/GPT
```
