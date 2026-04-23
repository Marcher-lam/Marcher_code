# Char2Vec 学习文档

## 1. 算法基础认知

Char2Vec（Character-level Embedding）是一种基于字符级别的文本表示方法。与 Word2Vec 以词为基本单位不同，Char2Vec 以字符（或子词 subword）为最小粒度来学习向量表示。这种方法在处理形态丰富的语言（如德语、土耳其语）和 OOV（Out-of-Vocabulary）问题时尤为有效。

典型代表包括 fastText（Facebook, 2016）和 ELMo 的字符级 CNN 编码器。

## 2. 核心原理

Char2Vec 的核心思想是：**词的语义可以从其组成的字符或子词片段中推断出来**。例如，"unhappiness" 可以分解为 "un-" + "happy" + "-ness"，即使整个词未出现在训练语料中，模型仍可通过已知子词片段推断其含义。

常见方法：
- **字符级 n-gram**：将词拆分为字符级 n-gram（如 "apple" → ["ap", "pp", "pl", "le"]），对 n-gram 向量求和或平均
- **字符级 CNN**：对字符序列做卷积操作提取特征
- **字符级 RNN**：用循环网络逐字符读取并编码

## 3. 数学公式与推导

### fastText 风格的字符 n-gram 嵌入

给定词 $w$，其字符 n-gram 集合为 $\mathcal{G}_w$（$3 \leq n \leq 6$），词的表示为：

$$\mathbf{v}_w = \frac{1}{|\mathcal{G}_w|} \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

其中 $\mathbf{z}_g$ 是 n-gram $g$ 的嵌入向量。

### 字符级 CNN

输入词 $w = c_1 c_2 \ldots c_L$（$L$ 个字符），先做字符嵌入：

$$\mathbf{E} = [\mathbf{e}_{c_1}, \mathbf{e}_{c_2}, \ldots, \mathbf{e}_{c_L}] \in \mathbb{R}^{d_c \times L}$$

然后做一维卷积：

$$h_k = \text{ReLU}(\mathbf{W}_{\text{conv}} \cdot \mathbf{E}_{[:, k:k+f-1]} + b)$$

再做最大池化得到固定长度的词表示：

$$\mathbf{v}_w = \text{MaxPool}([h_1, h_2, \ldots, h_{L-f+1}])$$

## 4. 训练过程讲解

以 fastText 风格为例：

1. **构建字符 n-gram 词典**：遍历所有词，提取 3-6 gram，建立 n-gram 到索引的映射
2. **初始化嵌入矩阵**：包括字符 n-gram 嵌入和词嵌入
3. **训练**：类似 Word2Vec 的 Skip-gram，但用 n-gram 向量之和代替词向量作为输入
4. **推理**：新词通过其 n-gram 向量组合得到表示，天然支持 OOV

## 5. 应用场景

- 处理 OOV 词（人名、地名、新词等）
- 形态丰富的语言（德语复合词、土耳其语）
- 拼写纠错和模糊匹配
- 命名实体识别（NER）
- 作为深度 NLP 模型的字符级输入层

## 6. 优缺点分析

**优点**：
- 天然处理 OOV，任何新词都可分解为已知字符/n-gram
- 共享子词信息，相似形态的词向量自然接近
- 对拼写变体和错别字鲁棒
- 适合词表开放的任务

**缺点**：
- 字符级序列更长，训练和推理速度比词级慢
- 字符本身无明确语义，需要学习组合规则
- n-gram 可能产生过多特征
- 对于中文等非形态语言，字符级优势不如子词级明显

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from gensim.models import FastText
from gensim.test.utils import common_texts

model = FastText(
    sentences=common_texts,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,
    max_n=6,
    workers=4,
    epochs=10,
)

print("'human' 向量:", model.wv["human"][:5])

oov_word = "humanship"
print(f"OOV 词 '{oov_word}' 向量:", model.wv[oov_word][:5])
print("与 'human' 的相似度:", model.wv.similarity("human", oov_word))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from collections import defaultdict

class CharNgramEmbedding:
    def __init__(self, embed_dim=50, min_n=3, max_n=6):
        self.embed_dim = embed_dim
        self.min_n = min_n
        self.max_n = max_n
        self.ngram_vectors = defaultdict(lambda: np.random.randn(embed_dim) * 0.1)

    def get_ngrams(self, word):
        padded = f"<{word}>"
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i:i + n])
        return ngrams

    def get_word_vector(self, word):
        ngrams = self.get_ngrams(word)
        if not ngrams:
            return np.zeros(self.embed_dim)
        return np.mean([self.ngram_vectors[ng] for ng in ngrams], axis=0)

    def most_similar(self, word, vocab, topn=5):
        target = self.get_word_vector(word)
        scores = []
        for w in vocab:
            sim = np.dot(target, self.get_word_vector(w)) / (
                np.linalg.norm(target) * np.linalg.norm(self.get_word_vector(w)) + 1e-10
            )
            scores.append((w, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]

model = CharNgramEmbedding(embed_dim=20)
vocab = ["apple", "application", "apply", "banana", "orange"]

print("'apple' 的 n-grams:", model.get_ngrams("apple"))
print("'apple' 向量:", model.get_word_vector("apple")[:5])
print("'apples' (OOV) 与词汇的相似度:", model.most_similar("apples", vocab))
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.decomposition import PCA
import numpy as np

sentences = [["apple", "application", "apply", "banana", "orange", "fruit"]]
model = FastText(sentences=sentences, vector_size=50, min_n=3, max_n=6, min_count=1, epochs=50)

words = ["apple", "application", "apply", "apples", "banana", "orange"]
vecs = np.array([model.wv[w] for w in words])

pca = PCA(n_components=2)
coords = pca.fit_transform(vecs)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1])
for i, w in enumerate(words):
    color = "red" if w == "apples" else "blue"
    plt.annotate(w, (coords[i, 0], coords[i, 1]), color=color)
plt.title("Char2Vec: 红色为OOV词，蓝色为训练词")
plt.tight_layout()
plt.savefig("char2vec_pca.png", dpi=150)
plt.show()
```

## 10. 模型评估

- **OOV 覆盖率**：测试集中能成功生成向量的 OOV 词比例
- **形态学类比**：如 "walk" → "walked" 对应 "run" → "runned"
- **下游任务**：NER、POS tagging 等序列标注任务的 F1 值
- **词相似度**：与人工标注的相似度数据集计算相关性

## 11. 常见问题与易错点

- **n-gram 范围选择**：`min_n` 太小（1-2）会产生过多无意义片段，太大则失去细粒度
- **中文处理**：中文"字符级"就是字级别，n-gram 对中文的效果不如对英文显著
- **边界符号**：添加 `<` 和 `>` 作为词边界标记可以区分词首、词尾的 n-gram
- **哈希冲突**：fastText 使用哈希映射 n-gram，bucket 大小设置不当会导致冲突

## 12. 学习总结

Char2Vec 通过将词分解为字符级 n-gram 或子词片段来构建词表示，核心优势是天然处理 OOV 和共享形态信息。fastText 是最流行的实现，在 Word2Vec 的基础上引入了子词向量求和。Char2Vec 是从静态词嵌入走向更灵活文本表示的重要一步。

## 13. 练习题与思考题（含答案）

**Q1**：词 "unhappiness" 的字符 3-gram 集合（含边界标记）有哪些？

**A1**：`<un`, `unh`, `nha`, `hap`, `app`, `ppi`, `pin`, `ine`, `nes`, `ess`, `ss>`，共 11 个。

**Q2**：为什么 Char2Vec 能处理 OOV 而 Word2Vec 不能？

**A2**：Word2Vec 的词汇表是固定的，每个词对应唯一向量；Char2Vec 通过组合字符 n-gram 的向量来构建词表示，任何新词都可以分解为已知的字符 n-gram，因此天然支持 OOV。

**Q3**：字符级 CNN 和字符级 n-gram 的主要区别是什么？

**A3**：n-gram 方法是无参数的 bag-of-features 组合；CNN 通过卷积核学习字符序列的局部模式，能捕获字符间的顺序关系和更复杂的组合规则。

## 14. 学习路径建议

1. 掌握 Word2Vec → 2. 理解 Char2Vec / fastText（子词嵌入）→ 3. 学习 ELMo（字符级 + 上下文）→ 4. 学习 BERT（子词分词 + Transformer）→ 5. 探索 BPE / WordPiece 等子词分词算法
