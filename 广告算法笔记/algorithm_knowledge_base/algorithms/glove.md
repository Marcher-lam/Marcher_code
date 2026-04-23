# GloVe 学习文档

## 1. 算法基础认知

GloVe（Global Vectors for Word Representation）是斯坦福大学的 Pennington 等人于 2014 年提出的词嵌入方法。它结合了全局矩阵分解方法（如 LSA）和局部上下文窗口方法（如 Word2Vec）的优点，利用词的共现统计信息来学习词向量。

核心思想：**词向量之差应该编码词之间的比率关系，这种关系可以从共现概率的比率中推导出来**。

## 2. 核心原理

GloVe 基于**词-词共现矩阵**，统计语料中词 $j$ 在词 $i$ 上下文中出现的次数。关键观察：

对于词 $i$, $j$ 和探测词 $k$，共现概率的比率 $\frac{P_{ik}}{P_{jk}}$ 携带了丰富的语义信息。例如：

| 比率 | $k$ = "冰" | $k$ = "蒸汽" | $k$ = "水" |
|------|-----------|-------------|-----------|
| $P(k\mid$"固体"$)/P(k\mid$"气体"$)$ | 大 | 小 | ≈1 |

这种比率关系可以用向量运算建模：$w_i^T w_k + b_i + b_k \approx \log X_{ik}$。

## 3. 数学公式与推导

### 共现矩阵

$X_{ij}$ 表示词 $j$ 出现在词 $i$ 上下文窗口中的次数。

$$P_{ij} = \frac{X_{ij}}{\sum_k X_{ik}}$$

### 目标函数

GloVe 的加权最小二乘目标函数：

$$J = \sum_{i=1}^{V} \sum_{j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

其中：
- $\mathbf{w}_i$ 是词 $i$ 的目标向量
- $\tilde{\mathbf{w}}_j$ 是词 $j$ 的上下文向量
- $b_i$, $\tilde{b}_j$ 是偏置项
- $f(X_{ij})$ 是权重函数

### 权重函数

$$f(x) = \begin{cases} (x / x_{\max})^{\alpha} & x < x_{\max} \\ 1 & x \geq x_{\max} \end{cases}$$

默认 $x_{\max} = 100$, $\alpha = 0.75$。这个函数的作用是：给共现次数适中的词对较高权重，避免高频词对（如停用词）主导损失。

### 最终词向量

训练完成后，取目标向量和上下文向量的和作为最终词向量：

$$\mathbf{v}_i = \mathbf{w}_i + \tilde{\mathbf{w}}_i$$

## 4. 训练过程讲解

1. **构建共现矩阵**：遍历语料，统计所有词对的共现次数（加权：距离越近权重越大）
2. **初始化参数**：随机初始化 $\mathbf{W}$, $\tilde{\mathbf{W}}$, $b$, $\tilde{b}$
3. **优化**：使用 AdaGrad 或 SGD 最小化目标函数 $J$
4. **提取向量**：$\mathbf{v}_i = \mathbf{w}_i + \tilde{\mathbf{w}}_i$

共现矩阵只需构建一次，之后可以重复使用，这是 GloVe 相对 Word2Vec 的一个优势。

## 5. 应用场景

- 通用词向量预训练（提供 50/100/200/300 维的公开向量）
- 文本分类、情感分析的特征输入
- 命名实体识别（NER）
- 机器翻译中的词对齐
- 词类比和语义相似度计算

## 6. 优缺点分析

**优点**：
- 利用全局统计信息，训练更充分
- 训练速度快（共现矩阵只需计算一次）
- 在词类比任务上表现优秀
- 公开预训练向量质量高，广泛使用

**缺点**：
- 静态嵌入，无法处理多义词
- 需要存储共现矩阵（内存消耗大）
- 对语料预处理质量敏感
- 不支持 OOV（除非用子词扩展）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import gensim.downloader as api

info = api.info()
glove_model = api.load("glove-wiki-gigaword-50")

print("词汇表大小:", len(glove_model))
print("'king' 向量:", glove_model["king"][:5])

result = glove_model.most_similar("king", topn=5)
print("与 'king' 最相似的词:", result)

analogy = glove_model.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
print("king - man + woman ≈:", analogy)
```

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = "glove.6B.50d.txt"
w2v_output_file = "glove.6B.50d.w2v.txt"
glove2word2vec(glove_input_file, w2v_output_file)
model = KeyedVectors.load_word2vec_format(w2v_output_file)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from collections import defaultdict

class GloVeManual:
    def __init__(self, vocab_size, embed_dim=50):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_ctx = np.random.randn(vocab_size, embed_dim) * 0.1
        self.b = np.zeros(vocab_size)
        self.b_ctx = np.zeros(vocab_size)
        self.gradsq_W = np.ones_like(self.W)
        self.gradsq_W_ctx = np.ones_like(self.W_ctx)
        self.gradsq_b = np.ones_like(self.b)
        self.gradsq_b_ctx = np.ones_like(self.b_ctx)

    def weight_func(self, x, x_max=100, alpha=0.75):
        return (x / x_max) ** alpha if x < x_max else 1.0

    def train_step(self, cooccurrences, lr=0.05):
        total_loss = 0.0
        for i, j, x_ij in cooccurrences:
            diff = np.dot(self.W[i], self.W_ctx[j]) + self.b[i] + self.b_ctx[j] - np.log(x_ij)
            f_x = self.weight_func(x_ij)
            loss = f_x * diff * diff
            total_loss += loss

            grad_main = f_x * diff * self.W_ctx[j]
            grad_ctx = f_x * diff * self.W[i]
            grad_b = f_x * diff
            grad_b_ctx = f_x * diff

            self.gradsq_W[i] += grad_main ** 2
            self.gradsq_W_ctx[j] += grad_ctx ** 2
            self.gradsq_b[i] += grad_b ** 2
            self.gradsq_b_ctx[j] += grad_b_ctx ** 2

            self.W[i] -= lr * grad_main / np.sqrt(self.gradsq_W[i])
            self.W_ctx[j] -= lr * grad_ctx / np.sqrt(self.gradsq_W_ctx[j])
            self.b[i] -= lr * grad_b / np.sqrt(self.gradsq_b[i])
            self.b_ctx[j] -= lr * grad_b_ctx / np.sqrt(self.gradsq_b_ctx[j])

        return total_loss

    def get_embedding(self, idx):
        return self.W[idx] + self.W_ctx[idx]

np.random.seed(42)
model = GloVeManual(vocab_size=5, embed_dim=20)
coocs = [(0, 1, 5.0), (0, 2, 3.0), (1, 2, 4.0), (1, 3, 2.0), (2, 4, 6.0)]
for epoch in range(100):
    loss = model.train_step(coocs)
print(f"最终损失: {loss:.4f}")
print("词0的嵌入:", model.get_embedding(0)[:5])
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

words = ["king", "queen", "man", "woman", "paris", "france", "london", "england"]
glove = api.load("glove-wiki-gigaword-50")
vecs = np.array([glove[w] for w in words])

from sklearn.decomposition import PCA
coords = PCA(n_components=2).fit_transform(vecs)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1])
for i, w in enumerate(words):
    plt.annotate(w, (coords[i, 0], coords[i, 1]))
plt.title("GloVe 词向量 PCA 可视化")
plt.tight_layout()
plt.savefig("glove_pca.png", dpi=150)
plt.show()
```

可视化中可以看到 "king-queen" 和 "man-woman" 形成平行向量，验证了词向量中的线性关系。

## 10. 模型评估

- **词类比准确率**：Google Analogy Dataset 上的准确率是标准评测指标
- **词相似度相关性**：与 WordSim-353、SimLex-999 等人工标注数据集的 Spearman 相关性
- **下游任务**：NER、情感分析等任务的 F1 分数
- **GloVe 通常在词类比任务上优于 Word2Vec**，因为全局统计信息更适合捕获类比关系

## 11. 常见问题与易错点

- **共现矩阵内存**：词汇量 10 万时，共现矩阵需约 40GB（float32），需要对称化或稀疏存储
- **窗口加权**：GloVe 对共现计数使用距离加权（距离为 $d$ 的词贡献 $1/d$），不是简单计数
- **向量拼接**：取 $\mathbf{w} + \tilde{\mathbf{w}}$ 效果通常好于单独使用 $\mathbf{w}$
- **预训练语料**：不同语料（Wikipedia vs Common Crawl）训练的向量质量差异大

## 12. 学习总结

GloVe 通过对词共现矩阵做加权最小二乘分解来学习词向量，巧妙地将全局统计信息融入词嵌入。它的核心公式 $w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}$ 建立了向量空间与共现统计之间的桥梁。GloVe 和 Word2Vec 是静态词嵌入的两大代表，各有优势，共同局限是无法处理多义词。

## 13. 练习题与思考题（含答案）

**Q1**：GloVe 的权重函数 $f(x)$ 为什么要上界截断（$x \geq x_{\max}$ 时为 1）？

**A1**：高频词对（如 "the-the"）的共现次数极高，不截断会主导损失函数。截断确保高频对权重不超过 1，让模型更关注中等频率的有意义词对。

**Q2**：GloVe 和 Word2Vec 的本质区别是什么？

**A2**：Word2Vec 基于局部上下文窗口，通过采样训练；GloVe 基于全局共现矩阵，直接拟合共现统计的对数。Word2Vec 是隐式地利用共现信息，GloVe 则是显式地利用。

**Q3**：为什么 GloVe 取 $w_i + \tilde{w}_i$ 作为最终向量而不是只用 $w_i$？

**A3**：因为模型具有对称性（$w_i^T \tilde{w}_j \approx \log X_{ij}$ 等价于 $\tilde{w}_j^T w_i \approx \log X_{ji}$），两个向量都编码了有用信息。相加相当于集成两个视角的表示，通常效果更好。

## 14. 学习路径建议

1. 理解 One-Hot / TF-IDF → 2. 掌握 Word2Vec（局部上下文）→ 3. 学习 GloVe（全局统计）→ 4. 对比 Word2Vec 与 GloVe 的异同 → 5. 学习 ELMo / BERT（上下文嵌入）
