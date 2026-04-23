# One-Hot 学习文档

## 1. 算法基础认知

One-Hot 编码（独热编码）是最基础的离散特征表示方法。它的核心思想是：将每个类别变量映射为一个仅有一个位置为 1、其余位置全为 0 的向量。例如，词汇表有 4 个词 `["猫", "狗", "鸟", "鱼"]`，则"猫"表示为 `[1, 0, 0, 0]`，"狗"表示为 `[0, 1, 0, 0]`。

它是自然语言处理和分类特征工程中最原始的表示方式，也是理解 Word2Vec、GloVe 等词嵌入方法的起点。

## 2. 核心原理

假设词汇表大小为 $V$，对于第 $i$ 个词 $w_i$，其 One-Hot 向量为：

$$\mathbf{e}_i = [0, 0, \ldots, 1, \ldots, 0, 0] \in \mathbb{R}^V$$

其中只有第 $i$ 个分量为 1，其余全为 0。

**关键性质**：任意两个不同词的 One-Hot 向量内积为 0，即它们在向量空间中完全正交，无法表达语义相似性。

## 3. 数学公式与推导

给定词汇表 $\mathcal{V} = \{w_1, w_2, \ldots, w_V\}$，定义映射函数：

$$f: w_i \mapsto \mathbf{e}_i, \quad \mathbf{e}_i \in \{0, 1\}^V, \quad \mathbf{e}_i(j) = \begin{cases} 1 & j = i \\ 0 & j \neq i \end{cases}$$

两个词的余弦相似度：

$$\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \cdot \|\mathbf{e}_j\|} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

这证明了 One-Hot 编码无法捕获词之间的语义关系。

## 4. 训练过程讲解

One-Hot 编码无需训练，是一种确定性的映射方法。过程如下：

1. 构建词汇表：遍历语料，收集所有唯一词，建立词到索引的映射 `word2idx`
2. 编码：对每个词，生成一个长度为 $V$ 的零向量，将对应位置设为 1

## 5. 应用场景

- 分类特征编码（如性别、城市等类别变量输入机器学习模型）
- NLP 中词的初始表示（作为词嵌入层的前身）
- 多分类任务标签的表示方式

## 6. 优缺点分析

**优点**：
- 实现极其简单，无需训练
- 可解释性强，每个维度对应一个类别

**缺点**：
- 维度灾难：词汇量为 10 万时，每个词需要 10 万维向量
- 极度稀疏：仅一个非零元素，存储和计算效率低
- 语义鸿沟：任意两个词的相似度为 0，无法表达语义关系
- 无法处理未登录词（OOV）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = np.array([["猫"], ["狗"], ["鸟"], ["鱼"]]).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
onehot = encoder.fit_transform(data)

print("类别:", encoder.categories_)
print("One-Hot 编码:\n", onehot)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class OneHotEncoderManual:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def fit(self, words):
        unique_words = sorted(set(words))
        self.word2idx = {w: i for i, w in enumerate(unique_words)}
        self.idx2word = {i: w for i, w in enumerate(unique_words)}
        self.vocab_size = len(unique_words)

    def transform(self, word):
        vec = np.zeros(self.vocab_size)
        vec[self.word2idx[word]] = 1.0
        return vec

    def transform_batch(self, words):
        return np.array([self.transform(w) for w in words])

words = ["猫", "狗", "鸟", "鱼", "猫"]
enc = OneHotEncoderManual()
enc.fit(words)
for w in ["猫", "狗"]:
    print(f"{w} -> {enc.transform(w)}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

words = ["猫", "狗", "鸟", "鱼"]
vectors = np.eye(len(words))

plt.figure(figsize=(6, 4))
plt.imshow(vectors, cmap="Blues", aspect="auto")
for i in range(len(words)):
    for j in range(len(words)):
        plt.text(j, i, str(int(vectors[i, j])), ha="center", va="center")
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.title("One-Hot 编码矩阵")
plt.xlabel("维度索引")
plt.ylabel("词")
plt.tight_layout()
plt.savefig("onehot_visual.png", dpi=150)
plt.show()
```

可视化结果为一个单位矩阵，每行只有一个蓝色方块（值为 1），直观展示了稀疏性。

## 10. 模型评估

One-Hot 本身不是预测模型，评估维度主要是：
- **编码效率**：向量维度 = 词汇表大小，空间复杂度 $O(V)$
- **下游任务表现**：直接用 One-Hot 特征输入分类器通常效果差（维度过高导致稀疏）
- **改进方案**：降维（PCA）或使用稠密嵌入（Word2Vec、GloVe）

## 11. 常见问题与易错点

- **未处理新类别**：测试数据中出现训练集未见过的类别会导致编码失败，需要预留"未知"类别
- **维度爆炸**：对高基数特征（如用户 ID）直接 One-Hot 会导致内存溢出
- **与标签编码混淆**：标签编码（Label Encoding）将类别映射为整数，One-Hot 将其展开为二值向量，二者适用场景不同
- **忽略顺序信息**：One-Hot 假设类别间无序，若类别本身有序（如"低中高"），One-Hot 会丢失顺序信息

## 12. 学习总结

One-Hot 是最朴素的特征表示方法，它通过高维稀疏的二值向量唯一标识每个类别。虽然简单直观，但维度灾难和语义缺失是致命缺陷。理解 One-Hot 的局限性，是学习 Word2Vec 等稠密嵌入方法的重要前提——这些方法正是为了克服 One-Hot 的不足而提出的。

## 13. 练习题与思考题（含答案）

**Q1**：词汇表大小为 50000，每个 One-Hot 向量占多少字节（float64）？

**A1**：$50000 \times 8 = 400000$ 字节 ≈ 390 KB / 词，一篇 1000 词的文章需约 390 MB。

**Q2**：为什么两个不同词的 One-Hot 向量余弦相似度为 0？

**A2**：因为两个向量在不同的位置为 1，内积为 0，所以余弦相似度为 0。

**Q3**：One-Hot 编码和词嵌入（如 Word2Vec）的本质区别是什么？

**A3**：One-Hot 是稀疏的高维确定映射，维度等于词表大小；词嵌入是稠密的低维学习表示，维度通常为 100-300，且能表达语义相似性。

## 14. 学习路径建议

1. 掌握 One-Hot 编码原理 → 2. 学习 TF-IDF（加权改进）→ 3. 学习 Word2Vec（稠密嵌入）→ 4. 学习 GloVe（全局统计信息）→ 5. 学习 BERT（上下文相关嵌入）
