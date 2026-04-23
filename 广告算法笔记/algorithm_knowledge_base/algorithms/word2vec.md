# Word2Vec 学习文档

## 1. 算法基础认知

Word2Vec 是 Google 的 Mikolov 等人于 2013 年提出的词嵌入方法。它通过浅层神经网络从大规模文本中学习词的稠密向量表示，使得语义相近的词在向量空间中距离接近。

Word2Vec 提出了两种架构：**CBOW**（Continuous Bag-of-Words，连续词袋模型）和 **Skip-gram**（跳字模型），以及两种加速训练策略：**Negative Sampling**（负采样）和 **Hierarchical Softmax**（层次 Softmax）。

## 2. 核心原理

**CBOW**：利用上下文词预测中心词。输入是周围若干词的 One-Hot，输出是中心词的概率分布。

**Skip-gram**：利用中心词预测上下文词。输入是中心词的 One-Hot，输出是上下文词的概率分布。Skip-gram 在实践中效果通常更好。

核心假设（分布式假设）：出现在相似上下文中的词具有相似的含义。

## 3. 数学公式与推导

### Skip-gram 目标函数

给定语料 $\mathcal{C}$，窗口大小 $m$，Skip-gram 最大化：

$$\mathcal{L} = \sum_{w_t \in \mathcal{C}} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t)$$

其中条件概率用 Softmax 定义：

$$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}_{w_O}^{\top} \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}_w^{\top} \mathbf{v}_{w_I})}$$

其中 $\mathbf{v}_{w_I}$ 是输入词向量，$\mathbf{v}_{w_O}$ 是输出词向量，$V$ 是词汇表大小。

### 负采样

直接计算 Softmax 分母复杂度 $O(V)$，负采样将其简化为二分类：

$$\log \sigma(\mathbf{v}_{w_O}^{\top} \mathbf{v}_{w_I}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_k}^{\top} \mathbf{v}_{w_I})]$$

其中 $\sigma$ 是 Sigmoid 函数，$K$ 个负样本从噪声分布 $P_n(w) \propto \text{freq}(w)^{3/4}$ 中采样。

### Hierarchical Softmax

用二叉树（哈夫曼树）代替 Softmax，将计算复杂度从 $O(V)$ 降到 $O(\log V)$。每个叶子节点是词汇表中的词，沿路径做二分类。

## 4. 训练过程讲解

**Skip-gram 训练流程**：

1. 初始化词嵌入矩阵 $W \in \mathbb{R}^{V \times d}$（$d$ 为嵌入维度）
2. 遍历语料中的每个词 $w_t$，提取窗口内的上下文词
3. 对每个 (中心词, 上下文词) 对：
   - 正样本：上下文词，标签为 1
   - 负样本：从噪声分布采样 $K$ 个词，标签为 0
4. 通过 Sigmoid 计算损失，反向传播更新词向量
5. 训练完成后，$W$ 即为词嵌入矩阵

## 5. 应用场景

- 文本分类、情感分析中的词向量特征
- 命名实体识别（NER）、词性标注等序列标注任务
- 机器翻译中的词表示
- 推荐系统中的 item 嵌入（Item2Vec）
- 词类比：`国王 - 男人 + 女人 ≈ 女王`

## 6. 优缺点分析

**优点**：
- 稠密低维表示（通常 100-300 维），效率高
- 捕获语义相似性（"猫"和"狗"向量接近）
- 训练速度快，支持大规模语料
- 预训练向量可迁移使用

**缺点**：
- 静态嵌入，同一词在不同语境中向量相同（多义词问题）
- 无法处理 OOV（未登录词）
- 依赖语料质量和规模
- 只考虑局部上下文窗口，未利用全局统计信息

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

print("示例语料:", common_texts[:3])

model = Word2Vec(
    sentences=common_texts,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    negative=5,
    epochs=10,
)

print("词 'human' 的向量（前10维）:", model.wv["human"][:10])
print("词 'human' 的最相似词:", model.wv.most_similar("human", topn=3))
print("词汇表大小:", len(model.wv))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SkipGramManual:
    def __init__(self, vocab_size, embed_dim=50):
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx, context_idx, neg_indices, lr=0.025):
        v_c = self.W_in[center_idx]
        loss = 0.0
        grad = np.zeros_like(v_c)

        v_o = self.W_out[context_idx]
        score = np.dot(v_o, v_c)
        loss += -np.log(self.sigmoid(score) + 1e-10)
        grad += (self.sigmoid(score) - 1) * v_o
        self.W_out[context_idx] -= lr * (self.sigmoid(score) - 1) * v_c

        for neg_idx in neg_indices:
            v_n = self.W_out[neg_idx]
            score = np.dot(v_n, v_c)
            loss += -np.log(self.sigmoid(-score) + 1e-10)
            grad += self.sigmoid(score) * v_n
            self.W_out[neg_idx] -= lr * self.sigmoid(score) * v_c

        self.W_in[center_idx] -= lr * grad
        return loss

    def get_embedding(self, idx):
        return self.W_in[idx]

np.random.seed(42)
V, d = 10, 20
model = SkipGramManual(V, d)
center = np.random.randint(V)
context = np.random.randint(V)
negs = np.random.choice([i for i in range(V) if i != center and i != context], size=3, replace=False)
loss = model.train_pair(center, context, negs)
print(f"Loss: {loss:.4f}")
print(f"词{center}的嵌入向量: {model.get_embedding(center)[:5]}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn.decomposition import PCA

model = Word2Vec(sentences=common_texts, vector_size=50, window=5, min_count=1, sg=1, epochs=50)
words = list(model.wv.key_to_index.keys())
vecs = np.array([model.wv[w] for w in words])

pca = PCA(n_components=2)
coords = pca.fit_transform(vecs)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
for i, w in enumerate(words):
    plt.annotate(w, (coords[i, 0], coords[i, 1]))
plt.title("Word2Vec 词向量 PCA 可视化")
plt.tight_layout()
plt.savefig("word2vec_pca.png", dpi=150)
plt.show()
```

## 10. 模型评估

- **内部评估**：词类比任务（king - man + woman ≈ queen），衡量语义关系的捕获能力
- **外部评估**：将词向量作为下游任务（分类、NER）的输入特征，比较 F1 分数
- **相似度评估**：与人工标注的词相似度数据集（如 WordSim-353）计算 Spearman 相关性

## 11. 常见问题与易错点

- **窗口大小选择**：小窗口（2-5）捕获语法关系，大窗口（5-10）捕获语义关系
- **负样本数量**：小语料 5-20，大语料 2-5
- **最小词频**：`min_count` 过低会引入噪声，通常设为 5
- **CBOW vs Skip-gram**：CBOW 训练快，适合小语料；Skip-gram 效果好，适合大语料
- **重训练**：每次重新训练结果不同（随机初始化 + 采样），需固定随机种子

## 12. 学习总结

Word2Vec 通过浅层神经网络学习词的稠密向量，核心思想是"上下文相似的词语义相似"。CBOW 用上下文预测中心词，Skip-gram 用中心词预测上下文。负采样和层次 Softmax 解决了 Softmax 的计算瓶颈。Word2Vec 的最大局限是静态嵌入，无法区分多义词，BERT 等上下文模型解决了这一问题。

## 13. 练习题与思考题（含答案）

**Q1**：Skip-gram 的负采样中，为什么噪声分布用 $P_n(w) \propto \text{freq}(w)^{3/4}$ 而不是均匀分布？

**A1**：$3/4$ 次幂是对高频词的适度降权（相比原始词频）同时仍保持高频词被采样的概率高于低频词，兼顾了采样效率和训练效果。

**Q2**：Word2Vec 的 "king - man + woman ≈ queen" 性质说明了什么？

**A2**：说明词向量空间中存在线性结构，性别关系可以用一个方向向量表示，这体现了嵌入捕获了语义关系的几何结构。

**Q3**：为什么 Word2Vec 无法处理多义词？

**A3**：Word2Vec 为每个词学习唯一一个静态向量，而多义词在不同语境中含义不同。例如"苹果"在水果语境和公司语境中应使用不同向量，但 Word2Vec 只能给出一个混合向量。

## 14. 学习路径建议

1. 理解 One-Hot 和 TF-IDF → 2. 掌握 Word2Vec 原理（CBOW / Skip-gram）→ 3. 学习 GloVe（全局统计信息）→ 4. 学习 fastText（子词嵌入）→ 5. 学习 BERT（上下文嵌入）
