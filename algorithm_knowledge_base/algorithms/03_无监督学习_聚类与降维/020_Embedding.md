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

## 4. 训练过程讲解
### 训练步骤
1. **数据准备**：收集并清洗数据，划分训练/测试集
2. **特征工程**：标准化、编码等预处理
3. **模型初始化**：设置超参数
4. **模型训练**：使用训练数据拟合参数
5. **交叉验证**：K折CV选择最优超参数
6. **模型评估**：测试集最终评估

## 5. 应用场景

Embedding在以下领域有广泛应用：

- 客户细分与用户画像
- 信用评分与风险评估
- 医疗诊断辅助决策
- 文本分类与情感分析
- 推荐系统中的特征处理

在工业实践中，Embedding通常与完整的数据管道配合使用。选择Embedding时需要根据数据特点、性能要求和计算资源综合考量。

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import numpy as np

class EmbeddingScratch:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr, self.n_iter, self.losses = lr, n_iter, []
    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            err = X @ self.w + self.b - y
            self.losses.append(np.mean(err**2))
            self.w -= self.lr * (2/n) * X.T @ err
            self.b -= self.lr * (2/n) * np.sum(err)
        return self
    def predict(self, X): return X @ self.w + self.b

np.random.seed(42)
X = np.random.randn(200, 3)
y = 2*X[:,0] - X[:,1] + 0.5*X[:,2] + np.random.randn(200)*0.1
m = EmbeddingScratch().fit(X, y)
print(f"Loss: {m.losses[-1]:.6f}")
```

## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 14. 学习路径建议

### 前置知识
线性代数、概率统计、Python、NumPy

### 学习顺序
1. 先理解原理：掌握Embedding核心思想和数学基础
2. 动手实践：运行代码，观察实验结果
3. 深入理解：阅读原始论文，理解设计动机
4. 项目实战：真实数据集上应用Embedding

### 进阶方向
集成学习、特征工程、AutoML

### 推荐资源
- 搜索Embedding原始论文和综述
- GitHub优秀实现
- Coursera/Stanford相关课程

