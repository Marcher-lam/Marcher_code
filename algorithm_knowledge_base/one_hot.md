# One-Hot Encoding 独热编码 学习文档

> One-Hot Encoding是机器学习中将离散类别转换为二进制向量表示的基本技术

---

## 1. 算法基础认知

### 1.1 一句话定义

**One-Hot Encoding（独热编码）** 是一种使用二进制向量表示分类变量（或类别）的方法，其中每个类别对应一个位置为1、其余位置为0的向量。

### 1.2 直觉类比

想象一个"城市路灯系统"：每盏灯代表一个类别，只有当前灯亮（值为1），其他灯都灭（值为0）。例如，如果有4个类别，我们用4盏灯表示：类别A是 [1,0,0,0]，类别B是[0,1,0,0]，以此类推。每时刻只有一盏灯亮起，这就是"独热"的含义——同一时刻只有一个"热"（激活）。

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 1960s | 数字电路的"独热"概念出现 |
| 1980s | 神经网络中的One-Hot表示 |
| 2000s | 特征工程中的标准处理 |
| 2010s | 深度学习NLP的词表表示 |
| 2020s | Transformer的Token Embedding |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 特征编码 / 离散化 |
| 输出 | 二进制向量 |
| 核心 | 互斥表示 |

### 1.5 前置知识

- 线性代数基础
- Python基础
- 机器学习概念

---

## 2. 核心原理

### 2.1 基本原理

将类别变量转换为二进制向量：

| 类别 | One-Hot编码 |
|------|------------|
| A | [1, 0, 0, 0, ..., 0] |
| B | [0, 1, 0, 0, ..., 0] |
| C | [0, 0, 1, 0, ..., 0] |
| D | [0, 0, 0, 1, ..., 0] |
| ... | ... |

### 2.2 数学表示

**编码函数**：
$$f: \mathcal{C} \rightarrow \{0,1\}^K$$

其中 $K$ 是类别数量，满足：
$$f(c_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 逆向解码

**解码函数**：
$$f^{-1}: \{0,1\}^K \rightarrow \mathcal{C}$$
$$f^{-1}(v) = \text{argmax}(v)$$

### 2.4 工作流程

```python
# One-Hot 编码流程
def one_hot_encode(categories):
    # 1. 获取唯一类别
    unique_cats = sorted(set(categories))
    
    # 2. 创建映射
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    
    # 3. 编码
    encoded = []
    for cat in categories:
        vector = [0] * len(unique_cats)
        vector[cat_to_idx[cat]] = 1
        encoded.append(vector)
    
    return np.array(encoded)
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $K$ | 类别数量 |
| $c$ | 类别 |
| $v$ | 编码向量 |
| $n$ | 样本数量 |

### 3.2 编码矩阵

**输入向量**：
$$\mathcal{C} = [c_1, c_2, ..., c_n]^T$$

**One-Hot矩阵**：
$$V = [v_1, v_2, ..., v_n]^T \in \mathbb{R}^{n \times K}$$

其中：
$$v_i[j] = \begin{cases} 1 & \text{if } c_i = \text{idx}^{-1}(j) \\ 0 & \text{otherwise} \end{cases}$$

### 3.3 维度关系

**特征空间维度**：
- 原始：1维（类别本身）
- One-Hot后：$K$维

**信息守恒**：使用One-Hot后的熵不变：
$$H(\mathcal{C}) = H(V) = \log_2 K$$

### 3.4 与Label Encoding对比

| 方法 | 维度 | 距离关系 | 适用模型 |
|------|------|----------|----------|
| Label | 1 | 等距 | 决策树 |
| One-Hot | K | 稀疏 | 神经网络 |

### 3.5 Embedding扩展

**可学习的One-Hot（Embedding）**：
$$e = W \cdot v$$

其中 $W \in \mathbb{R}^{K \times D}$ 是可学习的嵌入矩阵，$D$ 是嵌入维度。

---

## 4. Python实现

### 4.1 基础实现

```python
import numpy as np
import pandas as pd
import torch


def one_hot_encode(categories):
    """
    将类别列表转换为One-Hot编码
    
    Args:
        categories: list, 类别列表
    
    Returns:
        numpy.ndarray, One-Hot编码矩阵
    """
    # 获取唯一类别并排序（保证一致性）
    unique_cats = sorted(set(categories))
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    
    n_samples = len(categories)
    n_classes = len(unique_cats)
    
    # 初始化零矩阵
    one_hot = np.zeros((n_samples, n_classes))
    
    # 填充
    for i, cat in enumerate(categories):
        one_hot[i, cat_to_idx[cat]] = 1
    
    return one_hot


def one_hot_decode(one_hot_matrix):
    """
    One-Hot解码为类别列表
    
    Args:
        one_hot_matrix: One-Hot编码矩阵
    
    Returns:
        list, 类别列表
    """
    indices = np.argmax(one_hot_matrix, axis=1)
    return indices


def one_hot_encode_pandas(categories):
    """使用pandas实现"""
    df = pd.DataFrame({'category': categories})
    one_hot_df = pd.get_dummies(df, columns=['category'], prefix_sep='_')
    return one_hot_df.values


class OneHotEncoder:
    """One-Hot编码器类"""
    
    def __init__(self, categories):
        self.categories = sorted(set(categories))
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_cat = {idx: cat for cat, idx in self.cat_to_idx.items()}
        
        self.n_classes = len(self.categories)
        self.fitted = True
    
    def transform(self, categories):
        """转换"""
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        n_samples = len(categories)
        one_hot = np.zeros((n_samples, self.n_classes))
        
        for i, cat in enumerate(categories):
            if cat in self.cat_to_idx:
                one_hot[i, self.cat_to_idx[cat]] = 1
            else:
                raise ValueError(f"Unknown category: {cat}")
        
        return one_hot
    
    def inverse_transform(self, one_hot):
        """逆向转换"""
        indices = np.argmax(one_hot, axis=1)
        return [self.idx_to_cat[idx] for idx in indices]
    
    def fit(self, categories):
        """拟合"""
        self.categories = sorted(set(categories))
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_cat = {idx: cat for cat, idx in self.cat_to_idx.items()}
        self.n_classes = len(self.categories)
        self.fitted = True
        return self
    
    def fit_transform(self, categories):
        """拟合并转换"""
        return self.fit(categories).transform(categories)
```

### 4.2 PyTorch实现

```python
import torch
import torch.nn as nn


class OneHotEmbedding(nn.Module):
    """可学习的One-Hot Embedding"""
    
    def __init__(self, num_classes, embedding_dim):
        super(OneHotEmbedding, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # 可学习的嵌入
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, indices):
        """
        Args:
            indices: (batch_size,) 类别索引
        
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.embedding(indices)


class OneHotLinear(nn.Module):
    """One-Hot后接线性层"""
    
    def __init__(self, num_classes, output_dim):
        super(OneHotLinear, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, output_dim)
    
    def forward(self, one_hot):
        """
        Args:
            one_hot: (batch_size, num_classes)
        
        Returns:
            output: (batch_size, output_dim)
        """
        return self.linear(one_hot)


def create_one_hot_indices(num_classes, batch_size):
    """创建One-Hot索引张量"""
    indices = torch.randint(0, num_classes, (batch_size,))
    one_hot = F.one_hot(indices, num_classes).float()
    return one_hot, indices
```

### 4.3 Sklearn实现

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


class OneHotEncoderAdvanced:
    """高级One-Hot编码器，处理缺失值"""
    
    def __init__(self, handle_unknown='ignore', sparse_output=False):
        self.encoder = OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=sparse_output,
            categories='auto'
        )
        self.fitted = False
    
    def fit(self, X):
        """拟fit"""
        X = np.array(X).reshape(-1, 1)
        self.encoder.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        """转换"""
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        X = np.array(X).reshape(-1, 1)
        return self.encoder.transform(X)
    
    def fit_transform(self, X):
        """拟合并转换"""
        X = np.array(X).reshape(-1, 1)
        return self.encoder.fit_transform(X)
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.encoder.get_feature_names_out()


def demo_sklearn():
    """Sklearn演示"""
    # 数据
    data = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
    
    # One-Hot编码
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(data)
    
    print("原始数据:", data)
    print("\nOne-Hot编码后:")
    print(encoded)
    print("\n类别名称:", encoder.get_feature_names_out())
    
    # 逆转换
    reversed_data = encoder.inverse_transform(encoded)
    print("\n逆转换结果:", reversed_data)
```

### 4.4 Keras/TensorFlow实现

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding


class OneHotKeras:
    """Keras中实现One-Hot"""
    
    @staticmethod
    def to_one_hot(labels, num_classes):
        """转换为One-Hot"""
        return tf.one_hot(labels, depth=num_classes)
    
    @staticmethod
    def from_one_hot(one_hot):
        """从One-Hot还原"""
        return tf.argmax(one_hot, axis=-1)


class CategoryEmbedding:
    """类别嵌入（可学习的One-Hot）"""
    
    def __init__(self, num_classes, embedding_dim):
        self.embedding = Embedding(num_classes, embedding_dim)
    
    def call(self, indices):
        return self.embedding(indices)
```

---

## 5. 代码示例

### 5.1 完整示例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def demo_one_hot():
    """完整演示"""
    
    print("=" * 60)
    print("One-Hot Encoding 演示")
    print("=" * 60)
    
    # 示例数据
    cities = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Beijing', 'Shanghai']
    
    print("\n原始数据:", cities)
    
    # One-Hot编码
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(cities)
    
    print("\nOne-Hot编码:")
    print(encoded)
    
    # 特征名称
    feature_names = encoder.get_feature_names_out(['city'])
    print("\n特征名称:", feature_names)
    
    # 可视化
    plt.figure(figsize=(10, 5))
    df = pd.DataFrame(encoded, columns=feature_names)
    sns.heatmap(df.T, cmap='Blues', cbar=True)
    plt.title('One-Hot Encoding Visualization')
    plt.xlabel('Sample')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('one_hot_visualization.png', dpi=150)
    plt.close()
    
    return encoded


def demo_neural_network():
    """神经网络中的One-Hot"""
    
    print("\n" + "=" * 60)
    print("神经网络中的One-Hot")
    print("=" * 60)
    
    # 类别数据
    labels = [0, 1, 2, 0, 1, 2, 0]
    num_classes = 3
    
    # PyTorch One-Hot
    labels_tensor = torch.tensor(labels)
    one_hot = F.one_hot(labels_tensor, num_classes).float()
    
    print("原始标签:", labels)
    print("\nOne-Hot编码:")
    print(one_hot)
    
    # 嵌入
    embedding = nn.Embedding(num_classes, 4)
    embedded = embedding(labels_tensor)
    
    print("\n嵌入向量 (dim=4):")
    print(embedded)
    
    return one_hot, embedded


def demo_text_classification():
    """文本分类中的One-Hot"""
    
    print("\n" + "=" * 60)
    print("文本分类中的One-Hot")
    print("=" * 60)
    
    # 词汇表
    vocab = ['the', 'cat', 'sat', 'on', 'mat']
    vocab_size = len(vocab)
    
    # 单词到索引
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 句子
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    
    # One-Hot编码每个词
    one_hot_matrix = np.zeros((len(sentence), vocab_size))
    
    for i, word in enumerate(sentence):
        one_hot_matrix[i, word2idx[word]] = 1
    
    print("句子:", sentence)
    print("\nOne-Hot矩阵:")
    print(one_hot_matrix)
    
    # 可视化
    plt.figure(figsize=(12, 4))
    sns.heatmap(one_hot_matrix, cmap='Blues', 
              xticklabels=vocab, yticklabels=range(len(sentence)))
    plt.title('Sentence One-Hot Encoding')
    plt.xlabel('Vocabulary')
    plt.ylabel('Word Position')
    plt.tight_layout()
    plt.savefig('sentence_one_hot.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    demo_one_hot()
    demo_neural_network()
    demo_text_classification()
```

---

## 6. 应用场景

### 6.1 机器学习应用

| 应用 | 说明 |
|------|------|
| **分类特征** | 类别转数值 |
| **神经网络** | Softmax输入 |
| **决策树** | 类别特征 |

### 6.2 自然语言处理

| 应用 | 说明 |
|------|------|
| **词表表示** | 词袋模型 |
| **词性标注** | POS标签 |
| **命名实体** | NER标签 |

### 6.3 深度学习

| 应用 | 说明 |
|------|------|
| **Word2Vec** | 词的分布式表示 |
| **Transformer** | Token ID |
| **SentencePiece** | 子词表示 |

### 6.4 代码示例

```python
# 处理缺失类别的One-Hot
def handle_unknown_categories(train_categories, test_categories):
    """处理训练和测试中的不同类别"""
    
    train_unique = set(train_categories)
    test_unique = set(test_categories)
    
    # 找出未见过的类别
    unknown = test_unique - train_unique
    
    if unknown:
        print(f"警告: 测试集中有未知类别: {unknown}")
        # 替换为_unknown_
        test_categories = ['_unknown_' if c in unknown else c for c in test_categories]
    
    # 重新编码
    encoder = OneHotEncoder()
    train_encoded = encoder.fit_transform(train_categories)
    test_encoded = encoder.transform(test_categories)
    
    return train_encoded, test_encoded
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **互斥** | 每个类别独立表示 |
| **直观** | 容易理解和实现 |
| **通用** | 适合大多数模型 |
| **无序** | 不引入类别顺序关系 |

### 7.2 缺点

| 缺点 | 说明 | 缓解 |
|------|------|------|
| **维度爆炸** | K类 -> K维 | 特征选择 |
| **稀疏** | 内存占用大 | 稀疏矩阵 |
| **无顺序** | 不适合有序数据 | Label Encoding |

### 7.3 对比Label Encoding

| 方法 | 适用模型 | 维度 | 顺序关系 |
|------|----------|------|----------|
| One-Hot | NN, SVM | K | 无 |
| Label | 决策树 | 1 | 有 |
| Embedding | 深度学习 | D | 学习得到 |

---

## 8. 常见问题与易错点

### 8.1 问题1：维度爆炸

**场景**：类别数量非常大（例如：词汇表10万词）

**解决方案**：使用稀疏编码或Hash编码
```python
# 稀疏编码
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=True)
sparse_onehot = encoder.fit_transform(data.reshape(-1, 1))
```

### 8.2 问题2：新类别

**问题**：测试集出现训练集没有的类别

**解决方案**：
```python
# 处理未知类别
if new_category not in encoder.categories_:
    # 忽略或映射到未知token
    pass
```

### 8.3 问题3：多标签

**问题**：一个样本多个标签

**解决方案**：不使用One-Hot，使用多标签编码

---

## 9. 学习总结

### 9.1 核心要点

1. **互斥表示**：每个类别独立维度
2. **稀疏**：大量0
3. **可逆**：可精确还原

### 9.2 关键公式

$$v_i[j] = \delta_{ij}$$

（Kronecker delta）

### 9.3 学习路径

One-Hot → Label Encoding → Embedding → Tokenizer

---

## 10. 练习题

### 10.1 基础题

1. One-Hot为什么不能用于类别有顺序的情况
2. 实现 One-Hot 到 Label Encoding 的转换

### 10.2 进阶题

3. 实现高维稀疏One-Hot的内存优化
4. 设计处理未知类别的策略

### 10.3 答案

<details>
<summary>答案1</summary>

因为One-Hot将每个类别视为完全独立的位置，不存在大小顺序关系。A=[1,0,0]和B=[0,1,0]之间的距离与A和C=[0,0,1]的距离相同，无法表示"第i类比第j类大"这样的关系。

</details>

<details>
<summary>答案2</summary>

```python
def one_hot_to_label(one_hot):
    return np.argmax(one_hot, axis=1)
```

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段

1. 理解原理
2. 实现基础One-Hot
3. 对比Label Encoding

### 11.2 第二阶段

1. 深度学习应用
2. 结合Embedding
3. 处理实际数据

### 11.3 第三阶段

1. 优化大规模编码
2. 特征工程
3. 项目实践

---

## 12. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_representation():
    """可视化One-Hot表示"""
    
    categories = ['A', 'B', 'C', 'D']
    encoder = OneHotEncoder()
    
    encoded = encoder.fit_transform(categories * 3)
    
    plt.figure(figsize=(10, 3))
    sns.heatmap(encoded, cmap='Blues')
    plt.title('One-Hot Representation')
    plt.show()
```

---

## 13. 模型评估

### 13.1 评估维度

| 维度 | 指标 |
|------|------|
| **可逆性** | 编码=解码 |
| **内存** | 稀疏度 |
| **效率** | 计算时间 |

### 13.2 代码

```python
def evaluate_encoding(categories, one_hot):
    metrics = {
        'accuracy': (decode_one_hot(one_hot) == categories).all(),
        'sparsity': (one_hot == 0).sum() / one_hot.size,
        'memory': one_hot.nbytes,
    }
    return metrics
```

---

## 14. 进阶内容

### 14.1 变体

| 变体 | 描述 |
|------|------|
| **Dummy Variable** | 处理多重共线性 |
| **Count Vectorizer** | 词频向量 |
| **Binary** | 二进制编码 |

### 14.2 统计编码

| 方法 | 描述 |
|------|------|
| **Target Encoding** | 使用目标变量 |
| **WOE** | Weight of Evidence |
| **Frequency** | 类别频率 |

### 14.3 深度学习进展

- One-Hot → Token Embedding
- 可学习的表示
- 分布式表示

---

**文档结束**

*参考：Sklearn.preprocessing.OneHotEncoder*