# One-Hot编码 学习文档

> 最基础的特征表示方法——理解向量化表示的起点

---

## 1. 算法基础认知

### 1.1 什么是One-Hot编码

**One-Hot编码** 将每个类别映射为一个只有一个位置为1、其余为0的向量。

```
词汇表: ["电影", "音乐", "体育", "科技"]

"电影" → [1, 0, 0, 0]
"音乐" → [0, 1, 0, 0]
"体育" → [0, 0, 1, 0]
"科技" → [0, 0, 0, 1]
```

### 1.2 为什么需要One-Hot

| 问题 | 说明 |
|------|------|
| **类别无法计算** | "电影"和"音乐"是文字，无法输入模型 |
| **数值编码有歧义** | 用1,2,3,4编码会引入虚假的大小关系 |
| **One-Hot无序** | 每个类别独占一个维度，无大小关系 |

### 1.3 在推荐系统中的应用

- 类别特征的基础编码（用户ID、物品ID、城市等）
- 实际工程中通常作为Embedding的输入

---

## 2. 核心原理

### 2.1 编码方式

对于N个类别的变量，One-Hot编码产生N维向量：

$$\text{OneHot}(c_i) = [0, 0, ..., 1, ..., 0, 0]$$

其中第 $i$ 个位置为1，其余为0。

### 2.2 独热矩阵

对于词汇表 $\{w_1, w_2, ..., w_V\}$：

$$\mathbf{I} = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \\ 0 & 0 & \cdots & 1 \end{pmatrix} \in \mathbb{R}^{V \times V}$$

每个词是单位矩阵的一行。

---

## 3. 数学公式

### 3.1 编码函数

$$\text{OneHot}(x, i) = \begin{cases} 1 & \text{if } x = c_i \\ 0 & \text{otherwise} \end{cases}$$

### 3.2 词向量

$$\mathbf{e}_i = [0, 0, ..., 1, ..., 0] \in \mathbb{R}^{V}$$

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 简单直观 | **维度灾难**：V个类别→V维向量 |
| 无偏（类别独立） | **稀疏**：只有1个非零元素 |
| 容易实现 | **无法表达语义关系** |
| 不引入虚假序关系 | 无法计算相似度 |

> One-Hot的核心问题：任意两个不同词的向量正交（余弦相似度=0），"猫"和"狗"的距离与"猫"和"汽车"一样远。

---

## 7. 调库实现

```python
"""
One-Hot 编码实现
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 方法1: 手动实现
def one_hot_encode(categories, all_categories):
    """手动One-Hot编码"""
    cat_to_idx = {c: i for i, c in enumerate(all_categories)}
    n = len(all_categories)
    result = np.zeros((len(categories), n))
    for i, c in enumerate(categories):
        result[i, cat_to_idx[c]] = 1
    return result

# 方法2: sklearn
encoder = OneHotEncoder(sparse_output=False)
cities = np.array([['北京'], ['上海'], ['北京'], ['广州']])
encoded = encoder.fit_transform(cities)
print(f"输入: {cities.flatten()}")
print(f"编码:\n{encoded}")
print(f"类别: {encoder.categories_[0]}")
```

---

## 12. 学习总结

1. One-Hot是最基础的类别编码方式
2. **维度灾难和稀疏性**是其致命缺点
3. 在推荐系统中通常作为Embedding层的前置输入
4. 理解One-Hot是理解Word2Vec、Embedding的基础

---

## 14. 学习路径

```
One-Hot → TF-IDF（加权表示）→ Word2Vec（稠密向量）→ Embedding
```
