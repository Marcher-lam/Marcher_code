# K近邻(KNN) 学习文档

> 用最近的K个邻居投票或平均进行预测，最直观的非参数学习方法。

> 来源线索：本节内容根据原书中关于"Nonparametric Models"的相关章节(Ch 3.10.1)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：KNN通过找训练集中距离最近的K个样本，用它们的标签（分类）或数值（回归）进行预测。

**直觉类比**：搬到一个新城市，想判断这个区域是否安全。你问最近的3个邻居他们对安全的看法，少数服从多数。这就是KNN——"近朱者赤，近墨者黑"。

**历史背景**：KNN由Fix & Hodges (1951)提出，是最早的模式识别方法之一。原书Ch 3.10.1中作为非参数回归/分类方法的代表引入。

**算法定位**：非参数学习/惰性学习。不训练模型，预测时才计算。

**前置知识**：距离度量、概率基础。

## 2. 核心原理

**核心思想**：相似的输入应该有相似的输出。对于新样本$x$，在训练集中找K个最近邻，用它们的响应值$y$的均值（回归）或多数投票（分类）作为预测。

**工作流程**：
1. 存储全部训练数据
2. 对新样本$x$，计算到所有训练样本的距离
3. 选取距离最小的K个邻居
4. 回归：$\hat{y} = \frac{1}{K}\sum_{i \in N_K(x)} y_i$；分类：多数投票

**关键概念**：
- **距离度量**：欧氏距离$d(x,x') = \|x-x'\|_2$，曼哈顿距离，闵可夫斯基距离
- **K值选择**：K小→复杂边界（过拟合），K大→平滑（欠拟合）
- **惰性学习**：无训练过程，所有计算在预测时进行

## 3. 数学公式与推导

### KNN回归

$$\hat{f}(x) = \frac{1}{K}\sum_{i \in N_K(x)} y_i$$

其中$N_K(x)$是$x$的K个最近邻的索引集。

### KNN分类

$$\hat{y} = \arg\max_c \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$

### 偏差-方差分析

- $K=1$：偏差低、方差高（过拟合）
- $K=N$：偏差高、方差低（欠拟合，退化为全局均值）
- 最优K在中间，通常通过交叉验证选择

### 理论性质

当$N \to \infty, K \to \infty, K/N \to 0$时，KNN回归是贝叶斯最优的一致估计。

## 4-6. 简要

### 应用
1. 推荐系统（相似用户推荐）
2. 原书中的非参数值函数估计
3. 缺失值填补

### 优缺点
**优点**：直觉简单、无需训练、天然多分类
**缺点**：预测慢$O(Nd)$、高维距离失效（维度灾难）、需要特征标准化

## 7-8. 核心实现

```python
"""KNN：手工实现"""
import numpy as np
from collections import Counter

class KNN:
    """K近邻算法"""
    def __init__(self, k=5, task='regression'):
        self.k = k
        self.task = task

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            dists = self._euclidean(self.X_train, x.reshape(1, -1).repeat(len(self.X_train), axis=0))
            k_idx = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx]
            if self.task == 'regression':
                predictions.append(np.mean(k_labels))
            else:
                counter = Counter(k_labels)
                predictions.append(counter.most_common(1)[0][0])
        return np.array(predictions)

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y_reg = np.sin(X[:, 0]) + X[:, 1]**2 + 0.1*np.random.randn(n)
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(int)

    # 回归
    knn_r = KNN(k=5, task='regression').fit(X, y_reg)
    y_pred = knn_r.predict(X[:20])
    mse = np.mean((y_pred - y_reg[:20])**2)
    print(f"KNN回归 MSE: {mse:.4f}")

    # 分类
    knn_c = KNN(k=5, task='classification').fit(X, y_cls)
    y_pred_c = knn_c.predict(X[:20])
    acc = np.mean(y_pred_c == y_cls[:20])
    print(f"KNN分类 准确率: {acc:.2f}")
```

## 9-14. 简要

### 12. 学习总结
KNN：$\hat{y}(x) = \frac{1}{K}\sum_{i \in N_K(x)} y_i$。非参数惰性学习，K控制偏差-方差权衡。

### 13. 练习题
**Q1**：KNN在高维空间中为什么效果差？
**A1**：维度灾难——高维空间中所有点对的距离趋于相同，"最近邻"不再有意义。距离比$\max d / \min d \to 1$当$d \to \infty$。

### 14. 学习路径
**前置**：距离度量 | **进阶**：核回归、局部多项式回归、近似最近邻(ANN)
**资源**：原书Ch 3.10.1、Hastie et al. "ESL" Ch 13.3
