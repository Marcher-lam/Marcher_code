# KNN K近邻 学习文档

> 最简单的分类/回归算法——近朱者赤，近墨者黑

---

## 1. 算法基础认知

**KNN（K-Nearest Neighbors）** 是一种基于实例的学习算法：对于新样本，找到训练集中最近的K个样本，用它们的标签进行投票（分类）或平均（回归）。

---

## 3. 数学公式

### 距离度量

$$d(x, x') = \|x - x'\|_p$$

| 度量 | 公式 |
|------|------|
| 欧氏距离 | $\sqrt{\sum(x_i - x'_i)^2}$ |
| 曼哈顿距离 | $\sum|x_i - x'_i|$ |
| 余弦相似度 | $\frac{x \cdot x'}{\|x\|\|x'\|}$ |

### 分类决策

$$\hat{y} = \arg\max_c \sum_{i \in N_K(x)} \mathbb{1}[y_i = c]$$

### 回归决策

$$\hat{y} = \frac{1}{K}\sum_{i \in N_K(x)} y_i$$

---

## 7. 调库实现

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
print(f"KNN准确率: {accuracy_score(y_test, knn.predict(X_test)):.4f}")
```

---

## 8. 手工实现

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # 计算所有距离
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # 找最近的K个
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            # 多数投票
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(predictions)

# 测试
X = np.array([[1,1],[1,2],[2,2],[8,8],[9,8],[8,9]])
y = np.array([0,0,0,1,1,1])
knn = KNN(k=3)
knn.fit(X, y)
print(f"预测 [3,3]: {knn.predict(np.array([[3,3]]))}")  # 应该是0
```

---

## 12. 学习总结

1. KNN = "找最近的K个邻居，少数服从多数"
2. 简单但有效，无需训练（惰性学习）
3. 缺点：预测慢（需遍历所有样本）、维度灾难
4. 在推荐中：基于物品相似度的推荐、UserCF本质上是KNN
