# K近邻法 学习文档

## 1. 算法基础认知
基于实例的学习方法，预测时使用最近的K个邻居进行决策。

## 2. 核心原理
- 距离度量：欧氏距离、曼哈顿距离等
- 决策规则：分类用多数表决，回归用平均值

## 3. 数学公式
距离：$d(oldsymbol{x}, oldsymbol{x}') = \|oldsymbol{x} - oldsymbol{x}'\|$ 
k个最近邻的多数表决决定类别。

## 4. 训练过程
存储所有训练样本，预测时计算距离并排序。

## 5. 应用场景
- 小数据集分类
- 局部模式识别

## 6. 优缺点
优点：简单直观、无假设；缺点：计算量大、维度灾难

## 7. 代码实现
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## 8. 手工实现
```python
import numpy as np
from collections import Counter

class KNNScratch:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```
