# KNN 学习文档

## 1. 算法基础认知

K近邻算法（K-Nearest Neighbors, KNN）是一种基本的"懒惰学习"（lazy learning）算法。它没有显式的训练过程，而是在预测时直接计算待预测样本与所有训练样本的距离，找到最近的 $k$ 个邻居，通过投票（分类）或取均值（回归）来决定输出。

KNN 体现了"近朱者赤，近墨者黑"的朴素思想。

## 2. 核心原理

**分类决策规则**：多数投票法——$k$ 个邻居中出现次数最多的类别作为预测结果。

**回归决策规则**：取 $k$ 个邻居目标值的均值（或加权均值）。

**距离度量**：KNN 的核心是距离计算，常用距离包括：
- 欧氏距离（最常用）
- 曼哈顿距离
- 闵可夫斯基距离

KNN 是一种非参数模型，不对数据分布做假设。

## 3. 数学公式与推导

**欧氏距离：**

$$d(x_i, x_j) = \sqrt{\sum_{k=1}^{n}(x_{ik} - x_{jk})^2}$$

**曼哈顿距离：**

$$d(x_i, x_j) = \sum_{k=1}^{n}|x_{ik} - x_{jk}|$$

**闵可夫斯基距离（广义形式）：**

$$d(x_i, x_j) = \left(\sum_{k=1}^{n}|x_{ik} - x_{jk}|^p\right)^{1/p}$$

**分类预测（多数投票）：**

$$\hat{y} = \arg\max_{c}\sum_{i \in N_k(x)} I(y_i = c)$$

**回归预测（加权平均）：**

$$\hat{y} = \frac{\sum_{i \in N_k(x)} w_i \cdot y_i}{\sum_{i \in N_k(x)} w_i}, \quad w_i = \frac{1}{d(x, x_i)}$$

## 4. 训练过程讲解

KNN **没有显式训练过程**（因此称为懒惰学习），"训练"只是存储训练数据。

**预测过程**：
1. 计算测试样本与所有训练样本的距离
2. 按距离排序，选取前 $k$ 个最近邻
3. 分类：统计 $k$ 个邻居的类别，多数投票
4. 回归：取 $k$ 个邻居目标值的均值

为加速搜索，通常使用 KD-Tree 或 Ball Tree 数据结构。

## 5. 应用场景

- 推荐系统（基于相似用户/物品推荐）
- 手写数字识别（低维特征）
- 文本分类（结合 TF-IDF 特征）
- 广告系统中的相似用户匹配（look-alike）
- 缺失值填补（用最近邻的均值填充）

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 简单直观，无需训练 | 预测速度慢（需计算所有距离） |
| 无参数假设，适用面广 | 对高维数据效果差（维度灾难） |
| 天然支持多分类 | 对特征尺度敏感 |
| 可用于分类和回归 | 需要大量存储空间 |
| 对异常值不敏感（k较大时） | $k$ 值选择对结果影响大 |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append((k, scores.mean()))

best_k = max(k_scores, key=lambda x: x[1])[0]
print(f"最优k值: {best_k}")

model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from collections import Counter

class KNNManual:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.distance = distance

    def _compute_distance(self, x1, x2):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(x1 - x2))

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

k_values = range(1, 31)
train_acc = []
test_acc = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_acc.append(knn.score(X_train_scaled, y_train))
    test_acc.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_values, train_acc, 'b-', label='训练准确率')
plt.plot(k_values, test_acc, 'r-', label='测试准确率')
plt.xlabel('k值')
plt.ylabel('准确率')
plt.title('KNN: k值对模型性能的影响')
plt.legend()
plt.tight_layout()
plt.savefig("knn_k_selection.png", dpi=150)
plt.show()
```

- $k=1$ 时训练准确率为 100%（过拟合），测试准确率波动大
- $k$ 过大时模型过于平滑（欠拟合）
- 通常选择测试准确率最高处的 $k$ 值

## 10. 模型评估

- **准确率**：分类任务的主要指标
- **交叉验证**：KNN 对数据划分敏感，需交叉验证选择 $k$
- **混淆矩阵**：查看各类别的分类情况
- **运行时间**：评估搜索效率（考虑 KD-Tree 加速）

## 11. 常见问题与易错点

- **特征未标准化**：不同尺度特征对距离影响不均，必须标准化
- **k 值选择不当**：$k$ 太小（过拟合，噪声敏感）或太大（欠拟合，边界模糊）
- **维度灾难**：高维空间中所有点对之间的距离趋于相同，KNN 失效
- **数据不平衡**：多数类样本会主导投票结果，可使用距离加权
- **k 为偶数**：可能产生平票，sklearn 会选择较小标签

## 12. 学习总结

KNN 是最直观的机器学习算法——"看邻居怎么投票就怎么决定"。它没有训练过程，但预测时需要计算与所有训练样本的距离。关键要点：标准化特征、通过交叉验证选择 $k$、注意高维数据下的维度灾难问题。

## 13. 练习题与思考题（含答案）

**Q1**: 为什么 KNN 预测时计算量大？

> A: 每次预测都需要计算测试样本与所有 $m$ 个训练样本的距离，时间复杂度 $O(mn)$（$n$ 为特征维度）。可使用 KD-Tree（$O(\log m)$）或 Ball Tree 加速。

**Q2**: $k=1$ 和 $k=N$（样本总数）时，KNN 分别退化为什么？

> A: $k=1$ 时模型记住训练数据，训练集准确率 100%，极易过拟合；$k=N$ 时预测结果始终为多数类，退化为将所有样本预测为训练集中最多的类别。

**Q3**: 什么是维度灾难？为什么它影响 KNN？

> A: 在高维空间中，所有点对之间的距离趋于相同（最大距离与最小距离之比趋近 1），"最近邻"不再有实际意义，导致 KNN 分类效果急剧下降。一般建议特征维度不超过 20-30。

## 14. 学习路径建议

```
KNN → k-D tree（加速搜索）→ 距离度量学习 → 局部敏感哈希(LSH) → 向量近似搜索
```

理解 KNN 后，学习 KD-Tree 来加速最近邻搜索，再深入了解近似最近邻算法（如 LSH、HNSW）在大规模数据中的应用。
