# LDA 学习文档

## 1. 算法基础认知

线性判别分析（Linear Discriminant Analysis, LDA）是一种经典的有监督降维和分类方法，由 Ronald Fisher 于 1936 年提出。与 PCA 无监督最大化方差不同，LDA 的目标是**最大化类间差异、最小化类内差异**，即找到最能区分不同类别的投影方向。

## 2. 核心原理

LDA 基于 **Fisher 判别准则**：寻找投影方向 $w$，使得投影后不同类别尽可能分开，同一类别尽可能紧凑。

具体来说，LDA 优化目标为：

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

- $S_B$：类间散度矩阵（Between-class scatter）
- $S_W$：类内散度矩阵（Within-class scatter）
- $J(w)$ 越大，表示类间分离越好、类内聚合越好

## 3. 数学公式与推导

**类内散度矩阵**：

$$S_W = \sum_{c=1}^{C} S_c = \sum_{c=1}^{C} \sum_{x \in \text{class } c} (x - \mu_c)(x - \mu_c)^T$$

**类间散度矩阵**：

$$S_B = \sum_{c=1}^{C} n_c (\mu_c - \mu)(\mu_c - \mu)^T$$

其中 $\mu_c$ 是第 $c$ 类的均值向量，$\mu$ 是全局均值，$n_c$ 是第 $c$ 类的样本数。

**广义特征值问题**：

对 $J(w)$ 求极值，等价于求解广义特征值问题：

$$S_B w = \lambda S_W w$$

即 $S_W^{-1} S_B w = \lambda w$，对 $S_W^{-1} S_B$ 做特征分解，取前 $k$ 个最大特征值对应的特征向量。

**降维维度上界**：LDA 最多降到 $C - 1$ 维（$C$ 为类别数），因为 $S_B$ 的秩至多为 $C-1$。

## 4. 训练过程讲解

1. 计算每个类别的均值向量 $\mu_c$
2. 计算全局均值 $\mu$
3. 构造类内散度矩阵 $S_W$ 和类间散度矩阵 $S_B$
4. 求解 $S_W^{-1} S_B$ 的特征值和特征向量
5. 按特征值降序排列，取前 $k$ 个特征向量组成投影矩阵 $W$
6. 将数据投影：$Z = XW$

## 5. 应用场景

- 人脸识别（Fisherfaces，与 Eigenfaces 对应）
- 文本分类中的降维
- 广告点击率预估中的特征压缩
- 医学诊断中的多指标判别
- 适合类别信息已知的降维场景

## 6. 优缺点分析

**优点**：
- 利用了类别标签信息，降维后分类效果通常优于 PCA
- 有闭式解，计算高效
- 降维后的特征具有判别性

**缺点**：
- 最多降到 $C-1$ 维，维度受限于类别数
- 假设各类别服从相同协方差的高斯分布
- 当 $S_W$ 奇异（特征维度 > 样本数）时需正则化
- 对非线性分布效果差

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

print(f"Original: {X_train.shape} -> Reduced: {X_lda.shape}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

y_pred = lda.predict(X_test)
print(f"Classification accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class LDAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        classes = np.unique(y)
        n_features = X.shape[1]
        mean_overall = X.mean(axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in classes:
            X_c = X[y == c]
            mean_c = X_c.mean(axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            diff = (mean_c - mean_overall).reshape(-1, 1)
            S_B += len(X_c) * diff @ diff.T
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(S_W) @ S_B)
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]].T
        self.explained_variance_ratio_ = eigenvalues[idx[:self.n_components]] / eigenvalues[idx].sum()
        return self

    def transform(self, X):
        return X @ self.components.T

lda_s = LDAScratch(n_components=2)
lda_s.fit(X_train, y_train)
X_lda_s = lda_s.transform(X_train)
print("Manual LDA shape:", X_lda_s.shape)
print("Variance ratio:", lda_s.explained_variance_ratio_)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for c in np.unique(y_train):
    mask = y_train == c
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {c}', alpha=0.7)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection (Iris Dataset)')
plt.legend()
plt.tight_layout()
plt.show()
```

与 PCA 对比观察：LDA 降维后不同类别通常分得更开，因为 LDA 利用了类别标签。

## 10. 模型评估

- **分类准确率**：LDA 本身也是分类器，可直接评估
- **类间/类内散度比**：$J(w) = w^T S_B w / w^T S_W w$，越大越好
- **降维可视化**：观察投影后类别是否可分
- **与 PCA 对比**：在同一数据集上对比两种降维方法的分类效果

## 11. 常见问题与易错点

- **维度限制**：$K$ 类问题最多降到 $K-1$ 维，无法降到更高维度
- **$S_W$ 奇异问题**：高维小样本时 $S_W$ 不可逆，需加正则项（sklearn 中 `shrinkage` 参数）
- **与 PCA 混淆**：PCA 无监督（最大化方差），LDA 有监督（最大化类间/类内比）
- **与主题模型 LDA 混淆**：Latent Dirichlet Allocation 也缩写为 LDA，是完全不同的算法

## 12. 学习总结

LDA 是有监督降维的代表算法。它的 Fisher 判别准则思想影响深远，从经典的线性分类器到现代深度学习中的 center loss 都有体现。理解 LDA 的关键在于理解"最大化类间差异、最小化类内差异"这一优化目标。

## 13. 练习题与思考题（含答案）

**Q1**：三分类问题，原始特征维度为 10，LDA 最多能降到几维？为什么？

> 答：2 维。因为 $S_B$ 的秩至多为 $C - 1 = 2$，最多只有 2 个非零特征值对应的有效投影方向。

**Q2**：PCA 和 LDA 的根本区别是什么？

> 答：PCA 是无监督的，目标是最小化重构误差（等价于最大化投影方差）；LDA 是有监督的，目标是最大化类间散度与类内散度的比值。

**Q3**：什么时候应该用 PCA 而非 LDA？

> 答：当没有标签信息时只能用 PCA；当类别数很多但每类样本很少时，LDA 的参数估计不可靠，PCA 可能更稳定。

## 14. 学习路径建议

- **前置知识**：协方差矩阵、特征分解、PCA
- **下一步学习**：Kernel LDA（非线性扩展）、QDA（二次判别分析）
- **进阶方向**：Fisher 判别在深度学习中的应用（center loss、contrastive loss）
