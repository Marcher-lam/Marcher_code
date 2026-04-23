# PCA 学习文档

## 1. 算法基础认知

主成分分析（Principal Component Analysis, PCA）是最经典的无监督降维方法，由 Karl Pearson 于 1901 年提出。其核心目标是在保留尽可能多信息的前提下，将高维数据投影到低维空间。PCA 通过线性变换找到数据方差最大的方向，实现特征压缩和去相关。

## 2. 核心原理

PCA 的核心思想：**寻找一组正交基，使得数据在这组基上的投影方差依次最大化**。

直觉理解：数据在某个方向上的方差越大，说明该方向携带的信息越多。PCA 依次找到方差最大的方向（第一主成分）、与之正交且方差次大的方向（第二主成分），依此类推。

等价表述：
- **最大方差视角**：最大化投影方差
- **最小重构误差视角**：最小化从低维重构原数据的误差
- **SVD 视角**：对中心化矩阵做 SVD

## 3. 数学公式与推导

**步骤一：中心化**

$$\tilde{X} = X - \bar{X}, \quad \bar{X} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**步骤二：协方差矩阵**

$$C = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$

**步骤三：特征分解**

$$C = V \Lambda V^T$$

其中 $V$ 的列向量是特征向量（主成分方向），$\Lambda$ 的对角元素 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$ 是特征值。

**步骤四：投影**

选择前 $k$ 个特征向量组成 $W = [v_1, v_2, \ldots, v_k]$，则降维结果为：

$$Z = \tilde{X} W \in \mathbb{R}^{n \times k}$$

**解释方差比**：第 $j$ 个主成分的解释方差比为 $\lambda_j / \sum_{i=1}^d \lambda_i$。

## 4. 训练过程讲解

1. 对数据按特征列做中心化（减去均值）
2. 计算协方差矩阵 $C$
3. 对 $C$ 做特征分解（或对中心化矩阵做 SVD，更稳定）
4. 按特征值降序排列，选择前 $k$ 个特征向量
5. 将原始数据投影到选定的特征向量上

选择 $k$ 的常用方法：
- 累积解释方差比达到 85%-95%
- 肘部法则（观察特征值下降的拐点）

## 5. 应用场景

- 高维数据可视化（降至 2D/3D）
- 图像压缩与人脸识别（Eigenfaces）
- 特征去相关与降噪
- 广告 CTR 预估中的特征降维
- 金融风控中的多指标综合

## 6. 优缺点分析

**优点**：
- 无监督方法，不需要标签
- 降维后特征线性无关
- 计算高效，有闭式解
- 降噪效果显著

**缺点**：
- 只能捕捉线性关系
- 主成分的解释性可能较差
- 对数据的尺度敏感，需要先标准化
- 异常值会严重影响结果

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

data = load_wine()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Original: {X.shape} -> Reduced: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

pca_s = PCAScratch(n_components=2)
X_pca_s = pca_s.fit_transform(X_scaled)
print("Manual PCA explained variance ratio:", pca_s.explained_variance_ratio_)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {label}', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA Projection (Wine Dataset)')
plt.legend()
plt.tight_layout()
plt.show()
```

## 10. 模型评估

- **累积解释方差比**：$\sum_{j=1}^{k} \lambda_j / \sum_{i=1}^d \lambda_i$，衡量保留了多少信息
- **重构误差**：$\|X - \hat{X}\|_F / \|X\|_F$
- **下游任务性能**：降维后在分类器上的准确率

## 11. 常见问题与易错点

- **忘记标准化**：PCA 对特征尺度敏感，不同量纲的特征必须先做 StandardScaler
- **混淆 PCA 与 LDA**：PCA 是无监督（最大化方差），LDA 是有监督（最大化类间差异）
- **PCA 不适合非线性数据**：对流形结构数据，应考虑 Kernel PCA、t-SNE、UMAP
- **中心化 vs 标准化**：PCA 严格要求中心化；标准化是可选的但推荐

## 12. 学习总结

PCA 是降维的基石算法。理解 PCA 的三个等价视角（最大方差、最小重构误差、SVD）是深入理解降维理论的关键。PCA 也为后续学习 Kernel PCA、增量 PCA、稀疏 PCA 等变体奠定基础。

## 13. 练习题与思考题（含答案）

**Q1**：PCA 中，第一主成分对应协方差矩阵的什么？

> 答：协方差矩阵的最大特征值对应的特征向量方向。

**Q2**：数据有 100 个特征，PCA 降到 10 维后，累积解释方差比为 70%，是否合适？

> 答：取决于具体场景。如果 70% 已足够（如可视化目的），则可以接受。如果下游任务需要更高保真度，应增大保留维度。通常建议 85%-95%。

**Q3**：PCA 为什么能降噪？

> 答：噪声通常分布在方差较小的方向上（对应较小的特征值），PCA 丢弃这些方向就相当于过滤了噪声。

## 14. 学习路径建议

- **前置知识**：协方差矩阵、特征分解、SVD
- **下一步学习**：LDA、Kernel PCA、t-SNE、UMAP
- **进阶方向**：增量 PCA（在线学习）、稀疏 PCA、PCA 在图像处理中的应用
