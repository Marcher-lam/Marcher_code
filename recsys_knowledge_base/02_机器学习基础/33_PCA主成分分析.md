# PCA主成分分析 学习文档

> 无监督降维的经典方法

---

## 1. 算法基础认知

**PCA（Principal Component Analysis）** 通过正交变换将高维数据投影到方差最大的方向，实现降维。

---

## 3. 数学公式

### 3.1 步骤

1. 数据中心化: $X' = X - \bar{X}$
2. 计算协方差矩阵: $C = \frac{1}{n}X'^T X'$
3. 对C做特征分解: $C = V\Lambda V^T$
4. 取前k个最大特征值对应的特征向量
5. 投影: $Z = X' V_k$

### 3.2 方差解释比

$$\text{explained\_ratio} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}$$

---

## 7. 调库实现

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(1000, 50)
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

print(f"原始维度: {X.shape[1]}")
print(f"降维后: {X_reduced.shape[1]}")
print(f"方差解释比: {pca.explained_variance_ratio_.sum():.4f}")
```

---

## 12. 学习总结

1. PCA = 找方差最大的方向做投影
2. 无监督降维，不依赖标签
3. 在推荐中：特征降维、可视化、去噪
4. 局限：线性方法，可能丢失非线性信息
