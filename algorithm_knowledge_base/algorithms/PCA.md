
# PCA 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
PCA（Principal Component Analysis，主成分分析）是一种无监督降维算法，通过线性变换将数据投影到正交的新坐标系，使得投影后方差最大，从而用更少的维度保留最多的信息。

### 1.2 直觉类比
想象你在三维空间有很多点，你想找到一个方向（主成分）来观察这些点，使得点的分布"最展开"（方差最大）。找到第一个方向后，再找第二个方向（与第一个正交），使得在第一个方向投影后的"残余"方差最大，以此类推。

### 1.3 历史背景
PCA由Karl Pearson于1901年首次提出，Harold Hotelling于1933年进一步发展。PCA是最经典的降维方法，在统计学、机器学习、信号处理等领域广泛应用。

### 1.4 算法定位
- 类型：无监督学习
- 输出：降维后的特征向量
- 模型类别：非参数模型（线性降维）

### 1.5 前置知识
- 线性代数（协方差矩阵、特征值分解）
- 概率统计（均值、方差）
- Python 编程（NumPy、scikit-learn）

## 2. 核心原理
### 2.1 核心原理
PCA的核心思想是"最大化方差，最小化信息损失"——通过线性变换找到数据方差最大的投影方向，这些方向（主成分）构成新的正交坐标系。

### 2.2 工作流程
1. 数据中心化（减去均值）
2. 计算协方差矩阵
3. 对协方差矩阵进行特征值分解
4. 选择前k个最大特征值对应的特征向量
5. 将数据投影到这些特征向量上

### 2.3 关键概念解释
- **主成分**：数据方差最大的投影方向
- **协方差矩阵**：衡量各维度之间相关性的矩阵
- **特征向量**：协方差矩阵在变换后方向不变的向量
- **特征值**：数据在该方向上的方差

### 2.4 几何解释
PCA找到的是数据在特征空间中的主轴方向。第一主成分是数据分布最长轴的方向，第二主成分是与第一主成分正交且方差次大的方向，以此类推。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $X$ | 数据矩阵 $(n \times d)$ |
| $x_i$ | 第i个样本 $(1 \times d)$ |
| $\mu$ | 均值向量 $(1 \times d)$ |
| $\Sigma$ | 协方差矩阵 $(d \times d)$ |
| $w_i$ | 第i个主成分方向 |
| $\lambda_i$ | 第i个特征值 |

### 3.2 问题形式化
寻找正交矩阵 $W = [w_1, w_2, ..., w_d]$，使得数据投影到 $w_i$ 方向后的方差最大化：
$$\max_{w_i} \text{Var}(X w_i) \quad \text{s.t.} \quad w_i^T w_i = 1, w_i^T w_j = 0 (i \neq j)$$

### 3.3 目标函数
$$\max_W J(W) = \sum_{i=1}^{k} w_i^T \Sigma w_i \quad \text{s.t.} \quad W^T W = I$$

### 3.4 推导过程
**Step 1: 数据中心化**
$$\bar{x}_i = x_i - \mu$$

**Step 2: 计算协方差矩阵**
$$\Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T = \frac{1}{n-1} X_c^T X_c$$

**Step 3: 特征值分解**
求解特征值方程：
$$\Sigma w = \lambda w$$

**Step 4: 选择主成分**
选择前k个最大特征值对应的特征向量：
$$w_1, w_2, ..., w_k$$

**Step 5: 投影数据**
$$X_{new} = X W_k$$

### 3.5 最终解/算法步骤
1. 中心化：$X_c = X - \mu$
2. 协方差：$\Sigma = \frac{1}{n-1} X_c^T X_c$
3. 特征分解：$[\Lambda, W] = \text{eig}(\Sigma)$
4. 排序：按特征值降序排列
5. 投影：$X_{pca} = X_c W_k$

## 4. 训练过程讲解
### 4.1 数据预处理
- 数据中心化（必须）
- 数据标准化（可选，StandardScaler）
- 缺失值处理

### 4.2 参数初始化
- n_components：主成分数量
- svd_solver：'full', 'arpack', 'randomized'
- whitening：是否白化

### 4.3 迭代过程
PCA使用闭式解（特征值分解），无需迭代。使用随机SVD可以处理大规模数据。

### 4.4 收敛条件
特征值分解一次性完成，不涉及收敛。

### 4.5 超参数及推荐范围
- n_components: 1到min(n_samples, n_features)
- svd_solver: 'randomized'（大数据），'full'（小数据）
- whiten: False或True

## 5. 应用场景
### 5.1 典型应用
- **数据降维**：减少特征数量，加速后续算法
- **数据可视化**：将高维数据降到2-3维进行可视化
- **噪声过滤**：去除方差小的主成分
- **特征提取**：提取主要信息特征

### 5.2 适用数据特征
- 数据线性相关
- 特征维度较高
- 需要去除冗余特征

### 5.3 不适用场景
- 数据非线性结构
- 需要保留所有信息
- 类别信息重要

## 6. 优缺点分析
### 6.1 优点
- 计算效率高（闭式解）
- 可解释性强
- 无参数问题
- 去除数据冗余

### 6.2 缺点
- 假设线性关系
- 只能处理实数数据
- 可能丢失重要信息
- 对异常值敏感

### 6.3 与同类算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| PCA | 简单高效 | 线性假设 | 一般降维 |
| LDA | 有监督 | 最多c-1维 | 分类降维 |
| t-SNE | 非线性 | 计算慢 | 可视化 |
| UMAP | 保持结构 | 需调参 | 可视化 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler

# 1. 生成示例数据
X, y = make_blobs(n_samples=500, centers=3, n_features=5, 
                  cluster_std=1.5, random_state=42)

# 2. 数据标准化（PCA前通常需要标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA降维
pca = PCA(n_components=0.95)  # 保留95%方差
X_pca = pca.fit_transform(X_scaled)

print(f"原始维度: {X_scaled.shape[1]}")
print(f"降维后维度: {X_pca.shape[1]}")
print(f"累计解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

# 4. 可视化方差解释比例
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
        pca.explained_variance_ratio_)
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('各主成分解释方差')

plt.subplot(1, 3, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
plt.xlabel('主成分数')
plt.ylabel('累计解释方差')
plt.title('累计解释方差')
plt.legend()

plt.subplot(1, 3, 3)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
        pca.singular_values_)
plt.xlabel('主成分')
plt.ylabel('奇异值')
plt.title('奇异值分布')

plt.tight_layout()
plt.show()

# 5. 二维可视化
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('原始数据 (前2维)')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('PCA降维后')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')

plt.tight_layout()
plt.show()

# 6. 主成分载荷（各特征对主成分的贡献）
loadings = pca_2d.components_.T
plt.figure(figsize=(10, 5))
plt.bar(range(5), loadings[:, 0], alpha=0.7, label='PC1')
plt.bar(range(5), loadings[:, 1], alpha=0.7, label='PC2')
plt.xlabel('原始特征')
plt.ylabel('载荷')
plt.title('主成分载荷')
plt.legend()
plt.show()

# 7. 重构与降噪
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

# 添加噪声
X_noisy = X + np.random.normal(0, 0.5, X.shape)

# PCA降噪
pca_denoise = PCA(n_components=2)
X_denoised = pca_denoise.inverse_transform(pca_denoise.fit_transform(X_noisy))

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(X_noisy[:50, i], alpha=0.5, label='噪声')
    plt.plot(X[:50, i], linewidth=2, label='原始')
    plt.plot(X_denoised[:50, i], linestyle='--', label='去噪')
    plt.title(f'特征{i+1}')
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.show()
```

### 7.3 运行结果示例
```
原始维度: 5
降维后维度: 3
累计解释方差: [0.45 0.72 0.95]
```

## 8. 手工代码实现
### 8.1 核心算法手写
```python
import numpy as np

class PCAManual:
    """手工实现主成分分析(PCA)"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """训练PCA模型"""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # 1. 计算均值并中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. 计算协方差矩阵
        # 方法1：直接计算
        # cov = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # 方法2：使用SVD（更数值稳定）
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. 特征向量就是V的转置
        self.components_ = Vt.T
        
        # 4. 计算方差（奇异值的平方除以n-1）
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        
        # 5. 确定主成分数
        if self.n_components is None:
            self.n_components = n_features
        
        # 6. 截取前n_components个
        self.components_ = self.components_[:, :self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        
        # 7. 计算解释方差比例
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(self.explained_variance_)
        )
        
        return self
    
    def transform(self, X):
        """投影数据到主成分空间"""
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X):
        """训练并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca):
        """从PCA空间重构回原始空间"""
        return X_pca @ self.components_.T + self.mean_

# 测试手工实现
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    
    # 加载数据
    iris = load_iris()
    X = iris.data
    
    # 手工实现
    pca_manual = PCAManual(n_components=2)
    X_manual = pca_manual.fit_transform(X)
    
    # sklearn实现
    pca_sklearn = PCA(n_components=2)
    X_sklearn = pca_sklearn.fit_transform(X)
    
    print("=== PCA手工实现 vs sklearn ===")
    print(f"手工实现方差比例: {pca_manual.explained_variance_ratio_}")
    print(f"sklearn方差比例: {pca_sklearn.explained_variance_ratio_}")
    print(f"\n投影结果相关系数: {np.corrcoef(X_manual.flatten(), X_sklearn.flatten())[0,1]:.6f}")
```

### 8.2 与调库结果对比
| 指标 | 手工实现 | sklearn |
|------|----------|---------|
| 解释方差比 | 相同 | 相同 |
| 投影方向 | 相同 | 相同 |
| 数值精度 | 略低 | 优化过 |

## 9. 可视化与结果理解
### 9.1 方差解释可视化
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# 完整方差分析
pca_full = PCA().fit(X_scaled)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 6), pca_full.explained_variance_ratio_, alpha=0.7)
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('各主成分方差贡献')

plt.subplot(1, 2, 2)
plt.plot(range(1, 6), np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
plt.axhline(y=0.95, color='g', linestyle='--', label='95%')
plt.xlabel('主成分数')
plt.ylabel('累计解释方差')
plt.title('累计方差曲线')
plt.legend()
plt.tight_layout()
plt.show()
```

### 9.2 双标图（Biplot）
```python
# 双标图：同时显示样本和变量
def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, alpha=0.5)
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0]*3, coeff[i, 1]*3, color='r', alpha=0.7)
        if labels is not None:
            plt.text(coeff[i, 0]*3.1, coeff[i, 1]*3.1, labels[i], 
                    color='r', ha='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
biplot(X_pca, pca.components_.T, [f'Feature{i}' for i in range(5)])
```

### 9.3 结果解读
- 第一主成分捕获了数据最大方差方向
- 前几个主成分通常能捕获大部分方差
- 载荷图显示各原始特征对主成分的贡献

## 10. 模型评估
### 10.1 评估指标选择
- **解释方差比**：各主成分捕获的信息比例
- **累计解释方差**：选取足够的主成分数
- **重构误差**：降维后的信息损失

### 10.2 维度选择
```python
# 使用累计解释方差选择维度
for threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
    n_comp = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= threshold) + 1
    print(f"保留{threshold*100}%信息需要{n_comp}个主成分")
```

### 10.3 重构误差评估
```python
# 计算不同维度下的重构误差
for n_comp in range(1, 6):
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    error = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"n_components={n_comp}, 重构误差: {error:.4f}")
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 未进行中心化（导致结果错误）
- 未标准化（特征尺度不同）
- 异常值影响

### 11.2 模型层面常见错误
- 过度降维（丢失重要信息）
- 使用PCA作为分类器
- 忽视线性假设

### 11.3 调参层面常见误区
- 盲目追求高解释方差
- 不理解n_components参数含义
- 忽视后续任务需求

## 12. 学习总结
### 12.1 核心要点回顾
- PCA通过最大化方差找到数据的主轴方向
- 主成分是协方差矩阵的特征向量
- 降维后保留尽可能多的信息
- 需要数据中心化

### 12.2 关键公式汇总
- 协方差矩阵：$\Sigma = \frac{1}{n-1} X_c^T X_c$
- 特征值分解：$\Sigma w = \lambda w$
- 投影：$X_{new} = X_c W_k$

### 12.3 与前序/后续算法联系
- **前置算法**：数据预处理、标准化
- **后续算法**：LDA（有监督降维）、t-SNE（非线性可视化）

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 为什么PCA前需要进行数据中心化？
2. 解释主成分和特征向量的关系。
3. 如何选择合适的主成分数量？

### 13.2 进阶思考题
1. PCA和LDA的本质区别是什么？
2. 为什么PCA假设的是线性关系？

### 13.3 详细答案与解析
1. **答案**：数据中心化后，均值变为0，协方差矩阵计算更简单，且主成分方向不受数据平移影响。
2. **答案**：主成分方向就是协方差矩阵的特征向量，特征值表示该方向上的方差大小。
3. **答案**：可以使用累计解释方差（通常选择90%-95%）或交叉验证确定。

## 14. 学习路径建议建议
### 14.1 前置知识
- 线性代数基础
- 概率统计基础
- Python编程

### 14.2 平行算法
- LDA（监督降维）
- ICA（独立成分分析）
- SVD（矩阵分解）

### 14.3 进阶算法
- 核PCA（非线性）
- 稀疏PCA
- 增量PCA

### 14.4 推荐资源
- 《Pattern Recognition and Machine Learning》- Bishop
- 《Introduction to to Statistical Learning》
- scikit-learn PCA官方文档
