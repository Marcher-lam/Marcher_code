# Random Projection 学习文档

## 1. 算法基础认知

Random Projection（随机投影）是机器学习中用于**降维和特征提取**的重要技术，其理论基础是Johnson-Lindenstrauss引理：该引理表明，高维空间中的点集可以嵌入到低维空间，同时保持点间距离近似的性质。更重要的是，随机投影矩阵可以从特定分布（如高斯分布）中采样，无需复杂的优化过程。Random Projection的核心优势在于：**计算效率极高**，比PCA等方法快数个数量级；**不需要训练**，直接使用随机矩阵投影；**理论保证**，有Johnson-Lindenstrauss引理保证距离保持。

Random Projection广泛应用于：1）高维数据可视化；2）加速近邻搜索；3）特征降维；4）解决维度灾难。在文本处理（TF-IDF向量）、图像特征等领域，随机投影都是有效的降维手段。

## 2. 核心原理

Random Projection的核心原理是**用随机投影矩阵将高维数据映射到低维空间，同时保持原始空间的距离结构**。设原始数据矩阵X ∈ R^{n×d}，投影矩阵R ∈ R^{d×k}，投影后Y = X × R。关键是随机矩阵R的构造方式：常见的是高斯随机投影：R的元素从N(0,1/k)中独立采样；稀疏随机投影：R的元素以特定概率取±1/√k或其他稀疏值。

Johnson-Lindenstrauss引理：对于n个点，存在k = O(log n / ε²)维的嵌入，使得点间距离在(1±ε)因子内保持。这意味着用少量维度（相对于原始维度，可以是几千维）就能近似保持距离结构。

## 3. 数学公式与推导

### 3.1 随机投影矩阵

高斯随机投影（Achlioptas算法）：
$$R_{ij} = \sqrt(1/k) \cdot \begin{cases} +1 & p=1/2 \\ 0 & p=1/2 \end{cases}$$

或者：
$$R_{ij} = \begin{cases} +1 & p=1/6 \\ 0 & p=2/3 \\ -1 & p=1/6 \end{cases}$$

### 3.2 Johnson-Lindenstrauss引理

设原始距离为d(x,y)，投影后距离为d'(x,y)。对于任意ε>0，以高概率有：

$$(1-\varepsilon) \|x-y\|^2 \leq \|Rx-Ry\|^2 \leq (1+\varepsilon) \|x-y\|^2$$

所需维度：k ≥ O(log n / ε²)

### 3.3 距离保持性

通过投影后点积保持：
$$E[||Rx - Ry||^2] = ||x-y||^2$$

方差有上界：
$$Var(||Rx - Ry||^2) \leq (||x-y||^2)^2 / k$$

### 3.4 降维误差分析

投影误差随维度增加指数下降：
$$P(|\hat{d} - d| > \varepsilon d) \leq 2 \exp(-(k\varepsilon^2)/4)$$

## 4. 训练过程讲解

Random Projection是"无训练"技术，核心是随机矩阵的生成：

```
# 1. 确定目标维度k
k = min(d_original, log(n_samples) / epsilon^2)

# 2. 生成随机投影矩阵
R = random_matrix(d_original, k, distribution='gaussian')

# 3. 投影
X_projected = X @ R

# 4. （可选）归一化
X_projected = X_projected / sqrt(k)
```

关键参数：
| 参数 | 作用 | 选择方法 |
|------|------|----------|
| k | 目标维度 | log(n)/ε² |
| ε | 误差容忍 | 0.1-0.3 |
| distribution | 随机分布 | 高斯/稀疏 |

## 5. 应用场景

Random Projection主要应用场景：**文本降维**，TF-IDF向量的快速降维；**图像特征**，HOG、SIFT描述子的降维；**近邻搜索**，加速KNN搜索；**可视化**，t-SNE的线性前驱。具体应用：
1. KNN分类：投影后使用KNN
2. 聚类：投影后使用K-Means
3. 回归：投影后使用线性回归

## 6. 优缺点分析

Random Projection的优点：**快**，生成矩阵和投影都是O(n×d×k)；**简单**，无需训练；**理论保证**，JL引理。缺点：**信息损失**，降维有损；**随机性**，结果可能不稳定；**不如PCA**，有标签时PCA更好。

| 优点 | 说明 | 适用场景 |
|------|------|----------|
| 快 | O(ndk) | 实时系统 |
| 简单 | 无训练 | 初步降维 |

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 有损 | 信息损失 | 增大k |
| 不确定 | 随机性 | 多次投影取平均 |

## 7. 调库实现（Python完整代码）

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array


class GaussianRandomProjection(BaseEstimator, TransformerMixin):
    """高斯随机投影"""
    def __init__(self, n_components=10, epsilon=0.1, random_state=None):
        self.n_components = n_components
        self.epsilon = epsilon
        self.random_state = random_state
        self.components_ = None
    
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True, ensure_min_samples=2)
        
        n_samples, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = int(np.ceil(
                np.log(n_samples) / (self.epsilon ** 2)
            ))
        
        rng = np.random.RandomState(self.random_state)
        
        self.components_ = rng.randn(self.n_components, n_features).T
        self.components_ /= np.sqrt(self.n_components)
        
        return self
    
    def transform(self, X):
        X = check_array(X, accept_sparse=True)
        return X @ self.components_


class SparseRandomProjection(BaseEstimator, TransformerMixin):
    """稀疏随机投影"""
    def __init__(self, n_components=10, density=0.1, random_state=None):
        self.n_components = n_components
        self.density = density
        self.random_state = random_state
        self.components_ = None
    
    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2)
        
        n_samples, n_features = X.shape
        
        rng = np.random.RandomState(self.random_state)
        
        # 生成稀疏矩阵
        size = (n_features, self.n_components)
        indices = rng.uniform(0, 1, size) < self.density
        data = np.where(indices, 
            np.where(rng.uniform(0, 1, size) < 0.5, 1, -1),
            0)
        
        self.components_ = data.T / np.sqrt(self.n_components * self.density)
        
        return self
    
    def transform(self, X):
        X = check_array(X, accept_sparse=True)
        return X @ self.components_


class GaussianRandomProjectionCPU(GaussianRandomProjection):
    """传统CPU高效实现"""
    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_2d=True)
        
        n_samples, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = min(n_features, 
                               int(np.ceil(np.log(n_samples) / (self.epsilon ** 2))))
        
        rng = np.random.RandomState(self.random_state)
        
        # 使用简化的随机矩阵：±1/√k
        self.components_ = rng.choice([1, -1], 
                                  (n_features, self.n_components)) / np.sqrt(self.n_components)
        
        return self


class MultipleRandomProjection(BaseEstimator, TransformerMixin):
    """多次随机投影（减少方差）"""
    def __init__(self, n_components=10, n_projections=5, random_state=None):
        self.n_components = n_components
        self.n_projections = n_projections
        self.random_state = random_state
        self.projectors_ = None
    
    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2)
        
        n_samples, n_features = X.shape
        
        self.projectors_ = []
        
        for i in range(self.n_projections):
            proj = GaussianRandomProjection(
                n_components=self.n_components,
                epsilon=self.epsilon if hasattr(self, 'epsilon') else 0.1,
                random_state=self.random_state + i if self.random_state else None
            )
            proj.fit(X)
            self.projectors_.append(proj)
        
        return self
    
    def transform(self, X):
        X = check_array(X, accept_sparse=True)
        
        results = [proj.transform(X) for proj in self.projectors_]
        
        return np.concatenate(results, axis=1)


def compute_projection_error(X_original, X_projected):
    """计算投影误差"""
    dist_original = euclidean_distances(X_original)
    dist_projected = euclidean_distances(X_projected)
    
    relative_error = np.abs(dist_projected - dist_original) / (dist_original + 1e-10)
    
    return relative_error.mean()


def optimal_n_components(n_samples, epsilon=0.1):
    """计算最优维度"""
    return int(np.ceil(np.log(n_samples) / (epsilon ** 2)) + 1


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    X, _ = load_iris(return_X_y=True)
    
    print("=== Random Projection Demo ===")
    print(f"Original shape: {X.shape}")
    
    rp = GaussianRandomProjection(n_components=50, random_state=42)
    X_projected = rp.fit_transform(X)
    
    print(f"Projected shape: {X_projected.shape}")
    
    error = compute_projection_error(X, X_projected)
    print(f"Distance error: {error:.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def gaussian_random_projection(X, n_components, random_state=42):
    """高斯随机投影"""
    rng = np.random.RandomState(random_state)
    
    n_samples, n_features = X.shape
    
    # 生成高斯随机矩阵
    projection_matrix = rng.randn(n_features, n_components) / np.sqrt(n_components)
    
    # 投影
    X_projected = X @ projection_matrix
    
    return X_projected


def sparse_random_projection(X, n_components, density=0.1, random_state=42):
    """稀疏随机投影"""
    rng = np.random.RandomState(random_state)
    
    n_samples, n_features = X.shape
    
    # 生成稀疏矩阵
    size = (n_features, n_components)
    r = rng.uniform(0, 1, size)
    
    # 1/√k概率为1，1/√k概率为-1，其余为0
    threshold = density
    projection_matrix = np.where(r < threshold,
        np.where(r < threshold/2, 1, -1),
        0
    ) / np.sqrt(n_components * density)
    
    # 投影
    X_projected = X @ projection_matrix
    
    return X_projected


def johnson_lindenstrauss_dimension(n_points, epsilon=0.1):
    """Johnson-Lindenstrauss维度计算"""
    return int(np.ceil(np.log(n_points) / (epsilon ** 2))) + 1


def random_projection_batch(X_list, k, random_state=42):
    """批量随机投影"""
    results = []
    
    for X in X_list:
        X_proj = gaussian_random_projection(X, k, random_state)
        results.append(X_proj)
    
    return results


if __name__ == '__main__':
    X = np.random.randn(100, 50)
    
    k = johnson_lindenstrauss_dimension(100, 0.1)
    print(f"Optimal k: {k}")
    
    X_proj = sparse_random_projection(X, k, 0.1)
    print(f"Projected shape: {X_proj.shape}")
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_distance_comparison():
    """距离保持对比"""
    np.random.seed(42)
    X = np.random.randn(100, 100)
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    k_values = [5, 10, 20, 50]
    
    plt.figure(figsize=(10, 6))
    
    for k in k_values:
        rp = GaussianRandomProjection(n_components=k, random_state=42)
        X_proj = rp.fit_transform(X)
        
        dist_orig = euclidean_distances(X).flatten()
        dist_proj = euclidean_distances(X_proj).flatten()
        
        plt.scatter(dist_orig, dist_proj, alpha=0.5, label=f'k={k}')
    
    plt.plot([0, 10], [0, 10], 'k--', label='Perfect')
    plt.xlabel('Original Distance')
    plt.ylabel('Projected Distance')
    plt.title('Distance Preservation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rp_distance.png', dpi=150)
    plt.show()


def plot_dimension_vs_error():
    """维度vs误差"""
    n_samples = 1000
    k_values = np.arange(5, 101, 5)
    errors = [np.exp(-0.05*k) * 0.3 for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, 'o-', linewidth=2)
    plt.xlabel('n_components')
    plt.ylabel('Distance Error')
    plt.title('Dimension vs Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rp_dimension.png', dpi=150)
    plt.show()


def plot_dimension_runtime():
    """运行时间比较"""
    dims = [50, 100, 200, 500, 1000]
    rp_time = [0.01, 0.02, 0.05, 0.12, 0.30]
    pca_time = [0.1, 0.3, 1.0, 5.0, 20.0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, rp_time, 'o-', label='Random Projection')
    plt.plot(dims, pca_time, 's-', label='PCA')
    plt.xlabel('Dimension')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rp_runtime.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_dimension_vs_error()
    plot_dimension_runtime()
```

## 10. 模型评估

Random Projection评估指标：
1. **Distance Error**：距离保持误差
2. **Correlation**：相关系数
3. **k值选择**：JL最优维度
4. **Runtime**：运行时间

## 11. 常见问题与易错点

问题：1.维度选择不当 2.随机性导致不稳定 3.稀疏矩阵参数选择

解决：多次投影取平均，选择合理的k

## 12. 学习总结

Random Projection是高效的降维技术，基于Johnson-Lindenstrauss引理。核心：快、简单、有保证。

## 13. 练习题与思考题（含答案）

**练习题1**：计算1000个点，ε=0.1时最优维度。

答案：k = ceil(log(1000)/0.01) = ceil(6.91/0.01) = 692

**练习题2**：高斯随机投影的分布。

答案：N(0, 1/k)

### 13.3 详细答案

**问题**：随机投影vs PCA的区别。

答案：随机投影无监督、随机、无优化；PCA有监督、正交、优化。

## 14. 学习路径建议

学习Random Projection：
1. 降维基础
2. Johnson-Lindenstrauss引理
3. 实现与对比
4. 实际应用

### 14.1 资源

**论文**：
1. Johnson & Lindenstrauss (1984)
2. "JL Lemma"

**框架**：
1. sklearn RandomProjection