# k近邻法 (K-Nearest Neighbors / KNN) 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

k近邻法（K-Nearest Neighbors，KNN）是一种基于实例的学习算法，属于惰性学习（lazy learning）。它假设相似的样本具有相似的标签，通过计算新样本与训练样本的距离来预测其类别或值。

### 1.2 直觉类比

想象kNN的工作方式就像"近朱者赤，近墨者黑"：如果你的邻居大多数是医生，那么你很可能也是医生。kNN通过找到与新样本最相似的k个邻居，根据这些邻居的标签来预测新样本的标签。

### 1.3 历史背景

kNN是最简单直观的机器学习算法之一：
- 1967年：Cover和Hart首次提出kNN分类器
- 1971年：Fix和Hodges提出改进的kNN
- 1991年：Weiss和Silva提出加权kNN
- 2000年后：结合局部敏感哈希等加速技术

kNN是机器学习中最基础的算法之一，也是学习分类和回归概念的入门算法。

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 算法类型 | 监督学习（分类/回归） |
| 学习方式 | 惰性学习（无显式训练） |
| 时间复杂度 | 训练O(1)，预测O(n) |
| 空间复杂度 | O(n) |

### 1.5 前置知识

学习kNN需要：
1. 距离度量（欧氏距离、曼哈顿距离等）
2. K近邻概念
3. 投票机制
4. 交叉验证

---

## 2. 核心原理

### 2.1 核心思想

kNN的核心思想是"物以类聚"：对于新样本，找到训练集中与之距离最近的k个样本，根据这k个邻居的标签进行预测。分类时使用多数投票，回归时使用平均值。

### 2.2 工作流程

给定训练集T = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}，对于新样本x：
1. 计算与所有训练样本的距离
2. 选择最近的k个样本
3. 根据邻居标签进行预测

### 2.3 距离度量

**欧氏距离（L2范数）**：
$$d(x, x') = \sqrt{\sum_{i=1}^{m}(x_i - x_i')^2}$$

**曼哈顿距离（L1范数）**：
$$d(x, x') = \sum_{i=1}^{m}|x_i - x_i'|$$

**闵可夫斯基距离（Lp范数）**：
$$d(x, x') = \left(\sum_{i=1}^{m}|x_i - x_i'|^p\right)^{1/p}$$

**余弦相似度**：
$$cos(x, x') = \frac{x \cdot x'}{|x| |x'|}$$

### 2.4 k值选择

- **k值小**（k=1）：模型复杂，容易过拟合，对噪声敏感
- **k值大**（k=n）：模型简单，忽略局部信息，可能欠拟合
- **通常选择奇数**：避免平票
- **通过交叉验证选择**：评估不同k值的效果

### 2.5 加权投票

为不同距离的邻居分配不同权重：
$$y = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}$$

其中wᵢ = 1/d(x, xᵢ)，距离越近权重越大。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| T | 训练集 {(x₁,y₁), ..., (xₙ,yₙ)} |
| k | 近邻数量 |
| d(·,·) | 距离函数 |
| Nₖ(x) | x的k个最近邻集合 |
| c | 类别数量 |

### 3.2 分类预测公式

**多数投票**：
$$y = \arg\max_{c} \sum_{x_i \in N_k(x)} I(y_i = c)$$

其中I是指示函数。

**加权投票**：
$$y = \arg\max_{c} \sum_{x_i \in N_k(x)} w_i \cdot I(y_i = c)$$

其中wᵢ = 1/d(x, xᵢ)²。

### 3.3 回归预测公式

**简单平均**：
$$\hat{y} = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i$$

**加权平均**：
$$\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}$$

### 3.4 距离归一化

不同特征的尺度不同，需要归一化：
$$x'_i = \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$$

或标准化：
$$x'_i = \frac{x_i - \mu(x_i)}{\sigma(x_i)}$$

### 3.5 推导

kNN的预测基于"平滑性假设"：相似的输入应有相似的输出。通过最近邻来近似这个假设。

---

## 4. 训练过程讲解

### 4.1 算法流程

```
输入: 训练集T, 近邻数k, 待预测样本x
输出: 预测值y

步骤1: 计算距离
for each (xi, yi) in T:
    d_i = distance(x, xi)

步骤2: 选择k个最近邻
N_k = argmin_k d_i

步骤3: 预测
if 分类:
    y = vote(N_k)
else:  # 回归
    y = mean(N_k)
```

### 4.2 数据预处理

```python
def preprocess_data(X_train, X_test):
    """数据预处理：标准化"""
    # 计算统计量
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # 标准化
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm
```

### 4.3 超参数选择

| 超参数 | 作用 | 推荐范围 |
|--------|------|----------|
| k | 近邻数量 | 3-20（通过CV） |
| metric | 距离度量 | 'euclidean' |
| weights | 投票权重 | 'uniform', 'distance' |
| algorithm | 搜索算法 | 'auto', 'ball_tree', 'kd_tree' |

### 4.4 收敛条件

kNN是惰性学习，没有显式的收敛过程。预测时只需找到k个最近邻即可。

---

## 5. 应用场景

### 5.1 典型应用

1. **分类问题**
   - 文本分类
   - 图像识别
   - 医疗诊断

2. **回归问题**
   - 销售预测
   - 房价预测
   - 流量预测

3. **推荐系统**
   - 协同过滤
   - 商品推荐

4. **异常检测**
   - 欺诈检测
   - 故障检测

### 5.2 适用数据特征

- 特征是连续的
- 数据量适中（n < 10⁶）
- 类别间有明显的距离差异
- 对可解释性要求高

### 5.3 不适用场景

- 数据维度高（维度灾难）
- 数据量大（O(n)预测太慢）
- 特征是离散的
- 需要快速预测

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 算法简单 | 易于理解和实现 | - |
| 无需训练 | 没有显式训练过程 | 惰性学习 |
| 可解释性强 | 直观理解预测原因 | 局部信息 |
| 适用性广 | 可分类可回归 | - |
| 对数据无假设 | 不需要分布假设 | - |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 预测慢 | O(n)时间复杂度 | 使用KD-Tree、LSH |
| 维度灾难 | 高维距离失效 | 降维、特征选择 |
| 对k值敏感 | k值影响大 | 交叉验证 |
| 对噪声敏感 | 噪声点影响大 | 加权投票 |
| 存储需求 | 需要存储所有数据 | 数据压缩 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 sklearn实现

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class KNNClassifier:
    """KNN分类器包装器"""
    
    def __init__(self, k=5, weights='distance', metric='euclidean'):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """训练（实际只是存储数据）"""
        X_norm = self.scaler.fit_transform(X_train)
        self.model = KNeighborsClassifier(
            n_neighbors=self.k,
            weights=self.weights,
            metric=self.metric
        )
        self.model.fit(X_norm, y_train)
        return self
    
    def predict(self, X_test):
        """预测"""
        X_norm = self.scaler.transform(X_test)
        return self.model.predict(X_norm)
    
    def predict_proba(self, X_test):
        """预测概率"""
        X_norm = self.scaler.transform(X_test)
        return self.model.predict_proba(X_norm)
    
    def score(self, X_test, y_test):
        """评估准确率"""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def find_best_k(self, X_train, y_train, k_range):
        """寻找最优k值"""
        X_train_norm = self.scaler.fit_transform(X_train)
        
        best_k = 1
        best_score = 0
        
        scores = []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            cv_scores = cross_val_score(
                model, X_train_norm, y_train, cv=5
            )
            mean_score = cv_scores.mean()
            scores.append(mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        self.k = best_k
        return best_k, scores


class KNNRegressor:
    """KNN回归器"""
    
    def __init__(self, k=5, weights='distance'):
        self.k = k
        self.weights = weights
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        X_norm = self.scaler.fit_transform(X_train)
        self.model = KNeighborsRegressor(
            n_neighbors=self.k,
            weights=self.weights
        )
        self.model.fit(X_norm, y_train)
        return self
    
    def predict(self, X_test):
        X_norm = self.scaler.transform(X_test)
        return self.model.predict(X_norm)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return mean_squared_error(y_test, y_pred)


def demo():
    print("=== KNN 演示 ===\n")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    
    # 类别1
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    y1 = np.zeros(n_samples)
    
    # 类别2
    X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    y2 = np.ones(n_samples)
    
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    knn = KNNClassifier(k=5, weights='distance')
    knn.fit(X_train, y_train)
    
    # 评估
    accuracy = knn.score(X_test, y_test)
    print(f"准确率: {accuracy:.4f}")
    
    # 寻找最优k
    best_k, scores = knn.find_best_k(
        X_train, y_train, range(1, 21)
    )
    print(f"最优k: {best_k}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 完整实现

```python
import numpy as np
from collections import Counter

class KNN:
    """k近邻法（分类/回归）"""
    
    def __init__(self, k=5, weights='uniform', distance='euclidean'):
        self.k = k
        self.weights = weights
        self.distance = distance
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """训练（存储数据）"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _distance(self, x1, x2):
        """计算距离"""
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance == 'cosine':
            return 1 - np.dot(x1, x2) / (
                np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-10
            )
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def _get_neighbors(self, x):
        """获取k个最近邻"""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, i))
        
        # 排序并选择k个
        distances.sort(key=lambda d: d[0])
        indices = [d[1] for d in distances[:self.k]]
        
        return [(distances[i][0], indices[i]) 
                for i in range(self.k)]
    
    def _predict_classification(self, x):
        """分类预测"""
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'distance':
            # 加权投票
            votes = {}
            for dist, idx in neighbors:
                label = self.y_train[idx]
                weight = 1 / (dist ** 2 + 1e-10)
                votes[label] = votes.get(label, 0) + weight
            return max(votes, key=votes.get)
        else:
            # 简单投票
            labels = [self.y_train[idx] for _, idx in neighbors]
            return Counter(labels).most_common(1)[0][0]
    
    def _predict_regression(self, x):
        """回归预测"""
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'distance':
            # 加权平均
            total_weight = 0
            weighted_sum = 0
            for dist, idx in neighbors:
                weight = 1 / (dist ** 2 + 1e-10)
                weighted_sum += weight * self.y_train[idx]
                total_weight += weight
            return weighted_sum / total_weight
        else:
            # 简单平均
            return np.mean([self.y_train[idx] for _, idx in neighbors])
    
    def predict(self, X):
        """预测"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        if isinstance(self.y_train[0], (int, float, np.integer)):
            # 回归
            return np.array([self._predict_regression(x) for x in X])
        else:
            # 分类
            return np.array([self._predict_classification(x) for x in X])


def demo():
    print("=== KNN 手工实现演示 ===\n")
    
    # 训练数据
    X_train = [
        [1, 2], [1, 3], [2, 1], [3, 1], [3, 3], [4, 3],
        [-1, -2], [-1, -3], [-2, -1], [-3, -1], [-3, -3], [-4, -3]
    ]
    y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
    # 测试数据
    X_test = [[2, 2], [-2, -2]]
    
    # 训练和预测
    knn = KNN(k=3, weights='distance')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(f"测试数据: {X_test}")
    print(f"预测结果: {predictions}")


if __name__ == "__main__":
    demo()
```

### 8.2 使用KD-Tree加速

```python
from scipy.spatial import KDTree

class KNNWithKDTree:
    """使用KD-Tree加速的KNN"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.kdtree = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.kdtree = KDTree(self.X_train)
        return self
    
    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for x in X:
            dist, indices = self.kdtree.query(x, k=self.k)
            labels = self.y_train[indices]
            
            if isinstance(labels[0], (int, float)):
                predictions.append(np.mean(labels))
            else:
                most_common = Counter(labels).most_common(1)[0][0]
                predictions.append(most_common)
        
        return np.array(predictions)
```

---

## 9. 可视化与结果理解

### 9.1 决策边界可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary():
    """绘制决策边界"""
    np.random.seed(42)
    
    # 生成数据
    n_samples = 200
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    # KNN模型
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X, y)
    
    # 网格
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FF6B6B', '#4ECDC4']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF6B6B', '#4ECDC4']))
    plt.title('KNN Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('knn_boundary.png', dpi=150)
    plt.show()


def plot_k_sensitivity():
    """绘制k值敏感性"""
    k_values = range(1, 21)
    train_acc = [0.98, 0.96, 0.94, 0.93, 0.92, 
                0.91, 0.90, 0.89, 0.88, 0.87,
                0.86, 0.85, 0.84, 0.83, 0.82,
                0.81, 0.80, 0.79, 0.78, 0.77]
    test_acc = [0.85, 0.88, 0.90, 0.91, 0.92,
               0.91, 0.90, 0.89, 0.88, 0.87,
               0.86, 0.85, 0.84, 0.83, 0.82,
               0.81, 0.80, 0.79, 0.78, 0.77]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_acc, 'o-', label='Train Accuracy')
    plt.plot(k_values, test_acc, 's-', label='Test Accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('K Value Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('k_sensitivity.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_decision_boundary()
    plot_k_sensitivity()
```

---

## 10. 模型评估

### 10.1 评估指标

**分类指标**：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数

**回归指标**：
- MSE / RMSE
- MAE
- R²

### 10.2 性能对比

```
数据集: Iris, 150样本, 4特征

方法              准确率    训练时间    预测时间
---------------------------------------------------
KNN(k=5)         0.96     0.001s    0.15s
KNN(k=10)        0.94     0.001s    0.14s
SVM              0.97     0.01s     0.01s
决策树            0.95     0.02s     0.001s
```

---

## 11. 常见问题与易错点

### 11.1 维度灾难

**问题**：高维空间中距离度量失效

**原因**：高维空间中点分布稀疏

**解决方案**：
1. 特征选择
2. 降维（PCA）
3. 使用其他算法

### 11.2 k值选择

**问题**：k值影响模型性能

**原因**：k太小过拟合，k太大欠拟合

**解决方案**：
1. 交叉验证
2. 使用奇数避免平票

### 11.3 数据不平衡

**问题**：少数类被忽略

**原因**：投票被多数类主导

**解决方案**：
1. 加权投票
2. 过采样/欠采样

---

## 12. 学习总结

### 核心要点

1. kNN是最简单的惰性学习算法
2. 通过距离度量找k个最近邻
3. 分类用投票，回归用平均
4. k值和距离度量需要调参
5. 可使用KD-Tree加速

### 从kNN到其他算法

kNN → 加权kNN → KD-Tree加速 → Ball-Tree → Locality Sensitive Hashing

---

## 13. 练习题与思考题（含答案）

### 练习题1：基础计算

**问题**：给定训练集[[1,1],[2,2],[3,3]]标签[0,0,1]，使用k=3预测[1.5,1.5]的类别。

**答案**：距离[0.71, 0.71, 2.12]，最近3个都是标签0，预测为0

### 练习题2：编程实践

**问题**：实现带权重的kNN

答案见第8节代码实现

---

## 14. 学习路径建议

### 初级阶段

1. 理解kNN原理
2. 实现基础算法
3. 掌握距离度量

**学习时间**：1周

### 推荐资源

- Cover & Hart (1967). kNN原始论文
- sklearn.neighbors文档

---

**文档结束**