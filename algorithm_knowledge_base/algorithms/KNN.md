# KNN 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
K近邻（K-Nearest Neighbors, KNN）是一种基于实例的学习算法，它不需要显式的训练过程，而是通过计算待分类样本与训练集样本之间的距离，找出最近的K个邻居，根据这些邻居的类别标签进行投票来决定待分类样本的类别。

### 1.2 直觉类比
KNN的工作方式非常类似于人类的生活经验。想象你搬到了一个新小区，你想知道这个小区住户的整体素质。你会怎么做？你可能会去询问与你最近的几个邻居（K个邻居）的看法。如果5个邻居中有4个说这里的住户都很友好、乐于助人，只有1个邻居说有时会有噪音干扰，你就会倾向于认为这是个不错的社区。这就是KNN的核心思想——"近朱者赤，近墨者黑"。K值的选择就像是你选择询问多少个邻居一样：问得太少可能被少数人误导，问得太多又可能忽略了你最想了解的那个群体的特征。

### 1.3 历史背景
KNN算法是最简单也是最直观的机器学习算法之一。其基本思想可以追溯到1951年，Fix和Hodgins在最邻近判别分析（Nearest Neighbor Discriminant Analysis）的工作中奠定了理论基础。1967年，Cover和Hart首次正式提出了K近邻算法的概念，并分析了其渐进收敛性质。KNN算法在1960-1970年代被广泛应用于模式识别领域，包括光学字符识别（OCR）、语音识别和图像分类等。由于其简单直观的特点，KNN至今仍是机器学习中不可或缺的baseline模型，也是许多初学者学习机器学习的入门算法。

### 1.4 算法定位
- **类型**：监督学习（Supervised Learning）
- **输出**：离散类别（K分类）或连续值（回归）
- **模型类别**：非参数模型（基于实例的学习）
- **学习范式**：惰性学习（Lazy Learning）
- **求解方法**：距离度量 + K近邻搜索 + 投票机制

### 1.5 前置知识
- **线性代数**：向量范数（曼哈顿、欧几里得）
- **概率论**：投票机制的概率解释
- **算法复杂度**：时间复杂度、空间复杂度
- **Python编程**：NumPy、scikit-learn

---

## 2. 核心原理

### 2.1 核心思想
KNN的核心思想可以概括为"物以类聚，人以群分"。给定一个待分类的样本，KNN算法不会像传统机器学习算法那样学习一个通用的模型参数，而是直接利用训练数据本身。当需要预测新样本的类别时，算法会找到训练集中与该新样本距离最近的K个样本（K个近邻），然后根据这K个近邻的类别标签进行投票，票数最多的类别即为新样本的预测类别。这种"基于实例"的方法使得KNN对数据的分布没有假设，因此可以适应各种复杂的数据分布。

### 2.2 工作流程
1. **确定距离度量**：选择合适的距离函数（如欧氏距离）
2. **确定K值**：选择最近的邻居数量K
3. **寻找近邻**：在训练集中找到与待预测样本距离最近的K个样本
4. **投票决策**：统计K个近邻的类别，票数最多的类别作为预测结果
5. **返回预测**：输出预测的类别标签

### 2.3 关键概念解释
- **K值**：K近邻中的K，表示选择多少个最近的邻居参与投票。K值的选择对算法结果有重要影响：K值太小容易受到噪声影响，K值太大容易忽略局部特征。K值通常选择奇数以避免平票情况。
- **距离度量**：用于衡量样本之间相似性或距离的函数。常用的距离度量包括：欧氏距离、曼哈顿距离、切比雪夫距离、余弦相似度、马氏距离等。选择何种距离度量取决于具体应用场景和数据特征。
- **决策边界**：将特征空间划分为不同类别的边界。在KNN中，决策边界是通过所有样本点的类别投票自然形成的，往往不是线性的，而是呈现复杂的不规则形状。
- **维度灾难**：当特征维度很高时，高维空间中的距离度量会失去意义，所有点到目标点的距离趋于相近，导致KNN失效。这是KNN面临的主要挑战之一。

### 2.4 几何/直观解释
KNN的决策边界可以通过维诺图（Voronoi Diagram）来直观理解。在二维特征空间中，每个训练样本点相当于一个"势力范围"的中心点，这些中心点将整个平面划分成若干个区域，每个区域内的点都以该区域中心点为最近的训练样本。K=1时的决策边界就是这些区域的边界线。K>1时，决策边界会变得更加平滑，因为需要考虑多个邻居的意见。在高维空间中，KNN的工作方式类似，但难以可视化。核函数方法（如KD树）可以加速近邻搜索，避免每次预测时都需要遍历所有训练样本。

---

## 3. 数学公式与推导

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| X ∈ R^(n×d) | 特征矩阵，n个样本，d维特征 |
| y ∈ {1,2,...,c}^n | 标签向量，c个类别 |
| K | 近邻数量 |
| d(x_i, x_j) | 样本x_i和x_j之间的距离 |
| N_K(x) | 距离x最近的K个样本的集合 |
| vote(c_i) | 类别c_i获得的票数 |

### 3.2 问题形式化
**分类问题**：对于给定的新样本x，找到其K个近邻，根据投票结果决定类别：
$$\hat{y} = \arg\max_{c} \sum_{x_i \in N_K(x)} \mathbb{I}[y_i = c]$$

**回归问题**：对于给定的新样本x，计算K个近邻的真实值的加权平均：
$$\hat{y} = \frac{1}{K} \sum_{x_i \in N_K(x)} y_i$$

### 3.3 常见距离度量
1. **欧氏距离（L2范数）**：
$$d(x_i, x_j) = \sqrt{\sum_{l=1}^{d} (x_{il} - x_{jl})^2}$$

2. **曼哈顿距离（L1范数）**：
$$d(x_i, x_j) = \sum_{l=1}^{d} |x_{il} - x_{jl}|$$

3. **切比雪夫距离（L∞范数）**：
$$d(x_i, x_j) = \max_{l} |x_{il} - x_{jl}|$$

4. **余弦相似度**：
$$\cos\theta = \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||}$$

5. **马氏距离**：
$$d(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)}$$

其中Σ是协方差矩阵。

6. **明可夫斯基距离**：
$$d(x_i, x_j) = \left(\sum_{l=1}^{d} |x_{il} - x_{jl}|^p\right)^{1/p}$$

当p=1时为曼哈顿距离，p=2时为欧氏距离，p→∞时为切比雪夫距离。

### 3.4 距离推导过程
**Step 1：距离计算**
对于特征向量x_i和x_j，首先计算它们在每个维度上的差的绝对值，然后根据距离类型进行聚合。

**Step 2：近邻选择**
计算待预测样本与所有训练样本的距离，排序后选择最小的K个。

**Step 3：投票计算**
统计K个近邻中各类别的数量：
$$\text{vote}(c) = \sum_{x_i \in N_K(x)} \mathbb{I}[y_i = c]$$

**Step 4：预测输出**
选择票数最多的类别作为预测结果：
$$\hat{y} = \arg\max_c \text{vote}(c)$$

### 3.5 带权重的KNN
为近邻的投票添加距离权重，距离越近权重越大：

**距离加权投票**：
$$\hat{y} = \arg\max_c \sum_{x_i \in N_K(x)} w_i \cdot \mathbb{I}[y_i = c]$$

其中权重可以定义为：
- $w_i = \frac{1}{d(x, x_i)^2}$（反距离加权）
- $w_i = \exp(-d(x, x_i)^2 / h)$（高斯核加权）
- $w_i = \frac{1}{K}$（均匀加权）

**加权KNN回归**：
$$\hat{y} = \frac{\sum_{i \in N_K(x)} w_i y_i}{\sum_{i \in N_K(x)} w_i}$$

---

## 4. 训练过程讲解

### 4.1 数据预处理
1. **特征标准化**
   KNN对特征的尺度非常敏感，因为距离计算直接依赖于特征值。不同尺度的特征会导致某些特征主导距离计算。建议使用Z-score标准化或Min-Max归一化：
   
   ```python
   # Z-score标准化
   X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
   
   # Min-Max归一化
   X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
   ```

2. **缺失值处理**
   - 删除含有缺失值的样本
   - 用均值/中位数填充
   - 使用更复杂的插补方法

3. **特征选择/降维**
   当特征维度很高时，需要进行特征选择或降维（如PCA、特征选择），以避免维度灾难。

### 4.2 参数初始化
KNN算法是惰性学习算法，没有显式的"训练"过程。算法的"参数"主要是：
- **K值**：通常是奇数，如1, 3, 5, 7, ...
- **距离度量**：默认为欧氏距离
- **权重策略**：均匀权重或距离加权

### 4.3 迭代过程
KNN不需要迭代训练，但预测过程需要遍历训练集。完整的预测流程：

```
输入：待预测样本x，训练集X_train, y_train，K值，距离函数
输出：预测类别y_pred

1. 对每个训练样本x_i in X_train，计算d(x, x_i)
2. 排序所有距离，选择最小的K个样本
3. 统计这K个样本的类别标签
4. 返回票数最多的类别
```

### 4.4 收敛条件
KNN没有"收敛"的概念，因为算法不需要学习参数。算法效果取决于：
- K值的选择
- 距离度量的选择
- 数据的预处理

### 4.5 超参数及推荐范围
- **K（近邻数量）**：
  - 小数据集：K = 3, 5, 7
  - 一般数据集：K = 3, 5, 7, 9, 11
  - 大数据集：K = 15, 21, 31（需平衡准确率和速度）
  
- **距离度量**：
  - 数值特征：'euclidean'（欧氏距离）
  - 高维稀疏：'cosine'（余弦相似度）
  - 曼哈顿城市：'manhattan'
  
- **权重**：
  - 'uniform'（均匀权重）
  - 'distance'（距离加权）

---

## 5. 应用场景

### 5.1 典型应用
1. **推荐系统**
   - 电商平台的商品推荐：找到与你购买历史最相似的K个用户，推荐他们购买但你未购买的商品
   - 电影/音乐推荐：协同过滤的基础
   
2. **模式识别**
   - 手写数字识别：识别手写的0-9数字
   - 人脸识别：找到最相似的K个人脸数据库
  
3. **数据分类**
   - 文本分类：垃圾邮件过滤
   - 网络入侵检测

4. **回归分析**
   - 预测房价：基于K个最相似房屋的均价
   - 预测销量

### 5.2 适用数据特征
- **数据量**：中小规模（几千到几万样本）
- **特征维度**：低维到中维（避免维度灾难）
- **类别数**：二分类或多分类
- **数据分布**：任意分布（无参数假设）

### 5.3 不适用场景
1. **大规模数据**：每次预测都需要遍历全部训练数据，时间复杂度O(n)
2. **高维数据**：维度灾难问题，距离度量失效
3. **特征尺度差异大**：未标准化的数据
4. **实时性要求高**：需要大量计算

---

## 6. 优缺点分析

### 6.1 优点
1. **简单直观**：算法思想简单，易于理解和实现
2. **无需训练**：没有显式的训练过程，没有过拟合风险
3. **适应性强**：对数据分布没有假设，可以处理任意分布的数据
4. **可解释性强**：可以解释为什么做出这样的预测
5. **多功能**：可以通过修改K值和距离度量适应不同任务

### 6.2 缺点
1. **预测时间长**：时间复杂度O(n)，不适合大规模数据
2. **维度灾难**：高维空间中距离度量失效
3. **对特征尺度敏感**：需要标准化
4. **K值选择困难**：K值对结果影响大
5. **不抗噪声**：容易被异常点影响

---

## 7. 调库实现（sklearn完整代码）

```python
"""
K近邻分类器 - sklearn实现
作者：算法学习文档
功能：使用sklearn的KNeighborsClassifier类进行分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 生成模拟数据集
print("=" * 50)
print("1. 生成模拟数据集")
print("=" * 50)

# 使用moon数据集（非线性）
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
y = np.where(y == 0, -1, 1)

print(f"数据集大小: {X.shape}")
print(f"类别分布: 类+1: {np.sum(y==1)}, 类-1: {np.sum(y==-1)}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 特征标准化（重要！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train_scaled.shape}")
print(f"测试集大小: {X_test_scaled.shape}")

# 4. 使用不同的K值训练KNN
print("\n" + "=" * 50)
print("2. 不同K值的KNN分类器")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11]
results = {}

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,           # K值
        weights='uniform',     # 均匀权重
        metric='euclidean',    # 欧氏距离
        algorithm='auto',      # 自动选择最优算法
        n_jobs=-1             # 并行计算
    )
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[k] = acc
    print(f"K={k:2d}: 准确率 = {acc:.4f}")

# 5. 距离加权KNN
print("\n" + "=" * 50)
print("3. 距离加权KNN")
print("=" * 50)

knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',      # 距离加权
    metric='euclidean'
)
knn_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = knn_weighted.predict(X_test_scaled)
acc_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"距离加权 KNN: 准确率 = {acc_weighted:.4f}")

# 6. 使用Manhattan距离
print("\n" + "=" * 50)
print("4. 不同距离度量")
print("=" * 50)

for metric in ['euclidean', 'manhattan', 'chebyshev']:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"距离度量 '{metric}': 准确率 = {accuracy_score(y_test, y_pred):.4f}")

# 7. 交叉验证选择最佳K值
print("\n" + "=" * 50)
print("5. 交叉验证选择最佳K值")
print("=" * 50)

k_range = range(1, 31, 2)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"K={k:2d}: CV准确率 = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

best_k = k_range[np.argmax(cv_scores)]
print(f"\n最佳K值: {best_k}, CV准确率: {max(cv_scores):.4f}")

# 8. 使用最佳K值训练最终模型
print("\n" + "=" * 50)
print("6. 最佳模型评估")
print("=" * 50)

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)

print(f"训练集准确率: {best_knn.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_best):.4f}")

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_best))

print("\n分类报告:")
print(classification_report(y_test, y_pred_best))

# 9. 查看最近邻
print("\n" + "=" * 50)
print("7. 最近邻信息")
print("=" * 50)

distances, indices = best_knn.kneighbors(X_test_scaled[:5])
print("测试集前5个样本的K个最近邻:")
for i in range(5):
    print(f"样本{i}: 最近的{best_k}个邻居索引 = {indices[i]}, 距离 = {distances[i]}")

# 10. 可视化
def plot_knn_decision_boundary(X, y, model, scaler, title):
    """绘制KNN决策边界"""
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                       np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='+1 类', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='-1 类', alpha=0.6)
    
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'knn_{title}.png', dpi=150)
    plt.show()

# 绘制不同K值的决策边界
for k in [1, 3, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    plot_knn_decision_boundary(X_test, y_test, knn, scaler, f'K{k}')
```

运行上述代码，将输出：
- 不同K值的准确率对比
- 距离加权与均匀权重的对比
- 不同距离度量的对比
- 交叉验证结果
- 决策边界可视化

---

## 8. 手工代码实现（NumPy）

```python
"""
K近邻分类器 - 纯NumPy实现
作者：算法学习文档
功能：从零实现KNN分类器，深入理解算法原理
"""

import numpy as np
import matplotlib.pyplot as plt

class KNNFromScratch:
    """
    K近邻分类器的完整实现
    
    参数:
    -------
    k : int, 默认5
        近邻数量
    distance_metric : str, 默认'euclidean'
        距离度量方式
    weights : str, 默认'uniform'
        权重类型：'uniform' 或 'distance'
    """
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
    def _compute_distance(self, x1, x2):
        """
        计算两个样本之间的距离
        
        参数:
        -------
        x1, x2 : array-like
            两个样本的特征向量
        返回:
        -------
        float: 距离值
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_all_distances(self, x):
        """
        计算待预测样本与所有训练样本的距离
        
        参数:
        -------
        x : array-like
            待预测样本
        返回:
        -------
        distances : array
            与所有训练样本的距离数组
        """
        n_train = self.X_train.shape[0]
        distances = np.zeros(n_train)
        
        for i in range(n_train):
            distances[i] = self._compute_distance(x, self.X_train[i])
        
        return distances
    
    def _get_k_nearest_neighbors(self, x):
        """
        找到K个最近邻
        
        参数:
        -------
        x : array-like
            待预测样本
        返回:
        -------
        indices : array
            K个最近邻的索引
        distances : array
            对应的距离
        """
        distances = self._compute_all_distances(x)
        
        # 使用argpartition快速找到最小的K个元素
        k_smallest_indices = np.argpartition(distances, self.k)[:self.k]
        
        # 排序这K个索引
        k_sorted_indices = k_smallest_indices[np.argsort(distances[k_smallest_indices])]
        
        return k_sorted_indices, distances[k_sorted_indices]
    
    def _vote(self, neighbor_indices, neighbor_distances):
        """
        投票决定类别
        
        参数:
        -------
        neighbor_indices : array
            K个最近邻的索引
        neighbor_distances : array
            对应的距离
        返回:
        -------
        predicted_class : int
            预测的类别
        """
        neighbor_labels = self.y_train[neighbor_indices]
        
        if self.weights == 'uniform':
            # 均匀权重
            unique_labels = np.unique(neighbor_labels)
            votes = np.zeros(len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                votes[i] = np.sum(neighbor_labels == label)
            
            return unique_labels[np.argmax(votes)]
        
        elif self.weights == 'distance':
            # 距离加权
            unique_labels = np.unique(neighbor_labels)
            weighted_votes = np.zeros(len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                # 找到该类别的所有近邻索引
                label_mask = neighbor_labels == label
                label_distances = neighbor_distances[label_mask]
                
                # 使用逆距离加权
                if np.any(label_distances == 0):
                    # 如果有距离为0的（完全相同），直接返回该类别
                    return label
                
                weights = 1.0 / (label_distances ** 2)
                weighted_votes[i] = np.sum(weights)
            
            return unique_labels[np.argmax(weighted_votes)]
    
    def fit(self, X, y):
        """
        训练模型（实际上是保存训练数据）
        
        参数:
        -------
        X : array-like, shape (n_samples, n_features)
            训练特征矩阵
        y : array-like, shape (n_samples,)
            训练标签向量
        返回:
        -------
        self
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"KNN模型已训练，包含 {len(y)} 个样本")
        
        return self
    
    def predict(self, X):
        """
        预测多样本
        
        参数:
        -------
        X : array-like, shape (n_samples, n_features)
            待预测样本矩阵
        返回:
        -------
        predictions : array
            预测的类别标签数组
        """
        X = np.array(X)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # 找到K个最近邻
            neighbor_indices, neighbor_distances = self._get_k_nearest_neighbors(X[i])
            
            # 投票
            predictions[i] = self._vote(neighbor_indices, neighbor_distances)
        
        return predictions
    
    def predict_single(self, x):
        """
        预测单个样本
        
        参数:
        -------
        x : array-like
            待预测样本
        返回:
        -------
        predicted_class : int
            预测的类别
        """
        neighbor_indices, neighbor_distances = self._get_k_nearest_neighbors(x)
        return self._vote(neighbor_indices, neighbor_distances)


class KDTree:
    """
    KD树的简化实现，用于加速K近邻搜索
    """
    
    def __init__(self, data):
        self.data = np.array(data)
        self.root = self._build_tree(range(len(data)), depth=0)
    
    def _build_tree(self, indices, depth):
        """递归构建KD树"""
        if len(indices) == 0:
            return None
        
        axis = depth % self.data.shape[1]
        
        # 按选中的轴排序
        sorted_indices = sorted(indices, key=lambda i: self.data[i, axis])
        
        mid = len(sorted_indices) // 2
        
        node = {
            'index': sorted_indices[mid],
            'axis': axis,
            'left': self._build_tree(sorted_indices[:mid], depth + 1),
            'right': self._build_tree(sorted_indices[mid + 1:], depth + 1)
        }
        
        return node
    
    def query(self, x, k=1):
        """查询K个最近邻"""
        # 这是一个简化的实现
        # 实际中需要用优先队列管理距离
        distances = np.linalg.norm(self.data - x, axis=1)
        indices = np.argpartition(distances, k)[:k]
        return indices, distances[indices]


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    
    # 生成两类数据
    n = 100
    
    # 类1：中心在(1, 1)
    X1 = np.random.randn(n, 2) + [1, 1]
    y1 = np.ones(n)
    
    # 类2：中心在(-1, -1)
    X2 = np.random.randn(n, 2) + [-1, -1]
    y2 = -np.ones(n)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    return X, y


def plot_knn_result(X, y, model, title):
    """绘制KNN结果"""
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                       np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策区域
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # 绘制数据点
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='类 1', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='类 -1', alpha=0.6)
    
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'knn_custom_{title}.png', dpi=150)
    plt.show()


# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("K近邻分类器 - 手工实现")
    print("=" * 50)
    
    # 1. 创建测试数据
    X, y = create_test_data()
    
    # 2. 划分训练集和测试集
    n_train = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # 3. 标准化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    # 4. 训练不同K值的KNN
    print("\n测试不同的K值...")
    for k in [1, 3, 5, 7]:
        knn = KNNFromScratch(k=k, distance_metric='euclidean', weights='uniform')
        knn.fit(X_train_scaled, y_train)
        
        y_pred = knn.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"K={k}: 准确率 = {accuracy:.4f}")
    
    # 5. 测试距离加权
    print("\n测试距离加权...")
    knn_weighted = KNNFromScratch(k=5, weights='distance')
    knn_weighted.fit(X_train_scaled, y_train)
    y_pred_weighted = knn_weighted.predict(X_test_scaled)
    print(f"距离加权 KNN: 准确率 = {np.mean(y_pred_weighted == y_test):.4f}")
    
    # 6. 可视化
    print("\n生成可视化...")
    plot_knn_result(X_test_scaled, y_test, knn, 'K5')
    
    # 7. 与sklearn对比
    print("\n与sklearn对比...")
    from sklearn.neighbors import KNeighborsClassifier
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test_scaled)
    print(f"sklearn KNN: 准确率 = {np.mean(y_pred_sklearn == y_test):.4f}")
    
    print("\n完成!")
```

运行上述代码，将输出：
- 不同K值的准确率对比
- 均匀权重与距离权重的对比
- 与sklearn实现的对比结果
- 决策边界可视化

---

## 9. 可视化与结果理解

### 9.1 决策边界可视化
KNN的决策边界是自然形成的，不需要显式学习。通过可视化可以直观理解：
- K=1时：决策边界非常不规则，容易过拟合
- K越大：决策边界越平滑
- K=N时：总是预测多数类

### 9.2 K值选择影响
```python
import matplotlib.pyplot as plt

# 测试不同K值
k_values = [1, 3, 5, 9, 15]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, 'bo-')
plt.xlabel('K值')
plt.ylabel('准确率')
plt.title('K值与准确率的关系')
plt.grid(True)
plt.savefig('knn_k_selection.png')
plt.show()
```

### 9.3 结果分析
1. **过拟合**：K值太小，决策边界不光滑
2. **欠拟合**：K值太大，可能忽略局部特征
3. **最佳K值**：通过交叉验证确定

---

## 10. 模型评估

### 10.1 评估指标
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = knn.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率: {precision_score(y_test, y_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
```

### 10.2 K折交叉验证
```python
from sklearn.model_selection import cross_val_score

for k in range(1, 21, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(f"K={k}: {scores.mean():.4f} +/- {scores.std()*2:.4f}")
```

### 10.3 学习曲线
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=5), X, y, cv=5
)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'b-', label='训练集')
plt.plot(train_sizes, test_scores.mean(axis=1), 'r-', label='测试集')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.legend()
plt.title('KNN学习曲线')
plt.savefig('knn_learning_curve.png')
plt.show()
```

---

## 11. 常见问题与易错点

### 11.1 常见问题
1. **预测时间太长**
   - 原因：每次预测都要遍历所有训练样本
   - 解决：使用KD树或Ball树索引，或者减少K值

2. **效果不好**
   - 原因：特征未标准化，或K值选择不当
   - 解决：标准化特征，通过交叉验证选择K值

3. **维度灾难**
   - 原因：特征维度太高
   - 解决：降维（PCA）或特征选择

### 11.2 易错点
1. **忽略特征标准化**
   - 未标准化的特征会导致某些维度主导距离计算

2. **K值选择错误**
   - K=N意味着总是预测多数类
   - K=1意味着没有平滑，过于敏感

3. **距离度量选择错误**
   - 欧氏距离对尺度敏感
   - 余弦距离更适合文本数据

4. **忽略类别不平衡**
   - 少数类可能被忽略

---

## 12. 学习总结

### 12.1 核心要点回顾
1. **基于实例**：不需要显式训练，直接利用训练数据
2. **投票机制**：K个近邻投票决定类别
3. **惰性学习**：模型在预测时才"学习"
4. **距离度量**：欧氏距离最常用

### 12.2 学习建议
1. **从简单开始**：先理解K=1的情况
2. **实践出真知**：使用真实数据集实验
3. **理解可视化**：观察K值对边界的影响

### 12.3 进一步学习方向
1. **KD树**：加速近邻搜索
2. **加权KNN**：距离加权投票
3. **特征工程**：标准化、特征选择
4. **相似度度量**：根据数据特性选择

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题
1. **KNN算法属于？**
   A. 参数模型   B. 非参数模型   C. 生成模型   D. 判别模型
   **答案：B**（无显式参数）

2. **K=N时，KNN预测结果？**
   A. 总是预测多数类   B. 总是预测少数类   C. 随机预测   D. 无法预测
   **答案：A**（K=N时，只看多数类）

3. **欧氏距离适用于？**
   A. 所有数据   B. 已标准化的数据   C. 文本数据   D. 图像数据
   **答案：B**（标准化后的数据）

### 13.2 简答题
1. **为什么KNN需要特征标准化？**
   答案：不同特征的量纲不同，未标准化会导致某些特征主导距离计算。例如，年龄范围0-100和工资范围0-10000，后者会主导距离。

2. **K值大小对结果有什么影响？**
   - K值太小：决策边界不规则，容易过拟合，噪声影响大
   - K值太大：决策边界过于平滑，可能欠拟合，忽略局部特征

3. **KNN如何处理多分类问题？**
   答案：KNN天然支持多分类，只��要��计K个近邻中各类别的票数，选择票数最多的类别即可。

### 13.3 编程题
1. **实现带权重的KNN**
   ```python
   class WeightedKNN:
       def __init__(self, k=5):
           self.k = k
           
       def predict(self, X_train, y_train, x_test):
           distances = np.linalg.norm(X_train - x_test, axis=1)
           k_indices = np.argpartition(distances, self.k)[:self.k]
           k_distances = distances[k_indices]
           
           # 距离加权
           weights = 1.0 / (k_distances + 1e-10)
           unique_labels = np.unique(y_train[k_indices])
           
           best_label = unique_labels[0]
           best_weight = 0
           
           for label in unique_labels:
               mask = y_train[k_indices] == label
               weight = np.sum(weights[mask])
               if weight > best_weight:
                   best_weight = weight
                   best_label = label
           
           return best_label
   ```

2. **实现KD树加速搜索**
   使用scikit-learn的NearestNeighbors类：
   ```python
   from sklearn.neighbors import NearestNeighbors
   
   # 构建索引
   nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
   nn.fit(X_train)
   
   # 快速搜索
   distances, indices = nn.kneighbors(X_test)
   ```

---

## 14. 学习路径建议建议

### 14.1 入门路径
- **Week 1**：理解KNN原理，手工实现基本版本
- **Week 2**：学习sklearn使用，理解K值选择
- **Week 3**：可视化决策边界，理解算法行为

### 14.2 进阶方向
- 推荐系统
- 协同过滤
- 距离度量学习

### 14.3 推荐资源
- 《机器学习》- 周志华（第10章）
- 《Pattern Recognition》- Duda, Hart & Stork
- scikit-learn官方文档

---

*本学习文档到此结束，建议结合metrics.md和optimization.md一起学习，以更全面地理解机器学习评估指标和优化方法。*