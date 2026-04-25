# Mean Shift（均值漂移）学习文档

> 无监督聚类算法，基于核密度估计找到数据密度最高点

---

## 1. 算法基础认知

**一句话定义**：Mean Shift（均值漂移）是由Fukunaga等人在1975年提出的无监督聚类算法，通过迭代计算数据点的漂移向量（指向密度最高方向），自动找到数据的聚类中心，无需预先指定聚类数量。

**直觉类比**：Mean Shift就像"水往低处流"的反向过程。想象把很多小球放在一张起伏不平的表面上，小球会滚向山谷（局部最低点）。在数据空间中，Mean Shift让每个数据点"滚向"局部密度最高的区域。具体做法是：对每个点，计算它周围所有点的"重心"，然后把点向那个方向移动一点点。重复这个过程，最终所有点都会聚集到几个"坑"里——这就是聚类中心。

**历史背景**：
- 1975年，Fukunaga和Hostetler在论文"The mean shift: a robust approach to feature space analysis"中首次提出
- 2002年，Comaniciu和Meer将其应用于图像分割和跟踪
- 现在是计算机视觉中的经典算法

**算法定位**：
- 类型：无监督学习 → 聚类
- 输出：聚类标签和中心
- 模型类型：基于密度

**前置知识**：
- [必备]：概率统计基础（密度估计）
- [必备]：聚类概念（K-means）
- [推荐]：核函数（Gaussian kernel）

---

## 2. 核心原理

### 2.1 传统聚类的问题

| 方法 | 需要预设K | 对噪声敏感 | 形状假设 |
|------|----------|-----------|----------|
| K-means | 是 | 敏感 | 球形 |
| 层次聚类 | 否 | 中等 | 灵活 |
| **Mean Shift** | **否** | **鲁棒** | **任意** |

### 2.2 Mean Shift的核心思想

**核心洞察**：数据点会自然聚集在密度高的区域！

流程：
1. 对每个数据点，计算核窗口内的加权重心
2. 把点移到重心位置
3. 重复直到收敛

### 2.3 整体流程图

```
         输入数据点
            │
            ▼
    ┌───────────────┐
    │ 计算漂移向量  │ ← 加权重心 - 当前点
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ 更新位置    │ ← 向密度高处移动
    └───────┬───────┘
            │
            ▼
       收敛？
            │
         ┌───┴───┐
         │ 是   │否
         ▼     ▼
      输出   继续
```

---

## 3. 数学公式与推导

### 3.1 核密度估计

**多元核密度**：

$$K_h(x) = \frac{1}{h^D} K\left(\frac{x}{h}\right)$$

其中h是带宽，K是核函数。

**Gaussian核**（最常用）：

$$K(x) = \frac{1}{(2\pi)^{D/2}} \exp\left(-\frac{\|x\|^2}{2}\right)$$

### 3.2 漂移向量

**密度梯度**：

$$\nabla f(x) = \frac{2}{h^2} \frac{\sum_{i=1}^n x_i g\left(\left\| \frac{x - x_i}{h} \right\|^2\right)}{\sum_{i=1}^n g\left(\left\| \frac{x - x_i}{h} \right\|^2\right)}$$

其中$g(x) = -K'(x)$是核的导数。

**简化形式**：

$$m(x) = \frac{\sum_{i=1}^n x_i g\left(\left\| \frac{x - x_i}{h} \right\|^2\right)}{\sum_{i=1}^n g\left(\left\| \frac{x - x_i}{h} \right\|^2\right)} - x$$

这就是**Mean Shift向量**！

### 3.3 更新规则

迭代公式：

$$x_{t+1} = x_t + m(x_t)$$

展开：

$$x_{t+1} = \frac{\sum_{i} x_i K\left(\frac{x_t - x_i}{h}\right)}{\sum_{i} K\left(\frac{x_t - x_i}{h}\right)}$$

### 3.4 权重解释

每个邻居点的权重：

$$w_i = K\left(\frac{\|x - x_i\|}{h}\right)$$

距离近的点权重高，距离远的权重低。

---

## 4. 训练过程讲解

### 4.1 算法流程

```
    初始化：x_i = 原始数据点，h=带宽
    
    迭代（对每个点）：
    ┌───────────────────────────────────┐
    │ 1. 找到窗口内的所有点              │
    │    neighbors = {x_j | ||x_i - x_j|| < h}│
    │                                   │
    │ 2. 计算加权重心                  │
    │    m = Σ(neighbor · K) / Σ(K)      │
    │                                   │
    │ 3. 移动点                     │
    │    x_i_new = m                  │
    │                                   │
    │ 4. 检查收敛                    │
    │    if ||m - x_i|| < ε: 停止    │
    └───────────────────────────────────┘
    
    聚类：收敛后，相近的点→同一聚类
```

### 4.2 带宽h的选择

| h | 效果 |
|---|------|
| 太小 | 过多小聚类 |
| 太大 | 过度平滑 |
| 自适应 | 最佳 |

### 4.3 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| bandwidth h | 5-20%数据范围 | 核窗口大小 |
| eps | 1e-3 | 收敛阈值 |
| max_iter | 300 | 最大迭代 |

### 4.4 实现技巧

| 技巧 | 说明 |
|------|------|
| KD-tree | 加速邻居搜索 |
| 提前终止 | 节省计算 |
| 并行化 | 加速 |

---

## 5. 应用场景

### 5.1 图像分割

```python
# 像素级聚类
# 简化色彩空间 (R,G,B) → Mean Shift
```

### 5.2 目标跟踪

```python
# 在视频中跟踪目标
# 跟踪密度最高的区域
```

### 5.3 聚类分析

```python
# 任意形状的数据聚类
# 无需预设K
```

### 5.4 异常检测

```python
# 密度低的区域→异常点
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **无需预设K** | 自动发现聚类数 |
| **形状灵活** | 可发现任意形状 |
| **鲁棒** | 对噪声不敏感 |
| **可解释** | 基于密度解释 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算重** | O(n²d)或更高 |
| **h敏感** | 带宽选择困难 |
| **内存大** | 需要全部数据 |
| **只聚类** | 不直接做分类 |

### 6.3 改进方案

| 改进 | 方法 |
|------|------|
| MED Shift | 用L1核替代Gaussian |
| Mean Shift++ | 更高效的初始化 |
| Selective Mean Shift | 加速邻居搜索 |

---

## 7. 调库实现

### 7.1 OpenCV实现（推荐）

```python
import cv2
import numpy as np

# 图像分割
img = cv2.imread('image.jpg')
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Mean Shift
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.1)
shifted = cv2.pyrMeanShiftFiltering(img_lab, sp=10, sr=20, maxLevel=1, criteria=criteria)
```

### 7.2 scikit-learn实现

```python
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

# 生成数据
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.6)

# Mean Shift聚类
ms = MeanShift(bandwidth=2)
ms.fit(X)

labels = ms.labels_
centers = ms.cluster_centers_

print(f"发现聚类数: {len(centers)}")
```

### 7.3 手动实现

```python
import numpy as np


def gaussian_kernel(distance, bandwidth):
    """Gaussian核权重"""
    return np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))


def mean_shift_step(X, point, bandwidth):
    """单步Mean Shift"""
    distances = np.linalg.norm(X - point, axis=1)
    weights = gaussian_kernel(distances, bandwidth)
    
    # 加权平均
    if weights.sum() > 0:
        new_point = (weights[:, np.newaxis] * X).sum(axis=0) / weights.sum()
    else:
        new_point = point
        
    return new_point


def mean_shift(X, bandwidth=2.0, max_iter=300, eps=1e-3):
    """Mean Shift聚类"""
    X = np.array(X)
    N = len(X)
    points = X.copy()
    labels = np.zeros(N, dtype=int)
    
    converged = np.zeros(N, dtype=bool)
    cluster_id = 0
    
    for i in range(N):
        if converged[i]:
            continue
            
        point = points[i].copy()
        
        for _ in range(max_iter):
            # 计算漂移
            new_point = mean_shift_step(X, point, bandwidth)
            
            # 检查收敛
            if np.linalg.norm(new_point - point) < eps:
                break
            point = new_point
            
        # 找相近点归为一类
        distances = np.linalg.norm(points - point, axis=1)
        cluster_points = distances < bandwidth * 0.5
        
        if not np.any(converged[cluster_points]):
            labels[cluster_points] = cluster_id
            points[cluster_points] = point
            converged[cluster_points] = True
            cluster_id += 1
        else:
            labels[cluster_points] = labels[converged][np.where(converged)[0][0]]
    
    return labels, points


# 使用示例
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成三个聚类
    X1 = np.random.randn(100, 2) + [0, 0]
    X2 = np.random.randn(100, 2) + [5, 5]
    X3 = np.random.randn(100, 2) + [-5, 5]
    X = np.vstack([X1, X2, X3])
    
    # Mean Shift聚类
    labels, centers = mean_shift(X, bandwidth=2.0)
    
    print(f"发现聚类: {len(np.unique(labels))}")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
import numpy as np
from scipy.spatial import cKDTree


class MeanShiftClustering:
    """完整Mean Shift聚类"""
    
    def __init__(self, bandwidth=2.0, max_iter=300, eps=1e-4, n_jobs=1):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.eps = eps
        self.n_jobs = n_jobs
        self.cluster_centers_ = None
        self.labels_ = None
        
    def _gaussian_kernel(self, distances):
        """Gaussian核"""
        return np.exp(-0.5 * (distances / self.bandwidth) ** 2)
    
    def fit(self, X):
        """训练"""
        X = np.array(X)
        N, D = X.shape
        
        # 构建KD树加速搜索
        tree = cKDTree(X)
        
        # 存储每个点的收敛位置
        points = X.copy()
        converged = np.zeros(N, dtype=bool)
        cluster_centers = []
        
        for i in range(N):
            if converged[i]:
                continue
                
            point = X[i].copy()
            
            for _ in range(self.max_iter):
                # 找邻居
                indices = tree.query_ball_point(point, self.bandwidth)
                
                if len(indices) == 0:
                    break
                
                # 计算加权重心
                neighbors = X[indices]
                distances = np.linalg.norm(neighbors - point, axis=1)
                weights = self._gaussian_kernel(distances)
                
                if weights.sum() > 0:
                    new_point = (weights[:, np.newaxis] * neighbors).sum(axis=0) / weights.sum()
                else:
                    new_point = point
                    
                # 检查收敛
                if np.linalg.norm(new_point - point) < self.eps:
                    break
                point = new_point
            
            # 判断是否是新的聚类中心
            is_new = True
            for center in cluster_centers:
                if np.linalg.norm(point - center) < self.bandwidth * 0.5:
                    is_new = False
                    break
            
            if is_new:
                cluster_centers.append(point)
            
            points[i] = point
            
            # 标记相近点
            for j in range(N):
                if not converged[j] and np.linalg.norm(points[j] - point) < self.bandwidth * 0.5:
                    converged[j] = True
        
        self.cluster_centers_ = np.array(cluster_centers)
        
        # 分配标签
        self.labels_ = np.zeros(N, dtype=int)
        for i, center in enumerate(self.cluster_centers_):
            distances = np.linalg.norm(X - center, axis=1)
            self.labels_[distances < self.bandwidth] = i
            
        return self
    
    def predict(self, X):
        """预测新点"""
        X = np.array(X)
        
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.cluster_centers_ - x, axis=1)
            labels[i] = np.argmin(distances)
            
        return labels


def demo():
    """演示"""
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=500, centers=4, 
                    cluster_std=0.8, random_state=42)
    
    ms = MeanShiftClustering(bandwidth=1.5)
    ms.fit(X)
    
    print(f"聚类数: {len(ms.cluster_centers_)}")
    print(f"标签分布: {np.bincount(ms.labels_)}")


if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

### 9.1 聚类可视化

```python
import matplotlib.pyplot as plt

def plot_mean_shift(X, labels, centers):
    """可视化"""
    
    plt.figure(figsize=(10, 8))
    
    # 画数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, 
                     cmap='viridis', alpha=0.6, s=20)
    
    # 画聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], c='red', 
               marker='x', s=200, linewidths=3)
    
    plt.colorbar(scatter)
    plt.title('Mean Shift聚类结果')
    plt.show()
```

### 9.2 收敛过程

```python
def plot_convergence(points_history):
    """展示收敛过程"""
    
    for i, points in enumerate(points_history[::10]):
        plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
        plt.title(f'Iteration {i*10}')
        plt.pause(0.1)
```

---

## 10. 模型评估

### 10.1 聚类质量

| 指标 | 说明 |
|------|------|
| Silhouette Score | 聚类质量 |
| Davies-Bouldin | 类内/类间距离 |

### 10.2 对比K-means

| 方法 | 准确率 | 稳定性 | 时间 |
|------|--------|--------|------|
| K-means | 取决于K | 敏感 | 快 |
| Mean Shift | 自动 | 鲁棒 | 慢 |

---

## 11. 常见问题与易错点

### 11.1 带宽选择

**问题**：h太大会合并聚类，h太小会过度分割

**解决**：
- 多次实验
- 用交叉验证

### 11.2 计算效率

**问题**：O(n²d)太慢

**解决**：
- KD-tree加速
- 采样

### 11.3 边界情况

**问题**：边界点可能不收敛

**解决**：
- 设置最大迭代
- 多起点

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 向密度高处迭代移动 |
| 核心 | m(x) = 加权重心 |
| 无需预设K | 自动发现 |
| 形状任意 | 无形状假设 |

### 12.2 公式

**漂移向量**：
$$m(x) = \frac{\sum x_i K(\|x-x_i\|^2)}{\sum K(\|x-x_i\|^2)} - x$$

### 12.3 扩展

- **Mean Shift++**：更好初始化
- **Selective**：加速

---

## 13. 练习题

### 13.1 基础

1. Mean Shift向量指向哪里？
2. 为什么不需要预设K？

### 13.2 进阶

1. 和K-means的区别？
2. h的影响？

---

## 14. 学习路径

1. 密度估计
2. 核函数
3. Mean Shift原理
4. 实现与调参

---

## 附录

### 参考

- 论文：Fukunaga 1975
- 库：OpenCV, sklearn

---

**文档结束**