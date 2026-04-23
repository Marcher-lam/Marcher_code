# ALS (交替最小二乘) 学习文档

> 推荐系统中广泛使用的矩阵分解算法，高效处理稀疏评分数据。

---

## 1. 算法基础认知

### 1.1 发展背景

ALS（Alternating Least Squares，交替最小二乘）是一种经典的矩阵分解算法，最早应用于推荐系统中的协同过滤。由 Netflix Prize 竞赛（2006-2009）而广为人知，是处理大规模稀疏评分矩阵的标准方法。

### 1.2 核心定位

| 特性 | 描述 |
|------|------|
| 类型 | 矩阵分解/协同过滤 |
| 输入 | 用户-物品评分矩阵（稀疏） |
| 输出 | 用户隐向量 + 物品隐向量 |
| 优点 | 可并行、收敛快、处理稀疏 |

### 1.3 应用场景

- 推荐系统：预测用户对物品的评分
- 隐因子模型：学习用户和物品的隐向量表示
- 降维技术：将高维稀疏向量投影到低维稠密空间

---

## 2. 核心原理

### 2.1 矩阵分解问题

给定用户-物品评分矩阵 $R \in \mathbb{R}^{m \times n}$（$m$ 用户，$n$ 物品），目标是分解为两个低秩矩阵：

$$R \approx U \times V^T$$

其中：
- $U \in \mathbb{R}^{m \times k}$：用户隐向量矩阵（$k$ 为隐因子维度）
- $V \in \mathbb{R}^{n \times k}$：物品隐向量矩阵

### 2.2 损失函数

$$\min_{U,V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i \cdot V_j^T)^2 + \lambda (\|U\|^2_F + \|V\|^2_F)$$

其中：
- $\Omega$ 为有评分的 (i, j) 对集合
- $\lambda$ 为正则化系数

### 2.3 交替优化

ALS 的核心思想是**交替固定**：

1. **固定 V，求 U**：当 $V$ 固定时，问题变为求解 $m$ 个独立的线性回归
2. **固定 U，求 V**：当 $U$ 固定时，问题变为求解 $n$ 个独立的线性回归
3. 交替迭代，直到收敛

---

## 3. 数学公式与推导

### 3.1 优化用户矩阵 $U$

当 $V$ 固定时，最小化：

$$\min_U \sum_{j \in I_i} (R_{ij} - U_i V_j^T)^2 + \lambda \|U_i\|^2$$

其中 $I_i$ 为用户 $i$ 评过分的物品集合。

令 $\frac{\partial L}{\partial U_i} = 0$：

$$U_i = (V_{I_i}^T V_{I_i} + \lambda I)^{-1} V_{I_i}^T R_{i,I_i}$$

其中 $V_{I_i}$ 为 $V$ 的行子集（用户 $i$ 评分过的物品）。

### 3.2 优化物品矩阵 $V$

当 $U$ 固定时：

$$V_j = (U_{U_j}^T U_{U_j} + \lambda I)^{-1} U_{U_j}^T R_{U_j,j}$$

其中 $U_j$ 为评分过物品 $j$ 的用户集合。

### 3.3 算法流程

```
Input: R, k, lambda, max_iter
Output: U, V

1. 随机初始化 U, V
2. for iter in range(max_iter):
3.     for i in range(m):
4.         更新 U[i]:
5.         U[i] = (V[I_i]^T V[I_i] + lambda*I)^{-1} V[I_i]^T R[i, I_i]
6.     for j in range(n):
7.         更新 V[j]:
8.         V[j] = (U[U_j]^T U[U_j] + lambda*I)^{-1} U[U_j]^T R[U_j, j]
9. return U, V
```

### 3.4 收敛性证明

ALS 的目标函数非单调递减，每一步交替优化都是凸优化，因此保证收敛。

---

## 4. 训练过程讲解

### 4.1 参数设置

| 参数 | 说明 | 典型值 |
|------|------|--------|
| k | 隐因子维度 | 10-200 |
| lambda | 正则化系数 | 0.01-0.1 |
| max_iter | 最大迭代次数 | 10-30 |
| tolerance | 收敛阈值 | 1e-4 |

### 4.2 初始化策略

1. **随机初始化**：$U, V \sim N(0, 0.1)$
2. **SVD 初始化**：对密集子矩阵做 SVD
3. **均值初始化**：使用全局均值

### 4.3 并行化

ALS 可以高效并行化：

- 每个用户 $i$ 的 $U_i$ 更新独立
- 每个物品 $j$ 的 $V_j$ 更新独立
- 适合 MapReduce 或 Spark

### 4.4 稀疏矩阵处理

对于稀疏矩阵 $R$，只需要存储非零元素：

```python
# COO 格式
coordinates = [(i, j, rating), ...]
```

更新时使用稀疏矩阵乘法加速。

---

## 5. 应用场景

### 5.1 推荐系统

- **电影推荐**：MovieLens 数据集
- **商品推荐**：电商平台
- **音乐推荐**：Spotify 等

### 5.2 代码示例

```python
import numpy as np
from scipy.sparse import csr_matrix

def als_recommend(R, k=50, lambda_=0.01, max_iter=20):
    """ALS 推荐算法"""
    
    m, n = R.shape
    
    # 初始化
    np.random.seed(42)
    U = np.random.randn(m, k) * 0.1
    V = np.random.randn(n, k) * 0.1
    
    # 迭代
    for iter in range(max_iter):
        # 更新 U
        for i in range(m):
            rated_items = np.where(R[i] > 0)[0]
            if len(rated_items) == 0:
                continue
            V_rated = V[rated_items]
            R_rated = R[i, rated_items]
            U[i] = np.linalg.solve(
                V_rated.T @ V_rated + lambda_ * np.eye(k),
                V_rated.T @ R_rated
            )
        
        # 更新 V
        for j in range(n):
            rating_users = np.where(R[:, j] > 0)[0]
            if len(rating_users) == 0:
                continue
            U_rated = U[rating_users]
            R_rated = R[rating_users, j]
            V[j] = np.linalg.solve(
                U_rated.T @ U_rated + lambda_ * np.eye(k),
                U_rated.T @ R_rated
            )
    
    return U, V
```

---

## 6. 优缺点分析

### 6.1 优点

1. **可并行**：用户/物品更新完全独立
2. **收敛快**：交替凸优化保证收敛
3. **稀疏友好**：只需要处理非零元素
4. **可扩展**：适合大规模数据

### 6.2 缺点

1. **隐因子假设**：假设低秩结构
2. **初始化敏感**：不同初始化结果不同
3. **只处理评分**：不处理隐式反馈

### 6.3 改进方向

- **BPR-ALS**：处理隐式反馈
- **SVD++**：加入隐式因子
- **神经ALS**：用神经网络替代线性模型

---

## 7. 调库实现

### 7.1 Scipy 实现

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class ALS:
    """交替最小二乘"""
    
    def __init__(self, n_factors=50, regularization=0.01, max_iter=20):
        self.n_factors = n_factors
        self.regularization = regularization
        self.max_iter = max_iter
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, R):
        """训练模型"""
        R = csr_matrix(R)
        m, n = R.shape
        
        # 稀疏矩阵的行索引和值
        row, col = R.nonzero()
        ratings = R.data
        
        # 初始化
        np.random.seed(42)
        self.user_factors = np.random.randn(m, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n, self.n_factors) * 0.1
        
        # 迭代
        for it in range(self.max_iter):
            # 更新用户
            for i in range(m):
                mask = row == i
                if mask.sum() == 0:
                    continue
                items = col[mask]
                V = self.item_factors[items]
                r = ratings[mask]
                A = V.T @ V + self.regularization * np.eye(self.n_factors)
                b = V.T @ r
                self.user_factors[i] = np.linalg.solve(A, b)
            
            # 更新物品
            for j in range(n):
                mask = col == j
                if mask.sum() == 0:
                    continue
                users = row[mask]
                U = self.user_factors[users]
                r = ratings[mask]
                A = U.T @ U + self.regularization * np.eye(self.n_factors)
                b = U.T @ r
                self.item_factors[j] = np.linalg.solve(A, b)
                
        return self
    
    def predict(self, user, item):
        """预测评分"""
        return np.dot(self.user_factors[user], self.item_factors[item])
    
    def recommend(self, user, n=10):
        """推荐 TOP-N"""
        scores = self.user_factors @ self.item_factors.T
        return np.argsort(scores[user])[::-1][:n]


# 示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    m, n = 100, 50
    R = np.random.randn(m, n)
    R = (R > 2).astype(float) * 5  # 稀疏二值化
    
    # 训练
    als = ALS(n_factors=20, regularization=0.1, max_iter=10)
    als.fit(R)
    
    # 推荐
    top_items = als.recommend(0, n=5)
    print(f"用户 0 的 TOP-5 推荐: {top_items}")
```

### 7.2 Implicit 库（处理隐式反馈）

```python
# pip install implicit
import implicit
from scipy.sparse import csr_matrix

# 准备数据（隐式反馈）
user_item = csr_matrix((np.ones(len(data)), (users, items)))

# 训练 ALS
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=20
)
model.fit(user_item)

# 预测
item_ids = model.recommend(user_id, user_item, N=10)
```

---

## 8. 手工代码实现

### 8.1 完整 ALS 实现

```python
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class ALSMatrix:
    """ALS 矩阵分解完整实现"""
    
    def __init__(self, n_factors=50, regularization=0.01, 
                 max_iter=20, tolerance=1e-4):
        self.n_factors = n_factors
        self.regularization = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, R, verbose=True):
        """
        训练 ALS 模型
        
        参数:
            R: 评分矩阵 (m x n)，0 表示无评分
        """
        R = np.array(R)
        m, n = R.shape
        
        # 构建用户-物品索引
        self.user_rated = {}
        self.item_rated = {}
        
        for i in range(m):
            self.user_rated[i] = np.where(R[i] > 0)[0]
        for j in range(n):
            self.item_rated[j] = np.where(R[:, j] > 0)[0]
        
        # 初始化因子矩阵
        np.random.seed(42)
        self.user_factors = np.random.randn(m, self.n_factors) * 0.01
        self.item_factors = np.random.randn(n, self.n_factors) * 0.01
        
        # 迭代
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # 更新用户因子
            for i in range(m):
                items = self.user_rated[i]
                if len(items) == 0:
                    continue
                    
                V = self.item_factors[items]
                r = R[i, items]
                
                A = V.T @ V + self.regularization * np.eye(self.n_factors)
                b = V.T @ r
                self.user_factors[i] = np.linalg.solve(A, b)
            
            # 更新物品因子
            for j in range(n):
                users = self.item_rated[j]
                if len(users) == 0:
                    continue
                    
                U = self.user_factors[users]
                r = R[users, j]
                
                A = U.T @ U + self.regularization * np.eye(self.n_factors)
                b = U.T @ r
                self.item_factors[j] = np.linalg.solve(A, b)
            
            # 计算损失
            loss = self._compute_loss(R)
            improvement = prev_loss - loss
            
            if verbose:
                print(f"Iter {iteration+1}: Loss = {loss:.4f}, 改善 = {improvement:.4f}")
            
            if improvement < self.tolerance:
                break
                
            prev_loss = loss
            
        return self
    
    def _compute_loss(self, R):
        """计算 RMSE + 正则化损失"""
        m, n = R.shape
        error = 0
        count = 0
        
        for i in range(m):
            items = self.user_rated[i]
            if len(items) == 0:
                continue
            pred = self.user_factors[i] @ self.item_factors[items].T
            error += np.sum((R[i, items] - pred) ** 2)
            count += len(items)
        
        rmse = np.sqrt(error / count)
        reg = self.regularization * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2)
        ) / 2
        
        return rmse + reg
    
    def predict(self, user, item):
        """预测单个评分"""
        if user >= len(self.user_factors) or item >= len(self.item_factors):
            return 0
        return np.dot(self.user_factors[user], self.item_factors[item])
    
    def predict_matrix(self):
        """预测整个评分矩阵"""
        return self.user_factors @ self.item_factors.T
    
    def recommend(self, user, n=10, exclude_known=True):
        """为用户推荐物品"""
        scores = self.user_factors[user] @ self.item_factors.T
        
        # 排除已评分
        if exclude_known and user in self.user_rated:
            scores[self.user_rated[user]] = -np.inf
        
        # 返回 top-n
        top_items = np.argsort(scores)[::-1][:n]
        return [(j, scores[j]) for j in top_items]


def demo():
    """演示 ALS"""
    print("=== ALS 交替最小二乘演示 ===\n")
    
    # 生成模拟评分数据
    np.random.seed(42)
    n_users, n_items = 100, 50
    k = 10  # 真实隐因子维度
    
    # 生成真实因子
    U_true = np.random.randn(n_users, k) * 3
    V_true = np.random.randn(n_items, k) * 3
    
    # 生成评分矩阵
    R_true = U_true @ V_true.T
    R = R_true + np.random.randn(n_users, n_items) * 2
    
    # 稀疏化（只保留 10% 评分）
    mask = np.random.rand(n_users, n_items) > 0.9
    R_sparse = R * mask
    
    print(f"原始维度: {n_users} x {n_items}")
    print(f"非零评分: {np.sum(R_sparse > 0)}")
    
    # 训练 ALS
    als = ALSMatrix(n_factors=20, regularization=0.1, max_iter=15)
    als.fit(R_sparse)
    
    # 预测
    pred_matrix = als.predict_matrix()
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean((R_true - pred_matrix) ** 2))
    print(f"\n预测 RMSE: {rmse:.4f}")
    
    # 推荐
    recommendations = als.recommend(0, n=5)
    print(f"\n用户 0 的 TOP-5 推荐:")
    for item, score in recommendations:
        print(f"  物品 {item}: {score:.4f}")


if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

### 9.1 收敛曲线

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence():
    """绘制收敛曲线"""
    iterations = range(1, 16)
    losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.4, 1.35, 1.3, 1.28, 1.26, 
             1.25, 1.24, 1.23, 1.23, 1.23]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('迭代次数')
    plt.ylabel('损失 (RMSE + 正则化)')
    plt.title('ALS 收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('als_convergence.png', dpi=150)
    plt.show()


def plot_factors():
    """可视化隐因子"""
    np.random.seed(42)
    factors = np.random.randn(50, 20)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(factors, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='因子值')
    plt.xlabel('隐因子维度')
    plt.ylabel('物品')
    plt.title('物品隐因子矩阵热图')
    plt.tight_layout()
    plt.savefig('als_factors.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_als(R_true, R_pred, test_mask):
    """评估 ALS 模型"""
    # RMSE
    rmse = np.sqrt(mean_squared_error(R_true[test_mask], R_pred[test_mask]))
    
    # MAE
    mae = mean_absolute_error(R_true[test_mask], R_pred[test_mask])
    
    # Precision@K
    def precision_at_k(R_true, R_pred, k=10):
        precisions = []
        for user in range(R_true.shape[0]):
            true_top = set(np.argsort(R_true[user])[::-1][:k])
            pred_top = set(np.argsort(R_pred[user])[::-1][:k])
            precisions.append(len(true_top & pred_top) / k)
        return np.mean(precisions)
    
    p_at_10 = precision_at_k(R_true, R_pred, k=10)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': p_at_10
    }
```

### 10.2 常用数据集

| 数据集 | 用户数 | 物品数 | 评分数 | 稀疏度 |
|--------|--------|--------|--------|--------|
| MovieLens | 600 | 9000 | 10万 | 1.8% |
| Netflix | 48万 | 17万 | 1亿 | 0.1% |
| Yelp | 1M | 1M | 5M | 0.1% |

---

## 11. 常见问题与易错点

### 11.1 隐因子维度选择

**问题**：如何选择合适的 $k$？

**解答**：
- $k$ 太小：欠拟合
- $k$ 太大：过拟合，时间增加
- 经验：$k$ 取 10-200，通过验证集选择

### 11.2 稀疏数据问题

**问题**：冷启动问题

**解答**：
1. 使用内容特征初始化
2. 混合协同过滤和内容过滤
3. 加入正则化

### 11.3 数值稳定性

**问题**：矩阵奇异

**解答**：
1. 增加正则化 $\lambda$
2. 使用伪逆
3. 抖动（Jitter）

---

## 12. 学习总结

**核心要点**：

1. **交替优化**：固定 $U$ 求 $V$，固定 $V$ 求 $U$
2. **闭式解**：每步为岭回归
3. **稀疏处理**：只更新有评分的用户/物品
4. **可扩展**：适合大规模推荐系统

**学习建议**：

1. 理解矩阵分解的动机
2. 推导交替优化的闭式解
3. 在实际数据集上实验

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 推导 ALS 的更新公式
2. 手动实现 ALS 并在模拟数据上验证
3. 比较不同 $k$ 对 RMSE 的影响

### 13.2 进阶练习

1. 加入偏置项的 ALS 实现
2. 隐式反馈的 ALS 实现
3. 结合时间衰减的 ALS

### 13.3 思考题

1. ALS 与 SGD 的区别？
2. 如何处理大规模稀疏数据？

---

### 13.4 详细答案与解析

#### 练习1：更新公式推导

**问题**：推导固定 $V$ 时 $U$ 的更新公式。

**损失函数**：
$$L = \sum_{j \in I_i} (R_{ij} - U_i V_j^T)^2 + \lambda \|U_i\|^2$$

**求导**：
$$\frac{\partial L}{\partial U_i} = -2 \sum_j V_j (R_{ij} - U_i V_j^T) + 2\lambda U_i = 0$$

整理：
$$\sum_j V_j V_j^T U_i^T - \sum_j V_j R_{ij} + \lambda U_i = 0$$

$$U_i (\sum_j V_j V_j^T + \lambda I) = \sum_j V_j R_{ij}$$

$$U_i = (\sum_j V_j R_{ij}) (\sum_j V_j V_j^T + \lambda I)^{-1}$$

写成矩阵形式：
$$U_i = (V_{I_i}^T R_{i,I_i}) (V_{I_i}^T V_{I_i} + \lambda I)^{-1}$$

这正是岭回归的闭式解。

#### 练习2：与 SGD 对比

** ALS vs SGD**：

| 特性 | ALS | SGD |
|------|-----|-----|
| 收敛速度 | 快 | 慢 |
| 并行性 | 高 | 低 |
| 内存 | 高 | 低 |
| 调参 | 易 | 难 |

ALS 适合稠密数据或可并行环境，SGD 适合超大规模数据。

---

## 14. 学习路径建议

### 入门阶段

1. 了解协同过滤基本思想
2. 学习矩阵分解
3. 掌握 ALS 推导

### 进阶阶段

1. 实现完整 ALS
2. 加入偏置和隐式反馈
3. 在真实数据集上实验

### 高级阶段

1. 分布式 ALS 实现
2. 神经协同过滤
3. 图神经网络推荐

**推荐路线**：

```
协同过滤 → SVD → ALS → BPR → NeuMF → LightGCN
```

**ALS 是推荐系统的基础，掌握它是进入推荐领域的必经之路。**