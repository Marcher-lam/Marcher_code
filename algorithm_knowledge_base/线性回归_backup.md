# 线性回归 (Linear Regression) 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

线性回归是最基础但最重要的监督学习算法，用于建立连续目标变量与一个或多个特征之间的线性关系模型。它是理解所有机器学习算法的起点。

### 1.2 直觉类比

想象线性回归的工作方式就像根据年龄预测身高：如果知道"年龄每增加1岁，身高平均增加5厘米"这个规律，就可以根据年龄预测身高。线性回归就是从数据中学习这个"规律"的数学方法。

### 1.3 历史背景

线性回归有着悠久的历史：
- 1805年：Legendre提出最小二乘法
- 1809年：高斯独立提出并给出误差分析
- 19世纪：最小二乘法成为标准统计方法
- 20世纪：计算机统计的时代到来
- 2000年代：正则化回归成为特征选择的工具

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 算法类型 | 监督学习（回归） |
| 模型类型 | 线性模型 |
| 目标 | 预测连续值 |
| 输出 | $\hat{y} = w^T x + b$ |

### 1.5 前置知识

学习线性回归需要：
1. 基础代数（向量、矩阵运算）
2. 导数与微分
3. 概率基础（高斯分布）

---

## 2. 核心原理

### 2.1 核心思想

线性回归的核心思想是假设目标变量y与特征x之间存在线性关系，通过最小二乘法找到一条"最合适"的直线（或超平面）来拟合数据。

### 2.2 工作流程

给定训练集$T = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$，其中$x_i \in \mathbb{R}^d, y_i \in \mathbb{R}$：

1. 建立模型：$\hat{y} = w^T x + b$
2. 定义损失：$L = \sum_{i=1}^n (y_i - \hat{y}_i)^2$
3. 优化：找到使L最小的参数w, b

### 2.3 几何解释

在二维情况下，线性回归找到一条直线，使得所有数据点到这条直线的垂直距离的平方和最小。

### 2.4 向量形式

将b并入w，x增广1列：
$$\hat{y} = X \theta$$

其中$\theta = [w; b]$，$X$最后添加全1列。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| X | 特征矩阵 | (n, d) |
| y | 目标向量 | (n,) |
| w | 权重向量 | (d,) |
| b | 偏置 | 标量 |
| θ | 参数向量 | (d+1,) |
| θ* | 最优参数 | (d+1,) |
| n | 样本数 | 标量 |
| d | 特征数 | 标量 |

### 3.2 目标函数

**最小二乘损失**：
$$J(\theta) = \frac{1}{2n} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$$

向量形式：
$$J(\theta) = \frac{1}{2n} \|X\theta - y\|^2_2$$

$h_\theta(x)$是假设函数，即模型的预测。

### 3.3 解析解（闭式解）

对目标函数求导并令导数为0：

$$\frac{\partial J}{\partial \theta} = \frac{1}{n} X^T(X\theta - y) = 0$$

展开：
$$X^T X \theta = X^T y$$

**最优参数**：
$$\theta^* = (X^T X)^{-1} X^T y$$

这是正规方程（Normal Equation）。

### 3.4 正则化

**L2正则化（岭回归）**：
$$J(\theta) = \frac{1}{2n}\|y - X\theta\|^2_2 + \lambda\|\theta\|^2_2$$

解析解：
$$\theta^* = (X^T X + \lambda I)^{-1} X^T y$$

**L1正则化（LASSO）**：
$$J(\theta) = \frac{1}{2n}\|y - X\theta\|^2_2 + \lambda\|\theta\|_1$$

L1范数不可微，使用次梯度。

### 3.5 梯度下降

**批量梯度下降**：
$$\theta \leftarrow \theta - \eta \cdot \frac{1}{n} X^T(X\theta - y)$$

**随机梯度下降（SGD）**：
$$\theta \leftarrow \theta - \eta \cdot (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
def preprocess(X):
    """数据预处理"""
    # 检查缺失值
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    
    # 特征标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / (std + 1e-8)
    
    return X_norm, mean, std
```

### 4.2 参数初始化

```python
def initialize(d):
    """初始化参数"""
    return np.zeros(d)
```

### 4.3 训练循环

```python
def train(X, y, lr=0.01, n_iters=1000):
    """训练"""
    n, d = X.shape
    theta = initialize(d + 1)
    
    # 增广X
    X_b = np.c_[np.ones(n), X]
    
    for i in range(n_iters):
        prediction = X_b @ theta
        error = prediction - y
        gradient = X_b.T @ error / n
        theta -= lr * gradient
    
    return theta
```

### 4.4 超参数

| 超参数 | 作用 | 推荐范围 |
|--------|------|----------|
| learning_rate | 学习率 | 0.001-0.1 |
| n_iters | 迭代次数 | 1000-10000 |
| regularization | 正则化强度 | 0.001-1.0 |

---

## 5. 应用场景

### 5.1 典型应用

1. **预测问题**
   - 房价预测
   - 销售额预测
   - 股票价格预测

2. **趋势分析**
   - 经济增长趋势
   - 人口增长模型

3. **因果分析**
   - 广告投入效果
   - 政策影响评估

### 5.2 适用数据

- 特征与目标有线性关系
- 误差近似正态分布
- 样本数量足够

### 5.3 不适用数据

- 高度非线性关系
- 特征间强相关（多重共线性）
- 异常值多

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 简单 | 易于理解和实现 | - |
| 快速 | 参数少，训练快 | - |
| 可解释 | 系数有物理意义 | 特征独立 |
| 可扩展 | 可添加正则化 | - |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 线性假设 | 无法处理非线性 | 特征工程 |
| 对异常值敏感 | 最小二乘放大误差 | Huber损失 |
| 多重共线性 | 矩阵不可逆 | 岭回归/LASSO |

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 sklearn实现

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel:
    """线性回归模型"""
    
    def __init__(self, normalize=True, regularization='none', alpha=1.0):
        self.normalize = normalize
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.mean = None
        self.std = None
    
    def fit(self, X_train, y_train):
        """训练"""
        # 标准化
        if self.normalize:
            self.mean = np.mean(X_train, axis=0)
            self.std = np.std(X_train, axis=0)
            X_norm = (X_train - self.mean) / (self.std + 1e-8)
        else:
            X_norm = X_train
        
        # 选择模型
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_norm, y_train)
        return self
    
    def predict(self, X_test):
        """预测"""
        if self.normalize:
            X_norm = (X_test - self.mean) / (self.std + 1e-8)
        else:
            X_norm = X_test
        
        return self.model.predict(X_norm)
    
    def score(self, X_test, y_test):
        """评估"""
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
    
    def get_coefficients(self):
        """获取系数"""
        return self.model.coef_, self.model.intercept_


def demo():
    print("=== 线性回归 演示 ===\n")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    
    # 特征：面积、房龄、距离
    X = np.random.randn(n_samples, 3)
    # 真实关系: y = 5*面积 + 0.1*房龄 - 3*距离 + 噪声
    true_weights = np.array([5, 0.1, -3])
    y = X @ true_weights + np.random.randn(n_samples) * 0.5 + 10
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = LinearRegressionModel(regularization='ridge', alpha=0.1)
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 系数
    coef, intercept = model.get_coefficients()
    print(f"\n系数: {coef}")
    print(f"截距: {intercept}")


if __name__ == "__main__":
    demo()
```

### 7.2 统计模型实现（statsmodels）

```python
import statsmodels.api as sm

class StatsmodelsLinearRegression:
    """使用statsmodels的线性回归"""
    
    def __init__(self):
        self.model = None
        self.results = None
    
    def fit(self, X, y):
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合OLS模型
        self.model = sm.OLS(y, X_with_const)
        self.results = self.model.fit()
        
        return self
    
    def summary(self):
        return self.results.summary()
    
    def predict(self, X):
        X_with_const = sm.add_constant(X)
        return self.results.predict(X_with_const)
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 正规方程实现

```python
import numpy as np

class LinearRegressionScratch:
    """线性回归（手工实现）"""
    
    def __init__(self):
        self.theta = None
        self.mean = None
        self.std = None
    
    def fit(self, X, y):
        """使用正规方程训练"""
        # 数据预处理
        n, d = X.shape
        
        # 标准化
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_norm = (X - self.mean) / (self.std + 1e-8)
        
        # 增广矩阵（添加偏置列）
        X_b = np.c_[np.ones(n), X_norm]
        
        # 正规方程: theta = (X'X)^(-1) X'y
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        
        # 添加正则化防止奇异
        lambda_reg = 0.001
        XtX += lambda_reg * np.eye(d + 1)
        
        self.theta = np.linalg.solve(XtX, Xty)
        
        return self
    
    def predict(self, X):
        """预测"""
        # 标准化
        X_norm = (X - self.mean) / (self.std + 1e-8)
        
        # 增广
        n = X.shape[0]
        X_b = np.c_[np.ones(n), X_norm]
        
        return X_b @ self.theta
    
    def get_params(self):
        """获取参数"""
        return self.theta[1:], self.theta[0]
```

### 8.2 梯度下降实现

```python
class LinearRegressionGD:
    """梯度下降版线性回归"""
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None
    
    def fit(self, X, y):
        n, d = X.shape
        
        # 标准化
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_norm = (X - self.mean) / (self.std + 1e-8)
        
        # 增广
        X_b = np.c_[np.ones(n), X_norm]
        
        # 初始化
        self.theta = np.zeros(d + 1)
        
        # 梯度下降
        for _ in range(self.n_iters):
            grad = X_b.T @ (X_b @ self.theta - y) / n
            self.theta -= self.lr * grad
        
        return self
    
    def predict(self, X):
        X_norm = (X - self.mean) / (self.std + 1e-8)
        n = X.shape[0]
        X_b = np.c_[np.ones(n), X_norm]
        return X_b @ self.theta


def demo_scratch():
    print("=== 线性回归 手工实现演示 ===\n")
    
    # 测试数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3*X[:, 0] + 2*X[:, 1] + 1 + np.random.randn(100)*0.1
    
    # 训练
    model = LinearRegressionScratch()
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    
    # 评估
    mse = np.mean((y - y_pred)**2)
    print(f"MSE: {mse:.6f}")
    
    # 参数
    weights, bias = model.get_params()
    print(f"权重: {weights}")
    print(f"偏置: {bias}")


if __name__ == "__main__":
    demo_scratch()
```

---

## 9. 可视化与结果理解

### 9.1 一维可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_1d():
    """一维线性回归可视化"""
    np.random.seed(42)
    
    # 生成数据
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2*X.flatten() + 1 + np.random.randn(100)*0.5
    
    # 训练（使用sklearn）
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_pred, 'r-', linewidth=2, label='Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression (1D)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_1d.png', dpi=150)
    plt.show()


def plot_residuals():
    """残差图"""
    np.random.seed(42)
    
    x = np.linspace(0, 10, 100)
    y = 2*x + 1 + np.random.randn(100)*0.5
    y_pred = 2*x + 1
    residuals = y - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('residuals.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_1d()
    plot_residuals()
```

---

## 10. 模型评估

### 10.1 评估指标

**回归指标**：

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 均方误差 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | 平均绝对误差 |
| R² | $1-\frac{SS_{res}}{SS_{tot}}$ | 决定系数 |

### 10.2 性能对比

```
Boston Housing数据集:

方法              RMSE      R²
----------------------------------
LinearRegression  4.90      0.74
Ridge(α=1.0)     4.91      0.74
Lasso(α=1.0)     5.12      0.71
ElasticNet       4.88      0.75
```

---

## 11. 常见问题与易错点

### 11.1 多重共线性

**问题**：特征间高度相关，矩阵奇异

**原因**：$X^T X$接近奇异矩阵，不可逆

**解决方案**：使用岭回归（L2正则化）

### 11.2 特征尺度不同

**问题**：梯度下降不收敛

**原因**：特征尺度差异大

**解决方案**：特征标准化

### 11.3 异��值��感

**问题**：MSE放大异常值影响

**原因**：平方项放大误差

**解决方案**：使用Huber损失或MAE

---

## 12. 学习总结

### 核心要点

1. 线性回归是最基础的机器学习算法
2. 最小二乘目标 + 闭式解
3. 可加入L1/L2正则化
4. 预测连续值任务的首选

### 从线性回归到其他算法

Linear Regression → Ridge → LASSO → ElasticNet → Logistic Regression → Neural Networks

---

## 13. 练习题与思考题（含答案）

### 练习题1：基础计算

**问题**：给定X = [1, 2, 3]，y = [2, 4, 6]，求线性回归参数

**答案**：y = 2x，通过原点，参数w=2, b=0

### 练习题2：梯度推导

**问题**：推导批量梯度下降更新公式

答案见第3节

### 练习题3：编程实践

**问题**：实现带正则化的线性回归

答案见第8节

---

## 14. 学习路径建议

### 初级阶段（1周）

1. 理解最小二乘
2. 掌握正规方程
3. 实现基础版本

### 中级阶段（1周）

1. 学习梯度下降
2. 掌握正则化
3. 调参与优化

### 高级阶段（1周）

1. 理解收敛分析
2. 学习更复杂变体
3. 实际应用

### 推荐资源

- Bishop: Pattern Recognition and Machine Learning
- Hastie: The Elements of Statistical Learning

---

**文档结束**