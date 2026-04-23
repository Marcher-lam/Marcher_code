# 优化方法详解

> 理解优化是理解机器学习的关键——几乎所有机器学习问题最终都是优化问题。

---

## 1. 优化问题基础

### 1.1 什么是优化问题

在机器学习中，我们的目标是找到一组参数 $\theta$，使得损失函数 $L(\theta)$ 最小：

$$\theta^* = \arg\min_{\theta} L(\theta)$$

**核心要素**：
- **目标函数（损失函数）**：衡量模型好坏的标准
- **参数**：需要优化的变量
- **约束**（可选）：参数的取值范围

### 1.2 为什么优化很重要

```python
# 机器学习 = 表达 + 优化
# 表达：选择模型形式（如 y = wx + b）
# 优化：找到最好的参数 w 和 b

import numpy as np
import matplotlib.pyplot as plt

# 损失函数曲面示例
w = np.linspace(-2, 4, 100)
b = np.linspace(-2, 4, 100)
W, B = np.meshgrid(w, b)

# 假设的损失函数
L = (W - 1)**2 + (B - 2)**2

plt.contour(W, B, L, levels=20)
plt.colorbar(label='Loss')
plt.xlabel('w')
plt.ylabel('b')
plt.title('损失函数等高线图')
plt.plot(1, 2, 'r*', markersize=15, label='最优解')
plt.legend()
plt.show()
```

---

## 2. 梯度下降法

### 2.1 核心思想

**直觉理解**：想象你被困在山上，四周大雾弥漫。你想下山，但看不见路。最好的策略是什么？

**答案**：感受脚下的坡度，朝着最陡峭的方向向下走。

**数学表达**：
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

其中：
- $\theta$：参数
- $\eta$：学习率（步长）
- $\nabla L$：损失函数的梯度

### 2.2 梯度的意义

**梯度**指向函数增长最快的方向，**负梯度**指向函数下降最快的方向。

```python
import numpy as np

def loss_function(x):
    """示例：f(x) = x^2"""
    return x ** 2

def gradient(x):
    """梯度：f'(x) = 2x"""
    return 2 * x

# 梯度下降
x = 10  # 初始点
learning_rate = 0.1
history = [x]

for i in range(20):
    grad = gradient(x)
    x = x - learning_rate * grad
    history.append(x)
    print(f"迭代 {i+1}: x = {x:.4f}, loss = {loss_function(x):.4f}")
```

### 2.3 梯度下降的变体

#### 批量梯度下降（BGD）

使用**全部数据**计算梯度。

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        # 使用所有样本计算梯度
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradient

    return theta
```

**优点**：收敛稳定
**缺点**：慢，内存消耗大

#### 随机梯度下降（SGD）

每次使用**一个样本**计算梯度。

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        for i in range(m):
            # 使用单个样本
            random_idx = np.random.randint(m)
            xi = X[random_idx:random_idx+1]
            yi = y[random_idx:random_idx+1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradient.flatten()

    return theta
```

**优点**：快，能跳出局部最优
**缺点**：收敛不稳定

#### 小批量梯度下降（Mini-batch GD）

使用**一小批样本**计算梯度。**这是最常用的方法！**

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradient = (1/len(Xi)) * Xi.T.dot(Xi.dot(theta) - yi)
            theta = theta - learning_rate * gradient

    return theta
```

**优点**：兼顾速度和稳定性

### 2.4 梯度下降可视化

```python
import numpy as np
import matplotlib.pyplot as plt

# 损失函数 f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2*x

# 不同方法的收敛过程
x_bgd = 5
x_sgd = 5
x_mini = 5

history_bgd = [x_bgd]
history_sgd = [x_sgd]
history_mini = [x_mini]

lr = 0.1

for _ in range(20):
    x_bgd = x_bgd - lr * df(x_bgd)
    history_bgd.append(x_bgd)

for _ in range(20):
    # SGD添加噪声模拟随机性
    noise = np.random.randn() * 0.5
    x_sgd = x_sgd - lr * (df(x_sgd) + noise)
    history_sgd.append(x_sgd)

for _ in range(20):
    # Mini-batch的噪声较小
    noise = np.random.randn() * 0.2
    x_mini = x_mini - lr * (df(x_mini) + noise)
    history_mini.append(x_mini)

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(history_bgd, 'o-')
plt.title('Batch GD')
plt.xlabel('Iteration')
plt.ylabel('x')

plt.subplot(132)
plt.plot(history_sgd, 'o-')
plt.title('SGD')
plt.xlabel('Iteration')

plt.subplot(133)
plt.plot(history_mini, 'o-')
plt.title('Mini-batch GD')
plt.xlabel('Iteration')

plt.tight_layout()
plt.show()
```

---

## 3. 学习率

### 3.1 学习率的作用

学习率 $\eta$ 控制每一步更新的步长。

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### 3.2 学习率的影响

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def df(x):
    return 2*x

learning_rates = [0.01, 0.1, 0.9, 1.1]
x_init = 5

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, lr in enumerate(learning_rates):
    x = x_init
    history = [x]

    for _ in range(20):
        x = x - lr * df(x)
        history.append(x)

    # 绘制损失函数
    x_range = np.linspace(-10, 10, 100)
    axes[idx].plot(x_range, f(x_range), 'b-', label='f(x)')
    axes[idx].plot(history, [f(h) for h in history], 'ro-', label='Path')
    axes[idx].set_title(f'LR = {lr}')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

**三种情况**：
- **学习率太小**（0.01）：收敛太慢
- **学习率合适**（0.1）：快速收敛
- **学习率太大**（0.9）：震荡
- **学习率过大**（1.1+）：发散

### 3.3 学习率调度

#### 学习率衰减

```python
def learning_rate_decay(initial_lr, epoch, decay_rate=0.95):
    """指数衰减"""
    return initial_lr * (decay_rate ** epoch)

# 使用示例
initial_lr = 0.1
for epoch in range(10):
    lr = learning_rate_decay(initial_lr, epoch)
    print(f"Epoch {epoch}: LR = {lr:.4f}")
```

#### 常用调度策略

```python
import numpy as np
import matplotlib.pyplot as plt

epochs = 100

# 1. 阶梯衰减
def step_decay(epoch, initial_lr=0.1, drop_rate=0.5, epochs_drop=30):
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

# 2. 指数衰减
def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

# 3. 余弦退火
def cosine_annealing(epoch, initial_lr=0.1, T_max=100):
    return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2

# 可视化
plt.figure(figsize=(12, 4))

lrs_step = [step_decay(e) for e in range(epochs)]
lrs_exp = [exponential_decay(e) for e in range(epochs)]
lrs_cos = [cosine_annealing(e) for e in range(epochs)]

plt.plot(range(epochs), lrs_step, label='Step Decay')
plt.plot(range(epochs), lrs_exp, label='Exponential Decay')
plt.plot(range(epochs), lrs_cos, label='Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 4. 高级优化器

### 4.1 动量法（Momentum）

**思想**：模拟物理中的惯性，积累过去的梯度。

$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.lr * grads
        params = params + self.velocity
        return params
```

**优点**：
- 加速收敛
- 抑制震荡

### 4.2 AdaGrad

**思想**：自适应学习率，频繁更新的参数用小学习率。

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

```python
class AdaGradOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.G = None

    def update(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)

        self.G += grads ** 2
        params = params - self.lr / (np.sqrt(self.G) + self.epsilon) * grads
        return params
```

**缺点**：学习率单调递减，可能过早停止

### 4.3 RMSprop

**思想**：使用指数加权移动平均解决AdaGrad的学习率衰减问题。

```python
class RMSpropOptimizer:
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay_rate
        self.epsilon = epsilon
        self.G = None

    def update(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)

        self.G = self.decay * self.G + (1 - self.decay) * grads ** 2
        params = params - self.lr / (np.sqrt(self.G) + self.epsilon) * grads
        return params
```

### 4.4 Adam（最常用）

**思想**：结合Momentum和RMSprop的优点。

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params = params - self.lr / (np.sqrt(v_hat) + self.epsilon) * m_hat
        return params
```

### 4.5 优化器对比

```python
import numpy as np
import matplotlib.pyplot as plt

# 测试函数：Rosenbrock函数
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# 模拟不同优化器的路径
def simulate_optimizer(optimizer_class, x_init, y_init, **kwargs):
    optimizer = optimizer_class(**kwargs)
    x, y = x_init, y_init
    path = [(x, y)]

    for _ in range(1000):
        grads = rosenbrock_grad(x, y)
        params = np.array([x, y])
        params = optimizer.update(params, grads)
        x, y = params
        path.append((x, y))

        if np.sqrt(grads[0]**2 + grads[1]**2) < 1e-6:
            break

    return np.array(path)

# 可视化
plt.figure(figsize=(10, 8))

# 绘制等高线
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.5)

# 绘制优化路径（示例）
plt.plot(1, 1, 'r*', markersize=15, label='最优解')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock函数优化')
plt.legend()
plt.show()
```

---

## 5. 正则化

### 5.1 为什么需要正则化

**过拟合**：模型在训练集上表现很好，但在测试集上表现差。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# 生成数据
np.random.seed(42)
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * X).flatten() + np.random.randn(20) * 0.1

# 测试数据
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = np.sin(2 * np.pi * X_test).flatten()

# 高次多项式拟合（过拟合）
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

model_no_reg = LinearRegression()
model_no_reg.fit(X_poly, y)

model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X_poly, y)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X, y, c='b', s=50, label='训练数据')
plt.plot(X_test, y_test, 'g--', label='真实函数')
plt.plot(X_test, model_no_reg.predict(X_test_poly), 'r-', label='无正则化')
plt.ylim(-1.5, 1.5)
plt.title('过拟合（无正则化）')
plt.legend()

plt.subplot(122)
plt.scatter(X, y, c='b', s=50, label='训练数据')
plt.plot(X_test, y_test, 'g--', label='真实函数')
plt.plot(X_test, model_ridge.predict(X_test_poly), 'r-', label='Ridge正则化')
plt.ylim(-1.5, 1.5)
plt.title('正则化效果')
plt.legend()

plt.tight_layout()
plt.show()
```

### 5.2 L2正则化（Ridge）

**损失函数**：
$$L_{ridge} = L_{original} + \lambda \sum_{j=1}^{p} \theta_j^2$$

**理解**：
- 惩罚参数的平方和
- 使参数趋向于较小的值，但不为零
- 也叫权重衰减（Weight Decay）

```python
from sklearn.linear_model import Ridge

# L2正则化
ridge = Ridge(alpha=1.0)  # alpha就是lambda
ridge.fit(X_train, y_train)
```

**梯度**：
$$\frac{\partial L_{ridge}}{\partial \theta_j} = \frac{\partial L_{original}}{\partial \theta_j} + 2\lambda\theta_j$$

### 5.3 L1正则化（Lasso）

**损失函数**：
$$L_{lasso} = L_{original} + \lambda \sum_{j=1}^{p} |\theta_j|$$

**理解**：
- 惩罚参数的绝对值和
- 使部分参数恰好为零
- 具有特征选择作用

```python
from sklearn.linear_model import Lasso

# L1正则化
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 查看哪些特征被选中
print("非零参数:", np.sum(lasso.coef_ != 0))
```

### 5.4 L1 vs L2 对比

```python
import numpy as np
import matplotlib.pyplot as plt

# L1和L2的约束区域
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# L2约束区域（圆形）
theta1 = np.linspace(-1, 1, 100)
theta2_L2 = np.sqrt(1 - theta1**2)
axes[0].plot(theta1, theta2_L2, 'b')
axes[0].plot(theta1, -theta2_L2, 'b')
axes[0].set_xlabel('θ1')
axes[0].set_ylabel('θ2')
axes[0].set_title('L2 正则化约束区域')
axes[0].set_aspect('equal')
axes[0].grid(True)

# L1约束区域（菱形）
theta1_diamond = [0, 1, 0, -1, 0]
theta2_diamond = [1, 0, -1, 0, 1]
axes[1].plot(theta1_diamond, theta2_diamond, 'r')
axes[1].set_xlabel('θ1')
axes[1].set_ylabel('θ2')
axes[1].set_title('L1 正则化约束区域')
axes[1].set_aspect('equal')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

| 特性 | L1 (Lasso) | L2 (Ridge) |
|------|------------|------------|
| 约束形状 | 菱形 | 圆形 |
| 稀疏性 | 产生稀疏解 | 参数趋小但非零 |
| 特征选择 | 是 | 否 |
| 求解难度 | 不可导，需特殊方法 | 可导，易优化 |
| 适用场景 | 特征选择 | 一般正则化 |

### 5.5 Elastic Net

**损失函数**：
$$L_{elastic} = L_{original} + \lambda_1 \sum_{j=1}^{p} |\theta_j| + \lambda_2 \sum_{j=1}^{p} \theta_j^2$$

结合L1和L2的优点。

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio控制L1/L2比例
elastic.fit(X_train, y_train)
```

---

## 6. 过拟合 vs 欠拟合

### 6.1 定义

| 问题 | 表现 | 原因 |
|------|------|------|
| 欠拟合 | 训练集和测试集都差 | 模型太简单 |
| 过拟合 | 训练集好，测试集差 | 模型太复杂 |

### 6.2 诊断方法

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
# plot_learning_curve(model, X, y)
```

### 6.3 解决方案

**欠拟合**：
- 增加模型复杂度
- 添加更多特征
- 减小正则化强度

**过拟合**：
- 增加训练数据
- 减少特征数量
- 增加正则化强度
- 使用Dropout（神经网络）
- Early Stopping

---

## 7. 总结

### 7.1 优化器选择指南

| 场景 | 推荐优化器 |
|------|------------|
| 默认选择 | Adam |
| 简单凸问题 | SGD + Momentum |
| 需要精细调参 | SGD + Momentum + 学习率调度 |
| 稀疏数据 | AdaGrad |
| RNN/LSTM | RMSprop 或 Adam |

### 7.2 正则化选择指南

| 场景 | 推荐正则化 |
|------|------------|
| 默认选择 | L2 (Ridge) |
| 需要特征选择 | L1 (Lasso) |
| 特征相关 | Elastic Net |
| 深度学习 | Dropout + L2 |

### 7.3 调参建议

1. **学习率**：从0.01开始，根据loss曲线调整
2. **正则化强度**：用交叉验证选择
3. **Batch size**：32-256之间
4. **优化器**：Adam是安全选择

记住：**没有万能的优化方法，需要根据具体问题选择！**
