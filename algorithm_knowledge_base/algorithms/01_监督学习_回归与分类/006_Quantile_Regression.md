# Quantile Regression 分位数回归 学习文档

> 预测分布而非点估计，理解不确定性

---

## 1. 算法基础认知

### 1.1 一句话定义

Quantile Regression（分位数回归）是预测目标变量在不同分位数上值的方法，不仅给出点估计，还能给出预测的置信区间，比传统回归更丰富。

### 1.2 直觉类比

传统回归就像给出"平均身高170cm"，分位数回归则给出"有50%概率身高低于170cm，90%概率低于180cm"——它告诉你的是整个分布而不只是一个点！

想象你在问一个预测明年工资的问题。普通回归会说"预计年薪20万"，但这个回答信息量太少了！分位数回归更聪明：它会说"有10%概率低于15万，50%概率低于20万，90%概率低于30万"——这才是真正有用的预测，因为它告诉你"不确定性的范围"！

### 1.3 发展背景

- 1978年，Koenker和Bassett提出分位数回归（论文"Quantile Regression"）
- 2005年，Koenker的经典著作《Quantile Regression》出版
- 近年在深度学习中广泛使用（如神经分位数回归）

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 回归 → 分位数预测 |
| 输出 | 任意分位数下的预测值 |
| 方法 | 变体回归损失 |
| 无分布假设 | 不需要正态假设 |

---

## 2. 核心原理

### 2.1 为什么需要分位数回归？

**传统回归的局限**：只给均值，不给分布信息。

| 问题 | 均值回归 | 分位数回归 |
|------|----------|-------------|
| 只预测均值 | ✓ | ✗ |
| 给出不确定性 | ✗ | ✓ |
| 对异常值鲁棒 | ✗ | ✓ |
| 异方差建模 | ✗ | ✓ |

**举例**：收入预测。低收入群体和高收入群体，方差不同。均值回归只能给出平均22万，但分位数可以给出：低收入90%<15万，高收入90%>30万。

### 2.2 分位数概念

定义：$Q_\tau(Y)$ 是满足 $P(Y \leq Q_\tau(Y)) = \tau$ 的值。

```python
tau=0.5  # 中位数
tau=0.1  # 第10分位数（下尾）
tau=0.9  # 第90分位数（上尾）
```

### 2.3 分位数损失函数

分位数回归的核心是 **pinball loss**：

$$L_\tau(y, \hat{y}) = \begin{cases} \tau \cdot (y - \hat{y}) & if\ y > \hat{y} \\ (\tau - 1) \cdot (y - \hat{y}) & if\ y \leq \hat{y} \end{cases}$$

简化写法：
$$L_\tau(y, \hat{y}) = (\tau - \mathbb{I}(y < \hat{y})) \cdot (y - \hat{y})$$

或max形式：
$$L_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau-1)(y - \hat{y}))$$

### 2.4 为什么选这个损失？

该损失的期望是分位数：

$$E[L_\tau(y, \hat{y})] = E[|F_y(\hat{y}) - \tau|]$$

当预测值等于真实 $\tau$ 分位数时，损失最小。

---

## 3. 数学公式与推导

### 3.1 线性分位数回归

形式为：
$$Q_\tau(Y) = X^T \beta_\tau$$

通过最小化：
$$\min_{\beta} \sum_{i=1}^n \rho_\tau(y_i - X_i^T \beta)$$

### 3.2 对偶问题

通过引入辅助变量，可以转化为线性规划：

$$\min_{\beta, \epsilon} \tau \epsilon^+ + (1-\tau) \epsilon^-$$
$$s.t.\ y - X\beta = \epsilon^+ - \epsilon^-$$
$$\epsilon^+, \epsilon^- \geq 0$$

### 3.3 梯度

损失函数的次梯度：
$$\frac{\partial L_\tau}{\partial \hat{y}} = \begin{cases} \tau & if\ \hat{y} < y \\ \tau-1 & if\ \hat{y} > y \\ \in [\tau-1, \tau] & if\ \hat{y} = y \end{cases}$$

### 3.4 与均值的联系

- $\tau = 0.5$ 时是中位数回归
- $\tau \to 0$ 时接近最小值
- $\tau \to 1$ 时接近最大值
- $\tau = 0.5$ 且数据正态时 = 均值回归

---

## 4. 训练过程讲解

### 4.1 训练流程

```
输入: X, y, 分位数tau
输出: 模型参数beta

Step 1: 初始化参数 beta
Step 2: 对每个样本计算残差 r = y - X @ beta
Step 3: 计算分位数损失 L = pinball(r, tau)
Step 4: 反向传播/梯度下降更新 beta
Step 5: 重复直到收敛
```

### 4.2 参数解释

| 参数 | 说明 | 范围 |
|------|------|------|
| tau | 目标分位数 | (0,1) |
| alpha | 正则化强度 | >=0 |
| solver | 求解器 | l1,l2,elasticnet |

### 4.3 多分位数同时估计

```python
# sklearn 同时估计多个分位数
taus = [0.1, 0.25, 0.5, 0.75, 0.9]
models = {}

for tau in taus:
    model = QuantileRegressor(quantile=tau, alpha=0.1)
    model.fit(X, y)
    models[tau] = model
```

---

## 5. 应用场景

### 5.1 预测区间

传统预测只给一个值，分位数回归给出一个区间：

```python
# 预测5%和95%分位数
q_low = model_0.05.predict(X_new)
q_high = model_0.95.predict(X_new)
interval = q_high - q_low  # 预测区间
```

### 5.2 金融风险管理

金融中常用分位数估计风险：

| 分位数 | 金融含义 |
|--------|----------|
| 0.05 | 5%VaR |
| 0.10 | 10%VaR |
| 0.50 | 中位数 |

### 5.3 异方差建模

当数据方差随输入变化时：

```python
# 预测均值和方差
q_0.16 = model_0.16.predict(X)  # ≈ mean - sigma
q_0.84 = model_0.84.predict(X)  # ≈ mean + sigma

# 估计方差
sigma = (q_0.84 - q_0.16) / 2
```

### 5.4 质量控制

工业中预测产品指标的上下限：

```python
# 预测规格上下限
lower = model_0.01.predict(X)
upper = model_0.99.predict(X)
```

### 5.5 对比选择

| 场景 | 推荐 |
|------|------|
| 均值预测 | 线性回归 |
| 预测区间 | 分位数回归 |
| 单一值+不确定 | Bayesian回归 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 分布信息 | 直接得到任意分位 |
| 鲁棒性 | 对异常值不敏感 |
| 无分布假设 | 不需要正态假设 |
| 异方差 | 能建模方差变化 |
| 可解释 | 物理含义明确 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算稍慢 | 需要解线性规划 |
| 需要训练多个 | 每个tau一个模型 |
| 不提供概率 | 只是分位点，不是分布 |

### 6.3 注意事项

- 模型之间不保证单调性（$\tau_1 < \tau_2 \Rightarrow \hat{y}_1 \leq \hat{y}_2$）
- 可以加单调性约束解决
- 异常值多时用 $\tau=0.5$ 更鲁棒

---

## 7. 调库实现（Python + scikit-learn）

### 7.1 基本用法

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 生成示例数据
np.random.seed(42)
n = 500
X = np.random.randn(n, 2)
y = 2*X[:, 0] + X[:, 1] + 0.5*np.random.randn(n)

# 分位数回归
model = QuantileRegressor(quantile=0.5, alpha=0.1)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
print(f"预测形状: {y_pred.shape}")
print(f"预测前5个: {y_pred[:5]}")
```

### 7.2 预测区间

```python
# 同时训练多个分位数
taus = [0.05, 0.25, 0.5, 0.75, 0.95]
predictions = {}

for tau in taus:
    model = QuantileRegressor(quantile=tau, alpha=0.1)
    model.fit(X, y)
    predictions[tau] = model.predict(X)

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y, predictions[0.5], alpha=0.5, label='中位数')

# 区间
for tau in [0.05, 0.95]:
    plt.scatter(y, predictions[tau], alpha=0.3, label=f'{int(tau*100)}%')

plt.plot([-5, 5], [-5, 5], 'k--', label='完美预测')
plt.xlabel('真实值')
plt.ylabel('预测��')
plt.legend()
plt.title('分位数回归预测')
plt.savefig('quantile_regression.png', dpi=100)
plt.show()
```

### 7.3 参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'quantile': [0.1, 0.25, 0.5, 0.75, 0.9],
    'alpha': [0.001, 0.01, 0.1, 1.0]
}

model = QuantileRegressor()
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)

print(f"最优参数: {grid.best_params_}")
print(f"最优分数: {grid.best_score_:.3f}")
```

### 7.4 与深度学习结合

```python
import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """分位数损失"""
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        
    def forward(self, pred, target):
        diff = target - pred
        loss = torch.max(self.tau * diff, (self.tau - 1) * diff)
        return loss.mean()

class QuantileNet(nn.Module):
    """分位数回归网络"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 训练多个分位数
taus = [0.1, 0.5, 0.9]
models = {}

for tau in taus:
    model = QuantileNet(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = QuantileLoss(tau)
    
    # 训练循环
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        
        loss.backward()
        optimizer.step()
    
    models[tau] = model
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np
from scipy.optimize import linprog

class QuantileRegression:
    """分位数回归 - 手工实现（线性规划）"""
    
    def __init__(self, tau=0.5, alpha=0.0):
        self.tau = tau
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        
    def _pinball_loss(self, y_true, y_pred):
        """Pinball损失"""
        diff = y_true - y_pred
        return np.maximum(self.tau * diff, (self.tau - 1) * diff).mean()
    
    def fit(self, X, y):
        """用线性规划拟合"""
        n_samples, n_features = X.shape
        
        # 添加截距
        X_aug = np.column_stack([np.ones(n_samples), X])
        n_params = n_features + 1
        
        # 构造线性规划问题
        # min: tau @ u + (1-tau) @ v
        # s.t.: X_aug @ beta + u - v = y
        #       u >= 0, v >= 0
        
        c = np.zeros(n_params * 2)
        c[n_params:] = self.tau if self.tau <= 0.5 else (1 - self.tau)
        c[n_params-1] = self.tau if self.tau <= 0.5 else (1 - self.tau)
        
        # 等式约束
        A_eq = np.zeros((n_samples, n_params * 2))
        for i in range(n_samples):
            A_eq[i, :n_params] = X_aug[i]
            A_eq[i, n_params:] = -X_aug[i]
        
        b_eq = y
        
        # 边界
        bounds = [(None, None)] * (n_params * 2)
        
        # 求解
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        # 提取参数
        beta = result.x[:n_params]
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.coef_ + self.intercept_
    
    def fit_predict(self, X, y):
        """拟合+预测"""
        self.fit(X, y)
        return self.predict(X)


class QuantileRegressionGD:
    """分位数回归 - 梯度下降版本"""
    
    def __init__(self, tau=0.5, lr=0.01, n_iters=1000):
        self.tau = tau
        self.lr = lr
        self.n_iters = n_iters
        self.coef_ = None
        self.intercept_ = 0.0
        
    def _gradient(self, y_true, y_pred, X):
        """分位数损失的梯度"""
        mask = (y_true > y_pred).astype(float)
        grad_coef = -(self.tau - mask) @ X / len(y_true)
        grad_intercept = -(self.tau - mask).mean()
        
        return grad_coef, grad_intercept
    
    def fit(self, X, y):
        """梯度下降拟合"""
        n_samples, n_features = X.shape
        
        # 初始化
        self.coef_ = np.zeros(n_features)
        
        # 梯度下降
        for i in range(self.n_iters):
            y_pred = X @ self.coef_ + self.intercept_
            grad_coef, grad_intercept = self._gradient(y_true=y, y_pred=y_pred, X=X)
            
            self.coef_ -= self.lr * grad_coef
            self.intercept_ -= self.lr * grad_intercept
            
            # 学习率衰减
            self.lr *= 0.999
            
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成数据
    n = 500
    X = np.random.randn(n, 2)
    y = 2*X[:, 0] + X[:, 1] + np.random.randn(n)
    
    # 手工实现
    model = QuantileRegression(tau=0.5)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # sklearn实现
    from sklearn.linear_model import QuantileRegressor
    model_sklearn = QuantileRegressor(quantile=0.5, alpha=0.1)
    model_sklearn.fit(X, y)
    y_pred_sklearn = model_sklearn.predict(X)
    
    print("手工预测前5:", y_pred[:5])
    print("sklearn预测前5:", y_pred_sklearn[:5])
    
    # 预测区间
    taus = [0.05, 0.5, 0.95]
    predictions = {}
    
    for tau in taus:
        model = QuantileRegression(tau=tau)
        model.fit(X, y)
        predictions[tau] = model.predict(X)
    
    print("\n分位数预测区间示例:")
    print(f"5%: {predictions[0.05][:3]}")
    print(f"50%: {predictions[0.5][:3]}")
    print(f"95%: {predictions[0.95][:3]}")
```

---

## 9. 可视化与结果理解

### 9.1 预测区间可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
np.random.seed(42)
n = 200
X = np.sort(np.random.randn(n))
y = X + 0.5*np.random.randn(n)

# 多分位数预测
from sklearn.linear_model import QuantileRegressor

taus = [0.05, 0.25, 0.5, 0.75, 0.95]
predictions = {}

for tau in taus:
    model = QuantileRegressor(quantile=tau, alpha=0.1)
    model.fit(X.reshape(-1, 1), y)
    predictions[tau] = model.predict(X.reshape(-1, 1))

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))

# 数据点
ax.scatter(X, y, alpha=0.5, s=20, label='数据')

# 预测线
colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
for i, tau in enumerate(taus):
    ax.plot(X, predictions[tau], color=colors[i], linewidth=2, 
            label=f'Q{int(tau*100)}')

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('分位数回归 - 预测区间')
ax.legend()
plt.savefig('quantile_interval.png', dpi=100)
plt.show()
```

### 9.2 异方差检测

```python
# 当方差随X变化时
np.random.seed(42)
X = np.linspace(-3, 3, 300)
y = X + (0.5 + X**2) * np.random.randn(300)  # 方差随X增大

# 分位数回归检测
taus = [0.05, 0.5, 0.95]
predictions = {}

for tau in taus:
    model = QuantileRegressor(quantile=tau)
    model.fit(X.reshape(-1, 1), y)
    predictions[tau] = model.predict(X.reshape(-1, 1))

# 区间宽度
interval_width = predictions[0.95] - predictions[0.05]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 数据与预测
axes[0].scatter(X, y, alpha=0.3, s=10)
axes[0].plot(X, predictions[0.5], 'b-', label='中位数')
axes[0].fill_between(X, predictions[0.05], predictions[0.95], 
                     alpha=0.3, label='90%区间')
axes[0].legend()
axes[0].set_title('异方差数据')

# 区间宽度随X变化
axes[1].plot(X, interval_width, 'r-')
axes[1].set_xlabel('X')
axes[1].set_ylabel('区间宽度')
axes[1].set_title('区间宽度随X变化')

plt.tight_layout()
plt.savefig('heteroscedastic.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| Pinball Loss | 分位数损失 | $\frac{1}{n}\sum \rho_\tau(y - \hat{y})$ |
| Coverage | 区间覆盖率 | $P(\hat{y}_{low} \leq y \leq \hat{y}_{high})$ |
| Interval Width | 区间宽度 | $\hat{y}_{high} - \hat{y}_{low}$ |

### 10.2 评估代码

```python
def pinball_loss(y_true, y_pred, tau):
    """Pinball损失"""
    diff = y_true - y_pred
    return np.maximum(tau * diff, (tau - 1) * diff).mean()

def coverage(y, lower, upper):
    """覆盖率"""
    return np.mean((y >= lower) & (y <= upper))

# 评估
for tau in [0.1, 0.5, 0.9]:
    loss = pinball_loss(y, predictions[tau], tau)
    print(f"Q{int(tau*100)} pinball loss: {loss:.3f}")

# 区间评估
cov = coverage(y, predictions[0.05], predictions[0.95])
width = np.mean(predictions[0.95] - predictions[0.05])
print(f"Coverage: {cov:.1%}")
print(f"Average width: {width:.3f}")
```

---

## 11. 常见问题与易错点

### Q1: $\tau$ 如何选择？

**答案**：根据需求。$\tau=0.5$ 是中位数（鲁棒），$\tau=0.05/0.95$ 用于预测边界。

### Q2: 预测不单调？

**答案**：不同$\tau$的预测可能不满足单调性$\tau_1 < \tau_2 \Rightarrow \hat{y}_1 \leq \hat{y}_2$。可加��调��束解决。

### Q3: 和置信区间有何区别？

**答案**：置信区间是基于均值+方差假设的正态分位数，分位数回归无假设、更通用。

### Q4: 异常值多选哪个$\tau$？

**答案**：$\tau=0.5$（中位数）最鲁棒，因为最小化中位数损失。

### Q5: 训练多个模型太麻烦？

**答案**：可以用多输出网络同时预测多个$\tau$。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 损失函数 | Pinball loss |
| 分位数 | $Q_\tau$ 是$\tau$分位点 |
| 优势 | 分布信息+鲁棒+无假设 |
| 应用 | 预测区间+风险 |

### 12.2 公式汇总

Pinball损失：
$$L_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau-1)(y - \hat{y}))$$

分位数定义：
$$Q_\tau: P(Y \leq Q_\tau) = \tau$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. 当$\tau=0.5$时，分位数回归等价于：
   - A) 均值回归
   - B) 中位数回归
   - C) 最小值回归

2. Pinball损失对异常值：
   - A) 很敏感
   - B) 不敏感
   - C) 取决于$\tau$

3. 90%预测区间应用哪两个分位数：
   - A) 0.05, 0.95
   - B) 0.25, 0.75
   - C) 0.1, 0.9

### 13.2 简答题

1. 为什么分位数回归比均值回归更鲁棒？
2. 如何用分位数回归估计异方差？

### 13.3 编程题

1. 实现基于分位数回归的VaR计算。
2. 比较多$\tau$预测与贝叶斯不确定性。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
统计基础
    ↓
均值回归
    ↓
分位数概念
    ↓
分位数回归
    ↓
深度学习结合
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| 线性回归 | 均值版本 |
| 鲁棒回归 | M-估计 |
| 贝叶斯回归 | 概率分布 |

### 14.3 扩展阅读

- Koenker, R. (2005). Quantile Regression. Cambridge University Press.
- Meinshausen, N. (2006). Quantile Regression Techniques.

---

## 附录

### 参考

1. Koenker, R., Bassett, G. (1978). Regression Quantiles. Econometrica.
2. Koenker, R. (2005). Quantile Regression. Cambridge University Press.
3. sklearn.linear_model.QuantileRegressor 文档

---

**文档结束**