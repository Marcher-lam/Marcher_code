# 随机梯度下降(SGD) 学习文档

> SGD是深度学习和大规模优化的核心引擎，用随机采样梯度替代精确梯度，实现高效在线参数更新。

> 来源线索：本节内容根据原书中关于"Stochastic Gradient Methods"的相关章节(Ch 5.3)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：随机梯度下降（SGD）通过每次只用一个（或少量）样本的随机梯度来更新参数，高效地求解期望损失最小的优化问题。

**直觉类比**：想象你蒙着眼睛下山，目标是到达谷底（最小值）。批量梯度下降会仔细测量周围每一点的坡度再决定方向——精确但极慢。SGD只需踩一脚感受坡度就迈一步——粗糙但极快，而且在大规模数据下效果出乎意料地好。

**历史背景**：SGD的理论基础由Herbert Robbins和Sutton Monro在1951年奠定（随机近似理论）。随着深度学习的兴起，SGD成为训练神经网络的标准方法，其变体（Adam、AdaGrad等）更是现代机器学习的基石。

**算法定位**：随机优化/一阶方法。SGD是随机搜索问题的核心算法（原书Ch 5），也是本书统一框架中"策略函数近似"和"值函数近似"的基础优化工具。

**前置知识**：微积分（梯度、偏导数）、线性代数、概率论（期望、方差）、Python编程。

## 2. 核心原理

**核心思想**：当我们需要最小化$\min_x \mathbb{E}[F(x,W)]$时，精确计算期望梯度$\nabla \mathbb{E}[F(x,W)]$通常代价过高。SGD的关键洞察是：一次随机采样的梯度$\nabla F(x, W^{n+1})$是无偏估计，虽然噪声大但计算便宜，大量累积后可以收敛到最优解。

**工作流程**：

1. 初始化参数$x^0$
2. 在第$n$步，采样一个随机变量$W^{n+1}$
3. 计算随机梯度$g^n = \nabla F(x^n, W^{n+1})$
4. 更新参数：$x^{n+1} = x^n - \alpha_n g^n$
5. 重复步骤2-4直到收敛

**关键概念**：

- **随机梯度**：$\nabla F(x, W)$是真实梯度$\nabla \mathbb{E}[F(x,W)]$的无偏但噪声估计
- **步长（学习率）**$\alpha_n$：控制每步更新幅度，是SGD最重要的超参数
- **Mini-batch**：用一小批样本的平均梯度替代单样本梯度，平衡噪声和计算
- **收敛条件**：Robbins-Monro条件（$\sum \alpha_n = \infty$，$\sum \alpha_n^2 < \infty$）

```
批量梯度下降：  x ← x - α·∇E[F(x,W)]    精确但慢
SGD：          x ← x - α·∇F(x,wⁿ)      快但有噪声
Mini-batch：   x ← x - α·(1/B)Σ∇F(x,wⁱ) 平衡
```

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $x$ | 参数向量 |
| $F(x,W)$ | 随机目标函数 |
| $\nabla F(x,W)$ | 随机梯度 |
| $\alpha_n$ | 第$n$步的步长 |
| $W$ | 随机变量 |
| $g^n$ | 第$n$步的梯度估计 |

### 目标函数

SGD求解的优化问题：

$$\min_x \mathbb{E}[F(x, W)]$$

在机器学习中，这通常是经验风险最小化：

$$\min_\theta \frac{1}{N}\sum_{i=1}^{N} L(f(x_i;\theta), y_i)$$

### SGD更新公式

$$x^{n+1} = x^n - \alpha_n \nabla F(x^n, W^{n+1})$$

**为什么这样更新有效？**

因为随机梯度是无偏估计：

$$\mathbb{E}[\nabla F(x, W)] = \nabla \mathbb{E}[F(x, W)]$$

所以虽然单步方向有噪声，期望方向是正确的。

### Mini-batch SGD

使用$B$个样本的平均梯度：

$$x^{n+1} = x^n - \alpha_n \frac{1}{B}\sum_{i=1}^{B} \nabla F(x^n, W_i^{n+1})$$

梯度估计的方差降低为单样本的$1/B$。

### 收敛性条件（Robbins-Monro）

SGD保证收敛的步长条件：

$$\sum_{n=1}^{\infty} \alpha_n = \infty, \quad \sum_{n=1}^{\infty} \alpha_n^2 < \infty$$

典型的满足条件的步长：$\alpha_n = \frac{c}{n}$（调和步长）。

### SGD作为序贯决策问题

原书（Ch 5.6）指出SGD本身可以看作一个序贯决策问题：状态是当前参数$x^n$，动作是步长选择或梯度估计方法，转移由随机变量$W^{n+1}$驱动。

## 4. 训练过程讲解

### 数据预处理
- 特征标准化（zero-mean, unit-variance）
- 数据打乱（避免有序数据导致梯度偏差）
- Mini-batch划分

### 参数初始化
- 参数：随机小值初始化（如Xavier初始化）
- 步长调度：通常从较大值开始逐步衰减

### 迭代过程
1. 遍历所有mini-batch
2. 计算mini-batch梯度
3. 更新参数
4. 每个epoch结束后打乱数据
5. 重复直到验证损失不再下降

### 收敛条件
- 训练损失变化小于阈值
- 验证损失不再改善（早停）
- 最大epoch数

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $\alpha$ | 初始学习率 | [0.001, 0.1] | 0.01 |
| batch_size | Mini-batch大小 | [16, 512] | 32 |
| epochs | 训练轮数 | [10, 500] | 100 |
| lr_decay | 学习率衰减 | [0.1, 0.99] | 0.95 |

## 5. 应用场景

### 1. 深度学习模型训练
为什么适合：神经网络有数百万参数，批量梯度下降不可行，SGD（及变体Adam等）是唯一可行的训练方法。

### 2. 大规模线性模型
为什么适合：数据量太大无法一次加载到内存，SGD可以逐样本或逐批处理。

### 3. 在线学习
为什么适合：数据持续到达，SGD可以增量更新模型，不需要重新训练。

### 4. 强化学习策略优化
为什么适合：策略梯度和值函数近似都使用SGD更新参数。

### 不适用场景
- 需要高精度解（用L-BFGS等二阶方法）
- 目标函数不可微（用SPSA等无梯度方法）
- 数据量很小（直接用批量方法更高效）

## 6. 优缺点分析

### 优点
1. **计算高效**：每步只需一个（或少量）样本
2. **内存友好**：不需要加载全部数据
3. **在线适用**：可以处理流式数据
4. **隐式正则化**：SGD的噪声有正则化效果，防止过拟合

### 缺点
1. **收敛较慢**：噪声导致在最优解附近震荡
2. **步长敏感**：步长选择对性能影响巨大
3. **非凸问题**：可能陷入局部最优或鞍点
4. **方差大**：单样本梯度估计方差高

### 算法对比

| 特性 | SGD | 批量GD | Adam | L-BFGS |
|------|-----|--------|------|--------|
| 每步计算量 | O(d) | O(Nd) | O(d) | O(d²) |
| 内存需求 | O(d) | O(Nd) | O(d) | O(d²) |
| 收敛速度 | 慢 | 中 | 中 | 快 |
| 大规模适用 | 是 | 否 | 是 | 否 |
| 需要调步长 | 是 | 是 | 较少 | 否 |

## 7. 调库实现

```python
"""
使用PyTorch实现SGD训练线性回归
"""
import torch
import torch.nn as nn
import numpy as np

# 生成数据
np.random.seed(42)
torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 0.1 * torch.randn(n_samples)

# 定义模型
model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
losses = []
for epoch in range(100):
    # Mini-batch SGD
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, 32):
        batch_X = X[perm[i:i+32]]
        batch_y = y[perm[i:i+32]]

        optimizer.zero_grad()
        pred = model(batch_X).squeeze()
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print(f"\n学习到的权重: {model.weight.data.numpy().round(3)}")
print(f"真实权重:     {true_w.numpy()}")
```

## 8. 手工代码实现

```python
"""
从零实现SGD优化器
NumPy实现，包含标准SGD和Momentum SGD
"""
import numpy as np

class SGDOptimizer:
    """随机梯度下降优化器"""

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        """执行一步SGD更新"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # 动量更新：v = μv + g, θ = θ - αv
        self.velocity = self.momentum * self.velocity + grads
        params = params - self.lr * self.velocity
        return params


class LinearRegressionSGD:
    """用SGD训练的线性回归"""

    def __init__(self, n_features, lr=0.01, momentum=0.9):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.optimizer_w = SGDOptimizer(lr, momentum)
        self.optimizer_b = SGDOptimizer(lr, momentum)

    def predict(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y, epochs=100, batch_size=32):
        """Mini-batch SGD训练"""
        n = len(y)
        losses = []

        for epoch in range(epochs):
            # 打乱数据
            perm = np.random.permutation(n)
            epoch_loss = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                X_b, y_b = X[idx], y[idx]

                # 前向传播
                pred = self.predict(X_b)
                error = pred - y_b
                batch_loss = np.mean(error ** 2)
                epoch_loss += batch_loss * len(idx)

                # 计算梯度：∂L/∂w = (2/B) X^T(Xw+b-y)
                grad_w = (2.0 / len(idx)) * X_b.T @ error
                grad_b = (2.0 / len(idx)) * np.sum(error)

                # SGD更新
                self.weights = self.optimizer_w.step(self.weights, grad_w)
                self.bias = self.optimizer_b.step(np.array([self.bias]), grad_b)
                self.bias = self.bias[0]

            avg_loss = epoch_loss / n
            losses.append(avg_loss)

        return losses


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 500, 3
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(n)

    model = LinearRegressionSGD(d, lr=0.01, momentum=0.9)
    losses = model.fit(X, y, epochs=100, batch_size=32)

    print(f"学习权重: {model.weights.round(4)}")
    print(f"真实权重: {true_w}")
    print(f"偏差: {model.bias:.4f}")
    print(f"最终MSE: {losses[-1]:.6f}")
```

## 9. 可视化与结果理解

```python
"""
SGD训练过程可视化
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_sgd(losses_sgd, losses_momentum=None):
    """可视化SGD的训练损失曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(losses_sgd, label='SGD', alpha=0.8)
    if losses_momentum:
        ax.plot(losses_momentum, label='SGD + Momentum', alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('SGD训练损失收敛曲线')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sgd_convergence.png', dpi=150)
    plt.show()
```

**结果解读**：SGD的损失曲线会有波动（因随机噪声），但总体趋势向下。加入Momentum后收敛更平滑更快。

## 10. 模型评估

```python
"""SGD模型评估"""
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_sgd_model(model, X_test, y_test):
    """评估SGD训练的模型"""
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.4f}")
    return mse, r2
```

## 11. 常见问题与易错点

### 数据层面

1. **数据未打乱**
   - 现象：训练不收敛或收敛极慢
   - 原因：有序数据导致mini-batch不具代表性
   - 解决方案：每个epoch前打乱数据

2. **特征尺度不一致**
   - 现象：某些特征的梯度主导更新
   - 原因：特征量级差异大
   - 解决方案：标准化特征（StandardScaler）

### 模型层面

3. **学习率过大**
   - 现象：损失发散或剧烈震荡
   - 原因：步长太大跳过最优点
   - 解决方案：降低学习率，或使用学习率调度

4. **学习率过小**
   - 现象：训练极慢，长时间不收敛
   - 原因：步长太小
   - 解决方案：增大学习率，或使用自适应方法（Adam）

### 调参层面

5. **batch_size选择不当**
   - 现象：GPU利用率低（太小）或收敛质量差（太大）
   - 解决方案：通常32-256之间，根据硬件和任务调整

## 12. 学习总结

SGD的核心贡献在于用随机采样实现了大规模优化的可行性。它的理论基础是随机近似理论（Robbins-Monro），核心公式极其简洁：$x_{n+1} = x_n - \alpha_n \nabla F(x_n, W_{n+1})$。

**关键公式**：
1. SGD更新：$x^{n+1} = x^n - \alpha_n \nabla F(x^n, W^{n+1})$
2. Robbins-Monro条件：$\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$
3. Mini-batch：$x^{n+1} = x^n - \frac{\alpha_n}{B}\sum_{i=1}^B \nabla F(x^n, W_i^{n+1})$

在原书的统一框架中，SGD既是随机搜索（Ch 5）的核心方法，也是策略梯度（Ch 12）和值函数近似（Ch 16-17）的底层优化引擎。Adam、AdaGrad、RMSProp都是SGD的自适应步长变体（Ch 6）。

## 13. 练习题与思考题

### 基础题

**题目1**：设$f(x) = \frac{1}{2}x^2$，真实梯度为$f'(x) = x$。SGD使用随机梯度$g^n = x^n + \epsilon^n$（$\epsilon^n \sim N(0,1)$），步长$\alpha_n = 0.1$。从$x^0 = 5$出发，写出前3步更新。

**参考答案**：
设$\epsilon^1=0.3, \epsilon^2=-0.5, \epsilon^3=0.1$：
- $x^1 = 5 - 0.1 \times (5 + 0.3) = 5 - 0.53 = 4.47$
- $x^2 = 4.47 - 0.1 \times (4.47 - 0.5) = 4.47 - 0.397 = 4.073$
- $x^3 = 4.073 - 0.1 \times (4.073 + 0.1) = 4.073 - 0.4173 = 3.656$

可见$|x|$逐步减小，趋向最优$x^*=0$，但速度受噪声影响。

### 进阶题

**题目2**：证明Mini-batch SGD的梯度估计方差是单样本SGD的$1/B$倍。

**参考答案**：
设单个样本梯度方差为$\sigma^2$。Mini-batch梯度为$g_B = \frac{1}{B}\sum_{i=1}^B g_i$，其中$g_i$独立同分布。
$\text{Var}(g_B) = \frac{1}{B^2}\sum_{i=1}^B \text{Var}(g_i) = \frac{B\sigma^2}{B^2} = \frac{\sigma^2}{B}$。
方差降低为$1/B$，但计算量增加$B$倍。

### 开放思考题

**题目3**：原书（Ch 5.6）将SGD本身看作序贯决策问题。在这个视角下，状态、决策、随机信息分别是什么？步长$\alpha_n$的选择如何影响这个"决策"？

**参考答案方向**：
- 状态：当前参数$x^n$和可能的梯度历史
- 决策：步长$\alpha_n$的选择（甚至梯度估计方法的选择）
- 随机信息：采样到的$W^{n+1}$
- 步长选择是一个元优化问题：太大导致发散，太小导致收敛慢。原书Ch 6专门讨论最优步长策略。

## 14. 学习路径建议

**前置算法**：梯度下降法、概率论基础

**平行算法**：SPSA（无梯度方法）、牛顿法（二阶方法）

**进阶算法**：Adam、AdaGrad、RMSProp（自适应步长变体，Ch 6）、策略梯度（SGD在RL中的应用，Ch 12）

**推荐资源**：
1. 原书Ch 5 "Derivative-Based Stochastic Search"
2. Bottou et al. "Optimization Methods for Large-Scale Machine Learning" (2018)
3. Goodfellow et al. "Deep Learning" Ch 8 "Optimization for Deep Models"
