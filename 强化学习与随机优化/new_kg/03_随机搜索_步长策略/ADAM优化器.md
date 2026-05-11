# ADAM优化器 学习文档

> Adam结合动量和自适应步长，是深度学习中最常用的优化器，几乎成为神经网络训练的默认选择。

> 来源线索：本节内容根据原书中关于"ADAM"的相关章节(Ch 6.2.3.4)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：Adam（Adaptive Moment Estimation）是一种自适应学习率优化算法，通过计算梯度的一阶矩和二阶矩的指数移动平均来动态调整每个参数的学习率。

**直觉类比**：想象你在一片起伏的地形中寻找最低点。SGD就像只看脚下的坡度往前走，容易在窄谷中来回震荡。Adam则像带了两个记忆：一个记住最近坡度的方向（动量），另一个记住最近坡度的大小（自适应步长）。在陡坡处自动放慢脚步，在缓坡处迈大步，同时沿着之前的方向继续前进。

**历史背景**：Adam由Diederik Kingma和Jimmy Ba在2015年的ICLR论文中提出。它综合了AdaGrad（处理稀疏梯度）和RMSProp（处理非平稳目标）的优点，加上动量机制。很快成为深度学习社区最受欢迎的优化器。

**算法定位**：自适应步长/一阶优化方法。在原书中属于步长策略（Ch 6）的一种自适应方法，也是SGD的重要改进。

**前置知识**：SGD、梯度下降、指数移动平均、Python。

## 2. 核心原理

**核心思想**：Adam为每个参数维护两个移动平均：(1)梯度的移动平均（一阶矩，类似动量），(2)梯度平方的移动平均（二阶矩，类似RMSProp）。然后用这两个统计量来调整每步的更新幅度——梯度一直很大的方向用小步长，梯度小或偶尔出现的方向用大步长。

**工作流程**：

1. 初始化参数$x_0$，一阶矩$m_0=0$，二阶矩$v_0=0$
2. 在第$t$步，计算梯度$g_t$
3. 更新一阶矩：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
4. 更新二阶矩：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
5. 偏差校正：$\hat{m}_t = m_t/(1-\beta_1^t)$，$\hat{v}_t = v_t/(1-\beta_2^t)$
6. 更新参数：$x_t = x_{t-1} - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
7. 重复

**关键概念**：

- **一阶矩**$m_t$：梯度的指数移动平均，起到动量作用
- **二阶矩**$v_t$：梯度平方的指数移动平均，衡量梯度的历史大小
- **偏差校正**：因为$m_0=v_0=0$，初期估计偏向0，校正后更准确
- **自适应步长**：每个参数的学习率由其历史梯度大小决定

```
梯度 g_t ─→ 更新一阶矩 m_t (方向记忆)
    │
    └──→ 更新二阶矩 v_t (大小记忆)
              │
              ↓
     x ← x - α · m̂_t / (√v̂_t + ε)
              │       │
              │       └── 自适应缩放
              └── 动量方向
```

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $\alpha$ | 学习率 |
| $\beta_1$ | 一阶矩衰减率 |
| $\beta_2$ | 二阶矩衰减率 |
| $\epsilon$ | 数值稳定项 |
| $m_t$ | 一阶矩估计 |
| $v_t$ | 二阶矩估计 |

### 一阶矩更新

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

这是梯度的指数移动平均，$\beta_1$越大，历史信息权重越高。

### 二阶矩更新

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

这是梯度平方的指数移动平均，衡量梯度的历史变化幅度。

### 偏差校正

初始化$m_0=v_0=0$导致初期估计偏小。校正公式：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**为什么需要校正？**当$t$很小时，$\beta_1^t$接近1，分母接近0，使得$\hat{m}_t$远大于$m_t$，弥补初始零偏。当$t \to \infty$时，$\beta^t \to 0$，校正消失。

### 参数更新

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

- $\hat{m}_t$：方向（来自梯度的移动平均）
- $\sqrt{\hat{v}_t}$：步长缩放（梯度大的方向步长小）
- $\epsilon$：防止除零（通常$10^{-8}$）

### 与AdaGrad和RMSProp的关系

- **AdaGrad**：累积所有历史梯度平方$\sum g_i^2$，学习率单调递减，后期太小
- **RMSProp**：用指数移动平均替代累积，适合非平稳问题
- **Adam**：在RMSProp基础上加入动量（一阶矩）和偏差校正

## 4. 训练过程讲解

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $\alpha$ | 学习率 | [1e-4, 1e-2] | 0.001 |
| $\beta_1$ | 一阶矩衰减率 | [0.85, 0.99] | 0.9 |
| $\beta_2$ | 二阶矩衰减率 | [0.9, 0.999] | 0.999 |
| $\epsilon$ | 稳定项 | [1e-8, 1e-6] | 1e-8 |

### 训练技巧
- 学习率是最关键的超参数，建议用学习率调度（warmup + cosine decay）
- $\beta_1, \beta_2$默认值在大多数情况下效果很好
- 对于NLP任务（如Transformer训练），常用AdamW（Adam + 解耦权重衰减）

## 5. 应用场景

1. **深度学习训练**：几乎所有神经网络的标准优化器
2. **NLP/Transformer**：BERT、GPT等模型默认使用Adam/AdamW
3. **强化学习**：策略梯度、DQN等算法常用Adam
4. **计算机视觉**：ResNet等模型训练

## 6. 优缺点分析

### 优点
1. **自适应步长**：每个参数自动调整学习率
2. **少调参**：默认超参数在大多数情况下有效
3. **处理稀疏梯度**：适合NLP等稀疏场景
4. **收敛快**：初期收敛通常比SGD快

### 缺点
1. **泛化性不如SGD**：在某些CV任务上SGD+Momentum泛化更好
2. **可能不收敛**：在某些问题上Adam可能不收敛到最优点
3. **内存开销**：需要额外存储$m_t$和$v_t$

### 算法对比

| 特性 | Adam | SGD+Momentum | AdaGrad | RMSProp |
|------|------|-------------|---------|---------|
| 自适应步长 | 是 | 否 | 是 | 是 |
| 动量 | 是 | 是 | 否 | 否 |
| 稀疏梯度 | 好 | 差 | 好 | 中 |
| 泛化性能 | 中 | 好 | 差 | 中 |
| 调参难度 | 低 | 中 | 低 | 中 |

## 7. 调库实现

```python
"""PyTorch中使用Adam"""
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()
print("Adam优化训练完成")
```

## 8. 手工代码实现

```python
"""
从零实现Adam优化器
NumPy实现，与PyTorch Adam等价
"""
import numpy as np

class Adam:
    """Adam优化器"""

    def __init__(self, params_shape, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(params_shape)  # 一阶矩
        self.v = np.zeros(params_shape)  # 二阶矩
        self.t = 0                       # 时间步

    def step(self, params, grads):
        """执行一步Adam更新"""
        self.t += 1

        # 更新一阶矩：m = β₁m + (1-β₁)g
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # 更新二阶矩：v = β₂v + (1-β₂)g²
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        # 偏差校正
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # 参数更新
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ========== 测试：用Adam优化二次函数 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 目标：min f(x) = 0.5 * x^T A x - b^T x
    n = 5
    A = np.random.randn(n, n)
    A = A.T @ A + np.eye(n)  # 正定矩阵
    b = np.random.randn(n)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b

    x_opt = np.linalg.solve(A, b)  # 精确解

    # Adam优化
    x = np.random.randn(n) * 5  # 初始值
    adam = Adam(x.shape, lr=0.1)

    for i in range(500):
        g = grad(x)
        x = adam.step(x, g)
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: f(x)={f(x):.6f}, ||x-x*||={np.linalg.norm(x-x_opt):.6f}")

    print(f"\nAdam最优解: {x.round(4)}")
    print(f"精确解:     {x_opt.round(4)}")
    print(f"误差范数:   {np.linalg.norm(x-x_opt):.2e}")
```

## 9. 可视化与结果理解

```python
"""Adam vs SGD收敛对比"""
import matplotlib.pyplot as plt
import numpy as np

def compare_optimizers():
    # Rosenbrock函数的简化2D版本
    def f(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def grad(x): return np.array([
        -2*(1-x[0]) + 400*x[0]*(x[0]**2 - x[1]),
        200*(x[1] - x[0]**2)
    ])

    paths = {}
    for name, opt_fn in [('SGD', lambda g: 0.001*g),
                          ('Adam', 'adam')]:
        x = np.array([-1.5, 2.0])
        path = [x.copy()]
        if name == 'Adam':
            adam = Adam(x.shape, lr=0.01)
            for _ in range(500):
                g = grad(x)
                x = adam.step(x, g)
                path.append(x.copy())
        else:
            for _ in range(5000):
                g = grad(x)
                x = x - 0.001 * g
                path.append(x.copy())
        paths[name] = np.array(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, path in paths.items():
        ax.plot(path[:,0], path[:,1], label=name, alpha=0.7)
    ax.plot(1, 1, 'r*', markersize=15, label='最优点')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Adam vs SGD 优化路径对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('adam_vs_sgd.png', dpi=150)
    plt.show()

# compare_optimizers()
```

## 10. 模型评估

```python
def evaluate_optimizer(f_history):
    print(f"初始损失: {f_history[0]:.4f}")
    print(f"最终损失: {f_history[-1]:.4f}")
    print(f"收敛步数: {next(i for i,v in enumerate(f_history) if v < 0.01)}")
```

## 11. 常见问题与易错点

1. **学习率太大**
   - 现象：损失不收敛或震荡
   - 解决方案：降低$\alpha$到$10^{-4}$或使用warmup

2. **忘记偏差校正**
   - 现象：训练初期不稳定
   - 原因：初期$m_t, v_t$偏小
   - 解决方案：确保实现中包含偏差校正

3. **AdamW vs Adam**
   - AdamW将权重衰减解耦，正则化效果更好
   - 对Transformer等大模型推荐用AdamW

## 12. 学习总结

Adam的核心是"自适应步长+动量"的组合。一阶矩提供方向（动量），二阶矩提供缩放（自适应），偏差校正确保初期准确。

**关键公式**：
1. $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
2. $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
3. $\theta_{t+1} = \theta_t - \alpha \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$

Adam是SGD（Ch 5）的自适应步长变体，步长策略详见Ch 6。在原书框架中，Adam是"自适应步长策略"的一种具体实现。

## 13. 练习题与思考题

**题目1**：Adam中偏差校正为什么在训练初期重要但后期不重要？

**参考答案**：初期$t$小，$\beta^t$接近1，校正因子$1/(1-\beta^t)$很大，将偏小的$m_t, v_t$放大。当$t \to \infty$时$\beta^t \to 0$，校正因子趋近1，校正效果消失。

**题目2（开放）**：为什么在某些计算机视觉任务中SGD+Momentum比Adam泛化性能更好？

**参考答案方向**：Adam找到的极小值通常更"尖"（flat minima理论），而SGD因为噪声更大找到更"平"的极小值，平极小值泛化更好。此外Adam的自适应步长可能让训练过早收敛到次优点。

## 14. 学习路径建议

**前置**：SGD、梯度下降

**进阶**：AdamW、LAMB（大batch训练）、学习率调度（Cosine、Warmup）

**推荐资源**：原书Ch 6.2.3.4、Kingma & Ba (2015) Adam原始论文
