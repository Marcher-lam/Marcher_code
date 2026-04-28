# OSAVI步长策略 学习文档

> 针对近似值迭代的最优步长，在值函数估计误差和贝尔曼误差间取最优平衡。

> 来源线索：本节内容根据原书中关于"Optimal Stepsize for Approximate Value Iteration"的相关章节(Ch 6.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：OSAVI(Optimal Stepsize for Approximate Value Iteration)为值迭代的在线更新设计最优步长，平衡新观测的噪声和旧估计的偏差。

**核心问题**：值迭代中更新$\bar{V}^n(s) = (1-\alpha)\bar{V}^{n-1}(s) + \alpha[r + \gamma\bar{V}^{n-1}(s')]$。步长$\alpha$需要权衡：
- $\alpha$太大：噪声大（TD目标不稳定）
- $\alpha$太小：更新慢（Bellman误差传播慢）

**OSAVI最优步长**：

$$\alpha_n^{OSAVI} = \frac{\lambda^n \bar{\sigma}^{2,n-1}}{\lambda^n \bar{\sigma}^{2,n-1} + \sigma_\epsilon^2}$$

其中$\sigma_\epsilon^2$是TD误差的方差，$\bar{\sigma}^{2,n}$是值函数估计的方差，$\lambda$是遗忘因子。

**与BAKF的关系**：OSAVI是BAKF在值函数估计场景的特化应用，额外考虑了Bellman方程的特有结构（折扣因子$\gamma$的影响）。

## 4-8. 核心实现

```python
"""OSAVI步长策略"""
import numpy as np

class OSAVIStepsize:
    def __init__(self, sigma_td=1.0, gamma=0.95, lam=0.99):
        self.sigma_td = sigma_td
        self.gamma = gamma
        self.lam = lam

    def optimal_alpha(self, var_estimate, n):
        """计算最优步长"""
        return self.lam * var_estimate / (self.lam * var_estimate + self.sigma_td**2)

if __name__ == "__main__":
    np.random.seed(42)
    osavi = OSAVIStepsize(sigma_td=1.0, gamma=0.95)
    V = np.zeros(10)
    V_var = np.ones(10) * 10.0  # 初始方差

    for ep in range(500):
        s = np.random.randint(10)
        s_next = np.random.randint(10)
        r = 1.0 if s_next == 9 else -0.01
        td_target = r + 0.95 * V[s_next]
        td_error = td_target - V[s]

        alpha = osavi.optimal_alpha(V_var[s], ep)
        V[s] += alpha * td_error
        V_var[s] = (1-alpha) * V_var[s]

        if (ep+1) % 100 == 0:
            print(f"Ep {ep+1}: V={V.round(2)}, 平均α={osavi.optimal_alpha(V_var.mean(), ep):.4f}")
```

## 9-14. 简要

### 12. 学习总结
OSAVI：$\alpha = \lambda\bar{\sigma}^2/(\lambda\bar{\sigma}^2 + \sigma_\epsilon^2)$。为值函数在线更新设计的最优步长，是BAKF在RL中的特化。

### 13. 练习题
**Q1**：为什么RL中的步长选择比普通SGD更困难？
**A1**：RL中TD目标是"移动靶"（non-stationary）——它依赖于被更新的值函数本身。因此步长不仅要处理观测噪声，还要处理目标的非平稳性（Bellman误差传播）。

### 14. 学习路径
**前置**：BAKF、值迭代 | **进阶**：TD学习中的自适应步长
**资源**：原书Ch 6.4
