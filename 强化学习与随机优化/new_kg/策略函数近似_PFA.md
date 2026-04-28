# 策略函数近似(PFA) 学习文档

> 用参数化函数直接映射状态到动作，是最直觉的策略设计方法。

> 来源线索：本节内容根据原书中关于"Policy Function Approximations"的相关章节(Ch 11.3, Ch 12)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：PFA直接寻找策略$X^\pi(S|\theta)$的参数$\theta$，使得$\max_\theta \mathbb{E}\sum_t C(S_t, X^\pi(S_t|\theta))$。

**形式化**：

$$X^{PFA}(S_t|\theta) = \arg\max_{x \in \mathcal{X}_t} U_t(x|S_t, \theta)$$

其中$U_t$是参数化的效用函数。

**典型PFA策略**：
- **库存$(s,S)$策略**：$x = S - R$ if $R < s$, else $x = 0$
- **Boltzmann策略**：$\mathbb{P}(a|s) \propto e^{Q(s,a)/\tau}$
- **仿射策略**：$x = \theta_0 + \theta_1 s_1 + \theta_2 s_2$

**优化方法**：策略梯度（REINFORCE）、有限差分SGD、SPSA。

**与其他策略类的关系**：
- CFA：在确定性优化上加 tunable 参数
- VFA：先学值函数，再从中提取策略
- DLA：前瞻模拟做决策

## 4-8. 核心实现

```python
"""PFA：(s,S)库存策略参数优化"""
import numpy as np

def simulate_policy(params, n_episodes=100, T=20):
    """模拟(s,S)策略"""
    s, S = params
    total_reward = 0
    for _ in range(n_episodes):
        inv = 20
        for t in range(T):
            order = max(0, min(S - inv, 50 - inv)) if inv < s else 0
            demand = np.random.poisson(10)
            sold = min(inv + order, demand)
            reward = 10 * sold - 6 * order - 1 * max(0, inv + order - sold)
            inv = max(0, inv + order - sold)
            total_reward += reward
    return total_reward / n_episodes

def spsa_optimize(f, theta0, n_iter=100, alpha=0.1):
    """SPSA优化PFA参数"""
    theta = np.array(theta0, dtype=float)
    for n in range(n_iter):
        delta = np.random.choice([-1, 1], size=len(theta))
        f_plus = f(theta + alpha * delta)
        f_minus = f(theta - alpha * delta)
        grad = (f_plus - f_minus) / (2 * alpha * delta)
        theta += (1.0 / (n+1)) * grad
    return theta

if __name__ == "__main__":
    np.random.seed(42)
    best_params = spsa_optimize(simulate_policy, [5.0, 30.0], n_iter=50)
    print(f"最优(s,S)参数: s={best_params[0]:.1f}, S={best_params[1]:.1f}")
    print(f"对应平均利润: {simulate_policy(best_params):.1f}")
```

## 9-14. 简要

### 12. 学习总结
PFA：$X^\pi(S|\theta)$直接参数化策略。库存$(s,S)$、Boltzmann、仿射策略都是PFA。用SGD/SPSA优化参数。

### 13. 练习题
**Q1**：PFA相比VFA的优势？
**A1**：PFA直接优化策略（不需要先学值函数），参数少、计算快。缺点是需要预先指定策略形式（可能不是最优策略类）。

### 14. 学习路径
**前置**：MDP、策略梯度 | **进阶**：CFA、VFA、DLA
**资源**：原书Ch 11.3, Ch 12
