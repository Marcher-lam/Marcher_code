# 成本函数近似(CFA) 学习文档

> 在确定性优化模型中加入可调参数处理不确定性，是优化+学习的混合策略。

> 来源线索：本节内容根据原书中关于"Cost Function Approximation"的相关章节(Ch 11.4, Ch 13)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：CFA在确定性优化目标中引入修正参数$\theta$，通过调整修正项来间接处理随机性。

**形式化**：

$$X^{CFA}(S_t|\theta) = \arg\max_{x \in \mathcal{X}_t} \left\{C(S_t, x) + \theta^T \phi(S_t, x)\right\}$$

- $C(S_t, x)$：确定性成本函数
- $\theta^T\phi(S_t,x)$：可调修正项
- $\theta$：通过仿真或实际数据优化

**与其他策略类的对比**：

| 策略 | 核心思想 | 适用场景 |
|------|---------|---------|
| PFA | 直接参数化策略 | 简单规则策略 |
| CFA | 参数化修正项+优化 | 有优化结构的问题 |
| VFA | 近似值函数 | 长期决策 |
| DLA | 前瞻模拟 | 复杂不确定环境 |

**典型应用**：鲁棒最短路径（用$\theta$-百分位成本代替均值）是CFA。

## 4-8. 核心实现

```python
"""CFA：参数化修正的库存优化"""
import numpy as np

def cfa_policy(state, theta, capacity=50):
    """CFA策略：确定性优化+修正项"""
    inv, demand_forecast = state
    # 确定性决策：订到预测需求
    x_deterministic = max(0, demand_forecast - inv)
    # 修正项：theta[0]*安全库存 + theta[1]*库存水平因子
    correction = theta[0] * np.sqrt(demand_forecast) - theta[1] * inv / capacity
    x = max(0, min(capacity - inv, int(x_deterministic + correction)))
    return x

def evaluate_cfa(theta, n_sim=200):
    """仿真评估CFA参数"""
    total_reward = 0
    for _ in range(n_sim):
        inv = 20
        for t in range(20):
            demand = np.random.poisson(10)
            state = (inv, 12)  # 预测需求12
            order = cfa_policy(state, theta)
            sold = min(inv + order, demand)
            reward = 10*sold - 6*order - max(0, inv+order-sold)
            inv = max(0, inv + order - demand)
            total_reward += reward
    return total_reward / n_sim

if __name__ == "__main__":
    np.random.seed(42)
    # 网格搜索θ
    best_reward = -np.inf
    best_theta = None
    for t0 in np.arange(0, 5, 0.5):
        for t1 in np.arange(0, 2, 0.3):
            r = evaluate_cfa([t0, t1])
            if r > best_reward:
                best_reward = r
                best_theta = [t0, t1]
    print(f"最优θ: {best_theta}, 平均利润: {best_reward:.1f}")
```

## 9-14. 简要

### 12. 学习总结
CFA：$\arg\max_x\{C(S,x) + \theta^T\phi(S,x)\}$。在确定性优化上加可调修正项。适合已有优化模型但需要适应不确定性的场景。

### 13. 练习题
**Q1**：CFA和PFA的核心区别？
**A1**：PFA直接输出动作（$x = f(S|\theta)$），CFA在优化问题内修正目标（$\arg\max C + \theta^T\phi$）。CFA保留了优化结构，PFA可能违反约束。

### 14. 学习路径
**前置**：PFA、确定性优化 | **进阶**：VFA、DLA
**资源**：原书Ch 11.4, Ch 13
