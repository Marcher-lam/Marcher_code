# LSPE 最小二乘策略评估 学习文档

> 用最小二乘投影迭代更新值函数，是LSTD和TD(0)的折中。

> 来源线索：本节内容根据原书中关于"LSPE"的相关章节(Ch 16.3.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：LSPE(Least Squares Policy Evaluation)在每次迭代中用最小二乘投影更新值函数参数，收敛速度介于TD(0)和LSTD之间。

**更新**：

$$\theta^{n+1} = \arg\min_\theta \sum_{m=1}^n \left(\phi(s^m)^T\theta - [r^m + \gamma\phi(s^{m+1})^T\theta^n]\right)^2$$

每步求解一个最小二乘问题（而非像TD那样单步梯度）。

## 4-8. 核心实现

```python
"""LSPE：最小二乘策略评估"""
import numpy as np

class LSPE:
    def __init__(self, n_features, gamma=0.95):
        self.gamma = gamma
        self.theta = np.zeros(n_features)
        self.phis = []
        selfphis_next = []
        self.rewards = []

    def update(self, phi_s, r, phi_s_next):
        self.phis.append(phi_s)
        self.phis_next.append(phi_s_next)
        self.rewards.append(r)
        # 求解LS
        Phi = np.array(self.phis)
        targets = np.array(self.rewards) + self.gamma * (np.array(self.phis_next) @ self.theta)
        self.theta = np.linalg.lstsq(Phi, targets, rcond=None)[0]

if __name__ == "__main__":
    np.random.seed(42)
    lspe = LSPE(n_features=3)
    for n in range(500):
        s = np.random.randint(16)
        phi = np.array([1.0, s/16, (s/16)**2])
        s_next = np.random.randint(16)
        phi_next = np.array([1.0, s_next/16, (s_next/16)**2])
        r = 1.0 if s_next == 15 else -0.01
        lspe.update(phi, r, phi_next)
    print(f"LSPE参数: {lspe.theta.round(4)}")
```

## 9-14. 简要

### 12. 学习总结
LSPE：每次迭代用所有样本做最小二乘投影。折中LSTD（一次求解）和TD(0)（逐步更新）。

### 13. 练习题
**Q1**：LSPE和LSTD的区别？
**A1**：LSTD只求解一次$\theta=A^{-1}b$。LSPE每次迭代重解最小二乘，$\theta$逐渐收敛。LSPE更稳定但计算量更大。

### 14. 学习路径
**前置**：LSTD、TD学习 | **进阶**：GTD、Off-policy LSTD
**资源**：原书Ch 16.3.4
