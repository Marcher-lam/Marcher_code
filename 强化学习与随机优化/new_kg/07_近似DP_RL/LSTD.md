# LSTD 最小二乘时间差分 学习文档

> 用最小二乘法直接求解TD不动点，比TD(0)更高效地利用样本。

> 来源线索：本节内容根据原书中关于"LSTD"的相关章节(Ch 16.3.3)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：LSTD(Least Squares Temporal Difference)用所有样本同时求解TD不动点$\theta = A^{-1}b$，而非逐步更新。

**公式**：

$$\theta^{LSTD} = \left(\sum_n \phi(s^n)(\phi(s^n) - \gamma\phi(s^{n+1}))^T\right)^{-1} \sum_n \phi(s^n) r^{n+1}$$

其中$A = \sum_n \phi(s^n)(\phi(s^n) - \gamma\phi(s^{n+1}))^T$，$b = \sum_n \phi(s^n) r^{n+1}$。

**与TD(0)的区别**：TD(0)用随机梯度逐步逼近，LSTD直接求解线性系统。LSTD收敛更快（样本效率高），但计算$O(p^2)$每步。

## 4-8. 核心实现

```python
"""LSTD：最小二乘时间差分"""
import numpy as np

class LSTD:
    def __init__(self, n_features, gamma=0.95, lam=1e-6):
        self.gamma = gamma
        self.A = np.eye(n_features) * lam  # 正则化
        self.b = np.zeros(n_features)

    def update(self, phi_s, reward, phi_s_next):
        self.A += np.outer(phi_s, phi_s - self.gamma * phi_s_next)
        self.b += phi_s * reward

    def solve(self):
        return np.linalg.solve(self.A, self.b)

if __name__ == "__main__":
    np.random.seed(42)
    lstd = LSTD(n_features=3)
    for _ in range(1000):
        s = np.random.randint(16)
        features = np.array([1.0, s/16, (s/16)**2])
        s_next = np.random.randint(16)
        f_next = np.array([1.0, s_next/16, (s_next/16)**2])
        r = 1.0 if s_next == 15 else -0.01
        lstd.update(features, r, f_next)
    theta = lstd.solve()
    print(f"LSTD参数: {theta.round(4)}")
    for s in [0, 5, 10, 15]:
        f = np.array([1.0, s/16, (s/16)**2])
        print(f"  V({s}) = {f@theta:.3f}")
```

## 9-14. 简要

### 12. 学习总结
LSTD：$\theta = A^{-1}b$，$A=\sum\phi(\phi-\gamma\phi')^T$，$b=\sum\phi r$。直接求解TD不动点，样本效率最高。

### 13. 练习题
**Q1**：LSTD的计算瓶颈在哪？
**A1**：每步需要更新$p \times p$矩阵$A$并最终求逆，$O(p^2)$每步，$O(p^3)$求逆。 Sherman-Morrison可用于增量更新$A^{-1}$。

### 14. 学习路径
**前置**：TD学习、线性回归 | **进阶**：LSPE、GTD
**资源**：原书Ch 16.3.3、Bradtke & Barto (1996)
