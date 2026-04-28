# Trigg公式 学习文档

> 用追踪信号监控估计质量，当追踪信号偏离时自动调整步长。

> 来源线索：本节内容根据原书中关于"Trigg公式"的相关章节(Ch 6.2.3.2)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Trigg公式用指数平滑误差的追踪信号$T^n = |E^n|/MAD^n$来自适应调整步长。

**公式**：

平滑误差：$E^n = \beta \epsilon^n + (1-\beta)E^{n-1}$

平均绝对偏差：$MAD^n = \beta|\epsilon^n| + (1-\beta)MAD^{n-1}$

追踪信号：$T^n = |E^n|/MAD^n \in [0, 1]$

自适应步长：$\alpha_n = T^n \cdot \alpha_{max}$

**含义**：当估计偏差系统性地朝一个方向（$T^n$接近1），增大步长快速纠正；当随机震荡（$T^n$接近0），减小步长保持稳定。

## 4-8. 核心实现

```python
"""Trigg公式：追踪信号自适应步长"""
import numpy as np

class TriggStepsize:
    def __init__(self, beta=0.1, alpha_max=0.5):
        self.beta = beta
        self.alpha_max = alpha_max
        self.E = 0.0
        self.MAD = 0.0
        self.n = 0

    def get_alpha(self, error):
        self.n += 1
        self.E = self.beta * error + (1-self.beta) * self.E
        self.MAD = self.beta * abs(error) + (1-self.beta) * self.MAD
        if self.MAD < 1e-10:
            return self.alpha_max
        T = abs(self.E) / self.MAD
        return min(T * self.alpha_max, self.alpha_max)

if __name__ == "__main__":
    np.random.seed(42)
    ts = TriggStepsize(beta=0.1, alpha_max=0.5)
    theta = 0.0
    for n in range(200):
        true_mu = 5.0 if n < 100 else 8.0  # t=100时跳变
        w = true_mu + np.random.randn()
        error = w - theta
        alpha = ts.get_alpha(error)
        theta += alpha * error
        if n in [99, 100, 110, 150, 199]:
            print(f"n={n}: θ={theta:.2f}, α={alpha:.4f}")
```

## 9-14. 简要

### 12. 学习总结
Trigg公式：$\alpha_n = T^n \cdot \alpha_{max}$，追踪信号$T=|E|/MAD$。系统性偏差时增大步长，随机震荡时减小。

### 13. 练习题
**Q1**：$\beta$参数的作用是什么？
**A1**：$\beta$控制追踪信号的响应速度。$\beta$大→快速响应变化但噪声大；$\beta$小→平滑但延迟。通常$\beta \in [0.05, 0.2]$。

### 14. 学习路径
**前置**：指数平滑 | **进阶**：BAKF步长策略
**资源**：原书Ch 6.2.3.2
