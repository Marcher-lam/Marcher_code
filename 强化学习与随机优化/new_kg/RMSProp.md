# RMSProp 学习文档

> 用指数移动平均替代累积和解决AdaGrad步长过早衰减问题。

> 来源线索：本节内容根据原书中关于"RMSProp"的相关章节(Ch 6.2.3.6)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：RMSProp用梯度平方的指数移动平均$E[g^2]$代替AdaGrad的累积和，使步长不会无限衰减。

**更新公式**：

$$E[g^2]^{n+1} = \rho E[g^2]^n + (1-\rho)(g^n)^2$$
$$\theta^{n+1} = \theta^n - \frac{\eta}{\sqrt{E[g^2]^{n+1} + \epsilon}} g^n$$

**与AdaGrad的区别**：
- AdaGrad：$G^n = \sum_{m=1}^n (g^m)^2$（无限累积）
- RMSProp：$E[g^2]^n \approx \rho E[g^2]^{n-1} + (1-\rho)(g^n)^2$（遗忘因子$\rho$）

$\rho \approx 0.9$时有效窗口约10步，步长保持活跃。

## 4-8. 核心实现

```python
"""RMSProp优化器"""
import numpy as np

class RMSProp:
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps = lr, rho, eps
        self.Eg2 = None

    def step(self, theta, grad):
        if self.Eg2 is None:
            self.Eg2 = np.zeros_like(theta)
        self.Eg2 = self.rho * self.Eg2 + (1-self.rho) * grad**2
        theta -= self.lr * grad / (np.sqrt(self.Eg2) + self.eps)
        return theta

if __name__ == "__main__":
    np.random.seed(42)
    f = lambda t: t[0]**2 + 100*t[1]**2
    grad_f = lambda t: np.array([2*t[0], 200*t[1]])
    theta = np.array([10.0, 10.0])
    opt = RMSProp(lr=0.01)
    for i in range(200):
        theta = opt.step(theta, grad_f(theta))
        if (i+1) % 50 == 0:
            print(f"Step {i+1}: θ={theta.round(4)}, f={f(theta):.6f}")
```

## 9-14. 简要

### 12. 学习总结
RMSProp：$E[g^2] \leftarrow \rho E[g^2] + (1-\rho)g^2$，步长$\propto \eta/\sqrt{E[g^2]}$。解决了AdaGrad步长单调递减的问题，是Adam的核心组件之一。

### 13. 练习题
**Q1**：$\rho$接近1和接近0分别意味着什么？
**A1**：$\rho$接近1→长记忆（类似AdaGrad），步长变化慢；$\rho$接近0→短记忆，步长快速适应当前梯度。通常$\rho=0.9$。

### 14. 学习路径
**前置**：AdaGrad | **进阶**：Adam优化器
**资源**：原书Ch 6.2.3.6、Tieleman & Hinton (2012)
