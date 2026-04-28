# AdaGrad 学习文档

> 累积历史梯度平方自适应调整每维步长，频繁更新的参数步长更小。

> 来源线索：本节内容根据原书中关于"AdaGrad"的相关章节(Ch 6.2.3.5)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：AdaGrad根据参数维度$i$的历史梯度平方和$G_i^n$自适应缩放步长——梯度大的维度步长自动减小。

**更新公式**：

$$\theta_i^{n+1} = \theta_i^n - \frac{\eta}{\sqrt{G_i^n + \epsilon}} g_i^n$$

其中$G_i^n = \sum_{m=1}^n (g_i^m)^2$，$\epsilon$防止除零。

**直觉**：稀疏特征（如NLP中的罕见词）获得更大步长（$G_i$小），频繁特征步长衰减（$G_i$大）。

**问题**：$G_i^n$单调递增，步长只会越来越小，可能在最优点之前就停滞。RMSProp通过指数移动平均解决。

## 4-8. 核心实现

```python
"""AdaGrad优化器"""
import numpy as np

class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr, self.eps = lr, eps
        self.G = None

    def step(self, theta, grad):
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.G += grad**2
        theta -= self.lr * grad / (np.sqrt(self.G) + self.eps)
        return theta

if __name__ == "__main__":
    np.random.seed(42)
    f = lambda t: t[0]**2 + 100*t[1]**2  # 椭圆函数
    grad_f = lambda t: np.array([2*t[0], 200*t[1]])
    theta = np.array([10.0, 10.0])
    opt = AdaGrad(lr=1.0)
    for i in range(100):
        theta = opt.step(theta, grad_f(theta))
        if (i+1) % 20 == 0:
            print(f"Step {i+1}: θ={theta.round(4)}, f={f(theta):.4f}")
```

## 9-14. 简要

### 12. 学习总结
AdaGrad：$\theta \leftarrow \theta - \frac{\eta}{\sqrt{G+\epsilon}}\nabla f$。每维自适应步长，稀疏特征步长大。缺点：步长单调递减可能过早停滞。

### 13. 练习题
**Q1**：为什么AdaGrad适合稀疏数据？
**A1**：稀疏特征出现少，$G_i$小，步长$\eta/\sqrt{G_i}$大，能快速学习。频繁特征$G_i$大，步长自动衰减。

### 14. 学习路径
**前置**：SGD | **进阶**：RMSProp、Adam
**资源**：原书Ch 6.2.3.5、Duchi et al. (2011)
