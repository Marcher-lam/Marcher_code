# Kesten规则 学习文档

> 根据符号变化次数自适应调整步长，在搜索方向震荡时减小步长。

> 来源线索：本节内容根据原书中关于"Kesten规则"的相关章节(Ch 6.2.3.1)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Kesten规则在连续符号变化（震荡）时减小步长，在方向一致时保持步长不变。

**直觉**：如果梯度方向反复变号，说明步长太大在跳来跳去，应该减小。如果方向一致，说明还在收敛途中，可以保持。

**规则**：维护计数器$K^n$（符号变化次数），步长为：

$$\alpha_n = \frac{a}{b + K^n}$$

其中$K^n$在$(\theta^{n+1}-\theta^n)(\theta^n-\theta^{n-1}) < 0$（符号变化）时递增。

## 4-8. 核心实现

```python
"""Kesten规则：自适应步长"""
import numpy as np

class KestenStepsize:
    def __init__(self, a=1.0, b=1):
        self.a, self.b = a, b
        self.K = 0
        self.prev_delta = None

    def update(self, delta):
        """delta = θ^n - θ^{n-1}"""
        if self.prev_delta is not None and delta * self.prev_delta < 0:
            self.K += 1  # 符号变化，增加计数
        self.prev_delta = delta
        return self.a / (self.b + self.K)

if __name__ == "__main__":
    np.random.seed(42)
    ks = KestenStepsize(a=1.0, b=1)
    theta, true_mu = 0.0, 5.0
    for n in range(100):
        old_theta = theta
        alpha = ks.update(theta - old_theta) if n > 0 else 1.0
        w = true_mu + np.random.randn()
        theta = (1-alpha)*theta + alpha*w
        alpha = ks.update(theta - old_theta)
    print(f"Kesten规则: 最终θ={theta:.3f}, K={ks.K}")
```

## 9-14. 简要

### 12. 学习总结
Kesten规则：$\alpha_n = a/(b+K^n)$，$K^n$跟踪符号变化次数。震荡时自动减小步长，收敛时保持步长。

### 13. 练习题
**Q1**：Kesten规则相比调和步长$1/n$有什么优势？
**A1**：Kesten只在确实需要减小步长时（检测到震荡）才减小，而不是盲目递减。在收敛方向一致时保持较大步长，学习更快。

### 14. 学习路径
**前置**：调和步长 | **进阶**：Trigg公式、BAKF步长
**资源**：原书Ch 6.2.3.1
