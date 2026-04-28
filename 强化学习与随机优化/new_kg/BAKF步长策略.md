# BAKF步长策略 学习文档

> 基于偏差校正Kalman滤波的最优步长，在估计偏差和测量噪声间取最优平衡。

> 来源线索：本节内容根据原书中关于"BAKF Stepsize"的相关章节(Ch 6.3.3)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：BAKF(Bias-Adjusted Kalman Filter)步长是Kalman滤波框架下的最优步长，同时考虑估计偏差和观测噪声。

**核心公式**：

在Kalman滤波框架中，最优步长（Kalman增益）为：

$$\alpha_n = \frac{\bar{\sigma}^{2,n-1}}{\bar{\sigma}^{2,n-1} + \sigma_W^2/\lambda^n}$$

其中$\bar{\sigma}^{2,n}$是估计方差，$\sigma_W^2$是观测噪声方差，$\lambda$是遗忘因子。

**偏差校正**：

当存在系统性偏差$\delta^n$时，修正步长：

$$\alpha_n^{BAKF} = \frac{\bar{\sigma}^{2,n-1} + |\delta^n|}{\bar{\sigma}^{2,n-1} + \sigma_W^2/\lambda^n + |\delta^n|}$$

偏差大时步长自动增大（快速纠正），偏差小时步长减小（精确估计）。

## 4-8. 核心实现

```python
"""BAKF步长策略"""
import numpy as np

class BAKFStepsize:
    def __init__(self, sigma_w=1.0, lam=1.0):
        self.sigma_w = sigma_w
        self.lam = lam
        self.sigma_est = 10.0  # 初始估计方差
        self.bias = 0.0

    def get_alpha(self, error):
        self.bias = 0.9 * self.bias + 0.1 * error  # 跟踪偏差
        alpha = (self.sigma_est + abs(self.bias)) / \
                (self.sigma_est + self.sigma_w**2/self.lam + abs(self.bias))
        # 更新估计方差
        self.sigma_est = (1-alpha) * self.sigma_est
        return alpha

if __name__ == "__main__":
    np.random.seed(42)
    bs = BAKFStepsize(sigma_w=2.0)
    theta = 0.0
    for n in range(200):
        true_mu = 5.0 if n < 100 else 8.0
        w = true_mu + 2*np.random.randn()
        error = w - theta
        alpha = bs.get_alpha(error)
        theta += alpha * error
        if n in [0, 50, 99, 105, 150, 199]:
            print(f"n={n:3d}: θ={theta:.2f}, α={alpha:.4f}, bias={bs.bias:.2f}")
```

## 9-14. 简要

### 12. 学习总结
BAKF步长：$\alpha = (\bar{\sigma}^2 + |\delta|)/(\bar{\sigma}^2 + \sigma_W^2/\lambda + |\delta|)$。Kalman滤波最优增益+偏差校正，是最优步长策略之一。

### 13. 练习题
**Q1**：BAKF相比固定步长的核心优势是什么？
**A1**：BAKF根据估计不确定性和系统性偏差自适应调整步长。偏差大时自动增大步长快速纠正（检测到跳变），偏差小时减小步长提高精度。固定步长无法区分这两种情况。

### 14. 学习路径
**前置**：Kalman滤波、遗忘因子 | **进阶**：OSAVI步长策略
**资源**：原书Ch 6.3.3
