# Adam优化算法 学习文档

## 1. 算法基础认知
自适应学习率的梯度下降算法，结合了动量和RMSProp。

## 2. 核心原理
计算一阶矩和二阶矩的指数移动平均。

## 3. 数学公式
$m_t = eta_1 m_{t-1} + (1-eta_1)g_t$
$v_t = eta_2 v_{t-1} + (1-eta_2)g_t^2$
$	heta_t = 	heta_{t-1} - lpha rac{m_t}{\sqrt{v_t} + \epsilon}$

## 4. 训练过程
自适应调整每个参数的学习率。

## 5. 应用场景
几乎所有深度学习训练

## 6. 优缺点
优点：收敛快、参数少；缺点：可能泛化性差

## 7. 代码实现
```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 8. 手工实现
```python
import numpy as np

class AdamScratch:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
```
