# 值函数近似(VFA) 学习文档

> 用参数化函数近似值函数，是大规模MDP的核心求解策略。

> 来源线索：本节内容根据原书中关于"Value Function Approximation"的相关章节(Ch 11.5, Ch 16-17)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：VFA用$\bar{V}(s|\theta)$近似真实值函数$V^*(s)$，通过TD学习或回归方法训练参数$\theta$。

**核心形式**：

$$\bar{V}(s|\theta) = \theta^T \phi(s)$$

或非线性形式：$\bar{V}(s|\theta) = f_\theta(s)$（神经网络）

**训练目标**：最小化Bellman误差：

$$\min_\theta \sum_s \left(\bar{V}(s|\theta) - [r(s) + \gamma \mathbb{E}\bar{V}(s'|\theta)]\right)^2$$

**方法分类**：
- **TD学习**：在线更新$\theta \leftarrow \theta + \alpha \delta \nabla\bar{V}(s|\theta)$
- **拟合值迭代**：定期用回归重新拟合
- **LSTD/LSPE**：用最小二乘法批量估计

**VFA在原书中的地位**：是四种策略类之一（PFA/CFA/VFA/DLA），覆盖TD学习(Ch 16)、Q-Learning(Ch 17)、深度RL。

## 4-8. 核心实现

```python
"""VFA：线性值函数近似 + TD学习"""
import numpy as np

class LinearVFA:
    """线性值函数近似"""
    def __init__(self, n_features, lr=0.01, gamma=0.95):
        self.theta = np.random.randn(n_features) * 0.01
        self.lr = lr
        self.gamma = gamma

    def V(self, features):
        return self.theta @ features

    def update(self, features, reward, next_features, done):
        td_target = reward + self.gamma * self.V(next_features) * (1-done)
        td_error = td_target - self.V(features)
        self.theta += self.lr * td_error * features
        return abs(td_error)

def state_features(s, n_states=16):
    """状态特征：one-hot + 位置编码"""
    f = np.zeros(n_states + 2)
    f[s] = 1.0
    f[-2] = s / n_states
    f[-1] = (s / n_states) ** 2
    return f

if __name__ == "__main__":
    np.random.seed(42)
    vfa = LinearVFA(n_features=18, lr=0.01)
    for ep in range(1000):
        s = np.random.randint(16)
        for step in range(30):
            s_next = np.random.randint(16)
            r = 1.0 if s_next == 15 else -0.01
            done = s_next == 15
            err = vfa.update(state_features(s), r, state_features(s_next), done)
            s = s_next
            if done: break
    values = [vfa.V(state_features(s)) for s in range(16)]
    print(f"VFA估计的状态值: {[f'{v:.2f}' for v in values]}")
```

## 9-14. 简要

### 12. 学习总结
VFA：$\bar{V}(s|\theta) = \theta^T\phi(s)$，TD更新$\theta \leftarrow \theta + \alpha\delta\nabla\bar{V}$。大规模MDP的核心方法，深度RL的基础。

### 13. 练习题
**Q1**：线性VFA的致命问题是什么？
**A1**：线性特征无法表达复杂函数。如果$\phi(s)$选择不当，近似误差很大。深度神经网络VFA(DQN)通过自动学习特征解决这个问题。

### 14. 学习路径
**前置**：TD学习、线性回归 | **进阶**：DQN、LSTD、深度VFA
**资源**：原书Ch 11.5, Ch 16-17、Sutton & Barto Ch 9
