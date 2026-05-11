# REINFORCE算法 学习文档

> 最基本的策略梯度算法，用蒙特卡洛回报直接优化策略参数。

> 来源线索：本节内容根据原书中关于"REINFORCE"的相关章节(Ch 12.8)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：REINFORCE用完整轨迹的回报$G_t$作为策略梯度的权重，是最简单的策略梯度算法。

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi(a|s;\theta) \cdot G_t\right]$$

**REINFORCE更新**：

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot G_t$$

其中$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k+1}$是折扣回报。

**带基线**：减去基线$b(s_t)$减小方差：

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot (G_t - b(s_t))$$

基线不影响期望（$\mathbb{E}[\nabla\log\pi \cdot b] = 0$），但大幅减小方差。

## 4-8. 核心实现

```python
"""REINFORCE算法"""
import numpy as np

class REINFORCE:
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99):
        self.nA = n_actions
        self.lr = lr
        self.gamma = gamma
        # 策略参数：每个状态的动作偏好
        self.theta = np.random.randn(n_states, n_actions) * 0.01

    def get_probs(self, s):
        exp_t = np.exp(self.theta[s] - self.theta[s].max())
        return exp_t / exp_t.sum()

    def select(self, s):
        return np.random.choice(self.nA, p=self.get_probs(s))

    def update(self, trajectory):
        """trajectory: [(s, a, r), ...]"""
        G = 0
        returns = []
        for _, _, r in reversed(trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        for (s, a, _), G in zip(trajectory, returns):
            probs = self.get_probs(s)
            grad = -probs  # ∇log π(a|s) 对所有a
            grad[a] += 1.0  # 选中的a加1
            self.theta[s] += self.lr * G * grad
        return returns[0]

if __name__ == "__main__":
    np.random.seed(42)
    agent = REINFORCE(16, 4, lr=0.01)
    for ep in range(500):
        s = np.random.randint(16)
        traj = []
        for step in range(20):
            a = agent.select(s)
            s_next = np.random.randint(16)
            r = 1.0 if s_next == 15 else -0.01
            traj.append((s, a, r))
            s = s_next
            if s == 15: break
        ret = agent.update(traj)
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}: 回报={ret:.2f}")
```

## 9-14. 简要

### 12. 学习总结
REINFORCE：$\theta \leftarrow \theta + \alpha \nabla\log\pi(a|s) \cdot G_t$。最简单的策略梯度，方差大但无偏。加基线减小方差。

### 13. 练习题
**Q1**：为什么基线$b(s)$不改变梯度的期望？
**A1**：$\mathbb{E}[\nabla\log\pi(a|s)b(s)] = b(s)\sum_a \nabla\pi(a|s) = b(s)\nabla\sum_a\pi(a|s) = b(s)\nabla 1 = 0$。概率和恒为1，梯度为0。

### 14. 学习路径
**前置**：策略梯度定理 | **进阶**：Actor-Critic、PPO、A3C
**资源**：原书Ch 12.8、Williams (1992)
