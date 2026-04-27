# 时序差分 (TD) 学习文档

> 结合 MC 的采样和 DP 的自举——每步都能学习的强化学习方法。

> 来源线索：本节内容根据原书中关于"时序差分"的相关章节（第13章13.5.2节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** TD 学习在每一步都用当前奖励加上下一状态的估计价值来更新当前状态价值，无需等待回合结束，是 MC 和动态规划的折中。

**直觉类比：** MC 像"考完期末考试才知道总分"，TD 像"每次小测验后就知道自己大概水平"。TD 不等回合结束，每走一步就根据即时奖励和下一步的估值来修正当前的估值。

**历史背景：** TD 学习由 Sutton 于 1988 年提出，是强化学习最核心的方法之一。TD(0) 是最简单的形式，TD(λ) 引入了资格迹。

**算法定位：** 强化学习、model-free、在线学习。

**前置知识：** MDP、蒙特卡罗方法、价值函数。

---

## 2-3. 核心原理与数学公式

### TD 更新规则

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

其中 $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ 称为 **TD 误差**。

### SARSA（同策略 TD 控制）

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

### Q-Learning（异策略 TD 控制）

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

---

## 4-8. 代码实现

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states=16, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return self.Q[state].argmax()

    def update(self, s, a, r, s_next):
        td_target = r + self.gamma * self.Q[s_next].max()
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error
        return td_error

agent = QLearningAgent(n_states=16, n_actions=4)
for step in range(1000):
    s = np.random.randint(16)
    a = agent.choose_action(s)
    s_next = np.random.randint(16)
    r = 1.0 if s_next == 15 else -0.01
    agent.update(s, a, r, s_next)
print(f"Q值表:\n{agent.Q.round(2)}")
```

---

## 9-14. 练习与路径

**题1：** Q-Learning 为什么是"异策略"（off-policy）？

**参考答案：** Q-Learning 的更新目标使用 $\max_{a'} Q(s', a')$（贪心动作），而非实际执行的动作。这意味着它可以用任何策略（如 ε-greedy）收集数据，但学习的是最优策略的 Q 值。行为策略和目标策略可以不同，所以是"异策略"。

### 学习路径
- 前置：MDP、蒙特卡罗方法
- 进阶：SARSA(λ)、DQN、Double Q-Learning
