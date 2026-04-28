# SARSA 学习文档

> SARSA是在策略的时序差分控制算法，因更新元组(S,A,R,S',A')而得名，比Q-Learning更安全但更保守。

> 来源线索：本节内容根据原书中关于Q-Learning和TD控制的相关章节(Ch 17.2-17.5)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：SARSA是一种在策略(on-policy)的TD控制算法，用实际执行动作的Q值（而非最大Q值）来更新，因此策略的探索行为会直接影响学习。

**直觉类比**：Q-Learning像一个乐观的规划者——它假设未来总是选最优动作。SARSA像一个谨慎的执行者——它考虑自己实际会做的动作（包括探索时的随机选择）。在悬崖边上，Q-Learning学到的最优路径贴着悬崖走（假设不犯错），但实际执行时可能因探索掉下去。SARSA学到更安全的路线。

**历史背景**：SARSA由Rummery & Niranjan (1994)和Sutton (1996)独立发展。名称来自五元组$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**算法定位**：在策略TD控制。与Q-Learning（离策略）对应。

## 2. 核心原理

**更新公式**：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**与Q-Learning的区别**：
- Q-Learning：TD目标 = $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$（最优动作的Q值）
- SARSA：TD目标 = $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$（实际选择的动作的Q值）

## 3-8. 核心实现

```python
"""SARSA vs Q-Learning对比"""
import numpy as np

class SARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, a_next, done):
        td_target = r + self.gamma * self.Q[s_next, a_next] * (1 - done)
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    n_states, n_actions = 16, 4
    agent = SARSAAgent(n_states, n_actions)
    # 简单训练
    for ep in range(1000):
        s = np.random.randint(n_states)
        a = agent.choose_action(s)
        for step in range(20):
            s_next = np.random.randint(n_states)
            r = 1.0 if s_next == 15 else -0.01
            done = (s_next == 15) or step >= 19
            a_next = agent.choose_action(s_next)
            agent.update(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next
            if done: break
    print(f"SARSA训练完成, Q值范围: [{agent.Q.min():.2f}, {agent.Q.max():.2f}]")
```

## 9-14. 简要

### 12. 学习总结
SARSA：$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$。在策略，考虑探索风险。

### 13. 练习题
**Q1**：在确定性环境中，SARSA和Q-Learning的结果会一样吗？
**A1**：如果策略是贪心的（ε=0），两者完全相同，因为$a' = \arg\max Q(s',\cdot)$。

### 14. 学习路径
**前置**：Q-Learning | **进阶**：SARSA(λ)、Expected SARSA
**资源**：原书Ch 17、Sutton & Barto Ch 6.4
