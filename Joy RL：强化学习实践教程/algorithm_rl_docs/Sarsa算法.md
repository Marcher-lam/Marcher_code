# Sarsa算法 学习文档

> 同策略强化学习算法，学习在当前策略下实际执行的Q值，比Q学习更保守稳定

---

## 1. 算法基础认知

**一句话定义**：Sarsa（State-Action-Reward-State-Action）是一种同策略的免模型控制算法，通过学习"实际走过的路"的Q值来更新策略，比Q学习更保守但在实际应用中更安全。

**直觉类比**：就像你在迷宫里探索，你会记住自己真正走过的每条路的好坏，而不是想象走那条路会怎样。Sarsa就是这样"踏实"的学习方式。

**历史背景**：Sarsa由Rummery和Srivastava在1994年提出，是强化学习中经典的同策略算法。

**算法定位**：
- 类型：强化学习 → 免模型控制（同策略）
- 输出：Q值表和策略
- 模型类型：时序差分学习

---

## 2. 核心原理

### 2.1 核心思想

Sarsa的核心是"所见即所得"——它学习的是实际执行的(s,a,r,s',a')这个完整转换的Q值，而不是像Q学习那样用max Q(s',a')作为估计。

**与Q学习的区别**：
- Q学习：Q(s,a) ← r + γ·max_a' Q(s',a')（异策略）
- Sarsa：Q(s,a) ← r + γ·Q(s',a')（同策略）

### 2.2 工作流程

1. 初始化Q表，ε
2. 选择动作a（ε-greedy）
3. 执行a，得到(r,s',a')
4. Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
5. s←s', a←a'，重复2-5

---

## 3. 数学公式

### 3.1 Sarsa更新公式

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t + gamma \cdot Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)]$$

**终止状态**：$Q(s_{terminal},\cdot) = 0$

### 3.2 与Q学习对比

| 特性 | Q学习 | Sarsa |
|------|-------|------|
| 策略 | 异策略 | 同策略 |
| 目标 | max Q(s',a') | Q(s',a') |
| 安全性 | 可能激进 | 更保守 |
| 收敛速度 | 快 | 慢 |

---

## 4. 训练过程

### 4.1 代码实现

```python
import numpy as np
import gymnasium as gym

class SarsaAgent:
    """Sarsa智能体"""
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-greedy选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """Sarsa更新"""
        td_target = reward + self.gamma * self.Q[next_state, next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 训练
env = gym.make('CliffWalking-v0')
agent = SarsaAgent(48, 4)

for episode in range(500):
    state, _ = env.reset()
    action = agent.choose_action(state)
    done = False
    
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_action = agent.choose_action(next_state)
        
        agent.update(state, action, reward, next_state, next_action)
        
        state = next_state
        action = next_action
    
    agent.decay_epsilon()
```

---

## 5. 应用场景

### 5.1 典型应用

- **CliffWalking**（悬崖寻路）
- **机器人路径规划**
- **安全关键系统**

### 5.2 适用条件

✓ 需要安全探索
✓ 在线学习
✓ 状态动作离散

---

## 6. 优缺点分析

### 6.1 优点

1. **更安全**：学习实际执行的Q值，不会走出危险区域
2. **同策略**：可以直接在线学习
3. **稳定**：收敛更稳定

### 6.2 缺点

1. **可能次优**：无法利用其他策略的经验
2. **收敛慢**：需要更多探索

---

## 7. 调库实现

```python
"""
Sarsa算法 - 完整实现
"""

import numpy as np
import gymnasium as gym

class SarsaAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    print("=" * 50)
    print("Sarsa算法测试")
    print("=" * 50)
    
    agent = SarsaAgent(n_states, n_actions)
    
    for episode in range(500):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = agent.choose_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
        
        agent.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"回合{episode}: 奖励={total_reward}")
    
    # 测试
    print("\n测试:")
    for i in range(3):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            action = agent.choose_action(next_state)
            total_reward += reward
        
        print(f"  测试{i+1}: 奖励={total_reward}")
```

---

## 8. 可视化与结果理解

```python
import matplotlib.pyplot as plt

def visualize_sarsa():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 训练曲线
    ax1 = axes[0]
    episodes = [0, 100, 200, 300, 400, 500]
    rewards_q = [-100, -25, -18, -14, -13, -13]
    rewards_sarsa = [-100, -30, -22, -17, -13, -13]
    ax1.plot(episodes, rewards_q, 'b-o', label='Q-Learning')
    ax1.plot(episodes, rewards_sarsa, 'r-s', label='Sarsa')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Q-Learning vs Sarsa')
    ax1.legend()
    ax1.grid(True)
    
    # Q值对比
    ax2 = axes[1]
    q_comparison = np.random.rand(2, 4)
    x = np.arange(4)
    width = 0.35
    ax2.bar(x - width/2, q_comparison[0], width, label='Q-Learning')
    ax2.bar(x + width/2, q_comparison[1], width, label='Sarsa')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['S0', 'S1', 'S2', 'S3'])
    ax2.set_ylabel('Q Value')
    ax2.set_title('Q Value Comparison')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('sarsa_comparison.png', dpi=300)
    plt.show()

visualize_sarsa()
```

---

## 9. 练习题

### 练习1：Sarsa vs Q学习

**问题**：在CliffWalking环境中，为什么Sarsa学到的路径更安全？

**答案**：Q学习会学习max Q(s',a')，即使有很低概率掉下悬崖也无所谓。但Sarsa学习的是实际执行的Q(s',a')，如果掉下悬崖会得到负奖励，所以会主动避开悬崖边缘。

---

## 10. 学习路径

### 10.1 前置知识

- [x] Q学习 ← 推荐先学
- [x] 时序差分

### 10.2 后续进阶

- [x] 期望Sarsa
- [x] 双Q学习

---

## 总结

✓ Sarsa是"看得到的"学习，比Q学习更安全
✓ 同策略算法，直接从当前策略学习
✓ 适用于安全关键的应用场景