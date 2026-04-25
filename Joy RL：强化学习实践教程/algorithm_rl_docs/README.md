# 强化学习算法知识库

> 包含35+核心强化学习算法的完整学习文档，基于《Joy RL：强化学习实践教程》

---

## 📚 内容概览

| 类别 | 数量 | 说明 |
|------|------|------|
| 基础理论 | 6 | MDP、DP、MC、TD |
| Q学习系列 | 10 | Q、Sarsa、DQN及各种变体 |
| 策略梯度 | 7 | REINFORCE、PPO、A2C |
| 连续控制 | 4 | DDPG、TD3、SAC |
| 分布式 | 3 | IMPALA、Ape-X等 |
| 补充资源 | 5 | 数学推导、选择指南 |

---

## 📖 学习路径

### 入门路径
```
Q学习 → DQN → PPO → SAC
```

### 进阶路径
```
Q学习 → DoubleDQN → Rainbow → SAC
```

---

## 🏆 推荐算法

| 任务 | 推荐 |
|------|------|
| 入门学习 | Q学习 |
| Atari游戏 | Rainbow DQN |
| 连续控制 | SAC |
| 通用场景 | PPO |

---

## 快速开始

```python
# Q学习（最简单）
import numpy as np
import gymnasium as env

Q = np.zeros((500, 6))
for episode in range(1000):
    s = env.reset()
    while True:
        a = np.argmax(Q[s]) if np.random()>0.1 else env.action_space.sample()
        s2, r, d, _ = env.step(a)
        Q[s,a] += 0.1*(r + 0.99*max(Q[s2]) - Q[s,a])
        if d: break
```

```python
# PPO（推荐，生产环境）
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1").learn(50000)
```

---

*共35+文档，持续更新*