# 强化学习基础

强化学习（Reinforcement Learning）是机器学习的重要分支，通过智能体与环境的交互学习最优策略。

## 🎯 核心概念

### 基本要素
- **智能体（Agent）**：学习和决策的主体
- **环境（Environment）**：智能体所处的外部世界
- **状态（State）**：环境的当前情况
- **动作（Action）**：智能体可以采取的行为
- **奖励（Reward）**：环境给予的反馈信号

### 强化学习循环
```
状态 → 智能体 → 动作 → 环境 → 奖励 + 新状态
  ↑                              ↓
  ←←←←←←←←←←←←←←←←←←←←←←←←←
```

## 📚 主要算法

### 1. 基于价值的方法
- Q-Learning
- DQN（Deep Q-Network）
- Double DQN
- Dueling DQN

### 2. 基于策略的方法
- Policy Gradient
- REINFORCE
- Actor-Critic
- A3C / A2C

### 3. 模型无关的方法
- DDPG
- SAC（Soft Actor-Critic）
- TD3

## 📖 学习资源

### 书籍
- 《强化学习》（Reinforcement Learning: An Introduction）- Sutton & Barto
- 《Deep Reinforcement Learning Hands-On》

### 课程
- Stanford CS234: Reinforcement Learning
- David Silver's RL Course

### 环境
- OpenAI Gym
- Gymnasium
- MuJoCo
- Atari Games

## 💡 实践项目

### 初级
- [ ] CartPole平衡
- [ ] MountainCar爬坡
- [ ] FrozenLake迷宫

### 中级
- [ ] Atari游戏（Pong、Breakout）
- [ ] 连续控制（BipedalWalker）
- [ ] 自动驾驶（简单场景）

### 高级
- [ ] AlphaGo类型游戏
- [ ] 机器人控制
- [ ] 多智能体协作

## 🔗 与其他机器学习的区别

| 特性 | 监督学习 | 无监督学习 | 强化学习 |
|------|---------|-----------|---------|
| 标签 | 有 | 无 | 延迟奖励 |
| 反馈 | 即时 | 无 | 延迟 |
| 数据 | 静态 | 静态 | 动态交互 |
| 目标 | 预测 | 结构发现 | 最大化累积奖励 |

## 💻 技术栈

```python
import gymnasium as gym
import numpy as np
from collections import deque
import random
```

## 📝 学习路径

1. 马尔可夫决策过程（MDP）
2. Bellman方程
3. Q-Learning
4. 深度Q网络（DQN）
5. 策略梯度
6. Actor-Critic
7. 现代算法（SAC、PPO）
