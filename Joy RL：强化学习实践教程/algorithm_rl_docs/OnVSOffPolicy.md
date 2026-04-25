# Off-Policy vs On-Policy 学习文档

> 强化学习算法的策略分类

---

## 1. 基本概念

### 1.1 On-Policy（同策略）

- 学习当前正在执行的策略
- 行为策略 = 目标策略
- 例子：Sarsa, A2C, PPO

### 1.2 Off-Policy（异策略）

- 可以从历史数据学习
- 行为策略 ≠ 目标策略
- 例子：Q学习, DQN, DDPG

---

## 2. 对比

| 特性 | On-Policy | Off-Policy |
|------|------------|------------|
| 数据利用 | 低（一次性的） | 高（可回放） |
| 实现 | 简单 | 复杂（需要回放） |
| 样本效率 | 低 | 高 |
| 收敛 | 稳定 | 可能不稳定 |
| 探索 | 显式控制 | 可隐式 |

---

## 3. 代表算法

### On-Policy
- REINFORCE
- Actor-Critic
- A2C/A3C
- PPO
- Sarsa

### Off-Policy
- Q-learning
- DQN/ Rainbow
- DDPG/ TD3
- SAC