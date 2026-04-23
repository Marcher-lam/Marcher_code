# SAC（Soft Actor-Critic）学习文档

## 1. 算法基础认知

SAC 是一种基于最大熵框架的离线策略（off-policy）RL 算法，适用于连续动作空间。在广告系统中用于连续权重空间的动态调权。

## 2. 核心原理

### 最大熵目标

$$
\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t (r_t + \alpha H(\pi(\cdot|s_t))) \right]
$$

其中 α 是温度参数，H 是策略熵。

### 关键特性

- 自动温度调节（automated entropy tuning）
- 双 Q 网络（twin Q networks）减少过估计
- 重参数化技巧（reparameterization trick）

## 3. 在广告中的应用

- 连续动作空间的出价调整
- 多目标权重动态调权
- RL-based 冷启动策略

## 4. 与其他方法对比

| 方法 | 适用场景 | 响应速度 | 复杂度 | 稳定性 |
|------|---------|---------|--------|--------|
| RL (PPO/SAC) | 连续权重空间 | ★慢 | 高 | ★需调优 |

## 5. 学习总结

SAC 在广告系统中用于连续动作空间的决策优化。相比 PPO，SAC 样本效率更高但实现更复杂。
