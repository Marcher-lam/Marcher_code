# GAEs（Generalized Advantage Estimation）学习文档

> 平衡偏差与方差的Advantage估计方法

---

## 1. 算法基础认知

**一句话定义**：GAE（Generalized Advantage Estimation）通过调整λ参数，在单步TD的高偏差和MC的低偏差之间取得平衡，是A2C/PPO等算法的核心组件。

**历史背景**：由Schulman等人在2016年提出，作为Actor-Critic的标准Advantage估计方法。

---

## 2. 核心原理

### 2.1 λ参数

- λ=0：单步TD，低方差，高偏差
- λ=1：MC，无偏差，高方差

$$GAE(\lambda): \hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$

### 2.2 公式

```python
def compute_gae(values, rewards, gamma=0.99, lam=0.95):
    """
    计算GAE
    
    Args:
        values: V(s)列表
        rewards: 奖励列表
        gamma: 折扣因子
        lam: GAE参数 (0-1)
    """
    advantages = []
    gae = 0
    
    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # 终止状态
        else:
            next_value = values[t + 1]
        
        # TD误差
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE累积
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages
```

---

## 3. 与其他方法的比较

| 方法 | λ | 偏差 | 方差 |
|------|---|------|------|
| TD(0) | 0 | 高 | 低 |
| TD(λ) | 0-1 | 中 | 中 |
| MC | 1 | 无 | 高 |
| GAE | 自定义 | 可调 | 可调 |

---

## 4. 总结

✓ 灵活的偏差-方差权衡
✓ A2C/PPO的核心组件
✓ 可根据任务调参