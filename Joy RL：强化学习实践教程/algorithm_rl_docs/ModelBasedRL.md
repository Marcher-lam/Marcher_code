# 模型-based vs 模型-free 强化学习

> 强化学习的两大范式

---

## 1. 模型-based

**了解环境 dynamics**：学习 p(s'|s,a) 和 r(s,a)

### 优点
- 样本效率高
- 可以规划
- 无需真实交互

### 缺点
- 需要学习模型
- 有模型误差

### 代表算法
- World Models
- PlaNet
- Dreamer
- MuZero

---

## 2. 模型-free

**直接学习策略/价值**，不学习环境模型

### 优点
- 实现简单
- 无模型误差

### 缺点
- 样本效率低
- 需要大量交互

### 代表算法
- Q-learning
- PPO
- SAC

---

## 3. 对比

| 特性 | 模型-based | 模型-free |
|------|-----------|----------|
| 样本效率 | 高 | 低 |
| 实现复杂度 | 高 | 低 |
| 规划能力 | 有 | 无 |
| 理论保证 | 弱 | 强 |

---

## 4. 世界模型学习

```python
class WorldModel(nn.Module):
    def __init__(self):
        self.encoder = VAE(...)  # 图像编码
        self.transition = LSTM(...)  # 状态转移
        self.reward = MLP(...)      # 奖励预测
    
    def forward(self, state, action):
        encoded = self.encoder(state)
        next_encoded = self.transition(encoded, action)
        reward = self.reward(encoded, action)
        return next_encoded, reward
```

---

## 总结

- 模型-based: 效率高但复杂
- 模型-free: 简单但样本需求大
- MuZero: 无模型也能规划