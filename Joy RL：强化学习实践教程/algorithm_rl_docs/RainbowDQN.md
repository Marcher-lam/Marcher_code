# Rainbow DQN 学习文档

> DQN的集大成者，融合7种技术

---

## 1. 算法基础认知

**一句话定义**：Rainbow DQN是DQN的"完全体"，融合了Double DQN、Priority Replay、Dueling、Noisy等7种技术。

**历史背景**：由Hessel等人在2017年提出，在Atari游戏上达到SOTA。

---

## 2. 融合的7种技术

1. **Double DQN** - 减少过估计
2. **Prioritized Experience Replay** - 优先级采样
3. **Dueling DQN** - 网络结构
4. **NoisyNet** - 噪声探索
5. **Distributional RL** - 分布Q值
6. **N-step Learning** - 多步TD
7. **Value Scaling** - 值缩放

---

## 3. 核心代码结构

```python
class RainbowDQN:
    """融合7种技术的DQN"""
    
    def __init__(self, state_dim, action_dim):
        # 1. Duelling + 分布Q + Noisy网络
        self.q_net = DistributionalDuelingNoisyNet(state_dim, action_dim)
        
        # 2. Double: 分离选择和评估
        self.target_net = copy.deepcopy(self.q_net)
        
        # 3. 优先级回放
        self.replay = PrioritizedReplayBuffer()
        
        # 4. N-step
        self.n_step_buffer = []
    
    def update(self):
        # 5. 优先级采样
        batch, indices, weights = self.replay.sample()
        
        # 6. Double DQN更新
        ...
```

---

## 4. 在Atari上的表现

| 算法 | 中位数分数 |
|------|----------|
| DQN | 100% |
| DDQN | 128% |
| Dueling | 137% |
| Noisy | 148% |
| PER | 164% |
| Rainbow | **223%** |

---

## 5. 总结

✓ DQN的完全体
✓ 7种技术融合
✓ Atati游戏SOTA