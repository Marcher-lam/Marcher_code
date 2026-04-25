# TAR（Target Network with Automatic Replacement）

> 自适应目标网络更新频率

---

## 1. 算法基础认知

**一句话定义**：TAR根据训练动态自动调整目标网络的更新频率，而不是固定每隔N步更新。

---

## 2. 核心代码

```python
class TARSDQN:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.tau = 0.01  # 初始软更新系数
    
    def update_target(self, loss):
        # 根据loss大小自适应调整
        if loss > 1.0:
            self.tau = 0.01  # 高loss，慢更新
        else:
            self.tau = 0.1   # 低loss，快更新
        
        # 软更新
        for target, main in zip(self.target_net.parameters(), self.q_net.parameters()):
            target.data.copy_(self.tau * main.data + (1-self.tau) * target.data)
```

---

## 总结

✓ 自适应更新频率 ✓ 稳定训练