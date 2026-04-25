# Noisy DQN 学习文档

> 通过参数化噪声增加探索的DQN改进

---

## 1. 算法基础认知

**一句话定义**：Noisy DQN在网络权重中加入可学习的噪声参数，用噪声驱动探索，无需ε-greedy策略。

**直觉类比**：ε-greedy是"偶尔随机行动"，Noisy DQN是"网络参数本身就有随机性"——像喝醉的酒保，有时候手抖把不同酒混一起。

**历史背景**：由Fortunato等人在2017年提出，作为简单有效的探索改进。

---

## 2. 核心原理

### 2.1 噪声层

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(in_dim, out_dim))
        self.sigma = nn.Parameter(torch.full((in_dim, out_dim), sigma))
    
    def forward(self, x):
        # 采样噪声
        epsilon = torch.randn_like(self.weight)
        # 权重 = μ + σ * noise
        weight = self.mu + self.sigma * epsilon
        return F.linear(x, weight)
```

### 2.2 优点

- 自主探索，无需ε参数
- 可梯度优化
- 适应性探索

---

## 3. 总结

✓ 噪声驱动探索
✓ 无需ε-greedy
✓ 可端到端训练