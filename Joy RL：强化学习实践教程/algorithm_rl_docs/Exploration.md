# 探索与利用（Exploration vs Exploitation）

> 强化学习的核心挑战

---

## 1. 基本概念

**探索(Exploration)**：尝试新动作，发现潜在高回报
**利用(Exploitation)**：使用已知最好的动作，最大化当前回报

---

## 2. 探索策略

### 2.1 ε-Greedy
```python
if random.random() < epsilon:
    action = random()  # 探索
else:
    action = argmax(Q)  # 利用
```

### 2.2 Boltzmann
```python
probs = softmax(Q / temperature)
action = sample(probs)
```

### 2.3 UCB
```python
Q_UCB = Q + c * sqrt(ln(N) / N(s,a))
action = argmax(Q_UCB)
```

### 2.4 噪声网络 (Noisy)
在网络参数中加入噪声

---

## 3. 代码实现

```python
class ExplorationScheduler:
    def __init__(self, strategy='epsilon'):
        if strategy == 'epsilon':
            self.epsilon = 1.0
            self.decay = 0.995
        elif strategy == 'boltzmann':
            self.temp = 1.0
        elif strategy == 'ucb':
            self.counts = defaultdict(int)
            self.c = 1.0
    
    def select_action(self, Q, state):
        if self.strategy == 'epsilon':
            if random.random() < self.epsilon:
                return random(len(Q))
            self.epsilon *= self.decay
            return argmax(Q)
        # ... 其他策略
```

---

## 4. 总结

| 方法 | 特点 | 适用 |
|------|------|------|
| ε-greedy | 简单 | 离散动作 |
| Boltzmann | 平滑 | 连续动作 |
| UCB | 理论保证 | 表格 |
| Noisy | 可学习 | 深度RL |