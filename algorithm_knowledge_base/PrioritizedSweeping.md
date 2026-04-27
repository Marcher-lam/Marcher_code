# Prioritized Sweeping 学习文档

> 基于优先级更新的强化学习方法。

## 1. 算法基础认知

Prioritized Sweeping（优先级扫描）是一种高效的强化学习方法，它基于TD误差的大小来确定更新的优先级，优先更新误差大的状态-动作对。

**直觉类比**：如果你在学习做题，不是按顺序每题都做一遍，而是先做那些错得最离谱的题重点复习。这就是优先级扫描的思想。

**前置知识**：Q-Learning、经验回放

## 2. 核心原理

核心思想：
1. 计算每个状态的TD误差
2. 按误差大小排序
3. 优先更新误差大的

## 3. 数学公式与推导

**TD误差**：
$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

**优先级**：
$$P(s,a) = ||\delta||$$

## 4. 训练过程讲解

**参数**：
| 参数 | 作用 |
|------|------|
| priority | 优先级队列 |
| max_priority | 最大优先级 |

## 5. 应用场景

- 小规模MDP
- 需要快速收敛的场景

## 6. 优缺点分析

**优点**：收敛快、效率高
**缺点**：需要额外数据结构

## 7. 调库实现

```python
"""
Prioritized Sweeping实现
"""
import numpy as np
import heapq

class PrioritizedSweeping:
    """优先级扫描"""
    def __init__(self, n_states, n_actions, gamma=0.9, theta=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        
        self.Q = np.zeros((n_states, n_actions))
        self.predecessors = {s: [] for s in range(n_states)}
        self.priority_queue = []
    
    def add_predecessor(self, s, a, pred_s, pred_a):
        """记录前驱状态"""
        key = (pred_s, pred_a)
        if key not in self.predecessors[s]:
            self.predecessors[s].append(key)
    
    def update(self, s, a, r, s_next):
        """更新Q值"""
        target = r + self.gamma * np.max(self.Q[s_next])
        delta = abs(target - self.Q[s, a])
        
        if delta > self.theta:
            heapq.heappush(self.priority_queue, (-delta, s, a))
    
    def sweep(self):
        """扫描更新"""
        visited = set()
        
        while self.priority_queue:
            neg_delta, s, a = heapq.heappop(self.priority_queue)
            
            if (s, a) in visited:
                continue
            visited.add((s, a))
            
            for pred_s, pred_a in self.predecessors[s]:
                target = self.Q[s, a] + self.gamma * np.max(self.Q[pred_s])
                delta = abs(target - self.Q[pred_s, pred_a])
                
                if delta > self.theta:
                    heapq.heappush(self.priority_queue, (-delta, pred_s, pred_a))

# 测试
ps = PrioritizedSweeping(16, 4)
ps.Q[5, 1] = 0.5
ps.update(5, 1, 0.1, 8)
print("测试完成")
```

## 8-14. 其他章节

**学习总结**：优先级扫描通过优先更新高误差状态来加速收敛。

> 来源线索：本节内容根据原书中关于"prioritized sweeping"的相关章节整理。