# Q-ACS 多智能体学习 学习文档

> 结合Q-Learning和ACO的多智能体学习方法。

## 1. 算法基础认知

Q-ACS是将Q-Learning算法与蚂蚁系统（Ant System）相结合的多智能体学习方法，由赵刚等人在多主体强化学习研究中提出。该方法同时利用强化学习的试错机制和蚂蚁群体的协作通信来解决MDP问题和组合优化问题。

**直觉类比**：想象一群蚂蚁不仅记住每条路的长度（像Q值），还在路上留下更多信息素。Q-ACS结合了这种"记忆"和"通信"两种机制。

**前置知识**：Q-Learning、Ant System

## 2. 核心原理

**核心机制**：
- 信息素同时表示Q值
- 蚂蚁在移动过程中更新信息素
- 通过信息素协作学习

## 3. 数学公式与推导

**信息素更新（结合Q-Learning思想）**：
$$\tau(s,a) \leftarrow (1-\rho)\tau(s,a) + \rho[R + \gamma \max_{a'}\tau(s',a')]$$

**路径选择**：
$$P(s \to s') = \frac{[\tau(s,s')]^\alpha [\eta(s,s')]^\beta}{\sum [\tau(s,s_i)]^\alpha [\eta(s,s_i)]^\beta}$$

其中 $\eta(s,s') = 1/d(s,s')$

## 4. 训练过程讲解

**算法流程**：
```
1. 初始化信息素
2. 对每只蚂蚁：
   a) 从起点出发
   b) 按概率选择下一城市
   c) 更新信息素（局部）
3. 选择最佳蚂蚁，更新信息素（全局）
4. 重复直到收敛
```

**参数**：
| 参数 | 作用 | 典型值 |
|------|------|--------|
| α | 信息素重要性 | 1.0 |
| β | 启发式重要性 | 2.0 |
| ρ | 信息素挥发率 | 0.1 |
| γ | 折扣因子 | 0.95 |

## 5. 应用场景

- 旅行商问题（TSP）
- 车辆路径问题（VRP）
- 网络路由
- Job Shop调度

## 6. 优缺点分析

**优点**：
1. 结合Q-Learning和ACO的优点
2. 可解决MDP问题
3. 分布式协作
4. 全局搜索能力强

**缺点**：
1. 参数敏感
2. 收敛速度一般
3. 需要仔细调参

## 7. 调库实现

```python
"""
Q-ACS算法实现
"""
import numpy as np
from scipy.spatial import distance_matrix

class QACS:
    """Q-ACS算法"""
    def __init__(self, n_ants=20, n_iter=100, alpha=1.0, beta=2.0, 
                 rho=0.1, gamma=0.95, q=100.0):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.gamma = gamma
        self.Q = q
        self.best_route = None
        self.best_length = float('inf')
    
    def fit(self, cities):
        """训练"""
        n = len(cities)
        dist = distance_matrix(cities, cities)
        np.fill_diagonal(dist, float('inf'))
        
        eta = 1.0 / dist
        np.fill_diagonal(eta, 0)
        
        tau = np.ones((n, n))
        
        for _ in range(self.n_iter):
            all_routes = []
            all_lengths = []
            
            for _ in range(self.n_ants):
                route = [0]
                visited = {0}
                
                while len(route) < n:
                    curr = route[-1]
                    probs = []
                    for next_city in range(n):
                        if next_city in visited:
                            probs.append(0)
                        else:
                            tau_val = tau[curr, next_city] ** self.alpha
                            eta_val = eta[curr, next_city] ** self.beta
                            probs.append(tau_val * eta_val)
                    
                    probs = np.array(probs)
                    probs /= probs.sum()
                    next_city = np.random.choice(n, p=probs)
                    route.append(next_city)
                    visited.add(next_city)
                
                length = sum(dist[route[i], route[i+1]] for i in range(n-1))
                length += dist[route[-1], route[0]]
                
                all_routes.append(route)
                all_lengths.append(length)
                
                if length < self.best_length:
                    self.best_length = length
                    self.best_route = route.copy()
            
            tau = (1 - self.rho) * tau
            
            for route, length in zip(all_routes, all_lengths):
                deposit = self.Q / length
                for i in range(n-1):
                    j, k = route[i], route[i+1]
                    tau[j, k] += deposit * self.gamma
                    tau[k, j] += deposit * self.gamma
        
        return self.best_route, self.best_length

# 测试
np.random.seed(42)
cities = np.random.rand(10, 2) * 100

qacs = QACS(n_ants=15, n_iter=50)
route, length = qacs.fit(cities)

print(f"最佳路径: {route}")
print(f"长度: {length:.2f}")
```

## 8. 手工代码实现

```python
"""
Q-ACS手工实现
"""
import numpy as np

def qacs_solve(cities, n_ants=10, n_iter=30):
    """Q-ACS求解"""
    n = len(cities)
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(cities[i] - cities[j])
    np.fill_diagonal(dist, np.inf)
    
    eta = 1.0 / dist
    np.fill_diagonal(eta, 0)
    
    tau = np.ones((n, n))
    best_route, best_length = None, np.inf
    
    for _ in range(n_iter):
        routes, lengths = [], []
        
        for _ in range(n_ants):
            route = [0]
            visited = {0}
            
            while len(route) < n:
                curr = route[-1]
                probs = []
                for j in range(n):
                    if j in visited:
                        probs.append(0)
                    else:
                        probs.append(tau[curr,j] * eta[curr,j])
                probs = np.array(probs)
                probs /= probs.sum()
                next_city = np.random.choice(n, p=probs)
                route.append(next_city)
                visited.add(next_city)
            
            length = sum(dist[route[i], route[i+1]] for i in range(n-1))
            length += dist[route[-1], route[0]]
            routes.append(route)
            lengths.append(length)
            
            if length < best_length:
                best_length = length
                best_route = route.copy()
        
        tau = 0.9 * tau
        for route, length in zip(routes, lengths):
            deposit = 100 / length
            for i in range(n-1):
                tau[route[i], route[i+1]] += deposit
                tau[route[i+1], route[i]] += deposit
    
    return best_route, best_length

cities = np.array([[0,0], [1,3], [4,1], [3,4], [2,2]])
route, length = qacs_solve(cities)
print(f"路径: {route}, 长度: {length:.2f}")
```

## 9-14. 其他章节

**学习总结**：Q-ACS结合了Q-Learning的学习机制和ACO的协作机制，是多智能体学习的重要方法。

**核心创新**：
- 信息素作为Q值
- 强化学习与群体智能结合

> 来源线索：本节内容根据原书中关于"Q-ACS multiagent learning method"的相关章节整理。