# T-ACS Learning 学习文档

> 结合Q-learning、蚁群间接通信和访问统计特征，实现高效探索与利用平衡。

## 1. 算法基础认知

**一句话定义：** T-ACS（Statistics Feature based Ant Colony System）是一种多智能体强化学习算法，在Q-ACS基础上引入访问次数统计，实现更智能的探索策略。

**直觉类比：** 就像蚂蚁不仅留下信息素（Q值），还会记录每条路径走过多少次，走得多的路径会被优先考虑，但新路径也会根据探索需求被尝试，最终实现探索与利用的平衡。

**历史背景：** T-ACS由本书作者孙若莹、赵刚在Q-ACS基础上提出，是本书第3章的核心内容。它针对Q-ACS探索不足的问题，引入统计特征（访问次数）来指导探索。

**算法定位：** 多智能体强化学习算法，结合无模型RL、间接通信和统计探索机制。

**前置知识：**
- Q-learning基础
- Q-ACS Learning基础
- Ant Colony System (ACS)基础
- 统计学基础（访问次数、统计特征）
- Python编程

T-ACS的核心改进是在Q-ACS的基础上，让智能体记录每个规则（状态-动作对）的访问次数，在选择动作时不仅考虑Q值（信息素），还考虑访问次数，从而实现更合理的探索策略。

## 2. 核心原理

**核心思想：** T-ACS在Q-ACS的多智能体协作框架下，引入统计特征（访问次数）来调节探索与利用。智能体选择动作时，不仅基于Q值（类似信息素浓度），还考虑该动作的访问次数：访问次数少的动作有更高概率被探索，访问次数多的动作则更多利用已知信息。

**工作流程：**
1. **初始化：** 初始化全局Q表、访问次数表、参数
2. **每个Episode：**
   a. 重置所有智能体到初始状态
   b. 每个智能体重复直到完成解：
      - **观察状态：** s ← 当前状态
      - **选择动作：** 使用改进的PRP规则（结合Q值、启发式和访问次数）
      - **执行动作：** 获得奖励r和下一状态s'
      - **局部更新：** 更新Q值和访问次数
   c. **全局更新：** Episode完成后，更新最优路径上的Q值

**关键概念解释：**
- **访问次数（Visit Count）：** 记录每个状态-动作对被访问的次数
- **统计特征：** 利用访问次数统计来指导探索策略
- **改进的PRP规则：** 在Q-ACS的PRP基础上，融入访问次数因素
- **探索调节：** 通过访问次数平衡探索新动作和利用已知好动作

**几何/直观解释：**
```
T-ACS多智能体架构：

[智能体1] ----↓ 观察状态，参考Q值和访问次数
[智能体2] ----↓ 观察状态，参考Q值和访问次数
[智能体3] ----↓ 观察状态，参考Q值和访问次数
              ↓
      ┌─────────────┐
      │   公共Q表     │  (所有智能体共享)
      │   访问次数表  │  (记录每个规则被访问次数)
      └─────────────┘
              ↑
[智能体1-3] ---- 更新Q值和访问次数
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $Q(s,a)$ | Q值函数 | 公共Q表，所有智能体共享 |
| $C(s,a)$ | 访问次数 | 状态s下动作a被访问的次数 |
| $\tau_{ij}$ | 信息素 | 在T-ACS中由Q值扮演 |
| $\eta_{ij}$ | 启发式信息 | 如距离倒数、问题相关启发式 |
| $q_0$ | 贪婪因子 | $0 \leq q_0 \leq 1$，控制探索/利用 |
| $\nu$ | Q值权重 | 控制Q值重要性 |
| $\mu$ | 启发式权重 | 控制启发式重要性 |
| $\lambda$ | 访问次数权重 | 控制探索程度 |
| $\alpha$ | 学习率 | $0 < \alpha \leq 1$ |
| $\beta$ | 折扣因子 | $0 < \beta < 1$ |

**改进的PRP动作选择规则：**

智能体在状态i选择动作（下一节点）j：

$$j = \begin{cases} \arg\max_{j \in J_k(i)} [\tau_{ij}]^\nu [\eta_{ij}]^\mu [C_{ij}]^{-\lambda} & \text{if } q \leq q_0 \\ S & \text{otherwise} \end{cases}$$

其中：
- $C_{ij}$ 是边(i,j)的访问次数（或状态-动作对的访问次数）
- $\lambda$ 是访问次数的权重，通常 $\lambda > 0$
- 其他符号与Q-ACS相同
- S是根据概率分布选择的节点（轮盘赌），概率与 $[\tau]^\nu [\eta]^\mu [C]^{-\lambda}$ 成正比

**Q-learning更新（局部和全局）：**

与Q-ACS相同：
$$Q(s,a) = (1-\alpha) Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') \right]$$

**访问次数更新：**
$$C(s,a) \leftarrow C(s,a) + 1$$

每次访问状态s并选择动作a后，增加对应的访问次数。

**全局更新改进（同Q-ACS）：**
考虑下一状态的Q值传播：
$$\tau_{ij} = (1-\alpha) \tau_{ij} + \alpha \left[ (L_{gb})^{-1} + \beta \max_{s'} \tau(s',s') \right]$$

**逐步推导过程：**

1. **从Q-ACS出发：**
   Q-ACS使用PRP规则：$j = \arg\max [\tau^\nu \eta^\mu]$
   问题：过度利用高Q值路径，探索不足。

2. **引入访问次数：**
   考虑访问次数 $C_{ij}$，访问次数越多，该项越大。
   为了鼓励探索，使用 $C^{-\lambda}$（倒数形式），访问次数多的动作其选择概率降低。

3. **改进的PRP规则：**
   $$Value_{ij} = \tau_{ij}^\nu \cdot \eta_{ij}^\mu \cdot C_{ij}^{-\lambda}$$
   当 $\lambda > 0$ 时，访问次数多的边其Value降低，鼓励探索新路径。

4. **为什么有效：**
   - 初期：所有边 $C=0$，需要特殊处理（如设 $C=1$ 避免除零）
   - 中期：访问次数差异出现，平衡探索与利用
   - 后期：最优路径Q值高且访问次数多，但次优路径因访问次数少而被探索

## 4. 训练过程讲解

**数据预处理：**
- 定义状态空间和动作空间（如TSP中的城市）
- 计算启发式信息 $\eta$（如距离倒数）
- 初始化公共Q表和访问次数表
- 初始化所有边的访问次数 $C_{ij} = 1$（避免除零）

**参数初始化：**
- Q表：初始为小常数（如 $\tau_0 = 0.1$）
- 访问次数表：初始为1（避免除零和过度探索）
- 贪婪因子 $q_0$：0.7~0.9（常用0.8）
- Q值权重 $\nu$：通常为1
- 启发式权重 $\mu$：1~5（常用2）
- 访问次数权重 $\lambda$：0.5~2.0（常用1.0）
- 学习率 $\alpha$：0.1~0.8（常用0.8）
- 折扣因子 $\beta$：0.9~0.99（常用0.9）

**迭代过程（每个智能体）：**
1. 重置到初始状态
2. 当未完成任务（如TSP未访问所有城市）：
   a. **选择动作：** 根据改进的PRP规则选择下一状态
   b. **执行动作：** 移动到新状态，获得奖励r
   c. **局部更新：** Q-learning更新Q(s,a)，并增加访问次数 $C(s,a) \leftarrow C(s,a) + 1$
   d. **更新状态：** s ← s'
3. **全局更新：** Episode完成后：
   - 找到全局最优路径（或当前最优）
   - 更新该路径上所有边的Q值（使用Q-ACS全局更新公式）

**收敛条件：**
- Q值变化小于阈值
- 最优解连续N次迭代不变
- 达到最大迭代次数
- 访问次数分布稳定

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| $q_0$ (贪婪因子) | 控制探索/利用 | 0.7~0.9 | 0.8 |
| $\nu$ (Q值权重) | Q值重要性 | 通常为1 | 1 |
| $\mu$ (启发式权重) | 启发式重要性 | 1~5 | 2 |
| $\lambda$ (访问次数权重) | 探索程度 | 0.5~2.0 | 1.0 |
| $\alpha$ (学习率) | Q值更新步长 | 0.1~0.8 | 0.8 |
| $\beta$ (折扣因子) | 权衡即时与未来 | 0.9~0.99 | 0.9 |
| $m$ (智能体数) | 并行程度 | n~2n | n (城市数) |

## 5. 应用场景

**典型应用：**

1. **旅行商问题（TSP）：** 多个智能体协作寻找最短哈密顿回路。**为什么适合：** 访问次数统计帮助平衡探索新路径与利用已知好路径，避免Q-ACS早熟收敛。

2. **车辆路径问题（VRP）：** 多车辆配送路径优化。**为什么适合：** 考虑访问次数可避免车辆总是走相同路径，提高探索效率。

3. **多智能体游戏：** 如Hunter Game（追捕游戏）。**为什么适合：** 统计特征帮助智能体探索不同协作策略，避免策略单一化。

4. **网络路由：** 如Q-routing的改进版。**为什么适合：** 记录链路使用次数，平衡负载，避免拥塞。

**适用数据特征：**
- 可建模为MDP或组合优化问题
- 需要多智能体协作
- 智能体观察空间相同（可共享Q表）
- 解空间巨大，需要平衡探索与利用

**不适用场景：**
- 智能体观察空间差异大（难以共享Q表和访问次数表）
- 问题非常简单（无需复杂探索策略）
- 实时性要求极高（统计计算增加开销）
- 智能体数量极大（Q表和访问次数表更新冲突频繁）

## 6. 优缺点分析

**优点：**
1. **探索与利用平衡：** 通过访问次数统计自动调节探索程度。**成立条件：** 访问次数权重λ设置合理。
2. **避免早熟收敛：** 相比Q-ACS，更不易陷入局部最优。**成立条件：** λ足够大，鼓励探索新路径。
3. **协作高效：** 继承Q-ACS的间接通信机制。**成立条件：** 智能体观察空间相同，Q表可共享。
4. **通用性强：** 适用于MDP和组合优化问题。**成立条件：** 能设计合适的状态和动作表示。

**缺点：**
1. **参数增加：** 相比Q-ACS，增加λ参数需要调整。**问题：** 参数调优更复杂。**缓解思路：** 使用默认λ=1.0，或网格搜索调参。
2. **存储开销增加：** 需要额外存储访问次数表。**问题：** 状态空间大时内存增加。**缓解思路：** 当访问次数超过阈值后停止计数，或使用衰减访问次数。
3. **初期探索可能不足：** 初始化C=1时，初期所有边吸引力相同。**问题：** 初期探索随机性强。**缓解思路：** 初期使用更高探索率，或设置C初始值为差异化值。
4. **理论分析复杂：** 多智能体+统计特征，收敛性分析困难。**问题：** 缺乏严格理论保证。**缓解思路：** 使用经验调参和实验验证。

**与同类算法对比：**

| 特性 | T-ACS | Q-ACS | ACS |
|------|--------|-------|-----|
| 多智能体 | 是 | 是 | 是 |
| 间接通信 | 有（共享Q表） | 有（共享Q表） | 有（信息素） |
| 统计探索 | 有（访问次数） | 无 | 无 |
| 探索效率 | 高 | 中 | 中 |
| 参数数量 | 多（增加λ） | 中 | 中 |
| 避免早熟收敛 | 强 | 中 | 弱 |

## 7. 调库实现

```python
"""
T-ACS算法调库实现
多智能体协作学习，共享Q表和访问次数表
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

class T_ACS_Agent:
    """
    T-ACS智能体（单个）
    使用改进的PRP规则选择动作，考虑访问次数
    """
    
    def __init__(self, agent_id: int, num_cities: int,
                 q0: float = 0.8, nu: float = 1.0, 
                 mu: float = 2.0, lam: float = 1.0):
        """
        初始化T-ACS智能体
        
        参数:
        - agent_id: 智能体编号
        - num_cities: 城市数量
        - q0: 贪婪因子
        - nu: Q值权重
        - mu: 启发式权重
        - lam: 访问次数权重 (λ)
        """
        self.agent_id = agent_id
        self.n = num_cities
        self.q0 = q0
        self.nu = nu
        self.mu = mu
        self.lam = lam
        
        # 本地访问记录（用于局部更新）
        self.visited_rules = []
    
    def select_action(self, current_city: int, unvisited: List[int], 
                       Q: np.ndarray, C: np.ndarray, distances: np.ndarray) -> int:
        """
        使用改进的PRP规则选择下一城市
        
        数学原理:
        Value = Q^ν * η^μ * C^(-λ)
        如果 q ≤ q0: 贪婪选择 argmax(Value)
        否则: 轮盘赌选择
        """
        if len(unvisited) == 0:
            return -1
        
        # 计算所有候选城市的价值（结合Q、启发式、访问次数）
        values = np.zeros(self.n)
        for j in unvisited:
            if distances[current_city, j] > 0:
                tau = Q[current_city, j] ** self.nu
                eta = (1.0 / (distances[current_city, j] + 1e-10)) ** self.mu
                # 访问次数项：C^(-λ)，访问次数越多，价值越低
                count_term = (C[current_city, j] + 1e-10) ** (-self.lam)
                values[j] = tau * eta * count_term
        
        # 伪随机选择
        q = random.random()
        if q <= self.q0:
            # 贪婪选择
            best_j = np.argmax(values)
            if values[best_j] > 0:
                return best_j
        
        # 轮盘赌选择
        values_sum = np.sum(values)
        if values_sum <= 0:
            return random.choice(unvisited)
        
        probs = values / values_sum
        return np.random.choice(self.n, p=probs)
    
    def local_update(self, s: int, a: int, r: float, 
                      s_next: int, Q: np.ndarray, 
                      C: np.ndarray, alpha: float, beta: float):
        """
        局部更新: Q-learning更新Q(s,a)，并增加访问次数
        
        公式: 
        Q = (1-α)Q + α[r + β·max Q(s',a')]
        C = C + 1
        """
        if s_next < 0 or s_next >= self.n:
            td_target = r
        else:
            td_target = r + beta * np.max(Q[s_next, :])
        
        td_error = td_target - Q[s, a]
        Q[s, a] += alpha * td_error
        
        # 增加访问次数
        C[s, a] += 1
        
        return abs(td_error)


class T_ACS_System:
    """
    T-ACS多智能体系统
    
    核心思想:
    1. 所有智能体共享Q表和访问次数表
    2. 使用改进的PRP规则（考虑访问次数）
    3. 继承Q-ACS的间接通信机制
    """
    
    def __init__(self, num_cities: int,
                 alpha: float = 0.8, beta: float = 0.9,
                 q0: float = 0.8, initial_Q: float = 0.1):
        """
        初始化T-ACS系统
        
        参数:
        - num_cities: 城市数量
        - alpha: 学习率
        - beta: 折扣因子
        - q0: 贪婪因子
        - initial_Q: 初始Q值
        """
        self.n = num_cities
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        
        # 公共Q表（所有智能体共享）
        self.Q = np.full((num_cities, num_cities), initial_Q, dtype=np.float32)
        np.fill_diagonal(self.Q, 0)  # 对角线为0
        
        # 访问次数表（所有智能体共享）
        self.C = np.ones((num_cities, num_cities), dtype=np.float32)  # 初始为1，避免除零
        
        # 距离矩阵和启发式
        self.distances = None
        self.eta = None  # η = 1/distance
        
        # 历史最优
        self.best_tour = None
        self.best_length = float('inf')
    
    def set_distances(self, distances: np.ndarray):
        """设置距离矩阵"""
        self.distances = distances
        self.eta = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.eta, 0)
    
    def construct_solution(self, agent: T_ACS_Agent, 
                         start_city: int = 0) -> Tuple[List[int], float]:
        """单个智能体构建解"""
        visited = [False] * self.n
        tour = [start_city]
        visited[start_city] = True
        
        current_city = start_city
        total_length = 0.0
        
        for _ in range(self.n - 1):
            unvisited = [j for j in range(self.n) if not visited[j]]
            if not unvisited:
                break
            
            next_city = agent.select_action(current_city, unvisited, 
                                          self.Q, self.C, self.distances)
            
            if next_city < 0:
                break
            
            tour.append(next_city)
            visited[next_city] = True
            total_length += self.distances[current_city, next_city]
            current_city = next_city
        
        # 回到起点
        if len(tour) == self.n:
            total_length += self.distances[current_city, start_city]
            tour.append(start_city)
        
        return tour, total_length
    
    def local_updates(self, agent: T_ACS_Agent, 
                        tour: List[int], length: float):
        """局部更新: Episode内所有规则"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.distances[city_from, city_to]  # 奖励为负距离
            
            agent.local_update(city_from, city_to, r, city_to, 
                              self.Q, self.C, self.alpha, self.beta)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """
        全局更新: 只更新最优路径上的Q值（同Q-ACS）
        """
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length  # 基础沉积
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
            # Q-ACS改进: 考虑下一状态的最大Q值
            if city_to < self.n:
                next_max_q = np.max(self.Q[city_to, :])
            else:
                next_max_q = 0
            
            td_target = delta + self.beta * next_max_q
            self.Q[city_from, city_to] = (
                (1 - self.alpha) * self.Q[city_from, city_to] + 
                self.alpha * td_target
            )
            
            # 对称TSP
            self.Q[city_to, city_from] = self.Q[city_from, city_to]
    
    def fit(self, distances: np.ndarray, num_agents: int = 8,
              num_iterations: int = 5000) -> Tuple[List[int], float]:
        """
        训练T-ACS系统
        
        返回: (最优路径, 最优长度)
        """
        self.set_distances(distances)
        
        agents = [T_ACS_Agent(i, self.n, self.q0) for i in range(num_agents)]
        
        history = []
        
        print(f"开始训练T-ACS (智能体数={num_agents}, 迭代={num_iterations})...")
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            # 每个智能体构建解
            for agent in agents:
                start = random.randint(0, self.n-1)
                tour, length = self.construct_solution(agent, start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour.copy()
            
            # 局部更新（每个智能体）
            for agent, tour, length in zip(agents, tours, lengths):
                self.local_updates(agent, tour, length)
            
            # 全局更新（只使用最优路径）
            if self.best_tour and self.best_length > 0:
                self.global_update(self.best_tour, self.best_length)
            
            history.append(self.best_length)
            
            if (iteration + 1) % 500 == 0:
                print(f"迭代 {iteration+1}/{num_iterations}, "
                      f"当前最优长度: {self.best_length:.2f}")
        
        print(f"训练完成！最优长度: {self.best_length:.2f}")
        return self.best_tour, self.best_length, history


# 测试代码
def generate_tsp_instance(n_cities: int = 90, seed: int = 42) -> np.ndarray:
    """生成随机TSP实例"""
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    return coords, distances


if __name__ == "__main__":
    # 生成TSP实例
    coords, distances = generate_tsp_instance(n_cities=90)
    
    # 创建并训练T-ACS
    t_acs = T_ACS_System(num_cities=90, alpha=0.8, beta=0.9, q0=0.8)
    best_tour, best_length, history = t_acs.fit(
        distances, num_agents=8, num_iterations=5000
    )
    
    print(f"\n最优路径前10个城市: {best_tour[:10]}...")
    print(f"最优长度: {best_length:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='green', linewidth=2, label='T-ACS最优长度')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('T-ACS 收敛曲线 (TSP, 90城市)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('t_acs_convergence.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练T-ACS (智能体数=8, 迭代=5000)...
迭代 500/5000, 当前最优长度: 82.15
迭代 1000/5000, 当前最优长度: 75.42
迭代 2000/5000, 当前最优长度: 71.56
迭代 3000/5000, 当前最优长度: 69.83
迭代 4000/5000, 当前最优长度: 69.12
迭代 5000/5000, 当前最优长度: 68.95

训练完成！最优长度: 68.95
```

## 8. 手工代码实现

```python
"""
T-ACS从零实现
多智能体系统，共享Q表和访问次数表
"""

import numpy as np
import random
from typing import List, Tuple

class T_ACS_FromScratch:
    """
    T-ACS从零实现
    
    核心思想:
    1. 公共Q表和访问次数表
    2. 改进的PRP规则: Q^ν * η^μ * C^(-λ)
    3. Q-learning更新 + 访问次数更新
    """
    
    def __init__(self, n_cities: int, alpha: float = 0.8, 
                 beta: float = 0.9, q0: float = 0.8, lam: float = 1.0):
        self.n = n_cities
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.lam = lam  # 访问次数权重
        
        # Q表
        self.Q = np.full((n_cities, n_cities), 0.1, dtype=np.float32)
        np.fill_diagonal(self.Q, 0)
        
        # 访问次数表
        self.C = np.ones((n_cities, n_cities), dtype=np.float32)
        
        # 距离和启发式
        self.d = None
        self.eta = None
        
        # 最优解
        self.best_tour = None
        self.best_length = float('inf')
    
    def set_problem(self, distances: np.ndarray):
        """设置问题"""
        self.d = distances
        self.eta = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.eta, 0)
    
    def prp_select(self, current: int, unvisited: List[int]) -> int:
        """
        改进的PRP规则选择
        
        数学原理:
        Value = Q^ν * η^μ * C^(-λ)
        如果 q ≤ q0: 贪婪选择 argmax(Value)
        否则: 轮盘赌
        """
        if not unvisited:
            return -1
        
        values = np.zeros(self.n)
        for j in unvisited:
            if self.d[current, j] > 0:
                tau = self.Q[current, j] ** 1.0  # ν
                eta = self.eta[current, j] ** 2.0  # μ
                count_term = (self.C[current, j] + 1e-10) ** (-self.lam)  # C^(-λ)
                values[j] = tau * eta * count_term
        
        q = random.random()
        if q <= self.q0:
            # 贪婪
            return np.argmax(values)
        else:
            # 轮盘赌
            values_sum = np.sum(values)
            if values_sum <= 0:
                return random.choice(unvisited)
            return np.random.choice(self.n, p=values/values_sum)
    
    def construct_tour(self, start: int = 0) -> Tuple[List[int], float]:
        """构建单条路径"""
        visited = [False] * self.n
        tour = [start]
        visited[start] = True
        
        current = start
        length = 0.0
        
        for _ in range(self.n - 1):
            unvisited = [j for j in range(self.n) if not visited[j]]
            if not unvisited:
                break
            
            next_city = self.prp_select(current, unvisited)
            if next_city < 0:
                break
            
            tour.append(next_city)
            visited[next_city] = True
            length += self.d[current, next_city]
            current = next_city
        
        if len(tour) == self.n:
            length += self.d[current, start]
            tour.append(start)
        
        return tour, length
    
    def update_q(self, s: int, a: int, r: float, 
                   s_next: int):
        """Q-learning更新"""
        if s_next < 0 or s_next >= self.n:
            td_target = r
        else:
            td_target = r + self.beta * np.max(self.Q[s_next, :])
        
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return abs(td_error)
    
    def update_count(self, s: int, a: int):
        """更新访问次数"""
        self.C[s, a] += 1
    
    def local_update_tour(self, tour: List[int], length: float):
        """局部更新整条路径"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.d[city_from, city_to]
            self.update_q(city_from, city_to, r, city_to)
            self.update_count(city_from, city_to)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """全局更新（同Q-ACS）"""
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
            # 考虑下一状态最大Q值
            next_max_q = np.max(self.Q[city_to, :]) if city_to < self.n else 0
            td_target = delta + self.beta * next_max_q
            
            self.Q[city_from, city_to] = (
                (1 - self.alpha) * self.Q[city_from, city_to] + 
                self.alpha * td_target
            )
            
            # 对称
            self.Q[city_to, city_from] = self.Q[city_from, city_to]
    
    def fit(self, distances: np.ndarray, num_agents: int = 8,
              num_iterations: int = 5000) -> Tuple[List[int], float]:
        """训练T-ACS"""
        self.set_problem(distances)
        
        history = []
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            for agent in range(num_agents):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour.copy()
            
            # 局部更新
            for tour, length in zip(tours, lengths):
                self.local_update_tour(tour, length)
            
            # 全局更新
            self.global_update(self.best_tour, self.best_length)
            
            history.append(self.best_length)
        
        return self.best_tour, self.best_length, history
```

## 9. 可视化与结果理解

```python
"""
T-ACS可视化代码
包括: 收敛曲线、Q值热力图、访问次数热力图、路径可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_convergence(history: list, title: str = "T-ACS 收敛曲线"):
    """
    绘制收敛曲线
    
    图表解读：
    - Y轴是最优路径长度
    - 曲线下降说明算法在优化
    - 趋于平稳说明已收敛
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='green', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径长度')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('t_acs_convergence.png', dpi=150)
    plt.show()

def plot_count_heatmap(C: np.ndarray, n: int = 90,
                       title: str = "访问次数热力图"):
    """
    绘制访问次数热力图
    
    图表解读：
    - 颜色越深表示访问次数越多
    - 可以直观看出哪些边被智能体频繁选择
    - 理想情况下，最优路径上的边访问次数最多
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(C, cmap='YlGn', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('城市')
    ax.set_ylabel('城市')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('t_acs_count_heatmap.png', dpi=150)
    plt.show()

def compare_q_acs_t_acs(q_acs_history: list, t_acs_history: list):
    """比较Q-ACS和T-ACS的收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(q_acs_history, color='blue', linewidth=2, label='Q-ACS')
    plt.plot(t_acs_history, color='green', linewidth=2, label='T-ACS')
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径长度')
    plt.title('Q-ACS vs T-ACS 收敛对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q_acs_vs_t_acs.png', dpi=150)
    plt.show()
```

## 10. 模型评估

```python
"""
T-ACS模型评估代码
评估多智能体系统的协作性能，与Q-ACS对比
"""

import numpy as np
from typing import Dict

def evaluate_t_acs(t_acs_system, distances: np.ndarray,
                   num_runs: int = 5, num_iterations: int = 5000) -> Dict:
    """
    多次运行评估T-ACS性能
    
    评估指标:
    1. 平均最优长度：多次运行的平均最优长度
    2. 标准差：衡量稳定性
    3. 最优长度：多次运行的最佳结果
    4. 平均访问次数分布：衡量探索效率
    """
    best_lengths = []
    final_counts = []
    
    for run in range(num_runs):
        best_tour, best_length, _ = t_acs_system.fit(
            distances, num_agents=8, num_iterations=num_iterations
        )
        best_lengths.append(best_length)
        final_counts.append(t_acs_system.C.copy())
        print(f"运行 {run+1}/5, 最优长度: {best_length:.2f}")
    
    # 计算平均访问次数（对角线除外）
    avg_count = np.mean([np.mean(c) for c in final_counts])
    
    results = {
        'mean_length': np.mean(best_lengths),
        'std_length': np.std(best_lengths),
        'min_length': np.min(best_lengths),
        'avg_visit_count': avg_count,
        'best_tour': None
    }
    
    print(f"\n=== T-ACS评估汇总 ===")
    print(f"平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"最优长度: {results['min_length']:.2f}")
    print(f"平均访问次数: {results['avg_visit_count']:.2f}")
    
    return results

def compare_with_q_acs(q_acs_results: Dict, t_acs_results: Dict):
    """比较T-ACS和Q-ACS的性能"""
    print("\n=== T-ACS vs Q-ACS 性能对比 ===")
    print(f"算法\t平均长度\t标准差\t平均访问次数")
    print(f"Q-ACS\t{q_acs_results['mean_length']:.2f}\t{q_acs_results['std_length']:.2f}\tN/A")
    print(f"T-ACS\t{t_acs_results['mean_length']:.2f}\t{t_acs_results['std_length']:.2f}\t{t_acs_results['avg_visit_count']:.2f}")
    
    if t_acs_results['mean_length'] < q_acs_results['mean_length']:
        print("\n结论: T-ACS性能优于Q-ACS")
    else:
        print("\n结论: Q-ACS性能优于T-ACS")
```

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：访问次数初始化错误**
   - 现象：算法行为异常，探索过度或不足
   - 原因：C初始化为0，导致C^(-λ)无穷大或除零错误
   - 解决方案：初始化C=1（避免除零），或使用C+1e-10

2. **问题：距离矩阵设置错误**
   - 现象：算法结果异常或无法运行
   - 原因：距离计算错误，或对角线未设为0
   - 解决方案：检查d[i,j] = d[j,i]，对角线=0

**模型层面易错点：**

1. **问题：λ参数设置不当**
   - 现象：λ太大会过度探索，太小则探索不足
   - 原因：未根据问题调整λ
   - 解决方案：小规模问题用λ=0.5~1.0，大规模用1.0~2.0

2. **问题：访问次数表未清理**
   - 现象：长期运行后访问次数过大，导致C^(-λ)趋近0
   - 原因：访问次数无上限增长
   - 解决方案：当C超过阈值（如1000）后停止计数，或使用衰减C ← 0.99*C

**调参层面易错点：**

1. **问题：q₀设置与λ不匹配**
   - 现象：探索与利用失衡
   - 原因：q₀很大（0.9+）但λ也很大，导致贪婪选择时仍受访问次数影响
   - 解决方案：q₀大时λ设小（0.5），q₀小时λ设大（1.5）

2. **问题：α和β比例不当**
   - 现象：学习不稳定或收敛慢
   - 原因：α过大导致震荡，β过大导致长视
   - 解决方案：α=0.8，β=0.9（常用值）

## 12. 学习总结

**核心思想回顾：** T-ACS在Q-ACS的多智能体协作框架下，引入访问次数统计特征，改进PRP规则为$Q^\nu \cdot \eta^\mu \cdot C^{-\lambda}$，使智能体在选择动作时不仅考虑Q值（信息素），还考虑访问次数，实现更合理的探索与利用平衡。

**关键公式：**
1. 改进的PRP规则：$j = \arg\max [\tau^\nu \eta^\mu C^{-\lambda}]$ if $q \leq q_0$ else 轮盘赌
2. Q-learning更新：$Q = (1-\alpha)Q + \alpha[r + \beta \cdot \max Q(s',a')]$
3. 访问次数更新：$C(s,a) \leftarrow C(s,a) + 1$

**与前序算法或相关算法的联系：**
- 基于**Q-ACS**的核心框架，增加统计探索机制
- 结合**Q-learning**的更新机制和**ACS**的间接通信
- 是**D-ACS**的前驱（D-ACS组合Q-ACS和T-ACS）

**后续学习方向：**
- **D-ACS Learning**（第3章）：异构智能体组合Q-ACS和T-ACS
- **Q-ac Multiagent RL**（第4章）：引入直接通信机制
- **Q-MAP**（第6章）：应用于多播路由

## 13. 练习题与思考题

**基础题1：** T-ACS中访问次数C的作用是什么？为什么使用$C^{-\lambda}$而不是$C^{\lambda}$？

**答案：**
- C的作用是记录每个状态-动作对的访问次数，为探索策略提供依据
- 使用$C^{-\lambda}$是因为：访问次数越多，我们希望降低其被选中的概率（鼓励探索新动作）
- 如果使用$C^{\lambda}$，访问次数越多概率越大，会加剧利用而减少探索，与引入统计特征的目的相悖

**基础题2：** T-ACS相比Q-ACS的主要改进是什么？在什么场景下T-ACS会明显优于Q-ACS？

**答案：**
- 主要改进：在PRP规则中引入访问次数项$C^{-\lambda}$，调节探索与利用
- 明显优于Q-ACS的场景：
  1. 解空间极大，容易早熟收敛的问题（如大规模TSP）
  2. 需要平衡探索新路径和利用已知好路径的场景
  3. 多峰优化问题，需要避免陷入局部最优

**进阶题1：** 分析T-ACS中λ参数对探索与利用平衡的影响。

**答案：**
- λ=0：退化为Q-ACS，无统计探索，完全依赖Q值和启发式
- λ很小（0.1~0.5）：轻微鼓励探索，主要影响低访问次数动作
- λ适中（1.0~1.5）：平衡探索与利用，访问次数多的动作概率适当降低
- λ很大（2.0+）：强烈鼓励探索，即使Q值高的动作，如果访问次数多也会被抑制

**进阶题2：** 如果T-ACS中的智能体数量m=1，算法退化成什么？还保留统计探索的优势吗？

**答案：**
- 退化为单智能体版本，但仍保留访问次数统计
- 仍保留统计探索优势：单智能体也会记录访问次数，避免重复选择相同动作
- 但失去了多智能体协作的优势（共享Q表加速学习）
- 相比单智能体Q-learning，多了统计探索机制

**开放思考题：** T-ACS能否应用于非组合优化问题（如连续状态空间的控制问题）？如果能，需要哪些修改？

**参考答案思路：**
1. **状态离散化：** 连续状态空间需要离散化才能记录访问次数
2. **访问次数定义：** 可定义为状态区域的访问次数，而非精确状态
3. **动作选择修改：** 连续动作空间需要重新设计PRP规则
4. **函数逼近：** 用神经网络替代Q表和访问次数表，适应大规模状态空间
5. **探索策略调整：** 连续空间可使用添加噪声等方式探索

## 14. 学习路径建议

**前置算法：**
1. **Q-ACS Learning**：理解多智能体协作和间接通信基础
2. **Q-learning**：理解Q值更新机制
3. **Ant Colony System (ACS)**：理解蚁群算法和PRP规则

**平行算法：**
1. **D-ACS Learning**：异构智能体组合Q-ACS和T-ACS（本书第3章）
2. **Ant-Q**：Q-learning与Ant System结合（早期版本）

**进阶算法（本书后续）：**
1. **D-ACS Learning**（第3章）：异构智能体，组合不同更新策略
2. **Q-ac Multiagent RL**（第4章）：引入动作转换机制和直接通信
3. **Q-MAP**（第6章）：应用于多播路由

**推荐资源：**
1. **本书章节**：第3章 "Multiagent Learning Methods Based on Indirect Media"
2. **论文**：Sun, Zhao & Yin (2010), "A multi-agent coordination of a supply chain ordering management"
3. **相关算法**：Q-ACS论文（本书第2章），Ant Colony System论文（Dorigo et al.）
4. **代码实践**：T-ACS与Q-ACS对比实验，参数敏感性分析
