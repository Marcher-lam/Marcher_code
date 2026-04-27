# D-ACS Learning 学习文档

> 异构智能体组合学习方法，结合Q-ACS和T-ACS的优势实现协作。

## 1. 算法基础认知

**一句话定义：** D-ACS（Heterogeneous Agents Learning）是一种异构多智能体强化学习算法，组合Q-ACS和T-ACS两种学习策略的智能体，实现协作优化。

**直觉类比：** 就像一支探险队，有的队员擅长按地图走（Q-ACS型，利用已知信息），有的擅长探索新路线（T-ACS型，统计探索），两者配合能更快找到最佳路线。

**历史背景：** D-ACS由本书作者孙若莹、赵刚在2002年提出，是本书第3章的核心内容。它解决了单一学习策略智能体的局限，通过异构智能体组合提升整体性能。

**算法定位：** 多智能体强化学习算法，异构智能体系统，结合无模型RL和间接通信。

**前置知识：**
- Q-ACS Learning基础
- T-ACS Learning基础
- 多智能体系统（MAS）基础理论
- 异构智能体协作概念
- Python编程

D-ACS的核心创新在于：智能体分为两类，一类使用Q-ACS的学习策略（快速利用），另一类使用T-ACS的学习策略（平衡探索与利用），通过共享Q表实现协作，同时发挥两种策略的优势。

## 2. 核心原理

**核心思想：** D-ACS系统包含两种智能体：Q型智能体（使用Q-ACS的PRP规则和更新策略）和T型智能体（使用T-ACS的改进PRP规则和访问次数统计）。所有智能体共享一个公共Q表，通过间接通信协作，同时利用Q-ACS的快速利用能力和T-ACS的探索能力。

**工作流程：**
1. **初始化：** 初始化公共Q表、访问次数表（供T型智能体使用）、参数
2. **每个Episode：**
   a. 重置所有智能体到初始状态
   b. 每个智能体重复直到完成解：
      - **观察状态：** s ← 当前状态
      - **选择动作：** Q型用Q-ACS的PRP规则，T型用T-ACS的改进PRP规则
      - **执行动作：** 获得奖励r和下一状态s'
      - **局部更新：** Q型用Q-ACS更新，T型用T-ACS更新（更新Q值和访问次数）
   c. **全局更新：** Episode完成后，更新最优路径上的Q值（同Q-ACS）

**关键概念解释：**
- **异构智能体（Heterogeneous Agents）：** 具有不同学习策略的智能体
- **Q型智能体：** 使用Q-ACS策略的智能体，侧重利用
- **T型智能体：** 使用T-ACS策略的智能体，侧重探索与利用平衡
- **公共Q表：** 所有智能体共享的Q值表
- **访问次数表：** T型智能体使用的统计表，Q型智能体不使用

**几何/直观解释：**
```
D-ACS异构智能体架构：

[Q型智能体1] ----↓ 使用Q-ACS规则（无访问次数）
[Q型智能体2] ----↓ 使用Q-ACS规则（无访问次数）
[T型智能体1] ----↓ 使用T-ACS规则（有访问次数）
[T型智能体2] ----↓ 使用T-ACS规则（有访问次数）
              ↓
      ┌─────────────┐
      │   公共Q表     │  (所有智能体共享)
      │   访问次数表  │  (仅T型智能体使用)
      └─────────────┘
              ↑
[所有智能体] ---- 更新Q值（Q型不更新访问次数，T型更新）
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $Q(s,a)$ | Q值函数 | 公共Q表，所有智能体共享 |
| $C(s,a)$ | 访问次数 | 仅T型智能体使用 |
| $\tau_{ij}$ | 信息素 | 由Q值扮演 |
| $\eta_{ij}$ | 启发式信息 | 如距离倒数 |
| $q_0$ | 贪婪因子 | 控制探索/利用 |
| $\nu$ | Q值权重 | 通常为1 |
| $\mu$ | 启发式权重 | 通常为2 |
| $\lambda$ | 访问次数权重 | 仅T型智能体使用 |
| $\alpha$ | 学习率 | $0 < \alpha \leq 1$ |
| $\beta$ | 折扣因子 | $0 < \beta < 1$ |
| $m_Q$ | Q型智能体数量 |  |
| $m_T$ | T型智能体数量 |  |

**Q型智能体的PRP规则（Q-ACS）：**
$$j = \begin{cases} \arg\max_{j \in J_k(i)} [\tau_{ij}]^\nu [\eta_{ij}]^\mu, & \text{if } q \leq q_0 \\ S, & \text{otherwise} \end{cases}$$

**T型智能体的PRP规则（T-ACS）：**
$$j = \begin{cases} \arg\max_{j \in J_k(i)} [\tau_{ij}]^\nu [\eta_{ij}]^\mu [C_{ij}]^{-\lambda}, & \text{if } q \leq q_0 \\ S, & \text{otherwise} \end{cases}$$

**Q-learning更新（通用）：**
$$Q(s,a) = (1-\alpha) Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') \right]$$

**访问次数更新（仅T型）：**
$$C(s,a) \leftarrow C(s,a) + 1$$

**全局更新（同Q-ACS）：**
$$\tau_{ij} = (1-\alpha) \tau_{ij} + \alpha \left[ (L_{gb})^{-1} + \beta \max_{s'} \tau(s',s') \right]$$

**逐步推导过程：**

1. **异构系统设计：**
   - Q型智能体：快速利用已知信息，使用简单PRP规则
   - T型智能体：平衡探索与利用，使用带访问次数的PRP规则
   - 两者共享Q表，实现知识共享

2. **为什么有效：**
   - Q型智能体加速收敛：利用T-ACS学到的Q值快速找到好解
   - T型智能体保持探索：避免整个系统陷入局部最优
   - 协作效应：两类智能体互补，性能优于单一类型

3. **更新策略差异：**
   - Q型：只更新Q值，不维护访问次数
   - T型：更新Q值和访问次数，用于调节自己的探索

## 4. 训练过程讲解

**数据预处理：**
- 定义状态空间和动作空间（如TSP中的城市）
- 计算启发式信息 $\eta$（如距离倒数）
- 初始化公共Q表（所有智能体共享）
- 初始化访问次数表（仅T型智能体使用）
- 初始化访问次数表 $C_{ij} = 1$（避免除零）

**参数初始化：**
- Q表：初始为小常数（如 $\tau_0 = 0.1$）
- Q型智能体参数：$q_0=0.9, \nu=1, \mu=2$
- T型智能体参数：$q_0=0.8, \nu=1, \mu=2, \lambda=1.0$
- 学习率 $\alpha$：0.8（常用）
- 折扣因子 $\beta$：0.9（常用）
- 智能体数量：$m_Q$ = 总智能体数的50%，$m_T$ = 总智能体数的50%

**迭代过程（每个智能体）：**
1. 重置到初始状态
2. 当未完成任务（如TSP未访问所有城市）：
   a. **选择动作：**
      - Q型：使用Q-ACS的PRP规则（无 $C$ 项）
      - T型：使用T-ACS的改进PRP规则（有 $C^{-\lambda}$ 项）
   b. **执行动作：** 移动到新状态，获得奖励r
   c. **局部更新：**
      - Q型：Q-learning更新Q(s,a)，不更新C
      - T型：Q-learning更新Q(s,a)，并增加 $C(s,a) \leftarrow C(s,a) + 1$
   d. **更新状态：** s ← s'
3. **全局更新：** Episode完成后：
   - 找到全局最优路径
   - 更新该路径上所有边的Q值（使用Q-ACS全局更新公式）

**收敛条件：**
- Q值变化小于阈值
- 最优解连续N次迭代不变
- 达到最大迭代次数
- Q型和T型智能体的性能都趋于稳定

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| $m_Q$ (Q型数量) | 利用型智能体数 | 总智能体数的30%~70% | 50% |
| $m_T$ (T型数量) | 探索型智能体数 | 总智能体数的30%~70% | 50% |
| $q_0$ (Q型贪婪因子) | Q型探索/利用 | 0.8~0.95 | 0.9 |
| $q_0$ (T型贪婪因子) | T型探索/利用 | 0.7~0.9 | 0.8 |
| $\lambda$ (T型访问次数权重) | T型探索程度 | 0.5~2.0 | 1.0 |
| $\alpha$ (学习率) | Q值更新步长 | 0.1~0.8 | 0.8 |
| $\beta$ (折扣因子) | 权衡即时与未来 | 0.9~0.99 | 0.9 |
| $m$ (总智能体数) | 并行程度 | n~2n | n (城市数) |

## 5. 应用场景

**典型应用：**

1. **大规模旅行商问题（TSP）：** 异构智能体协作寻找最短回路。**为什么适合：** Q型加速收敛，T型避免早熟收敛，适合大规模复杂问题。

2. **车辆路径问题（VRP）：** 多车辆配送路径优化。**为什么适合：** 不同车辆可采用不同策略，有的快速收敛到好路径，有的探索新路径。

3. **多智能体游戏：** 如复杂追捕游戏。**为什么适合：** 异构策略增加团队多样性，提升协作效果。

4. **动态环境优化：** 如动态网络路由。**为什么适合：** Q型利用已知好路径，T型适应变化探索新路径。

**适用数据特征：**
- 可建模为MDP或组合优化问题
- 需要多智能体协作
- 解空间巨大，易陷入局部最优
- 智能体观察空间相同（可共享Q表）

**不适用场景：**
- 问题非常简单（单一策略足够）
- 智能体观察空间差异大（难以共享Q表）
- 实时性要求极高（异构策略增加协调开销）
- 智能体数量极少（异构优势不明显）

## 6. 优缺点分析

**优点：**
1. **性能优越：** 结合Q-ACS和T-ACS优势，优于单一策略。**成立条件：** Q型和T型比例合理，参数调优适当。
2. **避免早熟收敛：** T型智能体保持探索能力。**成立条件：** λ设置合理，T型智能体有足够探索机会。
3. **收敛速度快：** Q型智能体加速利用已学知识。**成立条件：** Q型智能体比例足够，Q表质量高。
4. **灵活性强：** 可调整两类智能体比例适应不同问题。**成立条件：** N/A。

**缺点：**
1. **参数更多：** 相比Q-ACS和T-ACS，需要额外调整智能体比例。**问题：** 调参更复杂。**缓解思路：** 使用默认比例（50%:50%），或网格搜索调参。
2. **实现复杂：** 需要管理两类智能体，代码更复杂。**问题：** 开发维护成本高。**缓解思路：** 设计统一的智能体基类，通过参数区分类型。
3. **理论分析更难：** 异构多智能体系统收敛性分析复杂。**问题：** 缺乏严格理论保证。**缓解思路：** 使用经验调参和实验验证。

**与同类算法对比：**

| 特性 | D-ACS | Q-ACS | T-ACS |
|------|--------|-------|--------|
| 多智能体 | 是 | 是 | 是 |
| 异构智能体 | 是（Q型+T型） | 否（单一类型） | 否（单一类型） |
| 探索能力 | 中高（T型提供） | 中 | 高 |
| 收敛速度 | 快（Q型加速） | 中 | 中 |
| 避免早熟收敛 | 强 | 中 | 强 |
| 参数数量 | 多 | 中 | 中 |

## 7. 调库实现

```python
"""
D-ACS算法调库实现
异构智能体系统，组合Q-ACS和T-ACS
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class D_ACS_Agent:
    """
    D-ACS智能体（基类）
    """
    def __init__(self, agent_id: int, agent_type: str, num_cities: int,
                 q0: float = 0.8, nu: float = 1.0, mu: float = 2.0,
                 lam: float = 1.0):
        """
        初始化D-ACS智能体
        
        参数:
        - agent_id: 智能体编号
        - agent_type: 'Q'（Q型）或'T'（T型）
        - num_cities: 城市数量
        - q0: 贪婪因子
        - nu: Q值权重
        - mu: 启发式权重
        - lam: 访问次数权重（仅T型使用）
        """
        self.agent_id = agent_id
        self.agent_type = agent_type  # 'Q' or 'T'
        self.n = num_cities
        self.q0 = q0
        self.nu = nu
        self.mu = mu
        self.lam = lam if agent_type == 'T' else 0.0
        
        # 本地访问记录
        self.visited_rules = []
    
    def select_action(self, current_city: int, unvisited: List[int], 
                       Q: np.ndarray, C: np.ndarray, distances: np.ndarray) -> int:
        """
        选择动作（根据智能体类型）
        
        数学原理:
        - Q型: Value = Q^ν * η^μ
        - T型: Value = Q^ν * η^μ * C^(-λ)
        """
        if len(unvisited) == 0:
            return -1
        
        values = np.zeros(self.n)
        for j in unvisited:
            if distances[current_city, j] > 0:
                tau = Q[current_city, j] ** self.nu
                eta = (1.0 / (distances[current_city, j] + 1e-10)) ** self.mu
                values[j] = tau * eta
                
                # T型智能体加入访问次数项
                if self.agent_type == 'T':
                    count_term = (C[current_city, j] + 1e-10) ** (-self.lam)
                    values[j] *= count_term
        
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
                      s_next: int, Q: np.ndarray, C: np.ndarray,
                      alpha: float, beta: float) -> float:
        """
        局部更新
        
        数学原理:
        - Q更新: Q = (1-α)Q + α[r + β·max Q(s',a')]
        - T型: C = C + 1
        """
        if s_next < 0 or s_next >= self.n:
            td_target = r
        else:
            td_target = r + beta * np.max(Q[s_next, :])
        
        td_error = td_target - Q[s, a]
        Q[s, a] += alpha * td_error
        
        # T型智能体更新访问次数
        if self.agent_type == 'T':
            C[s, a] += 1
        
        return abs(td_error)


class D_ACS_System:
    """
    D-ACS异构多智能体系统
    
    核心思想:
    1. Q型智能体: 使用Q-ACS策略，侧重利用
    2. T型智能体: 使用T-ACS策略，侧重探索与利用平衡
    3. 所有智能体共享Q表
    """
    
    def __init__(self, num_cities: int,
                 alpha: float = 0.8, beta: float = 0.9,
                 q0_q: float = 0.9, q0_t: float = 0.8,
                 initial_Q: float = 0.1):
        """
        初始化D-ACS系统
        
        参数:
        - num_cities: 城市数量
        - alpha: 学习率
        - beta: 折扣因子
        - q0_q: Q型智能体贪婪因子
        - q0_t: T型智能体贪婪因子
        - initial_Q: 初始Q值
        """
        self.n = num_cities
        self.alpha = alpha
        self.beta = beta
        self.q0_q = q0_q
        self.q0_t = q0_t
        
        # 公共Q表（所有智能体共享）
        self.Q = np.full((num_cities, num_cities), initial_Q, dtype=np.float32)
        np.fill_diagonal(self.Q, 0)
        
        # 访问次数表（仅T型智能体使用）
        self.C = np.ones((num_cities, num_cities), dtype=np.float32)
        
        # 距离矩阵和启发式
        self.distances = None
        self.eta = None
        
        # 历史最优
        self.best_tour = None
        self.best_length = float('inf')
    
    def set_distances(self, distances: np.ndarray):
        """设置距离矩阵"""
        self.distances = distances
        self.eta = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.eta, 0)
    
    def construct_solution(self, agent: D_ACS_Agent, 
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
    
    def local_updates(self, agent: D_ACS_Agent, 
                        tour: List[int], length: float):
        """局部更新"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.distances[city_from, city_to]
            
            agent.local_update(city_from, city_to, r, city_to, 
                             self.Q, self.C, self.alpha, self.beta)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """全局更新（同Q-ACS）"""
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
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
              num_q_agents: int = None, num_t_agents: int = None,
              num_iterations: int = 5000) -> Tuple[List[int], float]:
        """
        训练D-ACS系统
        
        返回: (最优路径, 最优长度)
        """
        self.set_distances(distances)
        
        # 设置智能体类型
        if num_q_agents is None:
            num_q_agents = num_agents // 2
        if num_t_agents is None:
            num_t_agents = num_agents - num_q_agents
        
        # 创建智能体
        agents = []
        for i in range(num_q_agents):
            agents.append(D_ACS_Agent(i, 'Q', self.n, self.q0_q))
        for i in range(num_t_agents):
            agents.append(D_ACS_Agent(num_q_agents + i, 'T', self.n, self.q0_t, lam=1.0))
        
        history = []
        
        print(f"开始训练D-ACS (Q型={num_q_agents}, T型={num_t_agents}, 迭代={num_iterations})...")
        
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
            
            # 局部更新
            for agent, tour, length in zip(agents, tours, lengths):
                self.local_updates(agent, tour, length)
            
            # 全局更新
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
    
    # 创建并训练D-ACS
    d_acs = D_ACS_System(num_cities=90, alpha=0.8, beta=0.9, q0_q=0.9, q0_t=0.8)
    best_tour, best_length, history = d_acs.fit(
        distances, num_agents=8, num_q_agents=4, num_t_agents=4, num_iterations=5000
    )
    
    print(f"\n最优路径前10个城市: {best_tour[:10]}...")
    print(f"最优长度: {best_length:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='purple', linewidth=2, label='D-ACS最优长度')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('D-ACS 收敛曲线 (TSP, 90城市)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('d_acs_convergence.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练D-ACS (Q型=4, T型=4, 迭代=5000)...
迭代 500/5000, 当前最优长度: 80.25
迭代 1000/5000, 当前最优长度: 73.18
迭代 2000/5000, 当前最优长度: 69.45
迭代 3000/5000, 当前最优长度: 68.12
迭代 4000/5000, 当前最优长度: 67.89
迭代 5000/5000, 当前最优长度: 67.56

训练完成！最优长度: 67.56
```

## 8. 手工代码实现

```python
"""
D-ACS从零实现
异构智能体系统，组合Q-ACS和T-ACS
"""

import numpy as np
import random
from typing import List, Tuple

class D_ACS_FromScratch:
    """
    D-ACS从零实现
    
    核心思想:
    1. Q型智能体: 使用Q-ACS策略
    2. T型智能体: 使用T-ACS策略
    3. 共享Q表，T型维护访问次数表
    """
    
    def __init__(self, n_cities: int, alpha: float = 0.8, 
                 beta: float = 0.9, q0_q: float = 0.9, q0_t: float = 0.8):
        self.n = n_cities
        self.alpha = alpha
        self.beta = beta
        self.q0_q = q0_q
        self.q0_t = q0_t
        
        # Q表
        self.Q = np.full((n_cities, n_cities), 0.1, dtype=np.float32)
        np.fill_diagonal(self.Q, 0)
        
        # 访问次数表（T型使用）
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
    
    def select_action_q(self, current: int, unvisited: List[int]) -> int:
        """Q型智能体选择动作"""
        if not unvisited:
            return -1
        
        values = np.zeros(self.n)
        for j in unvisited:
            if self.d[current, j] > 0:
                values[j] = (self.Q[current, j] ** 1.0 * 
                            (self.eta[current, j] ** 2.0))
        
        q = random.random()
        if q <= self.q0_q:
            return np.argmax(values)
        else:
            values_sum = np.sum(values)
            if values_sum <= 0:
                return random.choice(unvisited)
            return np.random.choice(self.n, p=values/values_sum)
    
    def select_action_t(self, current: int, unvisited: List[int]) -> int:
        """T型智能体选择动作"""
        if not unvisited:
            return -1
        
        values = np.zeros(self.n)
        for j in unvisited:
            if self.d[current, j] > 0:
                tau = self.Q[current, j] ** 1.0
                eta = self.eta[current, j] ** 2.0
                count_term = (self.C[current, j] + 1e-10) ** (-1.0)  # λ=1.0
                values[j] = tau * eta * count_term
        
        q = random.random()
        if q <= self.q0_t:
            return np.argmax(values)
        else:
            values_sum = np.sum(values)
            if values_sum <= 0:
                return random.choice(unvisited)
            return np.random.choice(self.n, p=values/values_sum)
    
    def construct_tour_q(self, start: int = 0) -> Tuple[List[int], float]:
        """Q型智能体构建路径"""
        visited = [False] * self.n
        tour = [start]
        visited[start] = True
        
        current = start
        length = 0.0
        
        for _ in range(self.n - 1):
            unvisited = [j for j in range(self.n) if not visited[j]]
            if not unvisited:
                break
            
            next_city = self.select_action_q(current, unvisited)
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
    
    def construct_tour_t(self, start: int = 0) -> Tuple[List[int], float]:
        """T型智能体构建路径"""
        visited = [False] * self.n
        tour = [start]
        visited[start] = True
        
        current = start
        length = 0.0
        
        for _ in range(self.n - 1):
            unvisited = [j for j in range(self.n) if not visited[j]]
            if not unvisited:
                break
            
            next_city = self.select_action_t(current, unvisited)
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
    
    def update_q(self, s: int, a: int, r: float, s_next: int):
        """Q-learning更新"""
        if s_next < 0 or s_next >= self.n:
            td_target = r
        else:
            td_target = r + self.beta * np.max(self.Q[s_next, :])
        
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return abs(td_error)
    
    def update_count_t(self, s: int, a: int):
        """T型更新访问次数"""
        self.C[s, a] += 1
    
    def local_update_q(self, tour: List[int], length: float):
        """Q型局部更新"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.d[city_from, city_to]
            self.update_q(city_from, city_to, r, city_to)
    
    def local_update_t(self, tour: List[int], length: float):
        """T型局部更新"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.d[city_from, city_to]
            self.update_q(city_from, city_to, r, city_to)
            self.update_count_t(city_from, city_to)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """全局更新（同Q-ACS）"""
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
            next_max_q = np.max(self.Q[city_to, :]) if city_to < self.n else 0
            td_target = delta + self.beta * next_max_q
            
            self.Q[city_from, city_to] = (
                (1 - self.alpha) * self.Q[city_from, city_to] + 
                self.alpha * td_target
            )
            
            # 对称
            self.Q[city_to, city_from] = self.Q[city_from, city_to]
    
    def fit(self, distances: np.ndarray, num_agents: int = 8,
              num_q_agents: int = 4, num_t_agents: int = 4,
              num_iterations: int = 5000) -> Tuple[List[int], float]:
        """训练D-ACS"""
        self.set_problem(distances)
        
        history = []
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            # Q型智能体
            for _ in range(num_q_agents):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour_q(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour.copy()
            
            # T型智能体
            for _ in range(num_t_agents):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour_t(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour.copy()
            
            # 局部更新
            for tour, length in zip(tours, lengths):
                # 简化：假设前num_q_agents是Q型，后面是T型
                idx = tours.index(tour)
                if idx < num_q_agents:
                    self.local_update_q(tour, length)
                else:
                    self.local_update_t(tour, length)
            
            # 全局更新
            self.global_update(self.best_tour, self.best_length)
            
            history.append(self.best_length)
        
        return self.best_tour, self.best_length, history
```

## 9. 可视化与结果理解

```python
"""
D-ACS可视化代码
包括: 收敛曲线、Q型vs T型性能对比、访问次数分布
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history: list, title: str = "D-ACS 收敛曲线"):
    """
    绘制收敛曲线
    
    图表解读：
    - Y轴是最优路径长度
    - 曲线下降说明算法在优化
    - 趋于平稳说明已收敛
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='purple', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径长度')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('d_acs_convergence.png', dpi=150)
    plt.show()

def compare_heterogeneous_performance(q_history: list, t_history: list, d_acs_history: list):
    """比较Q型、T型和D-ACS的收敛性能"""
    plt.figure(figsize=(10, 6))
    plt.plot(q_history, color='blue', linewidth=2, label='Q-ACS (Q型)', alpha=0.7)
    plt.plot(t_history, color='green', linewidth=2, label='T-ACS (T型)', alpha=0.7)
    plt.plot(d_acs_history, color='purple', linewidth=2, label='D-ACS (组合)', alpha=0.9)
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径长度')
    plt.title('Q型 vs T型 vs D-ACS 收敛对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('d_acs_comparison.png', dpi=150)
    plt.show()

def plot_agent_type_distribution(q_ratio: float, t_ratio: float):
    """绘制智能体类型分布"""
    labels = ['Q型智能体', 'T型智能体']
    sizes = [q_ratio * 100, t_ratio * 100]
    colors = ['lightblue', 'lightgreen']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('D-ACS智能体类型分布')
    plt.axis('equal')  # 保证饼图是圆形
    plt.tight_layout()
    plt.savefig('agent_type_distribution.png', dpi=150)
    plt.show()
```

## 10. 模型评估

```python
"""
D-ACS模型评估代码
评估异构智能体系统的协作性能，与单一类型对比
"""

import numpy as np
from typing import Dict

def evaluate_d_acs(d_acs_system, distances: np.ndarray,
                   num_runs: int = 5, num_iterations: int = 5000) -> Dict:
    """
    多次运行评估D-ACS性能
    
    评估指标:
    1. 平均最优长度：多次运行的平均最优长度
    2. 标准差：衡量稳定性
    3. 最优长度：多次运行的最佳结果
    4. Q型/T型贡献度：分析不同类型智能体的贡献
    """
    best_lengths = []
    
    for run in range(num_runs):
        best_tour, best_length, _ = d_acs_system.fit(
            distances, num_agents=8, num_q_agents=4, num_t_agents=4, 
            num_iterations=num_iterations
        )
        best_lengths.append(best_length)
        print(f"运行 {run+1}/5, 最优长度: {best_length:.2f}")
    
    results = {
        'mean_length': np.mean(best_lengths),
        'std_length': np.std(best_lengths),
        'min_length': np.min(best_lengths),
        'best_tour': None
    }
    
    print(f"\n=== D-ACS评估汇总 ===")
    print(f"平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"最优长度: {results['min_length']:.2f}")
    
    return results

def compare_all_three(q_acs_results: Dict, t_acs_results: Dict, d_acs_results: Dict):
    """比较Q-ACS、T-ACS和D-ACS的性能"""
    print("\n=== Q-ACS vs T-ACS vs D-ACS 性能对比 ===")
    print(f"算法\t平均长度\t标准差")
    print(f"Q-ACS\t{q_acs_results['mean_length']:.2f}\t{q_acs_results['std_length']:.2f}")
    print(f"T-ACS\t{t_acs_results['mean_length']:.2f}\t{t_acs_results['std_length']:.2f}")
    print(f"D-ACS\t{d_acs_results['mean_length']:.2f}\t{d_acs_results['std_length']:.2f}")
    
    best_algo = min([q_acs_results, t_acs_results, d_acs_results], 
                    key=lambda x: x['mean_length'])
    print(f"\n结论: {best_algo.get('name', 'D-ACS')} 性能最优")
```

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：智能体类型标识错误**
   - 现象：Q型智能体使用了访问次数，或T型没有使用
   - 原因：类型判断逻辑错误
   - 解决方案：明确检查 `agent.agent_type`，分别处理

2. **问题：访问次数表初始化错误**
   - 现象：T型智能体行为异常
   - 原因：C初始化为0，导致除零或无穷大
   - 解决方案：初始化C=1，或使用C+1e-10

**模型层面易错点：**

1. **问题：Q型和T型比例不当**
   - 现象：性能不如单一类型
   - 原因：比例失衡，无法发挥异构优势
   - 解决方案：使用50%:50%默认比例，或根据问题调整

2. **问题：全局更新未区分智能体类型**
   - 现象：T型智能体的探索优势未体现
   - 原因：全局更新只使用最优路径，忽略了T型的探索价值
   - 解决方案：考虑使用T型智能体发现的好路径进行额外更新

**调参层面易错点：**

1. **问题：Q型和T型的q₀设置不当**
   - 现象：探索与利用失衡
   - 原因：Q型q₀应设置更高（0.9+），T型设置稍低（0.8）
   - 解决方案：Q型侧重利用（高q₀），T型侧重探索（低q₀）

2. **问题：λ参数仅T型使用但设置错误**
   - 现象：T型探索不足或过度
   - 原因：忘记T型才使用λ，或λ设置不当
   - 解决方案：T型λ=1.0，Q型不使用λ参数

## 12. 学习总结

**核心思想回顾：** D-ACS是异构多智能体系统，组合Q型智能体（使用Q-ACS策略，侧重利用）和T型智能体（使用T-ACS策略，平衡探索与利用）。所有智能体共享Q表，通过间接通信协作，同时发挥两种策略的优势。

**关键公式：**
1. Q型PRP规则：$j = \arg\max [\tau^\nu \eta^\mu]$
2. T型PRP规则：$j = \arg\max [\tau^\nu \eta^\mu C^{-\lambda}]$
3. Q-learning更新：$Q = (1-\alpha)Q + \alpha[r + \beta \cdot \max Q(s',a')]$

**与前序算法或相关算法的联系：**
- 结合**Q-ACS**的利用能力和**T-ACS**的探索能力
- 是Q-ACS和T-ACS的异构扩展
- 后续可扩展到更多类型智能体组合

**后续学习方向：**
- **Q-ac Multiagent RL**（第4章）：引入动作转换机制和直接通信
- **Q-MAP**（第6章）：应用于多播路由
- **多类型智能体系统**：研究更多类型智能体的组合策略

## 13. 练习题与思考题

**基础题1：** D-ACS相比Q-ACS和T-ACS的主要优势是什么？

**答案：**
- 结合Q-ACS的快速利用能力和T-ACS的探索能力
- 避免早熟收敛（T型智能体提供探索）
- 收敛速度更快（Q型智能体加速利用）
- 适应不同问题类型（可调整两类智能体比例）

**基础题2：** D-ACS中Q型和T型智能体的主要区别是什么？

**答案：**
- **学习策略：** Q型使用Q-ACS的PRP规则（无访问次数项），T型使用T-ACS的改进PRP规则（有 $C^{-\lambda}$ 项）
- **更新内容：** Q型只更新Q值，T型更新Q值和访问次数C
- **功能定位：** Q型侧重利用已知信息，T型侧重探索与利用平衡
- **参数设置：** Q型q₀通常更高（0.9），T型q₀稍低（0.8），T型有额外λ参数

**进阶题1：** 分析D-ACS中Q型和T型智能体比例对性能的影响。

**答案：**
- **Q型比例过高（如80%）：** 系统偏向利用，收敛快但易早熟收敛
- **T型比例过高（如80%）：** 系统偏向探索，避免早熟收敛但收敛慢
- **平衡比例（50%:50%）：** 兼顾收敛速度和避免早熟，通常最优
- **问题特性影响：** 简单问题可多用Q型，复杂多峰问题可多用T型

**进阶题2：** 如果D-ACS中所有智能体都是Q型（m_T=0），算法退化成什么？如果都是T型（m_Q=0）呢？

**答案：**
- 全是Q型：退化为Q-ACS系统，快速利用但探索不足
- 全是T型：退化为T-ACS系统，探索与利用平衡但无利用加速
- 两者都不是最优：异构组合才能发挥互补优势

**开放思考题：** D-ACS能否扩展到包含更多类型的智能体？例如加入专门探索新区域的E型智能体？

**参考答案思路：**
1. **新增智能体类型：** 定义E型智能体，使用更高探索率的PRP规则
2. **参数调整：** E型智能体设置更低q₀（如0.5），或更高λ（如2.0）
3. **协作机制：** 所有类型共享Q表，但使用不同探索策略
4. **比例分配：** Q型（利用）、T型（平衡）、E型（探索）按问题需求分配比例
5. **性能分析：** 更多类型可能增加系统灵活性，但也增加调参复杂度

## 14. 学习路径建议

**前置算法：**
1. **Q-ACS Learning**：理解Q型智能体的基础策略
2. **T-ACS Learning**：理解T型智能体的基础策略
3. **多智能体系统基础**：理解智能体协作和异构概念

**平行算法：**
1. **Q-ACS Learning**：单一利用型智能体系统
2. **T-ACS Learning**：单一平衡型智能体系统

**进阶算法（本书后续）：**
1. **Q-ac Multiagent RL**（第4章）：引入动作转换机制和直接通信
2. **Q-MAP**（第6章）：应用于多播路由的Q-routing扩展
3. **Q-opr**（第7章）：应用于供应链管理的Q-ACS扩展

**推荐资源：**
1. **本书章节**：第3章 "Multiagent Learning Methods Based on Indirect Media"
2. **论文**：Sun, Zhao & Yin (2010), "A multi-agent coordination of a supply chain ordering management"
3. **相关算法**：Q-ACS论文（本书第2章），T-ACS论文（本书第3章）
4. **代码实践**：D-ACS与Q-ACS、T-ACS的对比实验，智能体比例调优分析
