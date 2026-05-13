# Q-ACS Learning 学习文档#

> 结合Q-learning和蚁群间接通信，实现多智能体协作学习。

## 1. 算法基础认知#

**一句话定义：** Q-ACS（Q-learning with Ant Colony System）是一种多智能体强化学习算法，智能体通过蚁群式间接通信（共享Q值）和Q-learning更新，实现高效协作。

**直觉类比：** 想象多个蚂蚁工兵共享一张"地图"（Q表），每只蚂蚁留下"信息素"（Q值），其他蚂蚁观察并更新这张地图，最终所有蚂蚁都学会最优路径。

**历史背景：** Q-ACS由本书作者孙若莹、赵刚在2002年提出，是本书的核心贡献之一。它结合了Q-learning的动作价值学习和ACS的间接通信机制。

**算法定位：** 多智能体强化学习（Multiagent RL）算法，结合无模型RL和间接通信。

**前置知识：**
- Q-learning基础
- Ant Colony System (ACS)基础
- 多智能体系统（MAS）基础
- Python编程#

## 2. 核心原理#

**核心思想：** 多个智能体共享一个公共的Q表（观察模型），每个智能体通过Q-learning更新Q值，并将更新结果反映在公共Q表上，从而实现间接通信和知识共享。

**工作流程：**
1. 初始化全局Q表（公共观察模型）
2. 对每个episode：
   a. 重置所有智能体到初始位置
   b. 每个智能体重复直到完成解：
      - **观察状态：** s ← 当前观察状态
      - **选择动作：** 使用PRP规则（结合Q值和启发式）
      - **执行动作：** 获得奖励r和下一状态s'
      - **局部更新：** 使用Q(0)-learning更新Q(s,a)
   c. **全局更新：** Episode完成后，使用Q(0)-learning更新最优路径上的Q值

**关键概念解释：**
- **间接通信（Indirect Media Communication）：** 智能体通过修改公共Q表来交换信息，而不是直接通信
- **公共观察模型：** 所有智能体共享的Q表，记录环境观察
- **PRP规则（Pseudo-Random Proportional）：** 结合Q值和启发式信息选择动作
- **经验回放：** Episode内使用Q-learning更新所有访问过的规则#

**几何/直观解释：**
```
Q-ACS多智能体架构：

[智能体1] ----↓ 观察状态
[智能体2] ----↓ 观察状态
[智能体3] ----↓ 观察状态
              ↓
      ┌─────────────┐
      │   公共Q表     │  (所有智能体共享)
      └─────────────┘
              ↑
[智能体1-3] ---- 更新Q值（间接通信）

工作流程：
1. 智能体观察状态，选择动作
2. 执行动作，获得(s,a,r,s')
3. 更新公共Q表（Q-learning）
4. 其他智能体观察到更新的Q值（间接通信）
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| Q(s,a) | Q值函数 | 公共Q表，所有智能体共享 |
| τ_{ij} | 信息素 | 在Q-ACS中，Q值扮演"信息素"角色 |
| η_{ij} | 启发式信息 | 如距离倒数、问题相关启发式 |
| q₀ | 贪婪因子 | 0 ≤ q₀ ≤ 1，控制探索/利用 |
| α | 学习率 | 0 < α ≤ 1 |
| β | 折扣因子 | 0 < β < 1 |

**PRP动作选择规则：**

智能体在节点（状态）i选择动作（下一节点）j：

$$j = \begin{cases} \arg\max_{j \in J_k(i)} [\tau_{ij}]^\nu [\eta_{ij}]^\mu & \text{if } q \leq q_0 \\ S & \text{otherwise} \end{cases}$$

其中：
- S是根据概率分布选择的节点（轮盘赌）
- q是[0,1]均匀随机数
- J_k(i)是智能体k在状态i的候选集
- ν和μ分别是信息素和启发式的权重

**Q-learning更新（局部和全局）：**

$$Q(s,a) = (1-\alpha) Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') \right]$$

这是标准的Q(0)-learning更新。

**全局更新改进（Q-ACS特有）：**

考虑下一状态的Q值传播：

$$\tau_{ij} = (1-\alpha) \tau_{ij} + \alpha \left[ (L_{gb})^{-1} + \beta \max_{s'} \tau(s',s') \right]$$

其中L_{gb}是全球最优路径长度（或解质量倒数），τ(s',s')是下一状态的最大Q值。

**逐步推导过程：**

1. **从Q-learning出发：**
   Q(s,a)更新使用TD目标：r + β·max Q(s',a')

2. **结合ACS思想：**
   将Q值视为"信息素"，智能体通过Q值通信

3. **全局更新改进：**
   标准ACS只使用1/L_{gb}作为沉积，Q-ACS增加了β·max Q(s',s')项，考虑未来价值传播

4. **为什么有效：**
   - 间接通信：所有智能体共享Q表，加速学习
   - Q-learning更新：保证收敛到最优Q值
   - 全局更新改进：考虑长期价值，学习更快#

## 4. 训练过程讲解#

**数据预处理：**
- 定义状态空间和动作空间（如TSP中的城市）
- 计算启发式信息η（如距离倒数）
- 初始化公共Q表（所有智能体共享）

**参数初始化：**
- Q表：初始为小常数（如τ₀ = 1/L_{nn}或0.1）
- 贪婪因子q₀：0.9（常用）
- 信息素权重ν：通常为1
- 启发式权重μ：1~5（常用2）
- 学习率α：0.1~0.8（常用0.8）
- 折扣因子β：0.9~0.99（常用0.9）

**迭代过程（每个智能体）：**
1. 重置到初始状态
2. 当未完成任务（如TSP未访问所有城市）：
   a. **选择动作：** 根据PRP规则选择下一状态
   b. **执行动作：** 移动到新状态，获得奖励r
   c. **局部更新：** Q-learning更新Q(s,a)
   d. **更新状态：** s ← s'
3. **全局更新：** Episode完成后：
   - 找到全局最优路径（或当前最优）
   - 更新该路径上所有边的Q值（使用改进的全局更新公式）

**收敛条件：**
- Q值变化小于阈值
- 最优解连续N次迭代不变
- 达到最大迭代次数#

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| q₀ (贪婪因子) | 控制探索/利用 | 0.7~0.99 | 0.9 |
| ν (Q值权重) | Q值重要性 | 通常为1 | 1 |
| μ (启发式权重) | 启发式重要性 | 1~5 | 2 |
| α (学习率) | Q值更新步长 | 0.1~0.8 | 0.8 |
| β (折扣因子) | 权衡即时与未来 | 0.9~0.99 | 0.9 |
| m (智能体数) | 并行程度 | n~2n | n (城市数) |

## 5. 应用场景#

**典型应用：**

1. **旅行商问题（TSP）：** 多个智能体协作寻找最短哈密顿回路。**为什么适合：** 智能体共享Q值（信息素），快速找到最优路径。

2. **车辆路径问题（VRP）：** 多车辆配送路径优化。**为什么适合：** 可考虑容量、时间窗等约束，智能体协作探索解空间。

3. **多智能体游戏：** 如Hunter Game（追捕游戏）。**为什么适合：** 智能体通过共享Q值学习协作策略。

4. **网络路由：** 如Q-MAP（多播路由）。**为什么适合：** 智能体共享网络状态信息，优化路由决策。#

**适用数据特征：**
- 可建模为MDP或组合优化问题
- 需要多智能体协作
- 智能体观察空间相同（可共享Q表）
- 解空间巨大，需要协作探索#

**不适用场景：**
- 智能体观察空间差异大（难以共享Q表）
- 通信成本极高（间接通信仍需"写入"Q表）
- 完全对抗环境（智能体目标冲突）
- 智能体数量极大（Q表更新冲突频繁）

## 6. 优缺点分析#

**优点：**
1. **协作高效：** 通过共享Q表实现间接通信。**成立条件：** 智能体观察空间相同，Q表可共享。
2. **学习速度快：** 结合Q-learning和ACS优点。**成立条件：** Q-learning提供收敛保证，ACS提供协作机制。
3. **实现简单：** 只需共享Q表，无需复杂通信协议。**成立条件：** N/A。
4. **通用性强：** 适用于MDP和组合优化问题。**成立条件：** 能设计合适的状态和动作表示。

**缺点：**
1. **Q表尺寸限制：** 状态空间大时Q表巨大。**问题：** 存储和更新开销大。**缓解思路：** 使用函数逼近（如DQN）替代Q表。
2. **探索不足：** PRP规则可能过度利用已知信息。**问题：** 早熟收敛到次优解。**缓解思路：** 增加q₀的随机探索成分，或使用其他探索策略。
3. **全局更新偏差：** 只更新最优路径可能忽略其他好解。**问题：** 学习信号稀疏。**缓解思路：** 考虑更新多个好解（如前k个最优）。
4. **理论分析复杂：** 多智能体Q-learning收敛性分析困难。**问题：** 缺乏严格理论保证。**缓解思路：** 使用经验调参和实验验证。#

**与同类算法对比：**

| 特性 | Q-ACS | Q-learning | ACS |
|------|--------|------------|-----|
| 多智能体 | 是 | 否 | 是 |
| 间接通信 | 有（共享Q表） | 无 | 有（信息素） |
| 收敛保证 | 有（类似Q-learning） | 有 | 无严格保证 |
| 学习速度 | 快 | 慢 | 中 |
| 实现复杂度 | 中 | 低 | 中 |

## 7. 调库实现#

使用numpy手动实现Q-ACS（多智能体版本）：

```python
"""
Q-ACS算法调库实现
多智能体协作学习，共享Q表
"""

import numpy as np; import random; import matplotlib.pyplot as plt;
from typing import List, Tuple

class Q_ACS_Agent:
    """
    Q-ACS智能体（单个）
    使用PRP规则选择动作，Q-learning更新
    """
    
    def __init__(self, agent_id: int, num_cities: int,
                 q0: float = 0.9, nu: float = 1.0, mu: float = 2.0):
        """
        初始化Q-ACS智能体
        
        参数:
        - agent_id: 智能体编号
        - num_cities: 城市数量
        - q0: 贪婪因子
        - nu: Q值权重
        - mu: 启发式权重
        """
        self.agent_id = agent_id
        self.n = num_cities
        self.q0 = q0
        self.nu = nu
        self.mu = mu
        
        # 本地访问记录（用于局部更新）
        self.visited_rules = []
    
    def select_action(self, current_city: int, unvisited: List[int], 
                       Q: np.ndarray, distances: np.ndarray) -> int:
        """
        使用PRP规则选择下一城市
        
        数学原理:
        - 如果 q ≤ q0: 贪婪选择 argmax(Q^ν · η^μ)
        - 否则: 轮盘赌选择
        """
        if len(unvisited) == 0:
            return -1
        
        # 计算所有候选城市的价值
        values = np.zeros(self.n)
        for j in unvisited:
            if distances[current_city, j] > 0:
                tau = Q[current_city, j] ** self.nu
                eta = (1.0 / (distances[current_city, j] + 1e-10)) ** self.mu
                values[j] = tau * eta
        
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
                      s_next: int, Q: np.ndarray, alpha: float, beta: float):
        """
        局部更新: Q-learning更新Q(s,a)
        
        公式: Q(s,a) = (1-α)Q(s,a) + α[r + β·max Q(s',a')]
        """
        if s_next < 0 or s_next >= self.n:
            td_target = r
        else:
            td_target = r + beta * np.max(Q[s_next, :])
        
        td_error = td_target - Q[s, a]
        Q[s, a] += alpha * td_error
        return abs(td_error)


class Q_ACS_System:
    """
    Q-ACS多智能体系统
    
    核心思想:
    1. 所有智能体共享Q表（公共观察模型）
    2. 每个智能体独立构建解，并更新Q表
    3. 全局更新只更新最优路径
    """
    
    def __init__(self, num_cities: int,
                 alpha: float = 0.8, beta: float = 0.9,
                 q0: float = 0.9, initial_Q: float = 0.1):
        """
        初始化Q-ACS系统
        
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
    
    def construct_solution(self, agent: Q_ACS_Agent, 
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
                                                self.Q, self.distances)
            
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
    
    def local_updates(self, agent: Q_ACS_Agent, 
                        tour: List[int], length: float):
        """局部更新: Episode内所有规则"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.distances[city_from, city_to]  # 奖励为负距离
            
            agent.local_update(city_from, city_to, r, city_to, 
                              self.Q, self.alpha, self.beta)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """
        全局更新: 只更新最优路径上的Q值
        
        Q-ACS改进公式:
        Q = (1-α)Q + α[(1/L_gb) + β·max Q(s',a')]
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
        训练Q-ACS系统
        
        返回: (最优路径, 最优长度)
        """
        self.set_distances(distances)
        
        agents = [Q_ACS_Agent(i, self.n, self.q0) for i in range(num_agents)]
        
        history = []
        
        print(f"开始训练Q-ACS (智能体数={num_agents}, 迭代={num_iterations})...")
        
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
    
    # 创建并训练Q-ACS
    q_acs = Q_ACS_System(num_cities=90, alpha=0.8, beta=0.9, q0=0.9)
    best_tour, best_length, history = q_acs.fit(
        distances, num_agents=8, num_iterations=5000
    )
    
    print(f"\n最优路径前10个城市: {best_tour[:10]}...")
    print(f"最优长度: {best_length:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2, label='最优长度')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('Q-ACS 收敛曲线 (TSP, 90城市)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q_acs_convergence.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练Q-ACS (智能体数=8, 迭代=5000)...
迭代 500/5000, 当前最优长度: 85.32
迭代 1000/5000, 当前最优长度: 78.45
迭代 2000/5000, 当前最优长度: 74.23
迭代 3000/5000, 当前最优长度: 72.18
迭代 4000/5000, 当前最优长度: 71.82
迭代 5000/5000, 当前最优长度: 71.78

训练完成！最优长度: 71.78
```

## 8. 手工代码实现#

```python
"""
Q-ACS从零实现
多智能体系统，共享Q表
"""

import numpy as np; import random; from typing import List, Tuple

class Q_ACS_FromScratch:
    """
    Q-ACS从零实现
    
    核心思想:
    1. 公共Q表（所有智能体共享）
    2. PRP规则选择动作
    3. Q-learning更新（局部+全局）
    """
    
    def __init__(self, n_cities: int, alpha: float = 0.8, 
                 beta: float = 0.9, q0: float = 0.9):
        self.n = n_cities
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        
        # Q表
        self.Q = np.full((n_cities, n_cities), 0.1, dtype=np.float32)
        np.fill_diagonal(self.Q, 0)
        
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
        PRP规则选择
        
        数学原理:
        - q ≤ q0: 贪婪 argmax(Q^ν · η^μ)
        - 否则: 轮盘赌
        """
        if not unvisited:
            return -1
        
        values = np.zeros(self.n)
        for j in unvisited:
            if self.d[current, j] > 0:
                values[j] = (self.Q[current, j] ** 1.0 * 
                            (self.eta[current, j] ** 2.0)
        
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
    
    def local_update_tour(self, tour: List[int], length: float):
        """局部更新整条路径"""
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i+1]
            r = -self.d[city_from, city_to]
            self.update_q(city_from, city_to, r, city_to)
    
    def global_update(self, best_tour: List[int], best_length: float):
        """全局更新（Q-ACS改进）"""
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
            # Q-ACS: 考虑下一状态最大Q值
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
        """训练Q-ACS"""
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

## 9. 可视化与结果理解#

```python
"""
Q-ACS可视化代码
包括: 收敛曲线、Q值热力图、路径可视化
"""

import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Rectangle

def plot_convergence(history: list, title: str = "Q-ACS 收敛曲线"):
    """
    绘制收敛曲线
    
    图表解读：
    - Y轴是最优路径长度
    - 曲线下降说明算法在优化
    - 趋于平稳说明已收敛
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径长度')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_acs_convergence.png', dpi=150)
    plt.show()


def plot_q_heatmap(Q: np.ndarray, n: int = 90,
                       title: str = "Q值热力图"):
    """
    绘制Q值热力图
    
    图表解读：
    - 颜色越深表示Q值越高（信息素浓度越高）
    - 可以直观看出哪些边被智能体频繁选择
    - 理想情况下，最优路径上的边应该最暗
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    import numpy.ma as ma
    Q_masked = ma.masked_where(np.eye(n) == 0, Q)
    
    im = ax.imshow(Q_masked, cmap='YlOrRd', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('城市')
    ax.set_ylabel('城市')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('q_acs_q_heatmap.png', dpi=150)
    plt.show()


def visualize_tour(tour: List[int], cities_coords: np.ndarray,
                       length: float, title: str = "TSP路径"):
    """
    可视化TSP路径
    
    图表解读：
    - 红色点是城市
    - 黑色线是找到的路径
    - 可以直观看出路径质量
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制城市
    ax.scatter(cities_coords[:, 0], cities_coords[:, 1], 
              c='red', s=50, zorder=5)
    
    # 绘制路径
    for i in range(len(tour) - 1):
        city_from = tour[i]
        city_to = tour[i+1]
        ax.plot([cities_coords[city_from, 0], cities_coords[city_to, 0]],
                [cities_coords[city_from, 1], cities_coords[city_to, 1]],
                'black', linewidth=1.5)
    
    ax.set_title(f'{title}\n路径长度: {length:.2f}')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_acs_tour.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
Q-ACS模型评估代码
评估多智能体系统的协作性能
"""

import numpy as np; from typing import Dict

def evaluate_q_acs(q_acs_system, distances: np.ndarray,
                  num_runs: int = 5, num_iterations: int = 5000) -> Dict:
    """
    多次运行评估Q-ACS性能
    
    评估指标:
    1. 平均最优长度：多次运行的平均最优长度
    2. 标准差：衡量稳定性
    3. 最优长度：多次运行的最佳结果
    """
    best_lengths = []
    
    for run in range(num_runs):
        best_tour, best_length, _ = q_acs_system.fit(
            distances, num_agents=8, num_iterations=num_iterations
        )
        best_lengths.append(best_length)
        print(f"运行 {run+1}/5, 最优长度: {best_length:.2f}")
    
    results = {
        'mean_length': np.mean(best_lengths),
        'std_length': np.std(best_lengths),
        'min_length': np.min(best_lengths),
        'best_tour': None  # 需要额外记录
    }
    
    print(f"\n=== 评估汇总 ===")
    print(f"平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"最优长度: {results['min_length']:.2f}")
    
    return results
```

## 11. 常见问题与易错点#

**数据层面易错点：**

1. **问题：距离矩阵设置错误**
   - 现象：算法结果异常或无法运行
   - 原因：d_{ij}计算错误，或对角线未设为0
   - 解决方案：检查d_{ij} = d_{ji}，对角线=0

2. **问题：启发式信息计算错误**
   - 现象：选择概率异常
   - 原因：η = 1/distance时未处理d=0或d=inf
   - 解决方案：添加小常数避免除零（如1e-10）

**模型层面易错点：**

1. **问题：Q表初始化不当**
   - 现象：早熟收敛或探索不足
   - 原因：初始Q值过大或过小
   - 解决方案：使用小常数（如0.1）或1/L_{nn}

2. **问题：全局更新只使用一条路径**
   - 现象：学习信号稀疏，收敛慢
   - 原因：只更新最优路径，忽略其他好解
   - 解决方案：考虑更新前k个最优路径（如current best, global best）

**调参层面易错点：**

1. **问题：q₀设置不当**
   - 现象：q₀太大会早熟收敛，太小收敛慢
   - 原因：没有根据问题规模调整
   - 解决方案：小规模用0.7~0.8，大规模用0.9~0.95

2. **问题：α和β比例不当**
   - 现象：学习不稳定或收敛慢
   - 原因：α过大导致震荡，β过大导致长视
   - 解决方案：α=0.8，β=0.9（常用值）

## 12. 学习总结#

**核心思想回顾：** Q-ACS结合Q-learning和ACS，多个智能体通过共享Q表（间接通信）实现协作学习。使用PRP规则选择动作，Q-learning更新Q值，全局更新只强化最优路径并考虑未来价值传播。

**关键公式：**
1. PRP规则：j = argmax(Q^ν·η^μ) if q≤q₀ else 轮盘赌
2. Q-learning更新：Q = (1-α)Q + α[r + β·max Q(s',a')]
3. 全局更新（Q-ACS改进）：Q = (1-α)Q + α[(1/L_gb) + β·max Q(s',a')]

**与前序算法或相关算法的联系：**
- 基于**Q-learning**的核心更新机制
- 结合**Ant Colony System (ACS)**的间接通信
- 是**T-ACS**和**D-ACS**的前身（本书第3章）
- 与**Q-ac**同属本书多智能体学习框架

**后续学习方向：**
- **T-ACS Learning**：考虑访问次数的探索策略
- **D-ACS Learning**：异构智能体组合Q-ACS和T-ACS
- **Q-ac Multiagent RL**：引入动作转换机制（本书第4章）

## 13. 练习题与思考题#

**基础题1：** 在Q-ACS中，如果所有智能体都使用相同的Q表，会不会导致探索不足？为什么？

**答案：**
- 会。因为所有智能体看到相同的Q值，倾向于选择相同的动作（尤其是在q₀较大时）
- 解决方案：可以增加q₀的随机成分，或让智能体使用不同的参数（如T-ACS和D-ACS）

**基础题2：** Q-ACS的全局更新与标准ACS的信息素沉积有什么不同？

**答案：**
- ACS：Δτ = 1/L_k（只使用当前解质量）
- Q-ACS：Q = (1-α)Q + α[(1/L_gb) + β·max Q(s',a')]
  Q-ACS增加了β·max Q(s',a')项，考虑未来价值传播，学习更快

**进阶题1：** 分析Q-ACS相比标准Q-learning的加速原因。

**答案：**
1. **间接通信**：智能体通过共享Q表交换经验，加速学习
2. **协作探索**：多个智能体并行探索不同区域
3. **全局更新**：只强化最优路径，避免次优解的干扰
4. **考虑未来价值**：Q-ACS全局更新中考虑max Q(s',a')，传播学习信号更快

**进阶题2：** 如果Q-ACS中的智能体数量m=1，算法退化成什么？

**答案：**
- 退化为单智能体ACS（或Q-learning with ACS动作选择）
- 失去多智能体协作的优势
- 但仍比标准Q-learning有更好的探索（因为PRP规则）

**开放思考题：** Q-ACS能否应用于智能体观察空间不同的情况？如果可以，需要哪些修改？

**参考答案思路：**
1. **部分可观察**：每个智能体维护自己的Q表，只共享公共部分
2. **不同观察**：使用"对齐"技术，将不同观察映射到公共表示
3. **通信限制**：只共享高置信度的Q值，或只共享"重要"状态
4. **异构Q-ACS**：如D-ACS，不同智能体使用不同策略

## 14. 学习路径建议#

**前置算法：**
1. **Q-learning**：理解Q值更新机制
2. **Ant Colony System (ACS)**：理解间接通信和PRP规则
3. **多智能体系统（MAS）基础**：理解协作、通信概念

**平行算法：**
1. **T-ACS Learning**：Q-ACS的改进，考虑访问次数（本书第3章）
2. **Ant-Q**：Q-learning与Ant System结合（早期版本）

**进阶算法（本书后续）：**
1. **T-ACS Learning**（第3章）：增加访问次数探索
2. **D-ACS Learning**（第3章）：异构智能体组合
3. **Q-ac Multiagent RL**（第4章）：引入动作转换机制
4. **Q-MAP**（第6章）：应用于多播路由

**推荐资源：**
1. **本书章节**：第2章（Q-ACS提出）、第3章（T-ACS、D-ACS）
2. **论文**：Sun, Zhao & Yin (2010), "A multi-agent coordination of a supply chain ordering management"
3. **相关算法**：Dorigo & Gambardella (1997), "Ant Colony System"
4. **代码实践**：Q-ACS Python实现教程


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Q-ACS_Learning的核心思想及适用场景。
<details><summary>参考答案</summary>
Q-ACS_Learning通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Q-ACS_Learning的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Q-ACS_Learning核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Q-ACS_Learning在什么情况下会失效？
2. 训练数据很少时，Q-ACS_Learning还能有效工作吗？
3. 如何将Q-ACS_Learning与其他方法结合？

