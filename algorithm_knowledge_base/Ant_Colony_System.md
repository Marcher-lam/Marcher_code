# Ant Colony System (ACS) 学习文档#

> 改进的蚁群系统，结合局部更新和全局更新，用于组合优化。

## 1. 算法基础认知#

**一句话定义：** Ant Colony System（ACS）是Ant System的改进版本，增加了局部更新规则和伪随机比例选择规则，用于解决TSP等组合优化问题。

**直觉类比：** 想象蚂蚁不仅留下信息素，还在行走过程中不断挥发路径上的信息素（局部更新），使其他蚂蚁有机会探索新路径。同时，只有最优路径才获得大量信息素沉积（全局更新）。

**历史背景：** ACS由Dorigo和Gambardella在1997年提出，是Ant System的重要改进。它引入了局部更新机制和更精细的选择策略。

**算法定位：** 蚁群优化（ACO）家族的核心算法，结合了正反馈、负反馈（挥发）和启发式信息。

**前置知识：**
- Ant System (AS)基础
- 组合优化基础（TSP、VRP等）
- 概率论基础
- Python编程#

## 2. 核心原理#

**核心思想：** ACS在Ant System基础上增加了局部更新规则（蚂蚁每走一步就更新边上的信息素），并使用伪随机比例规则（exploitation vs exploration）选择下一节点。

**工作流程：**
1. 初始化信息素τ为τ₀
2. 重复直到收敛或达到最大迭代：
   a. **构造解：** 每只蚂蚁独立构造完整解
      - 使用伪随机比例规则选择下一节点
      - 每走一步执行局部更新（挥发）
   b. **评估解：** 计算每只蚂蚁的解质量L_k
   c. **全局更新：** 只有最优蚂蚁（或全局最优）更新信息素

**关键概念解释：**
- **伪随机比例规则（PRP）：** 以概率q₀使用贪婪选择（exploitation），以1-q₀使用轮盘赌（exploration）
- **局部更新：** 蚂蚁每走一步就更新边：τ ← (1-α)·τ + α·τ₀
- **全局更新：** 只有最优解才更新：τ ← (1-α)·τ + α·Δτ（Δτ = 1/L_best）
- **q₀参数：** 控制探索与利用的平衡，通常q₀=0.9#

**几何/直观解释：**
```
ACS流程图：

初始: τ = τ₀

蚂蚁1 (路径: A-B-C-D, 长度=50)
  A→B: 局部更新 τ -= α(τ-τ₀)
  B→C: 局部更新
  C→D: 局部更新

蚂蚁2 (路径: A-C-B-D, 长度=60)

全局更新（只有最优蚂蚁-蚂蚁1）：
  A-B: τ += α/L_best (L_best=50)
  B-C: τ += α/L_best
  C-D: τ += α/L_best
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| τ_{ij} | 信息素 | 边(i,j)上的信息素浓度 |
| η_{ij} | 启发式信息 | 通常为1/d_{ij}（距离倒数）|
| q₀ | 贪婪因子 | 0 ≤ q₀ ≤ 1，通常为0.9 |
| α | 信息素权重 | 局部和全局更新中使用 |
| β | 启发式权重 | 通常为2 |
| L_k | 蚂蚁k的路径长度 | 用于计算Δτ |
| q₀ | 探索/利用平衡 | 随机选择 vs 贪婪选择 |

**伪随机比例规则（PRP）：**

蚂蚁在节点i选择节点j：

$$j = \begin{cases} \arg\max_{l \in J_k(i)} [\tau_{il}]^\alpha [\eta_{il}]^\beta & \text{if } q \leq q_0 \\ J & \text{otherwise} \end{cases}$$

其中q是[0,1]均匀分布的随机数，J是通过轮盘赌选择：

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in J_k(i)} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$

**局部更新规则：**

当蚂蚁走过边(i,j)时（构造解过程中）：

$$\tau_{ij} = (1-\alpha) \tau_{ij} + \alpha \tau_0$$

其中τ₀是初始信息素值。

**全局更新规则：**

对于每个迭代的最优蚂蚁（或全局历史最优）：

$$\tau_{ij} = (1-\alpha) \tau_{ij} + \alpha \Delta \tau_{ij}$$

其中：

$$\Delta \tau_{ij} = \begin{cases} 1 / L_{best} & \text{if } (i,j) \in \text{global-best-tour} \\ 0 & \text{otherwise} \end{cases}$$

L_best可以是：
- 当前迭代最优路径长度（iteration-best）
- 全局历史最优路径长度（global-best）
- 或两者的组合#

**逐步推导过程：**

1. **初始化：** τ_{ij} = τ₀ ∀(i,j)

2. **构造解：** 每只蚂蚁从起点出发
   - 如果q ≤ q₀：贪婪选择 argmax τ^α·η^β
   - 否则：轮盘赌选择（概率正比于τ^α·η^β）
   - 每走一步执行局部更新：τ ← (1-α)τ + ατ₀

3. **全局更新：** 找到最优路径
   - 计算L_best（最短路径长度）
   - 更新最优路径上的边：τ ← (1-α)τ + α/L_best

4. **收敛判断：** 如果最优解稳定或达到最大迭代。

**为什么有效：** 局部更新使已走路径信息素挥发，鼓励探索；全局更新强化最优路径，形成正反馈。#

## 4. 训练过程讲解#

**数据预处理：**
- 构建图：节点集V（城市）和边集E
- 计算距离矩阵d_{ij}和启发式η_{ij} = 1/d_{ij}
- 确定候选节点集（如TSP中的未访问城市）

**参数初始化：**
- 信息素τ₀：1/(n·L_nn) 或常数0.1
- 贪婪因子q₀：0.8~0.9（常用0.9）
- 信息素权重α：通常为1
- 启发式权重β：1~5（常用2）
- 挥发率：1-α，通常α=0.1~0.5

**迭代过程：**
1. 每只蚂蚁构造完整解：
   a. 从起点出发，初始化候选集
   b. 当候选集非空：
      - 根据PRP规则选择下一节点
      - 移动到该节点，执行局部更新
      - 从候选集移除该节点
   c. 返回完成的解和长度L_k

2. 找到当前迭代最优路径

3. 全局更新：
   - 使用当前最优或全局最优L_best
   - 更新最优路径上的边

4. 记录历史最优解

5. 重复直到收敛或达到最大迭代。

**收敛条件：**
- 最优解连续N次迭代不变
- 信息素差异小于阈值
- 达到最大迭代次数。

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| τ₀ (初始信息素) | 初始探索能力 | 小常数或1/(n·L_nn) | 0.1 |
| q₀ (贪婪因子) | 控制探索/利用 | 0.7~0.9 | 0.9 |
| α (信息素权重) | 更新步长 | 0.1~1 | 1 |
| β (启发式权重) | 启发式重要性 | 1~5 | 2 |
| m (蚂蚁数) | 并行搜索程度 | n~2n | n (城市数) |
| L_best | 全局更新依据 | current-best或global-best | global-best |

## 5. 应用场景#

**典型应用：**

1. **旅行商问题（TSP）：** 寻找访问所有城市的最短回路。**为什么适合：** ACS是专为TSP设计的，表现优异。

2. **车辆路径问题（VRP）：** 多车辆配送路径优化。**为什么适合：** 可考虑容量、时间窗等约束，本书第5章有应用。

3. **作业车间调度：** 最小化完成时间的任务调度。**为什么适合：** 可建模为图搜索问题。

4. **网络路由：** 数据包在网络中的路径选择。**为什么适合：** 分布式特性与ACO天然契合，本书第6章有应用。

**适用数据特征：**
- 可建模为图搜索或组合优化问题
- 需要全局优化
- 解空间巨大，无法穷举
- 启发式信息可计算

**不适用场景：**
- 连续优化问题（需离散化）
- 动态变化频繁的问题（信息素过时）
- 实时性要求极高（构建解需要时间）
- 约束极其复杂的问题（难以设计启发式）

## 6. 优缺点分析#

**优点：**
1. **改进的探索能力：** 局部更新使蚂蚁探索新路径。**成立条件：** 局部更新参数α设置合理。
2. **更快收敛：** 伪随机规则加速收敛。**成立条件：** q₀设置合理（不过大或过小）。
3. **解质量高：** 全局更新只强化最优解。**成立条件：** 能找到足够好的当前最优。
4. **通用性强：** 适用于多种组合优化问题。**成立条件：** 能设计合适的启发式表示。

**缺点：**
1. **参数更多：** 相比Ant System有更多参数调优。**问题：** 参数间相互作用复杂。**缓解思路：** 使用经验值或网格搜索。
2. **早熟收敛：** 可能陷入局部最优。**问题：** 信息素过度集中。**缓解思路：** 增加q₀（更多探索）或使用MMAS的边界限制。
3. **对TSP偏向：** 其他问题需要重新设计启发式。**问题：** 通用性受限。**缓解思路：** 针对不同问题设计特定启发式。
4. **计算开销：** 每只蚂蚁构造O(n²)复杂度。**问题：** 大规模问题耗时。**缓解思路：** 使用候选列表（candidate list）加速。

**与同类算法对比：**

| 特性 | Ant System | ACS | MMAS |
|------|-------------|-----|------|
| 局部更新 | 无 | 有 | 有（简化） |
| 选择规则 | 纯概率 | 伪随机（PRP） | 伪随机 |
| 全局更新 | 所有蚂蚁 | 只有最优蚂蚁 | 最优蚂蚁 |
| 信息素边界 | 无 | 无 | 有 |
| 收敛速度 | 慢 | 快 | 快且稳定 |

## 7. 调库实现#

使用numpy手动实现ACS（因为scikit-learn没有ACO实现）：

```python
"""
Ant Colony System (ACS) 算法调库实现
用于解决TSP（旅行商问题）
"""

import numpy as np; import random; import matplotlib.pyplot as plt; from typing import List, Tuple

class AntColonySystem:
    """
    Ant Colony System (ACS)
    改进的蚁群系统，包含局部更新和全局更新
    """
    
    def __init__(self, num_cities: int,
                 alpha: float = 1.0, beta: float = 2.0,
                 q0: float = 0.9, evaporation_rate: float = 0.1,
                 initial_pheromone: float = 0.1):
        """
        初始化ACS
        
        参数:
        - num_cities: 城市数量n
        - alpha: 信息素权重（τ^α）
        - beta: 启发式权重（η^β）
        - q0: 贪婪因子（概率q ≤ q0时贪婪选择）
        - evaporation_rate: 挥发率（1-α中的α）
        - initial_pheromone: 初始信息素τ₀
        """
        self.n = num_cities
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.rho = evaporation_rate  # 挥发率 = 1 - alpha_in_formula
        
        # 信息素矩阵 τ[i,j]
        self.pheromones = np.full((num_cities, num_cities), initial_pheromone)
        np.fill_diagonal(self.pheromones, 0)  # 对角线为0
        
        # 距离矩阵和启发式信息
        self.distances = None
        self.heuristics = None  # η = 1/distance
    
    def set_distances(self, distances: np.ndarray):
        """
        设置距离矩阵并计算启发式信息
        
        启发式: η_{ij} = 1 / d_{ij}
        """
        self.distances = distances
        # 避免除零
        self.heuristics = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.heuristics, 0)
    
    def select_next_city(self, current: int, unvisited: List[int]) -> int:
        """
        使用伪随机比例规则（PRP）选择下一城市
        
        数学原理:
        - 如果 q ≤ q0: 贪婪选择 argmax(τ^α · η^β)
        - 否则: 轮盘赌选择（概率 ∝ τ^α · η^β）
        """
        if len(unvisited) == 0:
            return -1
        
        # 计算所有候选城市的概率/价值
        probabilities = np.zeros(self.n)
        greedy_value = float('-inf')
        greedy_city = -1
        
        for j in unvisited:
            if self.distances[current, j] <= 0:
                continue
            value = (self.pheromones[current, j] ** self.alpha * 
                    (self.heuristics[current, j] ** self.beta))
            
            if value > greedy_value:
                greedy_value = value
                greedy_city = j
            
            probabilities[j] = value
        
        # 伪随机选择
        q = random.random()
        if q <= self.q0 and greedy_city >= 0:
            # 贪婪选择
            return greedy_city
        else:
            # 轮盘赌选择
            prob_sum = np.sum(probabilities[unvisited])
            if prob_sum <= 0:
                return random.choice(unvisited)
            
            probs = probabilities.copy()
            probs[:] = 0
            for j in unvisited:
                probs[j] = probabilities[j] / prob_sum
            
            return np.random.choice(self.n, p=probs)
    
    def local_update(self, city_from: int, city_to: int):
        """
        局部更新：蚂蚁走过边后更新信息素
        
        公式: τ = (1-ρ)·τ + ρ·τ₀
        简化: τ = (1-ρ)·τ + ρ·τ₀ (τ₀通常是initial_pheromone)
        """
        tau0 = np.mean(self.pheromones)  # 简化: 使用平均值作为τ₀
        self.pheromones[city_from, city_to] = (
            (1 - self.rho) * self.pheromones[city_from, city_to] + 
            self.rho * tau0
        )
        # 对称TSP
        self.pheromones[city_to, city_from] = self.pheromones[city_from, city_to]
    
    def construct_solution(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        单只蚂蚁构建解（TSP）
        
        返回: (路径, 路径长度)
        """
        visited = [False] * self.n
        tour = [start_city]
        visited[start_city] = True
        
        current_city = start_city
        total_length = 0.0
        
        for _ in range(self.n - 1):
            # 找到未访问城市
            unvisited = [j for j in range(self.n) if not visited[j] and self.distances[current_city, j] > 0]
            
            if not unvisited:
                break
            
            # 选择下一城市
            next_city = self.select_next_city(current_city, unvisited)
            
            if next_city < 0:
                break
            
            # 更新路径
            tour.append(next_city)
            visited[next_city] = True
            total_length += self.distances[current_city, next_city]
            
            # 局部更新
            self.local_update(current_city, next_city)
            
            current_city = next_city
        
        # 回到起点
        if len(tour) == self.n:
            total_length += self.distances[current_city, start_city]
            tour.append(start_city)
        
        return tour, total_length
    
    def global_update(self, best_tour: List[int], best_length: float):
        """
        全局更新：只有最优路径更新信息素
        
        公式: τ = (1-α)·τ + α·Δτ
        Δτ = 1 / L_best (如果边在最优路径上)
        """
        # 挥发（全局）
        self.pheromones *= (1 - self.rho)
        
        # 沉积（只有最优路径）
        if best_length <= 0:
            return
        
        delta_tau = 1.0 / best_length
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            
            self.pheromones[city_from, city_to] += delta_tau
            self.pheromones[city_to, city_from] += delta_tau  # 对称TSP
        
        # 确保不出现0
        self.pheromones = np.maximum(self.pheromones, 1e-10)
    
    def fit(self, distances: np.ndarray, num_ants: int = 20,
              num_iterations: int = 100) -> Tuple[List[int], float]:
        """
        训练ACS
        
        返回: (最优路径, 最优长度)
        """
        self.set_distances(distances)
        
        best_tour = None
        best_length = float('inf')
        history = []
        
        print(f"开始训练ACS (蚂蚁数={num_ants}, 迭代={num_iterations})...")
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            # 每只蚂蚁构建解
            for ant in range(num_ants):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_solution(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()
            
            # 全局更新（使用当前最优或全局最优）
            best_idx = np.argmin(lengths)
            self.global_update(tours[best_idx], lengths[best_idx])
            
            history.append(best_length)
            
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration+1}/{num_iterations}, "
                      f"当前最优长度: {best_length:.2f}")
        
        print(f"训练完成！最优长度: {best_length:.2f}")
        return best_tour, best_length, history


# 测试代码
def generate_tsp_instance(n_cities: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成随机TSP实例"""
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    
    # 计算距离矩阵
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    return coords, distances


if __name__ == "__main__":
    # 生成TSP实例
    coords, distances = generate_tsp_instance(n_cities=20)
    
    # 创建并训练ACS
    acs_solver = AntColonySystem(num_cities=20, alpha=1.0, beta=2.0, q0=0.9)
    best_tour, best_length, history = acs_solver.fit(
        distances, num_ants=20, num_iterations=100
    )
    
    print(f"\n最优路径: {best_tour}")
    print(f"最优长度: {best_length:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2, label='最优长度')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('ACS 收敛曲线 (TSP, 20城市)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('acs_convergence.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练ACS (蚂蚁数=20, 迭代=100)...
迭代 20/100, 当前最优长度: 432.56
迭代 40/100, 当前最优长度: 398.12
迭代 60/100, 当前最优长度: 385.43
迭代 80/100, 当前最优长度: 381.25
迭代 100/100, 当前最优长度: 380.18

训练完成！最优长度: 380.18
```

## 8. 手工代码实现#

```python
"""
ACS从零实现
实现核心蚁群系统算法
"""

import numpy as np; import random; from typing import List, Tuple

class ACSFromScratch:
    """
    ACS算法从零实现
    
    核心思想:
    1. 伪随机比例规则（PRP）
    2. 局部更新（每步挥发）
    3. 全局更新（只有最优蚂蚁）
    """
    
    def __init__(self, n_cities: int, alpha: float = 1.0, beta: float = 2.0,
                 q0: float = 0.9, rho: float = 0.1):
        self.n = n_cities
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.rho = rho
        
        self.tau = np.full((n_cities, n_cities), 0.1, dtype=np.float32)
        np.fill_diagonal(self.tau, 0)
        
        self.d = None
        self.eta = None
    
    def set_problem(self, distances: np.ndarray):
        """设置问题"""
        self.d = distances
        self.eta = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.eta, 0)
    
    def select_next(self, current: int, visited: List[bool]) -> int:
        """伪随机比例规则"""
        unvisited = [j for j in range(self.n) if not visited[j] and self.d[current, j] > 0]
        
        if not unvisited:
            return -1
        
        # 计算价值
        values = np.zeros(self.n)
        for j in unvisited:
            values[j] = (self.tau[current, j] ** self.alpha * 
                        (self.eta[current, j] ** self.beta))
        
        # 伪随机选择
        q = random.random()
        if q <= self.q0:
            # 贪婪
            return np.argmax(values)
        else:
            # 轮盘赌
            values_sum = np.sum(values)
            if values_sum <= 0:
                return random.choice(unvisited)
            probs = values / values_sum
            return np.random.choice(self.n, p=probs)
    
    def local_update(self, i: int, j: int):
        """局部更新"""
        tau0 = 0.1  # 简化
        self.tau[i, j] = (1 - self.rho) * self.tau[i, j] + self.rho * tau0
        self.tau[j, i] = self.tau[i, j]  # 对称
    
    def construct_tour(self, start: int = 0) -> Tuple[List[int], float]:
        """构建单条路径"""
        visited = [False] * self.n
        tour = [start]
        visited[start] = True
        
        current = start
        length = 0.0
        
        for _ in range(self.n - 1):
            next_city = self.select_next(current, visited)
            if next_city < 0:
                break
            
            tour.append(next_city)
            visited[next_city] = True
            length += self.d[current, next_city]
            
            self.local_update(current, next_city)
            
            current = next_city
        
        # 回到起点
        if len(tour) == self.n:
            length += self.d[current, start]
            tour.append(start)
        
        return tour, length
    
    def global_update(self, best_tour: List[int], best_length: float):
        """全局更新"""
        # 挥发
        self.tau *= (1 - self.rho)
        
        # 沉积
        if best_length <= 0:
            return
        
        delta = 1.0 / best_length
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            self.tau[city_from, city_to] += delta
            self.tau[city_to, city_from] += delta
        
        self.tau = np.maximum(self.tau, 1e-10)
    
    def fit(self, distances: np.ndarray, num_ants: int = 20,
              num_iterations: int = 100) -> Tuple[List[int], float]:
        """训练ACS"""
        self.set_problem(distances)
        
        best_tour = None
        best_length = float('inf')
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            for ant in range(num_ants):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()
            
            # 全局更新（使用当前最优）
            best_idx = np.argmin(lengths)
            self.global_update(tours[best_idx], lengths[best_idx])
        
        return best_tour, best_length
```

## 9. 可视化与结果理解#

```python
"""
ACS可视化代码
包括: 收敛曲线、信息素热力图、路径可视化
"""

import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Rectangle

def plot_convergence(history: list, method: str = "ACS"):
    """
    绘制收敛曲线
    
    图表解读：
    - Y轴是最优路径长度
    - 曲线下降说明算法在优化
    - 趋于平稳说明已收敛
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history, color='blue', linewidth=2, label='最优长度')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('路径长度')
    ax.set_title(f'{method} 收敛曲线')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('acs_convergence.png', dpi=150)
    plt.show()


def visualize_tour(tour: List[int], cities_coords: np.ndarray,
                       length: float, title: str = "TSP路径"):
    """
    可视化TSP路径
    
    图表解读：
    - 红色点是城市
    - 黑色线是蚂蚁找到的路径
    - 可以直观看出路径质量
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制城市
    ax.scatter(cities_coords[:, 0], cities_coords[:, 1], 
              c='red', s=50, zorder=5)
    
    # 标注城市编号
    for i in range(len(cities_coords)):
        ax.text(cities_coords[i, 0], cities_coords[i, 1], str(i),
                ha='center', va='center', fontsize=8)
    
    # 绘制路径
    for i in range(len(tour) - 1):
        city_from = tour[i]
        city_to = tour[i+1]
        ax.plot([cities_coords[city_from, 0], cities_coords[city_to, 0]],
                [cities_coords[city_from, 1], cities_coords[city_to, 1]],
                'black', linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'{title}\n路径长度: {length:.2f}')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acs_tour.png', dpi=150)
    plt.show()


def plot_pheromone_heatmap(tau: np.ndarray, n: int = 20,
                          title: str = "信息素热力图"):
    """
    绘制信息素热力图
    
    图表解读：
    - 颜色越深表示信息素浓度越高
    - 可以直观看出哪些边被蚂蚁频繁选择
    - 理想情况下，最优路径上的边应该最暗
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 只显示上三角（对称矩阵）
    import numpy.ma as ma
    tau_masked = ma.masked_where(np.tril(np.ones((n, n)), tau) == 0, tau)
    
    im = ax.imshow(tau_masked, cmap='YlOrRd', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('城市')
    ax.set_ylabel('城市')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('acs_pheromones.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
ACS模型评估代码
评估算法的优化性能
"""

import numpy as np; from typing import Dict

def evaluate_acs(acs_solver, distances: np.ndarray,
                num_runs: int = 5, num_iterations: int = 100) -> Dict:
    """
    多次运行评估ACS性能
    
    评估指标:
    1. 平均最优长度：多次运行的平均最优长度
    2. 标准差：衡量稳定性
    3. 最优长度：多次运行的最佳结果
    """
    best_lengths = []
    
    for run in range(num_runs):
        best_tour, best_length = acs_solver.fit(
            distances, num_ants=20, num_iterations=num_iterations
        )
        best_lengths.append(best_length)
        print(f"运行 {run+1}/{num_runs}, 最优长度: {best_length:.2f}")
    
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
   - 解决方案：检查d_{ij} = d_{ji}，对角线=0。

2. **问题：启发式信息计算错误**
   - 现象：选择概率异常
   - 原因：η = 1/distance时未处理d=0或d=inf
   - 解决方案：添加小常数避免除零（如1e-10）。

**模型层面易错点：**

1. **问题：局部更新实现错误**
   - 现象：探索能力不足或过度挥发
   - 原因：τ₀设置不当或更新公式错误
   - 解决方案：τ₀使用初始信息素值，公式τ ← (1-ρ)τ + ρτ₀。

2. **问题：全局更新只更新当前最优**
   - 现象：收敛慢或陷入局部最优
   - 原因：当前最优可能不是好解
   - 解决方案：考虑使用全局历史最优（global-best）更新。

**调参层面易错点：**

1. **问题：q₀设置不当**
   - 现象：q₀太大会早熟收敛，太小收敛慢
   - 原因：没有根据问题调整
   - 解决方案：TSP用0.9，更复杂问题用0.7~0.8。

2. **问题：α和β比例不当**
   - 现象：过度依赖信息素或启发式
   - 原因：参数设置不合理
   - 解决方案：通常α=1，β=2~5，启发式更重要。

## 12. 学习总结#

**核心思想回顾：** ACS在Ant System基础上增加了局部更新（每步挥发）和伪随机比例规则（PRP），使蚂蚁在构造解过程中就进行探索，只有最优蚂蚁才进行全局更新。

**关键公式：**
1. PRP规则：j = argmax(τ^α·η^β) if q≤q₀ else 轮盘赌
2. 局部更新：τ ← (1-ρ)τ + ρτ₀
3. 全局更新：τ ← (1-α)τ + α/L_best

**与前序算法或相关算法的联系：**
- 是**Ant System (AS)**的改进版本
- 与**MMAS**的区别：MMAS有信息素边界限制
- 是**Q-ACS**等混合算法的基础（本书第2、3章）

**后续学习方向：**
- **MAX-MIN Ant System (MMAS)**：信息素边界，避免早熟收敛
- **Q-ACS Learning**：结合Q-learning和ACS（本书核心贡献）
- **MACS-VRPTW**：多蚁群系统解决带时间窗的车辆路径

## 13. 练习题与思考题#

**基础题1：** 在ACS中，如果q₀=1会发生什么？如果q₀=0会发生什么？

**答案：**
- q₀=1：总是贪婪选择，完全没有探索，快速陷入局部最优
- q₀=0：总是轮盘赌，没有利用，收敛极慢

**基础题2：** 为什么ACS需要局部更新？Ant System为什么不需要？

**答案：**
- ACS的局部更新使已走路径信息素挥发，鼓励其他蚂蚁探索新路径
- Ant System只在构造完成后更新，没有中间的探索机制

**进阶题1：** 分析ACS中α（信息素权重）和β（启发式权重）的作用。

**答案：**
- α控制信息素的重要性：α越大，历史信息影响越大
- β控制启发式重要性：β越大，距离（启发式）影响越大
- 通常β>α，因为启发式提供更直接的指导

**进阶题2：** ACS的局部更新使用τ₀，这个值应该如何设置？

**答案：**
- τ₀可以是初始信息素值
- 或使用平均信息素值
- 目的是提供一个"基础水平"，避免信息素过于集中在少数边

**开放思考题：** ACS中的局部更新和全局更新能否使用不同的α值？为什么？

**参考答案思路：**
- 可以，局部更新通常用更小的α（如0.1），全局更新用更大的α（如1.0）
- 原因：局部更新是探索机制，应该用小的更新步长；全局更新是强化机制，可以用大的更新步长
- 这样可以更好地平衡探索与利用

## 14. 学习路径建议#

**前置算法：**
1. **Ant System (AS)**：ACS的前身，理解基础蚁群行为
2. **组合优化基础**：理解TSP、VRP等问题形式化
3. **概率论基础**：理解轮盘赌等概率选择

**平行算法：**
1. **Genetic Algorithms**：另一种元启发式算法
2. **Particle Swarm Optimization**：群智能的另一代表

**进阶算法（本书后续）：**
1. **MAX-MIN Ant System (MMAS)**（第1、3、5章）：信息素边界
2. **Q-ACS Learning**（第2、3章）：结合Q-learning和ACS
3. **T-ACS Learning**（第3章）：考虑访问次数的探索
4. **D-ACS Learning**（第3章）：异构蚁群学习

**推荐资源：**
1. **教材**：Dorigo & Stützle, "Ant Colony Optimization" (2004)
2. **论文**：Dorigo & Gambardella (1997), "Ant Colony System: A Cooperating Learning Approach"
3. **本书章节**：第1、2、3、5章
4. **代码实践**：ACO算法Python实现教程
