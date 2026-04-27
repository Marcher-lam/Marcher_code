# MAX-MIN Ant System (MMAS) 学习文档#

> 改进的蚁群系统，限制信息素边界避免早熟收敛。

## 1. 算法基础认知#

**一句话定义：** MAX-MIN Ant System（MMAS）是ACS的改进版本，通过限制信息素在[τ_min, τ_max]范围内，避免早熟收敛并保持探索能力。

**直觉类比：** 想象蚂蚁的信息素不能无限积累（上限限制），也不能完全挥发消失（下限限制）。这样即使某条路径暂时领先，其他路径仍有机会被探索。

**历史背景：** MMAS由Stützle和Hoos在2000年提出，是蚁群优化中的重要改进。它在ACS基础上增加了信息素边界机制，解决早熟收敛问题。

**算法定位：** 蚁群优化（ACO）家族的改进算法，属于元启发式（Metaheuristic）优化方法。

**前置知识：**
- Ant System (AS)基础
- Ant Colony System (ACS)基础
- 组合优化基础（TSP、VRP等问题）
- Python编程#

## 2. 核心原理#

**核心思想：** MMAS在ACS基础上增加了信息素边界τ_min和τ_max。所有边的信息素被限制在[τ_min, τ_max]范围内，这样即使某条路径信息素很高，也不会过度集中；即使某条路径信息素很低，也不会完全消失。

**工作流程：**
1. 初始化信息素τ为τ_max（鼓励初期探索）
2. 重复直到收敛或达到最大迭代：
   a. **构造解：** 每只蚂蚁使用PRP规则构造完整解
   b. **计算质量：** 评估每只蚂蚁的解
   c. **更新信息素：**
      - **挥发：** τ ← (1-ρ)·τ
      - **沉积：** 只有最优蚂蚁（当前最优或全局最优）才更新
      - **限制边界：** τ = max(τ_min, min(τ, τ_max))

**关键概念解释：**
- **τ_max：** 信息素上限，防止过度集中
- **τ_min：** 信息素下限，保持基本探索能力
- **最优蚂蚁：** 只有当前最优或全局最优才更新信息素
- **动态边界：** τ_max和τ_min可根据解质量动态调整#

**几何/直观解释：**
```
MMAS信息素边界示意图：

无边界(ACS):  τ可以→∞ (过度集中) 或→0 (探索丧失)

MMAS有边界:
τ_max ─────────────── 上限 (最优路径信息素)
         /
        /
       /
τ_min ─────────────── 下限 (保持基本探索)
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| τ_{ij} | 信息素 | 边(i,j)上的信息素浓度 |
| τ_max | 信息素上限 | 根据最优解质量计算 |
| τ_min | 信息素下限 | 根据τ_max和分支因子计算 |
| ρ | 挥发率 | 0 ≤ ρ < 1 |
| f(s) | 解的质量 | 如路径长度、成本等 |
| L_best | 最优解长度 | 当前最优或全局最优 |

**信息素更新规则：**

只有最优蚂蚁（当前最优或全局最优）才更新：

$$\tau_{ij} = (1-\rho) \tau_{ij} + \rho \Delta \tau_{ij}$$

其中：

$$\Delta \tau_{ij} = \begin{cases} 1 / f(s) & \text{if } (i,j) \in \text{global-best-tour} \\ 0 & \text{otherwise} \end{cases}$$

**信息素边界限制：**

$$\tau_{ij} = \max(\tau_{\min}, \min(\tau_{ij}, \tau_{\max}))$$

**动态边界计算：**

τ_max根据最优解质量：

$$\tau_{\max} = \frac{1}{\rho \cdot f(s_{best})}$$

τ_min根据τ_max和分支因子b（平均候选节点数）：

$$\tau_{\min} = \frac{\tau_{\max} (1 - \sqrt[b]{p_{best})}{\sqrt[b]{p_{best}} - 1}$$

其中p_best是最优解被选中的概率（通常设为0.05）。

**逐步推导过程：**

1. **初始化：** τ_{ij} = τ_max

2. **构造解：** 使用PRP规则（同ACS）

3. **信息素更新：**
   - **挥发：** τ ← (1-ρ)·τ
   - **沉积：** 只有最优蚂蚁才更新Δτ = 1/f(s_best)
   - **限制：** τ = max(τ_min, min(τ, τ_max))

4. **收敛判断：** 如果最优解稳定或达到最大迭代，停止。

**为什么有效：** 边界限制防止信息素过度集中（避免早熟收敛），同时保持探索能力（τ_min确保即使最差边也有机会被选择）。

## 4. 训练过程讲解#

**数据预处理：**
- 构建图：节点集V（城市）和边集E
- 计算启发式信息η（如距离倒数）
- 确定候选节点集（如TSP中的未访问城市）

**参数初始化：**
- 信息素τ_max：根据1/(ρ·L_best)初始化
- 挥发率ρ：0.5~0.95（常用0.75）
- α：1（信息素权重）
- β：2~5（启发式权重）
- 蚂蚁数量m：通常=城市数或更大#

**迭代过程：**
1. 每只蚂蚁构造完整解：
   - 从起点出发
   - 根据PRP规则选择下一节点
   - 使用轮盘赌，确保概率选择
   - 直到所有节点访问完毕（TSP）

2. 评估每只蚂蚁的解质量L_k

3. 信息素更新：
   - **挥发：** 所有边：τ ← (1-ρ)·τ
   - **沉积：** 只有全局最优蚂蚁才更新：Δτ = 1/L_best
   - **限制边界：** 所有边限制在[τ_min, τ_max]

4. 记录历史最优解

5. 动态调整τ_max和τ_min（可选）

6. 重复直到收敛或最大迭代。

**收敛条件：**
- 最优解连续N次迭代不变
- 信息素差异小于阈值
- 达到最大迭代次数#

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| τ_max (信息素上限) | 防止过度集中 | 根据L_best计算 | 1/(ρ·L_best) |
| τ_min (信息素下限) | 保持探索能力 | 根据τ_max计算 | 公式见上文 |
| ρ (挥发率) | 控制信息素挥发 | 0.5~0.95 | 0.75 |
| α (信息素权重) | 信息素重要性 | 0.5~2 | 1 |
| β (启发式权重) | 启发式重要性 | 2~5 | 2 |
| m (蚂蚁数) | 并行搜索程度 | n~2n | n (城市数) |

## 5. 应用场景#

**典型应用：**

1. **旅行商问题（TSP）：** 寻找访问所有城市的最短回路。**为什么适合：** MMAS是TSP的经典算法，表现优异。

2. **车辆路径问题（VRP）：** 多车辆配送路径优化。**为什么适合：** 本书第5章有应用，可考虑容量、时间窗等约束。

3. **作业车间调度：** 最小化完成时间的任务调度。**为什么适合：** 可建模为图搜索问题。

4. **网络路由：** 数据包在网络中的路径选择。**为什么适合：** 分布式特性与ACO天然契合。

**适用数据特征：**
- 可建模为图搜索或组合优化问题
- 需要全局优化
- 解空间巨大，无法穷举
- 启发式信息可计算#

**不适用场景：**
- 连续优化问题（需离散化）
- 动态变化频繁的问题（信息素过时）
- 实时性要求极高（构造解需要时间）
- 约束极其复杂的问题（难以设计启发式）

## 6. 优缺点分析#

**优点：**
1. **避免早熟收敛：** 信息素边界防止过度集中。**成立条件：** τ_max和τ_min设置合理。
2. **保持探索能力：** τ_min确保即使最差边也有机会。**成立条件：** τ_min > 0。
3. **解质量高：** 只有最优蚂蚁更新，强化最优解。**成立条件：** 最优蚂蚁选择合理（当前最优或全局最优）。
4. **理论分析更完善：** 有收敛性分析。**成立条件：** 参数满足边界条件。

**缺点：**
1. **计算参数复杂：** τ_max和τ_min需要计算。**问题：** 公式复杂，需要问题质量。**缓解思路：** 使用简化版本，固定边界。
2. **参数敏感：** 边界参数影响性能。**问题：** 需要仔细调参。**缓解思路：** 使用网格搜索或自适应参数。
3. **只有最优蚂蚁更新：** 可能收敛慢。**问题：** 学习信号稀疏。**缓解思路：** 使用当前最优+全局最优共同更新。
4. **早熟收敛风险未完全消除：** 信息素仍可能过度集中。**问题：** 复杂问题仍需其他机制。**缓解思路：** 结合局部搜索（如2-opt、3-opt）。

**与同类算法对比：**

| 特性 | Ant System | ACS | MMAS |
|------|-------------|-----|------|
| 信息素边界 | 无 | 无 | 有（τ_min, τ_max) |
| 更新蚂蚁 | 所有蚂蚁 | 最优蚂蚁 | 只有最优蚂蚁 |
| 早熟收敛 | 容易 | 中等 | 较难 |
| 收敛速度 | 慢 | 快 | 中快 |
| 解质量 | 中 | 好 | 很好 |

## 7. 调库实现#

使用numpy手动实现MMAS（因为scikit-learn没有ACO实现）：

```python
"""
MAX-MIN Ant System (MMAS) 算法调库实现
用于解决TSP（旅行商问题）
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

class MMAS:
    """
    MAX-MIN Ant System (MMAS)
    改进的蚁群系统，使用信息素边界
    """
    
    def __init__(self, num_cities: int,
                 alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.75, initial_pheromone: float = 1.0):
        """
        初始化MMAS
        
        参数:
        - num_cities: 城市数量
        - alpha: 信息素重要性权重
        - beta: 启发式重要性权重
        - rho: 挥发率
        - initial_pheromone: 初始信息素（会动态调整）
        """
        self.n = num_cities
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # 信息素矩阵: τ[i,j]
        self.pheromones = np.full((num_cities, num_cities), 
                                   initial_pheromone, dtype=np.float32)
        np.fill_diagonal(self.pheromones, 0)  # 对角线为0
        
        # 距离矩阵和启发式信息
        self.distances = None
        self.heuristics = None  # η = 1/distance
        
        # 信息素边界
        self.tau_max = initial_pheromone
        self.tau_min = initial_pheromone * 0.1  # 初始简化
        
        # 历史最优解
        self.best_tour = None
        self.best_length = float('inf')
    
    def set_distances(self, distances: np.ndarray):
        """
        设置距离矩阵并计算启发式信息
        
        启发式: η_{ij} = 1 / d_{ij}
        """
        self.distances = distances
        # 避免除零
        self.heuristics = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.heuristics, 0)
    
    def compute_tau_max_min(self, best_length: float, b: float = 2.0):
        """
        计算信息素边界τ_max和τ_min
        
        公式:
        τ_max = 1 / (ρ * L_best)
        τ_min = τ_max * (1 - b) / (b * (avg_neighbors - 1))  # 简化
        """
        if best_length <= 0:
            return
        
        # τ_max
        self.tau_max = 1.0 / (self.rho * best_length)
        
        # τ_min (简化版本，假设平均分支因子b=5)
        b = min(self.n - 1, 5)  # 分支因子
        p_best = 0.05  # 最优解被选中的概率
        self.tau_min = (self.tau_max * (1 - np.power(p_best, 1.0/b))) / (
            np.power(p_best, 1.0/b) - 1 + 1e-10
        )
        self.tau_min = max(self.tau_min, 1e-10)  # 确保最小值
    
    def select_next_city(self, current: int, visited: List[bool]) -> int:
        """
        选择下一城市（PRP规则）
        
        数学原理:
        p_{ij} ∝ τ_{ij}^α · η_{ij}^β
        """
        probs = np.zeros(self.n)
        
        for j in range(self.n):
            if not visited[j] and self.distances[current, j] > 0:
                tau = self.pheromones[current, j] ** self.alpha
                eta = self.heuristics[current, j] ** self.beta
                probs[j] = tau * eta
        
        # 轮盘赌选择
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
            return np.random.choice(self.n, p=probs)
        else:
            # 如果没有可选城市，随机选一个未访问的
            unvisited = [j for j in range(self.n) if not visited[j]]
            return random.choice(unvisited)
    
    def construct_tour(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        单只蚂蚁构造解（TSP）
        
        返回: (路径, 路径长度)
        """
        visited = [False] * self.n
        tour = [start_city]
        visited[start_city] = True
        
        current_city = start_city
        total_length = 0.0
        
        for _ in range(self.n - 1):
            next_city = self.select_next_city(current_city, visited)
            tour.append(next_city)
            visited[next_city] = True
            total_length += self.distances[current_city, next_city]
            current_city = next_city
        
        # 回到起点
        total_length += self.distances[current_city, start_city]
        tour.append(start_city)
        
        return tour, total_length
    
    def update_pheromones(self, best_tour: List[int], best_length: float):
        """
        更新信息素: 挥发 + 沉积（只有最优蚂蚁）
        
        公式: τ = (1-ρ)τ + ρ·Δτ
        Δτ = 1/L_best (如果边在最优路径上)
        """
        # 1. 挥发
        self.pheromones *= (1 - self.rho)
        
        # 2. 沉积（只有最优蚂蚁）
        if best_length <= 0:
            return
        
        delta_tau = self.rho / best_length
        
        for i in range(self.n):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            self.pheromones[city_from, city_to] += delta_tau
            self.pheromones[city_to, city_from] += delta_tau  # 对称TSP
        
        # 3. 限制边界
        self.pheromones = np.clip(self.pheromones, 
                                   self.tau_min, self.tau_max)
    
    def fit(self, distances: np.ndarray, num_ants: int = 20,
              num_iterations: int = 100) -> Tuple[List[int], float]:
        """
        训练MMAS
        
        返回: (最优路径, 最优长度)
        """
        self.set_distances(distances)
        
        # 初始化τ_max
        self.compute_tau_max_min(np.sum(distances) / self.n)  # 初始估计
        
        history = []
        
        print(f"开始训练MMAS (蚂蚁数={num_ants}, 迭代={num_iterations})...")
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            # 每只蚂蚁构造解
            for ant in range(num_ants):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_tour = tour.copy()
                    self.best_length = length
            
            # 更新信息素（只有最优蚂蚁）
            best_idx = np.argmin(lengths)
            self.update_pheromones(tours[best_idx], lengths[best_idx])
            
            # 动态调整边界
            self.compute_tau_max_min(self.best_length)
            
            history.append(self.best_length)
            
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration+1}/{num_iterations}, "
                      f"当前最优长度: {self.best_length:.2f}")
        
        print(f"训练完成！最优长度: {self.best_length:.2f}")
        return self.best_tour, self.best_length, history


# 测试代码: 生成TSP实例
def generate_tsp_instance(n_cities: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成随机TSP实例（城市坐标和距离矩阵）
    """
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
    
    # 创建并训练MMAS
    mmas_solver = MMAS(num_cities=20, alpha=1.0, beta=2.0, rho=0.75)
    best_tour, best_length, history = mmas_solver.fit(
        distances, num_ants=20, num_iterations=100
    )
    
    print(f"\n最优路径: {best_tour}")
    print(f"最优长度: {best_length:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2, label='最优长度')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('MMAS 收敛曲线 (TSP, 20城市)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mmas_convergence.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练MMAS (蚂蚁数=20, 迭代=100)...
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
MMAS从零实现
实现核心MMAS算法，包含信息素边界
"""

import numpy as np
import random
from typing import List, Tuple

class MMASFromScratch:
    """
    MMAS算法从零实现
    
    核心思想:
    1. 信息素边界τ_min和τ_max
    2. 只有最优蚂蚁更新
    3. 挥发 + 沉积
    """
    
    def __init__(self, n_cities: int, alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.75):
        self.n = n_cities
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # 信息素
        self.tau = np.full((n_cities, n_cities), 1.0, dtype=np.float32)
        np.fill_diagonal(self.tau, 0)
        
        # 距离和启发式
        self.d = None
        self.eta = None
        
        # 边界
        self.tau_max = 1.0
        self.tau_min = 0.1
        
        # 最优解
        self.best_tour = None
        self.best_length = float('inf')
    
    def set_problem(self, distances: np.ndarray):
        """设置问题"""
        self.d = distances
        self.eta = 1.0 / (distances + 1e-10)
        np.fill_diagonal(self.eta, 0)
    
    def update_bounds(self, best_length: float):
        """更新边界"""
        if best_length <= 0:
            return
        self.tau_max = 1.0 / (self.rho * best_length)
        # 简化: τ_min = 0.1 * τ_max
        self.tau_min = 0.1 * self.tau_max
        self.tau_min = max(self.tau_min, 1e-10)
    
    def select_next(self, current: int, visited: List[bool]) -> int:
        """PRP规则选择"""
        probs = np.zeros(self.n)
        
        for j in range(self.n):
            if not visited[j] and self.d[current, j] > 0:
                probs[j] = (self.tau[current, j] ** self.alpha * 
                            self.eta[current, j] ** self.beta)
        
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            unvisited = [j for j in range(self.n) if not visited[j]]
            return random.choice(unvisited)
        
        probs /= probs_sum
        return np.random.choice(self.n, p=probs)
    
    def construct_tour(self, start: int = 0) -> Tuple[List[int], float]:
        """构造单条路径"""
        visited = [False] * self.n
        tour = [start]
        visited[start] = True
        
        current = start
        length = 0.0
        
        for _ in range(self.n - 1):
            next_city = self.select_next(current, visited)
            tour.append(next_city)
            visited[next_city] = True
            length += self.d[current, next_city]
            current = next_city
        
        # 回到起点
        length += self.d[current, start]
        tour.append(start)
        
        return tour, length
    
    def update_pheromones(self, best_tour: List[int], best_length: float):
        """更新信息素"""
        # 挥发
        self.tau *= (1 - self.rho)
        
        # 沉积（只有最优蚂蚁）
        if best_length <= 0:
            return
        
        delta_tau = self.rho / best_length
        
        for i in range(self.n):
            city_from = best_tour[i]
            city_to = best_tour[i+1]
            self.tau[city_from, city_to] += delta_tau
            self.tau[city_to, city_from] += delta_tau  # 对称
        
        # 限制边界
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
    
    def fit(self, distances: np.ndarray, num_ants: int = 20,
              num_iterations: int = 100) -> Tuple[List[int], float]:
        """训练MMAS"""
        self.set_problem(distances)
        self.update_bounds(np.sum(distances) / self.n)
        
        for iteration in range(num_iterations):
            tours = []
            lengths = []
            
            for ant in range(num_ants):
                start = random.randint(0, self.n-1)
                tour, length = self.construct_tour(start)
                tours.append(tour)
                lengths.append(length)
                
                if length < self.best_length:
                    self.best_tour = tour.copy()
                    self.best_length = length
            
            # 更新信息素
            self.update_pheromones(self.best_tour, self.best_length)
            
            # 更新边界
            self.update_bounds(self.best_length)
        
        return self.best_tour, self.best_length
```

## 9. 可视化与结果理解#

```python
"""
MMAS可视化代码
包括: 收敛曲线、信息素热力图、路径可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_convergence(history: list, method: str = "MMAS"):
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
    plt.ylabel('路径长度')
    plt.title(f'{method} 收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mmas_convergence.png', dpi=150)
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
    for i in range(len(tour)-1):
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
    plt.savefig('mmas_tour.png', dpi=150)
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
    tau_masked = ma.masked_where(np.tril(np.ones((n, n))) == 0, tau)
    
    im = ax.imshow(tau_masked, cmap='YlOrRd', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('城市')
    ax.set_ylabel('城市')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('mmas_pheromones.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
MMAS模型评估代码
评估算法的优化性能
"""

import numpy as np
from typing import Dict

def evaluate_mmas(mmas_solver, distances: np.ndarray,
                num_runs: int = 5, num_iterations: int = 100) -> Dict:
    """
    多次运行评估MMAS性能
    
    评估指标:
    1. 平均最优长度：多次运行的平均最优长度
    2. 标准差：衡量稳定性
    3. 最优长度：多次运行的最佳结果
    """
    best_lengths = []
    
    for run in range(num_runs):
        best_tour, best_length = mmas_solver.fit(
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

1. **问题：信息素边界计算错误**
   - 现象：τ_max或τ_min异常
   - 原因：L_best=0或公式实现错误
   - 解决方案：检查L_best>0，使用简化边界（如τ_min=0.1*τ_max）。

2. **问题：只有最优蚂蚁更新失效**
   - 现象：收敛极慢
   - 原因：当前最优蚂蚁质量差
   - 解决方案：使用全局历史最优蚂蚁更新，或结合当前最优。

**调参层面易错点：**

1. **问题：挥发率ρ设置不当**
   - 现象：ρ太大会丢失历史信息，太小会早熟收敛
   - 原因：没有根据问题规模调整
   - 解决方案：小规模用0.5~0.7，大规模用0.75~0.95。

2. **问题：α和β比例不当**
   - 现象：过度依赖信息素或启发式
   - 原因：参数设置不合理
   - 解决方案：通常α=1，β=2~5，启发式更重要。

## 12. 学习总结#

**核心思想回顾：** MMAS在ACS基础上增加了信息素边界τ_min和τ_max，所有边的信息素被限制在此范围内。只有最优蚂蚁（当前最优或全局最优）才更新信息素，从而强化最优解同时保持探索能力。

**关键公式：**
1. 信息素更新：τ = (1-ρ)τ + ρ·Δτ，其中Δτ = 1/L_best
2. 边界限制：τ = max(τ_min, min(τ, τ_max))
3. τ_max = 1/(ρ·L_best)，τ_min根据分支因子计算

**与前序算法或相关算法的联系：**
- 是**Ant Colony System (ACS)**的改进版本
- 与**Ant System (AS)**的区别：MMAS有信息素边界，只有最优蚂蚁更新
- 是**Modified MMAS**（本书第5章）的基础
- 与**Q-ACS**等混合算法结合形成本书的多主体学习框架

**后续学习方向：**
- **Modified MMAS**：本书第5章，应用于CVRP和VRPTW
- **Q-ACS Learning**：结合Q-learning和ACS（本书核心贡献）
- **MACS-VRPTW**：多蚁群系统解决带时间窗的车辆路径

## 13. 练习题与思考题#

**基础题1：** 在MMAS中，如果τ_min=0且τ_max=∞会发生什么？这相当于哪个算法？

**答案：**
- τ_min=0：信息素可以完全挥发，失去探索能力
- τ_max=∞：信息素可以无限积累，过度集中
- 相当于**Ant System (AS)**或**ACS**（如果没有边界），因为没有边界限制。

**基础题2：** 为什么MMAS只有最优蚂蚁才更新信息素？这有什么优缺点？

**答案：**
- **优点**：强化最优解，避免次优解的信息素干扰，收敛更快
- **缺点**：学习信号稀疏，如果最优蚂蚁质量差（如初期），收敛会很慢
- **改进**：可以使用"当前最优+全局最优"共同更新，或增加"第二最优"蚂蚁更新。

**进阶题1：** 推导τ_max和τ_min的计算公式（简化版本）。如果平均分支因子b=5，p_best=0.05，L_best=100，ρ=0.75，请计算τ_max和τ_min。

**答案：**
- τ_max = 1/(ρ·L_best) = 1/(0.75×100) = 0.0133
- τ_min简化：τ_min = 0.1×τ_max = 0.00133
- 完整公式：τ_min = τ_max × (1-b^(1/b))/(b^(1/b)-1) ≈ τ_max × 0.45 (当b=5)

**进阶题2：** 分析MMAS的时间复杂度（每次迭代）。与ACS相比如何？

**答案：**
- **每次迭代复杂度**：O(m·n²)（m蚂蚁，每只构造O(n²)选择概率）
- **信息素更新**：O(n)（只有最优路径上的边）
- 与ACS相比：ACS更新所有蚂蚁，复杂度O(m·n)；MMAS只有最优蚂蚁，更新更快但可能收敛更慢。

**开放思考题：** MMAS中的信息素边界能否动态调整？例如根据迭代次数或解的质量变化？

**参考答案思路：**
1. **根据迭代次数**：初期τ_max较大鼓励探索，后期τ_max减小鼓励利用
2. **根据解质量变化**：如果连续多次迭代最优解不变，增大τ_min增加探索；如果解质量提升，减小τ_min鼓励利用
3. **根据蚂蚁多样性**：计算信息素分布的熵，熵小（集中）时增大τ_min，熵大时减小τ_min
4. **自适应MMAS**：结合多种动态调整策略

## 14. 学习路径建议#

**前置算法：**
1. **Ant System (AS)**：理解基础蚁群行为
2. **Ant Colony System (ACS)**：理解PRP规则和局部/全局更新
3. **组合优化基础**：理解TSP、VRP等问题形式化

**平行算法：**
1. **Genetic Algorithms**：另一种元启发式算法
2. **Particle Swarm Optimization**：群智能的另一代表

**进阶算法（本书后续）：**
1. **Modified MMAS**（第5章）：应用于CVRP和VRPTW，动态调整边界
2. **Q-ACS Learning**（第2、3章）：结合Q-learning和ACS，本书核心贡献
3. **MACS-VRPTW**：多蚁群系统解决VRPTW

**推荐资源：**
1. **教材**：Dorigo & Stützle, "Ant Colony Optimization" (2004)
2. **论文**：Stützle & Hoos (2000), "MAX-MIN ant system"
3. **本书章节**：第1、3、5章
4. **代码实践**：ACO算法Python实现教程
