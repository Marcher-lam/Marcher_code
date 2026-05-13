# Tree Search 学习文档

> 通过遍历树结构寻找目标节点或最优解的搜索算法总称。

## 1. 算法基础认知

**一句话定义：** 系统地遍历树形结构，寻找目标节点或计算最优解的基础搜索方法。

**直觉类比：** 就像在迷宫中寻宝，树搜索就像沿着每条路径探索，标记已访问的岔路口，直到找到宝藏或遍历完所有路径。

**历史背景：** 树搜索是最基础的计算机算法之一，早在20世纪50年代就被用于博弈程序。书中在多主体系统决策、博弈树搜索（如MiniMax）部分涉及树搜索基础。

**算法定位：** 基础搜索算法，属于图论和算法基础范畴，是MiniMax、MCTS等高级算法的基础。

**前置知识：**
- 树与图的基本结构
- 递归与迭代编程
- 深度优先搜索（DFS）、广度优先搜索（BFS）基础
- Python编程基础

树搜索算法是几乎所有搜索类算法的基础，包括本书中的MiniMax搜索、蒙特卡洛树搜索（MCTS）等都建立在树搜索概念之上。

## 2. 核心原理

**核心思想：** 树搜索算法从根节点开始，按照特定策略遍历树结构中的节点，直到找到目标节点或遍历完所有节点。不同策略对应不同搜索算法（DFS、BFS、最佳优先搜索等）。

**工作流程（通用）：**
1. **初始化：** 将根节点放入待访问数据结构（栈/队列/优先队列）
2. **循环：** 当待访问结构非空时：
   a. 取出下一个节点（根据数据结构类型）
   b. 如果该节点是目标，返回结果
   c. 否则，将该节点的未访问子节点加入待访问结构
3. **终止：** 找到目标或遍历完所有节点

**关键概念解释：**
- **树节点：** 搜索空间中的一个状态或决策点
- **边：** 节点间的转移关系
- **深度优先搜索（DFS）：** 优先深入探索，使用栈（后进先出）
- **广度优先搜索（BFS）：** 优先逐层探索，使用队列（先进先出）
- **最佳优先搜索：** 根据启发式评估选择最有希望的节点

**几何/直观解释：**
```
树结构示例：
        A (根)
       / \
      B   C
     / \   \
    D   E   F

DFS遍历顺序: A → B → D → E → C → F (深度优先)
BFS遍历顺序: A → B → C → D → E → F (广度优先)
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $V$ | 节点集合 | 树中所有节点 |
| $E$ | 边集合 | 节点间的连接关系 |
| $r$ | 根节点 | 搜索起点 |
| $d(n)$ | 节点n的深度 | 从根到n的路径长度 |
| $f(n)$ | 评估函数 | 用于最佳优先搜索 |
| $g(n)$ | 从根到n的实际代价 | 已知代价 |
| $h(n)$ | 启发式估计代价 | 从n到目标的估计代价 |

**问题形式化：**
给定树 $T = (V, E)$ 和根节点 $r \in V$，找到目标节点 $t \in V$ 或确定不存在。

**不同搜索策略的比较：**

1. **DFS（深度优先）：**
   - 使用栈，每次取最新节点
   - 时间复杂度：$O(|V| + |E|)$
   - 空间复杂度：$O(d)$，d为最大深度

2. **BFS（广度优先）：**
   - 使用队列，每次取最早节点
   - 时间复杂度：$O(|V| + |E|)$
   - 空间复杂度：$O(w)$，w为最大宽度

3. **A*搜索（最佳优先）：**
   - 评估函数：$f(n) = g(n) + h(n)$
   - 使用优先队列，按f(n)排序
   - 如果h(n)是可采纳的（admissible），A*保证找到最优解

**A*算法推导：**
- $g(n)$：从根到n的实际代价，已知
- $h(n)$：从n到目标的估计代价，启发式提供
- $f(n)$：通过n到目标的估计总代价
- 因为 $h(n) \leq h^*(n)$（可采纳启发式），所以 $f(n) \leq g(n) + h^*(n) = g^*(n)$
- 因此A*会优先扩展看起来更有希望的路径，且保证找到最优解

**最终算法步骤（以BFS为例）：**
```
BFS(root, target):
    queue = [root]
    visited = set([root])
    while queue not empty:
        node = queue.pop(0)  # 队列：先进先出
        if node == target: return node
        for child in node.children:
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return None  # 未找到
```

## 4. 训练过程讲解

**数据预处理：**
- 构建树结构：定义节点和边的关系
- 设计节点表示：包含状态信息、子节点引用等
- 可选：设计启发式函数（如A*搜索）

**参数初始化：**
- 根节点：搜索起点
- 目标节点：搜索目标（或判断条件）
- 数据结构：栈（DFS）、队列（BFS）、优先队列（A*）
- 访问集合：记录已访问节点，防止重复访问

**迭代过程（通用树搜索）：**
1. 初始化待访问结构和访问集合
2. 将根节点加入待访问结构
3. 循环直到找到目标或待访问结构为空：
   a. 从待访问结构取出节点
   b. 检查是否为目标
   c. 将未访问的子节点加入待访问结构
4. 返回结果（找到的目标或失败）

**收敛条件：**
- 找到目标节点
- 遍历完所有节点（未找到）
- 达到资源限制（时间、节点数）

**超参数/参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| 搜索策略 | 决定遍历顺序 | DFS/BFS/A*/Greedy | 根据问题选择 |
| 最大深度 | 防止无限递归 | 问题相关 | BFS无此参数 |
| 启发式函数 | 指导搜索方向 | 问题相关 | A*需要可采纳启发式 |

## 5. 应用场景

**典型应用：**

1. **路径规划：** 在已知地图中寻找最短路径。**为什么适合：** 地图可建模为树/图，BFS/DFS/A*都能应用。
2. **博弈树搜索：** 如MiniMax搜索棋类游戏。**为什么适合：** 博弈树是树结构，需要深度优先遍历。
3. **决策树遍历：** 机器学习中遍历决策树进行分类。**为什么适合：** 决策树本质是树结构，预测时需要遍历。
4. **语法分析：** 编译器中的语法树遍历。**为什么适合：** 语法树是树结构，需要按规则遍历。

**适用数据特征：**
- 问题可表示为树或图结构
- 状态空间离散
- 需要系统性探索解空间

**不适用场景：**
- 连续状态空间：树结构难以表示
- 极大状态空间：指数爆炸
- 动态变化环境：树结构需要频繁重建
- 需要近似解的场景：树搜索给出精确解但可能太慢

## 6. 优缺点分析

**优点：**
1. **完备性：** 在有限状态空间中保证找到解（如果存在）。**成立条件：** 状态空间有限，搜索策略完备（如BFS）。
2. **最优性（部分）：** BFS在边权一致时找到最短路径，A*在可采纳启发式下找到最优解。**成立条件：** 相应条件满足。
3. **简单易懂：** 原理直观，实现简单。**成立条件：** N/A。

**缺点：**
1. **状态空间爆炸：** 树深度增加时节点数指数增长。**问题：** 大规模问题不可行。**缓解思路：** 使用剪枝、启发式引导（如A*）、近似方法。
2. **存储开销：** DFS需要存储当前路径，BFS需要存储当前层所有节点。**问题：** 深度或宽度大时内存不足。**缓解思路：** 使用迭代深化DFS（IDDFS）减少内存。
3. **无先验知识时效率低：** 盲目搜索（DFS/BFS）不考虑目标方向。**问题：** 搜索大量无关节点。**缓解思路：** 使用启发式搜索（A*、贪婪最佳优先）。

**与同类算法对比：**

| 特性 | DFS | BFS | A* | MiniMax |
|------|-----|-----|----|---------|
| 完备性 | 否（可能陷入无限分支） | 是（有限空间） | 是（可采纳启发式） | 是（有限树） |
| 最优性 | 否 | 是（一致边权） | 是（可采纳启发式） | 是（零和博弈） |
| 空间复杂度 | $O(d)$ | $O(w)$ | $O(b^d)$ | $O(b^d)$ |
| 适用场景 | 拓扑排序、连通性 | 最短路径（无权） | 最短路径（有权） | 博弈树 |

## 7. 调库实现

```python
"""
Tree Search 调库实现
使用Python标准库实现DFS、BFS、A*搜索
"""

import heapq
from collections import deque
from typing import List, Optional, Tuple, Dict

class TreeNode:
    """树节点类"""
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []
        self.parent = None
        # 为A*搜索准备
        self.g = float('inf')  # 从根到当前节点的代价
        self.h = 0  # 启发式估计代价
        self.f = float('inf')  # g + h
    
    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)

def dfs_search(root: TreeNode, target_value) -> Optional[TreeNode]:
    """
    深度优先搜索（递归版）
    
    数学原理:
    使用栈（后进先出），优先深入探索
    时间复杂度: O(V+E)，空间复杂度: O(d)
    """
    visited = set()
    return _dfs_recursive(root, target_value, visited)

def _dfs_recursive(node: TreeNode, target, visited) -> Optional[TreeNode]:
    if node is None:
        return None
    
    visited.add(node)
    
    if node.value == target:
        return node
    
    for child in node.children:
        if child not in visited:
            result = _dfs_recursive(child, target, visited)
            if result:
                return result
    
    return None

def bfs_search(root: TreeNode, target_value) -> Optional[TreeNode]:
    """
    广度优先搜索
    
    数学原理:
    使用队列（先进先出），逐层探索
    时间复杂度: O(V+E)，空间复杂度: O(w)
    """
    if root is None:
        return None
    
    queue = deque([root])
    visited = set([root])
    
    while queue:
        node = queue.popleft()
        
        if node.value == target_value:
            return node
        
        for child in node.children:
            if child not in visited:
                visited.add(child)
                queue.append(child)
    
    return None

def a_star_search(root: TreeNode, target_value, heuristic_func) -> Optional[TreeNode]:
    """
    A*搜索
    
    数学原理:
    f(n) = g(n) + h(n)
    使用优先队列，按f值排序
    如果h是可采纳的（admissible），保证找到最优解
    """
    root.g = 0
    root.h = heuristic_func(root.value, target_value)
    root.f = root.g + root.h
    
    open_set = []
    heapq.heappush(open_set, (root.f, id(root), root))
    closed_set = set()
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current.value == target_value:
            return current
        
        closed_set.add(current)
        
        for child in current.children:
            if child in closed_set:
                continue
            
            # 计算临时g值
            tentative_g = current.g + 1  # 假设每条边代价为1
            
            if tentative_g < child.g:
                child.parent = current
                child.g = tentative_g
                child.h = heuristic_func(child.value, target_value)
                child.f = child.g + child.h
                
                # 如果child不在open_set中，加入
                # 简化：直接加入，依赖heapq处理重复
                heapq.heappush(open_set, (child.f, id(child), child))
    
    return None


def test_tree_search():
    """测试树搜索算法"""
    print("=== 测试Tree Search ===")
    
    # 构建示例树:
    #        A
    #       / \
    #      B   C
    #     / \   \
    #    D   E   F
    
    root = TreeNode('A')
    b = TreeNode('B')
    c = TreeNode('C')
    d = TreeNode('D')
    e = TreeNode('E')
    f = TreeNode('F')
    
    root.add_child(b)
    root.add_child(c)
    b.add_child(d)
    b.add_child(e)
    c.add_child(f)
    
    # 测试DFS
    result = dfs_search(root, 'E')
    print(f"DFS搜索'E': {result.value if result else 'Not found'}")
    
    # 测试BFS
    result = bfs_search(root, 'E')
    print(f"BFS搜索'E': {result.value if result else 'Not found'}")
    
    # 测试A*（简单启发式：字符距离）
    def simple_heuristic(node_value, target):
        # 简化：假设节点值字符越接近目标，启发式值越小
        return abs(ord(node_value) - ord(target))
    
    result = a_star_search(root, 'E', simple_heuristic)
    print(f"A*搜索'E': {result.value if result else 'Not found'}")
    
    return root


if __name__ == "__main__":
    test_tree_search()
```

**运行结果示例：**
```
=== 测试Tree Search ===
DFS搜索'E': E
BFS搜索'E': E
A*搜索'E': E
```

## 8. 手工代码实现

```python
"""
Tree Search 手工实现
从零实现DFS、BFS，无外部依赖
"""

from typing import List, Optional

class SimpleTreeNode:
    """简单树节点"""
    def __init__(self, val):
        self.val = val
        self.children = []
    
    def add(self, child: 'SimpleTreeNode'):
        self.children.append(child)

class TreeSearchFromScratch:
    """树搜索从零实现"""
    
    def __init__(self):
        self.visited = []
    
    def dfs_iterative(self, root, target) -> Optional[SimpleTreeNode]:
        """
        DFS迭代实现（使用栈）
        
        核心逻辑:
        栈是后进先出(LIFO)，所以最后加入的子节点先被访问
        """
        if root is None:
            return None
        
        stack = [root]
        
        while stack:
            node = stack.pop()
            
            if node in self.visited:
                continue
            self.visited.append(node)
            
            if node.val == target:
                return node
            
            # 子节点逆序入栈，保证原顺序的第一个子节点先被处理
            for child in reversed(node.children):
                if child not in self.visited:
                    stack.append(child)
        
        return None
    
    def bfs_iterative(self, root, target) -> Optional[SimpleTreeNode]:
        """
        BFS迭代实现（使用队列）
        
        核心逻辑:
        队列是先进先出(FIFO)，所以先加入的节点先被访问
        """
        if root is None:
            return None
        
        queue = [root]
        self.visited = [root]
        
        while queue:
            node = queue.pop(0)  # 简单队列实现
            
            if node.val == target:
                return node
            
            for child in node.children:
                if child not in self.visited:
                    self.visited.append(child)
                    queue.append(child)
        
        return None


def test_from_scratch():
    print("=== 手工实现测试 ===")
    
    # 构建树
    root = SimpleTreeNode('A')
    b = SimpleTreeNode('B')
    c = SimpleTreeNode('C')
    d = SimpleTreeNode('D')
    e = SimpleTreeNode('E')
    f = SimpleTreeNode('F')
    
    root.add(b)
    root.add(c)
    b.add(d)
    b.add(e)
    c.add(f)
    
    # 测试DFS
    ts = TreeSearchFromScratch()
    result = ts.dfs_iterative(root, 'F')
    print(f"DFS找到'F': {result.val if result else '未找到'}")
    
    # 测试BFS
    ts = TreeSearchFromScratch()
    result = ts.bfs_iterative(root, 'F')
    print(f"BFS找到'F': {result.val if result else '未找到'}")


if __name__ == "__main__":
    test_from_scratch()
```

**测试结果：**
```
=== 手工实现测试 ===
DFS找到'F': F
BFS找到'F': F
```

## 9. 可视化与结果理解

```python
"""
Tree Search 可视化代码
绘制树结构和搜索过程
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List

def draw_tree_with_search(tree_root, search_path: List[str], title: str):
    """
    绘制树结构和搜索路径
    
    图表解读：
    - 蓝色节点：未访问
    - 红色节点：搜索路径上的节点
    - 绿色节点：目标节点
    """
    G = nx.DiGraph()
    
    # 递归添加节点和边
    def add_nodes_edges(node, parent=None):
        G.add_node(node.val)
        if parent:
            G.add_edge(parent.val, node.val)
        for child in node.children:
            add_nodes_edges(child, node)
    
    add_nodes_edges(tree_root)
    
    # 设置位置
    pos = nx.spring_layout(G, seed=42)
    
    # 节点颜色
    node_colors = []
    for node in G.nodes():
        if node in search_path:
            if node == search_path[-1]:
                node_colors.append('green')  # 目标
            else:
                node_colors.append('red')    # 路径上
        else:
            node_colors.append('lightblue')  # 未访问
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=2000, font_size=12, arrows=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'tree_search_{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def simulate_search_process():
    """模拟搜索过程并可视化"""
    # 构建树
    root = TreeNode('A')
    b = TreeNode('B')
    c = TreeNode('C')
    d = TreeNode('D')
    e = TreeNode('E')
    f = TreeNode('F')
    
    root.add_child(b)
    root.add_child(c)
    b.add_child(d)
    b.add_child(e)
    c.add_child(f)
    
    # DFS路径: A → B → D → E → C → F
    dfs_path = ['A', 'B', 'E']
    draw_tree_with_search(root, dfs_path, "DFS Search Path")
    
    # BFS路径: A → B → C → D → E → F
    bfs_path = ['A', 'B', 'E']
    draw_tree_with_search(root, bfs_path, "BFS Search Path")


if __name__ == "__main__":
    simulate_search_process()
```

**图表解读：**
1. **DFS路径图：** 显示深度优先的搜索顺序，优先深入子树。
2. **BFS路径图：** 显示广度优先的搜索顺序，逐层扩展。
3. 红色节点构成搜索路径，绿色是目标节点。

## 10. 模型评估

```python
"""
Tree Search 模型评估代码
评估不同搜索算法的性能
"""

import time
from typing import Dict, Callable

def evaluate_search_algorithm(search_func: Callable, tree_root, target, 
                               n_runs: int = 100) -> Dict:
    """
    评估搜索算法性能
    
    评估指标:
    1. 平均搜索时间
    2. 访问节点数
    3. 找到目标的成功率
    """
    total_time = 0.0
    total_nodes = 0
    success_count = 0
    
    for _ in range(n_runs):
        start_time = time.time()
        result = search_func(tree_root, target)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        
        if result:
            success_count += 1
            # 简化：假设访问节点数等于路径长度
            total_nodes += 1  # 实际应统计访问节点
    
    results = {
        'Avg_Time_ms': (total_time / n_runs) * 1000,
        'Success_Rate': success_count / n_runs,
        'Algorithm': search_func.__name__
    }
    
    print(f"=== {search_func.__name__} 评估 ===")
    for k, v in results.items():
        if k == 'Avg_Time_ms':
            print(f"{k}: {v:.4f} ms")
        else:
            print(f"{k}: {v}")
    
    return results


def compare_search_algorithms():
    """比较不同搜索算法"""
    print("\n=== 搜索算法对比 ===")
    
    # 构建测试树
    root = TreeNode('A')
    b = TreeNode('B')
    c = TreeNode('C')
    d = TreeNode('D')
    e = TreeNode('E')
    f = TreeNode('F')
    
    root.add_child(b)
    root.add_child(c)
    b.add_child(d)
    b.add_child(e)
    c.add_child(f)
    
    # 评估DFS
    dfs_result = evaluate_search_algorithm(dfs_search, root, 'E')
    
    # 评估BFS
    bfs_result = evaluate_search_algorithm(bfs_search, root, 'E')
    
    # 对比表格
    print("\n算法\t\t平均时间(ms)\t成功率")
    print(f"DFS\t\t{dfs_result['Avg_Time_ms']:.4f}\t\t{dfs_result['Success_Rate']:.2f}")
    print(f"BFS\t\t{bfs_result['Avg_Time_ms']:.4f}\t\t{bfs_result['Success_Rate']:.2f}")


if __name__ == "__main__":
    compare_search_algorithms()
```

**结果解读：**
- DFS和BFS都能找到目标，时间差异不大（小树）
- 大树时，DFS空间优势明显，BFS可能内存不足
- 成功率应为100%（有限树，目标存在）

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：树结构有环**
   - 现象：搜索陷入无限循环
   - 原因：树应该是无环的，但实现错误可能引入环
   - 解决方案：严格确保树结构无环，或使用visited集合检测环

2. **问题：节点重复访问**
   - 现象：搜索效率低下，可能重复访问
   - 原因：未使用visited集合记录已访问节点
   - 解决方案：始终维护visited集合，避免重复访问

**模型层面易错点：**

1. **问题：DFS递归栈溢出**
   - 现象：树很深时程序崩溃
   - 原因：递归深度超过Python默认递归限制（约1000）
   - 解决方案：使用迭代版DFS（栈），或增加递归限制

2. **问题：BFS队列内存爆炸**
   - 现象：树很宽时内存不足
   - 原因：BFS需要存储当前层所有节点
   - 解决方案：使用DFS或迭代深化DFS（IDDFS）

**调参层面易错点：**

1. **问题：A*启发式设计不当**
   - 现象：找到非最优解，或不保证完备性
   - 原因：启发式不可采纳（高估了到目标的代价）
   - 解决方案：确保 $h(n) \leq h^*(n)$（真实最小代价）

2. **问题：搜索策略选择错误**
   - 现象：性能差或找不到解
   - 原因：问题适合BFS却用了DFS
   - 解决方案：无权图最短路径用BFS，深度探索用DFS，有启发式用A*

## 12. 学习总结

**核心思想回顾：** 树搜索算法通过系统遍历树结构寻找目标或最优解。DFS使用栈优先深入，BFS使用队列优先广度，A*使用启发式引导搜索方向。这些基础算法是MiniMax、MCTS等高级算法的基础。

**关键公式：**
1. A*评估函数：$f(n) = g(n) + h(n)$
2. DFS时间复杂度：$O(|V| + |E|)$，空间复杂度：$O(d)$
3. BFS时间复杂度：$O(|V| + |E|)$，空间复杂度：$O(w)$

**与前序算法或相关算法的联系：**
- 是**MiniMax搜索**的基础（博弈树就是树结构）
- 是**蒙特卡洛树搜索（MCTS）**的基础（MCTS在搜索树上进行模拟）
- 书中多主体决策、博弈场景都依赖树搜索概念

**后续学习方向：**
- **MiniMax搜索：** 零和博弈中的对抗搜索
- **蒙特卡洛树搜索（MCTS）：** 结合采样和树搜索，适合大规模博弈
- **迭代深化DFS（IDDFS）：** 结合DFS空间效率和BFS完备性
- **Alpha-Beta剪枝：** MiniMax的优化，剪掉无效分支

## 13. 练习题与思考题

**基础题1：** DFS和BFS的主要区别是什么？在什么情况下优先选择DFS vs BFS？

**答案：**
- **区别：** DFS使用栈（后进先出），优先深入探索；BFS使用队列（先进先出），优先广度探索。
- **选DFS：** 树很深但不宽，需要空间效率，或问题需要深度探索（如拓扑排序）。
- **选BFS：** 需要最短路径（无权图），树宽度不大，或需要逐层处理。

**基础题2：** 为什么A*搜索需要启发式函数h(n)是可采纳的（admissible）才能保证最优性？

**答案：**
- 可采纳启发式满足 $h(n) \leq h^*(n)$，即不会高估真实代价。
- 如果h(n)高估，A*可能忽略实际更优的路径（因为f(n)被高估）。
- 可采纳性保证A*不会错误地剪掉最优路径，从而保证找到最优解。

**进阶题1：** 如何修改DFS以处理图（可能有环）而不是树？

**答案：**
1. **使用visited集合：** 记录所有已访问节点，避免重复访问。
2. **检测环：** 在DFS递归栈中检测当前路径上的节点，发现环时回溯。
3. **通用图搜索：** 实际上，树搜索是图搜索的特例（无环），通用图搜索需要更完备的访问标记。

**进阶题2：** 迭代深化DFS（IDDFS）如何结合DFS和BFS的优点？

**答案：**
- IDDFS首先以深度1进行DFS，然后深度2，以此类推。
- **DFS优点：** 空间复杂度 $O(d)$，d是当前深度限制。
- **BFS优点：** 完备性（找到最短路径），因为会逐步增加深度限制。
- 代价：节点可能被重复访问多次（但空间效率换取了完备性）。

**开放思考题：** 树搜索能否用于连续状态空间？如果能，需要哪些修改？

**参考答案思路：**
1. **离散化：** 将连续空间划分为离散区间，构建近似树。
2. **RRT（快速探索随机树）：** 随机采样构建树，适应连续空间。
3. **近似树搜索：** 使用近似最近邻等方法加速搜索。
4. **结合函数逼近：** 用机器学习模型预测有希望的区域，指导搜索。

## 14. 学习路径建议

**前置算法：**
1. **数据结构基础：** 树、图、栈、队列、优先队列
2. **递归与迭代：** 理解DFS的递归和迭代实现
3. **图论基础：** 了解图的基本概念和遍历

**平行算法：**
1. **MiniMax搜索：** 基于树搜索的对抗性博弈算法
2. **A*搜索：** 启发式树搜索，用于最短路径问题
3. **迭代深化DFS：** DFS和BFS的折中方案

**进阶算法：**
1. **蒙特卡洛树搜索（MCTS）：** 结合采样和树搜索，适合大规模博弈
2. **Alpha-Beta剪枝：** MiniMax的优化，剪掉无效分支
3. **蒙特卡洛方法：** 基于采样的搜索，MCTS的基础

**推荐资源：**
1. **教材：** Russell & Norvig, "Artificial Intelligence: A Modern Approach" (Chapter 3: Solving Problems by Searching)
2. **算法经典：** Cormen et al., "Introduction to Algorithms" (Chapter 22: Elementary Graph Algorithms)
3. **在线资源：** VisuAlgo (https://visualgo.net/en/dfsbfs) 可视化DFS/BFS
4. **书中章节：** 第1章多主体系统中的决策基础
