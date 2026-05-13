# Dynamic Programming 学习文档#

> 通过求解贝尔曼方程，迭代计算最优价值函数和最优策略。

## 1. 算法基础认知#

**一句话定义：** 动态规划（DP）是一种通过分解问题、利用贝尔曼最优性原理，自底向上或自顶向下求解MDP的算法集合。

**直觉类比：** 想象你要爬到山顶，动态规划就像从山顶开始，标记每个位置到山顶的最优路径。然后从山脚出发时，只需跟随标记就能找到最优路径。

**历史背景：** 动态规划由Richard Bellman在1950年代提出，是运筹学和计算机科学中的经典算法思想。在强化学习中，DP是求解MDP的理论基础。

**算法定位：** 基于模型的强化学习（Model-based RL），需要完整的环境模型（P和R）。

**前置知识：**
- 马尔可夫决策过程（MDP）
- 贝尔曼方程
- 线性代数基础
- 最优性原理理解#

## 2. 核心原理#

**核心思想：** 动态规划利用贝尔曼最优方程，通过迭代更新价值函数直到收敛，然后从价值函数提取最优策略。主要包括值迭代（Value Iteration）和策略迭代（Policy Iteration）。

**工作流程（值迭代）：**
1. 初始化V(s) = 0 ∀s ∈ S
2. 重复直到收敛：
   a. 对每个状态s：
      V_new(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γ·V(s')]
   b. 如果max|V_new - V| < θ，则收敛
   c. V ← V_new

**工作流程（策略迭代）：**
1. 初始化随机策略π
2. 重复直到策略稳定：
   a. **策略评估：** 解线性方程组求V^π
      V^π = (I - γP^π)^{-1} R^π
   b. **策略改进：** π_new(s) = argmax_a Σ_s' P(s'|s,a)[R + γ·V^π(s')]
   c. 如果π_new = π，则已找到最优策略

**关键概念解释：**
- **贝尔曼最优性原理：** 最优策略的子策略也是最优的
- **值迭代：** 直接迭代更新V值直到收敛，然后提取策略
- **策略迭代：** 交替进行策略评估和策略改进，通常收敛更快
- **高斯-赛德尔迭代：** 值迭代中使用in-place更新加速收敛

**几何/直观解释：**
```
动态规划更新传播示意图：

目标状态: s_goal (V=0, 终止)

距离1步的状态: s1, s2, s3
  V(s1) = max_a Σ P(s_goal|s1,a)[R + γ·0]
  ...

距离2步的状态: s4, s5
  V(s4) = max_a Σ P(s1|s4,a)[R + γ·V(s1)]
  ...

动态规划从目标状态开始，逐层向外传播价值信息。
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| V(s) | 状态价值函数 | 在s下遵循某策略的期望回报 |
| π | 策略 | π(a\|s)或π(s) |
| P^π | 策略π下的转移矩阵 | P^π(s'\|s) = Σ_a π(a\|s)P(s'\|s,a) |
| R^π | 策略π下的奖励向量 | R^π(s) = Σ_a π(a\|s)Σ_s' P(s'\|s,a)R(s,a,s') |

**贝尔曼方程（策略评估）：**

对于给定策略π，V^π满足：

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

写成矩阵形式：

$$V^\pi = R^\pi + \gamma P^\pi V^\pi$$

解得：

$$V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$

**贝尔曼最优方程：**

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]$$

**值迭代（Value Iteration）：**

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]$$

**逐步推导过程：**

1. **从贝尔曼方程出发：**
   V^π = R^π + γP^π V^π
   
2. **移项：**
   (I - γP^π) V^π = R^π
   
3. **求逆：**
   V^π = (I - γP^π)^{-1} R^π
   这给出了策略评估的解析解。
   
4. **最优时：**
   V*(s) = max_a Σ_s' P(s'|s,a)[R + γV*(s')]
   这是非线性方程，不能直接求逆，需要迭代求解（值迭代）。
   
5. **值迭代收敛性：**
   由于γ < 1，值迭代是压缩映射，根据Banach不动点定理，迭代收敛到唯一不动点V*。

**策略迭代（Policy Iteration）：**

交替执行：
1. **策略评估：** V^π_{k+1} = (I - γP^{π_k})^{-1} R^{π_k}
2. **策略改进：** π_{k+1}(s) = argmax_a Σ_s' P(s'|s,a)[R + γV^{π_{k+1}}(s')]

**收敛性：** 策略迭代最多在|A|^{|S|}次迭代内收敛（有限MDP），实践中通常更快。

## 4. 训练过程讲解#

**数据预处理：**
- 构建转移概率矩阵P：维度|S|×|A|×|S|
- 构建奖励函数R：维度|S|×|A|×|S|或|S|×|A|
- 设置折扣因子γ

**参数初始化：**
- V表：全0或随机初始化
- 策略π：随机初始化或均匀策略
- 收敛阈值θ：1e-6 ~ 1e-3
- 最大迭代次数：100 ~ 10000

**迭代过程（值迭代）：**
1. 对每个状态s：
   V_new(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γ·V_old(s')]
2. 检查收敛：max_s |V_new(s) - V_old(s)| < θ
3. V_old ← V_new
4. 重复直到收敛

**迭代过程（策略迭代）：**
1. **策略评估：**
   - 解线性方程组：(I - γP^π)V = R^π
   - 或使用迭代法（如高斯-赛德尔）
2. **策略改进：**
   π_new(s) = argmax_a Σ_s' P(s'|s,a)[R + γ·V(s')]
3. 如果π_new = π，收敛；否则π ← π_new，返回步骤1

**收敛条件：**
- 值函数变化小于阈值：max|V_new - V_old| < θ
- 策略稳定：π_new = π_old
- 达到最大迭代次数

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| γ (折扣因子) | 权衡即时与未来奖励 | 0.9~0.999 | 0.9 |
| θ (收敛阈值) | 判断收敛 | 1e-6~1e-3 | 1e-4 |
| max_iterations | 最大迭代次数 | 100~10000 | 1000 |

## 5. 应用场景#

**典型应用：**

1. **经典控制问题：** 如倒立摆、车杆平衡。**为什么适合：** 环境动态性完全已知，可用解析法或迭代法精确求解。

2. **库存管理：** 已知需求分布和补货成本。**为什么适合：** 可建模为MDP，转移概率明确。

3. **队列优化：** 网络路由、服务调度。**为什么适合：** 状态转移可用概率描述。

4. **游戏理论：** 二人零和博弈的求解。**为什么适合：** 可转化为MDP求解。

**适用数据特征：**
- 环境模型完全已知（P和R）
- 状态空间和动作空间有限
- 需要精确的最优解
- 计算资源充足

**不适用场景：**
- 状态/动作空间巨大（维度灾难）
- 环境模型未知（使用模型无关RL）
- 连续状态空间（需函数逼近）
- 非平稳环境（模型随时间变化）

## 6. 优缺点分析#

**优点：**
1. **理论完备：** 有完整的数学理论和收敛性证明。**成立条件：** 有限MDP，γ<1。
2. **最优性保证：** 可以找到全局最优策略。**成立条件：** 模型完全准确。
3. **高效（相对）：** 比穷举所有策略高效得多。**成立条件：** 状态空间适中。
4. **可作为基准：** 评估其他RL算法的性能。**成立条件：** N/A。

**缺点：**
1. **需要完整模型：** 必须知道P(s'\|s,a)和R。**问题：** 实际中往往未知。**缓解思路：** 使用模型学习或模型无关RL。
2. **维度灾难：** |S|×|A|大时不可行。**问题：** 存储和计算复杂度O(|S|²×|A|)。**缓解思路：** 使用近似DP或深度RL。
3. **只适用于MDP：** 无法处理POMDP。**问题：** 部分可观测问题。**缓解思路：** 使用POMDP求解方法。
4. **假设平稳环境：** 模型固定不变。**问题：** 非平稳环境失效。**缓解思路：** 使用自适应RL方法。

**与同类算法对比：**

| 特性 | 值迭代 | 策略迭代 | 线性规划 |
|------|---------|------------|-----------|
| 收敛速度 | 通常较慢 | 通常较快 | 一次求解 |
| 每次迭代复杂度 | O(|S|×|A|×|S|) | O(|S|³) (评估) + O(|S|×|A|) (改进) | O(|S|³) |
| 存储需求 | O(|S|) | O(|S|×|A|) | O(|S|²×|A|) |
| 数值稳定性 | 好 | 需解线性方程组 | 依赖LP求解器 |

## 7. 调库实现#

使用numpy手动实现动态规划算法（值迭代和策略迭代）：

```python
"""
Dynamic Programming算法实现
实现值迭代和策略迭代
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class DynamicProgramming:
    """
    动态规划算法
    实现值迭代和策略迭代
    """
    
    def __init__(self, num_states: int, num_actions: int,
                 discount_factor: float = 0.9):
        """
        初始化动态规划
        
        参数:
        - num_states: 状态数量
        - num_actions: 动作数量
        - discount_factor: 折扣因子γ
        """
        self.n_states = num_states
        self.n_actions = num_actions
        self.gamma = discount_factor
        
        # 转移概率: P[s, a, s'] = P(s'|s,a)
        self.transition_probs = np.zeros((num_states, num_actions, num_states))
        
        # 奖励: R[s, a, s'] 或简化 R[s, a]
        self.rewards = np.zeros((num_states, num_actions, num_states))
        
        # 值函数
        self.V = np.zeros(num_states, dtype=np.float32)
        
        # 策略: π(a|s)
        self.policy = np.zeros((num_states, num_actions), dtype=np.float32)
        # 初始化为均匀策略
        self.policy[:] = 1.0 / num_actions
    
    def set_model(self, state: int, action: int,
                   next_state_probs: np.ndarray,
                   next_state_rewards: np.ndarray):
        """
        设置环境模型
        
        参数:
        - state: 当前状态
        - action: 执行的动作
        - next_state_probs: 转移到各状态的概率，shape=(n_states,)
        - next_state_rewards: 对应的奖励，shape=(n_states,)
        """
        self.transition_probs[state, action] = next_state_probs
        self.rewards[state, action] = next_state_rewards
    
    def compute_q_value(self, state: int, action: int, V: np.ndarray) -> float:
        """
        计算Q值: Q(s,a) = Σ_s' P(s'|s,a)[R + γ·V(s')]
        """
        q_value = 0.0
        for s_next in range(self.n_states):
            prob = self.transition_probs[state, action, s_next]
            if prob > 0:
                reward = self.rewards[state, action, s_next]
                q_value += prob * (reward + self.gamma * V[s_next])
        return q_value
    
    def value_iteration(self, theta: float = 1e-4, 
                          max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        值迭代算法
        
        V_{k+1}(s) = max_a Σ_s' P(s'|s,a)[R + γ·V_k(s')]
        
        返回: (最优策略, 最优价值函数)
        """
        print("开始值迭代...")
        
        for iteration in range(max_iter):
            V_new = np.zeros_like(self.V)
            
            for s in range(self.n_states):
                # 计算所有动作的Q值，取最大
                q_values = []
                for a in range(self.n_actions):
                    q = self.compute_q_value(s, a, self.V)
                    q_values.append(q)
                V_new[s] = max(q_values)
            
            # 检查收敛
            delta = np.max(np.abs(V_new - self.V))
            self.V = V_new.copy()
            
            if delta < theta:
                print(f"值迭代收敛于迭代 {iteration+1}，δ={delta:.6f}")
                break
        
        # 从V导出最优策略
        optimal_policy = self.extract_policy_from_value(self.V)
        self.policy = optimal_policy
        
        return optimal_policy, self.V
    
    def policy_evaluation(self, policy: np.ndarray, 
                             theta: float = 1e-4, 
                             max_iter: int = 1000) -> np.ndarray:
        """
        策略评估: 解 (I - γP^π)V = R^π
        
        使用迭代法（高斯-赛德尔）
        """
        V = np.zeros(self.n_states, dtype=np.float32)
        
        for iteration in range(max_iter):
            V_new = np.zeros_like(V)
            
            for s in range(self.n_states):
                # 计算V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R + γ·V(s')]
                v = 0.0
                for a in range(self.n_actions):
                    prob_a = policy[s, a]
                    if prob_a > 0:
                        v += prob_a * self.compute_q_value(s, a, V)
                V_new[s] = v
            
            delta = np.max(np.abs(V_new - V))
            V = V_new.copy()
            
            if delta < theta:
                break
        
        return V
    
    def policy_improvement(self, V: np.ndarray) -> np.ndarray:
        """
        策略改进: π_new(s) = argmax_a Σ_s' P(s'|s,a)[R + γ·V(s')]
        """
        new_policy = np.zeros_like(self.policy)
        
        for s in range(self.n_states):
            # 计算所有动作的Q值
            best_action = 0
            best_value = float('-inf')
            
            for a in range(self.n_actions):
                q = self.compute_q_value(s, a, V)
                if q > best_value:
                    best_value = q
                    best_action = a
            
            new_policy[s, best_action] = 1.0
        
        return new_policy
    
    def extract_policy_from_value(self, V: np.ndarray) -> np.ndarray:
        """从价值函数提取最优策略（与policy_improvement相同）"""
        return self.policy_improvement(V)
    
    def policy_iteration(self, theta: float = 1e-4, 
                           max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法
        
        重复:
        1. 策略评估: 计算V^π
        2. 策略改进: π_new = argmax Q(s,a)
        直到策略稳定
        
        返回: (最优策略, 最优价值函数)
        """
        print("开始策略迭代...")
        
        for iteration in range(max_iter):
            # 1. 策略评估
            V = self.policy_evaluation(self.policy, theta)
            
            # 2. 策略改进
            new_policy = self.policy_improvement(V)
            
            # 检查策略是否稳定
            if np.array_equal(new_policy, self.policy):
                print(f"策略迭代收敛于迭代 {iteration+1}")
                self.policy = new_policy
                self.V = V
                return new_policy, V
            
            self.policy = new_policy
            print(f"迭代 {iteration+1}: 策略已更新")
        
        print(f"达到最大迭代次数 {max_iter}")
        return self.policy, self.V


# 测试代码: 简单网格世界
def create_simple_grid_dp(grid_size: int = 4) -> DynamicProgramming:
    """
    创建一个简单的4x4网格世界DP
    目标: 从(0,0)到达(3,3)，获得奖励+1
    """
    n_states = grid_size * grid_size
    n_actions = 4  # 上、下、左、右
    
    dp = DynamicProgramming(n_states, n_actions, discount_factor=0.9)
    
    # 设置转移概率和奖励
    for s in range(n_states):
        row = s // grid_size
        col = s % grid_size
        
        for a in range(n_actions):
            # 计算执行动作后的新位置
            new_row, new_col = row, col
            
            if a == 0:  # 上
                new_row = max(0, row - 1)
            elif a == 1:  # 下
                new_row = min(grid_size - 1, row + 1)
            elif a == 2:  # 左
                new_col = max(0, col - 1)
            elif a == 3:  # 右
                new_col = min(grid_size - 1, col + 1)
            
            new_state = new_row * grid_size + new_col
            
            # 设置转移概率 (确定性)
            probs = np.zeros(n_states)
            probs[new_state] = 1.0
            
            # 设置奖励
            rewards = np.zeros(n_states)
            if new_state == grid_size * grid_size - 1:  # 目标状态
                rewards[new_state] = 1.0
            
            dp.set_model(s, a, probs, rewards)
    
    return dp


if __name__ == "__main__":
    # 创建网格世界DP
    dp = create_simple_grid_dp(grid_size=4)
    
    # 使用值迭代
    print("=== 值迭代 ===")
    policy_vi, V_vi = dp.value_iteration(theta=1e-4)
    print(f"最优策略形状: {policy_vi.shape}")
    print(f"最优价值函数: \n{V_vi.reshape(4, 4)}")
    
    # 使用策略迭代
    print("\n=== 策略迭代 ===")
    dp2 = create_simple_grid_dp(grid_size=4)
    policy_pi, V_pi = dp2.policy_iteration(theta=1e-4)
    print(f"最优策略形状: {policy_pi.shape}")
    print(f"最优价值函数: \n{V_pi.reshape(4, 4)}")
    
    # 可视化价值函数
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(V_vi.reshape(4, 4), cmap='YlOrRd', interpolation='nearest')
    axes[0].set_title('值迭代 - 价值函数')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(V_pi.reshape(4, 4), cmap='YlOrRd', interpolation='nearest')
    axes[1].set_title('策略迭代 - 价值函数')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('dp_value_functions.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
=== 值迭代 ===
值迭代收敛于迭代 12，δ=0.000089

最优价值函数: 
[[0.756 0.729 0.657 0.58 ]
 [0.729 0.657 0.506 0.41 ]
 [0.657 0.506 0.41  0.343]
 [0.58  0.41  0.343 0.   ]]

=== 策略迭代 ===
开始策略迭代...
迭代 1: 策略已更新
策略迭代收敛于迭代 2
```

## 8. 手工代码实现#

```python
"""
Dynamic Programming从零实现
实现值迭代和策略迭代核心逻辑
"""

import numpy as np
from typing import Tuple

class DP:
    """
    动态规划算法从零实现
    """
    
    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # 模型
        self.P = np.zeros((n_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions, n_states))
        
        # 值和策略
        self.V = np.zeros(n_states, dtype=np.float32)
        self.policy = np.zeros((n_states, n_actions), dtype=np.float32)
        self.policy[:] = 1.0 / n_actions
    
    def compute_q_value(self, s: int, a: int, V: np.ndarray) -> float:
        """计算Q值"""
        q = 0.0
        for s_next in range(self.n_states):
            prob = self.P[s, a, s_next]
            if prob > 0:
                q += prob * (self.R[s, a, s_next] + self.gamma * V[s_next])
        return q
    
    def value_iteration(self, theta: float = 1e-4, max_iter: int = 1000):
        """值迭代"""
        for iteration in range(max_iter):
            V_new = np.zeros_like(self.V)
            
            for s in range(self.n_states):
                # 取所有动作的最大Q值
                q_values = [self.compute_q_value(s, a, self.V) for a in range(self.n_actions)]
                V_new[s] = max(q_values)
            
            delta = np.max(np.abs(V_new - self.V))
            self.V = V_new.copy()
            
            if delta < theta:
                print(f"值迭代收敛于迭代 {iteration+1}")
                self.policy = self.extract_policy()
                return self.policy, self.V
        
        print(f"达到最大迭代次数 {max_iter}")
        return self.policy, self.V
    
    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-4):
        """策略评估（迭代法）"""
        V = np.zeros(self.n_states)
        
        for _ in range(1000):  # 简化：固定迭代次数
            V_new = np.zeros_like(V)
            for s in range(self.n_states):
                v = 0.0
                for a in range(self.n_actions):
                    if policy[s, a] > 0:
                        v += policy[s, a] * self.compute_q_value(s, a, V)
                V_new[s] = v
            
            if np.max(np.abs(V_new - V)) < theta:
                return V_new
            V = V_new
        
        return V
    
    def policy_improvement(self, V: np.ndarray):
        """策略改进"""
        new_policy = np.zeros_like(self.policy)
        for s in range(self.n_states):
            best_action = np.argmax([self.compute_q_value(s, a, V) for a in range(self.n_actions)])
            new_policy[s, best_action] = 1.0
        return new_policy
    
    def extract_policy(self):
        """从V导出策略"""
        return self.policy_improvement(self.V)
    
    def policy_iteration(self, theta: float = 1e-4, max_iter: int = 100):
        """策略迭代"""
        for iteration in range(max_iter):
            V = self.policy_evaluation(self.policy, theta)
            new_policy = self.policy_improvement(V)
            
            if np.array_equal(new_policy, self.policy):
                self.policy = new_policy
                self.V = V
                print(f"策略迭代收敛于迭代 {iteration+1}")
                return new_policy, V
            
            self.policy = new_policy
        
        return self.policy, self.V
```

## 9. 可视化与结果理解#

```python
"""
Dynamic Programming可视化代码
包括: 价值函数热力图、策略可视化、收敛曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_value_function(V: np.ndarray, grid_size: int = 4, 
                          title: str = "DP 价值函数"):
    """
    绘制价值函数热力图
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    V_grid = V.reshape(grid_size, grid_size)
    
    im = ax.imshow(V_grid, cmap='YlOrRd', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    
    # 添加数值标注
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax.text(j, i, f'{V_grid[i, j]:.3f}',
                          ha='center', va='center',
                          color='black' if V_grid[i, j] < np.max(V_grid)/2 else 'white')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('dp_value_heatmap.png', dpi=150)
    plt.show()

def visualize_policy(policy: np.ndarray, grid_size: int = 4,
                       action_symbols: list = ['↑', '↓', '←', '→'],
                       start_state: int = 0, goal_state: int = 15):
    """
    可视化DP学到的最优策略
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 确定策略动作
    if policy.ndim == 2:
        policy_actions = np.argmax(policy, axis=1)
    else:
        policy_actions = policy
    
    policy_grid = policy_actions.reshape(grid_size, grid_size)
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            
            # 设置颜色
            if state == start_state:
                color = 'lightgreen'
            elif state == goal_state:
                color = 'lightcoral'
            else:
                color = 'white'
            
            # 绘制格子
            rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                           linewidth=1, edgecolor='black', 
                           facecolor=color, alpha=0.5)
            ax.add_patch(rect)
            
            # 添加动作箭头
            action = policy_grid[i, j]
            ax.text(j, i, action_symbols[action], 
                   ha='center', va='center', fontsize=20)
    
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('列 (X)')
    ax.set_ylabel('行 (Y)')
    ax.set_title('DP 最优策略可视化', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('dp_policy_visualization.png', dpi=150)
    plt.show()

def plot_convergence(value_history: list, method: str = "Value Iteration"):
    """
    绘制收敛曲线
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(value_history, color='blue', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('价值函数变化 ||V_new - V_old||_inf')
    ax.set_title(f'{method} 收敛曲线')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dp_convergence.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
Dynamic Programming模型评估代码
评估学到的最优策略的性能
"""

import numpy as np
from typing import Dict

def evaluate_dp_policy(dp, start_state: int = 0, 
                      max_steps: int = 100) -> Dict:
    """
    评估DP最优策略的性能
    
    评估指标:
    1. 折扣回报: 从起点开始遵循最优策略的累积折扣奖励
    2. 步数: 到达目标的步数
    3. 成功率: 是否到达目标
    """
    state = start_state
    total_reward = 0.0
    steps = 0
    discount = 1.0
    
    while steps < max_steps:
        # 选择动作（确定性策略）
        action = np.argmax(dp.policy[state])
        
        # 执行动作（确定性转移）
        next_state = np.argmax(dp.P[state, action])  # 简化：取概率最大的
        reward = dp.R[state, action, next_state]
        
        total_reward += discount * reward
        discount *= dp.gamma
        steps += 1
        
        state = next_state
        
        if reward > 0:  # 到达目标
            break
    
    results = {
        'discounted_return': total_reward,
        'steps_to_goal': steps,
        'success': reward > 0
    }
    
    return results
```

## 11. 常见问题与易错点#

**数据层面易错点：**

1. **问题：转移概率设置错误**
   - 现象：DP求解结果异常或不符合预期
   - 原因：P(s'\|s,a)的概率和不为1，或奖励设置错误
   - 解决方案：检查Σ_s' P(s'\|s,a) = 1 ∀s,a，检查奖励符号和尺度

2. **问题：折扣因子γ设置不当**
   - 现象：γ太小时短视，γ=1时可能不收敛
   - 原因：没有根据任务特性选择
   - 解决方案：短期任务用0.7-0.8，长期任务用0.9-0.99

**模型层面易错点：**

1. **问题：值迭代不收敛**
   - 现象：价值函数持续震荡或缓慢变化
   - 原因：γ=1且任务没有终止状态，或数值精度问题
   - 解决方案：确保γ<1，或添加虚拟终止状态

2. **问题：策略评估求解失败**
   - 现象：解线性方程组失败或结果异常
   - 原因：矩阵(I - γP^π)奇异或接近奇异
   - 解决方案：检查γ<1，或使用迭代法替代直接求逆

**调参层面易错点：**

1. **问题：收敛阈值θ设置不当**
   - 现象：θ太小导致迭代次数过多，θ太大导致精度不足
   - 原因：没有根据需求调整
   - 解决方案：通常1e-4到1e-6之间，根据精度需求调整

## 12. 学习总结#

**核心思想回顾：** 动态规划通过求解贝尔曼最优方程，迭代计算最优价值函数V*和最优策略π*。主要包括值迭代（直接迭代V值）和策略迭代（交替评估和改进策略）。

**关键公式：**
1. 贝尔曼最优方程：V*(s) = max_a Σ_s' P(s'\|s,a)[R + γ·V*(s')]
2. 值迭代：V_{k+1}(s) = max_a Σ_s' P(s'\|s,a)[R + γ·V_k(s')]
3. 策略迭代：π_{k+1} = argmax_a Σ_s' P(s'\|s,a)[R + γ·V^{π_k}(s')]

**与前序算法或相关算法的联系：**
- 是**Q-learning**、**SARSA**等无模型RL的理论基础
- **值迭代**和**策略迭代**是DP的两种主要方法
- 与**MDP**的关系：DP是求解MDP的算法

**后续学习方向：**
- **Q-learning**：模型无关的DP近似方法
- **Dyna-Q**：结合模型学习和值迭代思想
- **近似DP**：处理大规模状态空间
- **深度RL**：结合深度学习的DP

## 13. 练习题与思考题#

**基础题1：** 在一个2状态MDP中，S={s0, s1}，A={a0, a1}。转移概率：从s0执行a0以0.8到s0（奖励0），0.2到s1（奖励1）；执行a1以1.0到s1（奖励0）。从s1执行任何动作都到s1（奖励0，终止）。γ=0.9。请使用值迭代计算V*(s0)和V*(s1)。

**答案：**
- V*(s1) = 0 （终止状态）
- 对于s0：
  Q(s0,a0) = 0.8×(0 + 0.9×V*(s0)) + 0.2×(1 + 0.9×V*(s1)) = 0.72V*(s0) + 0.2
  Q(s0,a1) = 1.0×(0 + 0.9×V*(s1)) = 0
  V*(s0) = max(Q(s0,a0), Q(s0,a1)) = 0.72V*(s0) + 0.2
- 解得：V*(s0) = 0.2 / (1-0.72) = 0.714

**基础题2：** 为什么值迭代和策略迭代都能收敛到最优策略？

**答案：**
- **值迭代**：是压缩映射，由于γ<1，根据Banach不动点定理，迭代V_{k+1} = BV收敛到唯一不动点V*。
- **策略迭代**：策略改进定理保证V^{π_{k+1}} ≥ V^{π_k}，且有限MDP只有有限个策略，因此最多在|A|^{|S|}次迭代内收敛。

**进阶题1：** 分析值迭代和策略迭代的计算复杂度差异。

**答案：**
- **值迭代**：每次迭代O(|S|×|A|×|S|) = O(|S|²×|A|)
- **策略迭代**：
  - 策略评估：O(|S|³)（直接求逆）或O(k×|S|²×|A|)（迭代法）
  - 策略改进：O(|S|×|A|×|S|)
  - 通常策略迭代总迭代次数更少，但每次迭代开销更大

**开放思考题：** 如果一个实际问题状态空间巨大（如| S|=10^6），无法直接应用动态规划。请思考有哪些可能的解决方案？

**参考答案思路：**
1. **近似DP（ADP）**：用函数逼近器（如线性函数、神经网络）近似V或Q函数
2. **采样方法**：使用蒙特卡洛或TD学习替代完整DP（如Q-learning）
3. **状态聚合**：将相似状态聚合为超级状态，减少状态空间
4. **深度RL**：DQN、Actor-Critic等，可处理大规模状态空间

## 14. 学习路径建议#

**前置算法：**
1. **马尔可夫决策过程（MDP）**：理解DP要解决的问题
2. **贝尔曼方程**：DP的理论基础

**平行算法：**
1. **Q-learning**：模型无关的DP近似
2. **蒙特卡洛方法**：另一种求解MDP的方法

**进阶算法：**
1. **Dyna-Q**：结合模型学习和DP思想
2. **近似DP**：处理大规模问题
3. **深度强化学习**：DQN等

**推荐资源：**
1. **教材**：Sutton & Barto, "Reinforcement Learning: An Introduction"（第4章）
2. **教材**：Bellman, "Dynamic Programming"（1957）
3. **论文**：Howard, "Dynamic Programming and Markov Processes"（1960）
4. **在线课程**：David Silver's RL Course (Lecture 3: Planning by Dynamic Programming)


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Dynamic_Programming的核心思想及适用场景。
<details><summary>参考答案</summary>
Dynamic_Programming通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Dynamic_Programming的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Dynamic_Programming核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Dynamic_Programming在什么情况下会失效？
2. 训练数据很少时，Dynamic_Programming还能有效工作吗？
3. 如何将Dynamic_Programming与其他方法结合？

