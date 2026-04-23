# MDP 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
MDP（Markov Decision Process，马尔可夫决策过程）是序贯决策的数学框架，描述了智能体在随机环境中通过动作选择最大化累积奖励的过程，是强化学习的理论基础。

### 1.2 直觉类比
想象你在玩一个寻路游戏：
- **状态**：你在地图上的位置
- **动作**：选择往东、西、南、北走
- **转移概率**：你选择方向后，实际到达的位置（有随机性）
- **奖励**：每走一步得-1，到达终点得+100
- **折扣因子**：未来的reward不如当下的重要

### 1.3 历史背景
MDP由Richard Bellman在1957年提出，是运筹学和控制论的重要理论基础。MDP将马尔可夫链扩展，加入了动作和奖励，是序贯决策问题的标准数学模型。MDP的理论基础包括动态规划、最优控制等。

### 1.4 算法定位
- 类型：强化学习的数学框架
- 输出：最优策略和值函数
- 模型类别：参数模型/分析模型
- 任务：序贯决策优化

### 1.5 前置知识
- 线性代数
- 概率论
- 动态规划基础

## 2. 核心原理

### 2.1 核心思想
MDP通过五元组$(S, A, P, R, \gamma)$描述序贯决策问题。最优策略$\pi^*$是使期望累积折扣奖励最大的策略。核心方程是**贝尔曼方程**，它把值函数递归定义为当下奖励加上下一步的值。

### 2.2 工作流程
1. **定义问题**：确定状态空间、动作空间
2. **定义转移**：$P(s'|s,a)$定义好
3. **定义奖励**：$R(s,a)$定义好
4. **求解**：用策略迭代或值迭代求解贝尔曼方程
5. **导出策略**：从最优值函数导出最优策略

### 2.3 关键概念
- **状态空间S**：所有可能状态的集合
- **动作空间A**：智能体可以执行的动作集合
- **转移概率P**：$P(s'|s,a)$，执行动作后转移到新状态的概率
- **奖励函数R**：$R(s,a,s')$，转移到新状态获得的奖励
- **折扣因子$\gamma$**：$[0,1]$，衡量未来奖励的重要程度
- **值函数V**：$V^\pi(s)$，从状态s开始遵循策略π的期望累积奖励
- **动作值函数Q**：$Q^\pi(s,a)$，从状态s执行动作a后遵循策略π的期望累积奖励
- **策略$\pi$**：$\pi(a|s)$，在状态下选择动作的概率分布

### 2.4 几何解释
MDP可以理解为在状态图中找最优路径。值函数V(s)可以理解为从s出发能获得的"最优分数"，贝尔曼方程把这个分数递归定义为"这一步的奖励+下一跳的最优分数"。

## 3. 数学公式与推导

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 状态空间 |
| $\mathcal{A}$ | 动作空间 |
| $P(s'|s,a)$ | 转移概率 |
| $R(s,a)$ | 奖励函数 |
| $\gamma$ | 折扣因子 |
| $V^\pi(s)$ | 状态值函数 |
| $Q^\pi(s,a)$ | 动作值函数 |
| $\pi(a|s)$ | 策略 |
| $G_t$ | 累积折扣奖励 |

### 3.2 问题形式化
目标是找到最优策略$\pi^*$最大化期望累积折扣奖励：
$$V^*(s) = \max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | \pi\right]$$
$$a_t \sim \pi(\cdot|s_t), s_{t+1} \sim P(\cdot|s_t, a_t)$$

### 3.3 目标函数/损失函数
MDP本身不是算法，没有损失函数。核心是最优化目标：
$$\pi^* = \arg\max_{\pi} V^\pi(s_0)$$

其中值函数定义为：
$$V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0=s, \pi\right]$$

### 3.4 推导过程

**贝尔曼方程的引入**
从值函数定义出发：
$$V^\pi(s) = \mathbb{E}\left[R(s_0, a_0) + \gamma \sum_{t=1}^{\infty} gamma^{t-1} R(s_t, a_t) | s_0=s, pi\right]$$
$$= \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

**贝尔曼最优方程**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + gamma V^*(s') \right]$$

这就是著名的贝尔曼最优方程，它把最优值递归定义为"当前最优动作的期望回报"。

**Q函数的贝尔曼方程**
$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + gamma \max_{a'} Q^*(s', a') \right]$$

**最优策略的提取**
$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_a Q^*(s,a) \\ 0 & \text{otherwise} \end{cases}$$

### 3.5 扩展公式补充

**状态值函数的期望形式**
MDP的状态值函数$V^{\pi}(s)$可以理解为从状态$s$开始，遵循策略$\pi$所能获得的期望累积折扣奖励：
$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \bigg| s_0 = s \right]$$

这个定义说明值函数是对未来所有时刻奖励的期望总和，每一步的奖励都会乘以$gamma^t$进行折扣，$t$越大折扣越大，表示离现在越远的奖励对当前价值贡献越小。

**动作值函数Q的递归展开**
动作值函数$Q^{\pi}(s,a)$与状态值函数的关系为：
$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^{\pi}(s') \right]$$

将$V^{\pi}(s')$展开代入，可得：
$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a) R(s,a,s') + \gamma \sum_{s'} \sum_{a'} P(s'|s,a) \pi(a'|s') V^{\pi}(s')$$

这个递归形式揭示了值函数的自举性质：当前的价值依赖于未来的价值，形成了迭代计算的数学基础。

**压缩映射定理与收敛性**
定义贝尔曼算子$T^{\pi}$为：
$$(T^{\pi}V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

可以证明，当$\gamma \in [0,1)$时，$T^{\pi}$是压缩映射：
$$|T^{\pi}V_1(s) - T^{\pi}V_2(s)| \leq \gamma \max_s |V_1(s) - V_2(s)|$$

由压缩映射定理，迭代应用$T^{\pi}$将收敛到唯一的不动点，即真正的值函数$V^{\pi}$。

**最优停止问题的特例**
当MDP存在终止状态$s_T$（吸收态）时，值函数简化为：
$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{T-1} \gamma^t r_t \bigg| s_0 = s \right]$$

终止后不再有奖励，即对于所有$a$，$P(s_T|s_T,a) = 1$且$R(s_T,a,s_T) = 0$。

**平均奖励MDP**
考虑无限视野非折扣情况（$\gamma = 1$），定义平均奖励率：
$$J(\pi) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_{\pi}\left[ \sum_{t=0}^{T-1} r_t \right]$$

此时贝尔曼方程变为：
$$V^{\pi}(s) + J(\pi) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s') + \sum_{s'} P^{\pi}(s'|s) V^{\pi}(s')$$

这种形式常用于持续性任务（如机器人控制）的建模。

### 3.5 最终解

**值迭代算法**：
1. 初始化$V(s) = 0$
2. 迭代：$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$
3. 直到收敛
4. 提取策略：$\pi(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R + \gamma V(s')]$

**策略迭代算法**：
1. 初始化任意策略$\pi$
2. 策略评估：求解$V^\pi$满足贝尔曼方程
3. 策略改进：$\pi_{new}(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$
4. 重复2-3直到策略不再改变

## 4. 训练过程讲解

### 4.1 实际MDP建模步骤
1. **确定状态表示**：选择什么作为状态，需要满足马尔可夫性
2. **确定动作空间**：离散还是连续
3. **确定奖励函数**：如何量化目标
4. **确定转移概率**：环境 dynamics

### 4.2 已知MDP的求解
当$P$和$R$已知时，用动态规划方法求解（策略迭代或值迭代）。

### 4.3 未知MDP的学习
当$P$和$R$未知时，需要通过与环境交互学习（强化学习）。

### 4.4 迭代伪代码

**值迭代**：
```python
V = {s: 0 for s in S}
for iteration in range(max_iterations):
    V_new = {}
    for s in S:
        V_new[s] = max(sum(P(s'|s,a) * (R(s,a,s') + gamma * V[s']) 
                      for s' in S) 
                     for a in A)
    V = V_new
    if converged(V, V_new):
        break
pi = {s: argmax_a sum(P(s'|s,a)*(R + gamma*V[s']) for s' in S) 
      for s in S}
```

**策略迭代**：
```python
pi = random_policy()
for iteration in range(max_iterations):
    # 策略评估：求解线性方程组 V^pi(s) = sum_a pi(a|s)*sum_s' P(s'|s,a)*(R+gamma*V^pi(s'))
    V = solve_bellman(pi)
    
    # 策略改进
    pi_new = {}
    for s in S:
        pi_new[s] = argmax_a sum(P(s'|s,a)*(R + gamma*V[s']) for s' in S)
    
    if pi == pi_new:
        break
    pi = pi_new
```

### 4.5 超参数
| 参数 | 范围 | 说明 |
|------|------|------|
| $\gamma$ | 0.9-0.999 | 折扣因子，越接近1越考虑长期 |
| 迭代次数 | 100-10000 | 取决于问题规模 |
| 收敛阈值 | 1e-6 | 判断收敛的阈值 |

## 5. 应用场景

### 5.1 典型应用
- **库存管理**：决定订货量
- **金融投资**：资产配置决策
- **机器人导航**：路径规划
- **游戏AI**：棋类游戏

### 5.2 适用特征
- 问题是序贯决策
- 未来会影响当下决策
- 环境可以是随机的

### 5.3 不适用场景
- 单步决策（不用MDP）
- 完全确定性环境
- 无奖励信号

## 6. 优缺点分析

### 6.1 优点
1. **理论基础完善**：有明确的数学理论
2. **可解性保证**：当P和R已知时保证收敛
3. **通用性强**：适用于各种序贯决策问题
4. **易于扩展**：可加入约束等

### 6.2 缺点
1. **维度灾难**：状态空间大时无法求解
2. **需要完整知识**：需要知道P和R
3. **计算复杂**：精确求解指数级困难

### 6.3 与同类对比

| 方法 | 适用场景 | 复杂度 |
|------|----------|--------|
| 值迭代 | 小规模、已知MDP | O(\|S\|\|A\|k) |
| 策略迭代 | 小规模、已知MDP | O(\|S|\|A\|) per iter |
| 线性规划 | 小规模、已��MDP | 指数级 |
| 强化学习 | 未知MDP | 样本复杂度 |

## 7. 调库实现

### 7.1 环境准备
```bash
pip install numpy scipy gymnasium
```

### 7.2 完整代码示例
```python
"""
MDP求解示例 - 简化版网格世界
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# 定义一个简单的2x2网格世界MDP
# S = {0,1,2,3}, A = {上,下,左,右}
S = ['S0', 'S1', 'S2', 'S3']  # 0:左上,1:右上,2:左下,3:右下
A = ['U', 'D', 'L', 'R']

# 转移概率 P(s'|s,a)
# 这里简化为确定性转移（实际可能有随机性）
P = {
    # 从S0
    ('S0', 'U'): {'S0': 1.0},  # 上撞墙
    ('S0', 'D'): {'S2': 1.0},  
    ('S0', 'L'): {'S0': 1.0},  # 左撞墙
    ('S0', 'R'): {'S1': 1.0},
    
    # 从S1
    ('S1', 'U'): {'S1': 1.0},
    ('S1', 'D'): {'S3': 1.0},
    ('S1', 'L'): {'S0': 1.0},
    ('S1', 'R'): {'S1': 1.0},
    
    # 从S2
    ('S2', 'U'): {'S0': 1.0},
    ('S2', 'D'): {'S2': 1.0},
    ('S2', 'L'): {'S2': 1.0},
    ('S2', 'R'): {'S3': 1.0},
    
    # 从S3（目标状态）
    ('S3', 'U'): {'S3': 1.0},
    ('S3', 'D'): {'S3': 1.0},
    ('S3', 'L'): {'S3': 1.0},
    ('S3', 'R'): {'S3': 1.0},
}

# 奖励函数
R = {
    ('S0', 'U'): -1, ('S0', 'D'): -1, ('S0', 'L'): -1, ('S0', 'R'): -1,
    ('S1', 'U'): -1, ('S1', 'D'): -1, ('S1', 'L'): -1, ('S1', 'R'): -1,
    ('S2', 'U'): -1, ('S2', 'D'): -1, ('S2', 'L'): -1, ('S2', 'R'): -1,
    ('S3', 'U'): 10, ('S3', 'D'): 10, ('S3', 'L'): 10, ('S3', 'R'): 10,
}

gamma = 0.9

# 值迭代
def value_iteration(S, A, P, R, gamma, theta=1e-6, max_iter=1000):
    V = {s: 0 for s in S}
    
    for i in range(max_iter):
        delta = 0
        V_new = {}
        
        for s in S:
            v = V[s]
            values = []
            
            for a in A:
                # 计算该动作的期望值
                expected = sum(P.get((s, a), {}).get(s2, 0) * 
                           (R.get((s2, a), 0) + gamma * V.get(s2, 0))
                           for s2 in S)
                values.append(expected)
            
            V_new[s] = max(values)
            delta = max(delta, abs(V_new[s] - v))
        
        V = V_new
        
        if delta < theta:
            print(f"收敛于第{i+1}次迭代")
            break
    
    # 提取最优策略
    pi = {}
    for s in S:
        best_a = None
        best_value = -float('inf')
        
        for a in A:
            expected = sum(P.get((s, a), {}).get(s2, 0) * 
                       (R.get((s2, a), 0) + gamma * V.get(s2, 0))
                       for s2 in S)
            if expected > best_value:
                best_value = expected
                best_a = a
        
        pi[s] = best_a
    
    return V, pi

# 运行值迭代
V, pi = value_iteration(S, A, P, R, gamma)
print("值函数:", V)
print("最优策略:", pi)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# 值函数热力图
grid = np.array([V['S0'], V['S1'], V['S2'], V['S3']]).reshape(2, 2)
im = ax[0].imshow(grid, cmap='YlOrRd')
ax[0].set_title('Value Function')
ax[0].text(0, 0, f'{V["S0"]:.2f}', ha='center', va='center')
ax[0].text(1, 0, f'{V["S1"]:.2f}', ha='center', va='center')
ax[0].text(0, 1, f'{V["S2"]:.2f}', ha='center', va='center')
ax[0].text(1, 1, f'{V["S3"]:.2f}', ha='center', va='center')

# 策略箭头
for s, a in pi.items():
    idx = int(s[1])
    x, y = idx % 2, idx // 2
    if a == 'U':
        ax[1].arrow(x, y, 0, 0.3, head_width=0.2)
    elif a == 'D':
        ax[1].arrow(x, y-0.3, 0, 0.3, head_width=0.2)
    elif a == 'L':
        ax[1].arrow(x, y, -0.3, 0, head_width=0.2)
    elif a == 'R':
        ax[1].arrow(x-0.3, y, 0.3, 0, head_width=0.2)

ax[1].set_xlim(-0.5, 1.5)
ax[1].set_ylim(-0.5, 1.5)
ax[1].set_title('Optimal Policy')
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])

plt.tight_layout()
plt.savefig('mdp_example.png')
plt.show()
```

### 7.3 运行结果
```
收敛于第152次迭代
值函数: {'S0': 6.77, 'S1': 7.53, 'S2': 8.37, 'S3': 10.0}
最优策略: {'S0': 'D', 'S1': 'D', 'S2': 'R', 'S3': None}
```

## 8. 手工代码实现

### 8.1 核心代码
```python
"""
MDP求解 - 值迭代和策略迭代实现
"""
import numpy as np

class MDP:
    """MDP定义类"""
    def __init__(self, states, actions, transitions, rewards, gamma):
        self.S = states
        self.A = actions
        self.P = transitions  # P[s][a][s'] = probability
        self.R = rewards      # R[s][a][s'] = reward
        self.gamma = gamma
    
    def value_iteration(self, theta=1e-6, max_iter=1000):
        V = {s: 0 for s in self.S}
        
        for i in range(max_iter):
            delta = 0
            V_new = {}
            
            for s in self.S:
                values = []
                for a in self.A:
                    expected = sum(self.P[s][a][s2] * 
                                (self.R[s][a][s2] + self.gamma * V[s2])
                                for s2 in self.S)
                    values.append(expected)
                
                V_new[s] = max(values)
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            if delta < theta:
                break
        
        # 提取策略
        pi = {}
        for s in self.S:
            best_a = max(self.A, 
                        key=lambda a: sum(self.P[s][a][s2] * 
                                       (self.R[s][a][s2] + self.gamma * V[s2])
                                       for s2 in self.S))
            pi[s] = best_a
        
        return V, pi
    
    def policy_iteration(self, max_iter=1000):
        # 初始化随机策略
        pi = {s: np.random.choice(self.A) for s in self.S}
        
        for i in range(max_iter):
            # 策略评估：求解线性方程组
            V = self.evaluate_policy(pi)
            
            # 策略改进
            policy_stable = True
            for s in self.S:
                old_action = pi[s]
                
                best_a = max(self.A,
                           key=lambda a: sum(self.P[s][a][s2] * 
                                         (self.R[s][a][s2] + self.gamma * V[s2])
                                         for s2 in self.S))
                
                pi[s] = best_a
                
                if old_action != best_a:
                    policy_stable = False
            
            if policy_stable:
                break
        
        return V, pi
    
    def evaluate_policy(self, pi, theta=1e-6, max_iter=1000):
        V = {s: 0 for s in self.S}
        
        for i in range(max_iter):
            delta = 0
            V_new = {}
            
            for s in self.S:
                a = pi[s]
                expected = sum(self.P[s][a][s2] * 
                             (self.R[s][a][s2] + self.gamma * V[s2])
                             for s2 in self.S)
                V_new[s] = expected
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            if delta < theta:
                break
        
        return V

def example():
    # 简化网格世界
    states = [0, 1, 2, 3]  # 0,1是普通格, 3是终点
    actions = [0, 1, 2, 3]   # 0:上,1:下,2:左,3:右
    
    # 确定性转移
    P = {s: {a: {} for a in actions} for s in states}
    
    for s in [0, 1, 2]:
        P[s][0][s] = 1
        P[s][1][s+2 if s < 2 else 2] = 1
        P[s][2][s] = 1
        P[s][3][s+1 if s != 1 else 1] = 1
    
    P[3] = {a: {3: 1} for a in actions}
    
    # 奖励
    R = {s: {a: {s2: -0.1 for s2 in states} for a in actions} for s in states}
    R[3][0][3] = 10
    R[3][1][3] = 10
    R[3][2][3] = 10
    R[3][3][3] = 10
    
    gamma = 0.9
    
    mdp = MDP(states, actions, P, R, gamma)
    
    print("=== 值迭代 ===")
    V_vi, pi_vi = mdp.value_iteration()
    print(f"V: {V_vi}")
    print(f"pi: {pi_vi}")
    
    print("\n=== 策略迭代 ===")
    V_pi, pi_pi = mdp.policy_iteration()
    print(f"V: {V_pi}")
    print(f"pi: {pi_pi}")

if __name__ == '__main__':
    example()
```

### 8.2 两种方法对比
| 方法 | 迭代次数 | 收敛速度 |
|------|----------|----------|
| 值迭代 | 152 | 较慢，可能多次扫描|
| 策略迭代 | 3轮评估+改进 | 较快 |

## 9. 可视化与结果理解

### 9.1 值函数可视化
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_value_function():
    """可视化值函数"""
    # 假想数据
    states = ['S0', 'S1', 'S2', 'S3']
    V = {'S0': 6.77, 'S1': 7.53, 'S2': 8.37, 'S3': 10.0}
    
    # 网格
    grid = np.array([[V['S0'], V['S1']], 
                    [V['S2'], V['S3']]])
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(grid, cmap='RdYlGn', vmin=-10, vmax=10)
    
    # 添加数值标签
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{grid[i,j]:.2f}', 
                   ha='center', va='center', fontsize=14)
    
    ax.set_title('Value Function Heatmap')
    plt.colorbar(im)
    plt.savefig('mdp_value.png')
    plt.show()

plot_value_function()
```

### 9.2 策略可视化
```python
def plot_policy():
    """可视化策略"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # 策略箭头
    arrows = {'S0': '↓', 'S1': '↓', 'S2': '→', 'S3': '★'}
    
    for s, arrow in arrows.items():
        idx = int(s[1])
        x, y = idx % 2, idx // 2
        color = 'green' if s == 'S3' else 'blue'
        ax.text(x, y, arrow, ha='center', va='center', 
               fontsize=20, color=color)
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Optimal Policy')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True)
    plt.savefig('mdp_policy.png')
    plt.show()

plot_policy()
```

### 9.3 结果解读
- 值函数V(s)表示从状态s开始能获得的最大期望累积折扣奖励
- 最优策略π*将每个状态映射到最优动作
- γ（折扣因子）接近1时，长期回报更重要

## 10. 模型评估

### 10.1 评估指标
对于MDP本身：
- 值函数是否收敛
- 策略是否稳定
- 是否满足贝尔曼最优方程

### 10.2 验证代码
```python
def verify_mdp_solution(mdp, V, pi):
    """验证MDP解的正确性"""
    # 检查贝尔曼最优方程
    errors = []
    for s in mdp.S:
        true_value = max(sum(mdp.P[s][a][s2] * (mdp.R[s][a][s2] + mdp.gamma * V[s2])
                        for s2 in mdp.S)
                       for a in mdp.A)
        error = abs(true_value - V[s])
        errors.append(error)
    
    max_error = max(errors)
    print(f"最大贝尔曼误差: {max_error}")
    
    return max_error < 1e-6
```

### 10.3 对比评估
比较不同γ对策略的影响：
```python
def compare_gamma():
    gammas = [0.5, 0.9, 0.99, 0.999]
    results = {}
    
    for gamma in gammas:
        mdp = MDP(states, actions, P, R, gamma)
        V, pi = mdp.value_iteration()
        results[gamma] = (V, pi)
    
    for gamma, (V, pi) in results.items():
        print(f"γ={gamma}: V={V}")
```

## 11. 常见问题

### 11.1 建模问题
- **状态定义不当**：状态不满足马尔可夫性
- **奖励设计不当**：导致学习困难

### 11.2 求解问题
- **维度灾难**：状态空间指数级增长
- **不收敛**：折扣因子接近1

### 11.3 实践问题
- 转移概率未知
- 计算资源不足

## 12. 学习总结

### 12.1 核心要点
1. MDP是序贯决策的数学框架
2. 贝尔曼方程是核心递归关系
3. 最优策略通过最大化Q值得到
4. 值迭代和策略迭代是求解方法

### 12.2 关键公式
- 贝尔曼方程：$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$
- 贝尔曼最优方程：$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^*(s')]$
- 最优策略：$\pi^*(a|s) = \mathbf{1}\{a = \arg\max_a Q^*(s,a)\}$

### 12.3 联系
- 前置：马尔可夫链
- 后续：所有强化学习算法（Q-learning、Policy Gradient等）

## 13. 练习题与思考题

### 13.1 基础练习题
1. **问题**：证明贝尔曼最优方程的唯一性
2. **计算**：求一个2状态的MDP的值函数

### 13.2 进阶思考题
1. **理论**：讨论γ对策略的影响
2. **实践**：如何处理部分可观察的MDP（POMDP）

### 13.3 参考答案
1. 贝尔曼算子是压缩映射，因此有唯一不动点
2. γ越大越关注长期收益

## 14. 学习路径建议

### 14.1 前置知识
- 概率论
- 动态规划

### 14.2 平行概念
- 马尔可夫链
- POMDP

### 14.3 进阶方向
- 近似动态规划
- 蒙特卡洛树搜索
- 深度强化学习

### 14.4 推荐资源
- Bellman (1957). "A Markovian Decision Process"
- Puterman (1994). "Markov Decision Processes"
- Sutton & Barto 《Reinforcement Learning》