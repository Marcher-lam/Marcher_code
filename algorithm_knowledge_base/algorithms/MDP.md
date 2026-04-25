# MDP 学习文档

> MDP (Markov Decision Process) 马尔可夫决策过程是序贯决策的数学框架,为强化学习提供理论基础,是理解智能决策行为的基石。

---

## 1. 算法基础认知

### 一句话定义
MDP是一种通过序贯决策最大化长期累积奖励的数学框架,核心包含状态、动作、转移概率和奖励。

### 直觉类比
**迷宫寻宝**:
- **状态**:你在哪里(房间位置)
- **动作**:往哪个方向走
- **转移**:走后会到哪个房间
- **奖励**:找到宝藏+100,碰到妖怪-100

智能体需要学会在不同状态做最优决策!

### 历史背景
- 1950s:Bellman提出动态规划
- 1960s:Puterman等发展MDP理论
- 现代:强化学习的数学基础

### 算法定位
- **类型**:序贯决策/数学框架
- **输出**:最优策略/值函数
- **模型类型**:随机过程+控制

### 前置知识
- 概率论基础
- 随机过程
- 动态规划

---

## 2. 核心原理

### 2.1 核心要素
MDP五元组: $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$

1. **状态空间** $\mathcal{S}$: 所有可能状态
2. **动作空间** $\mathcal{A}$: 智能体可采取的动作
3. **转移概率** $P(s'|s,a)$: 在状态s执行动作a转到s'的概率
4. **奖励函数** $R(s,a,s')$: 收到的奖励
5. **折扣因子** $\gamma$: 未来奖励的衰减

### 2.2 工作流程
```
环境
    ↓
状态 s ∈ S
    ↓
智能体 → 动作 a ∈ A(s)
    ↓
环境 → 转移 s' ~ P(·|s,a)
      奖励 r = R(s,a,s')
    ↓
重复直到终止
```

### 2.3 关键概念
- **马尔可夫性**: $P(s_{t+1}|s_t,a_t) = P(s_{t+1}|s_t,a_t,...,s_0)$
- **策略** $\pi(a|s)$: 在状态s选择动作a的概率
- **值函数** $V^\pi(s)$: 从状态s开始的期望累积奖励
- **动作值函数** $Q^\pi(s,a)$: 从状态s执行动作a后的期望累积奖励

### 2.4 架构图
```
┌─────────────────────────────────────────────┐
│              MDP框架                          │
│                                             │
│         ┌─────────┐                         │
│    s ─→│   智能体  │──→ a              │
│         └─────────┘    │                  │
│              ↑         │                  │
│              │         ↓                  │
│         ┌──────────────┐               │
│         │  环境/转移   │                │
│         │ P(s'|s,a)   │                │
│         └──────────────┘                │
│              ↑    r                    │
│              └─────────────────        │
│                                             │
│   目标: 最大化 Σ γ^t r_t                 │
└─────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 状态空间 |
| $\mathcal{A}$ | 动作空间 |
| $P(s'\|s,a)$ | 转移概率 |
| $R(s,a,s')$ | 奖励 |
| $\gamma$ | 折扣因子 |
| $\pi(a\|s)$ | 策略 |
| $V^\pi(s)$ | 状态值函数 |
| $Q^\pi(s,a)$ | 动作值函数 |

### 3.2 值函数定义

**状态值函数**:
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \big| s_0 = s\right]$$

**动作值函数**:
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \big| s_0 = s, a_0 = a\right]$$

### 3.3 贝尔曼方程

**值函数的贝尔曼方程**:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**动作值函数**:
$$Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s',a')]$$

### 3.4 最优值函数

定义最优值函数:
$$V^*(s) = \max_\pi V^\pi(s)$$
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

**贝尔曼最优方程**:
$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a)[R(s,a,s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s',a')]$$

### 3.5 最优策略

若 $Q^*(s,a) = \max_{a'} Q^*(s,a')$, 则最优策略:
$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s,a') \\ 0 & \text{otherwise} \end{cases}$$

---

## 4. 求解方法

### 4.1 值迭代

```python
"""
值迭代算法
"""

import numpy as np
from collections import defaultdict

class MDPValueIteration:
    """MDP值迭代求解"""
    
    def __init__(self, states, actions, transitions, rewards, gamma, theta=1e-6):
        self.S = states
        self.A = actions
        self.P = transitions  # P[s][a] = [(s', p, r), ...]
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0.0 for s in states}
    
    def value_iteration(self, max_iter=1000):
        """值迭代"""
        for _ in range(max_iter):
            delta = 0
            
            for s in self.S:
                v = self.V[s]
                
                # 计算所有动作的最大Q值
                max_q = -float('inf')
                
                for a in self.A:
                    q_value = 0
                    
                    for s_next, prob, r in self.P[s][a]:
                        q_value += prob * (r + self.gamma * self.V[s_next])
                    
                    max_q = max(max_q, q_value)
                
                self.V[s] = max_q
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
        
        return self.V
    
    def get_policy(self):
        """提取最优策略"""
        policy = {}
        
        for s in self.S:
            best_action = None
            best_value = -float('inf')
            
            for a in self.A:
                q_value = 0
                for s_next, prob, r in self.P[s][a]:
                    q_value += prob * (r + self.gamma * self.V[s_next])
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            
            policy[s] = best_action
        
        return policy


class MDPPolicyIteration:
    """策略迭代"""
    
    def __init__(self, states, actions, transitions, rewards, gamma, theta=1e-6):
        self.S = states
        self.A = actions
        self.P = transitions
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0.0 for s in states}
        self.policy = {s: list(actions)[0] for s in states}  # 随机初始化
    
    def policy_evaluation(self, max_iter=1000):
        """策略评估"""
        for _ in range(max_iter):
            delta = 0
            
            for s in self.S:
                v = self.V[s]
                
                a = self.policy[s]
                q_value = 0
                
                for s_next, prob, r in self.P[s][a]:
                    q_value += prob * (r + self.gamma * self.V[s_next])
                
                self.V[s] = q_value
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """策略提升"""
        policy_stable = True
        
        for s in self.S:
            old_action = self.policy[s]
            
            # 找到最优动作
            best_action = old_action
            best_value = -float('inf')
            
            for a in self.A:
                q_value = 0
                for s_next, prob, r in self.P[s][a]:
                    q_value += prob * (r + self.gamma * self.V[s_next])
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            
            self.policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def solve(self, max_iter=1000):
        """求解"""
        for _ in range(max_iter):
            self.policy_evaluation()
            
            if self.policy_improvement():
                break
        
        return self.V, self.policy
```

### 4.2 线性规划方法

将MDP转化为线性规划:
$$\min_{V} \sum_s V(s)$$

s.t. $$V(s) \geq \max_{a} \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

### 4.3 Q学习(无模型)

```python
"""
Q学习(无模型MDP求解)
"""

class QLearning:
    """Q学习"""
    
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.S = states
        self.A = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Q表
        self.Q = defaultdict(lambda: defaultdict(float))
    
    def select_action(self, s):
        """ε-greedy选择"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.A)
        
        Qs = {a: self.Q[s][a] for a in self.A}
        return max(Qs, key=Qs.get)
    
    def update(self, s, a, r, s_next):
        """Q更新"""
        max_q = max(self.Q[s_next].values(), default=0)
        
        # TD更新
        td_error = r + self.gamma * max_q - self.Q[s][a]
        self.Q[s][a] += self.alpha * td_error
    
    def learn(self, env, n_episodes=1000):
        """学习"""
        for _ in range(n_episodes):
            s = env.reset()
            done = False
            
            while not done:
                a = self.select_action(s)
                s_next, r, done, _ = env.step(a)
                self.update(s, a, r, s_next)
                s = s_next
```

---

## 5. 扩展与变体

### 5.1 部分可观MDP (POMDP)
状态不可完全观测,需要 belief state:

$$b(s) = P(s|o_{1:t}, a_{1:t-1})$$

### 5.2 连续状态/动作MDP
- 离散化近似
- 函数近似(值函数/策略)

### 5.3 平均奖励MDP
$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} r_t$$

### 5.4 Dec-POMDP
多智能体非合作MDP。

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 理论基础 | 数学严格 |
| 可解性 | 动态规划 |
| 通用性 | 各种决策问题 |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 维度爆炸 | 近似方法 |
| 需要模型 | 无模型学习 |
| 假设过强 | 放松假设 |

---

## 7. 应用场景

### 7.1 典型应用
- **机器人控制**: 移动机器人导航
- **游戏AI**: 棋类、游戏
- **资源调度**: 电网、物流
- **推荐系统**: 序列推荐

### 7.2 适用问���
- 序贯决策
- 有延迟奖励
- 环境可建模

---

## 8. 手工实现

```python
"""
MDP 核心简化版
"""

import numpy as np

class SimpleMDP:
    """简化MDP"""
    
    def __init__(self, states, actions, gamma=0.9):
        self.S = states
        self.A = actions
        self.gamma = gamma
        
        # P[s][a] = {s_next: (prob, reward)}
        self.P = {s: {a: {} for s in states, a in actions}
        self.V = {s: 0 for s in states}
    
    def add_transition(self, s, a, s_next, prob, reward):
        """添加转移"""
        if s_next not in self.P[s][a]:
            self.P[s][a][s_next] = [0, reward]
        
        self.P[s][a][s_next][0] += prob
    
    def value_iteration(self, theta=1e-6, max_iter=100):
        """值迭代"""
        for _ in range(max_iter):
            delta = 0
            
            for s in self.S:
                old_v = self.V[s]
                
                # 最大Q值
                max_q = max(
                    sum(p * (r + self.gamma * self.V[s_next]) 
                    for s_next, (p, r) in self.P[s][a].items()
                ) for a in self.A
                self.V[s] = max_q
                
                delta = max(delta, abs(old_v - self.V[s]))
            
            if delta < theta:
                break
    
    def optimal_policy(self):
        """最优策略"""
        policy = {}
        
        for s in self.S:
            best_a = None
            best_v = -float('inf')
            
            for a in self.A:
                v = sum(
                    p * (r + self.gamma * self.V[s_next])
                    for s_next, (p, r) in self.P[s][a].items()
                )
                if v > best_v:
                    best_v = v
                    best_a = a
            
            policy[s] = best_a
        
        return policy
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_value_function(V, states, save_path='value.png'):
    """值函数"""
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(V)), list(V.values()))
    plt.xticks(range(len(V)), list(V.keys()), rotation=45)
    plt.ylabel('Value')
    plt.title('Value Function')
    plt.savefig(save_path)
    plt.show()


def visualize_mdp():
    """MDP可视化"""
    # 略
    pass
```

---

## 10. 评估

```python
def evaluate_policy(env, policy, n_episodes=100):
    """评估"""
    rewards = []
    
    for _ in range(n_episodes):
        s = env.reset()
        r = 0
        done = False
        
        while not done:
            a = policy[s]
            s, reward, done, _ = env.step(a)
            r += reward
        
        rewards.append(r)
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards)
    }
```

---

## 11. 常见问题

### 11.1 维度爆炸
- 函数近似
- 深度强化学习

### 11.2 收敛慢
- 异步更新
- 启发式方法

---

## 12. 总结

### 核心要点
1. **五元组**: (S, A, P, R, γ)
2. **值函数**: V^π(s), Q^π(s,a)
3. **贝尔曼**: 递归方程
4. **最优**: 动态规划求解

### 算法链
```
MDP → 策略迭代 → 值迭代 → Q学习 → DQN
    ↓
  POMDP → 深度强化学习
```

---

## 13. 练习题

**习题1**: 贝尔曼方程

<details>
<summary>答案</summary>

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$$

</details>

**习题2**: 最优性条件

<details>
<summary>答案</summary>

V* 满足贝尔曼最优方程:
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^*(s')]$$

</details>

---

## 14. 学习路径

- **初级**: MDP定义,值函数理解
- **中级**: 策略/值迭代实现
- **高级**: 扩展到POMDP,深度RL

### 推荐资源
- **书籍**: "Reinforcement Learning" - Sutton & Barto
- **论文**: Puterman "Markov Decision Processes"