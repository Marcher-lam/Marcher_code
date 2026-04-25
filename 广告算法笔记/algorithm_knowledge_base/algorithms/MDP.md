# MDP（马尔可夫决策过程）学习文档

## 1. 算法基础认知

MDP（Markov Decision Process）是强化学习的数学基础框架，由 Bellman 在 1957 年系统化。它为序贯决策问题提供了严格的数学描述，所有强化学习算法都建立在 MDP 之上。在广告系统中，出价、冷启动、多目标调控等问题都被建模为 MDP。

## 2. 核心原理

MDP 由五元组 (S, A, P, R, γ) 定义：

- **S**：状态空间，系统所有可能的状态集合
- **A**：动作空间，智能体可执行的动作集合
- **P(s'|s,a)**：状态转移概率，在状态 s 执行动作 a 后转移到 s' 的概率
- **R(s,a)**：奖励函数，在状态 s 执行动作 a 获得的即时奖励
- **γ ∈ [0,1]**：折扣因子，权衡当前与未来奖励

马尔可夫性质：未来只与当前状态有关，与历史无关。

$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1}|s_t, a_t)$$

## 3. 数学公式与推导

### 状态价值函数

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k+1} \mid s_t=s\right]$$

### 动作价值函数

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k+1} \mid s_t=s, a_t=a\right]$$

### Bellman 方程

状态价值的 Bellman 方程：

$$V^\pi(s) = \sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]$$

动作价值的 Bellman 方程：

$$Q^\pi(s, a) = R(s, a) + \gamma\sum_{s'}P(s'|s,a)\sum_{a'}\pi(a'|s')Q^\pi(s', a')$$

### 最优价值函数

$$V^*(s) = \max_\pi V^\pi(s)$$

$$Q^*(s, a) = R(s, a) + \gamma\sum_{s'}P(s'|s,a)\max_{a'}Q^*(s', a')$$

### 最优策略

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

## 4. 训练过程讲解

MDP 本身不是算法而是数学框架。求解 MDP 的方法包括：

1. **动态规划**（已知模型）：策略迭代、价值迭代
   - 策略评估：给定 π，计算 V^π
   - 策略改进：π'(s) = argmax_a Σ P(s'|s,a)[R + γV(s')]
   - 反复迭代直到收敛

2. **无模型方法**（未知模型）：Q-learning、SARSA
   - 通过交互学习 Q 函数

3. **策略梯度**：直接优化策略参数

广告出价的 MDP 求解通常使用无模型方法（RL 算法），因为环境动力学 P(s'|s,a) 未知。

## 5. 应用场景

- 广告出价的 RL 建模基础
- 冷启动动态决策
- 多目标调控的数学框架
- 路径规划、调度优化
- 机器人控制
- 金融投资决策

## 6. 优缺点分析

**优点**：
- 为序贯决策提供严格数学框架
- 可处理延迟奖励
- Bellman 方程提供了递归求解思路

**缺点**：
- 马尔可夫假设有时不成立（需要足够好的状态设计）
- 大规模 MDP 求解困难（维度灾难）
- 实际应用中 P(s'|s,a) 通常未知

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np

class MDP:
    def __init__(self, n_states, n_actions, transitions, rewards, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transitions
        self.R = rewards
        self.gamma = gamma

    def policy_evaluation(self, policy, theta=1e-6):
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for s_next in range(self.n_states):
                        v += action_prob * self.P[s][a][s_next] * (
                            self.R[s][a] + self.gamma * V[s_next])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break
        return V

    def policy_iteration(self):
        policy = np.ones([self.n_states, self.n_actions]) / self.n_actions
        while True:
            V = self.policy_evaluation(policy)
            policy_stable = True
            for s in range(self.n_states):
                old_action = np.argmax(policy[s])
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        action_values[a] += self.P[s][a][s_next] * (
                            self.R[s][a] + self.gamma * V[s_next])
                best_action = np.argmax(action_values)
                if old_action != best_action:
                    policy_stable = False
                policy[s] = np.eye(self.n_actions)[best_action]
            if policy_stable:
                return policy, V

    def value_iteration(self, theta=1e-6):
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        action_values[a] += self.P[s][a][s_next] * (
                            self.R[s][a] + self.gamma * V[s_next])
                best_value = np.max(action_values)
                delta = max(delta, abs(best_value - V[s]))
                V[s] = best_value
            if delta < theta:
                break
        policy = np.zeros([self.n_states, self.n_actions])
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    action_values[a] += self.P[s][a][s_next] * (
                        self.R[s][a] + self.gamma * V[s_next])
            policy[s][np.argmax(action_values)] = 1.0
        return policy, V
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class ValueIterationScratch:
    def __init__(self, rewards, transitions, gamma=0.99):
        self.R = rewards
        self.P = transitions
        self.gamma = gamma
        self.n_states = len(rewards)
        self.n_actions = len(rewards[0])

    def solve(self, max_iter=1000, tol=1e-6):
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_values[a] = self.R[s][a] + self.gamma * np.sum(
                        self.P[s][a] * V)
                V_new[s] = np.max(q_values)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                q_values[a] = self.R[s][a] + self.gamma * np.sum(
                    self.P[s][a] * V)
            policy[s] = np.argmax(q_values)
        return V, policy
```

## 9. 可视化与结果理解

- **价值函数热力图**：不同状态的价值分布
- **策略箭头图**：每个状态的最优动作方向
- **收敛曲线**：价值函数的最大变化量随迭代步数的下降
- **状态转移图**：MDP 的有向图表示

广告出价示例：横轴预算消耗率，纵轴时间进度，颜色表示价值函数。最优策略应表现为"前期探索、后期保守"。

## 10. 模型评估

- **策略收敛性**：策略迭代是否稳定
- **价值函数准确性**：与真实回报的误差
- **最优性验证**：是否存在更好的策略
- **模型假设合理性**：马尔可夫性是否成立

## 11. 常见问题与易错点

- **状态设计不满足马尔可夫性**：遗漏关键信息导致非马尔可夫
- **折扣因子 γ 设为 1**：无限回合下可能不收敛
- **奖励设计不当**：稀疏奖励导致学习困难
- **状态空间过大**：维度灾难，需要函数近似
- **转移概率不正确**：模型错误导致策略次优

## 12. 学习总结

MDP 是强化学习的数学基石，定义了状态、动作、转移、奖励和折扣因子五元组。Bellman 方程是求解 MDP 的核心工具，衍生出动态规划、Q-learning、策略梯度等算法族。在广告系统中，理解 MDP 建模是设计 RL 出价策略的第一步。

## 13. 练习题与思考题（含答案）

**Q1**：广告出价的 MDP 五元组是什么？

A1：S = {预算消耗率, 时间进度, CPA表现, 转化率}；A = {出价调整因子}；P = 环境动力学（通常未知）；R = α·转化量 - β·成本超标；γ = 0.99（重视长期累积回报）。

**Q2**：为什么需要折扣因子 γ？

A2：①数学上保证无限回合的价值函数有界；②实际中未来奖励的不确定性更高，应给予更低权重；③避免无限循环策略获得无穷回报。

**Q3**：策略迭代与价值迭代的区别？

A3：策略迭代交替进行策略评估和策略改进直到稳定；价值迭代直接迭代 Bellman 最优算子到收敛再提取策略。价值迭代通常更快，策略迭代每次迭代更有效。

## 14. 学习路径建议

前置知识：概率论 → 随机过程 → 动态规划
进阶方向：MDP → 动态规划 → Q-learning → DQN → PPO/SAC → 广告出价 RL 应用
