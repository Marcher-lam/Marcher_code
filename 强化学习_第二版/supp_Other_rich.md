## 深度补充：强化学习进阶主题

### 探索与利用的经典算法详解

**ε-greedy探索**：
最简单的探索策略，以ε概率随机探索，1-ε概率贪心利用。

**更新规则**：
$$ \pi(a|s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases} $$

**缺点**：探索效率低下，固定ε不自适应。

**UCB1（Upper Confidence Bound）**：
$$ A_t = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a) + \epsilon}} \right] $$

其中 $N(s,a)$ 是(s,a)被访问的次数，c是探索常数。

**优势**：理论保证累积遗憾界为 $O(\sqrt{T|\mathcal{A}|\ln T})$
**劣势**：需要存储访问次数，初始探索可能不够

**Softmax（Boltzmann）探索**：
$$ \pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)} $$

其中τ是温度参数，τ→0时退化为贪心策略，τ→∞时退化为均匀随机。

### 完整代码示例：自适应探索策略

```python
import numpy as np
import math

class AdaptiveExploration:
    """自适应探索策略：根据状态访问频率调整探索率"""
    
    def __init__(self, n_states, n_actions, init_epsilon=1.0, min_epsilon=0.01, 
                 decay_rate=0.995, ucb_c=1.0, softmax_tau=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.ucb_c = ucb_c
        self.softmax_tau = softmax_tau
        
        # 统计信息
        self.state_visits = np.zeros(n_states)
        self.sa_visits = np.zeros((n_states, n_actions))
        self.Q = np.zeros((n_states, n_actions))
    
    def epsilon_greedy(self, state):
        """ε-greedy动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def ucb_action(self, state):
        """UCB动作选择"""
        total_visits = self.state_visits[state] + 1
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            if self.sa_visits[state, a] == 0:
                # 未访问过的动作直接选择（乐观初始化）
                return a
            
            q_val = self.Q[state, a]
            bonus = self.ucb_c * math.sqrt(math.log(total_visits) / self.sa_visits[state, a])
            ucb_values[a] = q_val + bonus
        
        return np.argmax(ucb_values)
    
    def softmax_action(self, state):
        """Softmax动作选择"""
        q_vals = self.Q[state] / self.softmax_tau
        # 数值稳定性：减去最大值
        q_vals = q_vals - np.max(q_vals)
        probs = np.exp(q_vals) / np.sum(np.exp(q_vals))
        return np.random.choice(self.n_actions, p=probs)
    
    def adaptive_epsilon(self, state):
        """自适应ε：根据状态访问频率调整"""
        if self.state_visits[state] == 0:
            return 1.0  # 新状态，完全探索
        else:
            # 访问越多次，ε越小
            adaptive_eps = max(self.min_epsilon, 
                              self.epsilon * (self.decay_rate ** self.state_visits[state]))
            return adaptive_eps
    
    def select_action(self, state, method='adaptive'):
        """统一的动作选择接口"""
        if method == 'epsilon_greedy':
            return self.epsilon_greedy(state)
        elif method == 'ucb':
            return self.ucb_action(state)
        elif method == 'softmax':
            return self.softmax_action(state)
        elif method == 'adaptive':
            # 自适应：新状态用UCB，旧状态用ε-greedy
            if self.state_visits[state] < 10:
                return self.ucb_action(state)
            else:
                eps = self.adaptive_epsilon(state)
                if np.random.random() < eps:
                    return np.random.randint(self.n_actions)
                else:
                    return np.argmax(self.Q[state])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def update_stats(self, state, action, reward, next_state, done, lr=0.1, gamma=0.99):
        """更新统计信息和Q值"""
        # 更新访问次数
        self.state_visits[state] += 1
        self.sa_visits[state, action] += 1
        
        # Q-learning更新
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += lr * td_error
        
        # 衰减ε
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
```

### 函数逼近的理论基础

**线性函数逼近**：
$$ \hat{V}(s, \mathbf{w}) = \mathbf{w}^\top \phi(s) $$
其中 $\phi(s)$ 是状态s的特征向量，$\mathbf{w}$ 是权重向量。

**梯度下降更新**：
$$ \mathbf{w} \leftarrow \mathbf{w} - \frac{1}{2} \alpha \nabla_w \left( V^\pi(s) - \hat{V}(s, \mathbf{w}) \right)^2 $$
$$ = \mathbf{w} + \alpha \left( V^\pi(s) - \hat{V}(s, \mathbf{w}) \right) \phi(s) $$

**非线性函数逼近（神经网络）**：
$$ \hat{V}(s, \theta) = f_\theta(s) $$
使用反向传播和梯度下降更新参数θ。

**灾难性遗忘问题**：
当环境非平稳或任务切换时，神经网络可能忘记之前学习的知识。

**解决方案**：
1. **经验回放（Experience Replay）**：存储历史经验，随机采样训练
2. **弹性权重整合（EWC）**：保护重要参数不被大幅修改
3. **渐进式神经网络（Progressive Neural Networks）**：为新任务添加新列，保留旧列

### 完整代码示例：线性函数逼近TD

```python
import numpy as np

class LinearTDAgent:
    """线性函数逼近的TD(0)算法"""
    
    def __init__(self, n_features, n_actions, gamma=0.99, lr=0.01, lamda=0.0):
        self.weights = np.zeros((n_actions, n_features))  # 每个动作一个权重向量
        self.gamma = gamma
        self.lr = lr
        self.lamda = lamda  # λ=0时为TD(0)，λ>0时使用资格迹
        self.e_trace = np.zeros((n_actions, n_features))  # 资格迹
    
    def feature_vector(self, state, action=None):
        """将状态转换为特征向量（示例：简单编码）"""
        # 这里假设state已经是特征向量
        # 如果action不为None，可以构造状态-动作特征
        if action is not None:
            # 简单示例：state特征和one-hot action拼接
            n_actions = self.weights.shape[0]
            action_one_hot = np.zeros(n_actions)
            action_one_hot[action] = 1
            return np.concatenate([state, action_one_hot])
        return state
    
    def value(self, state, action):
        """计算Q(s,a)"""
        features = self.feature_vector(state, action)
        return np.dot(self.weights[action], features)
    
    def update_td0(self, state, action, reward, next_state, done):
        """TD(0)更新"""
        # 当前Q值
        q_current = self.value(state, action)
        
        # TD目标
        if done:
            td_target = reward
        else:
            # 贪心选择下一个动作
            next_q_values = [self.value(next_state, a) for a in range(self.weights.shape[0])]
            next_action = np.argmax(next_q_values)
            td_target = reward + self.gamma * self.value(next_state, next_action)
        
        # TD误差
        td_error = td_target - q_current
        
        # 更新权重
        features = self.feature_vector(state, action)
        self.weights[action] += self.lr * td_error * features
    
    def update_td_lambda(self, trajectory, rewards):
        """TD(λ)更新（使用资格迹）"""
        T = len(trajectory)
        self.e_trace = np.zeros_like(self.weights)  # 重置资格迹
        
        for t in range(T):
            state, action = trajectory[t]
            reward = rewards[t]
            
            # 计算TD目标
            if t < T-1:
                next_state, next_action = trajectory[t+1]
                td_target = reward + self.gamma * self.value(next_state, next_action)
            else:
                td_target = reward
            
            q_current = self.value(state, action)
            td_error = td_target - q_current
            
            # 更新资格迹
            features = self.feature_vector(state, action)
            self.e_trace[action] = self.gamma * self.lamda * self.e_trace[action] + features
            
            # 更新权重（所有动作的资格迹）
            for a in range(self.weights.shape[0]):
                self.weights[a] += self.lr * td_error * self.e_trace[a]
```

### 历史算法：Samuel的跳棋程序

**背景**：Arthur Samuel在1959年开发的跳棋程序，是强化学习早期代表作。

**核心思想**：
1. **自我对弈**：程序与自己下棋，从胜负中学习
2. **评分函数**：手工设计棋盘特征的线性组合
3. **奖励塑造**：根据棋盘评分提供中间奖励，而非只靠最终胜负

**评分特征示例**：
- 棋子数量差
- 王棋数量差
- 移动性（可行走步数）
- 控制中心程度

**更新规则（类似于TD学习）**：
$$ V(s) \leftarrow V(s) + \alpha (V(s') - V(s)) $$

**历史意义**：
- 证明了机器学习可以在复杂任务中超越人类专家
- 引入了自我对弈、奖励塑造等重要概念
- 为后来的TD学习和强化学习奠定基础

### 高级应用场景：推荐系统中的RL

**场景**：新闻推荐、视频推荐、商品推荐

**为什么使用RL**：
1. **延迟奖励**：用户点击后立即奖励，但长期满意度需要多步观察
2. **动态环境**：用户兴趣随时间变化，需要持续学习
3. **长期价值**：不仅优化点击率，还优化用户长期参与度和满意度

**状态设计**：
- 用户历史行为（点击、浏览、购买）
- 用户画像（年龄、性别、地域）
- 上下文信息（时间、设备、位置）
- 物品特征（类别、标签、价格）

**动作空间**：
- 推荐哪些物品（通常是top-k推荐）
- 推荐策略（多样性、新颖性调整）

**奖励设计**：
- 即时奖励：点击（+1）、购买（+10）、停留时间（归一化）
- 延迟奖励：用户满意度调查、长期留存

**挑战与解决方案**：
1. **稀疏奖励**：使用奖励塑造，增加中间奖励
2. **探索成本高**：使用离线评估、仿真环境
3. **冷启动**：结合监督学习初始化策略

### 理论扩展：强化学习的 regret bound

**定义**：Regret是在T步内，算法累积奖励与最优策略累积奖励的差值：
$$ \text{Regret}(T) = T \cdot V^{\pi^*} - \sum_{t=1}^T r_t $$

**常见算法的Regret界**：
| 算法 | Regret界 | 假设条件 |
|------|----------|----------|
| ε-greedy Q-learning | $O(T^{2/3})$ | 有限状态-动作空间 |
| UCB | $O(\sqrt{T \|\|S\|\|A\| \ln T})$ | 有限状态-动作空间 |
| Thompson Sampling | $O(\sqrt{T \|\|S\|\|A\| \ln T})$ | 有限状态-动作空间 |
| DQN（深度） | 无理论保证 | 依赖经验 |

**重要性**：Regret界衡量算法的样本效率，低Regret意味着更快学习。

### 更多练习题

**练习18：探索策略对比实验**
问题：在Multi-Armed Bandit环境中，比较ε-greedy、UCB、Softmax的性能。

答案要点：
1. 环境：10-arm Gaussian bandit，每个臂奖励～N(μ_i, 1)
2. 算法：三种探索策略，相同计算预算
3. 评估：累积奖励、累积遗憾
4. 预期：UCB和Thompson Sampling理论更优，ε-greedy简单但可能次优
5. 参数调优：ε衰减、UCB的c、Softmax的τ

**练习19：线性函数逼近的收敛性**
问题：证明线性TD(0)收敛到TD固定点（最小二乘解）。

答案要点：
1. TD固定点：$w_{TD} = A^{-1}b$，其中
   $A = \mathbb{E}[\phi_t (\phi_t - \gamma \phi_{t+1})^\top]$
   $b = \mathbb{E}[\phi_t r_t]$
2. 线性TD(0)更新：$\Delta w = \alpha (r_t + \gamma w^\top \phi_{t+1} - w^\top \phi_t) \phi_t$
3. 期望更新：$\mathbb{E}[\Delta w] = A w - b$
4. 收敛到 $w_{TD} = A^{-1}b$

**练习20：灾难性遗忘实验**
问题：设计一个实验，展示神经网络在连续任务中的灾难性遗忘。

答案要点：
1. 任务：连续学习两个CartPole任务（不同参数）
2. 算法：DQN，直接在新任务上训练
3. 观察：在新任务上性能提升，但在旧任务上性能下降
4. 解决方案：Experience Replay（存储旧任务经验）
5. 评估：在两个任务上的平均奖励