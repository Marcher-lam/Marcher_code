## 深度补充：TD算法高级主题

### 多步TD与资格迹的统一视角

TD(λ)与n-step TD可以通过**截断λ回报**统一表示：

$$ G_t^{(\lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)} $$

其中 $G_t^{(n)}$ 是n步回报。当λ=0时，退化为TD(0)；当λ=1时，接近蒙特卡洛。

**资格迹的三种形式**：
1. **累积迹（Accumulating Trace）**：$E_t = \gamma\lambda E_{t-1} + \nabla_\theta \log \pi(A_t|S_t)$
2. **替换迹（Replacing Trace）**：$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbf{1}(S_t=s)$
3. **Dutch Trace**：$E_t = \gamma\lambda E_{t-1} + \alpha \nabla_\theta \log \pi(A_t|S_t) Q(S_t,A_t)$

### 树备份算法（Tree Backup）详解

树备份是一种off-policy的n-step TD算法，通过期望树结构避免重要性采样：

**更新规则**：
$$ Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \delta_t \prod_{k=1}^{n-1} \rho_{t+k} $$

其中 $\rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$ 是重要性采样比，$\delta_t$ 是TD误差。

**优势**：完全避免重要性采样的方差问题
**劣势**：计算复杂度高，需要遍历所有可能的动作

### 强化学习中的偏差-方差困境

TD学习面临经典的偏差-方差权衡：

| 算法 | 偏差 | 方差 | 原因 |
|------|------|------|------|
| TD(0) | 高 | 低 | Bootstrap导致偏差，但只有单步噪声 |
| TD(λ) | 中 | 中 | λ参数控制偏差-方差权衡 |
| 蒙特卡洛 | 无 | 高 | 无bootstrap，但累计多步噪声 |
| 树备份 | 低 | 中 | 使用期望而非采样，降低方差 |

**数学推导**：
TD(0)的偏差来源：
$$ \mathbb{E}[\delta_t] = \mathbb{E}[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)] $$
$$ = \mathbb{E}[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t) $$
由于 $Q(S_{t+1}, A_{t+1})$ 是估计值而非真实值，存在bootstrap偏差。

### 完整代码示例：通用TD(λ)实现（支持多种资格迹）

```python
import numpy as np

class UniversalTDLambda:
    """通用TD(λ)实现，支持多种资格迹和n-step"""
    
    def __init__(self, n_states, n_actions, lamda=0.9, gamma=0.99, lr=0.01,
                 trace_type='accumulating'):
        self.Q = np.zeros((n_states, n_actions))
        self.lamda = lamda
        self.gamma = gamma
        self.lr = lr
        self.trace_type = trace_type
        self.n_states = n_states
        self.n_actions = n_actions
    
    def reset_eligibility(self):
        """重置资格迹"""
        self.E = np.zeros((self.n_states, self.n_actions))
    
    def update_td_lambda(self, trajectory, rewards):
        """
        通用TD(λ)更新（支持多种资格迹）
        trajectory: [(s0,a0), (s1,a1), ..., (s_T,a_T)]
        rewards: [r1, r2, ..., r_T]
        """
        T = len(trajectory)
        self.reset_eligibility()
        
        for t in range(T):
            s, a = trajectory[t]
            r = rewards[t]
            
            # 计算TD目标
            if t < T-1:
                s_next, a_next = trajectory[t+1]
                td_target = r + self.gamma * self.Q[s_next, a_next]
            else:
                td_target = r  # 终止状态
            
            td_error = td_target - self.Q[s, a]
            
            # 根据资格迹类型更新
            if self.trace_type == 'accumulating':
                # 累积迹：E = γλE + 1(s,a)
                self.E *= self.gamma * self.lamda
                self.E[s, a] += 1.0
            elif self.trace_type == 'replacing':
                # 替换迹：E = γλE，然后E(s,a) = 1
                self.E *= self.gamma * self.lamda
                self.E[s, a] = 1.0
            elif self.trace_type == 'dutch':
                # Dutch迹：E = γλE + α * ∇logπ * Q
                self.E *= self.gamma * self.lamda
                self.E[s, a] += self.lr * self.Q[s, a]
            
            # 更新Q值：所有状态-动作对根据资格迹权重更新
            self.Q += self.lr * td_error * self.E
    
    def update_n_step(self, trajectory, rewards, n):
        """
        n-step TD更新
        n: 步数
        """
        T = len(trajectory)
        
        for t in range(T):
            # 计算n步回报
            G = 0.0
            for k in range(min(n, T - t)):
                G += (self.gamma ** k) * rewards[t + k]
            
            # 添加bootstrap项
            if t + n < T:
                s_n, a_n = trajectory[t + n]
                G += (self.gamma ** n) * self.Q[s_n, a_n]
            
            # 更新Q值
            s, a = trajectory[t]
            td_error = G - self.Q[s, a]
            self.Q[s, a] += self.lr * td_error
```

### 完整代码示例：Expected Sarsa与Double Q-learning结合

```python
import numpy as np

class ExpectedDoubleQLearning:
    """Expected Double Q-learning：结合Double Q-learning和Expected Sarsa"""
    
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.99, lr=0.01):
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
    
    def select_action(self, state):
        """ε-greedy动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # 使用Q1+Q2的平均值选择动作
            return np.argmax(self.Q1[state] + self.Q2[state])
    
    def expected_q_value(self, state, Q):
        """计算期望Q值（用于Expected Sarsa）"""
        best_action = np.argmax(Q[state])
        expected = 0.0
        for a in range(self.n_actions):
            if a == best_action:
                prob = 1.0 - self.epsilon + self.epsilon / self.n_actions
            else:
                prob = self.epsilon / self.n_actions
            expected += prob * Q[state, a]
        return expected
    
    def update(self, s, a, r, s_next, done):
        """更新Q1或Q2（随机选择）"""
        if np.random.random() < 0.5:
            # 更新Q1，使用Q2评估
            if done:
                td_target = r
            else:
                td_target = r + self.gamma * self.expected_q_value(s_next, self.Q2)
            td_error = td_target - self.Q1[s, a]
            self.Q1[s, a] += self.lr * td_error
        else:
            # 更新Q2，使用Q1评估
            if done:
                td_target = r
            else:
                td_target = r + self.gamma * self.expected_q_value(s_next, self.Q1)
            td_error = td_target - self.Q2[s, a]
            self.Q2[s, a] += self.lr * td_error
    
    def get_optimal_policy(self):
        """获取最优策略（基于Q1+Q2）"""
        policy = np.zeros(self.Q1.shape[0], dtype=int)
        for s in range(self.Q1.shape[0]):
            policy[s] = np.argmax(self.Q1[s] + self.Q2[s])
        return policy
```

### 高级应用场景：金融交易中的TD算法

**场景1：高频股票交易**
- **问题**：需要在毫秒级别做出买卖决策，不能等待episode结束
- **TD优势**：单步更新，快速适应市场变化
- **实现要点**：
  - 状态：过去N分钟的价格变化、成交量、技术指标
  - 动作：买入、卖出、持有
  - 奖励：考虑交易成本后的净收益
  - 使用Double Q-learning减少过估计，避免过于乐观的交易策略

**场景2：期权对冲**
- **问题**：需要实时调整对冲组合，降低风险
- **TD优势**：可以在每个时间步更新对冲策略
- **实现要点**：
  - 状态：期权价格、标的资产价格、波动率、时间到期
  - 动作：调整对冲比例（Delta对冲）
  - 奖励：组合价值变化减去交易成本
  - 使用TD(λ)平衡偏差和方差

### 调参指南与最佳实践

**1. λ参数选择**
- **任务特点**：episode长度、噪声水平
- **经验法则**：
  - 短episode、低噪声：λ较大（0.7-0.9）
  - 长episode、高噪声：λ较小（0.3-0.6）
  - 极高噪声：λ=0（TD(0)）
- **网格搜索**：λ ∈ {0, 0.3, 0.5, 0.7, 0.9}

**2. 学习率α调整**
- **TD(0)**：α ≈ 0.1-0.5（单步更新，可以较大）
- **TD(λ)**：α ≈ 0.01-0.1（多步更新，需要较小）
- **自适应学习率**：α_t = α_0 / (1 + βt)（随时间衰减）

**3. 折扣因子γ设置**
- **短期任务**：γ较小（0.7-0.9）
- **长期任务**：γ较大（0.9-0.99）
- **无期限任务**：γ=0.99以上

### 理论扩展：TD学习的收敛性证明

**命题**：在有限MDP中，使用线性函数逼近的TD(0)算法，如果学习率满足Robbins-Monro条件：
$$ \sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty $$
则TD(0)几乎必然收敛到TD固定点（TD fixed point）。

**证明思路**：
1. TD(0)更新可以写作随机逼近：$V_{t+1} = V_t + \alpha_t (R_{t+1} + \gamma V(S_{t+1}) - V(S_t)) \phi(S_t)$
2. 其中$\phi(S_t)$是特征向量
3. 期望更新方向是：$-\nabla L(V)$，其中$L(V) = \mathbb{E}[(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))^2]$
4. 根据随机逼近理论，算法收敛到$L(V)$的驻点

### 更多练习题

**练习6：TD(λ)的λ参数实验设计**
问题：设计一个实验，在CartPole环境中比较不同λ值（0, 0.3, 0.5, 0.7, 0.9, 1.0）的性能。

答案要点：
1. 环境：CartPole-v1，状态空间4维，动作空间2维
2. 函数逼近：线性函数或小型神经网络
3. 评估指标：平均episode长度（100个episode平均）
4. 每个λ运行500个episode，记录学习曲线
5. 预期结果：中等λ（0.5-0.7）通常最优，平衡偏差和方差

**练习7：Double Q-learning的过估计分析**
问题：通过实验证明Double Q-learning减少了Q-learning的过估计偏差。

答案要点：
1. 创建一个简单环境（如4状态2动作），真实Q值已知
2. 分别运行Q-learning和Double Q-learning
3. 比较学习到的Q值与真实Q值的差异
4. 预期：Double Q-learning的Q值更接近真实值

**练习8：Expected Sarsa的方差分析**
问题：比较Sarsa、Expected Sarsa、Q-learning的方差。

答案要点：
1. 理论分析：Expected Sarsa使用期望，方差最小；Sarsa使用采样，方差中等；Q-learning使用max，方差最大
2. 实验验证：在相同环境中运行三种算法，记录Q值更新的方差
3. 结论：Expected Sarsa在稳定性上优于Sarsa和Q-learning