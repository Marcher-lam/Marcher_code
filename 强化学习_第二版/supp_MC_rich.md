## 深度补充：蒙特卡洛方法高级主题

### 增量蒙特卡洛更新详解

蒙特卡洛方法可以使用增量更新，避免存储所有回报：

**增量更新公式**：
$$ V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right) $$

其中 $G_t$ 是从时刻t开始的完整回报，$\alpha$ 是学习率。

**优势**：
- 不需要存储所有历史回报
- 可以处理非平稳环境（通过α>0）
- 内存效率更高

**数学推导**：
这实际上是随机梯度下降在损失函数 $L(V) = \mathbb{E}[(G_t - V(S_t))^2]$ 上的应用。

### 重要度采样的高级形式

**加权重要度采样（Weighted Importance Sampling）**：
$$ V(s) = \frac{\sum_{t\in\mathcal{T}(s)} \rho_t G_t}{\sum_{t\in\mathcal{T}(s)} \rho_t} $$

其中 $\rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$ 是重要性采样比。

**优势**：比普通重要度采样方差更小，且是无偏估计（在有限样本下）

**普通重要度采样 vs 加权重要度采样**：
| 方法 | 偏差 | 方差 | 适用场景 |
|------|------|------|----------|
| 普通重要度采样 | 无偏 | 高方差 | 样本量大时 |
| 加权重要度采样 | 有偏（但偏差小） | 低方差 | 样本量小时 |

### 每决策蒙特卡洛（Every-Visit MC）vs 首次访问蒙特卡洛（First-Visit MC）

**数学定义**：
- **首次访问**：只使用每个episode中第一次访问状态s的回报
- **每决策**：使用每个episode中所有访问状态s的回报

**方差对比**：
- 首次访问：方差较小，因为回报数量少
- 每决策：方差较大，但利用了更多信息

**实验建议**：通常每决策MC收敛更快，因为利用了更多数据。

### 完整代码示例：增量蒙特卡洛 with 重要度采样

```python
import numpy as np

class IncrementalMC:
    """增量蒙特卡洛算法（支持重要度采样）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99, lr=0.01):
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.lr = lr
        self.episode_data = []  # 存储当前episode的数据
    
    def generate_episode(self, env, policy, max_steps=1000):
        """生成一个episode"""
        episode = []
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        
        return episode
    
    def calculate_return(self, episode):
        """计算episode中每个时间步的回报"""
        T = len(episode)
        returns = np.zeros(T)
        G = 0.0
        
        # 从后向前计算回报
        for t in range(T-1, -1, -1):
            reward = episode[t][2]
            G = reward + self.gamma * G
            returns[t] = G
        
        return returns
    
    def update_with_importance_sampling(self, episode, behavior_policy, target_policy):
        """使用重要度采样更新（off-policy）"""
        returns = self.calculate_return(episode)
        T = len(episode)
        
        for t in range(T):
            state, action, _ = episode[t]
            G = returns[t]
            
            # 计算重要性采样比
            rho = 1.0
            for k in range(t, T):
                s_k, a_k, _ = episode[k]
                rho *= target_policy(s_k, a_k) / behavior_policy(s_k, a_k)
            
            # 使用加权重要度采样更新
            # 这里简化为普通更新，实际应使用加权更新
            self.Q[state, action] += self.lr * rho * (G - self.Q[state, action])
    
    def update_on_policy(self, episode):
        """On-policy蒙特卡洛更新"""
        returns = self.calculate_return(episode)
        T = len(episode)
        
        visited = set()
        for t in range(T):
            state, action, _ = episode[t]
            
            # 首次访问MC：只更新第一次访问的状态-动作对
            if (state, action) not in visited:
                G = returns[t]
                self.Q[state, action] += self.lr * (G - self.Q[state, action])
                visited.add((state, action))
```

### 完整代码示例：蒙特卡洛探索策略改进（MC-ES）

```python
import numpy as np

class MonteCarloES:
    """蒙特卡洛探索策略改进（Monte Carlo Exploring Starts）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99):
        self.Q = np.zeros((n_states, n_actions))
        self.returns_sum = np.zeros((n_states, n_actions))
        self.returns_count = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.n_actions = n_actions
    
    def generate_episode_with_start(self, env, start_state=None, start_action=None):
        """生成episode，可选指定起始状态和动作（探索起点）"""
        episode = []
        
        if start_state is not None:
            state = start_state
            if start_action is not None:
                action = start_action
            else:
                action = np.random.randint(self.n_actions)
        else:
            state = env.reset()
            action = np.random.randint(self.n_actions)
        
        episode.append((state, action, 0))  # 初始奖励为0
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            if not done:
                action = np.random.randint(self.n_actions)
            episode.append((next_state, action, reward))
            state = next_state
        
        return episode
    
    def update(self, episode):
        """更新Q值（首次访问MC）"""
        T = len(episode)
        returns = np.zeros(T)
        G = 0.0
        
        # 从后向前计算回报
        for t in range(T-2, -1, -1):  # 最后一个是终止状态
            reward = episode[t+1][2]  # 奖励在下一个时间步
            G = reward + self.gamma * G
            returns[t] = G
        
        visited = set()
        for t in range(T-1):  # 不包括终止状态
            state, action, _ = episode[t]
            if (state, action) not in visited:
                G = returns[t]
                self.returns_sum[state, action] += G
                self.returns_count[state, action] += 1
                self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                visited.add((state, action))
    
    def improve_policy(self):
        """根据当前Q值改进策略（贪婪策略）"""
        policy = np.zeros(self.Q.shape[0], dtype=int)
        for s in range(self.Q.shape[0]):
            policy[s] = np.argmax(self.Q[s])
        return policy
    
    def train(self, env, num_episodes=1000):
        """训练MC-ES算法"""
        for episode_idx in range(num_episodes):
            # 探索起点：随机选择一个状态和动作作为起始
            start_state = np.random.randint(self.Q.shape[0])
            start_action = np.random.randint(self.n_actions)
            
            # 生成episode
            episode = self.generate_episode_with_start(env, start_state, start_action)
            
            # 更新Q值
            self.update(episode)
            
            # 每100个episode打印进度
            if (episode_idx + 1) % 100 == 0:
                policy = self.improve_policy()
                print(f"Episode {episode_idx+1}/{num_episodes}, Policy: {policy[:10]}...")
```

### 高级应用场景：游戏AI中的蒙特卡洛树搜索（MCTS）

**MCTS的核心思想**：
虽然MCTS不是传统的蒙特卡洛方法，但它使用了蒙特卡洛模拟的思想。

**四个步骤**：
1. **选择（Selection）**：从根节点开始，使用UCB公式选择子节点，直到到达叶子节点
2. **扩展（Expansion）**：如果叶子节点不是终止状态，添加一个或多个子节点
3. **模拟（Simulation）**：从新节点开始，进行随机模拟（rollout）直到终止
4. **回溯（Backpropagation）**：将模拟结果回溯更新所有祖先节点的价值

**UCB公式**：
$$ UCB(s,a) = Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a) + \epsilon}} $$

其中 $N(s)$ 是访问状态s的次数，$N(s,a)$ 是访问(s,a)的次数。

**AlphaGo中的应用**：
- 使用策略网络（Policy Network）指导模拟
- 使用价值网络（Value Network）评估叶子节点
- 结合蒙特卡洛模拟和深度神经网络

### 理论扩展：蒙特卡洛方法的收敛性

**定理**：在有限MDP中，使用首次访问MC或每决策MC，如果：
1. 所有状态-动作对被无限次访问
2. 学习率 $\alpha_t$ 满足 Robbins-Monro 条件
则 $Q(s,a)$ 几乎必然收敛到 $Q^\pi(s,a)$。

**证明要点**：
1. MC更新是随机逼近：$Q_{t+1}(s,a) = Q_t(s,a) + \alpha_t (G_t - Q_t(s,a))$
2. 期望更新方向是 $-\nabla L(Q)$，其中 $L(Q) = \mathbb{E}[(G_t - Q(s,a))^2]$
3. 根据随机逼近理论，收敛到 $Q^\pi$

### 更多练习题

**练习9：重要度采样方差实验**
问题：设计实验比较普通重要度采样和加权重要度采样在off-policy学习中的方差。

答案要点：
1. 环境：简单网格世界（如4x4 FrozenLake）
2. 行为策略：随机策略（均匀随机）
3. 目标策略：贪婪策略（基于当前Q值）
4. 运行1000个episode，记录每次更新的方差
5. 预期：加权重要度采样方差更小

**练习10：首次访问 vs 每决策 MC**
问题：在什么情况下首次访问MC比每决策MC更好？什么情况下相反？

答案要点：
1. 首次访问MC：适用于需要快速收敛的场景，方差小
2. 每决策MC：适用于数据稀缺的场景，利用更多信息
3. 实验：在样本量不同时比较两者性能
4. 结论：样本量大时两者相近，样本量小时每决策MC更好

**练习11：MC-ES的探索起点**
问题：为什么MC-ES需要“探索起点”（Exploring Starts）？如果去掉会怎样？

答案要点：
1. 探索起点确保所有状态-动作对被访问，保证收敛
2. 去掉后，如果初始状态分布不均匀，可能导致某些状态-动作对从未被访问
3. 解决方案：使用ε-greedy或其他探索策略
4. 实验：比较有/无探索起点的性能差异