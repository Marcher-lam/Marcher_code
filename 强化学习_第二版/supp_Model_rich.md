## 深度补充：Model-Based强化学习高级主题

### Dyna架构的深度解析

Dyna算法结合了模型学习和直接强化学习，核心思想是：**用模型生成的模拟经验辅助真实经验**。

**Dyna-Q算法流程**：
1. **真实交互**：用当前策略与环境交互，得到(s,a,r,s')
2. **直接学习**：用真实经验更新Q值（如Q-learning）
3. **模型学习**：更新环境模型：M(s,a) → (r,s')
4. **规划**：从模型中采样n个模拟经验，用同样方式更新Q值

**数学形式**：
$$ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)] $$
这个更新既用于真实经验，也用于模拟经验。

**优势**：
- 模型可以利用少量真实经验生成大量模拟经验
- 加速学习，特别是在真实样本昂贵时
- 结合model-free和model-based的优点

### 完整代码示例：Dyna-Q实现

```python
import numpy as np
import random

class DynaQ:
    """Dyna-Q算法实现"""
    
    def __init__(self, n_states, n_actions, gamma=0.99, lr=0.1, epsilon=0.1, planning_steps=10):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # 模型： (s,a) -> (r, s')
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.planning_steps = planning_steps  # 每次真实步骤后的规划步数
        self.n_actions = n_actions
        self.visited_sa = set()  # 记录访问过的状态-动作对
    
    def select_action(self, state):
        """ε-greedy动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def update_model(self, state, action, reward, next_state):
        """更新环境模型"""
        self.model[(state, action)] = (reward, next_state)
        self.visited_sa.add((state, action))
    
    def q_learning_update(self, state, action, reward, next_state):
        """Q-learning更新（用于真实经验和模拟经验）"""
        if next_state is None:  # 终止状态
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error
        return td_error
    
    def planning(self):
        """从模型中采样进行规划"""
        if len(self.visited_sa) == 0:
            return
        
        for _ in range(self.planning_steps):
            # 随机选择一个访问过的状态-动作对
            s, a = random.choice(list(self.visited_sa))
            
            # 从模型中获取结果
            if (s, a) in self.model:
                r, s_next = self.model[(s, a)]
                self.q_learning_update(s, a, r, s_next)
    
    def train_episode(self, env, max_steps=1000):
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 更新模型
            self.update_model(state, action, reward, None if done else next_state)
            
            # Q-learning更新（真实经验）
            self.q_learning_update(state, action, reward, None if done else next_state)
            
            # 规划（模拟经验）
            self.planning()
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            state = next_state
        
        return total_reward, steps
```

### 完整代码示例：Dyna-Q+（带探索奖励）

```python
import numpy as np
import random
import time

class DynaQPlus:
    """Dyna-Q+算法：带有探索奖励的Dyna变体"""
    
    def __init__(self, n_states, n_actions, gamma=0.99, lr=0.1, epsilon=0.1, 
                 planning_steps=10, kappa=1e-4):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.n_actions = n_actions
        self.kappa = kappa  # 探索奖励系数
        
        # 记录每个状态-动作对的最后访问时间
        self.last_visit_time = {}
        self.current_time = 0
    
    def select_action(self, state):
        """ε-greedy动作选择（含探索奖励）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # 计算包含探索奖励的Q值
            augmented_q = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                # 基础Q值
                q_val = self.Q[state, a]
                
                # 探索奖励：很久没访问的动作获得额外奖励
                if (state, a) in self.last_visit_time:
                    time_since_visit = self.current_time - self.last_visit_time[(state, a)]
                    bonus = self.kappa * np.sqrt(time_since_visit)
                else:
                    bonus = self.kappa * np.sqrt(self.current_time + 1)
                
                augmented_q[a] = q_val + bonus
            
            return np.argmax(augmented_q)
    
    def update_model(self, state, action, reward, next_state):
        """更新模型和时间戳"""
        self.model[(state, action)] = (reward, next_state)
        self.last_visit_time[(state, action)] = self.current_time
        self.visited_sa.add((state, action))
    
    def q_learning_update(self, state, action, reward, next_state):
        """Q-learning更新"""
        if next_state is None:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error
        return td_error
    
    def planning(self):
        """规划步骤（使用模型生成的经验）"""
        if len(self.visited_sa) == 0:
            return
        
        for _ in range(self.planning_steps):
            s, a = random.choice(list(self.visited_sa))
            
            if (s, a) in self.model:
                r, s_next = self.model[(s, a)]
                self.q_learning_update(s, a, r, s_next)
    
    def train_episode(self, env, max_steps=1000):
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 更新模型和时间
            self.update_model(state, action, reward, None if done else next_state)
            self.current_time += 1
            
            # Q-learning更新
            self.q_learning_update(state, action, reward, None if done else next_state)
            
            # 规划
            self.planning()
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            state = next_state
        
        return total_reward, steps
```

### 蒙特卡洛树搜索（MCTS）高级主题

** Upper Confidence Bounds for Trees (UCT) **：
MCTS的核心选择策略，平衡探索和利用：

$$ UCT(s,a) = Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a) + \epsilon}} $$

其中：
- $Q(s,a)$ 是状态-动作对的平均价值
- $N(s)$ 是访问状态s的次数
- $N(s,a)$ 是访问(s,a)的次数
- $c$ 是探索常数（通常√2）

**四个阶段详解**：
1. **选择（Selection）**：从根节点开始，使用UCT选择子节点，直到到达叶子节点
2. **扩展（Expansion）**：如果叶子节点不是终止状态，添加一个或多个未访问的子节点
3. **模拟（Simulation）**：从新节点开始，使用默认策略（如随机）模拟到终止
4. **回溯（Backpropagation）**：将模拟结果回溯更新所有祖先节点的统计信息

### 模型学习算法：高斯过程与神经网络

**高斯过程模型（Gaussian Process Model）**：
- 非参数贝叶斯方法，提供不确定性估计
- 适用于连续状态空间的小样本学习
- 计算复杂度O(n³)，不适合大规模问题

**神经网络模型（Neural Network Model）**：
- 学习状态转移函数：$s_{t+1} = f_\theta(s_t, a_t)$
- 学习奖励函数：$r_t = g_\phi(s_t, a_t)$
- 可以用梯度下降训练，适合大规模问题

**代码示例（简单神经网络模型）**：
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ModelNetwork(nn.Module):
    """神经网络环境模型"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ModelNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # 输出：next_state (state_dim) + reward (1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        output = self.net(x)
        next_state = output[:, :-1]
        reward = output[:, -1]
        return next_state, reward
```

### 高级应用场景：自动驾驶规划

**场景**：自动驾驶车辆在复杂城市环境中规划路径

**为什么使用Model-Based RL**：
1. **安全性**：可以在模型中模拟危险场景，无需真实碰撞
2. **样本效率**：真实驾驶数据昂贵且危险，模型可以生成大量模拟经验
3. **长期规划**：模型可以进行多步预测，适合长期规划

**实现架构**：
- **状态**：车辆位置、速度、周围车辆状态、交通灯状态
- **动作**：加速度、转向角
- **模型**：学习车辆动力学模型 + 交通参与者行为模型
- **规划**：使用MCTS或Dyna进行路径规划

**挑战与解决方案**：
1. **模型误差累积**：使用ensemble模型（多个模型取平均）降低误差
2. **真实世界随机性**：在模型中加入噪声，提高鲁棒性
3. **计算效率**：使用简化模型进行快速规划，复杂模型进行精细评估

### 理论扩展：模型误差对规划的影响

**定理**：如果模型误差为ε（即 $|P_{true}(s'|s,a) - P_{model}(s'|s,a)| \leq \epsilon$），则规划得到的策略性能界限为：

$$ V^{\pi^*_{model}}(s) \geq V^{\pi^*_{true}}(s) - \frac{2\gamma\epsilon}{(1-\gamma)^2} $$

**证明思路**：
1. 模型误差导致价值函数误差：$\| V^{\pi}_{true} - V^{\pi}_{model} \|_\infty \leq \frac{\epsilon}{1-\gamma}$
2. 策略误差：选择错误动作的概率有界
3. 性能差异：通过贝尔曼方程推导

**实践意义**：模型精度直接影响最终性能，需要平衡模型复杂度和采样效率。

### 更多练习题

**练习15：Dyna-Q的规划步数调参**
问题：设计实验研究规划步数（planning_steps）对Dyna-Q性能的影响。

答案要点：
1. 环境：网格世界（如FrozenLake）
2. 测试不同planning_steps：{1, 5, 10, 20, 50}
3. 评估指标：达到最优策略所需的真实episode数
4. 预期结果：适当增加规划步数加速学习，但过多可能浪费计算
5. 最优值：通常在10-20之间

**练习16：MCTS的探索常数c选择**
问题：如何为特定任务选择合适的UCT探索常数c？

答案要点：
1. 理论基础：c = √2 在理论上保证收敛
2. 实践调参：根据任务特点调整
   - 探索性任务：增大c（如2-5）
   - 利用性任务：减小c（如0.5-1）
3. 自适应调整：根据搜索树深度动态调整c
4. 实验：在固定计算预算下，比较不同c的性能

**练习17：模型误差传播分析**
问题：分析模型误差如何在多步规划中传播？

答案要点：
1. 单步误差：模型预测next_state的误差
2. 多步累积：误差随规划步数指数增长
3. 数学推导：$error_k \leq \epsilon \sum_{i=0}^{k-1} \gamma^i \approx \frac{\epsilon}{1-\gamma}$
4. 缓解方法：限制规划步数、使用概率模型、ensemble方法