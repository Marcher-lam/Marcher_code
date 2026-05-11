## 深度补充：动态规划高级主题

### 策略迭代的收敛性证明

**定理**：策略迭代在有限MDP中保证收敛到最优策略。

**证明**：
1. **策略评估**：给定策略π，计算 $V^\pi$ 使用贝尔曼方程：
   $$ V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')] $$
   这是一个线性系统，有唯一解。

2. **策略改进**：贪心更新保证单调改进：
   $$ \pi'(s) = \arg\max_a Q^\pi(s,a) $$
   根据贝尔曼最优方程，如果 $Q^\pi(s, \pi'(s)) = V^\pi(s)$ 对所有s成立，则π已是最优。

3. **有限性**：有限MDP中策略数量有限（$|A|^{|S|}$），每次改进严格提升（或不变），故必在有限步内收敛。

### 值迭代的误差界

**定理**：值迭代第k次迭代的误差界为：
$$ \| V_k - V^* \|_\infty \leq \frac{\gamma^k}{1-\gamma} \| V_1 - V_0 \|_\infty $$

**推导**：
值迭代：$V_{k+1} = \mathcal{B} V_k$，其中$\mathcal{B}$是最优贝尔曼算子。
由于$\mathcal{B}$是$\gamma$-压缩映射：
$$ \| \mathcal{B} V - \mathcal{B} V' \|_\infty \leq \gamma \| V - V' \|_\infty $$
迭代k次后：
$$ \| V_k - V^* \|_\infty \leq \gamma^k \| V_0 - V^* \|_\infty $$
利用几何级数求和得到上述界。

### 广义策略迭代（GPI）框架

策略迭代和值迭代都是GPI的特例：

**GPI的两个同步过程**：
1. **策略评估**：给定策略π，计算 $V^\pi$ 或 $Q^\pi$
2. **策略改进**：给定价值函数V，改进策略π

**不同算法的GPI实现**：
| 算法 | 策略评估 | 策略改进 |
|------|----------|----------|
| 策略迭代 | 精确求解（迭代评估至收敛） | 完全贪心更新 |
| 值迭代 | 单次更新（不等待收敛） | 隐式改进（在值更新中） |
| 截断策略迭代 | 有限次评估迭代 | 完全贪心更新 |
| Q-learning | 单步样本更新 | 隐式改进（max操作） |

### 完整代码示例：异步动态规划

```python
import numpy as np
import random

class AsyncDP:
    """异步动态规划（异步策略迭代/值迭代）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99):
        self.V = np.zeros(n_states)
        self.Q = np.zeros((n_states, n_actions))
        self.policy = np.zeros(n_states, dtype=int)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
    
    def set_model(self, transitions):
        """设置环境模型：transitions[s][a] = [(prob, next_state, reward, done), ...]"""
        self.transitions = transitions
    
    def async_policy_evaluation(self, policy, num_iterations=1000):
        """异步策略评估：每次只更新一个状态"""
        for _ in range(num_iterations):
            # 随机选择一个状态
            s = random.randint(0, self.n_states - 1)
            
            # 计算该状态的贝尔曼期望方程
            v = 0.0
            a = policy[s]
            for prob, next_s, reward, done in self.transitions[s][a]:
                if done:
                    v += prob * reward
                else:
                    v += prob * (reward + self.gamma * self.V[next_s])
            
            # 更新该状态的价值
            self.V[s] = v
    
    def async_value_iteration(self, num_iterations=1000):
        """异步值迭代：每次只更新一个状态"""
        for _ in range(num_iterations):
            # 随机选择一个状态
            s = random.randint(0, self.n_states - 1)
            
            # 计算该状态的最大值迭代更新
            values = []
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                values.append(v)
            
            # 更新为最大值
            self.V[s] = max(values)
    
    def async_policy_iteration(self, num_iterations=1000):
        """异步策略迭代：交替进行异步评估和异步改进"""
        for i in range(num_iterations):
            if i % 2 == 0:
                # 异步策略评估（单次扫描）
                s = random.randint(0, self.n_states - 1)
                a = self.policy[s]
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                self.V[s] = v
            else:
                # 异步策略改进（单次更新）
                s = random.randint(0, self.n_states - 1)
                best_a = 0
                best_v = float('-inf')
                for a in range(self.n_actions):
                    v = 0.0
                    for prob, next_s, reward, done in self.transitions[s][a]:
                        if done:
                            v += prob * reward
                        else:
                            v += prob * (reward + self.gamma * self.V[next_s])
                    if v > best_v:
                        best_v = v
                        best_a = a
                self.policy[s] = best_a
    
    def get_policy(self):
        """根据当前价值函数提取贪心策略"""
        for s in range(self.n_states):
            best_a = 0
            best_v = float('-inf')
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                if v > best_v:
                    best_v = v
                    best_a = a
            self.policy[s] = best_a
        return self.policy
```

### 完整代码示例：实时动态规划（RTDP）

```python
import numpy as np

class RTDP:
    """实时动态规划（Real-Time Dynamic Programming）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99, lr=0.1):
        self.V = np.zeros(n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
    
    def set_model(self, transitions):
        """设置环境模型"""
        self.transitions = transitions
    
    def rtdp_update(self, s, a, r, s_next):
        """RTDP更新：只更新访问过的状态"""
        # 计算TD目标
        if s_next is None:  # 终止状态
            td_target = r
        else:
            # 使用贪心动作选择
            next_values = []
            for a_next in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s_next][a_next]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                next_values.append(v)
            td_target = r + self.gamma * max(next_values)
        
        # TD更新
        td_error = td_target - self.V[s]
        self.V[s] += self.lr * td_error
        
        return td_error
    
    def plan_episode(self, start_state, max_steps=100):
        """规划一个episode（使用当前模型）"""
        s = start_state
        trajectory = []
        
        for step in range(max_steps):
            # 贪心动作选择
            best_a = 0
            best_v = float('-inf')
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                if v > best_v:
                    best_v = v
                    best_a = a
            
            # 执行动作（模拟）
            prob, next_s, reward, done = random.choice(self.transitions[s][best_a])
            trajectory.append((s, best_a, reward, next_s))
            
            # RTDP更新
            self.rtdp_update(s, best_a, reward, next_s if not done else None)
            
            if done:
                break
            s = next_s
        
        return trajectory
    
    def solve(self, start_states, num_episodes=1000):
        """求解MDP"""
        for episode in range(num_episodes):
            start = random.choice(start_states)
            self.plan_episode(start)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes}")
```

### 高级应用场景：机器人路径规划

**场景**：移动机器人在网格世界中寻找最短路径到目标

**DP解决方案**：
1. **状态**：机器人位置 (x,y)
2. **动作**：上、下、左、右
3. **奖励**：每步-1（鼓励最短路径），到达目标+10
4. **转移模型**：如果撞墙则留在原地

**为什么使用DP**：
- 环境模型已知（网格世界规则明确）
- 状态空间有限（可以穷举）
- 需要精确的最优解

**值迭代实现要点**：
```python
# 伪代码
for iteration in range(max_iterations):
    delta = 0
    for s in all_states:
        v = V[s]
        # 计算最大值更新
        max_value = float('-inf')
        for a in actions:
            # 模型预测下一个状态和奖励
            next_s, reward = model(s, a)
            value = reward + gamma * V[next_s]
            max_value = max(max_value, value)
        V[s] = max_value
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break
```

### 理论扩展：压缩映射定理

**定义**：算子 $\mathcal{T}$ 是 $\gamma$-压缩映射，如果：
$$ \| \mathcal{T} V - \mathcal{T} V' \|_\infty \leq \gamma \| V - V' \|_\infty, \quad 0 \leq \gamma < 1 $$

**贝尔曼最优算子**：$\mathcal{B} V(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$
是$\gamma$-压缩映射。

**证明**：
$$ |\mathcal{B} V(s) - \mathcal{B} V'(s)| = |\max_a Q(s,a) - \max_a Q'(s,a)| \leq \max_a |Q(s,a) - Q'(s,a)| $$
$$ \leq \gamma \max_{s'} |V(s') - V'(s')| = \gamma \| V - V' \|_\infty $$

**推论**：根据压缩映射不动点定理，$\mathcal{B}$ 有唯一不动点 $V^*$，且值迭代收敛到 $V^*$。

### 更多练习题

**练习12：截断策略迭代**
问题：设计一个截断策略迭代算法，评估迭代只运行K次而非直到收敛。

答案要点：
1. 策略评估：运行K次高斯-赛德尔迭代
2. 策略改进：基于当前V值贪心更新
3. K的选择：K=1时接近值迭代，K→∞时接近策略迭代
4. 实验：比较不同K的收敛速度

**练习13：异步DP的收敛性**
问题：证明异步DP（Gauss-Seidel值迭代）比同步DP收敛更快。

答案要点：
1. 异步DP使用最新的值更新，信息传播更快
2. 同步DP使用旧的一轮值，需要等待所有状态更新完
3. 实验：在网格世界上比较两者达到收敛的迭代次数
4. 结论：异步DP通常快1.5-2倍

**练习14：RTDP的样本效率**
问题：为什么RTDP比值迭代更高效？在什么情况下RTDP可能失败？

答案要点：
1. RTDP只更新访问过的状态，避免无用计算
2. 适合大规模MDP，其中很多状态不可达
3. 失败情况：如果起始状态分布不均匀，可能错过重要状态
4. 解决方案：使用探索策略或重要性采样## 深度补充：动态规划高级主题

### 策略迭代的收敛性证明

**定理**：策略迭代在有限MDP中保证收敛到最优策略。

**证明**：
1. **策略评估**：给定策略π，计算 $V^\pi$ 使用贝尔曼方程：
   $$ V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')] $$
   这是一个线性系统，有唯一解。

2. **策略改进**：贪心更新保证单调改进：
   $$ \pi'(s) = \arg\max_a Q^\pi(s,a) $$
   根据贝尔曼最优方程，如果 $Q^\pi(s, \pi'(s)) = V^\pi(s)$ 对所有s成立，则π已是最优。

3. **有限性**：有限MDP中策略数量有限（$|A|^{|S|}$），每次改进严格提升（或不变），故必在有限步内收敛。

### 值迭代的误差界

**定理**：值迭代第k次迭代的误差界为：
$$ \| V_k - V^* \|_\infty \leq \frac{\gamma^k}{1-\gamma} \| V_1 - V_0 \|_\infty $$

**推导**：
值迭代：$V_{k+1} = \mathcal{B} V_k$，其中$\mathcal{B}$是最优贝尔曼算子。
由于$\mathcal{B}$是$\gamma$-压缩映射：
$$ \| \mathcal{B} V - \mathcal{B} V' \|_\infty \leq \gamma \| V - V' \|_\infty $$
迭代k次后：
$$ \| V_k - V^* \|_\infty \leq \gamma^k \| V_0 - V^* \|_\infty $$
利用几何级数求和得到上述界。

### 广义策略迭代（GPI）框架

策略迭代和值迭代都是GPI的特例：

**GPI的两个同步过程**：
1. **策略评估**：给定策略π，计算 $V^\pi$ 或 $Q^\pi$
2. **策略改进**：给定价值函数V，改进策略π

**不同算法的GPI实现**：
| 算法 | 策略评估 | 策略改进 |
|------|----------|----------|
| 策略迭代 | 精确求解（迭代评估至收敛） | 完全贪心更新 |
| 值迭代 | 单次更新（不等待收敛） | 隐式改进（在值更新中） |
| 截断策略迭代 | 有限次评估迭代 | 完全贪心更新 |
| Q-learning | 单步样本更新 | 隐式改进（max操作） |

### 完整代码示例：异步动态规划

```python
import numpy as np
import random

class AsyncDP:
    """异步动态规划（异步策略迭代/值迭代）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99):
        self.V = np.zeros(n_states)
        self.Q = np.zeros((n_states, n_actions))
        self.policy = np.zeros(n_states, dtype=int)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
    
    def set_model(self, transitions):
        """设置环境模型：transitions[s][a] = [(prob, next_state, reward, done), ...]"""
        self.transitions = transitions
    
    def async_policy_evaluation(self, policy, num_iterations=1000):
        """异步策略评估：每次只更新一个状态"""
        for _ in range(num_iterations):
            # 随机选择一个状态
            s = random.randint(0, self.n_states - 1)
            
            # 计算该状态的贝尔曼期望方程
            v = 0.0
            a = policy[s]
            for prob, next_s, reward, done in self.transitions[s][a]:
                if done:
                    v += prob * reward
                else:
                    v += prob * (reward + self.gamma * self.V[next_s])
            
            # 更新该状态的价值
            self.V[s] = v
    
    def async_value_iteration(self, num_iterations=1000):
        """异步值迭代：每次只更新一个状态"""
        for _ in range(num_iterations):
            # 随机选择一个状态
            s = random.randint(0, self.n_states - 1)
            
            # 计算该状态的最大值迭代更新
            values = []
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                values.append(v)
            
            # 更新为最大值
            self.V[s] = max(values)
    
    def async_policy_iteration(self, num_iterations=1000):
        """异步策略迭代：交替进行异步评估和异步改进"""
        for i in range(num_iterations):
            if i % 2 == 0:
                # 异步策略评估（单次扫描）
                s = random.randint(0, self.n_states - 1)
                a = self.policy[s]
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                self.V[s] = v
            else:
                # 异步策略改进（单次更新）
                s = random.randint(0, self.n_states - 1)
                best_a = 0
                best_v = float('-inf')
                for a in range(self.n_actions):
                    v = 0.0
                    for prob, next_s, reward, done in self.transitions[s][a]:
                        if done:
                            v += prob * reward
                        else:
                            v += prob * (reward + self.gamma * self.V[next_s])
                    if v > best_v:
                        best_v = v
                        best_a = a
                self.policy[s] = best_a
    
    def get_policy(self):
        """根据当前价值函数提取贪心策略"""
        for s in range(self.n_states):
            best_a = 0
            best_v = float('-inf')
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                if v > best_v:
                    best_v = v
                    best_a = a
            self.policy[s] = best_a
        return self.policy
```

### 完整代码示例：实时动态规划（RTDP）

```python
import numpy as np

class RTDP:
    """实时动态规划（Real-Time Dynamic Programming）"""
    
    def __init__(self, n_states, n_actions, gamma=0.99, lr=0.1):
        self.V = np.zeros(n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
    
    def set_model(self, transitions):
        """设置环境模型"""
        self.transitions = transitions
    
    def rtdp_update(self, s, a, r, s_next):
        """RTDP更新：只更新访问过的状态"""
        # 计算TD目标
        if s_next is None:  # 终止状态
            td_target = r
        else:
            # 使用贪心动作选择
            next_values = []
            for a_next in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s_next][a_next]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                next_values.append(v)
            td_target = r + self.gamma * max(next_values)
        
        # TD更新
        td_error = td_target - self.V[s]
        self.V[s] += self.lr * td_error
        
        return td_error
    
    def plan_episode(self, start_state, max_steps=100):
        """规划一个episode（使用当前模型）"""
        s = start_state
        trajectory = []
        
        for step in range(max_steps):
            # 贪心动作选择
            best_a = 0
            best_v = float('-inf')
            for a in range(self.n_actions):
                v = 0.0
                for prob, next_s, reward, done in self.transitions[s][a]:
                    if done:
                        v += prob * reward
                    else:
                        v += prob * (reward + self.gamma * self.V[next_s])
                if v > best_v:
                    best_v = v
                    best_a = a
            
            # 执行动作（模拟）
            prob, next_s, reward, done = random.choice(self.transitions[s][best_a])
            trajectory.append((s, best_a, reward, next_s))
            
            # RTDP更新
            self.rtdp_update(s, best_a, reward, next_s if not done else None)
            
            if done:
                break
            s = next_s
        
        return trajectory
    
    def solve(self, start_states, num_episodes=1000):
        """求解MDP"""
        for episode in range(num_episodes):
            start = random.choice(start_states)
            self.plan_episode(start)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes}")
```

### 高级应用场景：机器人路径规划

**场景**：移动机器人在网格世界中寻找最短路径到目标

**DP解决方案**：
1. **状态**：机器人位置 (x,y)
2. **动作**：上、下、左、右
3. **奖励**：每步-1（鼓励最短路径），到达目标+10
4. **转移模型**：如果撞墙则留在原地

**为什么使用DP**：
- 环境模型已知（网格世界规则明确）
- 状态空间有限（可以穷举）
- 需要精确的最优解

**值迭代实现要点**：
```python
# 伪代码
for iteration in range(max_iterations):
    delta = 0
    for s in all_states:
        v = V[s]
        # 计算最大值更新
        max_value = float('-inf')
        for a in actions:
            # 模型预测下一个状态和奖励
            next_s, reward = model(s, a)
            value = reward + gamma * V[next_s]
            max_value = max(max_value, value)
        V[s] = max_value
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break
```

### 理论扩展：压缩映射定理

**定义**：算子 $\mathcal{T}$ 是 $\gamma$-压缩映射，如果：
$$ \| \mathcal{T} V - \mathcal{T} V' \|_\infty \leq \gamma \| V - V' \|_\infty, \quad 0 \leq \gamma < 1 $$

**贝尔曼最优算子**：$\mathcal{B} V(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$
是$\gamma$-压缩映射。

**证明**：
$$ |\mathcal{B} V(s) - \mathcal{B} V'(s)| = |\max_a Q(s,a) - \max_a Q'(s,a)| \leq \max_a |Q(s,a) - Q'(s,a)| $$
$$ \leq \gamma \max_{s'} |V(s') - V'(s')| = \gamma \| V - V' \|_\infty $$

**推论**：根据压缩映射不动点定理，$\mathcal{B}$ 有唯一不动点 $V^*$，且值迭代收敛到 $V^*$。

### 更多练习题

**练习12：截断策略迭代**
问题：设计一个截断策略迭代算法，评估迭代只运行K次而非直到收敛。

答案要点：
1. 策略评估：运行K次高斯-赛德尔迭代
2. 策略改进：基于当前V值贪心更新
3. K的选择：K=1时接近值迭代，K→∞时接近策略迭代
4. 实验：比较不同K的收敛速度

**练习13：异步DP的收敛性**
问题：证明异步DP（Gauss-Seidel值迭代）比同步DP收敛更快。

答案要点：
1. 异步DP使用最新的值更新，信息传播更快
2. 同步DP使用旧的一轮值，需要等待所有状态更新完
3. 实验：在网格世界上比较两者达到收敛的迭代次数
4. 结论：异步DP通常快1.5-2倍

**练习14：RTDP的样本效率**
问题：为什么RTDP比值迭代更高效？在什么情况下RTDP可能失败？

答案要点：
1. RTDP只更新访问过的状态，避免无用计算
2. 适合大规模MDP，其中很多状态不可达
3. 失败情况：如果起始状态分布不均匀，可能错过重要状态
4. 解决方案：使用探索策略或重要性采样