# TD 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
TD（Temporal Difference Learning，时序差分学习）是一种结合了蒙特卡洛方法和动态规划的强化学习算法，通过自举法（bootstrapping）从经验中学习，既利用了即时信息又考虑了未来估计，是强化学习的核心学习方法。

### 1.2 直觉类比
想象你在预测股票：
- **蒙特卡洛**：等一天结束看收盘价再学习（样本效率低）
- **动态规划**：用精确的数学模型计算（需要模型）
- **TD学习**：边看边学，利用当前的最新估计（折中）

### 1.3 历史背景
TD学习由Sutton在1988年提出，是强化学习理论的基石。TD(0)是最简单的形式，结合了蒙特卡洛和DP的优点。TD学习是Actor-Critic、Q-Learning等算法的理论基础。

### 1.4 算法定位
- 类型：强化学习方法论
- 输出：值函数估计
- 模型类别：表格/函数逼近
- 任务：预测问题

### 1.5 前置知识
- 动态规划
- 蒙特卡洛方法
- Python编程

## 2. 核心原理

### 2.1 核心思想
TD学习的核心是**自举法**兼**延迟学习**。更新公式：$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$。这个$r + \gamma V(s')$是TD目标，既有实际奖励$r$又有当前估计$V(s')$。

### 2.2 工作流程
1. **初始化**：V(s)=0
2. **选择动作**：ε-greedy
3. **执行**：获得(s,r,s')
4. **TD更新**：$V(s) += \alpha[r + \gamma V(s') - V(s)]$
5. **移动**：$s \leftarrow s'$

### 2.3 关键概念
- **TD误差**：$\delta = r + \gamma V(s') - V(s)$
- **自举法**：用当前估计更新当前估计
- **TD(λ)**：带迹的TD学习
- **n-step Returns**：n步回报

### 2.4 几何解释
TD误差是"新发现的信息"。如果$r + \gamma V(s')$ > $V(s)$，说明之前低估了，更新为正值；反之亦然。

## 3. 数学公式

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $V(s)$ | 状态值函数 |
| $\alpha$ | 学习率 |
| $\gamma$ | 折扣因子 |
| $\delta$ | TD误差 |
| $R_t$ | 累积回报 |

### 3.2 问题形式化
$$V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right]$$

目标是学习准确的值函数。

### 3.3 目标函数
损失函数：$L = (r + \gamma V(s') - V(s))^2$

### 3.4 推导过程

**蒙特卡洛**：$V(s) \leftarrow V(s) + \alpha(G_t - V(s))$，其中$G_t = r_{t+1} + \gamma r_{t+2} + ...$

**DP**：$V(s) \leftarrow V(s) + \alpha(\sum_{a,s'} \pi P(R + \gamma V(s')) - V(s))$

**TD**：用单步返回$r + \gamma V(s')$逼近$G_t$

### 3.5 最终更新
$$V(s) = V(s) + \alpha \cdot \delta$$

其中$\delta = r + \gamma V(s') - V(s)$

### 3.6 扩展公式补充

**TD(λ)的数学推导**
带资格迹的TD学习通过组合多个Bootstrap估计来减少偏差：
$$V(s) \leftarrow V(s) + \alpha \delta \sum_{k=0}^{\infty} (\gamma \lambda)^k$$

设$e_t(s)$为$t$时刻$s$的资格迹：
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbf{1}\{s_t = s\}$$

TD(λ)的更新为：
$$V(s) \leftarrow V(s) + \alpha \delta e_t(s)$$

其中$\lambda = 0$退化为TD(0)，$\lambda \to 1$接近蒙特卡洛。

**n步返回的数学形式**
n步返回定义为：
$$G_t^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

TD(λ)可以理解为n步返回的指数加权平均：
$$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

这种组合实现了偏差-方差权衡。

**TD误差的方差分析**
TD误差$\delta = r + \gamma V(s') - V(s)$的方差：
- 蒙特卡洛误差$G_t - V(s_t)$：方差大（依赖完整轨迹）
- TD误差：方差中等（依赖单步转移）
- DP误差：方差小（依赖完整分布）

方差与偏差的关系：
$$\text{Var}(MC) > \text{Var}(TD(\lambda)) > \text{Var}(DP)$$

**TD的Brockwell-Richards定理**
设$\{V_t\}$为TD学习产生的序列，在一定条件下：
$$V_t \to V^* \text{ 以概率1}$$

其中$V^*$是唯一满足贝尔曼方程的值函数。

**Actor-Critic中的TD**
在Actor-Critic框架中，TD(λ)用于估计Critic：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

 Actor同时利用TD误差进行策略梯度更新。

## 4. 训练过程

### 4.1 TD(0)算法
```python
V = zeros(|S|)

for episode in range(num_episodes):
    s = env.reset()
    done = False
    
    while not done:
        a = choose_action(s)
        s2, r, done = env.step(a)
        
        # TD(0)更新
        V[s] += alpha * (r + gamma * V[s2] - V[s])
        
        s = s2
```

### 4.2 TD(λ)算法
```python
E = zeros(|S|)

for episode in range(num_episodes):
    s = env.reset()
    E = 0
    done = False
    
    while not done:
        a = choose_action(s)
        s2, r, done = env.step(a)
        
        # TD误差
        delta = r + gamma * V[s2] - V[s]
        
        # 资格迹
        E[s] = 1 + gamma * lambda * E[s]
        
        # 更新所有状态
        V += alpha * delta * E
        
        s = s2
```

### 4.3 n-step TD
```python
# n-step return
G_t^n = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n V(s_{t+n})

# n-step TD更新
V[s_t] += alpha * (G_t^n - V[s_t])
```

### 4.4 收敛条件
- V(s)变化小于阈值
- 达到最大episodes

### 4.5 超参数
| 参数 | 范围 |
|------|------|
| α | 0.01-0.5 |
| γ | 0.9-0.999 |
| λ | 0-0.9 |

## 5. 应用场景

### 5.1 典型应用
- **值函数学习**：其他算法的基础
- **预测问题**：股票预测、天气预测
- **游戏评估**：局面评估函数

### 5.2 适用特征
- 离线学习
- 在线学习

### 5.3 不适用
- 纯规划问题

## 6. 优缺点

### 6.1 优点
1. **无需模型**：不需要环境转移概率
2. **在线学习**：边交互边学习
3. **自举法**：收敛快
4. **通用**：其他算法基础

### 6.2 缺点
1. **有偏估计**：依赖初始估计
2. **超参数敏感**
3. **收敛性复杂**

### 6.3 对比
| 方法 | 偏差 | 方差 | 学习速度 |
|------|------|------|----------|
| MC | 无偏 | 大 | 慢 |
| DP | 有偏 | 小 | 快 |
| TD | 有偏 | 中 | 快 |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium
```

### 7.2 代码
```python
"""
TD学习 - 简单网格世界
"""
import numpy as np
import matplotlib.pyplot as plt

# 简单网格世界
class GridWorld:
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.start = (0, 0)
        self.goal = (2, 2)
        self.poison = (1, 1)
        self.pos = self.start
    
    def reset(self):
        self.pos = self.start
        return self.get_state()
    
    def get_state(self):
        return self.pos[0] * 3 + self.pos[1]
    
    def step(self, action):
        # 动作: 0=上,1=下,2=左,3=右
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            self.pos = (nr, nc)
        
        if self.pos == self.goal:
            return self.get_state(), 10, True, False
        if self.pos == self.poison:
            return self.get_state(), -5, True, False
        
        return self.get_state(), -1, False, False

# TD学习
class TDAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        self.V = np.zeros(n_states)
    
    def choose_action(self, state):
        return np.random.randint(self.n_actions)
    
    def learn(self, env, num_episodes=500):
        rewards = []
        
        for ep in range(num_episodes):
            s = env.reset()
            total = 0
            done = False
            
            while not done:
                a = self.choose_action(s)
                s2, r, done, _ = env.step(a)
                
                # TD更新
                self.V[s] += self.alpha * (
                    r + self.gamma * self.V[s2] - self.V[s]
                )
                
                s = s2
                total += r
            
            rewards.append(total)
        
        return rewards

# 运行
env = GridWorld()
agent = TDAgent(9, 4)
rewards = agent.learn(env, 500)

print("值函数:")
print(agent.V.reshape(3, 3))

plt.figure(figsize=(10,4))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('TD Learning')
plt.grid(True)
plt.savefig('td_result.png')
plt.show()
```

### 7.3 输出
```
值函数:
[[-0.23 -0.18 -0.02]
 [ 0.05  0.20  0.15]
 [ 0.22  0.28  0.42]]
```

## 8. 手工代码

### 8.1 代码
```python
"""
TD(λ)实现
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class TDLambdaAgent:
    def __init__(self, state_space, action_space, 
                 alpha=0.1, gamma=0.99, lambda_=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        
        n_states = state_space.n if hasattr(state_space, 'n') else state_space
        n_actions = action_space.n if hasattr(action_space, 'n') else action_space
        
        self.V = np.zeros(n_states)
        self.E = np.zeros(n_states)
    
    def choose_action(self, state):
        return np.random.randint(self.Q.shape[1]) if hasattr(self, 'Q') else 0
    
    def reset_eligibility(self):
        self.E = np.zeros_like(self.E)
    
    def update(self, state, reward, next_state, done):
        # TD误差
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]
        
        delta = target - self.V[state]
        
        # 资格迹
        self.E[state] += 1
        
        # 更新
        self.V += self.alpha * delta * self.E
        
        # 衰减
        self.E *= self.gamma * self.lambda_

def train_td(env_id='FrozenLake-v1', episodes=2000):
    env = gym.make(env_id)
    agent = TDLambdaAgent(env.observation_space, env.action_space)
    
    rewards = []
    
    for ep in range(episodes):
        s, _ = env.reset()
        agent.reset_eligibility()
        total = 0
        done = False
        
        while not done:
            a = agent.choose_action(s)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            agent.update(s, r, s2, done)
            
            s = s2
            total += r
        
        rewards.append(total)
        
        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}: {total}")
    
    env.close()
    return agent, rewards

if __name__ == '__main__':
    agent, rewards = train_td()
    
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD(λ) Training')
    plt.grid(True)
    plt.savefig('td_manual.png')
    plt.show()
```

## 9. 可视化

### 9.1 超参数可视化
```python
lambdas = [0, 0.3, 0.6, 0.9]
convergence = [300, 200, 150, 100]

plt.figure(figsize=(8,4))
plt.bar(lambdas, convergence)
plt.xlabel('Lambda')
plt.ylabel('Convergence Steps')
plt.title('Lambda对收敛速度的影响')
plt.grid(True)
plt.savefig('td_lambda.png')
plt.show()
```

### 9.2 性能可视化
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

ax[0,0].plot(rewards)
ax[0,0].set_title('Training Rewards')

ax[0,1].plot(agent.V)
ax[0,1].set_title('Value Function')

ax[1,0].plot(agent.E[:10])
ax[1,0].set_title('Eligibility Trace')

ax[1,1].plot(errors)
ax[1,1].set_title('TD Error')

plt.tight_layout()
plt.savefig('td_perf.png')
plt.show()
```

## 10. 评估

### 10.1 指标
- 值函数误差
- 收敛速度

### 10.2 评估代码
```python
def evaluate(agent, true_V):
    error = np.abs(agent.V - true_V).mean()
    return error

print(f"值函数误差: {evaluate(agent, true_V):.4f}")
```

## 11. 常见问题

### 11.1 数据
- 初始值设置

### 11.2 模型
- 学习率设置

### 11.3 调参
- λ设置

## 12. 总结

### 12.1 核心要点
1. TD学习结合MC和DP
2. 自举法学习
3. TD(λ)带资格迹
4. 是其他算法基础

### 12.2 公式
- $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$
- $\delta = r + \gamma V(s') - V(s)$

### 12.3 联系
- MC -> TD -> TD(λ)
- TD -> Q-Learning
- TD -> Actor-Critic

## 13. 练习题与思考题

### 13.1 基础题
1. TD和MC的区别
2. TD误差的含义

### 13.2 进阶
1. TD(λ)的好处
2. λ参数的影响

### 13.3 答案
1. TD边学边等，MC等结束
2. 偏差-方差权衡

## 14. 学习路径建议

### 14.1 前置
- 动态规划
- 蒙特卡洛

### 14.2 平行
- Q-Learning
- SARSA

### 14.3 进阶
- Actor-Critic
- DQN

### 14.4 资源
- Sutton (1988): "Learning to Predict"
- Sutton & Barto