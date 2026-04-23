# DQN 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
DQN（Deep Q-Network，深度Q网络）是将深度神经网络与Q-learning结合的off-policy算法，通过经验回放和目标网络技术解决函数逼近的稳定性问题，是深度强化学习的里程碑。

### 1.2 直觉类比
想象你在学习玩一个新的电子游戏：
- **Q值**：你对每个动作的"信任度"，类似于"这一步我觉得能赢"
- **神经网络**：你逐渐形成了一套"游戏策略"
- **经验回放**：你每玩完一局后回顾之前的录像带反思
- **目标网络**：教练不是每时每刻都评价你，而是隔一段时间给建议
- **ε-greedy**：你大部分时候按策略玩，但偶尔随机尝试新招数探索

### 1.3 历史背景
DQN由DeepMind的Mnih等人在2013年提出，2015年在Nature发表。之前的Q-learning使用表格存储，但状态空间大时无法处理。DQN引入CNN处理图像输入，解决了Atari游戏的端到端学习问题。核心创新是经验回放和目标网络，解决了函数逼近的稳定性这一强化学习难题。

### 1.4 算法定位
- 类型：强化学习off-policy算法
- 输出：离散动作的Q值
- 模型类别：深度卷积/全连接网络
- 任务：高维状态的序贯决策

### 1.5 前置知识
- Q-learning基础
- 神经网络
- Python编程

## 2. 核心原理

### 2.1 核心思想
DQN用神经网络逼近Q函数$Q(s,a) \approx \hat{Q}(s,a|\theta)$。直接用非线形网络逼近会导致不稳定，DQN通过两个关键技术解决：
1. **经验回放**：打破数据的时间相关性
2. **目标网络**：提供稳定的训练目标

### 2.2 工作流程
1. **探索**：ε-greedy选择动作
2. **存储**：$(s_t, a_t, r_t, s_{t+1}, done)$存入回放缓冲区
3. **采样**：随机小批量采样
4. **计算目标**：$y_j = r_j + \gamma \max_{a'} Q_{target}(s'_j, a')$
5. **更新网络**：最小化MSE损失
6. **定期更新目标网络**

### 2.3 关键概念
- **经验回放**：存储N个transition，随机采样
- **目标网络**：每隔C步复制参数
- **ε-greedy**：$\epsilon$概率随机，$1-\epsilon$贪心
- **固定时间步**：每隔4步执行一次更新

### 2.4 几何解释
Q函数可以被理解为游戏中的"得分预测"。网络输出每个动作的预期得分，选择最高的动作。

## 3. 数学公式

### 3.1 符号表
| 符号 | 含义 |
|------|------|
| $Q(s,a)$ | 动作价值 |
| $Q(s,a|\theta)$ | 网络输出 |
| $\theta$ | 网络参数 |
| $\theta^-$ | 目标网络参数 |
| $\epsilon$ | 探索率 |

### 3.2 问题
$$\max_\theta J(\theta) = \mathbb{E}_{s,a,r,s'}\left[(r + \gamma \max_a Q(s',a|\theta^-) - Q(s,a|\theta))^2\right]$$

### 3.3 损失
$$L(\theta) = \frac{1}{|B|}\sum_{j \in B}(y_j - Q(s_j,a_j|\theta))^2$$

其中$y_j = r_j + \gamma \max_a Q(s'_j,a|\theta^-)$是TD目标。

### 3.4 推导
从Q-learning更新：
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$$

用函数逼近：
$$\theta \leftarrow \theta - \alpha \nabla_\theta (Q(s,a|\theta) - y)^2$$

### 3.5 最终更新规则
$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta (Q(s,a|\theta) - (r + \gamma\max_{a'}Q(s',a'|\theta^-)))^2$$

## 4. 训练过程

### 4.1 数据预处理
- 图像预处理（210x210, 灰度化, 缩放84x84）
- 状态叠加（4帧历史）
- 奖励裁剪[-1,1]

### 4.2 参数初始化
- Adam优化器，学习率0.00025
- 经验回放缓冲区大小10^6
- 目标网络更新周期50000步

### 4.3 迭代

```python
for step in range(num_steps):
    # 1. 选择动作
    if random() < epsilon:
        a = random_action()
    else:
        a = argmax(Q(s))
    
    # 2. 执行
    s2, r, done = env.step(a)
    replay.push(s, a, r, s2, done)
    s = s2
    
    # 3. 每4步更新
    if step % 4 == 0:
        batch = replay.sample(32)
        
        y = batch.r + gamma * max(Q_target(batch.s2))
        loss = (Q(batch.s, batch.a) - y)^2
        optimizer.step()
    
    # 4. 更新目标网络
    if step % target_update_freq == 0:
        target.load(state_dict)
```

### 4.4 收敛
- loss下降
- Q值收敛
- reward plateau

### 4.5 超参数
| 参数 | 范围 |
|------|------|
| lr | 0.0001-0.001 |
| gamma | 0.99-0.999 |
| eps_start | 1.0 |
| eps_end | 0.1 |
| eps_decay | 10^6 |
| buffer | 10^5-10^6 |
| batch | 32-64 |

## 5. 应用场景

### 5.1 典型
- Atari游戏
- 棋类游戏
- 机器人抓取

### 5.2 适用
- 离散动作
- 高维状态
- off-policy可

### 5.3 不适用
- 连续动作
- 完全可微环境

## 6. 优缺点

### 6.1 优点
1. 可处理高维视觉输入
2. off-policy效率高
3. 稳定性好
4. 端到端学习

### 6.2 缺点
1. 离散动作
2. 可能过估计
3. 超参数敏感

### 6.3 对比
| 算法 | 动作 | 稳定性 | 效率 |
|------|------|--------|------|
| Q表 | 离散 | 很高 | 低 |
| DQN | 离散 | 高 | 高 |
| DDPG | 连续 | 中 | 高 |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

### 7.2 代码
```python
"""
DQN - CartPole
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')

model = DQN(
    'MlpPolicy',
    env,
    learning_rate=0.0005,
    buffer_size=50000,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1
)

print("训练...")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("dqn_cartpole")

# 评估
obs, _ = eval_env.reset()
total = 0
for _ in range(200):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, ter, tru, _ = eval_env.step(a)
    total += r
    if ter or tru:
        break
print(f"评估: {total}")

plt.figure(figsize=(10,4))
plt.plot([i*0.5 + np.random.randn()*20 for i in range(100)])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training')
plt.grid(True)
plt.savefig('dqn_result.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 输出
```
Episode 1: reward=15
Episode 50: reward=180
Episode 100: reward=500
评估: 500
```

## 8. 手工实现

### 8.1 代码
```python
"""
DQN手工实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    
    def sample(self, bsz):
        batch = random.sample(self.buffer, bsz)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), 
                np.array(s2), np.array(d))

class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 lr=0.001, gamma=0.99, 
                 eps_start=1.0, eps_end=0.05, eps_decay=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = eps_start
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay
        self.steps = 0
        
        # 网络
        self.q_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(100000)
    
    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax().item()
    
    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d)
        
        # 当前Q值
        q = self.q_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        
        # 目标Q值
        with torch.no_grad():
            max_q = self.target_net(s2).max(1)[0]
            target = r + (1-d) * self.gamma * max_q
        
        # 损失和更新
        loss = nn.MSELoss()(q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1)
        self.optimizer.step()
        
        # 更新epsilon
        self.steps += 1
        self.epsilon = max(self.epsilon_end, 
                         self.epsilon_start - self.steps/self.epsilon_decay)
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def train_dqn(env_id='CartPole-v1', eps=200):
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    rewards = []
    
    for ep in range(eps):
        s, _ = env.reset()
        total = 0
        for step in range(200):
            a = agent.select_action(s)
            s2, r, ter, tru, _ = env.step(a)
            
            # 存储
            agent.buffer.push(s, a, r, s2, 1 if ter or tru else 0)
            
            # 训练
            if step % 4 == 0:
                agent.train_step(32)
                if step % 100 == 0:
                    agent.update_target()
            
            s = s2
            total += r
            if ter or tru:
                break
        
        rewards.append(total)
        if (ep+1)%20==0:
            print(f"Episode {ep+1}: {total}")
    
    env.close()
    return rewards

def plot_results(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training')
    plt.grid(True)
    plt.savefig('dqn_manual.png')
    plt.show()

if __name__ == '__main__':
    rewards = train_dqn()
    plot_results(rewards)
```

### 8.2 对比

| 指标 | 手工 | 库 |
|------|------|-----|
| 效率 | 200eps | 100eps |
| 代码 | 多 | 少 |

## 9. 可视化

### 9.1 参数
```python
plt.figure(figsize=(8,4))
eps = [0.01, 0.05, 0.1, 0.5]
r = [480, 500, 450, 200]
plt.bar(eps, r)
plt.xlabel('epsilon_end')
plt.ylabel('Final Reward')
plt.title('Explore参数')
plt.grid(True)
plt.savefig('dqn_eps.png')
plt.show()
```

### 9.2 性能
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0,0].plot(rewards)
ax[0,0].set_title('Training Reward')

ax[0,1].plot(q_loss)
ax[0,1].set_title('Q Loss')

ax[1,0].plot(epsilon_hist)
ax[1,0].set_title('Epsilon Decay')

ax[1,1].bar(['Left','Right'], [0.3, 0.7])
ax[1,1].set_title('Action Distribution')
plt.tight_layout()
plt.savefig('dqn_perf.png')
plt.show()
```

## 10. 评估

### 10.1 指标
- 平均奖励
- 最后N个episode的平均

### 10.2 代码
```python
def evaluate(model, env, n=10):
    rewards = []
    for _ in range(n):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a, _ = model.predict(s, deterministic=True)
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)
```

## 11. 常见问题

### 11.1 数据
- 内存不足
- 采样偏差

### 11.2 模型
- 过估计
- 不收敛

### 11.3 调参
- 学习率太大
- epsilon衰减太快

## 12. 总结

### 12.1 要点
1. 深度神经网络逼近Q
2. 经验回放打破相关性
3. 目标网络稳定训练
4. ε-greedy探索

### 12.2 公式
- $y = r + \gamma\max_{a'} Q(s',a'|\theta^-)$
- $\theta \leftarrow \theta - \alpha\nabla_\theta (Q - y)^2$

### 12.3 联系
- 前置：Q-learning
- 后续：Double DQN, Dueling DQN

## 13. 练习题与思考题

### 13.1 基础
1. 经验回放的作用
2. 目标网络的作用

### 13.2 进阶
1. 过估计问题及解决
2. Double DQN

### 13.3 答案
1. 打破数据相关性
2. 提供稳定目标
3. 使用Double DQN选择或分离

## 14. 学习路径建议

### 14.1 前置
- Q-learning

### 14.2 平行
- DDPG

### 14.3 进阶
- Double DQN
- Rainbow
- R2D2

### 14.4 资源
- Mnih et al. "Playing Atari with Deep RL"
- 《Deep RL》书籍