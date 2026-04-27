# DQN 学习文档

## 1. 算法基础认知

深度Q网络（Deep Q-Network, DQN）是将深度学习与强化学习结合的里程碑算法：
- 1989年：Chris Watkins提出Q-learning，奠定表格型Q-learning基础
- 2013年：DeepMind团队（Volodymyr Mnih等）提出DQN，在Atari游戏中达到人类水平
- 2015年：DeepMind在Nature发表改进版DQN，增加目标网络、经验回放等优化
- 2017年：Rainbow DQN整合多项改进，成为DQN的集大成版本
- 2020年：DQN扩展到视觉导航、机器人控制等多个领域

**详细发展历程**：

**第一阶段：理论基础（1980s-2000s）**
- 1989年：Q-learning论文发表，成为表格型RL的标准方法
- 1990s：Q-learning被应用于简单任务，如GridWorld
- 2000s：函数近似Q-learning的早期尝试（线性拟合、小神经网络）

**第二阶段：深度RL突破（2013-2015）**
- 2013年：Nature DQN论文发表，里程碑式成果：
  - 卷积网络处理高维状态（84×84图像）
  - 经验回放打破数据相关性
  - 目标网络稳定训练
  - 57个Atari游戏中有49个达到人类水平
- 2015年：改进版DQN：
  - Double DQN（解决过估计）
  - 像素裁剪（稳定训练）
  - 优先经验回放（提升样本效率）

**第三阶段：集成改进（2016-2019）**
- 2016年：Dueling DQN（分解Q值为V和A，加速收敛）
- 2017年：Rainbow（集成6大改进：DDQN + PER + Dueling + Multi-step + Distributional + Noisy Nets）
- 2018年：Ape-X（探索效率提升，更优的探索策略）
- 2019年：R2D2（残差网络DQN，解决深度网络训练难度）

**第四阶段：现代发展（2020-至今）**
- 分布式RL：Apex-DQN、Distributed Prioritized Experience Replay
- 离线RL：Conservative Q-Learning（CQL）、Batch Constraint Q-learning
- 多智能体：QMIX、Multi-agent DQN
- 实时应用：工业界DQN优化（AlphaGo、广告推荐）

### 1.2 生活类比
DQN的核心是**用深度网络近似Q值，通过经验学习最优策略**：
| 类比场景 | 状态 | 动作 | Q值含义 | 训练过程 |
|----------|------|------|----------|----------|
| 玩Atari游戏 | 当前游戏画面 | 上下左右等操作 | 当前画面下做动作的期望回报 | 玩游戏积累经验，更新网络 |
| 学骑自行车 | 当前平衡/速度 | 转动车把/踩踏板 | 当前状态下做动作的回报 | 练习摔倒积累经验，学会骑车 |
| 股票交易 | 当前股价/指标 | 买入/卖出/持有 | 当前状态下做动作的收益 | 交易积累经验，优化策略 |

### 1.3 算法定位
| 维度 | 定位说明 |
|------|----------|
| 学习范式 | 强化学习、深度强化学习 |
| 模型属性 | 模型无关（Model-Free） |
| 策略类型 | 离线策略（Off-Policy） |
| 价值函数 | Q值函数（动作价值） |
| 核心创新 | 经验回放、目标网络 |

### 1.4 学习前置清单
#### 数学基础
- 强化学习：MDP、Q-learning、贝尔曼方程
- 深度学习：神经网络、梯度下降、损失函数
- 概率论：期望、方差

#### 编程基础
- Python 3.9+ 基础语法
- PyTorch 框架（张量、网络定义、优化器）
- Gymnasium（强化学习环境，可选）

> 扩展阅读：Mnih 2015 Nature论文《Human-level control through deep reinforcement learning》

## 2. 核心原理
### 2.1 核心机制：深度Q-learning + 两大创新
DQN解决表格型Q-learning无法处理高维状态的问题，核心创新：
1. **经验回放（Experience Replay）**：将经验$(s,a,r,s')$存入回放池，随机采样训练，打破数据相关性
2. **目标网络（Target Network）**：维护一个滞后更新的目标网络，计算Q-target，稳定训练

#### 机制ASCII示意图
```
+-------------------+  动作a_t  +-------------------+  经验(s,a,r,s')  +-------------------+
|   环境（如Atari） | --------> |     DQN智能体      | ----------------> |  经验回放池      |
| 返回状态s_t+1,奖励r_t | <-------- | 当前Q网络Q(s,a)  | <---------------- | (随机采样batch)  |
+-------------------+          | 目标网络Q'(s,a)  |  更新Q网络       +-------------------+
                                +-------------------+
                                    |       ^
                                    +-------+
                                    计算损失，反向传播
```

### 2.2 相关算法对比
| 算法 | 是否深度 | 经验回放 | 目标网络 | 稳定性 |
|------|----------|----------|----------|--------|
| Q-learning | 否 | 无 | 无 | 低（高维状态） |
| DQN | 是 | 有 | 有 | 中 |
| DDQN | 是 | 有 | 有（解耦） | 高（缓解过估计） |
| SARSA | 否 | 无 | 无 | 低 |

### 2.3 工程经验
1. 奖励裁剪：将奖励裁剪到[-1, 1]，避免Q值爆炸
2. 帧堆叠：Atari场景堆叠4帧灰度图，捕捉运动信息
3. ε-贪婪探索：初始ε=1.0，指数衰减到0.1
4. 目标网络更新：每C步复制当前网络参数到目标网络，C=100~1000

### 2.4 几何直观解释
Q值$Q(s,a)$是在状态s下选择动作a的期望累积回报：
- 深度网络将高维状态（如图像）映射到Q值向量
- 经验回放让网络学习多样化的经验，避免过拟合到近期轨迹
- 目标网络提供稳定的Q-target，避免堆栈式更新导致训练崩溃

> 知识链接：与`Q-learning.md`、`DDQN.md`同属Q-learning系列算法

## 3. 数学公式与推导
### 3.1 符号表
| 符号 | 含义 | 维度/范围 |
|------|------|-----------|
| $s_t$ | t时刻的状态 | 高维向量/图像 |
| $a_t$ | t时刻的动作 | 离散动作空间$A$ |
| $r_t$ | t时刻的奖励 | 实数，裁剪到[-1,1] |
| $\\gamma$ | 折扣因子 | $0 \\leq \\gamma \\leq 1$，通常0.99 |
| $Q(s,a; \\theta)$ | 当前Q网络，参数$\\theta$ | 输出$|A|$维Q值向量 |
| $Q'(s,a; \\theta^-)$ | 目标Q网络，参数$\\theta^-$ | 输出$|A|$维Q值向量 |
| $\\mathcal{D}$ | 经验回放池 | 容量$N$，通常$1e5 \\sim 1e6$ |

### 3.2 核心公式推导
#### 贝尔曼最优方程
$$Q^*(s,a) = \\mathbb{E}_{s' \\sim P}[r + \\gamma \\max_{a'} Q^*(s',a') | s,a]$$

#### DQN损失函数（MSE）
对每个样本$(s,a,r,s') \\sim \\mathcal{D}$：
$$y = r + \\gamma \\max_{a'} Q'(s',a'; \\theta^-)$$
$$L(\\theta) = \\mathbb{E}_{(s,a,r,s')}[ (y - Q(s,a; \\theta))^2 ]$$

#### 梯度更新
$$\\theta \\leftarrow \\theta - \\alpha \\nabla_\\theta L(\\theta)$$

### 3.3 算法伪代码
```
初始化：当前Q网络Q(θ)，目标网络Q'(θ⁻)=Q(θ)，经验池D
初始化：ε=1.0，帧堆叠缓冲
for episode=1 to M:
    初始化状态s（堆叠4帧）
    while 未终止：
        以ε概率随机选动作a，否则a=argmax Q(s,a)
        执行a，得到r，s'
        将(s,a,r,s')存入D
        采样batch B从D
        计算y = r + γ max_a' Q'(s',a')
        计算损失L = MSE(y, Q(s,a))
        梯度下降更新θ
        每隔C步：θ⁻ = θ
        s = s'
        衰减ε
    end while
end for
```

### 3.4 收敛性说明
DQN无严格收敛证明，但实践中通过经验回放和目标网络可以稳定训练，在Atari等基准上达到人类水平。

> 扩展阅读：Mnih 2015论文中的收敛实验

## 4. 训练过程讲解
### 4.1 数据预处理（Atari场景）
1. 帧处理：RGB转灰度，resize到84x84
2. 帧堆叠：连续4帧堆叠成84x84x4的输入
3. 奖励裁剪：奖励裁剪到[-1, 1]
4. 帧最大池化：连续2帧取最大值，减少闪烁

### 4.2 参数初始化表
| 参数 | 推荐值（Atari） | 推荐值（CartPole） | 说明 |
|------|------------------|---------------------|------|
| 学习率α | 1e-4 | 1e-3 | Adam优化器默认1e-3，Atari需更小 |
| 折扣因子γ | 0.99 | 0.99 | 长期回报权重 |
| 回放池容量 | 1e6 | 1e5 | 存储经验的数量 |
| 批量大小B | 32 | 64 | 每次更新采样的样本数 |
| 目标更新频率C | 10000 | 100 | Atari每1e4步更新，CartPole每100步 |
| 初始ε | 1.0 | 1.0 | 初始完全探索 |
| 最终ε | 0.1 | 0.05 | 最终探索率 |
| ε衰减率 | 0.995 | 0.99 | 每步衰减 |

### 4.3 训练流程
1. 初始化网络、回放池、参数
2. 循环多个episode：
   a. 重置环境，获取初始状态
   b. 循环直到终止：
      i. ε-贪婪选择动作
      ii. 执行动作，获取奖励和下一个状态
      iii. 存储经验到回放池
      iv. 采样batch，计算损失，更新当前网络
      v. 定期更新目标网络
      vi. 衰减ε
3. 评估性能，保存模型

#### 工程技巧
- 使用帧堆叠缓冲：维护一个队列存储最近4帧
- 优先经验回放（PER）：根据TD误差采样，提升效率
- 梯度裁剪：将梯度范数裁剪到10，避免梯度爆炸

### 4.4 收敛与调试
#### 收敛条件
- 平均回报（最近100episode）不再上升
- 损失曲线趋于平稳
- 评估时平均回报达到基准水平（如CartPole-v1 > 475）

#### 常见问题调试
| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 损失不下降 | 学习率过高/过低 | 调整学习率（1e-3 ~ 1e-5） |
| Q值爆炸 | 奖励未裁剪 | 裁剪奖励到[-1,1] |
| 训练不稳定 | 目标网络更新频率过高 | 增大C（目标更新间隔） |
| 探索不足 | ε衰减过快 | 降低衰减率，提高最终ε |

## 5. 应用场景
### 5.1 完整应用案例
#### 案例1：Atari游戏智能体
- 状态：84x84x4堆叠灰度帧
- 动作：18个Atari操作（上下左右等）
- 奖励：游戏得分变化
- 目标：最大化总游戏得分

#### 案例2：CartPole平衡
- 状态：[小车位置，小车速度，杆角度，杆角速度]（4维）
- 动作：左移/右移（2维）
- 奖励：每步存活得1分
- 目标：最大化存活步数（>475步为成功）

#### 案例3：机器人导航
- 状态：激光雷达扫描、IMU数据（高维）
- 动作：前进/后退/左转/右转（4维）
- 奖励：到达目标+10，碰撞-1，每步-0.01
- 目标：最小化导航时间

#### 案例4：游戏AI（如王者荣耀）
- 状态：游戏画面、英雄状态（高维）
- 动作：技能释放、移动等（离散动作）
- 奖励：击杀+5，死亡-5，推塔+10
- 目标：最大化胜率

#### 案例5：推荐系统
- 状态：用户历史行为、当前上下文（高维）
- 动作：推荐商品ID（离散）
- 奖励：点击+1，购买+5，未点击0
- 目标：最大化总奖励

### 5.2 适用场景特征
| 特征 | 说明 |
|------|------|
| 高维状态空间 | 图像、传感器数据等，表格Q-learning无法处理 |
| 离散动作空间 | 动作为有限个离散选项 |
| 离线策略学习 | 可以用历史经验训练 |
| 延迟奖励 | 动作后很久才获得最终奖励 |

### 5.3 不适用场景与替代方案
| 不适用场景 | 问题 | 替代方案 |
|----------|------|----------|
| 连续动作空间 | DQN只能处理离散动作 | DDPG、PPO（连续动作） |
| 极高维状态 | 图像分辨率过高，训练慢 | 使用CNN降维，或PPO |
| 需要确定性策略 | DQN输出随机策略（ε-贪婪） | DDPG（确定性策略） |
| 样本效率要求极高 | DQN样本利用率低 | 模型基RL、离线RL |

## 6. 优缺点分析
### 6.1 优点
1. **处理高维状态**
   - 条件：搭配CNN/MLP等深度网络
   - 说明：图像、传感器数据等可直接输入
2. **离线策略学习，样本利用率高**
   - 条件：有经验回放机制
   - 说明：可以复用历史经验，打破数据相关性
3. **理论继承Q-learning**
   - 条件：满足MDP假设
   - 说明：Q-learning的理论基础支撑DQN
4. **在多个基准上达到人类水平**
   - 条件：Atari等游戏场景
   - 说明：DeepMind验证在57个Atari游戏上超过人类
5. **开源实现丰富**
   - 条件：PyTorch/TensorFlow都有成熟实现
   - 说明：快速复现和二次开发成本低

### 6.2 缺点
1. **Q值过估计（Overestimation Bias）**
   - 问题：max操作导致Q值高估，影响策略质量
   - 解决方案：使用DDQN（解耦目标计算）
2. **训练不稳定**
   - 问题：超参数敏感，容易训练崩溃
   - 解决方案：目标网络、经验回放、梯度裁剪
3. **样本效率低**
   - 问题：需要数百万步经验才能收敛
   - 解决方案：优先经验回放、模型基RL
4. **仅支持离散动作**
   - 问题：无法处理连续动作空间
   - 解决方案：使用DDPG、PPO等连续控制算法
5. **超参数敏感**
   - 问题：学习率、回放池大小等影响性能大
   - 解决方案：网格搜索，参考基准参数

### 6.3 算法对比
| 算法 | 动作空间 | 过估计 | 训练稳定性 | 样本效率 |
|------|----------|--------|------------|----------|
| Q-learning | 离散 | 有 | 低（高维） | 低 |
| DQN | 离散 | 有 | 中 | 中 |
| DDQN | 离散 | 缓解 | 高 | 中 |
| DDPG | 连续 | 有（Actor-Critic） | 中 | 中 |

## 7. 调库实现
### 7.1 完整代码（CartPole-v1，基于PyTorch）
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
import gymnasium as gym

class DQN(nn.Module):
    """DQN网络（MLP）"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """DQN智能体"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        # 网络初始化
        self.current_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()
        
        # 优化器和回放池
        self.optimizer = optim.Adam(self.current_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """ε-贪婪选择动作"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.current_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self):
        """更新当前网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        # 采样batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.current_net(states).gather(1, actions)
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (~dones)
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_net.parameters(), 10)  # 梯度裁剪
        self.optimizer.step()
        return loss.item()
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.current_net.state_dict())

def train_dqn(
    env_id: str = "CartPole-v1",
    num_episodes: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995
) -> DQNAgent:
    """训练DQN智能体"""
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 选择动作
            action = agent.select_action(state, epsilon)
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            # 更新网络
            loss = agent.update()
            # 更新目标网络
            if agent.steps_done % agent.target_update_freq == 0:
                agent.update_target_net()
            state = next_state
            total_reward += reward
            agent.steps_done += 1
        # 衰减epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        # 每50episode打印一次
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: 总奖励 {total_reward}, epsilon {epsilon:.3f}")
    env.close()
    return agent

if __name__ == "__main__":
    agent = train_dqn(num_episodes=500)
    # 评估
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, epsilon=0.0)  # 评估时不探索
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    print(f"评估总奖励：{total_reward}")
    env.close()
```

### 7.2 运行结果示例
```
Episode 50: 总奖励 42, epsilon 0.778
Episode 100: 总奖励 156, epsilon 0.605
Episode 150: 总奖励 289, epsilon 0.471
Episode 200: 总奖励 342, epsilon 0.367
Episode 250: 总奖励 421, epsilon 0.286
Episode 300: 总奖励 489, epsilon 0.222
Episode 350: 总奖励 500, epsilon 0.173
Episode 400: 总奖励 500, epsilon 0.135
Episode 450: 总奖励 500, epsilon 0.105
Episode 500: 总奖励 500, epsilon 0.082
评估总奖励：500
```

### 7.3 超参数说明
| 超参数 | 取值范围 | 推荐值（CartPole） | 影响 |
|--------|----------|---------------------|------|
| lr | 1e-5~1e-2 | 1e-3 | 学习率，过高不稳定，过低收敛慢 |
| gamma | 0.9~0.999 | 0.99 | 折扣因子，越大越关注长期回报 |
| batch_size | 32~256 | 64 | 批量大小，越大训练越稳定 |
| target_update_freq | 10~1000 | 100 | 目标网络更新频率，过高不稳定 |

### 7.4 工程经验
1. 训练时关闭渲染，评估时开启，提升速度
2. 保存最佳模型：当评估奖励超过阈值时保存
3. 使用TensorBoard记录损失、奖励曲线，监控训练

## 8. 手工代码实现
### 8.1 简化版DQN（仅用PyTorch核心功能）
```python
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class SimpleDQN:
    """简化版DQN，核心逻辑手写"""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 简单MLP网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.steps = 0
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.net(state).argmax().item()
    
    def update(self):
        if len(self.buffer) < 64:
            return
        batch = random.sample(self.buffer, 64)
        states = torch.FloatTensor([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.BoolTensor([x[4] for x in batch]).unsqueeze(1)
        
        current_q = self.net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (~dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return loss.item()
```

### 8.2 说明
简化版去掉了复杂封装，保留DQN核心逻辑，适合理解算法本质。

## 9. 可视化与结果理解
### 9.1 可视化代码（训练曲线）
```python
import matplotlib.pyplot as plt
from collections import deque

def plot_training_curves(rewards: List[float], losses: List[float]):
    """可视化训练曲线"""
    plt.figure(figsize=(12, 4))
    # 奖励曲线（滑动平均）
    plt.subplot(1, 2, 1)
    smoothed_rewards = deque(maxlen=50)
    avg_rewards = []
    for r in rewards:
        smoothed_rewards.append(r)
        avg_rewards.append(np.mean(smoothed_rewards))
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("平均奖励（50episode滑动）")
    plt.title("训练奖励曲线")
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("更新步数")
    plt.ylabel("MSE损失")
    plt.title("训练损失曲线")
    plt.tight_layout()
    plt.show()

# 训练时记录rewards和losses，传入函数即可
# plot_training_curves(rewards, losses)
```

### 9.2 结果解读
1. **奖励曲线**：应逐步上升，最终稳定在环境最优值附近（CartPole为500）
2. **损失曲线**：应逐步下降，最终趋于平稳
3. **收敛判断**：奖励曲线平稳，损失不再下降

#### 常见问题
- 奖励波动大：ε衰减过慢，或回放池容量太小
- 损失不下降：学习率过高，或网络结构不合理

## 10. 模型评估
### 10.1 评估指标
| 指标 | 含义 | 优化方向 |
|------|------|----------|
| 平均奖励 | 最近100episode的平均回报 | 最大化 |
| 最大奖励 | 单次episode的最大回报 | 最大化 |
| 收敛episode | 达到最优性能的episode数 | 最小化 |
| 评估胜率 | 与基线模型对比的胜率 | 最大化 |

### 10.2 评估代码
```python
def evaluate_dqn(agent: DQNAgent, env_id: str = "CartPole-v1", num_episodes: int = 20) -> float:
    """评估DQN智能体"""
    env = gym.make(env_id)
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # 不探索
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"评估平均奖励（{num_episodes}次）：{avg_reward:.2f}")
    return avg_reward

# 评估训练好的agent
# evaluate_dqn(agent)
```

### 10.3 标准指标值（CartPole-v1）
| 指标 | 合格值 | 优秀值 |
|------|--------|--------|
| 平均奖励（100episode） | ≥200 | ≥475 |
| 收敛episode | ≤500 | ≤200 |
| 评估胜率（vs随机） | ≥80% | ≥95% |

### 10.4 超参数调优
网格搜索学习率和批量大小：
```python
def tune_dqn_hyperparameters():
    """调优DQN超参数"""
    best_reward = -np.inf
    best_params = {}
    for lr in [1e-4, 1e-3, 1e-2]:
        for batch_size in [32, 64, 128]:
            # 简化训练，仅跑100episode
            agent = DQNAgent(state_dim=4, action_dim=2, lr=lr, batch_size=batch_size)
            # 这里省略训练过程，实际需训练后评估
            # reward = evaluate_dqn(agent)
            # if reward > best_reward: ...
    print(f"最优参数：{best_params}")
```

## 11. 常见问题与易错点
### 5.1 常见陷阱
1. **忘记更新目标网络**
   - 现象：训练不稳定，损失震荡
   - 原因：目标网络参数未更新，Q-target失真
   - 解决：定期复制当前网络参数到目标网络

2. **回放池容量过小**
   - 现象：训练过拟合，泛化能力差
   - 原因：经验多样性不足
   - 解决：增大回放池容量到1e5以上

3. **奖励未裁剪**
   - 现象：Q值爆炸，损失NaN
   - 原因：奖励范围过大（如Atari奖励-30~30）
   - 解决：裁剪奖励到[-1, 1]

4. **ε衰减过快**
   - 现象：过早停止探索，陷入局部最优
   - 原因：ε很快衰减到0.05以下
   - 解决：降低衰减率，提高最终ε

5. **梯度爆炸**
   - 现象：损失NaN，参数更新异常
   - 原因：梯度范数过大
   - 解决：梯度裁剪（范数裁剪到10）

### 11.2 调试技巧
1. 打印每步的Q值、损失，验证更新逻辑
2. 可视化回放池中的经验，检查分布是否合理
3. 对比不同超参数的训练曲线，选择最优组合

### 11.3 工程最佳实践
1. 每1000步保存一次模型checkpoint，避免训练崩溃丢失进度
2. 使用TensorBoard记录训练指标，实时监控
3. 优先经验回放（PER）：根据TD误差采样，提升样本效率

## 12. 学习总结
### 12.1 核心思想回顾
DQN的核心是用深度神经网络近似Q值，通过**经验回放**打破数据相关性，通过**目标网络**稳定训练，解决高维状态空间的强化学习问题。

#### 思维导图（ASCII）
```
                    DQN
                     |
         +-----------+-----------+
         |           |           |
     核心创新      应用场景      相关算法
         |           |           |
  经验回放+目标网络 Atari/机器人 Q-learning/DDQN
```

### 12.2 必记公式
1. DQN损失：$L = \\mathbb{E}[(r + \\gamma \\max_{a'} Q'(s',a') - Q(s,a))^2]$
2. 贝尔曼最优方程：$Q^*(s,a) = \\mathbb{E}[r + \\gamma \\max_{a'} Q^*(s',a')]$
3. ε-贪婪选择：$a_t = \\begin{cases} \\text{随机动作} & \\text{概率}\\epsilon \\ \\arg\\max_a Q(s,a) & \\text{否则} \\end{cases}$

### 12.3 算法关系
```
Q-learning → DQN → DDQN → Rainbow DQN → 强化学习进阶算法（PPO、SAC）
```

> 知识链接：后续学习`DDQN.md`解决Q值过估计问题

## 13. 练习题与思考题
### 13.1 基础题（5道）
1. DQN的两大核心创新是什么？
<details>
<summary>答案</summary>
经验回放（Experience Replay）和目标网络（Target Network）。
</details>

2. 经验回放的作用是什么？
<details>
<summary>答案</summary>
打破数据的时间相关性，提升样本利用率，稳定训练。
</details>

3. 目标网络的作用是什么？
<details>
<summary>答案</summary>
提供稳定的Q-target，避免目标值随当前网络频繁变化导致训练不稳定。
</details>

4. DQN适用于什么场景？
<details>
<summary>答案</summary>
适用于高维状态、离散动作、有延迟奖励的场景，如Atari游戏、机器人控制。
</details>

5. DQN的过估计问题是什么？
<details>
<summary>答案</summary>
max操作导致Q值被高估，影响策略质量，可通过DDQN缓解。
</details>

### 13.2 进阶题（2道）
1. 推导DQN的损失函数来源。
<details>
<summary>推导思路</summary>
来自贝尔曼最优方程：$Q^*(s,a) = \\mathbb{E}[r + \\gamma \\max_{a'} Q^*(s',a')]$，用均方误差拟合，得到损失$L = (r + \\gamma \\max Q' - Q)^2$。
</details>

2. 如何修改DQN适配连续动作空间？
<details>
<summary>答案</summary>
DQN本身不支持连续动作，需改用DDPG、PPO等算法，使用Actor-Critic框架。
</details>

### 13.3 开放讨论题（2道）
1. 为什么DQN在Atari游戏上能超过人类水平？
2. DQN的样本效率为什么低？如何改进？

### 13.4 面试题（2道）
1. 请解释DQN的原理，并说明经验回放和目标网络的作用。
2. DQN的过估计问题如何解决？

### 13.5 代码实践题（2道）
1. 实现DDQN，对比与DQN的性能差异。
2. 修改DQN代码，加入优先经验回放（PER）。

## 14. 学习路径建议
### 14.1 前置学习顺序
1. 学习强化学习基础（MDP、Q-learning）
2. 学习PyTorch框架（张量、网络定义、优化器）
3. 动手实现DQN（CartPole场景）
4. 阅读Mnih 2015 Nature论文（可选）
5. 学习DDQN、Rainbow DQN等改进算法

### 14.2 学习资源表
| 资源类型 | 名称 | 链接 |
|----------|------|------|
| 论文 | Human-level control through deep reinforcement learning | https://www.nature.com/articles/nature14236 |
| 视频 | DQN详解（David Silver） | https://www.youtube.com/watch?v=2pWv7GOvuf0 |
| 博客 | DQN从零实现 | https://spinningup.openai.com/en/latest/algorithms/dqn.html |
| 书籍 | 《强化学习》（ Sutton & Barto ）第16章 | https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf |

### 14.3 知识链接
- 上一篇：[Q-learning.md](Q-learning.md) 学习表格型Q-learning
- 下一篇：[DDQN.md](DDQN.md) 解决过估计问题
- 升级学习：[Rainbow DQN.md](Rainbow DQN.md) 整合多项改进
- 关联：[PPO.md](PPO.md) 策略梯度算法代表

### 14.4 学习路线图（ASCII）
```
Q-learning → DQN → DDQN → Rainbow DQN → 策略梯度（PPO）→ 强化学习进阶
```

> 来源线索：本节内容根据原书中关于"第10章 深度强化学习基础"的相关章节整理、扩展与教学化改写。
