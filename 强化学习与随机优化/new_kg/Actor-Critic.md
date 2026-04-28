# Actor-Critic 学习文档

> Actor-Critic结合策略梯度的灵活性和值函数的低方差优势，是现代强化学习最实用的算法范式之一。

> 来源线索：本节内容根据原书中关于"The Actor-Critic Paradigm"的相关章节(Ch 17.8)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：Actor-Critic同时学习一个策略函数（Actor，决定动作）和一个值函数（Critic，评估动作好坏），通过Critic的低方差评估信号来指导Actor的策略改进。

**直觉类比**：想象一个演员（Actor）在舞台上表演，一个导演（Critic）在台下打分。演员根据导演的反馈调整表演方式：如果导演说"这段演得好"，演员会多做类似的表演；如果导演说"这段不行"，演员会减少。关键是导演不是等整场戏演完才打分，而是每一段都给即时反馈——这就是比REINFORCE更高效的原因。

**历史背景**：Actor-Critic的概念最早可追溯到Widrow et al.(1973)的"critic"学习框架。Barto, Sutton和Anderson在1983年将其应用于倒立摆控制。Sutton等人在2000年提出了策略梯度定理，为现代Actor-Critic方法（A2C、A3C、PPO等）奠定了理论基础。

**算法定位**：策略梯度+值函数联合学习。属于原书四类策略中的"值函数近似（VFA）"与"策略函数近似（PFA）"的混合方法。

**前置知识**：策略梯度定理、值函数、TD学习、神经网络基础。

## 2. 核心原理

**核心思想**：纯策略梯度方法（如REINFORCE）使用回合回报作为策略梯度估计的权重，方差很大。Actor-Critic的关键改进是：用一个学习的值函数（Critic）来估计期望回报，取代蒙特卡洛采样。这样既保留了策略梯度的灵活性（支持连续动作空间），又获得了TD方法的低方差优势。

**工作流程**：

1. 观察状态$S_t$
2. Actor根据当前策略$\pi_\theta(a|s)$选择动作$A_t$
3. 执行动作，获得奖励$R_{t+1}$，转移到$S_{t+1}$
4. Critic计算TD误差：$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$
5. 更新Critic参数：$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(S_t)$
6. 更新Actor参数：$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(A_t|S_t)$
7. 重复

**关键概念**：

- **Actor**：参数化策略$\pi_\theta(a|s)$，输出动作概率分布
- **Critic**：参数化值函数$V_w(s)$，评估状态价值
- **优势函数**$A(s,a) = Q(s,a) - V(s)$：衡量动作$a$比平均水平好多少
- **TD误差作为优势估计**：$\delta_t \approx A(S_t, A_t)$

```
      状态 S_t
     ╱       ╲
   Actor      Critic
  π_θ(a|s)   V_w(s)
     │           │
  采样动作     计算TD误差
  A_t         δ_t = R + γV(S') - V(S)
     │           │
     ↓           ↓
  执行动作    更新Critic: w ← w + α_w·δ·∇V
     ↓           ↓
  R_{t+1}    更新Actor: θ ← θ + α_θ·δ·∇log π
     ↓
  S_{t+1} → 循环
```

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $\pi_\theta(a\|s)$ | Actor策略（参数θ） |
| $V_w(s)$ | Critic值函数（参数w） |
| $\delta_t$ | TD误差 |
| $A(s,a)$ | 优势函数 |
| $\alpha_\theta$ | Actor学习率 |
| $\alpha_w$ | Critic学习率 |

### 策略梯度定理

策略梯度的核心公式：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[A^{\pi_\theta}(S,A) \nabla_\theta \log \pi_\theta(A|S)\right]$$

其中$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$是优势函数。

### 用TD误差近似优势

在实际实现中，用TD误差$\delta_t$近似优势函数：

$$A(S_t, A_t) \approx \delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$$

这是合理的，因为$\mathbb{E}[\delta_t] = \mathbb{E}[A(S_t, A_t)]$（TD误差是优势的无偏估计）。

### Actor更新

$$\theta \leftarrow \theta + \alpha_\theta \cdot \delta_t \cdot \nabla_\theta \log \pi_\theta(A_t|S_t)$$

直觉：如果$\delta_t > 0$（动作比预期好），增大该动作的概率；如果$\delta_t < 0$（比预期差），减小概率。

### Critic更新

$$w \leftarrow w + \alpha_w \cdot \delta_t \cdot \nabla_w V_w(S_t)$$

Critic用标准TD(0)学习更新值函数参数。

### 对比REINFORCE

REINFORCE用蒙特卡洛回报$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$作为权重：

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(A_t|S_t)$$

Actor-Critic用$\delta_t$替代$G_t$，方差大幅降低（因为$\delta_t$是单步误差而非整回合累积）。

## 4. 训练过程讲解

### 数据预处理
- 状态归一化（有助于神经网络训练稳定）
- 奖励缩放（避免梯度过大）

### 参数初始化
- Actor网络：小随机值（Xavier/He初始化）
- Critic网络：小随机值
- 输出层偏置：策略最后一层偏置初始化为0

### 迭代过程
1. 收集一步数据$(S_t, A_t, R_{t+1}, S_{t+1})$
2. Critic前向传播：计算$V_w(S_t)$和$V_w(S_{t+1})$
3. 计算TD误差$\delta_t$
4. 更新Critic参数
5. 计算策略梯度$\nabla_\theta \log \pi_\theta(A_t|S_t)$
6. 更新Actor参数
7. 重复

### 收敛条件
- 累积奖励达到目标
- 策略在评估中表现稳定
- 最大训练步数

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $\alpha_\theta$ | Actor学习率 | [1e-4, 1e-2] | 1e-3 |
| $\alpha_w$ | Critic学习率 | [1e-3, 1e-1] | 1e-2 |
| $\gamma$ | 折扣因子 | [0.99, 0.999] | 0.99 |
| hidden_dim | 隐藏层维度 | [32, 256] | 128 |

## 5. 应用场景

### 1. 连续控制
为什么适合：Actor输出连续动作（如力矩、位置），Critic评估价值。适用于机器人控制、自动驾驶。

### 2. 游戏AI
为什么适合：Atari、MuJoCo等环境，Actor-Critic可以学习复杂的策略。

### 3. 推荐系统
为什么适合：可以处理大规模离散动作空间，同时学习用户价值。

### 4. 资源调度
为什么适合：状态和动作空间大且连续，Actor-Critic能高效探索。

### 不适用场景
- 离散动作空间且状态空间小（Q-Learning更简单）
- 需要高样本效率的场景（模型基方法更高效）

## 6. 优缺点分析

### 优点
1. **低方差**：Critic提供稳定的梯度信号（相比REINFORCE）
2. **连续动作空间**：天然支持连续动作
3. **在线学习**：每步更新，不需要等待回合结束
4. **灵活策略**：可学习随机策略

### 缺点
1. **Critic偏差**：不准确的Critic会误导Actor
2. **训练不稳定**：两个网络同时训练可能相互干扰
3. **调参困难**：需要平衡Actor和Critic的学习率
4. **样本效率一般**：仍需大量交互

### 算法对比

| 特性 | Actor-Critic | REINFORCE | Q-Learning | PPO |
|------|-------------|-----------|------------|-----|
| 策略类型 | 在策略 | 在策略 | 离策略 | 在策略 |
| 方差 | 低 | 高 | 低 | 低 |
| 连续动作 | 支持 | 支持 | 不支持 | 支持 |
| 训练稳定性 | 中 | 差 | 中 | 好 |
| 样本效率 | 中 | 低 | 高 | 中 |

## 7. 调库实现

```python
"""
使用PyTorch实现Actor-Critic (A2C)
场景：CartPole-v1
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    """Actor-Critic网络（共享特征提取层）"""
    def __init__(self, n_obs, n_act, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor = nn.Linear(hidden, n_act)   # 策略输出
        self.critic = nn.Linear(hidden, 1)       # 值函数输出

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

def train_actor_critic(env_name='CartPole-v1', n_episodes=1000, gamma=0.99,
                       lr_actor=1e-3, lr_critic=1e-2):
    env = gym.make(env_name)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n

    model = ActorCritic(n_obs, n_act)
    optimizer = optim.Adam(model.parameters(), lr=lr_actor)

    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        log_probs, values, rewards = [], [], []
        done = False

        while not done:
            state_t = torch.FloatTensor(state)
            logits, value = model(state_t)

            # Actor: 从策略分布采样动作
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            state = next_state

        # 计算回报和优势
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        # 优势 = 回报 - 基线（Critic估计）
        advantages = returns - values.detach()

        # Actor损失（策略梯度）+ Critic损失（值函数回归）
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))
        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}, 平均奖励: {avg:.1f}")

    env.close()
    return model, episode_rewards

# model, rewards = train_actor_critic()
```

## 8. 手工代码实现

```python
"""
从零实现Actor-Critic
NumPy + 简单策略梯度，无任何RL库
使用CartPole风格的简单环境
"""
import numpy as np

class SimpleActorCritic:
    """简单的Actor-Critic实现（线性策略和值函数）"""

    def __init__(self, n_states, n_actions, lr_actor=0.01, lr_critic=0.05, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Actor参数：策略权重 (n_states × n_actions)
        self.theta = np.random.randn(n_states, n_actions) * 0.01
        # Critic参数：值函数权重 (n_states)
        self.w = np.random.randn(n_states) * 0.01

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def act(self, state_features):
        """Actor：根据策略选择动作"""
        logits = self.theta.T @ state_features
        probs = self.softmax(logits)
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def value(self, state_features):
        """Critic：估计状态值"""
        return self.w @ state_features

    def update(self, state, action, reward, next_state, done):
        """Actor-Critic更新"""
        # Critic计算TD误差
        v = self.value(state)
        v_next = self.value(next_state) if not done else 0.0
        delta = reward + self.gamma * v_next - v

        # 更新Critic：w ← w + α_w · δ · state
        self.w += self.lr_critic * delta * state

        # 更新Actor：θ ← θ + α_θ · δ · ∇log π(a|s)
        logits = self.theta.T @ state
        probs = self.softmax(logits)
        # ∇log π(a|s) = state - Σ π(a')·state = (one_hot(a) - probs) ⊗ state
        grad = np.outer(state, np.eye(self.n_actions)[action] - probs)
        self.theta += self.lr_actor * delta * grad

        return delta


# ========== 测试：简单网格世界 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 简单环境：4个状态特征，2个动作
    n_features = 4
    n_actions = 2
    agent = SimpleActorCritic(n_features, n_actions)

    # 模拟训练（简化）
    total_rewards = []
    for ep in range(2000):
        # 随机状态特征
        state = np.random.randn(n_features)
        total_reward = 0
        for step in range(20):
            action, prob = agent.act(state)
            # 简单奖励：第一个动作好
            reward = 1.0 if action == 0 else -0.1
            next_state = np.random.randn(n_features)
            done = (step >= 19)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)

    # 验证学到了偏好动作0
    test_state = np.ones(n_features)
    action, _ = agent.act(test_state)
    print(f"测试动作: {action} (期望=0)")
    print(f"最后100轮平均奖励: {np.mean(total_rewards[-100:]):.2f}")
    print("Actor-Critic训练完成")
```

## 9. 可视化与结果理解

```python
"""Actor-Critic训练可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_ac_training(episode_rewards):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 滑动平均奖励
    window = 50
    smoothed = [np.mean(episode_rewards[max(0,i-window):i+1])
                for i in range(len(episode_rewards))]
    ax1.plot(smoothed)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('Actor-Critic 学习曲线')
    ax1.grid(True, alpha=0.3)

    # 奖励分布（前半 vs 后半）
    mid = len(episode_rewards) // 2
    ax2.hist(episode_rewards[:mid], bins=30, alpha=0.5, label='前半训练')
    ax2.hist(episode_rewards[mid:], bins=30, alpha=0.5, label='后半训练')
    ax2.set_xlabel('Episode奖励')
    ax2.set_ylabel('频次')
    ax2.set_title('奖励分布变化')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('actor_critic_results.png', dpi=150)
    plt.show()
```

**结果解读**：学习曲线应显示平均奖励逐步上升并趋于稳定。后半训练的奖励分布应比前半更集中在高值区域。

## 10. 模型评估

```python
"""评估Actor-Critic策略"""
def evaluate_ac_policy(model, env_name='CartPole-v1', n_episodes=100):
    import gymnasium as gym
    import torch
    env = gym.make(env_name)
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state)
                logits, _ = model(state_t)
                action = torch.argmax(logits).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    print(f"评估({n_episodes}回合): 平均={np.mean(rewards):.1f}, 标准差={np.std(rewards):.1f}")
    return np.mean(rewards)
```

## 11. 常见问题与易错点

### 数据层面

1. **状态特征未归一化**
   - 现象：训练不稳定，梯度爆炸
   - 原因：不同状态特征量级差异大
   - 解决方案：对状态做标准化或使用BatchNorm

2. **奖励尺度不当**
   - 现象：Critic值函数过大或过小
   - 原因：奖励绝对值太大或太小
   - 解决方案：奖励缩放到合理范围（如[-1, 1]）

### 模型层面

3. **Actor和Critic学习率不匹配**
   - 现象：Critic学习太慢无法提供准确评估，或学习太快覆盖Actor信号
   - 原因：两个网络的学习率需要协调
   - 解决方案：通常Critic学习率比Actor大2-10倍

4. **共享特征层导致的梯度冲突**
   - 现象：训练不稳定
   - 原因：Actor和Critic的梯度方向冲突
   - 解决方案：使用独立的网络，或调整共享层的梯度比例

### 调参层面

5. **折扣因子选择**
   - 现象：短期行为过多（γ小）或远期信号太弱（γ大）
   - 解决方案：CartPole用0.99，长期问题用0.995-0.999

## 12. 学习总结

Actor-Critic的核心贡献是将策略梯度（灵活但高方差）和值函数近似（稳定但受限）结合。Actor负责"做什么"，Critic负责"做得好不好"。Critic提供的TD误差$\delta_t$既是值函数更新的信号，也是策略梯度的低方差替代权重。

**关键公式**：
1. Actor更新：$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(A_t|S_t)$
2. Critic更新：$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(S_t)$
3. TD误差：$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$

Actor-Critic是REINFORCE（纯策略梯度）的自然改进。在原书框架中，它体现了VFA和PFA策略的混合——Actor是PFA，Critic是VFA。后续的A2C/A3C（多线程同步）、PPO（裁剪目标）等都是Actor-Critic的工程优化版本。

## 13. 练习题与思考题

### 基础题

**题目1**：在Actor-Critic中，为什么用TD误差$\delta_t$替代蒙特卡洛回报$G_t$作为策略梯度的权重能降低方差？

**参考答案**：
$G_t$是整回合的累积折扣奖励，受多个时间步的随机性影响，方差很大。$\delta_t$只涉及一步的随机性（$R_{t+1}$和$S_{t+1}$），但通过Critic的自举间接利用了后续信息。$\delta_t$是$A(S_t,A_t)$的无偏估计，但方差远低于$G_t$，因为Critic已经综合了历史经验来估计$V(S)$。

### 进阶题

**题目2**：如果Critic的值函数完全不准确（比如$V(s)=0$对所有$s$），Actor-Critic会退化成什么？如果Critic完全准确呢？

**参考答案**：
- Critic完全不准确（$V(s)=0$）：$\delta_t = R_{t+1} + 0 - 0 = R_{t+1}$，退化为仅用即时奖励的1步REINFORCE，没有利用未来信息。
- Critic完全准确（$V=V^*$）：$\delta_t$精确等于优势函数$A(s,a)$，策略梯度无偏且方差最低，理论上一次更新就能找到最优方向。

### 开放思考题

**题目3**：原书将Actor-Critic归类为VFA和PFA的混合策略。在你看来，Actor-Critic与纯VFA方法（如Q-Learning）的本质区别是什么？什么情况下Actor-Critic比Q-Learning更合适？

**参考答案方向**：
- 本质区别：Q-Learning学习$Q(s,a)$表格/函数然后推导策略，Actor-Critic直接参数化策略$\pi_\theta$
- Actor-Critic更适合：(1)连续动作空间（Q-Learning需要离散化）；(2)需要随机策略的场景；(3)动作空间大时（Q-Learning的max操作代价高）
- Q-Learning更适合：(1)离散动作空间；(2)需要离策略学习利用历史数据；(3)简单问题不需要策略的显式参数化

## 14. 学习路径建议

**前置算法**：策略梯度定理、TD学习、神经网络基础

**平行算法**：REINFORCE（纯策略梯度）、Q-Learning（纯值函数）

**进阶算法**：A2C/A3C（异步Actor-Critic）、PPO（近端策略优化）、SAC（Soft Actor-Critic）、DDPG（确定性策略梯度）

**推荐资源**：
1. 原书Ch 17.8 "The Actor-Critic Paradigm"
2. Sutton & Barto "Reinforcement Learning" Ch 13 "Policy Gradient Methods"
3. Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning" (A3C论文)
