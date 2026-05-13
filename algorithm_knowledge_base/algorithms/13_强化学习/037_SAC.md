# SAC 学习文档

> 用一句话说明这个算法的核心价值：SAC（Soft Actor-Critic）结合最大熵强化学习与双重Q网络，在探索、稳定性和样本效率之间取得最优平衡，是当前连续控制领域的标杆算法。

## 1. 算法基础认知

### 1.1 什么是SAC

SAC（Soft Actor-Critic）由 Tuomas Haarnoja 等人于 **2018年** 在论文 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" 中提出，后续在 2019 年的 "Soft Actor-Critic Algorithms and Applications" 中进一步完善自动温度调节机制。

SAC 基于**最大熵强化学习（Maximum Entropy RL）** 框架，在优化期望累积回报的同时，最大化策略的熵（随机性），鼓励智能体探索更多可能的行为。

**一句话定义**：SAC 是一种异策略、基于 Actor-Critic 架构的连续控制算法，通过在目标函数中加入策略熵正则项、结合双重 Q 网络、随机策略和自动温度调节，实现稳定、高效、鲁棒的连续动作控制。

### 1.2 三个直觉类比

| 类比场景 | 对应SAC组件 | 解释 |
|---------|------------|------|
| 学习时鼓励多样化方法 | 最大熵目标 | 不仅追求高分（回报），还鼓励尝试不同方法（熵），避免陷入单一低效模式 |
| 两个老师交叉打分 | 双重Q网络 | 用两个Q网络取较小值，避免对单一动作过度乐观，减少过估计 |
| 自动调节难度系数 | 自动温度调节α | 根据当前探索程度自动调节熵权重，探索不够时加大，探索充分时减小 |

### 1.3 算法定位表

| 属性 | SAC | 说明 |
|------|-----|------|
| 学习范式 | 免模型（Model-free） | 无需环境动力学模型 |
| 策略类型 | 异策略（Off-policy） | 可复用历史数据 |
| 动作空间 | 连续 | 输出高斯分布 |
| 策略类型 | 随机策略 | 天然支持探索 |
| 核心框架 | 最大熵强化学习 | 最大化回报+熵 |
| 网络架构 | Actor + 双Critic | 5个网络（含目标网络） |

### 1.4 前置知识清单

- [ ] **TD3 基本原理**：双重Q网络、目标网络、延迟更新
- [ ] **信息熵概念**：$H(X) = -\sum p(x)\log p(x)$，衡量随机性
- [ ] **重参数化技巧**：从高斯分布采样的可微操作
- [ ] **PyTorch 深度学习框架**：自动微分、优化器使用
- [ ] **Softmax / tanh 函数**：概率归一化和动作压缩

### 1.5 发展历程

```
2014  DDPG（深度确定性策略梯度，连续控制先驱）
  │
2018  TD3（双重Q+延迟更新，减少过估计）
  │
2018  SAC v1（最大熵+双重Q+固定温度α）
  │
2019  SAC v2（自动温度调节，无需手动设α）
  │
2020+ 工业级应用（机器人、自动驾驶、推荐系统）
```

## 2. 核心原理

### 2.1 最大熵强化学习

传统RL只最大化期望累积回报：

$$J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \right]$$

SAC 在此基础上**加入策略熵正则项**：

$$J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]$$

其中 $\alpha > 0$ 是**温度系数**，控制熵的权重。$\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|s)]$ 是策略熵。

**为什么加入熵？**

1. **鼓励探索**：高熵意味着策略更随机，不会过早收敛到次优策略
2. **提高鲁棒性**：学到的策略对环境扰动不敏感（多条路径都能到达目标）
3. **多模态行为**：同一状态下可能有多个好动作，熵正则保留这种多样性
4. **避免模式崩塌**：防止策略退化为确定性策略，丢失探索能力

### 2.2 四大核心技术

**技术1：随机策略（Gaussian Policy）**

策略网络输出动作的高斯分布参数 $(\mu, \sigma)$：

$$\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta(s)^2)$$

使用 tanh 压缩到动作范围 $[-1, 1]$，再缩放到环境允许范围。

**技术2：双重Q网络（Twin Q-networks）**

维护两个独立的Critic网络 $Q_{\phi_1}$、$Q_{\phi_2}$，取**较小值**作为目标：

$$Q(s,a) = \min(Q_{\phi_1}(s,a), Q_{\phi_2}(s,a))$$

这减少了Q值过估计偏差（继承自TD3）。

**技术3：自动温度调节（Automatic Entropy Tuning）**

将温度 $\alpha$ 视为可优化参数，目标：使策略熵自动收敛到目标熵 $\bar{\mathcal{H}}$：

$$\mathcal{L}(\alpha) = \mathbb{E}_{a\sim\pi} \left[ -\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}} \right]$$

目标熵通常设为 $\bar{\mathcal{H}} = -\dim(\mathcal{A})$（动作维度的负值）。

**技术4：重参数化技巧（Reparameterization Trick）**

使策略梯度可通过 Critic 传播：

$$a = \tanh(\mu + \sigma \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

将随机性从策略网络中分离，使得 Actor 损失可通过 Q 网络的反向传播计算梯度。

### 2.3 工作流程图

```
┌─────────────────────────────────────────────────────┐
│                   SAC 训练流程                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  状态 s ──→ Actor网络 ──→ μ, σ                      │
│                │         │                           │
│                │    重参数化采样                      │
│                │         │                           │
│                │         ▼                           │
│                │      动作 a ──→ 环境执行             │
│                │                     │               │
│                │                     ▼               │
│                │            (r, s', done)           │
│                │                     │               │
│                │              存入回放池              │
│                │                     │               │
│                │              采样 mini-batch        │
│                │                     │               │
│                ▼                     ▼               │
│  ┌─────────────────────────────────────────┐        │
│  │         更新 Critic（双网络）            │        │
│  │  目标: y = r + γ(min(Q'₁,Q'₂)          │        │
│  │              - α·log π(a'|s'))          │        │
│  │  损失: L = (y - Qᵢ(s,a))²             │        │
│  └─────────────────────────────────────────┘        │
│                │                                    │
│                ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │         更新 Actor                      │        │
│  │  损失: L = α·log π(a|s)               │        │
│  │        - min(Q₁,Q₂)(s,a)               │        │
│  └─────────────────────────────────────────┘        │
│                │                                    │
│                ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │      自动调节温度 α                     │        │
│  │  损失: L = -α·(log π + H̄)             │        │
│  └─────────────────────────────────────────┘        │
│                │                                    │
│                ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │    软更新目标网络 Q'ᵢ ← τQᵢ + (1-τ)Q'ᵢ│        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

### 2.4 SAC vs TD3 vs PPO 详细对比

| 特性 | SAC | TD3 | PPO |
|------|-----|-----|-----|
| 策略类型 | 随机（高斯） | 确定性 | 随机（ categorical/高斯） |
| 学习范式 | 异策略 | 异策略 | 同策略 |
| 熵正则 | 有（自动调节） | 无 | 有（固定或自适应） |
| 样本效率 | 最高 | 高 | 中等 |
| 探索能力 | 最强（熵驱动） | 弱（依赖噪声） | 中等 |
| 实现复杂度 | 高（5网络+α） | 中（6网络） | 低（2网络） |
| 训练稳定性 | 最高 | 高 | 中等 |
| 适用动作空间 | 连续 | 连续 | 连续/离散 |
| 推荐场景 | 机器人、连续控制 | 确定性控制 | 通用RL |

## 3. 数学公式与推导

### 3.1 完整符号约定表

| 符号 | 含义 | 维度/范围 |
|------|------|-----------|
| $s$ | 状态 | $\mathbb{R}^{d_s}$ |
| $a$ | 动作 | $\mathbb{R}^{d_a}$ |
| $r$ | 标量奖励 | $\mathbb{R}$ |
| $\pi_\theta$ | Actor策略网络（高斯） | 参数 $\theta$ |
| $\phi_i$ | 第 $i$ 个Critic网络参数 | $i=1,2$ |
| $\phi'_i$ | 第 $i$ 个目标Critic网络参数 | $i=1,2$ |
| $\alpha$ | 温度系数（熵权重） | $\mathbb{R}^+$ |
| $\bar{\mathcal{H}}$ | 目标熵 | $-\dim(\mathcal{A})$ |
| $\gamma$ | 折扣因子 | $(0, 1)$ |
| $\tau$ | 软更新系数 | $(0, 1)$ |
| $\mu_\theta(s)$ | 策略均值 | $\mathbb{R}^{d_a}$ |
| $\sigma_\theta(s)$ | 策略标准差 | $\mathbb{R}^{d_a}_{+}$ |

### 3.2 最大熵目标函数推导

**从原始MDP出发**。标准RL目标：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$$

加入熵正则：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t) + \alpha \underbrace{\mathcal{H}(\pi(\cdot|s_t))}_{\text{策略熵}} \right) \right]$$

展开策略熵：

$$\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|s)]$$

这个目标等价于修改后的 MDP，其中奖励为 $r'(s,a) = r(s,a) + \alpha \mathcal{H}(\pi(\cdot|s))$。

### 3.3 软贝尔曼方程推导

定义**软Q函数（Soft Q-function）**：

$$Q(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p} \left[ V(s') \right]$$

其中**软价值函数**：

$$V(s) = \mathbb{E}_{a\sim\pi} \left[ Q(s,a) - \alpha \log\pi(a|s) \right]$$

将V函数代入Q函数，得到**软贝尔曼方程**：

$$Q(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p, a' \sim \pi} \left[ Q(s',a') - \alpha \log\pi(a'|s') \right]$$

**物理直觉**：软贝尔曼方程在标准贝尔曼方程中加入了"探索奖励" $\alpha \log\pi(a'|s')$，鼓励策略保持随机性。

### 3.4 Critic 损失函数推导

两个Critic分别最小化**时序差分（TD）误差**：

$$\mathcal{L}(\phi_i, \mathcal{D}) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(s,a) - \hat{Q}(s,a) \right)^2 \right], \quad i=1,2$$

目标Q值（使用目标网络和双重Q最小值）：

$$\hat{Q}(s,a) = r + \gamma(1-d) \left[ \min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log\pi_\theta(a'|s') \right]$$

其中 $a' \sim \pi_\theta(\cdot|s')$ 是从**当前策略**采样的动作（非贪心），$d$ 是终止标志。

**为什么用当前策略采样而不是贪心？**

因为SAC的目标包含策略熵，需要在目标计算中体现策略的随机性。使用当前策略采样 $a'$ 并减去熵项 $-\alpha\log\pi(a'|s')$，等价于计算期望 $V(s') = \mathbb{E}_{a'\sim\pi}[Q(s',a') - \alpha\log\pi(a'|s')]$。

### 3.5 Actor 损失函数推导

Actor 的目标是**最大化期望软Q值**：

$$\max_\theta \mathbb{E}_{s\sim\mathcal{D}, a\sim\pi_\theta} \left[ Q(s,a) - \alpha \log\pi_\theta(a|s) \right]$$

等价于最小化：

$$\mathcal{L}(\theta) = \mathbb{E}_{s\sim\mathcal{D}} \left[ \alpha \log\pi_\theta(a|s) - Q(s,a) \right]$$

其中 $a$ 通过重参数化技巧采样：

$$a = f_\theta(\epsilon; s) = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

使用 $\min(Q_{\phi_1}, Q_{\phi_2})$ 代替 $Q$ 以减少过估计：

$$\mathcal{L}(\theta) = \mathbb{E}_{s\sim\mathcal{D}} \left[ \alpha \log\pi_\theta(a|s) - \min_{i=1,2} Q_{\phi_i}(s,a) \right]$$

**为什么损失是 $\alpha\log\pi - Q$ 而不是 $Q - \alpha\log\pi$？**

因为 PyTorch 的优化器默认执行梯度**下降**，而我们要**最大化** $Q - \alpha\log\pi$，所以取负号。

### 3.6 自动温度调节推导

温度 $\alpha$ 的目标是使策略熵逼近目标熵 $\bar{\mathcal{H}}$。

当实际熵 > 目标熵时，说明策略过于随机，需要减小 $\alpha$（减少探索）；反之则增大 $\alpha$。

损失函数：

$$\mathcal{L}(\alpha) = \mathbb{E}_{a\sim\pi_\theta} \left[ -\alpha \log\pi_\theta(a|s) - \alpha \bar{\mathcal{H}} \right]$$

$$= -\alpha \mathbb{E}_{a\sim\pi_\theta} \left[ \log\pi_\theta(a|s) + \bar{\mathcal{H}} \right]$$

$$= \alpha \left( \mathcal{H}(\pi_\theta(\cdot|s)) - \bar{\mathcal{H}} \right)$$

**梯度分析**：

$$\frac{\partial \mathcal{L}}{\partial \alpha} = \mathcal{H}(\pi_\theta) - \bar{\mathcal{H}}$$

当 $\mathcal{H} > \bar{\mathcal{H}}$ 时，$\frac{\partial \mathcal{L}}{\partial \alpha} > 0$，梯度下降使 $\alpha$ 减小 ✓
当 $\mathcal{H} < \bar{\mathcal{H}}$ 时，$\frac{\partial \mathcal{L}}{\partial \alpha} < 0$，梯度下降使 $\alpha$ 增大 ✓

### 3.7 重参数化技巧与 tanh 压缩的修正

原始高斯采样 $a \sim \mathcal{N}(\mu, \sigma^2)$ 后需要 tanh 压缩：

$$u = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$
$$a = \tanh(u)$$

此时 log_prob 需要修正（Jacobian 变换）：

$$\log\pi(a|s) = \log\mathcal{N}(u; \mu, \sigma^2) - \sum_i \log(1 - a_i^2 + \epsilon)$$

## 4. 训练过程讲解

### 4.1 数据预处理

| 步骤 | 操作 | 目的 |
|------|------|------|
| 状态归一化 | $(s - \mu_s) / \sigma_s$ | 消除不同维度量纲差异 |
| 动作裁剪 | $a \leftarrow \text{clip}(a, a_{min}, a_{max})$ | 确保动作在合法范围 |
| 奖励缩放 | $r \leftarrow r / r_{scale}$ | 防止奖励值过大导致Q发散 |

### 4.2 参数初始化表

| 参数 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| $\gamma$ | 折扣因子 | 0.99 | 远期奖励衰减 |
| $\tau$ | 软更新系数 | 0.005 | 目标网络追踪速度 |
| $\alpha$ 初始值 | 温度系数 | 0.2（自动调节时用 $\log(0.2)$） | 初始熵权重 |
| $\bar{\mathcal{H}}$ | 目标熵 | $-\dim(\mathcal{A})$ | 负的动作维度 |
| 学习率 | Actor/Critic/α | 3e-4 | Adam 默认 |
| 隐藏层大小 | 网络宽度 | 256 | 两层全连接 |
| 回放池容量 | 数据上限 | 1e6 | MuJoCo 推荐 |
| 批次大小 | mini-batch | 256 | 越大越稳定 |
| 初始随机步数 | 预热 | 1e4 | 填充回放池 |

### 4.3 网络架构详解

```
Actor 网络:
┌─────────────────────────────────┐
│  state (d_s)                    │
│     ↓                           │
│  Linear(d_s, 256) + ReLU        │
│     ↓                           │
│  Linear(256, 256) + ReLU        │
│     ↓                           │
│  ┌───────┬──────────┐           │
│  │ μ头    │ log_σ头  │           │
│  │Linear(256,d_a)   │Linear(256,d_a)│
│  │(无激活)          │clamp(-20,2)│
│  └───────┴──────────┘           │
│     ↓ 重参数化 + tanh           │
│  action (d_a)                   │
└─────────────────────────────────┘

Critic 网络 (×2):
┌─────────────────────────────────┐
│  state (d_s) + action (d_a)     │
│     ↓                           │
│  Linear(d_s+d_a, 256) + ReLU    │
│     ↓                           │
│  Linear(256, 256) + ReLU        │
│     ↓                           │
│  Linear(256, 1)                 │
│     ↓                           │
│  Q value (标量)                  │
└─────────────────────────────────┘
```

### 4.4 训练迭代详解

```
初始化:
  ├─ Actor: θ ~ N(0, 1/sqrt(fan_in))
  ├─ Critic1, Critic2: φ ~ N(0, 1/sqrt(fan_in))
  ├─ Critic1_target = Critic1, Critic2_target = Critic2
  ├─ log_alpha = log(0.2)
  └─ ReplayBuffer(capacity=1e6)

每回合 t:
  1. 观察状态 s_t
  2. Actor采样动作: a_t ~ π_θ(·|s_t)
  3. 执行动作: (r_t, s_{t+1}, done) ← env.step(a_t)
  4. 存储经验: buffer.push((s_t, a_t, r_t, s_{t+1}, done))

  5. if len(buffer) >= batch_size:
     a. 采样 mini-batch: (s,a,r,s',d) ~ buffer
     b. 用当前策略采样 a' ~ π_θ(·|s'), 计算 log_prob
     c. 计算目标Q:
        Q' = min(Q'₁(s',a'), Q'₂(s',a'))
        y = r + γ(1-d)(Q' - α·log_prob)
     d. 更新 Critic:
        L_critic = MSE(Q₁(s,a), y) + MSE(Q₂(s,a), y)
        critic_optimizer.step()
     e. 更新 Actor:
        a_new ~ π_θ(·|s), log_prob_new
        Q_new = min(Q₁(s,a_new), Q₂(s,a_new))
        L_actor = mean(α·log_prob_new - Q_new)
        actor_optimizer.step()
     f. 更新 α:
        L_alpha = mean(-α·(log_prob_new + H̄))
        alpha_optimizer.step()
     g. 软更新目标网络:
        Q'_i ← τ·Q_i + (1-τ)·Q'_i, i=1,2

收敛条件:
  - 滑动平均奖励稳定（窗口100）
  - Q值不再发散
  - α 收敛到合理范围（0.05~0.5）
```

### 4.5 工程经验

1. **预热回放池**：前 10000 步只用随机动作填充回放池，确保 Critic 初期有足够数据
2. **Critic 更新频率**：每步更新 1 次 Critic 和 Actor（同步更新），也可更新 2 次 Critic
3. **梯度裁剪**：裁剪 Actor 梯度到 `max_norm=1.0`，防止策略更新过大
4. **学习率调度**：Critic 学习率可以比 Actor 略大（如 1e-3 vs 3e-4），加快价值估计收敛

## 5. 应用场景

### 5.1 典型应用案例

**案例1：MuJoCo Pendulum-v1（钟摆倒立）**

| 属性 | 值 |
|------|-----|
| 状态 | 3维（角度cos、角度sin、角速度） |
| 动作 | 1维连续（扭矩，范围 [-2, 2]） |
| 奖励 | $-(\theta^2 + 0.1\dot\theta^2 + 0.001a^2)$ |
| 最优奖励 | 约 0 |
| SAC 训练回合 | ~50回合（~12500步）可收敛 |

**为什么SAC特别适合Pendulum？**

Pendulum 的奖励是连续的负值惩罚，需要精确控制，SAC 的随机策略天然提供平滑控制，且样本效率极高。

**案例2：MuJoCo HalfCheetah-v4（半猎豹奔跑）**

| 属性 | 值 |
|------|-----|
| 状态 | 17维（关节角度+角速度） |
| 动作 | 6维连续（关节扭矩） |
| 奖励 | 前进速度 - 能量消耗惩罚 |
| SAC 训练回合 | ~300回合可达到专业水平 |
| SOTA性能 | ~12000+ 奖励 |

**案例3：MuJoCo Humanoid-v4（人形机器人行走）**

| 属性 | 值 |
|------|-----|
| 状态 | 376维（全身关节） |
| 动作 | 17维连续 |
| 奖励 | 前进速度 + 姿态控制 |
| 训练难度 | 极高，需要 1000+ 回合 |
| SAC优势 | 最大熵保证多种稳定步态 |

**案例4：机器人灵巧手操作（Dexterous Manipulation）**

SAC 被广泛用于 OpenAI 的机械手控制任务，如灵巧手旋转魔方（SAC + curriculum learning）。

**案例5：自动驾驶横向控制**

| 属性 | 值 |
|------|-----|
| 状态 | 车辆位置、速度、航向角 |
| 动作 | 转向角、油门 |
| 挑战 | 安全约束、连续控制 |
| SAC优势 | 随机策略提供鲁棒性 |

### 5.2 适用/不适用场景

| 适用场景 | 原因 |
|---------|------|
| 连续动作空间控制 | SAC 专为连续动作设计 |
| 高维动作空间 | 随机策略+熵正则探索更充分 |
| 样本昂贵（机器人） | 异策略样本效率最高 |
| 需要鲁棒性 | 最大熵保留多种行为模式 |
| 奖励稀疏任务 | 熵正则鼓励探索未知区域 |

| 不适用场景 | 替代方案 |
|-----------|---------|
| 离散动作空间 | PPO / DQN |
| 需要极致确定性 | TD3 |
| 超大规模离散空间 | Rainbow / AlphaZero |
| 内存/计算受限 | PPO（更轻量） |

## 6. 优缺点分析

### 6.1 详细优点

| # | 优点 | 详细解释 | 适用条件 |
|---|------|---------|---------|
| 1 | **样本效率最高** | 异策略 + 经验回放 + 最大熵探索，样本利用率远超 PPO | 回放池 ≥ 1e6 |
| 2 | **训练极其稳定** | 双重Q + 熵正则 + 软更新，几乎不崩溃 | 正确实现所有组件 |
| 3 | **探索能力强** | 熵正则自动调节探索程度，避免局部最优 | 自动温度调节开启 |
| 4 | **鲁棒性好** | 随机策略保留多种行为模式，抗环境扰动 | α不过小 |
| 5 | **自动调参** | 温度α自动调节，减少超参数搜索 | 目标熵设为-dim(A) |

### 6.2 详细缺点

| # | 缺点 | 具体问题 | 解决方案 |
|---|------|---------|---------|
| 1 | **实现复杂** | 5个网络 + 自动调α，代码量大 | 参考CleanRL/SpinningUp |
| 2 | **计算开销大** | 每步需多次前向/反向传播 | 减小网络或批次大小 |
| 3 | **仅支持连续动作** | 离散动作需修改（SAC-Discrete） | 使用SAC-Discrete或换PPO |
| 4 | **Q值可能仍发散** | 熵项可能导致Q无界增长 | 监控Q值，裁剪奖励 |
| 5 | **调参仍有门槛** | 网络结构、学习率等仍需调优 | 参考论文默认参数 |

### 6.3 与相关算法对比

| 维度 | SAC | TD3 | PPO | DDPG |
|------|-----|-----|-----|------|
| 样本效率 | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| 训练稳定性 | ★★★★★ | ★★★★ | ★★★ | ★★ |
| 探索能力 | ★★★★★ | ★★ | ★★★ | ★★ |
| 实现难度 | ★★ | ★★★ | ★★★★★ | ★★★ |
| 连续控制性能 | ★★★★★ | ★★★★ | ★★★★ | ★★★ |

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque


class GaussianActor(nn.Module):
    """
    高斯策略网络
    输出动作分布参数 (mu, log_std)
    使用重参数化技巧采样，支持反向传播
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mu, log_std

    def sample(self, state):
        """
        重参数化采样
        返回: action, log_prob
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t) * self.max_action
        log_prob = dist.log_prob(x_t) - torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(-1, keepdim=True).expand_as(dist.log_prob(x_t))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """
    Q网络：输入 (state, action)，输出标量Q值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class SAC:
    """
    Soft Actor-Critic 完整实现
    包含: 双Critic, 随机Actor, 自动温度调节
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        hidden_dim=256,
        buffer_capacity=int(1e6),
        target_entropy=None
    ):
        self.actor = GaussianActor(state_dim, action_dim, max_action, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr
        )

        self.target_entropy = target_entropy or -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def choose_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            with torch.no_grad():
                mu, _ = self.actor(state_t)
                action = torch.tanh(mu) * self.max_action
            return action.cpu().numpy()[0]
        with torch.no_grad():
            action, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + self.gamma * q_next * (1 - dones)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, new_log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * new_log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (new_log_probs.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item()
        }


def train_sac(env_name='Pendulum-v1', max_episodes=200, max_steps=200):
    env = gym.make(env_name)
    agent = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0])
    )

    reward_history = []
    alpha_history = []

    for ep in range(max_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            info = agent.update(batch_size=256)
            total_reward += reward
            state = next_state
            if done:
                break

        reward_history.append(total_reward)
        alpha_history.append(info['alpha'] if info else 0)

        if (ep + 1) % 10 == 0:
            avg = np.mean(reward_history[-10:])
            print(
                f"Ep {ep+1}/{max_episodes} | "
                f"Reward: {total_reward:.1f} | "
                f"Avg10: {avg:.1f} | "
                f"Alpha: {alpha_history[-1]:.4f}"
            )

    return agent, reward_history, alpha_history


if __name__ == "__main__":
    agent, rewards, alphas = train_sac()
```

**运行结果示例**：
```
Ep 10/200 | Reward: -1234.5 | Avg10: -1567.8 | Alpha: 0.2000
Ep 50/200 | Reward: -345.2 | Avg10: -412.3 | Alpha: 0.1234
Ep 100/200 | Reward: -187.6 | Avg10: -195.4 | Alpha: 0.0856
Ep 150/200 | Reward: -132.1 | Avg10: -140.2 | Alpha: 0.0712
Ep 200/200 | Reward: -128.3 | Avg10: -130.5 | Alpha: 0.0689
```

## 8. 手工代码实现

从零实现SAC各组件核心逻辑，不依赖框架高层API：

```python
import numpy as np


def reparameterize_gaussian(mu, log_std, epsilon=None):
    """
    重参数化技巧：从高斯分布采样，梯度可传播
    mu: (batch, action_dim) 均值
    log_std: (batch, action_dim) 对数标准差
    返回: action, log_prob
    """
    if epsilon is None:
        epsilon = np.random.randn(*mu.shape)
    std = np.exp(log_std)
    u = mu + std * epsilon
    action = np.tanh(u)
    log_prob = (
        -0.5 * np.log(2 * np.pi) - log_std
        - 0.5 * epsilon ** 2
        - np.log(1 - action ** 2 + 1e-6)
    )
    return action, log_prob.sum(axis=-1, keepdims=True)


def soft_update_target(online_params, target_params, tau=0.005):
    """软更新目标网络参数"""
    new_target = {}
    for key in online_params:
        new_target[key] = (
            tau * online_params[key] + (1 - tau) * target_params[key]
        )
    return new_target


def compute_sac_targets(
    r, done, q1_next, q2_next, next_log_prob, alpha, gamma=0.99
):
    """
    计算SAC目标Q值
    y = r + γ(1-d)(min(Q'1, Q'2) - α·log π(a'|s'))
    """
    q_next = np.minimum(q1_next, q2_next)
    target = r + gamma * (1 - done) * (q_next - alpha * next_log_prob)
    return target


def compute_actor_loss(q1, q2, log_prob, alpha):
    """
    计算Actor损失
    L = mean(α·log_prob - min(Q1, Q2))
    """
    q_min = np.minimum(q1, q2)
    return np.mean(alpha * log_prob - q_min)


def compute_alpha_loss(log_prob, target_entropy, alpha):
    """
    计算温度损失
    L = mean(-α·(log_prob + H̄))
    """
    return np.mean(-alpha * (log_prob + target_entropy))


if __name__ == "__main__":
    np.random.seed(42)
    batch_size = 4
    action_dim = 2

    mu = np.random.randn(batch_size, action_dim)
    log_std = np.zeros((batch_size, action_dim)) - 0.5

    action, log_prob = reparameterize_gaussian(mu, log_std)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"Log prob shape: {log_prob.shape}")

    q1 = np.random.randn(batch_size, 1)
    q2 = np.random.randn(batch_size, 1)
    actor_loss = compute_actor_loss(q1, q2, log_prob, alpha=0.2)
    print(f"Actor loss: {actor_loss:.4f}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_sac_training(reward_history, alpha_history, window=10):
    """SAC训练过程全面可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(reward_history, alpha=0.3, color='blue', label='单回合奖励')
    if len(reward_history) >= window:
        moving_avg = np.convolve(
            reward_history, np.ones(window)/window, mode='valid'
        )
        axes[0].plot(
            range(window-1, len(reward_history)),
            moving_avg, color='red', linewidth=2,
            label=f'{window}回合滑动平均'
        )
    axes[0].set_xlabel('回合数')
    axes[0].set_ylabel('累积奖励')
    axes[0].set_title('SAC 训练奖励曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alpha_history, color='green', linewidth=1.5)
    axes[1].set_xlabel('回合数')
    axes[1].set_ylabel('温度系数 α')
    axes[1].set_title('自动温度调节过程')
    axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(reward_history[-50:], bins=20, color='purple', alpha=0.7)
    axes[2].set_xlabel('奖励值')
    axes[2].set_ylabel('频次')
    axes[2].set_title('最后50回合奖励分布')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sac_training.png', dpi=150)
    plt.show()


def plot_sac_vs_td3(sac_rewards, td3_rewards, window=10):
    """SAC vs TD3 性能对比"""
    plt.figure(figsize=(10, 6))
    for rewards, label, color in [
        (sac_rewards, 'SAC', 'blue'),
        (td3_rewards, 'TD3', 'orange')
    ]:
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), avg,
                     color=color, linewidth=2, label=label)
    plt.xlabel('回合数')
    plt.ylabel('累积奖励（滑动平均）')
    plt.title('SAC vs TD3 训练对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**结果解读要点**：

1. **奖励曲线**：应呈单调上升趋势，波动逐渐减小。若前期震荡剧烈属于正常（探索阶段）
2. **温度曲线**：α 应从初始值（~0.2）逐渐下降并收敛到较小值（~0.05-0.1），表示策略逐渐确定
3. **奖励分布**：后期分布应集中在高奖励区域，方差减小

## 10. 模型评估

```python
import numpy as np
import torch


def evaluate_sac(agent, env, episodes=20, max_steps=200):
    """完整评估SAC策略性能"""
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.choose_action(state, evaluate=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  最大奖励: {np.max(rewards):.2f}")
    print(f"  最小奖励: {np.min(rewards):.2f}")
    print(f"  中位数:   {np.median(rewards):.2f}")
    return avg_reward, std_reward


def evaluate_sample_efficiency(agent_builder, env, eval_interval=10):
    """
    评估样本效率
    记录不同训练步数下的评估奖励
    """
    results = {}
    total_steps = 0
    for ep in range(200):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            total_steps += 1
            state = next_state
        if (ep + 1) % eval_interval == 0:
            avg_r, std_r = evaluate_sac(agent, env, episodes=5)
            results[total_steps] = (avg_r, std_r)
    return results
```

**评估指标参考值（Pendulum-v1）**：

| 水平 | 平均奖励 | 说明 |
|------|---------|------|
| 随机策略 | ~ -1500 | 完全随机 |
| 初级训练 | ~ -800 | 50回合 |
| 中等训练 | ~ -300 | 100回合 |
| 收敛 | ~ -130 | 200回合 |
| 接近最优 | ~ -100 | 500回合 |
| 理论最优 | ~ 0 | - |

## 11. 常见问题与易错点

### 11.1 五大常见陷阱

| # | 问题现象 | 根本原因 | 解决方案 |
|---|---------|---------|---------|
| 1 | Q值发散、训练崩溃 | α 初始值过大或Critic学习率过高 | α初始设0.2，Critic lr=1e-4 |
| 2 | 策略不探索、陷入局部最优 | log_std 初始化过小，策略接近确定性 | log_std 初始化为 0，加 clamp |
| 3 | α 不收敛 | 目标熵设置错误（应为负动作维度） | 设 $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ |
| 4 | tanh 边界动作梯度消失 | log_prob 修正不正确 | 正确实现 Jacobian 修正项 |
| 5 | 目标网络滞后过大 | τ 太小导致追踪过慢 | τ = 0.005，或用 0.01 |

### 11.2 调试技巧

1. **监控 Q 值**：打印 Critic 损失和 Q 值均值，正常应逐渐减小并稳定
2. **监控 α**：应逐渐收敛，若持续增大说明探索不足，持续减小说明策略过于随机
3. **先验证 Critic**：单独训练 Critic（冻结 Actor），确认 Q 值估计合理后再联合训练
4. **梯度检查**：打印各网络梯度的范数，异常大的梯度说明训练不稳定
5. **简化测试**：先在 Pendulum 上验证，通过后再扩展到复杂环境

## 12. 学习总结

### 12.1 核心思想回顾

```
                    SAC 核心思想
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      最大熵目标      双重Q网络     自动温度调节
           │             │             │
    回报 + 熵正则    min(Q1,Q2)    α自适应调节
           │             │             │
    鼓励探索         减少过估计    平衡探索/利用
           │             │             │
           └─────────────┼─────────────┘
                         │
                  稳定高效连续控制
```

### 12.2 必记公式

1. **最大熵目标**：$J(\pi) = \mathbb{E}[\sum \gamma^t (r_t + \alpha \mathcal{H}(\pi))]$
2. **Critic 目标**：$y = r + \gamma(1-d)(\min(Q'_1, Q'_2) - \alpha \log\pi(a'|s'))$
3. **Actor 损失**：$\mathcal{L}_\theta = \mathbb{E}[\alpha\log\pi(a|s) - \min(Q_1,Q_2)(s,a)]$
4. **温度调节**：$\mathcal{L}_\alpha = \mathbb{E}[-\alpha(\log\pi(a|s) + \bar{\mathcal{H}})]$

### 12.3 算法关系图谱

```
DDPG ──→ TD3 ──→ SAC
 │         │        │
 │    双重Q    最大熵
 │    延迟更新  自动温度
 │         │        │
 └── 确定性策略     随机策略
                    │
              SAC-Discrete
```

## 13. 练习题与思考题

### 基础题

<details>
<summary>1. SAC中的"最大熵"具体指什么？为什么加入熵正则能改善探索？</summary>

**答案**：最大熵指在优化期望回报的同时最大化策略熵 $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|s)]$。策略熵衡量策略的随机性：高熵意味着策略更均匀地分配概率给不同动作，不会过早锁定到某个动作。加入熵正则等价于给每个动作添加内在奖励 $\alpha\log\pi(a|s)$，鼓励智能体尝试多种动作而非贪心选择当前最优动作。这解决了标准RL中"探索与利用"的平衡问题，使SAC不易陷入局部最优。
</details>

<details>
<summary>2. SAC 为什么要使用两个Critic网络并取最小值？</summary>

**答案**：这是从TD3继承的技术。由于函数近似误差，Q网络倾向于高估动作价值（过估计偏差），在 Actor-Critic 框架中会引导 Actor 选择被高估的次优动作。使用两个独立训练的Critic网络 $Q_{\phi_1}$ 和 $Q_{\phi_2}$，取 $\min(Q_1, Q_2)$ 作为目标，能有效抑制过估计。直觉：两个独立网络同时高估同一动作的概率低于单个网络。
</details>

<details>
<summary>3. 自动温度调节α的原理是什么？为什么目标熵设为动作维度的负值？</summary>

**答案**：自动温度调节通过优化损失 $\mathcal{L}(\alpha) = \alpha(\mathcal{H}(\pi) - \bar{\mathcal{H}})$ 来调节α。当实际熵 > 目标熵时，α 减小（降低探索权重）；反之 α 增大。目标熵设为 $-\dim(\mathcal{A})$ 是经验规则：对于维度为 $d$ 的连续动作，均匀分布的熵约为 $-d$（考虑 tanh 压缩后的分布），这个值使策略保持适度随机但不过于分散。
</details>

<details>
<summary>4. SAC 与 TD3 的本质区别是什么？</summary>

**答案**：核心区别在于策略类型和目标函数。TD3 使用确定性策略 $\pi(s) = a$，目标仅最大化期望回报；SAC 使用随机策略 $\pi(a|s) \sim \mathcal{N}(\mu, \sigma^2)$，目标是最大化期望回报加策略熵。这使得 SAC 天然具有探索能力，不需要额外噪声注入。此外 SAC 有自动温度调节机制。
</details>

<details>
<summary>5. 重参数化技巧在 SAC 中的作用是什么？</summary>

**答案**：策略网络输出高斯分布参数 $(\mu, \sigma)$，直接从分布采样 $a \sim \mathcal{N}(\mu, \sigma^2)$ 不可微。重参数化技巧将采样分解为确定性变换 + 外部噪声：$a = \mu + \sigma \odot \epsilon$，$\epsilon \sim \mathcal{N}(0,I)$。由于 $\mu, \sigma$ 是网络的确定性输出，梯度可以通过 Actor 网络反向传播到 Q 网络用于 Actor 更新。
</details>

### 进阶题

<details>
<summary>1. 证明SAC的Actor损失函数等价于最大化期望软Q值。</summary>

**答案**：SAC目标为最大化 $\mathbb{E}[Q(s,a) - \alpha\log\pi(a|s)]$（期望软Q值），其中 $a \sim \pi_\theta(\cdot|s)$。

取负号转化为最小化：$\mathcal{L}(\theta) = -\mathbb{E}[Q(s,a) - \alpha\log\pi(a|s)] = \mathbb{E}[\alpha\log\pi(a|s) - Q(s,a)]$。

使用重参数化 $a = f_\theta(\epsilon; s)$，期望变为对 $\epsilon \sim \mathcal{N}(0,I)$ 的期望：

$\mathcal{L}(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}}[\alpha\log\pi_\theta(f_\theta(\epsilon;s)|s) - Q(f_\theta(\epsilon;s), s)]$

这就是代码中 Actor 损失的来源。用 $\min(Q_1, Q_2)$ 替代 $Q$ 以减少过估计偏差。
</details>

<details>
<summary>2. 分析SAC的熵正则为什么等价于在奖励中加入状态依赖的内在奖励。</summary>

**答案**：SAC目标可写为 $J = \mathbb{E}[\sum \gamma^t (r_t + \alpha\mathcal{H}(\pi(\cdot|s_t)))]$。这等价于在每步的奖励中增加一项 $\alpha\mathcal{H}(\pi(\cdot|s_t))$，即内在奖励。这个内在奖励是**状态依赖**的：当策略在状态 $s_t$ 下分布越均匀（熵越高），内在奖励越大；当策略已高度确定时，内在奖励趋近于0。这种自适应的内在奖励机制自动调节了探索强度：在策略不确定的区域加强探索，在已确定的区域减少探索。
</details>

### 面试题

<details>
<summary>1. 如果SAC在某个任务上训练不收敛，你会从哪些方面排查？</summary>

**答案**：排查顺序：
1. **检查 Critic 损失**：若不收敛或爆炸，降低学习率或减小网络
2. **检查 Q 值范围**：若 Q 值增长无界，添加奖励裁剪或梯度裁剪
3. **检查 α 行为**：若 α 持续增大，说明策略过于确定性，检查 log_std 初始化
4. **检查回放池**：确保预热足够（至少 batch_size 的数据），数据分布不过时
5. **简化环境**：先在简单环境（Pendulum）验证实现正确性
6. **调整超参数**：$\gamma$、$\tau$、目标熵等是否合理
</details>

<details>
<summary>2. SAC能否应用于离散动作空间？如何修改？</summary>

**答案**：可以。SAC-Discrete（Christodoulou 2019）的修改要点：
1. Actor 输出动作概率分布 $\pi(a|s)$（Softmax），而非高斯参数
2. 重参数化技巧用 Gumbel-Softmax 代替 tanh 压缩
3. Q 网络输入状态（不拼接动作），输出所有动作的 Q 值 $Q(s, \cdot) \in \mathbb{R}^{|\mathcal{A}|}$
4. 目标Q值计算改为：$y = r + \gamma(1-d)\sum_a \pi(a'|s')(\min Q'_i(s',a') - \alpha\log\pi(a'|s'))$
</details>

### 代码练习

<details>
<summary>1. 修改SAC实现，添加Critic更新延迟（参考TD3），观察效果变化。</summary>

**提示**：在 `update()` 方法中，改为每隔 `policy_delay` 步才更新 Actor 和 α，而 Critic 每步都更新。设置 `policy_delay=2`。
</details>

<details>
<summary>2. 实现SAC-Discrete版本，在CartPole-v1上训练。</summary>

**提示**：Actor 输出 Softmax 概率，Critic 输出 `|A|` 维 Q 值，用 Gumbel-Softmax 重参数化。参考论文 "Soft Actor-Critic for Discrete Action Settings"。
</details>

## 14. 学习路径建议

### 14.1 前置学习路线

```
强化学习基础
    ├── MDP 与贝尔曼方程
    ├── Q-learning / DQN
    └── 策略梯度 / REINFORCE
          │
    Actor-Critic 架构
    ├── A2C / A3C（同策略AC）
    ├── PPO（现代同策略）
    └── DDPG（连续控制先驱）
          │
    连续控制进阶
    ├── TD3（确定性+双重Q）  ← 建议先学
    └── SAC（随机+最大熵）   ← 当前
          │
    扩展方向
    ├── SAC-Discrete
    ├── Offline SAC
    └── SAC + Curriculum Learning
```

### 14.2 推荐学习资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 论文 | Soft Actor-Critic (2018) | SAC v1，固定温度 |
| 论文 | Soft Actor-Critic: Algorithms and Applications (2019) | SAC v2，自动温度 |
| 教程 | OpenAI Spinning Up SAC | https://spinningup.openai.com/en/latest/algorithms/sac.html |
| 代码 | CleanRL SAC | https://github.com/vwxyzjn/cleanrl |
| 代码 | Stable-Baselines3 | https://github.com/DLR-RM/stable-baselines3 |
| 视频 | Pieter Abbeel RL课程 | YouTube: CS285 |
| 书籍 | Easy RL 教程 | 第8章 深度确定性策略梯度 |

### 14.3 知识链接

- **前置**：`TD3.md`、`DDPG.md`、`PPO.md`
- **相关**：`马尔可夫决策过程.md`（最大熵MDP理论）
- **进阶**：`逆强化学习.md`（最大熵IRL与SAC的理论联系）

> 来源线索：本节内容根据原书中关于"第8章 深度确定性策略梯度"和"SAC"的相关章节整理、扩展与教学化改写。
