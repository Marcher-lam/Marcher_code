# TD（时序差分）学习 学习文档

## 1. 算法基础认知

时序差分（Temporal Difference, TD）学习是强化学习中最核心的概念之一，由 Sutton 于 1988 年提出。TD 方法结合了**蒙特卡洛（MC）的采样思想**和**动态规划（DP）的自举（bootstrapping）思想**：

- 像 MC 一样：从环境交互经验中学习（无需模型）
- 像 DP 一样：基于其他估计值来更新当前估计（自举）

最简单的 TD(0) 在每一步交互后立即更新价值函数，无需等待 episode 结束。

## 2. 核心原理

### 2.1 Bootstrapping（自举）

自举是指用一个估计值去更新另一个估计值。TD 用 $R_{t+1} + \gamma V(S_{t+1})$ 代替 MC 中的完整回报 $G_t$，这个替代值称为 **TD 目标**：

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

$\delta_t$ 称为 **TD 误差**。

### 2.2 TD(0)

最基础的 TD 预测方法，一步自举：

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### 2.3 TD($\lambda$) 与资格迹

TD(0) 只回看一步。TD($\lambda$) 通过**资格迹（Eligibility Traces）**将多步回报融合：

前向视角：使用 $\lambda$-回报 $G_t^\lambda$ 作为目标：

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}$$

后向视角（在线实现）：维护资格迹 $E_t(s)$：

$$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbf{1}[S_t = s]$$

$$V(s) \leftarrow V(s) + \alpha \delta_t E_t(s), \quad \forall s$$

当 $\lambda=0$ 退化为 TD(0)，$\lambda=1$ 退化为 MC。

## 3. 数学公式与推导

### TD(0) 收敛性

TD(0) 在表格情况下对固定策略 $\pi$ 收敛到 $V^\pi$。关键条件是步长满足 Robbins-Monro 条件：

$$\sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

### TD 目标的偏差-方差分析

| 目标 | 偏差 | 方差 |
|------|------|------|
| TD 目标 $R_{t+1}+\gamma V(S_{t+1})$ | 有偏（依赖 $V$ 的估计误差） | 低 |
| MC 回报 $G_t$ | 无偏 | 高 |
| $G_t^\lambda$ | 中等 | 中等 |

$\lambda$ 控制了偏差-方差权衡：$\lambda$ 越大越接近 MC（无偏高方差），越小越接近 TD(0)（有偏低方差）。

### n-step TD 的统一视角

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

TD(0) 是 n=1 的特例，MC 是 $n \to \infty$ 的特例。

## 4. 训练过程讲解

### TD(0) 预测流程

1. **初始化**：$V(s) = 0$，$\forall s \in S$
2. **对每个 episode**：
   - 初始化 $S$
   - **循环每一步**：
     - $A \leftarrow \pi(S)$
     - 执行 $A$，观测 $R, S'$
     - $V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]$
     - $S \leftarrow S'$
   - 直到 $S'$ 为终止状态

### TD($\lambda$) 后向视角流程

1. **初始化**：$V(s)=0$，$E(s)=0$
2. **对每个 episode**：
   - 初始化 $S$，$E(s)=0, \forall s$
   - **循环每一步**：
     - $A \leftarrow \pi(S)$，执行 $A$，观测 $R, S'$
     - $\delta \leftarrow R + \gamma V(S') - V(S)$
     - $E(S) \leftarrow E(S) + 1$
     - **对所有 $s$**：$V(s) \leftarrow V(s) + \alpha \delta E(s)$，$E(s) \leftarrow \gamma \lambda E(s)$
     - $S \leftarrow S'$

## 5. 应用场景

- **价值函数近似**：深度 Q 网络（DQN）的基础
- **在线控制**：机器人、游戏 AI 的实时策略评估
- **资源分配**：网络拥塞控制
- **金融**：期权定价的 TD 方法
- **广告竞价**：实时评估不同竞价策略的长期价值

## 6. 优缺点分析

**优点**：
- 每步即可更新，无需等 episode 结束（在线学习）
- 方差低于 MC
- 可用于 continuing 任务
- 计算效率高

**缺点**：
- 有偏估计（bootstrap 引入偏差）
- 表格情况下收敛速度受步长影响大
- 函数近似下可能出现发散（需要目标网络等技巧）
- 对初始值敏感

## 7. 调库实现（Python）

```python
import numpy as np
from collections import defaultdict

def td0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=1.0):
    V = defaultdict(float)
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] * (1 - done) - V[state])
            state = next_state
    return V

def td_lambda_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=1.0, lam=0.8):
    V = defaultdict(float)
    for _ in range(num_episodes):
        state = env.reset()
        E = defaultdict(float)
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[next_state] * (1 - done) - V[state]
            E[state] += 1.0
            for s in E:
                V[s] += alpha * delta * E[s]
                E[s] *= gamma * lam
            state = next_state
    return V
```

## 8. 手工代码实现

```python
import numpy as np

class TD0:
    def __init__(self, n_states, alpha=0.1, gamma=1.0):
        self.V = np.zeros(n_states)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, r, s_next, done):
        target = r + self.gamma * self.V[s_next] * (1 - done)
        self.V[s] += self.alpha * (target - self.V[s])

class TDLambda:
    def __init__(self, n_states, alpha=0.1, gamma=1.0, lam=0.8):
        self.V = np.zeros(n_states)
        self.E = np.zeros(n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def reset_traces(self):
        self.E[:] = 0.0

    def update(self, s, r, s_next, done):
        delta = r + self.gamma * self.V[s_next] * (1 - done) - self.V[s]
        self.E[s] += 1.0
        self.V += self.alpha * delta * self.E
        self.E *= self.gamma * self.lam
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

def plot_td_convergence(V_history, true_V, state):
    errors = [np.abs(v[state] - true_V[state]) for v in V_history]
    plt.semilogy(errors)
    plt.xlabel('Episode')
    plt.ylabel('|V_est - V_true|')
    plt.title(f'TD Prediction Error (state {state})')
    plt.grid(True)
    plt.savefig('td_convergence.png', dpi=150)
```

关键可视化：
- **估计值收敛曲线**：观察 $V(s)$ 如何随 episode 收敛到真值
- **不同 $\lambda$ 的学习速度对比**：通常中间值（$\lambda \approx 0.7$）效果最好
- **TD 误差热力图**：$\delta_t$ 的大小反映预测的不准确程度

## 10. 模型评估

- **RMSE vs 真实 $V^\pi$**：$\sqrt{\frac{1}{|S|}\sum_s(\hat{V}(s) - V^\pi(s))^2}$
- **学习曲线**：RMSE 随 episode 数的变化
- **不同 $\lambda$ 的性能比较**：画出 RMSE-$\lambda$ 曲线，通常呈 U 形
- **样本效率**：达到特定精度所需的交互步数

## 11. 常见问题与易错点

- **混淆 TD 和 MC 的更新时机**：TD 每步更新，MC 每个 episode 结束后更新
- **终止状态处理**：$V(\text{terminal})$ 应为 0，用 `(1 - done)` 因子确保
- **步长 $\alpha$ 选择**：太大则震荡不收敛，太小则学习慢
- **资格迹爆炸**：在线更新时 $E(s)$ 可能过大，需截断或使用 replacing trace
- **TD($\lambda$) 前向和后向视角**：在离线情况下等价，在线时有细微差别

## 12. 学习总结

TD 学习是强化学习的核心范式。它通过 bootstrapping 实现了高效在线学习，平衡了 MC 的无偏性和 DP 的计算效率。TD(0) 是基础，TD($\lambda$) 通过资格迹灵活控制偏差-方差权衡。几乎所有现代深度强化学习算法（DQN、PPO、SAC）都建立在 TD 思想之上。

## 13. 练习题与思考题

**Q1**：证明 TD(0) 的更新目标是 $V^\pi$ 的有偏估计。

> **答案**：TD 目标为 $R_{t+1} + \gamma V(S_{t+1})$，其中 $V(S_{t+1})$ 是估计值而非真值，因此 $\mathbb{E}[R_{t+1} + \gamma V(S_{t+1})] \neq V^\pi(S_t)$（除非 $V = V^\pi$）。

**Q2**：为什么 $\lambda=1$ 的后向视角 TD 等价于 MC？

> **答案**：$\lambda=1$ 时，$E(s)$ 以 $\gamma^{k}$ 衰减，对某状态 $s$ 累积的 TD 误差修正恰好等于从该状态出发的完整回报 $G_t$ 与当前估计的差。

**Q3**：TD(0) 和 MC 哪个样本效率更高？为什么？

> **答案**：TD(0) 通常样本效率更高，因为它利用了马尔可夫性（bootstrapping），而 MC 不利用马尔可夫假设。但在非马尔可夫环境中，MC 可能更可靠。

## 14. 学习路径建议

1. **前置知识**：MC 预测、DP、MDP 基础
2. **本节掌握**：TD(0) 更新规则、TD($\lambda$)、资格迹
3. **进阶方向**：
   - TD 控制（SARSA、Q-learning）
   - 多步回报与 n-step TD
   - 函数近似下的 TD 学习（梯度 TD 方法）
4. **后续学习**：DQN、策略梯度、Actor-Critic
