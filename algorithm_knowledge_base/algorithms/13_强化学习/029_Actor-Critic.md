# Actor-Critic 学习文档

> Actor 学策略，Critic 学价值——策略梯度与值函数的完美结合。

> 来源线索：本节内容根据原书中关于"Actor-Critic"的相关章节（第13章13.5.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** Actor-Critic 同时训练策略网络（Actor）和价值网络（Critic），Critic 用 TD 误差为 Actor 提供低方差的梯度信号。

**直觉类比：** 策略梯度（REINFORCE）像"只有运动员"——只能通过最终比分判断表现好坏。Actor-Critic 加了一个"教练"（Critic）——每一步都给出即时反馈"这一步做得怎么样"。教练的评估比最终比分更有指导性，大大加快了学习速度。

**历史背景：** Actor-Critic 框架由 Barto 等人于 1983 年提出。A2C/A3C（Mnih et al., 2016）使其在大规模问题上实用化。

**算法定位：** 强化学习、策略优化+值函数估计。

**前置知识：** 策略梯度、TD 学习、价值函数。

---

## 2-3. 核心原理与数学公式

### 两个网络

- **Actor $\pi_\theta(a|s)$**：策略网络，输出动作概率
- **Critic $V_\phi(s)$**：价值网络，输出状态价值估计

### 更新规则

**Critic 更新**（TD 误差）：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

$$\mathcal{L}_{Critic} = \delta_t^2$$

**Actor 更新**（策略梯度，用 TD 误差替代回报）：

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

### 优势函数

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t$$

---

## 4-8. 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class A2CAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def step(self, state):
        probs, value = self.model(torch.FloatTensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(), dist.entropy()

    def update(self, rewards, log_probs, values, entropies):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        advantages = returns - torch.stack(values)
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -torch.stack(entropies).mean() * 0.01

        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

agent = A2CAgent(state_dim=4, action_dim=2)
state = np.random.randn(4)
action, lp, val, ent = agent.step(state)
print(f"动作:{action} 值估计:{val.item():.3f} 熵:{ent.item():.3f}")
```

---

## 9-14. 练习与路径

**题1：** Actor-Critic 相比 REINFORCE 的优势？

**参考答案：** REINFORCE 用完整回合回报 $G_t$ 作为梯度权重，方差大。Actor-Critic 用 Critic 的 TD 误差 $\delta_t$ 替代，方差显著降低，同时保持无偏（或低偏差）。此外 Actor-Critic 可以在线学习（无需等回合结束）。

### 学习路径
- 前置：策略梯度、TD 学习
- 进阶：A3C（异步）、PPO、SAC
- 推荐：Sutton & Barto, "Reinforcement Learning" 第13章


## 3. 数学公式与推导

Actor-Critic的数学基础：

### 价值函数
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$$

### 贝尔曼方程
$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\left[r + \gamma V^\pi(s')\right]$$

### 策略梯度
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right]$$


## 4. 训练过程讲解
### 训练步骤
1. **环境初始化**：创建RL环境
2. **策略初始化**：初始化策略/值函数
3. **交互采样**：执行动作收集经验
4. **策略评估**：计算回报/优势
5. **策略改进**：更新参数
6. **循环迭代**：重复采样和更新

## 5. 应用场景

Actor-Critic在以下领域有广泛应用：

- 游戏AI（Atari、围棋等）
- 机器人控制与导航
- 自动驾驶决策
- 推荐系统实时决策
- 资源调度与优化

在工业实践中，Actor-Critic通常与完整的数据管道配合使用。选择Actor-Critic时需要根据数据特点、性能要求和计算资源综合考量。

## 6. 优缺点分析

### 优点
1. **理论成熟**：有着坚实的理论基础和大量研究支撑
2. **效果可靠**：在适当场景下能取得稳定优秀的性能
3. **社区支持**：完善的开源实现和活跃社区生态
4. **可解释性**：决策过程在一定程度上可理解和解释
5. **易于使用**：主流框架提供简洁API

### 缺点
1. **数据依赖**：性能高度依赖训练数据质量和数量
2. **超参敏感**：某些超参数对结果影响较大
3. **计算开销**：大规模数据下需要较多计算资源
4. **泛化限制**：分布外数据上表现可能下降
5. **假设约束**：理论假设在实际数据中可能不成立


## 7. 调库实现（Python + 完整代码 + 注释）

以下是使用主流框架实现Actor-Critic的代码：

```python
import numpy as np, gymnasium as gym, random

env = gym.make('CartPole-v1')
alpha, gamma, epsilon = 0.1, 0.99, 0.1
from collections import defaultdict
q = defaultdict(lambda: np.zeros(env.action_space.n))

for ep in range(1000):
    s, _ = env.reset()
    total_r = 0
    done = False
    while not done:
        a = env.action_space.sample() if random.random() < epsilon else int(np.argmax(q[s]))
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        q[s][a] += alpha * (r + gamma * np.max(q[s2]) * (1-done) - q[s][a])
        s, total_r = s2, total_r + r
    if (ep+1) % 100 == 0: print(f"Ep {ep+1}: reward={total_r}")
env.close()
```

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import numpy as np, random
from collections import defaultdict

class ActorCriticAgent:
    def __init__(self, n_act, lr=0.1, gamma=0.99, eps=0.1):
        self.n_act, self.lr, self.gamma, self.eps = n_act, lr, gamma, eps
        self.q = defaultdict(lambda: np.zeros(n_act))
    def act(self, s):
        return random.randint(0,self.n_act-1) if random.random()<self.eps else int(np.argmax(self.q[s]))
    def update(self, s, a, r, s2, done):
        self.q[s][a] += self.lr * (r + self.gamma * np.max(self.q[s2]) * (1-done) - self.q[s][a])
```

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：Actor-Critic与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('Actor-Critic Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 12. 学习总结

### 核心要点
1. **基本原理**：Actor-Critic的核心思想和数学基础
2. **实现方法**：从调库到手工实现
3. **应用场景**：Actor-Critic适合的问题类型
4. **优化技巧**：超参数调优和正则化
5. **评估方法**：客观评估性能

### 关键概念
- Actor-Critic的损失函数设计原理
- 参数优化的数学推导
- 泛化能力与过拟合的平衡

### 进阶方向
掌握Actor-Critic后，可进一步学习相关的进阶方法和变体。


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Actor-Critic的核心思想及适用场景。
<details><summary>参考答案</summary>
Actor-Critic通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Actor-Critic的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Actor-Critic核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Actor-Critic在什么情况下会失效？
2. 训练数据很少时，Actor-Critic还能有效工作吗？
3. 如何将Actor-Critic与其他方法结合？


## 14. 学习路径建议

### 前置知识
概率论、MDP、Python、NumPy

### 学习顺序
1. 先理解原理：掌握Actor-Critic核心思想和数学基础
2. 动手实践：运行代码，观察实验结果
3. 深入理解：阅读原始论文，理解设计动机
4. 项目实战：真实数据集上应用Actor-Critic

### 进阶方向
多智能体RL、RLHF

### 推荐资源
- 搜索Actor-Critic原始论文和综述
- GitHub优秀实现
- Coursera/Stanford相关课程

