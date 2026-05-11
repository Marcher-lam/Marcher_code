# Rollout启发式策略(Rollout Heuristic Policy) 学习文档

> 用仿真rollout评估动作价值，是MCTS和前瞻策略的核心评估组件。

> 来源线索：本节内容根据原书中关于"Rollout Heuristic"的相关章节(Ch 19.8)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：Rollout策略从当前状态开始仿真到终止（或有限步），用多次仿真的平均回报估计每个动作的价值，选择平均回报最高的动作。

**直觉类比**：你在下棋时犹豫不决。于是你在脑中快速模拟"如果走这步会怎样"——每种走法模拟若干局，看哪种赢的概率最高。这就是Rollout：用快速仿真代替精确计算，估计每个动作的好坏。

**历史背景**：Rollout思想最早由Tesauro & Galperin (1997)在西洋双陆棋中应用。它启发了一系列前瞻算法，包括MCTS（蒙特卡洛树搜索）中的Simulation阶段就是Rollout。原书Ch 19.8将Rollout归为DLA（直接前瞻近似）策略的核心组件。

**算法定位**：直接前瞻/DLA策略。Rollout是DLA的基本实现——通过仿真"前瞻"未来，评估当前动作的价值。

**前置知识**：
- 马尔可夫决策过程（MDP）
- 蒙特卡洛仿真
- 策略评估
- Python编程

## 2. 核心原理

**核心思想**：不做精确的值函数计算，而是通过仿真快速估计。从当前状态$s$出发，对每个候选动作$a$，用基策略$\pi^{base}$快速仿真$N$次，取平均回报作为$Q(s,a)$的估计。

**工作流程**：

1. 在状态$s$，枚举所有候选动作$a_1, ..., a_m$
2. 对每个动作$a$：
   a. 执行$a$，转移到新状态
   b. 用基策略$\pi^{base}$仿真$N$次，每次最多$H$步
   c. 记录每次仿真的累积折扣回报$G^{(i)}$
3. 计算$\hat{Q}(s,a) = \frac{1}{N}\sum_{i=1}^N G^{(i)}$
4. 选择$\arg\max_a \hat{Q}(s,a)$

**关键概念**：

- **基策略$\pi^{base}$**：Rollout中使用的决策规则（贪心、随机、或预训练策略）
- **Rollout深度$H$**：仿真多少步（太浅→偏差大，太深→慢）
- **Rollout次数$N$**：更多rollout减小方差但增加时间
- **偏差-方差权衡**：浅rollout有截断偏差，少rollout有方差

**与MCTS的关系**：

```
MCTS的四个阶段：
1. Selection ──→ 2. Expansion ──→ 3. Simulation(=Rollout!) ──→ 4. Backpropagation
                                       ↑
                              用基策略快速仿真到终止
```

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $s$ | 当前状态 |
| $a$ | 候选动作 |
| $\pi^{base}$ | 基策略 |
| $N$ | Rollout次数 |
| $H$ | Rollout深度 |
| $G^{(i)}(s,a)$ | 第$i$次rollout的累积回报 |
| $\gamma$ | 折扣因子 |

### 动作价值估计

$$\hat{Q}(s, a) = \frac{1}{N}\sum_{i=1}^N G^{(i)}(s, a, \pi^{base})$$

其中$G^{(i)}$是第$i$次rollout从$s$执行$a$后用$\pi^{base}$的累积回报：

$$G^{(i)} = r_0 + \gamma r_1 + \gamma^2 r_2 + ... + \gamma^{H-1} r_{H-1}$$

### 无偏性分析

如果rollout深度$H$足够大（到达终止状态），且基策略是固定的，则$\hat{Q}(s,a)$是$Q^{\pi^{base}}(s,a)$的无偏估计。

### 截断偏差

当$H$不够大时，存在截断偏差：

$$\hat{Q}(s,a) = Q^{\pi^{base}}(s,a) - \underbrace{\gamma^H V^{\pi^{base}}(s_H)}_{\text{截断误差}}$$

截断误差以$\gamma^H$速率指数衰减。

### 方差分析

$$\text{Var}[\hat{Q}(s,a)] = \frac{\sigma^2}{N}$$

其中$\sigma^2$是单次rollout回报的方差。增加$N$以$O(1/\sqrt{N})$速率减小标准差。

### 一步改进定理

Rollout策略$\pi^{rollout}$至少不差于基策略$\pi^{base}$：

$$V^{\pi^{rollout}}(s) \geq V^{\pi^{base}}(s)$$

因为rollout在基策略之上增加了一步优化（选择最优动作而非按基策略行动）。

## 4. 训练过程讲解

### 参数初始化
- 基策略$\pi^{base}$：随机策略或简单贪心
- Rollout次数$N$：通常10-50
- Rollout深度$H$：通常10-100
- 折扣因子$\gamma$：0.9-0.99

### 在线决策过程
1. 在每个决策点，对所有候选动作执行$N$次rollout
2. 计算$\hat{Q}(s,a)$
3. 选择$\arg\max_a \hat{Q}(s,a)$
4. 执行选中的动作，观察转移，进入下一决策点

### 计算预算分配
- 总预算$B = N \times H \times |\mathcal{A}|$
- 浅+多次（$N$大，$H$小）vs 深+少次（$N$小，$H$大）
- 通常浅+多次更稳定

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $N$ | Rollout次数 | [10, 100] | 20 |
| $H$ | Rollout深度 | [10, 100] | 30 |
| $\gamma$ | 折扣因子 | [0.9, 0.99] | 0.95 |
| $\pi^{base}$ | 基策略 | 随机/贪心 | 随机 |

## 5. 应用场景

### 1. 棋类游戏
为什么适合：游戏规则明确，仿真快速。Rollout评估每个落子的胜率，选择最优落子。AlphaGo的早期版本就用了Rollout。

### 2. 路径规划
为什么适合：从当前位置模拟多条路径，选平均最快的。Rollout深度对应规划窗口。

### 3. 资源调度
为什么适合：从当前状态模拟多种调度方案，选平均收益最高的。

### 4. 机器人控制
为什么适合：物理仿真可以快速执行Rollout，评估不同控制动作的效果。

### 不适用场景
- 仿真非常慢（如大规模网络仿真）
- 状态空间连续且维度高（需要函数近似）
- 实时性要求极高（rollout时间不够）

## 6. 优缺点分析

### 优点
1. **无需模型**：只需要仿真器，不需要解析模型（成立条件：有可用的快速仿真器）
2. **非参数**：不做函数近似假设，直接估计
3. **保证改进**：一步改进定理保证不差于基策略
4. **实现简单**：核心代码不到50行

### 缺点
1. **计算量大**：每个决策点需要$N \times H \times |\mathcal{A}|$次仿真
2. **基策略依赖**：基策略太差则rollout估计也差
3. **方差问题**：高方差环境需要大量rollout
4. **不可并行化**（在单次决策内）：需要串行评估多个动作

### 算法对比

| 特性 | Rollout | MCTS | 值迭代 | 策略梯度 |
|------|---------|------|--------|---------|
| 需要模型 | 仿真器 | 仿真器 | 完整模型 | 仿真器 |
| 在线计算 | 是 | 是 | 否 | 否 |
| 计算量/步 | 高 | 高 | 一次性 | 高(训练) |
| 理论保证 | 一步改进 | 渐近最优 | 最优 | 局部最优 |
| 适用规模 | 小-中 | 中 | 小 | 大 |

## 7. 调库实现

```python
"""
Rollout启发式策略：完整的网格世界示例
"""
import numpy as np

class RolloutPolicy:
    """Rollout启发式策略"""

    def __init__(self, n_actions, gamma=0.95, n_rollouts=20, max_depth=30):
        self.nA = n_actions
        self.gamma = gamma
        self.N = n_rollouts
        self.H = max_depth

    def decide(self, state, sim_fn, reward_fn, done_fn, base_policy=None):
        """用rollout评估每个动作，返回最优动作"""
        if base_policy is None:
            base_policy = lambda s: np.random.randint(self.nA)

        q_values = np.zeros(self.nA)
        for a in range(self.nA):
            returns = []
            for _ in range(self.N):
                G = 0
                s = state
                # 执行动作a
                r = reward_fn(s, a)
                G = r
                s = sim_fn(s, a)
                # Rollout with base policy
                for d in range(1, self.H):
                    if done_fn(s):
                        break
                    a_base = base_policy(s)
                    r = reward_fn(s, a_base)
                    G += (self.gamma ** d) * r
                    s = sim_fn(s, a_base)
                returns.append(G)
            q_values[a] = np.mean(returns)
        return np.argmax(q_values), q_values

    def decide_batch(self, states, sim_fn, reward_fn, done_fn, base_policy=None):
        """批量决策：对多个状态同时执行rollout"""
        results = []
        for s in states:
            a, q = self.decide(s, sim_fn, reward_fn, done_fn, base_policy)
            results.append((a, q))
        return results


if __name__ == "__main__":
    np.random.seed(42)
    # 简单网格世界：4x4，目标是右下角(15)
    size = 4
    sim = lambda s, a: min(max(0, s + [1, -1, size, -size][a]), size*size-1)
    rew = lambda s, a: 1.0 if sim(s, a) == 15 else -0.01
    done = lambda s: s == 15

    rollout = RolloutPolicy(n_actions=4, n_rollouts=30, max_depth=20)

    for s in [0, 5, 10, 14]:
        a, q = rollout.decide(s, sim, rew, done)
        print(f"状态{s}: Rollout选动作{a}, Q值={q.round(3)}")
```

## 8. 手工代码实现

```python
"""
从零实现Rollout策略（纯NumPy）
包含完整的网格世界环境和对比实验
"""
import numpy as np

class GridWorld:
    """4x4网格世界"""
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.goal = size * size - 1

    def step(self, state, action):
        row, col = state // self.size, state % self.size
        if action == 0: row = max(row - 1, 0)      # 上
        elif action == 1: col = min(col + 1, self.size-1)  # 右
        elif action == 2: row = min(row + 1, self.size-1)  # 下
        elif action == 3: col = max(col - 1, 0)      # 左
        next_state = row * self.size + col
        reward = 1.0 if next_state == self.goal else -0.01
        done = next_state == self.goal
        return next_state, reward, done


class RolloutAgent:
    """Rollout策略智能体"""

    def __init__(self, env, n_rollouts=20, max_depth=30, gamma=0.95):
        self.env = env
        self.N = n_rollouts
        self.H = max_depth
        self.gamma = gamma

    def act(self, state):
        """用rollout选择最优动作"""
        q_values = np.zeros(self.env.n_actions)
        for a in range(self.env.n_actions):
            returns = []
            for _ in range(self.N):
                G = 0
                s = state
                s, r, done = self.env.step(s, a)
                G = r
                for d in range(1, self.H):
                    if done: break
                    a_base = np.random.randint(self.env.n_actions)
                    s, r, done = self.env.step(s, a_base)
                    G += (self.gamma ** d) * r
                returns.append(G)
            q_values[a] = np.mean(returns)
        return np.argmax(q_values), q_values

    def evaluate(self, n_episodes=100):
        """评估rollout策略"""
        total_rewards = []
        for _ in range(n_episodes):
            state = 0
            ep_reward = 0
            for step in range(50):
                action, _ = self.act(state)
                state, reward, done = self.env.step(state, action)
                ep_reward += reward
                if done: break
            total_rewards.append(ep_reward)
        return np.mean(total_rewards), np.std(total_rewards)


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)
    env = GridWorld(size=4)
    agent = RolloutAgent(env, n_rollouts=30, max_depth=20)

    print("=== Rollout策略 ===")
    for s in [0, 5, 10, 14]:
        a, q = agent.act(s)
        arrows = ['↑', '→', '↓', '←']
        print(f"状态{s}: 选{arrows[a]}, Q值={q.round(3)}")

    mean_r, std_r = agent.evaluate(n_episodes=50)
    print(f"\n50局评估: 平均奖励={mean_r:.3f} ± {std_r:.3f}")
```

## 9. 可视化与结果理解

```python
"""Rollout可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_rollout_convergence(env_state, rollout_agent, n_range=range(5, 51, 5)):
    """展示rollout次数对Q值估计的影响"""
    q_means = []
    q_stds = []
    for n in n_range:
        old_N = rollout_agent.N
        rollout_agent.N = n
        _, q = rollout_agent.act(env_state)
        q_means.append(q.max())
        q_stds.append(q.std())
        rollout_agent.N = old_N

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(n_range, q_means, yerr=q_stds, marker='o')
    ax.set_xlabel('Rollout次数 $N$')
    ax.set_ylabel('$\\hat{Q}(s, a^*)$')
    ax.set_title('Rollout估计随仿真次数的收敛')
    ax.grid(True, alpha=0.3)
    plt.savefig('rollout_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**结果解读**：
- Q值估计随rollout次数增加趋于稳定
- 标准差以$O(1/\sqrt{N})$速率减小
- 少量rollout（$N=10$）已可给出合理估计，但$N \geq 30$更稳定

## 10. 模型评估

```python
"""Rollout策略评估"""
import numpy as np

def compare_policies(rollout_agent, random_agent, env, n_episodes=100):
    """对比Rollout策略和随机策略"""
    # Rollout策略
    r_rewards = []
    for _ in range(n_episodes):
        s = 0; total = 0
        for _ in range(50):
            a, _ = rollout_agent.act(s)
            s, r, d = env.step(s, a)
            total += r
            if d: break
        r_rewards.append(total)

    # 随机策略
    rand_rewards = []
    for _ in range(n_episodes):
        s = 0; total = 0
        for _ in range(50):
            a = np.random.randint(env.n_actions)
            s, r, d = env.step(s, a)
            total += r
            if d: break
        rand_rewards.append(total)

    print(f"{'策略':<10} {'平均奖励':>8} {'标准差':>8} {'成功率':>8}")
    print("-" * 38)
    print(f"{'Rollout':<10} {np.mean(r_rewards):>8.3f} {np.std(r_rewards):>8.3f} "
          f"{sum(r>0 for r in r_rewards)/n_episodes:>8.1%}")
    print(f"{'随机':<10} {np.mean(rand_rewards):>8.3f} {np.std(rand_rewards):>8.3f} "
          f"{sum(r>0 for r in rand_rewards)/n_episodes:>8.1%}")
```

## 11. 常见问题与易错点

### 数据层面

1. **基策略太差**
   - 现象：Rollout估计的Q值都很低且相似
   - 原因：基策略（如随机策略）在很多环境下表现极差
   - 解决方案：使用贪心策略或预训练策略作为基策略

2. **Rollout深度不足**
   - 现象：Q值估计系统性偏低
   - 原因：截断偏差$\gamma^H V(s_H)$被丢弃
   - 解决方案：增加$H$或添加终态值估计$V(s_H)$

### 模型层面

3. **计算时间过长**
   - 现象：每个决策点耗时过长
   - 原因：$N \times H \times |\mathcal{A}|$过大
   - 解决方案：并行化rollout、减小$H$或使用approximation

4. **高方差环境**
   - 现象：Q值估计波动大，决策不稳定
   - 原因：环境随机性高
   - 解决方案：增加$N$、使用variance reduction技术

### 调参层面

5. **N和H的平衡**
   - 现象：不知道该增加$N$还是$H$
   - 解决方案：总预算$B = N \times H$固定时，浅+多次（$N$大$H$小）通常优于深+少次

## 12. 学习总结

Rollout用仿真估计动作价值：$\hat{Q}(s,a) = \frac{1}{N}\sum G(s,a,\pi^{base})$。核心三要素是基策略$\pi^{base}$、rollout次数$N$和深度$H$。一步改进定理保证Rollout策略不差于基策略。

**关键公式**：
1. 动作价值估计：$\hat{Q}(s,a) = \frac{1}{N}\sum_{i=1}^N G^{(i)}$
2. 截断偏差：$\gamma^H V(s_H)$
3. 方差：$\sigma^2/N$

Rollout是MCTS的Simulation阶段，也是DLA策略的基础实现。它的简单性使其成为在线决策的实用工具。

## 13. 练习题与思考题

### 基础题

**题目1**：rollout次数$N$和深度$H$如何平衡？计算预算有限时应优先增加哪个？

**参考答案**：增加$N$减小方差（$O(1/\sqrt{N})$），增加$H$减小偏差（截断误差$\gamma^H$）。计算预算有限时，通常优先增加$N$（减小方差）而非$H$（减小偏差），因为方差不减小时增加$H$只是更精确地估计一个含噪量。

**题目2**：证明Rollout策略至少不差于基策略（一步改进定理）。

**参考答案**：设基策略为$\pi^{base}$。Rollout策略$\pi^{rollout}$在每步选择$\arg\max_a Q^{\pi^{base}}(s,a)$而非$\pi^{base}(s)$。因为$\max_a Q^{\pi^{base}}(s,a) \geq Q^{\pi^{base}}(s, \pi^{base}(s))$，所以$V^{\pi^{rollout}}(s) \geq V^{\pi^{base}}(s)$。直观理解：rollout多做了一步优化（在基策略的值函数上贪心）。

### 进阶题

**题目3**：原书(Ch 19.8)提到"multi-step lookahead"。一步rollout（只优化一步）和多步rollout（在rollout内也做优化）有什么区别？

**参考答案**：
- 一步rollout：只在当前步做优化（选$\arg\max_a$），之后用基策略仿真
- 多步rollout：在rollout的每一步都做优化（嵌套rollout），计算量指数增长
- 多步rollout效果更好但计算量爆炸（$|\mathcal{A}|^d$，$d$是前瞻步数）
- MCTS是高效的折中：在树内做选择性扩展（非穷举），在树外用rollout

### 开放思考题

**题目4**：Rollout在什么意义上是"最优"的？它的局限性是什么？

**参考答案方向**：
- Rollout在一步前瞻意义下最优：给定基策略的值函数，选择最优动作
- 局限性：(1)只保证一步改进，不保证全局最优；(2)计算量随候选动作数线性增长；(3)基策略质量是天花板；(4)高方差环境需要大量rollout
- 改进方向：(1)用更好的基策略（如训练好的神经网络）；(2)MCTS在rollout基础上增加树搜索；(3)用variance reduction（如common random numbers）

## 14. 学习路径建议

**前置算法**：
- 马尔可夫决策过程（MDP）
- 蒙特卡洛仿真
- 策略评估

**平行算法**：
- MCTS（蒙特卡洛树搜索）—— Rollout + 树搜索
- 值迭代 —— 精确版的"前瞻"

**进阶算法**：
- MCTS和乐观MCTS —— 更高效的前瞻
- 两阶段随机规划 —— 规划版的前瞻
- 原书Ch 19.8的DLA策略族

**推荐资源**：
1. Powell, W.B. "Reinforcement Learning and Stochastic Optimization" Ch 19.8 —— DLA策略中的Rollout
2. Bertsekas, D.P. "Rollout Algorithms for Discrete Optimization: A Survey" (2013) —— Rollout算法综述
3. Tesauro, G. & Galperin, G. "On-line Policy Improvement using Monte-Carlo Search" (1997) —— Rollout的先驱论文
