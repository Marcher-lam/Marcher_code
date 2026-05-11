# 直接前瞻近似(DLA) 学习文档

> 通过前瞻模拟未来决策场景做当前决策，处理复杂不确定性的最强策略类。

> 来源线索：本节内容根据原书中关于"Direct Lookahead Approximations"的相关章节(Ch 11.6, Ch 19)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：DLA通过模拟未来可能的决策路径来选择当前最优动作，是四种策略类中最强大也最计算密集的方法。

**直觉类比**：下棋时，你在脑中模拟接下来几步的所有可能走法，评估每种走法的结果，然后选择当前最优的一步。这就是直接前瞻。

**历史背景**：前瞻策略自古就有（博弈论中的minimax搜索），现代版本包括蒙特卡洛树搜索(MCTS)和模型预测控制(MPC)。

**算法定位**：策略设计/前瞻策略。是PFA/CFA/VFA/DLA四种策略类中的第四种。

**前置知识**：MDP、蒙特卡洛仿真、树搜索。

## 2. 核心原理

**核心思想**：在决策时刻$t$，构建一个前瞻模型模拟未来$H$步，在模拟空间中求解最优决策序列，但只执行第一步。

**策略中的策略(Policy-within-a-Policy)**：DLA的前瞻模型本身也需要策略（如何近似未来）。这就是"policy-within-a-policy"结构：

$$X^{DLA}(S_t) = \arg\max_{x_t} \mathbb{E}\left[C(S_t, x_t) + \sum_{t'=t+1}^{t+H} C(S_{t'}, X^\pi(S_{t'}))\right]$$

**四种前瞻变体**：
1. **确定性前瞻**：用点预测代替随机变量
2. **随机前瞻**：采样多条路径取平均
3. **参数化前瞻**：在确定性模型上加修正参数
4. **混合前瞻**：组合VFA/CFA/PFA在前瞻中

## 3. 数学公式与推导

### 前瞻模型

在状态$S_t$，求解：

$$\max_{x_t,...,x_{t+H}} \mathbb{E}\sum_{t'=t}^{t+H} C(S_{t'}, x_{t'})$$

$$\text{s.t. } S_{t'+1} = S^M(S_{t'}, x_{t'}, W_{t'+1})$$

### 近似策略

- **结果聚合/采样**：用$N$个场景替代完整期望
- **阶段聚合**：缩短前瞻窗口$H$
- **策略近似**：前瞻内用简单策略（PFA/VFA）

### 与其他策略类的关系

| 策略 | 特点 | 计算量 |
|------|------|--------|
| PFA | 直接映射 | 低 |
| CFA | 修正优化 | 中 |
| VFA | 学习值函数 | 中 |
| DLA | 模拟未来 | 高 |

## 4-8. 核心实现

```python
"""DLA：简单前瞻搜索"""
import numpy as np

class DirectLookahead:
    def __init__(self, n_actions, horizon=3, n_scenarios=20, gamma=0.95):
        self.nA = n_actions
        self.H = horizon
        self.N = n_scenarios
        self.gamma = gamma

    def decide(self, state, sim_fn, reward_fn):
        """前瞻搜索：评估每个动作的期望价值"""
        action_values = np.zeros(self.nA)
        for a in range(self.nA):
            total = 0
            for _ in range(self.N):
                value = self._rollout(state, a, sim_fn, reward_fn, self.H)
                total += value
            action_values[a] = total / self.N
        return np.argmax(action_values)

    def _rollout(self, state, action, sim_fn, reward_fn, depth):
        """蒙特卡洛rollout"""
        r = reward_fn(state, action)
        if depth <= 0:
            return r
        s_next = sim_fn(state, action)
        # 简单策略：随机动作（可用更好策略替代）
        a_next = np.random.randint(self.nA)
        return r + self.gamma * self._rollout(s_next, a_next, sim_fn, reward_fn, depth-1)

if __name__ == "__main__":
    np.random.seed(42)
    dla = DirectLookahead(n_actions=4, horizon=3, n_scenarios=30)
    sim = lambda s, a: np.random.randint(16)
    rew = lambda s, a: 1.0 if (s + a) % 16 == 15 else -0.01
    for s in [0, 5, 10, 14]:
        a = dla.decide(s, sim, rew)
        print(f"状态{s}: DLA选择动作{a}")
```

## 9-14. 简要

### 12. 学习总结
DLA：前瞻模拟$H$步，选当前最优。四种策略类中最强但计算量最大。包含确定性/随机/参数化/混合四种变体。

### 13. 练习题
**Q1**：DLA和MCTS的关系？
**A1**：MCTS是DLA的一种具体实现——用蒙特卡洛采样近似期望，用UCB在搜索树中选择方向。DLA是策略类（框架），MCTS是算法（实例）。

### 14. 学习路径
**前置**：PFA、CFA、VFA | **进阶**：MCTS、MPC、随机规划
**资源**：原书Ch 11.6, Ch 19
