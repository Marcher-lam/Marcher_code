#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版批量生成 - 生成更详细的文档
为每个算法生成包含完整14章节的详细文档
"""

import re
from pathlib import Path
import os

def sanitize_filename(name):
    """清理文件名"""
    name = name.replace('/', '_').replace('\\', '_')
    name = re.sub(r'[\\:*?"<>|]', '', name)
    return name.strip()

def generate_detailed_doc(algo_name, category):
    """生成详细文档 - 根据算法类别填充内容"""
    
    # 根据类别选择详细模板
    if category == "TD":
        return gen_td_detailed(algo_name)
    elif category == "MC":
        return gen_mc_detailed(algo_name)
    elif category == "DP":
        return gen_dp_detailed(algo_name)
    elif category == "Deep":
        return gen_deep_detailed(algo_name)
    elif category == "Model":
        return gen_model_detailed(algo_name)
    elif category == "FA":
        return gen_fa_detailed(algo_name)
    elif category == "Exploration":
        return gen_exp_detailed(algo_name)
    else:
        return gen_other_detailed(algo_name)

def gen_td_detailed(algo_name):
    """生成TD类算法的详细文档"""
    is_q_learning = "Q学习" in algo_name or "Q(" in algo_name
    is_sarsa = "Sarsa" in algo_name
    is_td_lambda = "λ" in algo_name or "lambda" in algo_name.lower()
    
    # 基础详细模板
    doc = f"""# {algo_name} 学习文档

> {algo_name}是基于时序差分的强化学习算法，{"通过Q表格和TD学习找到最优策略" if is_q_learning else "通过时间差分更新价值函数"}。

---

## 1. 算法基础认知!

**一句话定义**：{algo_name}通过{"维护Q表格并使用TD误差更新，最终学到最优策略" if is_q_learning else "时间差分学习更新价值估计，结合蒙特卡洛和动态规划的优点"}。

**直觉类比**：想象你在走迷宫，每走一步就根据是否接近目标来修正对各个位置距离目标的估计。{algo_name}就是这种"边走边修正"的学习方法，它使用时间差分误差来不断调整价值估计。

**历史背景**：{algo_name}是强化学习领域的重要算法。{"Q-learning由Watkins于1989年提出，是第一个被证明收敛到最优策略的off-policy算法。" if is_q_learning else "时序差分学习由Sutton于1988年提出，成为强化学习三大基石之一（与动态规划、蒙特卡洛方法并列）。"}{algo_name}进一步发展了TD学习的理论和实践。

**算法定位**：
- 类型：强化学习 → {"控制（Control）" if is_q_learning or is_sarsa else "预测（Prediction）"}
- 输出：{"动作价值 Q(s,a)" if is_q_learning or is_sarsa else "状态价值 V(s)"}
- 模型类型：{"非参数模型（表格型）或参数模型（函数逼近）" if is_q_learning or is_sarsa else "非参数模型（表格型）"}
- On/Off Policy：{"Off-policy（可以学习与实际执行不同的策略）" if is_q_learning else "On-policy（学习实际执行的策略）" if is_sarsa else "N/A（预测算法）"}

**前置知识**：
- 马尔可夫决策过程（MDP）：状态、动作、奖励、转移概率的概念
- 贝尔曼方程：价值函数的递归关系，{algo_name}的理论基础
- {"Q-learning/Sarsa基础：理解动作价值函数的含义" if is_q_learning or is_sarsa else "TD学习基础：理解时间差分和bootstrap"}
- Python编程和NumPy使用：实现算法需要
- 基本概率论：理解期望、随机过程

---

## 2. 核心原理!

### 2.1 核心思想!

{algo_name}的核心思想是：{"通过智能体与环境的交互，不断更新对状态-动作价值的估计（Q值），最终学到最优策略。关键在于它使用下一个状态的最大Q值（Q-learning）或实际下一个动作（Sarsa）来更新当前Q值，使得能够学习到最优策略。" if is_q_learning or is_sarsa else "通过时间差分学习，结合蒙特卡洛的无偏性和动态规划的自举特性，高效估计价值函数。使用bootstrap（自举）方法，用当前估计值来更新估计值。"}

核心思想可以概括为：{"通过时间差分学习和贝尔曼最优方程，在状态-动作空间中迭代更新Q值，最终收敛到最优动作价值函数Q*。" if is_q_learning or is_sarsa else "通过TD误差不断更新价值估计，最终收敛到真实价值函数。"}

### 2.2 工作流程!

1. **初始化**：{"初始化Q表格（状态-动作价值表）" if is_q_learning or is_sarsa else "初始化价值函数V(s)"}
   - 输入：{"状态空间S、动作空间A、学习率α、折扣因子γ" if is_q_learning or is_sarsa else "状态空间S、学习率α、折扣因子γ"}
   - 输出：初始化的{"Q表格（通常初始化为0或小的随机值）" if is_q_learning or is_sarsa else "价值函数V(s)（通常初始化为0）"}

2. **交互循环**：智能体与环境交互
   - 观察当前状态s
   - 根据{"ε-greedy策略" if is_q_learning or is_sarsa else "当前策略π"}选择动作a
   - 执行动作，得到奖励r和下一个状态s'
   - 关键操作：{"根据贝尔曼方程更新Q(s,a)：Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]（Q-learning）" if is_q_learning else "更新价值函数：V(s) ← V(s) + α[r + γV(s') - V(s)]"}

3. **终止条件**：{"episode结束（如游戏结束、机器人到达目标、达到最大步数）" if is_q_learning or is_sarsa else "价值函数收敛或达到最大迭代次数"}

### 2.3 关键概念解释!

- **{"Q值（动作价值）" if is_q_learning or is_sarsa else "V值（状态价值）"}**：{"在状态s执行动作a后，按照某策略继续下去能获得的期望回报" if is_q_learning or is_sarsa else "在状态s下，按照某策略继续下去能获得的期望回报"}
- **TD误差**：{"δ = r + γ max_a' Q(s',a') - Q(s,a)（Q-learning）" if is_q_learning else "δ = r + γV(s') - V(s)"}
- **{"Off-policy（仅Q-learning）" if is_q_learning else "On-policy（仅Sarsa）" if is_sarsa else "Bootstrap"}**：{"Q-learning学习的是最优策略，不受实际行为策略限制" if is_q_learning else "Sarsa学习的是实际执行的策略" if is_sarsa else "使用当前估计值来更新估计值"}
- **ε-greedy探索**：以ε概率随机探索，以1-ε概率贪心利用当前最优动作
- **贝尔曼方程**：{"Q*(s,a) = E[r + γ max_a' Q*(s',a') | s,a]" if is_q_learning else "V(s) = E[r + γV(s')]"}，{algo_name}更新的理论基础

### 2.4 几何/直观解释!

{"Q-learning可以在状态-动作空间中看作是在不断'填色'：每个状态-动作对的价值逐渐被填充为真实的价值。通过多次访问和更新，整个Q表格会收敛到最优Q*。" if is_q_learning else "TD学习的更新可以看作是在时间维度上的'纠错'：每次得到一个奖励后，算法会比较之前的预测和实际结果，然后调整预测使其更准确。这类似于在走迷宫时，每走一步就根据是否接近目标来修正对各个位置距离目标的估计。"}

{"对于Sarsa，由于它使用实际执行的下一个动作来更新，它学习的是实际策略的价值，而不是最优策略。这在安全关键应用中更有优势，因为评估的是实际会执行的策略。" if is_sarsa else ""}

{"TD(λ)通过引入资格迹（eligibility trace），将TD(0)的单步更新和蒙特卡洛的完整轨迹信息结合起来。λ参数控制两者的权衡：λ=0时退化为TD(0)，λ=1时接近蒙特卡洛方法。" if is_td_lambda else ""}

---

## 3. 数学公式与推导!

### 3.1 符号约定!

| 符号 | 含义 |
|------|------|
| $S$ | 状态集合 |
| $A$ | 动作集合 |
| $R$ | 奖励 |
| $\gamma$ | 折扣因子 |
| $\alpha$ | 学习率 |
| ${"Q(s,a)$" if is_q_learning or is_sarsa else "V(s)$"} | {"动作价值函数" if is_q_learning or is_sarsa else "状态价值函数"} |

### 3.2 问题形式化!

给定马尔可夫决策过程 $M = \\langle S, A, P, R, \\gamma \\rangle$，我们的目标是找到{"最优策略 $\pi^*$ 使得期望回报最大" if is_q_learning or is_sarsa else "价值函数 V(s) 使得TD误差最小"}。

### 3.3 目标函数/损失函数!

{"对于Q-learning，目标是最小化TD误差的平方：" if is_q_learning else "目标是最小化TD误差："}
$$ L({"Q" if is_q_learning else "V"}) = \\mathbb{E}_{s,a,r,s'} \\left[ \\left( {"r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)" if is_q_learning else "r + \\gamma V(s') - V(s)"} \\right)^2 \\right] $$

**为什么选择这个损失函数？**
- TD误差衡量了当前估计与Bootstrap估计之间的差距
- 平方损失是连续可微的，便于梯度计算（虽然表格型不用梯度）
- 在表格型情况下，这等价于动态规划中的贝尔曼方程的固定点迭代

### 3.4 推导过程!

**Step 1：{"贝尔曼最优方程" if is_q_learning else "贝尔曼方程"}**

{"最优动作价值函数满足：" if is_q_learning else "状态价值函数满足："}
$$ {"Q^*(s,a)" if is_q_learning else "V(s)"} = \\mathbb{E} \\left[ {"r + \\gamma \\max_{a'} Q^*(s',a')" if is_q_learning else "r + \\gamma V(s')"} \\mid s{"，a" if is_q_learning else ""} \\right] $$

**Step 2：样本近似**

在实际应用中，我们用样本均值代替期望：

$$ {"Q(s,a)" if is_q_learning else "V(s)"} \\leftarrow {"Q(s,a)" if is_q_learning else "V(s)"} + \\alpha \\left[ {"r + \\gamma \\max_{a'} Q(s',a')" if is_q_learning else "r + \\gamma V(s')"} - {"Q(s,a)" if is_q_learning else "V(s)"} \\right] $$

### 3.5 最终解/算法步骤!

**{"Q-learning" if is_q_learning else "TD学习"}算法**：

```
初始化 {"Q(s,a)" if is_q_learning or is_sarsa else "V(s)"} 任意值（通常为0）
对于每个episode：
    初始化状态 s
    重复直到终止：
        根据{"ε-greedy" if is_q_learning or is_sarsa else "当前策略"}选择动作 a
        执行a，观察 r, s'
        {"Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]" if is_q_learning else "V(s) ← V(s) + α[r + γV(s') - V(s)]"}
        s ← s'
```

---

## 4. 训练过程讲解!

### 4.1 数据预处理!

**必要预处理**：
1. **状态表示**：
   - 离散状态：可以直接作为{"Q表格" if is_q_learning or is_sarsa else "V函数"}的索引
   - 连续状态：需要离散化或使用函数逼近（如神经网络）
   - 代码示例：
     ```python
     import numpy as np
     def discretize_state(state, state_ranges, bins_per_dim=10):
         discrete_state = []
         for i, (low, high) in enumerate(state_ranges):
             normalized = (state[i] - low) / (high - low)
             bin_idx = int(normalized * bins_per_dim)
             bin_idx = np.clip(bin_idx, 0, bins_per_dim - 1)
             discrete_state.append(bin_idx)
         return tuple(discrete_state)
     ```

2. **奖励设计**：
   - 稀疏奖励：只在关键节点给奖励
   - 密集奖励：每步都给反馈
   - 奖励塑形：添加中间奖励引导学习

### 4.2 参数初始化!

- **{"Q表格" if is_q_learning or is_sarsa else "V函数"}初始化**：通常初始化为0或小的随机值
- **理由**：零初始化简单且能保证在表格型情况下收敛

### 4.3 迭代过程!

```python
import numpy as np
import gymnasium as gym

# 训练循环示例
for episode in range(1000):
    state, _ = env.reset()
    done = False
    while not done:
        action = {"np.argmax(Q[state])" if is_q_learning else "policy.sample()"}
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 更新逻辑
        {"td_target = reward + 0.99 * np.max(Q[next_state]) * (not done)\n        td_error = td_target - Q[state][action]\n        Q[state][action] += 0.01 * td_error" if is_q_learning else "td_error = reward + 0.99 * V[next_state] * (not done) - V[state]\n        V[state] += 0.01 * td_error"}
        
        state = next_state
```

### 4.4 收敛条件!

- {"Q值" if is_q_learning or is_sarsa else "V值"}变化小于阈值（如1e-4）
- 达到最大episode数
- TD误差接近0

### 4.5 超参数及推荐范围!

| 超参数 | 推荐范围 | 默认值 |
|--------|----------|--------|
| $\alpha$ | 0.001-0.1 | 0.01 |
| $\gamma$ | 0.9-0.999 | 0.99 |
| $\epsilon$ | 0.01-0.3 | 0.1 |

---

## 5. 应用场景!

### 5.1 典型应用!

**应用1：游戏AI**
- 问题类型：序贯决策控制
- 为什么适合{"Q-learning" if is_q_learning else "TD学习"}：有明确的状态、动作、奖励定义
- 实际案例：{"DQN玩Atari游戏、AlphaGo" if is_q_learning else "TD-Gammon"}

**应用2：机器人控制**
- 问题类型：连续/离散控制
- 为什么适合：能处理高维状态空间

### 5.2 适用数据特征!

- 特征类型：{"离散或连续状态" if is_q_learning or is_sarsa else "离散状态"}
- 环境特性：需要能够多次交互采样

### 5.3 不适用场景!

1. 无法多次试错的任务
2. {"状态/动作空间极大且无有效泛化方法" if is_q_learning or is_sarsa else "状态空间极大"}
3. 需要可解释性的关键决策场景

---

## 6. 优缺点分析!

### 6.1 优点!

1. **{"无需环境模型" if is_q_learning or is_sarsa else "理论基础扎实"}**：{"Q-learning是model-free算法，不需要知道状态转移概率" if is_q_learning else "TD学习基于贝尔曼方程，有严格的理论保证"}
2. **可处理中等规模问题**：在状态空间不大时，表格型方法简单有效
3. **理论保证**：在表格型情况下，满足Robbins-Monro条件可保证收敛

### 6.2 缺点!

1. **样本效率低**：需要大量交互才能学到好策略
2. **超参数敏感**：学习率、折扣因子等超参数对性能影响大
3. **{"存在过估计偏差" if is_q_learning else "有偏差（bootstrap）"}**：{"Q-learning使用max操作，倾向于过估计Q值" if is_q_learning else "TD方法使用bootstrap，存在偏差"}

### 6.3 与同类算法对比!

| 维度 | {algo_name} | {"Q-learning" if is_sarsa else "Sarsa"} | Monte Carlo |
|------|---------|-----------|---------|
| 样本效率 | {"中等" if is_q_learning or is_sarsa else "低"} | {"中等" if is_sarsa else "中等"} | 低 |
| 收敛性 | {"保证收敛（表格型）" if is_q_learning or is_sarsa else "可能不收敛"} | 保证收敛 | 保证收敛 |

---

## 7. 调库实现!

### 7.1 环境准备!

```bash
pip install gymnasium numpy matplotlib
```

### 7.2 完整代码示例!

```python
"""
{algo_name} 调库实现示例
环境：CartPole-v1
目标：学习平衡杆的策略
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, n_states, n_actions):
        {"self.Q = np.zeros((n_states, n_actions))" if is_q_learning or is_sarsa else "self.V = np.zeros(n_states)"}
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint({"n_actions" if is_q_learning or is_sarsa else "2"})
        else:
            {"return np.argmax(self.Q[state])" if is_q_learning or is_sarsa else "return 0"}
    
    def update(self, s, a, r, s_next, done):
        {"td_target = r + 0.99 * np.max(self.Q[s_next]) * (not done)\n        td_error = td_target - self.Q[s][a]\n        self.Q[s][a] += 0.01 * td_error" if is_q_learning else "# 更新逻辑"}
        pass

# 主程序
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    {"agent = Agent(10, env.action_space.n)" if is_q_learning or is_sarsa else "agent = Agent(10)"}
    
    print("开始训练...")
    for episode in range(1000):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {{episode}}, Total Reward: {{total_reward}}")
    
    print("训练完成！")
```

---

## 8. 手工代码实现!

### 8.1 核心算法手写!

```python
# 简化版手工实现
import numpy as np

class Tabular{algo_name.replace(' ', '').replace('(','').replace(')','')}:
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99):
        {"self.Q = np.zeros((n_states, n_actions))" if is_q_learning or is_sarsa else "self.V = np.zeros(n_states)"}
        self.lr = lr
        self.gamma = gamma
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint({"n_actions" if is_q_learning or is_sarsa else "2"})
        else:
            {"return np.argmax(self.Q[state])" if is_q_learning or is_sarsa else "return 0"}
    
    def update(self, s, a, r, s_next, done):
        {"td_target = r + self.gamma * np.max(self.Q[s_next]) * (not done)\n        td_error = td_target - self.Q[s][a]\n        self.Q[s][a] += self.lr * td_error" if is_q_learning else "# 更新逻辑"}
    
    def train(self, env, episodes=500):
        rewards = []
        for ep in range(episodes):
            s = env.reset()[0]
            total = 0
            done = False
            while not done:
                a = self.choose_action(s)
                result = env.step(a)
                if len(result) == 4:
                    s_next, r, done, _ = result
                else:
                    s_next, r, terminated, truncated, _ = result
                    done = terminated or truncated
                self.update(s, a, r, s_next, done)
                s = s_next
                total += r
            rewards.append(total)
        return rewards
```

---

## 9. 可视化与结果理解!

### 9.1 关键参数可视化!

```python
import matplotlib.pyplot as plt

def plot_training_curve(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('{algo_name} Training Curve')
    plt.grid(True)
    plt.show()

# plot_training_curve(rewards)
```

---

## 10. 模型评估!

### 10.1 评估指标选择!

- 累计奖励：直接衡量策略性能
- 平均奖励：稳定性能评估

### 10.2 评估代码!

```python
def evaluate(agent, env, runs=10):
    scores = []
    for _ in range(runs):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a = {"np.argmax(agent.Q[s])" if is_q_learning or is_sarsa else "agent.choose_action(s, 0)"}
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = s_next
            total += r
        scores.append(total)
    print(f"Average: {{np.mean(scores):.2f}} +/- {{np.std(scores):.2f}}")
```

---

## 11. 常见问题与易错点!

### 11.1 数据层面常见错误!

**错误1：状态空间未正确离散化**
- 现象：学习速度极慢或完全不收敛
- 解决方案：使用离散化或函数逼近

**错误2：奖励设计不合理**
- 现象：智能体学不到有效策略
- 解决方案：设计合理的奖励函数

### 11.2 模型层面常见错误!

**错误1：探索不足**
- 现象：训练停滞
- 解决方案：使用适当的探索策略

**错误2：学习率设置不当**
- 现象：震荡不收敛或学习太慢
- 解决方案：使用自适应学习率

---

## 12. 学习总结!

### 12.1 核心要点回顾!

✓ **核心思想**：{algo_name}通过{"TD误差更新Q值，最终学到最优策略" if is_q_learning or is_sarsa else "TD误差更新价值函数，最终收敛到真实价值"}
✓ **数学本质**：基于贝尔曼方程的学习方法
✓ **优化目标**：最大化期望累计折扣回报
✓ **适用场景**：具有序贯决策特性的任务
✓ **局限性**：样本效率低，需要大量交互

### 12.2 关键公式汇总!

1. 更新公式：$$ {"Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]" if is_q_learning else "V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]"} $$
2. TD误差：$$ \\delta = {"r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)" if is_q_learning else "r + \\gamma V(s') - V(s)"} $$

### 12.3 最佳实践!

- ✓ 合理设计奖励函数
- ✓ 使用ε-greedy平衡探索与利用
- ✓ 监控训练曲线，及时调整超参数

### 12.4 与其他算法的联系!

- **前置算法**：多臂赌博机、动态规划基础
- **后续算法**：{"DQN、策略梯度方法" if is_q_learning else "Q-learning、Sarsa"}
- **相关算法**：同类算法对比

---

## 13. 练习题与思考题!

### 13.1 基础练习!

**练习1：概念理解**

问题：{algo_name}中的核心更新公式是什么？
A. {"Q(s,a) <- Q(s,a) + alpha * TD_error" if is_q_learning or is_sarsa else "V(s) <- V(s) + alpha * TD_error"}
B. 其他公式
C. 以上都有可能

**答案**：A

---

### 13.2 进阶思考!

**思考1：改进分析**

问题：{algo_name}在状态空间巨大时效果不佳，如何改进？

**答案**：
1. 使用函数逼近（线性、神经网络）
2. 使用经验回放提高样本效率
3. 改进探索策略

---

### 13.3 开放思考!

**思考2：创新应用**

问题：如何将{algo_name}应用到推荐系统？

**答案**：
1. 状态=用户画像、历史行为
2. 动作=推荐内容
3. 奖励=用户点击、停留时间等

---

## 14. 学习路径建议!

### 14.1 前置知识!

- [ ] **概率论**：条件概率、期望
- [ ] **线性代数**：向量、矩阵运算
- [ ] **Python基础**：数据类型、函数、类
- [ ] **强化学习基础**：MDP、贝尔曼方程

### 14.2 平行算法!

1. **{"Sarsa" if is_q_learning else "Q-learning"}**：{"On-policy" if is_q_learning else "Off-policy"}版本的TD控制算法
2. **蒙特卡洛方法**：基于完整轨迹的估计方法

### 14.3 进阶算法!

**短期目标**：
1. **DQN**：深度Q网络
2. **策略梯度方法**：直接优化策略

**长期目标**：
1. **深度强化学习**：DDPG、PPO、A3C
2. **前沿研究**：离线RL、元学习

### 14.4 推荐资源!

**教材**：
1. 《强化学习（第二版）》Sutton & Barto
2. 《深入浅出强化学习》

**在线课程**：
1. David Silver的强化学习课程
2. CS285：深度强化学习

---

**文档结束**

> 如果你觉得这个文档对你有帮助，请分享给更多学习强化学习的人！
> 如有错误或建议，欢迎指出，共同完善！
"""
    return doc

def gen_mc_detailed(algo_name):
    """生成MC类算法的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "蒙特卡洛").replace("TD(", "MC(").replace("bootstrap", "完整轨迹")

def gen_dp_detailed(algo_name):
    """生成DP类算法的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "动态规划").replace("bootstrap", "模型")

def gen_deep_detailed(algo_name):
    """生成Deep类的详细文档"""
    return gen_td_detailed(algo_name).replace("Q学习", algo_name).replace("表格型", "深度神经网络")

def gen_model_detailed(algo_name):
    """生成Model类的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "模型学习").replace("bootstrap", "规划")

def gen_fa_detailed(algo_name):
    """生成FA类的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "函数逼近").replace("bootstrap", "逼近")

def gen_exp_detailed(algo_name):
    """生成Exploration类的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "探索策略").replace("bootstrap", "探索")

def gen_other_detailed(algo_name):
    """生成Other类的详细文档"""
    return gen_td_detailed(algo_name).replace("时序差分", "相关").replace("bootstrap", "方法")

def main():
    """主函数：生成所有算法的详细文档"""
    output_dir = Path("/Users/marcher/Desktop/Marcher_code/强化学习_第二版")
    
    # 算法列表及其类别
    algorithms = [
        # TD类
        ("Q学习", "TD"), ("Sarsa", "TD"), ("TD学习", "TD"), ("TD(0)", "TD"), ("TD(λ)", "TD"),
        ("期望Sarsa", "TD"), ("n步自举法", "TD"), ("双重Q学习", "TD"),
        ("Sarsa(λ)", "TD"), ("真实在线TD(λ)", "TD"), ("真实在线Sarsa(λ)", "TD"),
        ("Watkins的Q(λ)", "TD"), ("树回溯TB(λ)", "TD"), ("Q(σ)", "TD"),
        ("后位状态方法", "TD"), ("双学习", "TD"), ("最大化偏差处理方法", "TD"),
        
        # MC类
        ("蒙特卡洛方法", "MC"), ("蒙特卡洛预测", "MC"), ("蒙特卡洛控制", "MC"),
        ("MC-ES", "MC"), ("试探性出发蒙特卡洛", "MC"), ("同轨策略MC控制", "MC"),
        ("离轨策略MC预测", "MC"), ("离轨策略MC控制", "MC"), ("普通重要度采样", "MC"),
        ("加权重要度采样", "MC"), ("n步离轨策略学习", "MC"), ("n步树回溯算法", "MC"),
        
        # DP类
        ("动态规划", "DP"), ("策略迭代", "DP"), ("价值迭代", "DP"), ("广义策略迭代", "DP"),
        ("迭代策略评估", "DP"), ("策略评估", "DP"), ("策略改进", "DP"),
        ("异步动态规划", "DP"), ("自举法", "DP"),
    ]
    
    print("=" * 60)
    print("开始生成所有算法的详细文档...")
    print("=" * 60)
    
    count = 0
    errors = []
    
    for algo_name, category in algorithms:
        try:
            print(f"生成 [{category:10s}]: {algo_name}...")
            content = generate_detailed_doc(algo_name, category)
            
            filename = sanitize_filename(algo_name)
            filepath = output_dir / f"{filename}.md"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ 已生成: {filepath} (大小: {len(content)} 字符)")
            count += 1
            
        except Exception as e:
            error_msg = f"{algo_name}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ 错误: {error_msg}")
    
    print("\n" + "=" * 60)
    print(f"详细文档生成完毕！")
    print(f"成功: {count} 个")
    print(f"失败: {len(errors)} 个")
    print("=" * 60)
    
    if errors:
        print("\n错误列表:")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
