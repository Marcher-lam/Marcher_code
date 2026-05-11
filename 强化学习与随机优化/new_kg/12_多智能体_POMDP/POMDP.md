# POMDP(部分可观测马尔可夫决策过程) 学习文档

> POMDP处理"你只能看到世界的部分信息"的决策问题，是现实世界序贯决策的通用框架。

> 来源线索：本节内容根据原书中关于"The POMDP Perspective"的相关章节(Ch 20.3)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：POMDP是MDP的推广，决策者无法直接观测到系统状态，只能获得与状态相关的噪声观测，必须基于不完全信息做出决策。

**直觉类比**：你是一个医生，无法直接看到患者体内的情况（真实状态），只能通过体温、血压、症状等检查结果（观测）来推断。你的治疗方案（动作）依赖于你当前的"诊断信念"（对真实状态的概率分布），而不是真实状态本身。

**历史背景**：POMDP的理论框架由Åström (1965)和Smallwood & Sondik (1973)发展。Sondik (1971)证明了有限视野POMDP的最优值函数是分段线性凸函数。POMDP在机器人、语音识别、医疗决策等领域有广泛应用。

**算法定位**：部分可观测/序贯决策框架。在原书中，POMDP是多智能体系统（Ch 20）的核心建模工具之一。

**前置知识**：MDP、贝叶斯推断、隐马尔可夫模型(HMM)。

## 2. 核心原理

**核心思想**：在POMDP中，智能体不观测真实状态$S_t$，只获得一个观测$O_t$（与$S_t$相关但不含全部信息）。智能体维护一个**信念状态**$b(s)$——对真实状态的概率分布。信念状态本身是马尔可夫的（POMDP可以转化为信念空间上的MDP），但信念空间通常是连续高维的。

**POMDP六元组**：$(\mathcal{S}, \mathcal{A}, \mathcal{O}, P, Z, r)$

1. $\mathcal{S}$：状态空间
2. $\mathcal{A}$：动作空间
3. $\mathcal{O}$：观测空间
4. $P(s'|s,a)$：状态转移概率
5. $Z(o|s',a)$：观测概率（在状态$s'$执行动作$a$后观测到$o$的概率）
6. $r(s,a)$：奖励函数

**工作流程**：

1. 维护信念状态$b(s)$
2. 基于信念选择动作$a$
3. 获得观测$o$
4. 贝叶斯更新信念：$b'(s') \propto Z(o|s',a) \sum_s P(s'|s,a) b(s)$
5. 重复

**关键概念**：

- **信念状态(belief state)**：对真实状态的后验分布$b(s) = P(S=s|历史)$
- **信念MDP**：POMDP在信念空间上等价于连续状态MDP
- **观测模型**$Z(o|s,a)$：定义了观测与状态的关系
- **信念更新**：贝叶斯规则

```
真实状态 S_t（不可见）
      ↓ 动作 a_t
状态转移 P(S_{t+1}|S_t, a_t)
      ↓
真实状态 S_{t+1}（仍不可见）
      ↓ 观测 o_{t+1}
观测生成 Z(o|S_{t+1}, a_t)
      ↓
智能体只看到 o_{t+1}
      ↓ 贝叶斯更新
信念状态 b_{t+1}(s) 更新
```

## 3. 数学公式与推导

### 信念更新

在时刻$t$，信念$b_t$，执行动作$a$，观测$o$后：

$$b_{t+1}(s') = \frac{Z(o|s',a) \sum_{s \in \mathcal{S}} P(s'|s,a) b_t(s)}{\sum_{s''} Z(o|s'',a) \sum_{s} P(s''|s,a) b_t(s)}$$

分子：先转移再观测的联合概率。分母：归一化常数（观测$o$的总概率）。

### 信念MDP

信念更新函数$S^M(b,a,o)$是确定性的（给定$b,a,o$，$b'$完全确定）。因此POMDP可以转化为信念空间上的MDP，但信念空间$B$是连续的。

### 值函数

信念空间上的值函数：

$$V(b) = \max_a \left[\sum_s b(s) r(s,a) + \gamma \sum_o P(o|b,a) V(b'^{a,o})\right]$$

### 精确求解的复杂性

有限视野POMDP的最优值函数是分段线性凸的，可以用$\alpha$-向量表示。但每步的$\alpha$-向量数量指数增长（维度灾难），精确求解POMDP是PSPACE-hard的。

## 4. 训练过程讲解

### 超参数表

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| $\gamma$ | 折扣因子 | 0.95 |
| $n_{particles}$ | 粒子数(粒子滤波) | 100-1000 |
| max_iter | 值迭代轮数 | 100 |

## 5. 应用场景

1. **机器人导航**：传感器有噪声，不能精确知道位置
2. **医疗决策**：不能完全观测患者内部状态
3. **对话系统**：不能确定用户意图
4. **多智能体系统**：不能观测其他智能体的私有信息

## 6. 优缺点分析

### 优点
1. **现实建模**：真实世界通常只能部分可观测
2. **理论完备**：有完整的数学框架

### 缺点
1. **计算极难**：PSPACE-hard
2. **信念空间连续**：难以精确表示
3. **维度灾难**：状态空间稍大就不可行

### 对比

| 特性 | MDP | POMDP | HMM |
|------|-----|-------|-----|
| 状态可观测 | 完全 | 部分 | 部分 |
| 动作控制 | 有 | 有 | 无 |
| 计算复杂度 | P | PSPACE | P |
| 信念维护 | 不需要 | 需要 | 需要 |

## 7. 调库实现

```python
"""
POMDP信念更新和简单求解
使用粒子滤波近似信念
"""
import numpy as np

class POMDP:
    """简单POMDP：老虎门问题(Tiger Problem)"""

    def __init__(self):
        self.states = ['tiger_left', 'tiger_right']
        self.actions = ['listen', 'open_left', 'open_right']
        self.observations = ['hear_left', 'hear_right']
        self.n_states = 2
        self.listen_accuracy = 0.85

    def transition(self, s, a):
        if a == 'listen':
            return s  # 听不改变状态
        return np.random.choice(self.n_states)  # 开门后状态重置

    def observe(self, s, a):
        if a == 'listen':
            if np.random.random() < self.listen_accuracy:
                return s  # 正确听到
            else:
                return 1 - s  # 听错
        return np.random.choice(self.n_states)  # 开门后随机观测

    def reward(self, s, a):
        if a == 'listen':
            return -1.0  # 听有代价
        if (a == 'open_left' and s == 0) or (a == 'open_right' and s == 1):
            return -100.0  # 遇到老虎
        return 10.0  # 安全开门

def bayes_update(belief, action, obs, pomdp):
    """贝叶斯信念更新"""
    p_correct = pomdp.listen_accuracy
    if action == 'listen':
        if obs == 0:  # hear_left
            belief_new = np.array([belief[0]*p_correct, belief[1]*(1-p_correct)])
        else:
            belief_new = np.array([belief[0]*(1-p_correct), belief[1]*p_correct])
    else:
        belief_new = np.array([0.5, 0.5])  # 开门后重置信念
    return belief_new / belief_new.sum()

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    pomdp = POMDP()
    belief = np.array([0.5, 0.5])  # 初始均匀信念
    true_state = 0  # 老虎在左边

    print("老虎门POMDP演示:")
    for t in range(10):
        # 简单策略：如果信念>0.8就开门，否则听
        if belief[0] > 0.8:
            action = 'open_right'
        elif belief[1] > 0.8:
            action = 'open_left'
        else:
            action = 'listen'

        obs = pomdp.observe(true_state, action)
        reward = pomdp.reward(true_state, action)
        belief = bayes_update(belief, action, obs, pomdp)
        true_state = pomdp.transition(true_state, action)

        print(f"t={t}: 动作={action}, 观测=hear_{'left' if obs==0 else 'right'}, "
              f"奖励={reward:.0f}, 信念=[L:{belief[0]:.2f}, R:{belief[1]:.2f}]")
```

## 8. 手工代码实现

```python
"""从零实现POMDP信念更新"""
import numpy as np

class POMDPSolver:
    def __init__(self, n_states, n_actions, n_obs, P, Z, R, gamma=0.95):
        self.ns, self.na, self.no = n_states, n_actions, n_obs
        self.P = P  # P[s',s,a]
        self.Z = Z  # Z[o,s',a]
        self.R = R  # R[s,a]
        self.gamma = gamma

    def belief_update(self, b, a, o):
        """贝叶斯信念更新"""
        b_new = np.zeros(self.ns)
        for s_next in range(self.ns):
            b_new[s_next] = self.Z[o, s_next, a] * np.sum(self.P[s_next, :, a] * b)
        total = b_new.sum()
        return b_new / total if total > 0 else np.ones(self.ns)/self.ns

    def solve_qmdp(self, n_vi_iters=100):
        """Q-MDP近似：先解MDP，再用MDP的Q值在信念上决策"""
        V = np.zeros(self.ns)
        for _ in range(n_vi_iters):
            Q = np.zeros((self.ns, self.na))
            for s in range(self.ns):
                for a in range(self.na):
                    Q[s, a] = self.R[s, a] + self.gamma * np.sum(self.P[:, s, a] * V)
            V = np.max(Q, axis=1)
        self.Q_mdp = Q

    def act(self, belief):
        """在信念上选择动作：∑_s b(s) Q(s,a)最大的a"""
        return np.argmax(belief @ self.Q_mdp)
```

## 9. 可视化与结果理解

```python
"""POMDP信念状态演变可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_belief_evolution(belief_history, action_history, reward_history):
    """绘制信念状态随时间的演变"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    T = len(belief_history)

    # 1. 信念概率演变
    beliefs = np.array(belief_history)
    axes[0].plot(range(T), beliefs[:, 0], 'b-o', label='P(tiger_left)', markersize=4)
    axes[0].plot(range(T), beliefs[:, 1], 'r-s', label='P(tiger_right)', markersize=4)
    axes[0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='决策阈值')
    axes[0].set_ylabel('信念概率')
    axes[0].set_title('信念状态演变')
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)

    # 2. 动作历史
    action_map = {'listen': 0, 'open_left': 1, 'open_right': 2}
    actions = [action_map.get(a, 0) for a in action_history]
    colors = ['green' if a == 0 else ('red' if a == 1 else 'blue') for a in actions]
    axes[1].bar(range(T), actions, color=colors, alpha=0.7)
    axes[1].set_ylabel('动作')
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['听', '开左门', '开右门'])
    axes[1].set_title('动作序列')

    # 3. 累积奖励
    cum_rewards = np.cumsum(reward_history)
    axes[2].plot(range(T), cum_rewards, 'k-o', markersize=4)
    axes[2].set_ylabel('累积奖励')
    axes[2].set_xlabel('时间步')
    axes[2].set_title('累积奖励曲线')

    plt.tight_layout()
    plt.savefig('pomdp_belief_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**结果解读**：
- 信念曲线随"听"动作逐步收敛到真实状态（概率接近1.0）
- 当信念超过阈值0.8时，触发开门动作
- 累积奖励曲线：听动作代价小（-1），正确开门获得+10，错误开门-100

## 10. 模型评估

```python
"""POMDP策略评估：与MDP上界比较"""
import numpy as np

def evaluate_pomdp_policy(pomdp_env, policy_fn, n_episodes=500, max_steps=50):
    """
    评估POMDP策略的实际表现
    指标：平均累积奖励、正确决策率、平均决策时间步
    """
    total_rewards = []
    correct_decisions = 0
    decision_steps = []

    for ep in range(n_episodes):
        belief = np.array([0.5, 0.5])
        true_state = np.random.choice(pomdp_env.n_states)
        ep_reward = 0
        decided = False

        for t in range(max_steps):
            action = policy_fn(belief)
            obs = pomdp_env.observe(true_state, action)
            reward = pomdp_env.reward(true_state, action)
            ep_reward += reward
            belief = bayes_update(belief, action, obs, pomdp_env)
            true_state = pomdp_env.transition(true_state, action)

            if action.startswith('open') and not decided:
                decision_steps.append(t)
                decided = True
                # 检查是否正确
                if (action == 'open_right' and true_state == 0) or \
                   (action == 'open_left' and true_state == 1):
                    correct_decisions += 1

        total_rewards.append(ep_reward)

    print(f"POMDP策略评估 ({n_episodes}回合):")
    print(f"  平均累积奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  正确决策率: {correct_decisions/n_episodes:.2%}")
    print(f"  平均决策时间步: {np.mean(decision_steps):.1f}")
    return np.mean(total_rewards)
```

**评估指标说明**：
- **平均累积奖励**：最直接的策略质量度量，与MDP完全信息上界对比
- **正确决策率**：开门时选择安全门的概率，衡量信念质量
- **平均决策时间步**：收集多少信息后才行动，反映探索-利用平衡

## 11. 常见问题与易错点

### 数据层面

1. **观测模型设计偏差**
   - 现象：信念更新后概率没有向正确方向收敛
   - 原因：观测概率$Z(o|s,a)$设定不合理，如观测正确率太低
   - 解决方案：通过历史数据估计观测模型的参数，或使用EM算法学习

2. **信念退化（粒子滤波）**
   - 现象：所有粒子坍缩到同一状态，失去多样性
   - 原因：粒子数太少或似然函数太尖锐
   - 解决方案：增加粒子数、使用重采样+扰动（如粒子群MCMC）

### 模型层面

3. **维度灾难**
   - 现象：状态空间稍大（$|\mathcal{S}|>20$）就无法精确求解
   - 原因：信念空间$\Delta^{|\mathcal{S}|-1}$的$\alpha$-向量数量指数增长
   - 解决方案：使用Q-MDP近似、POMCP蒙特卡洛方法、或点集值迭代

4. **信念更新数值不稳定**
   - 现象：信念概率出现NaN或负值
   - 原因：观测概率极小时归一化导致浮点下溢
   - 解决方案：在对数空间计算，或使用粒子滤波避免显式归一化

### 调参层面

5. **决策阈值选择不当**
   - 现象：过早开门（高错误率）或过度听（累积奖励低）
   - 原因：信念阈值设得太低或太高
   - 解决方案：通过仿真搜索最优阈值，或使用POMDP求解器自动计算最优策略

## 12. 学习总结

POMDP将"不知道真实状态"这一现实约束建模为信念状态$b(s)$，通过贝叶斯更新维护信念。POMDP是MDP在部分可观测下的自然推广，理论上可将POMDP转化为信念空间上的连续状态MDP。

**关键公式**：
1. 信念更新：$b'(s') \propto Z(o|s',a)\sum_s P(s'|s,a)b(s)$
2. 信念MDP值函数：$V(b) = \max_a [\sum_s b(s)r(s,a) + \gamma\sum_o P(o|b,a)V(b'^{a,o})]$
3. Q-MDP近似：$\pi(b) = \arg\max_a \sum_s b(s)Q^*_{MDP}(s,a)$

POMDP与前序知识紧密相连：它是MDP的推广（加入部分可观测），与HMM共享观测模型概念（但增加了动作控制），与贝叶斯推断的核心计算（后验更新）一致。后续学习中，精确求解方法包括点集值迭代、近似方法包括POMCP/DESPOT，多智能体扩展为Dec-POMDP。

## 13. 练习题与思考题

### 基础题

**题目1**：证明信念更新后的$b'$仍然是一个合法的概率分布（非负且和为1）。

**参考答案**：
$b'(s') = \frac{Z(o|s',a)\sum_s P(s'|s,a)b(s)}{\eta}$，其中分母$\eta = \sum_{s'}Z(o|s',a)\sum_s P(s'|s,a)b(s)$。
由于$Z, P, b$都是非负的，分子非负，故$b'(s')\geq 0$。
求和：$\sum_{s'}b'(s') = \frac{1}{\eta}\sum_{s'}Z(o|s',a)\sum_s P(s'|s,a)b(s) = \frac{\eta}{\eta} = 1$。证毕。

**题目2**：在老虎门问题中，初始信念$b=[0.5, 0.5]$，执行"听"动作后观测到"hear_left"，观测正确率为0.85。计算更新后的信念。

**参考答案**：
$b'(tiger\_left) \propto 0.85 \times (P(tiger\_left|tiger\_left, listen)\times 0.5 + P(tiger\_left|tiger\_right, listen)\times 0.5) = 0.85 \times 0.5 = 0.425$
$b'(tiger\_right) \propto 0.15 \times 0.5 = 0.075$
归一化：$b' = [0.425/(0.425+0.075), 0.075/(0.425+0.075)] = [0.85, 0.15]$

### 进阶题

**题目3**：解释Q-MDP近似为什么是POMDP的常用近似方法，以及它在什么情况下表现不佳。

**参考答案**：
Q-MDP先忽略部分可观测性，求解完全可观测MDP得到$Q^*(s,a)$，然后选择$\arg\max_a \sum_s b(s)Q^*(s,a)$。它假设决策后立即获得完全信息（"只要做对这一步"），因此严重低估了信息收集的价值。在观测质量高、决策后果不严重时表现好；在需要长时间收集信息（如听多次才敢开门）时表现差，因为它不会"耐心等待"。

### 开放思考题

**题目4**：原书(Ch 20)将多智能体系统建模为POMDP——每个智能体无法观测其他智能体的私有信息。请思考：当智能体数量增加时，POMDP的信念空间维度如何增长？有哪些实际的方法可以缓解？

**参考答案方向**：
- 信念空间维度随智能体数和状态空间指数增长（联合信念空间）
- 缓解方法：(1) 独立假设——假设智能体信念独立，降维到各智能体的边际信念；(2) 通信——智能体共享观测信息减少不确定性；(3) 分散化策略——每个智能体只基于局部观测决策（Dec-POMDP）；(4) 平均场近似——用群体的统计量代替个体状态

## 14. 学习路径建议

**前置算法**：
- 马尔可夫决策过程（MDP）
- 隐马尔可夫模型（HMM）
- 贝叶斯推断
- 贝尔曼方程

**平行算法**：
- 粒子滤波（Particle Filtering）
- 信息论（Information Theory）
- 贝叶斯优化

**进阶算法**：
- POMCP（部分可观测蒙特卡洛规划）
- DESPOT（确定性稀疏部分可观测树）
- 点集值迭代（Point-Based Value Iteration）
- Dec-POMDP（分散化POMDP，多智能体扩展）

**推荐资源**：
1. Powell, W.B. "Reinforcement Learning and Stochastic Optimization" (2022) Ch 20.3
2. Thrun, Burgard & Fox "Probabilistic Robotics" (2005) — 机器人中的POMDP应用
3. Kaelbling, Littman & Cassandra "Planning and Acting in Partially Observable Stochastic Domains" (1998) — POMDP经典综述
