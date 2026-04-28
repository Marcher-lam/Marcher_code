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

## 9-14. 简要补充

### 9. 可视化
绘制信念状态的演变轨迹（2D时为概率三角形中的路径）。

### 10. 评估
通过仿真比较POMDP策略与MDP策略（假设完全可观测）的性能差距。

### 11. 常见问题
1. **信念退化**：长时间不观测某些状态 → 粒子滤波中用重采样缓解
2. **计算复杂度**：用Q-MDP、POMCP等近似方法
3. **观测模型不准**：影响信念更新质量 → 从数据学习观测模型

### 12. 学习总结
POMDP将"不知道真实状态"建模为信念状态$b(s)$，通过贝叶斯更新维护信念。核心公式：$b'(s') \propto Z(o|s',a)\sum_s P(s'|s,a)b(s)$。POMDP是MDP在部分可观测下的自然推广。

### 13. 练习题
**Q1**：证明信念更新后的$b'$仍然是一个合法的概率分布（和为1）。
**A1**：$b'(s') = Z(o|s',a)\sum_s P(s'|s,a)b(s) / \eta$，分母$\eta = \sum_{s'}Z(o|s',a)\sum_s P(s'|s,a)b(s)$恰好是归一化因子，保证$\sum_{s'}b'(s')=1$。

### 14. 学习路径
**前置**：MDP、HMM、贝叶斯推断 | **进阶**：POMCP、DESPOT、点集值迭代
**资源**：原书Ch 20.3、Thrun et al. "Probabilistic Robotics"
