#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量扩展标准版文档到5k-10k字
为每类算法生成详细内容，替换通用模板文本
"""

import os
import re
from pathlib import Path

# 算法类别的详细内容模板
CATEGORY_TEMPLATES = {
    "TD": {
        "core_idea_zh": "通过时间差分学习和贝尔曼方程更新价值函数，结合蒙特卡洛的无偏性和动态规划的自举特性，在状态-动作空间中迭代更新Q值或V值。",
        "update_formula_latex": r"Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] \quad \text{(Q-learning)} \\ V(s) \leftarrow V(s) + \alpha \left[ r + \gamma V(s') - V(s) \right] \quad \text{(TD预测)}",
        "key_concepts_list": "TD误差、自举法（bootstrap）、λ参数（如适用）、资格迹（eligibility traces）、off-policy/on-policy区别",
        "math_details": """
### 3.4 推导过程

**Step 1：贝尔曼方程**
对于TD学习，我们基于贝尔曼方程：
$$ V^\pi(s) = \mathbb{E} \left[ r + \gamma V^\pi(s') \mid s \right] $$

**Step 2：样本近似**
在实际应用中，我们用样本均值代替期望：
$$ V(s) \leftarrow V(s) + \alpha \left[ r + \gamma V(s') - V(s) \right] $$

**Step 3：收敛性分析**
在满足Robbins-Monro条件时（$\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$），TD学习保证收敛到$V^\pi$。
""",
        "code_example": """
```python
import numpy as np

class TDAgent:
    def __init__(self, n_states, lr=0.01, gamma=0.99):
        self.V = np.zeros(n_states)
        self.lr = lr
        self.gamma = gamma
    
    def update(self, s, r, s_next, done):
        if done:
            td_target = r
        else:
            td_target = r + self.gamma * self.V[s_next]
        td_error = td_target - self.V[s]
        self.V[s] += self.lr * td_error
        return td_error
```
""",
        "applications": "游戏AI、机器人控制、推荐系统、交易策略优化"
    },
    
    "MC": {
        "core_idea_zh": "通过完整episode的回报采样估计价值函数或优化策略，无需环境模型，无偏差但方差较高，是强化学习三大基石之一（与动态规划、时序差分并列）。",
        "update_formula_latex": r"G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \\ V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right]",
        "key_concepts_list": "回报（Return）G_t、首次访问（First-Visit）MC、每次访问（Every-Visit）MC、重要度采样（Importance Sampling）、普通/加权重要度采样、ε-greedy策略",
        "math_details": """
### 3.4 推导过程

**Step 1：回报的定义**
从时刻t开始的折扣回报：
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T $$

**Step 2：大数定律的应用**
根据大数定律，当episode数量$N \rightarrow \infty$时，样本均值收敛到期望：
$$ \hat{V}_N(s) = \frac{1}{N} \sum_{i=1}^N G_t^{(i)}(s) \rightarrow V^\pi(s) \quad \text{a.s.} $$

**Step 3：增量式更新**
为避免存储所有回报，使用增量式更新：
$$ V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right] $$
这里$\alpha$是学习率，等价于给每个回报的权重为$\alpha(1-\alpha)^{k-1}$（指数衰减）。
""",
        "code_example": """
```python
def compute_returns(rewards, gamma=0.99):
    \"\"\"计算折扣回报G_t\"\"\"
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
```
""",
        "applications": "博弈游戏（围棋、象棋）、推荐系统（用户session）、广告投放、医疗治疗方案评估"
    },
    
    "DP": {
        "core_idea_zh": "基于马尔可夫决策过程的环境模型，通过贝尔曼方程迭代求解最优价值函数和最优策略，是强化学习中样本效率最高的基石方法。",
        "update_formula_latex": r"V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V_k(s') \right] \\ \pi'(s) = \arg\max_a \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V^\pi(s') \right]",
        "key_concepts_list": "贝尔曼方程、贝尔曼最优方程、策略评估（Policy Evaluation）、策略改进（Policy Improvement）、策略迭代（Policy Iteration）、价值迭代（Value Iteration）、广义策略迭代（GPI）",
        "math_details": """
### 3.4 推导过程

**Step 1：贝尔曼方程（对于策略π）**
$$ V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V^\pi(s') \right] $$

**Step 2：策略评估的迭代形式**
$$ V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V_k(s') \right] $$

**Step 3：策略改进定理**
如果策略π'在每一个状态s上都满足：
$$ Q^\pi(s, \pi'(s)) \geq V^\pi(s) $$
那么π'的价值函数满足$V^{\pi'} \geq V^\pi$，即π'是比π更优的策略。
""",
        "code_example": """
```python
def policy_evaluation(policy, P, R, gamma=0.99, theta=1e-4):
    \"\"\"策略评估：计算V^π\"\"\"
    V = np.zeros(len(P))
    while True:
        V_old = V.copy()
        delta = 0
        for s in range(len(P)):
            v = 0
            for a in range(len(P[s])):
                for s_next in P[s][a]:
                    prob, reward = P[s][a][s_next]
                    v += prob * (reward + gamma * V_old[s_next])
            V[s] = v
            delta = max(delta, abs(V_old[s] - V[s]))
        if delta < theta:
            break
    return V
```
""",
        "applications": "棋盘游戏AI、机器人路径规划（已知地图）、资源分配优化、生产计划调度"
    },
    
    "Deep": {
        "core_idea_zh": "使用深度神经网络作为函数逼近器，处理高维状态/动作空间，结合经验回放和稳定训练技巧，是深度强化学习的核心。",
        "update_formula_latex": r"L(\theta) = \mathbb{E} \left[ (r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta) )^2 \right] \\ \nabla_\theta L = \mathbb{E} \left[ (r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta)) \nabla_\theta Q(s,a; \theta) \right]",
        "key_concepts_list": "经验回放（Experience Replay）、目标网络（Target Network）、卷积神经网络（CNN）、Dueling Network、Double DQN、Rainbow、分布式RL",
        "math_details": """
### 3.4 推导过程

**Step 1：DQN的损失函数**
$$ L(\theta) = \mathbb{E}_{s,a,r,s'} \left[ (r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta) )^2 \right] $$

**Step 2：梯度计算**
$$ \nabla_\theta L = \mathbb{E} \left[ (r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta)) \nabla_\theta Q(s,a; \theta) \right] $$

**Step 3：目标网络**
使用目标网络$Q(s',a'; \theta^-)$来稳定训练，缓慢更新目标网络参数：
$$ \theta^- \leftarrow \tau \theta + (1-\tau) \theta^- $$
""",
        "code_example": """
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 训练时使用目标网络
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
```
""",
        "applications": "Atari游戏、围棋（AlphaGo）、机器人控制（连续/离散）、自动驾驶、推荐系统、金融交易"
    },
    
    "Model": {
        "core_idea_zh": "结合环境模型学习和规划，使用Dyna架构等框架，通过模型生成的模拟经验加速学习，提高样本效率。",
        "update_formula_latex": r"Model: \hat{P}, \hat{R} \leftarrow \text{data} \\ Planning: Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)] \\ \text{using both real and simulated experiences}",
        "key_concepts_list": "Dyna架构、模型学习（Model Learning）、轨迹采样（Trajectory Sampling）、优先遍历（Prioritized Sweeping）、实时动态规划（RTDP）、蒙特卡洛树搜索（MCTS）",
        "math_details": """
### 3.4 推导过程

**Step 1：模型学习**
从实际交互数据中学习环境模型$\hat{P}(s'|s,a)$和$\hat{R}(s,a,s')$：
$$ \hat{P}(s'|s,a) = \frac{N(s,a,s')}{\sum_{s''} N(s,a,s'')} $$
$$ \hat{R}(s,a,s') = \frac{\sum \text{rewards}}{N(s,a,s')} $$

**Step 2：Dyna架构**
每次实际交互后，用模型生成$k$个模拟经验：
$$ s', r \sim \hat{P}(\cdot|s,a), \hat{R}(s,a,\cdot) $$
然后用这些模拟经验进行Q-learning更新。

**Step 3：收敛性**
在表格型情况下，Dyna-Q保证收敛到Q*（如果模型准确）。
""",
        "code_example": """
```python
class DynaAgent:
    def __init__(self, n_states, n_actions, planning_steps=10):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # (s,a) -> (s', r)
        self.planning_steps = planning_steps
    
    def update(self, s, a, r, s_next):
        # 更新Q值（实际经验）
        td_target = r + gamma * np.max(self.Q[s_next])
        self.Q[s, a] += lr * (td_target - self.Q[s, a])
        
        # 学习模型
        self.model[(s, a)] = (s_next, r)
        
        # 规划：用模型生成模拟经验
        for _ in range(self.planning_steps):
            (s_sim, a_sim) = random.choice(list(self.model.keys()))
            s_next_sim, r_sim = self.model[(s_sim, a_sim)]
            td_target = r_sim + gamma * np.max(self.Q[s_next_sim])
            self.Q[s_sim, a_sim] += lr * (td_target - self.Q[s_sim, a_sim])
```
""",
        "applications": "机器人导航（已知/未知地图）、游戏AI（棋类游戏）、组合优化、元学习"
    },
    
    "FA": {
        "core_idea_zh": "使用函数逼近替代表格存储，通过线性或非线性函数将状态映射到价值，处理大规模或连续状态空间。",
        "update_formula_latex": r"V(s) \approx \hat{V}(s; w) = \phi(s)^T w \\ w \leftarrow w + \alpha \left[ r + \gamma \hat{V}(s'; w) - \hat{V}(s; w) \right] \phi(s) \\ \text{(半梯度下降，因为仅考虑 } \hat{V}(s'; w) \text{ 的梯度)}",
        "key_concepts_list": "线性函数逼近、非线性函数逼近（神经网络）、半梯度方法（Semi-gradient）、资格迹（Eligibility Traces）、瓦片编码（Tile Coding）、径向基函数（RBF）、傅立叶基（Fourier Basis）",
        "math_details": """
### 3.4 推导过程

**Step 1：函数逼近形式**
使用特征向量$\phi(s)$和权重$w$来近似价值函数：
$$ \hat{V}(s; w) = \phi(s)^T w $$

**Step 2：半梯度更新**
由于bootstrap，我们只有半梯度（只考虑当前$w$对$\hat{V}(s; w)$的梯度，不考虑$\hat{V}(s'; w)$的梯度）：
$$ w \leftarrow w + \alpha \left[ r + \gamma \hat{V}(s'; w) - \hat{V}(s; w) \right] \nabla_w \hat{V}(s; w) $$

**Step 3：收敛性**
对于线性函数逼近，半梯度TD保证收敛到TD不动点（但不是真正的TD不动点，因为半梯度）。
""",
        "code_example": """
```python
class LinearFA:
    def __init__(self, n_features, lr=0.01):
        self.w = np.random.randn(n_features) * 0.01
        self.lr = lr
    
    def phi(self, state):
        \"\"\"状态特征化（示例：瓦片编码）\"\"\"
        # 简化的瓦片编码
        return np.eye(1, n_features, state)[0]  # 实际应更复杂
    
    def value(self, state):
        return np.dot(self.phi(state), self.w)
    
    def update(self, state, reward, next_state, done, gamma):
        td_target = reward + gamma * self.value(next_state) * (not done)
        td_error = td_target - self.value(state)
        self.w += self.lr * td_error * self.phi(state)
```
""",
        "applications": "高维状态空间问题、连续状态空间、大规模MDP、深度强化学习的基础"
    },
    
    "Exploration": {
        "core_idea_zh": "平衡探索与利用，设计策略在未知环境中有效学习，避免陷入次优策略，是强化学习实用化的关键。",
        "update_formula_latex": r"\epsilon\text{-greedy}: \pi(a|s) = \begin{cases} 1-\epsilon+\frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases} \\ \text{UCB}: a_t = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right]",
        "key_concepts_list": "ε-greedy、UCB（置信度上界）、Thompson Sampling、Softmax（Boltzmann）探索、乐观初始化（Optimistic Initialization）、内在奖励（Intrinsic Motivation）",
        "math_details": """
### 3.4 推导过程

**Step 1：ε-greedy探索**
以$\epsilon$概率随机探索，以$1-\epsilon$概率贪心利用：
$$ \pi(a|s) = \begin{cases} 1-\epsilon+\frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases} $$

**Step 2：UCB探索**
上置信界探索，平衡探索与利用：
$$ a_t = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right] $$
其中$N(s,a)$是状态-动作对被访问的次数，$c$是探索常数。

**Step 3：理论保证**
UCB算法在k臂赌博机中保证对数后悔界（logarithmic regret）：
$$ R(T) \leq \sum_{a: \Delta_a > 0} \left( \frac{8 \ln T}{\Delta_a} + O(1) \right) $$
其中$\Delta_a = \mu^* - \mu_a$是最佳臂与臂a的期望奖励差。
""",
        "code_example": """
```python
class UCBAgent:
    def __init__(self, n_states, n_actions, c=2.0):
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions))
        self.c = c
        self.t = 0
    
    def choose_action(self, state):
        self.t += 1
        # UCB公式
        ucb_values = self.Q[state] + self.c * np.sqrt(np.log(self.t) / (self.N[state] + 1e-6))
        return np.argmax(ucb_values)
    
    def update(self, state, action, reward):
        self.N[state, action] += 1
        # 增量更新Q值
        self.Q[state, action] += (reward - self.Q[state, action]) / self.N[state, action]
```
""",
        "applications": "多臂赌博机、推荐系统、A/B测试、超参数调优、强化学习探索策略"
    },
    
    "Other": {
        "core_idea_zh": "强化学习相关的基础理论、应用领域或扩展方法，涵盖从多臂赌博机到神经科学模型的广泛内容。",
        "update_formula_latex": r"\text{根据具体算法确定，通常是前述类别的特例或组合}",
        "key_concepts_list": "马尔可夫决策过程、贝尔曼方程、最优控制、神经科学模型、遗传算法、进化策略、认知架构",
        "math_details": """
### 3.4 推导过程

**概述**
这类算法通常是前述TD、MC、DP等方法的变体或应用，具体推导取决于具体算法。

**常见形式**
- 遗传算法：通过选择、交叉、突变进化策略参数
- 神经科学模型：使用多巴胺等神经递质模拟TD误差
- 多臂赌博机：简化版的RL，单状态多动作
""",
        "code_example": """
```python
# 示例：多臂赌博机（Gaussian）
class GaussianBandit:
    def __init__(self, n_arms):
        self.true_means = np.random.randn(n_arms)
        self.estimated_means = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
    
    def select_arm(self):
        # UCB选择
        ucb = self.estimated_means + np.sqrt(2 * np.log(sum(self.N)) / (self.N + 1e-6))
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.N[arm] += 1
        self.estimated_means[arm] += (reward - self.estimated_means[arm]) / self.N[arm]
```
""",
        "applications": "理论研究、神经科学、进化计算、游戏AI历史、认知建模、推荐系统"
    }
}

def get_algorithm_category(filename):
    """根据文件名判断算法类别"""
    name = Path(filename).stem
    if any(x in name for x in ["Q学习", "Sarsa", "TD", "期望Sarsa", "n步", "双重", "树回溯", "Q(σ)"]):
        return "TD"
    elif any(x in name for x in ["蒙特卡洛", "MC-", "重要度采样"]):
        return "MC"
    elif any(x in name for x in ["动态规划", "策略迭代", "价值迭代", "自举法"]):
        return "DP"
    elif any(x in name for x in ["DQN", "深度", "REINFORCE", "策略梯度", "行动器-评判器"]):
        return "Deep"
    elif any(x in name for x in ["Dyna", "MCTS", "UCT", "预演", "规划", "RTDP"]):
        return "Model"
    elif any(x in name for x in ["函数逼近", "半梯度", "LSTD", "GTD", "资格迹", "λ-回报", "瓦片编码", "径向基"]):
        return "FA"
    elif any(x in name for x in ["ε-贪心", "UCB", "softmax", "高斯", "赌博机", "探索"]):
        return "Exploration"
    else:
        return "Other"

def expand_document(filepath, category, template):
    """扩展单个文档到5k-10k字"""
    try:
        # 读取文件
        content = None
        for enc in ['utf-8', 'gbk', 'latin-1', 'cp936']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except:
                continue
        
        if content is None:
            return False
        
        algo_name = Path(filepath).stem
        
        # 替换通用文本为详细内容
        replacements = {
            r'像在走迷宫时，每走一步就根据是否接近目标来修正对各个位置距离目标的估计': template['core_idea_zh'][:100] + "..." if len(template['core_idea_zh']) > 100 else template['core_idea_zh'],
            r'TD学习基于Sutton在1988年提出的时序差分学习理论': f"{algo_name}是强化学习领域的重要算法，有深厚的研究基础。",
            r'Episode 100, Average Score: 25\.34\nEpisode 200, Average Score: 38\.12\n\.{3}': '训练曲线示例（需实际运行代码生成）',
            r'通过Q表格和TD学习找到最优策略的off-policy算法，是强化学习中最基础的算法之一': template['core_idea_zh'],
            r'核心思想可以概括为：.*?\n': f"核心思想可以概括为：{template['core_idea_zh'][:150]}...\n",
            r'TD误差：δ = r \+ γV\(s\'\) - V\(s\)\n- Bootstrap：使用当前估计值': f"TD误差：{template['update_formula_latex'][:100]}...\n- 关键更新公式见数学公式章节",
            r'蒙特卡洛方法基于Sutton在1988年提出的时序差分学习理论': f"{algo_name}基于Sutton在1988年提出的时序差分学习理论，是强化学习三大基石之一。",
        }
        
        # 扩展第3章数学公式部分
        if "## 3. 数学公式与推导" in content:
            content = content.replace("## 3. 数学公式与推导", "## 3. 数学公式与推导\n" + template['math_details'])
        
        # 扩展代码示例
        if "```python" in content:
            # 只替换第一个代码块（示例）
            parts = content.split("```python", 1)
            if len(parts) > 1:
                before, after = parts
                # 插入详细代码示例
                content = before + "```python\n" + template['code_example'] + "\n```" + after
        
        # 替换关键概念
        if "TD误差、Bootstrap" in content or "回报G_t" in content:
            content = re.sub(r'TD误差、Bootstrap.*?\n', template['key_concepts_list'] + '\n', content, flags=re.DOTALL)
            content = re.sub(r'回报G_t.*?\n', template['key_concepts_list'] + '\n', content, flags=re.DOTALL)
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"错误 {filepath.name}: {e}")
        return False

def main():
    output_dir = Path("/Users/marcher/Desktop/Marcher_code/强化学习_第二版")
    
    # 要跳过的文件
    skip_files = ["TEMPLATE.md", "WRITING_SPEC.md", "PROMPT.md", "full.md", 
                  "Q学习_完整版.md", "Sarsa_完整版.md", "蒙特卡洛方法_完整版.md",
                  "动态规划_完整版.md", "策略迭代_完整版.md", "价值迭代_完整版.md",
                  "强化学习算法名称提取.md"]
    
    print("=" * 60)
    print("开始批量扩展标准版文档到5k-10k字...")
    print("=" * 60)
    
    expanded = 0
    total = 0
    
    for filepath in output_dir.glob("*.md"):
        if filepath.name in skip_files:
            continue
        
        total += 1
        category = get_algorithm_category(filepath.name)
        template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES["Other"])
        
        if expand_document(filepath, category, template):
            expanded += 1
            if expanded % 10 == 0:
                print(f"已扩展: {expanded}/{total}")
    
    print("\n" + "=" * 60)
    print(f"扩展完成！共处理{total}个文件，成功扩展{expanded}个")
    print("=" * 60)

if __name__ == "__main__":
    main()
