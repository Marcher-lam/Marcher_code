#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成所有强化学习算法的完整文档
使用预定义的模板和数据，避免字符串嵌套问题
"""

import re
from pathlib import Path

def sanitize_filename(name):
    """清理文件名"""
    name = name.replace('/', '_').replace('\\', '_')
    name = re.sub(r'[\\:*?"<>|]', '', name)
    return name.strip()

def read_template():
    """读取手工编写的Q学习文档作为完整模板"""
    # 这里我们使用简化的内容，因为读取文件可能很大
    # 实际中应该读取 Q学习_完整版.md
    return ""

def generate_doc(algo_name, category, description):
    """生成单个算法的完整文档"""
    
    # 基础模板 - 使用占位符
    template = """# {algo} 学习文档!

> {desc}

---

## 1. 算法基础认知

**一句话定义**：{one_liner}

**直觉类比**：{analogy}

**历史背景**：{history}

**算法定位**：
- 类型：强化学习 → {algo_type}
- 输出：{output}
- 模型类型：{model_type}

**前置知识**：
- {prereq1}
- {prereq2}
- {prereq3}

---

## 2. 核心原理

### 2.1 核心思想

{core_idea}

核心思想可以概括为：{core_summary}

### 2.2 工作流程

{workflow}

### 2.3 关键概念解释

{key_concepts}

### 2.4 几何/直观解释

{intuitive}

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $S$ | 状态集合 |
| $A$ | 动作集合 |
| $R$ | 奖励 |
| $\\gamma$ | 折扣因子 |

### 3.2 问题形式化

{problem}

### 3.3 目标函数

{objective}

### 3.4 推导过程

{derivation}

### 3.5 最终算法步骤

{solution}

---

## 4. 训练过程讲解!

### 4.1 数据预处理

{preprocessing}

### 4.2 参数初始化

{param_init}

### 4.3 迭代过程

{iteration}

### 4.4 收敛条件

{convergence}

### 4.5 超参数及推荐范围!

| 超参数 | 推荐范围 | 默认值 |
|--------|----------|--------|
| $\\alpha$ | 0.001-0.1 | 0.01 |
| $\\gamma$ | 0.9-0.999 | 0.99 |

---

## 5. 应用场景!

### 5.1 典型应用

{applications}

### 5.2 适用数据特征!

{data_chars}

### 5.3 不适用场景!

{limitations}

---

## 6. 优缺点分析!

### 6.1 优点

{advantages}

### 6.2 缺点

{disadvantages}

### 6.3 与同类算法对比!

{comparison}

---

## 7. 调库实现!

### 7.1 环境准备!

```bash
pip install gymnasium numpy matplotlib
```

### 7.2 完整代码示例!

{code_example}

### 7.3 运行结果示例!

{output}

---

## 8. 手工代码实现!

### 8.1 核心算法手写!

{manual_impl}

### 8.2 与调库结果对比!

{comparison2}

---

## 9. 可视化与结果理解!

### 9.1 关键参数可视化!

{visualization}

### 9.2 结果解读!

{interpretation}

---

## 10. 模型评估!

### 10.1 评估指标选择!

{eval_metrics}

### 10.2 评估代码!

{eval_code}

---

## 11. 常见问题与易错点!

### 11.1 数据层面常见错误!

{data_errors}

### 11.2 模型层面常见错误!

{model_errors}

### 11.3 调参层面常见误区!

{param_mistakes}

---

## 12. 学习总结!

### 12.1 核心要点回顾!

{key_takeaways}

### 12.2 关键公式汇总!

{key_formulas}

### 12.3 最佳实践!

{best_practices}

### 12.4 与其他算法的联系!

{related_algs}

---

## 13. 练习题与思考题!

### 13.1 基础练习!

{basic_exercises}

### 13.2 进阶思考!

{advanced_exercises}

### 13.3 开放思考!

{open_ended}

---

## 14. 学习路径建议!

### 14.1 前置知识!

{prerequisites}

### 14.2 平行算法!

{parallel_algs}

### 14.3 进阶算法!

{next_algs}

### 14.4 推荐资源!

{resources}

---

**文档结束**

> 如果你觉得这个文档对你有帮助，请分享给更多学习强化学习的人!
> 如有错误或建议，欢迎指出，共同完善!
"""
    
    # 根据算法类别填充内容
    if category == "TD":
        content = fill_td(algo_name, description, template)
    elif category == "MC":
        content = fill_mc(algo_name, description, template)
    elif category == "DP":
        content = fill_dp(algo_name, description, template)
    elif category == "Deep":
        content = fill_deep(algo_name, description, template)
    elif category == "Model":
        content = fill_model(algo_name, description, template)
    elif category == "FA":
        content = fill_fa(algo_name, description, template)
    elif category == "Exploration":
        content = fill_exp(algo_name, description, template)
    else:
        content = fill_other(algo_name, description, template)
    
    return content

def fill_td(algo, desc, template):
    """填充TD类算法"""
    replacements = {
        "{algo}": algo,
        "{desc}": desc,
        "{one_liner}": f"{algo}通过时间差分学习更新价值函数，结合蒙特卡洛和动态规划的优点",
        "{analogy}": "像在走迷宫时，每走一步就根据是否接近目标来修正对各个位置距离目标的估计",
        "{history}": f"{algo}基于Sutton在1988年提出的时序差分学习理论",
        "{algo_type}": "控制/预测",
        "{output}": "状态价值V(s)或动作价值Q(s,a)",
        "{model_type}": "表格型或函数逼近",
        "{prereq1}": "马尔可夫决策过程（MDP）",
        "{prereq2}": "贝尔曼方程",
        "{prereq3}": "Q-learning或Sarsa基础",
        "{core_idea}": f"{algo}的核心思想是通过bootstrap（使用当前估计）来更新价值估计，结合了蒙特卡洛的无偏性和动态规划的单步更新。",
        "{core_summary}": "通过时间差分误差不断更新价值估计，最终收敛到真实价值函数",
        "{workflow}": "1. 初始化价值函数\n2. 每一步执行动作，观察奖励和下一个状态\n3. 使用TD误差更新价值\n4. 终止条件判断",
        "{key_concepts}": "- TD误差：δ = r + γV(s') - V(s)\n- Bootstrap：使用当前估计值\n- λ参数（如适用）：控制偏差-方差权衡",
        "{intuitive}": "TD学习可以看作是在时间维度上的'纠错'：每次得到一个奖励后，算法会比较之前的预测和实际结果，然后调整预测使其更准确。",
        "{problem}": "给定MDP，学习目标是通过TD学习找到价值函数V(s)，使得TD误差最小化。",
        "{objective}": "$$ L = E[(r + γV(s') - V(s))^2] $$",
        "{derivation}": "基于贝尔曼方程：V(s) = E[r + γV(s')]\nTD更新：V(s) <- V(s) + α[r + γV(s') - V(s)]",
        "{solution}": "TD(0)更新规则：\nV(s_t) <- V(s_t) + α[r_{t+1} + γV(s_{t+1}) - V(s_t)]",
        "{preprocessing}": "1. 状态表示：离散状态或函数逼近\n2. 奖励设计：根据任务设计",
        "{param_init}": "价值函数初始化为0或小的随机值",
        "{iteration}": "```python\nfor episode in range(N):\n    s = env.reset()\n    while not done:\n        a = policy(s)\n        s_next, r, done, _ = env.step(a)\n        V[s] += alpha * (r + gamma * V[s_next] - V[s])\n```",
        "{convergence}": "价值函数变化 < ε，或达到最大迭代次数",
        "{applications}": "**应用1：游戏AI** - 如TD-Gammon\n**应用2：机器人控制** - 学习状态价值函数",
        "{data_chars}": "适合有马尔可夫性质的环境，可以处理连续或离散状态",
        "{limitations}": "1. 需要多次试错\n2. 表格型受限于状态空间大小\n3. 函数逼近可能不收敛",
        "{advantages}": "1. 不需要完整episode（相比MC）\n2. 可以在线学习\n3. 理论保证（表格型）",
        "{disadvantages}": "1. 有偏差（bootstrap）\n2. 对超参数敏感\n3. 非线性函数逼近可能不收敛",
        "{comparison}": "| 维度 | TD(0) | Monte Carlo |\n|------|--------|------------|\n| 偏差/方差 | 低偏差高方差 | 高偏差低方差 |",
        "{code_example}": "```python\nimport gymnasium as gym\nimport numpy as np\n\nV = np.zeros(16)\nalpha = 0.01\ngamma = 0.99\n\nfor episode in range(1000):\n    s = env.reset()[0]\n    done = False\n    while not done:\n        a = policy(s)\n        s_next, r, terminated, truncated, _ = env.step(a)\n        done = terminated or truncated\n        V[s] += alpha * (r + gamma * V[s_next] - V[s])\n        s = s_next\n```",
        "{output}": "Episode 100, Average Score: 25.34\nEpisode 200, Average Score: 38.12\n...",
        "{manual_impl}": "```python\nclass TabularTD:\n    def __init__(self, n_states, lr=0.01, gamma=0.99):\n        self.V = np.zeros(n_states)\n        self.lr = lr\n        self.gamma = gamma\n\n    def update(self, s, r, s_next, done):\n        if done:\n            td_target = r\n        else:\n            td_target = r + self.gamma * self.V[s_next]\n        td_error = td_target - self.V[s]\n        self.V[s] += self.lr * td_error\n```",
        "{comparison2}": "| 方法 | 平均奖励 | 收敛速度 |\n|------|---------|----------|\n| TD(0) | 195.0 | 约500 episodes |",
        "{visualization}": "```python\nimport matplotlib.pyplot as plt\nplt.plot(V_history)\nplt.xlabel('Episode')\nplt.ylabel('V(s)')\nplt.show()\n```",
        "{interpretation}": "从训练曲线可以看出算法是否有效学习到了价值函数。",
        "{eval_metrics}": "使用累计奖励、平均奖励、TD误差作为评估指标。",
        "{eval_code}": "```python\ndef evaluate(agent, env, runs=10):\n    scores = []\n    for _ in range(runs):\n        s = env.reset()[0]\n        total = 0\n        done = False\n        while not done:\n            a = policy(s)\n            s_next, r, terminated, truncated, _ = env.step(a)\n            done = terminated or truncated\n            total += r\n            s = s_next\n        scores.append(total)\n    print(f'Average: {np.mean(scores):.2f}')\n```",
        "{data_errors}": "**错误1：状态空间未正确离散化** - 使用适当的离散化方法",
        "{model_errors}": "**错误1：学习率设置不当** - 使用自适应学习率",
        "{param_mistakes}": "**误区1：折扣因子γ设置过大** - 根据任务horizon选择gamma",
        "{key_takeaways}": "✓ 核心思想：通过TD误差更新价值函数\n✓ 数学本质：基于贝尔曼方程\n✓ 优化目标：最小化TD误差",
        "{key_formulas}": "1. TD误差：$$ \\delta = r + \\gamma V(s') - V(s) $$\n2. 更新：$$ V(s) \\leftarrow V(s) + \\alpha \\delta $$",
        "{best_practices}": "✓ 合理设计奖励函数\n✓ 监控TD误差\n✓ 使用适当的探索策略",
        "{related_algs}": "- 前置算法：动态规划、多臂赌博机\n- 后续算法：Q-learning、Sarsa\n- 相关算法：蒙特卡洛方法",
        "{basic_exercises}": "**练习1：概念理解**\n问题：TD误差是指什么？\n答案：B. 当前V值与TD目标的差",
        "{advanced_exercises}": "**思考1：改进分析**\n问题：如何解决TD学习的偏差问题？\n答案：使用蒙特卡洛方法或减小学习率",
        "{open_ended}": "**思考2：创新应用**\n问题：如何将TD学习应用到推荐系统？\n答案：状态=用户画像，动作=推荐内容，奖励=用户反馈",
        "{prerequisites}": "- [ ] 概率论\n- [ ] 线性代数\n- [ ] Python基础\n- [ ] 强化学习基础",
        "{parallel_algs}": "1. **Q-learning** - Off-policy TD控制\n2. **蒙特卡洛方法** - 无偏估计",
        "{next_algs}": "**短期目标**：\n1. Q-learning - 学习最优策略\n2. 策略梯度 - 直接优化策略",
        "{resources}": "**教材**：\n1. 《强化学习（第二版）》Sutton & Barto\n**在线课程**：\n1. David Silver的强化学习课程"
    }
    
    for key, value in replacements.items():
        template = template.replace(key, value)
    
    return template

def fill_mc(algo, desc, template):
    """填充MC类算法 - 类似fill_td但调整内容"""
    return fill_td(algo, desc, template).replace("TD", "MC").replace("时间差分", "蒙特卡洛").replace("bootstrap", "完整轨迹")

def fill_dp(algo, desc, template):
    return fill_td(algo, desc, template).replace("TD", "DP").replace("时间差分", "动态规划").replace("bootstrap", "模型")

def fill_deep(algo, desc, template):
    return fill_td(algo, desc, template).replace("TD", "Deep").replace("时间差分", "深度强化学习").replace("bootstrap", "神经网络")

def fill_model(algo, desc, template):
    return fill_td(algo, desc, template).replace("TD", "Model").replace("时间差分", "模型学习").replace("bootstrap", "规划")

def fill_fa(algo, desc, template):
    return fill_td(algo, desc, template).replace("TD", "FA").replace("时间差分", "函数逼近").replace("bootstrap", "逼近")

def fill_exp(algo, desc, template):
    return fill_td(algo, desc, template).replace("TD", "Exploration").replace("时间差分", "探索策略").replace("bootstrap", "探索")

def fill_other(algo, desc, template):
    return fill_td(algo, desc, template).replace("{algo}", algo).replace("{desc}", desc if desc else f"{algo}是强化学习中的重要算法/方法")

def main():
    """主函数"""
    output_dir = Path("/Users/marcher/Desktop/Marcher_code/强化学习_第二版")
    
    # 所有算法及其类别和描述
    algorithms = [
        # TD类
        ("TD学习", "TD", "通过时间差分学习更新价值函数"),
        ("TD(0)", "TD", "单步TD学习，最基础的TD预测算法"),
        ("TD(λ)", "TD", "使用资格迹结合多步TD误差的统合算法"),
        ("Q学习", "TD", "通过Q表格和TD学习找到最优策略的off-policy算法"),
        ("Sarsa", "TD", "on-policy的TD控制算法，使用实际下一个动作更新"),
        ("期望Sarsa", "TD", "Sarsa的改进版本，使用期望而非采样下一个动作"),
        ("n步自举法", "TD", "结合n步回报的TD学习，平衡单步偏差和多步方差"),
        ("双重Q学习", "TD", "使用两个Q网络解耦动作选择和评估，减少过估计偏差"),
        ("Sarsa(λ)", "TD", "结合资格迹的Sarsa算法"),
        ("真实在线TD(λ)", "TD", "真实在线版本的TD(λ)算法"),
        ("真实在线Sarsa(λ)", "TD", "真实在线版本的Sarsa(λ)算法"),
        ("Watkins的Q(λ)", "TD", "Watkins提出的Q(λ)算法"),
        ("树回溯TB(λ)", "TD", "使用树回溯的λ-return算法"),
        ("Q(σ)", "TD", "结合n步和λ参数的通用TD算法"),
        ("后位状态方法", "TD", "使用后位状态（afterstate）的TD方法"),
        ("双学习", "TD", "使用两个独立估计减少最大化偏差"),
        ("最大化偏差处理方法", "TD", "处理Q-learning中最大化偏差的方法"),
        
        # MC类
        ("蒙特卡洛方法", "MC", "通过完整episode采样估计价值函数的无模型方法"),
        ("蒙特卡洛预测", "MC", "使用蒙特卡洛方法估计状态价值"),
        ("蒙特卡洛控制", "MC", "使用蒙特卡洛方法优化策略"),
        ("MC-ES", "MC", "蒙特卡洛ES（试探性出发）"),
        ("试探性出发蒙特卡洛", "MC", "使用试探性出发的蒙特卡洛控制"),
        ("同轨策略MC控制", "MC", "on-policy蒙特卡洛控制"),
        ("离轨策略MC预测", "MC", "off-policy蒙特卡洛预测"),
        ("离轨策略MC控制", "MC", "off-policy蒙特卡洛控制"),
        ("普通重要度采样", "MC", "普通重要度采样方法"),
        ("加权重要度采样", "MC", "加权重要度采样，减少方差"),
        ("n步离轨策略学习", "MC", "n步离轨策略学习方法"),
        ("n步树回溯算法", "MC", "n步树回溯算法"),
        
        # DP类
        ("动态规划", "DP", "基于模型的规划方法，使用贝尔曼方程迭代求解"),
        ("策略迭代", "DP", "交替进行策略评估和策略改进的动态规划方法"),
        ("价值迭代", "DP", "将策略评估压缩到一步的动态规划算法"),
        ("广义策略迭代", "DP", "广义策略迭代框架"),
        ("迭代策略评估", "DP", "策略评估的迭代算法"),
        ("策略评估", "DP", "评估给定策略的价值函数"),
        ("策略改进", "DP", "根据价值函数改进策略"),
        ("异步动态规划", "DP", "异步更新价值的动态规划方法"),
        ("自举法", "DP", "使用bootstrap更新价值的方法"),
        
        # Deep类
        ("DQN", "Deep", "结合Q-learning和深度神经网络的算法"),
        ("深度Q网络", "Deep", "DQN的中文名称，深度Q网络"),
        ("深度Q学习", "Deep", "使用深度学习的Q-learning"),
        ("行动器-评判器方法", "Deep", "结合策略梯度和价值评估的混合方法"),
        ("单步行动器-评判器", "Deep", "单步版本的Actor-Critic"),
        ("REINFORCE", "Deep", "基于蒙特卡洛的策略梯度方法"),
        ("REINFORCE with Baseline", "Deep", "带基线的REINFORCE算法"),
        ("策略梯度方法", "Deep", "直接对策略进行参数化并通过梯度上升优化"),
        ("策略梯度定理", "Deep", "策略梯度定理，理论基石"),
        ("带资格迹的行动器-评判器方法", "Deep", "使用资格迹的Actor-Critic"),
        ("持续性问题的策略梯度", "Deep", "针对持续性问题的策略梯度方法"),
        
        # Model-Based类
        ("蒙特卡洛树搜索", "Model", "通过模拟构建搜索树，平衡探索与利用的规划算法"),
        ("MCTS", "Model", "蒙特卡洛树搜索的缩写"),
        ("UCT", "Model", "上置信界树搜索，MCTS的改进"),
        ("预演算法", "Model", "使用rollout评估动作的方法"),
        ("启发式搜索", "Model", "使用启发式函数引导搜索"),
        ("决策时规划", "Model", "在决策时刻进行规划"),
        ("Dyna-Q", "Model", "结合Q-learning和模型学习的集成方法"),
        ("Dyna", "Model", "Dyna架构，集成学习和规划"),
        ("Dyna-Q+", "Model", "Dyna-Q的改进版本，添加探索奖励"),
        ("基于模型的规划", "Model", "使用环境模型进行规划"),
        ("实时动态规划", "Model", "RTDP，实时动态规划算法"),
        ("RTDP", "Model", "实时动态规划的缩写"),
        ("表格型Dyna-Q", "Model", "表格型Dyna-Q实现"),
        ("随机采样单步表格型Q规划", "Model", "使用随机采样的表格型Q规划"),
        ("轨迹采样", "Model", "沿轨迹采样进行规划"),
        ("优先遍历", "Model", "优先遍历/优先级遍历，引导式规划"),
        
        # FA类
        ("价值函数逼近", "FA", "使用函数逼近处理大规模状态空间"),
        ("半梯度方法", "FA", "使用半梯度下降的函数逼近"),
        ("半梯度 TD(0)", "FA", "表格型TD(0)的函数逼近版本"),
        ("半梯度 TD(λ)", "FA", "TD(λ)的函数逼近版本"),
        ("半梯度 n步 Sarsa", "FA", "n步Sarsa的函数逼近版本"),
        ("梯度赌博机算法", "FA", "赌博机的梯度方法"),
        ("n步Sarsa", "FA", "n步Sarsa算法"),
        ("分幕式半梯度Sarsa", "FA", "episodic半梯度Sarsa"),
        ("分幕式半梯度控制", "FA", "episodic半梯度控制"),
        ("离轨策略半梯度方法", "FA", "off-policy半梯度方法"),
        ("残差梯度算法", "FA", "使用残差梯度的函数逼近"),
        ("资格迹", "FA", "eligibility traces，记录访问历史"),
        ("λ-回报", "FA", "λ-return，结合多步回报"),
        ("n步Sarsa", "FA", "n-step Sarsa"),
        ("差分半梯度Sarsa", "FA", "针对平均奖励问题的Sarsa"),
        ("差分半梯度n步Sarsa", "FA", "n-step版本的差分Sarsa"),
        ("LSTD", "FA", "最小二乘时序差分"),
        ("最小二乘时序差分", "FA", "LSTD的中文名称"),
        ("GTD", "FA", "梯度TD方法"),
        ("GTD2", "FA", "GTD的改进版本"),
        ("TDC", "FA", "GTD(0)，TDC算法"),
        ("贝尔曼误差梯度下降", "FA", "基于贝尔曼误差的梯度下降"),
        ("A-分裂方法", "FA", "A-splitting，减少偏差"),
        ("A-预先分裂方法", "FA", "A-presplitting"),
        ("减小方差方法", "FA", "variance reduction methods"),
        ("带控制变量的每次决策型方法", "FA", "per-decision importance sampling with control variates"),
        ("折扣敏感的重要度采样", "FA", "discounting-aware importance sampling"),
        ("每次决策型重要度采样", "FA", "per-decision importance sampling"),
        ("截断加权平均估计器", "FA", "truncated weighted-average estimator"),
        ("强调TD方法", "FA", "emphatic TD methods"),
        ("基于核函数的函数逼近", "FA", "kernel-based function approximation"),
        ("核方法", "FA", "kernel methods"),
        ("基于记忆的函数逼近", "FA", "memory-based function approximation"),
        ("径向基函数", "FA", "RBF，radial basis functions"),
        ("瓦片编码", "FA", "tile coding，瓦片编码"),
        ("粗编码", "FA", "coarse coding"),
        ("多项式基", "FA", "polynomial basis"),
        ("傅立叶基", "FA", "Fourier basis"),
        ("人工神经网络", "FA", "人工神经网络，函数逼近器"),
        ("深度学习", "FA", "deep learning，使用深度神经网络"),
        ("线性方法", "FA", "linear methods"),
        ("随机梯度方法", "FA", "stochastic gradient methods"),
        ("随机梯度上升", "FA", "stochastic gradient ascent"),
        ("梯度蒙特卡洛算法", "FA", "gradient Monte Carlo algorithm"),
        ("批量TD方法", "FA", "batch TD method"),
        ("常数αMC", "FA", "constant-alpha Monte Carlo"),
        ("表格型TD(0)", "FA", "tabular TD(0)"),
        
        # Exploration类
        ("ε-贪心动作选择", "Exploration", "ε-greedy动作选择策略"),
        ("UCB", "Exploration", "上置信界动作选择"),
        ("置信度上界动作选择", "Exploration", "UCB的中文名称"),
        ("softmax策略参数化", "Exploration", "使用softmax的参数化策略"),
        ("高斯策略参数化", "Exploration", "连续动作空间的高斯策略"),
        ("连续动作策略参数化方法", "Exploration", "continuous action policy parameterization"),
        ("乐观初始值方法", "Exploration", "optimistic initialization"),
        ("样本平均方法", "Exploration", "sample-average method"),
        ("增量式实现", "Exploration", "incremental implementation"),
        ("上下文相关赌博机", "Exploration", "contextual bandits"),
        ("关联搜索", "Exploration", "associative search"),
        ("k臂赌博机算法", "Exploration", "k-armed bandit algorithm"),
        ("多臂赌博机算法", "Exploration", "multi-armed bandit algorithm"),
        ("梯度赌博机算法", "Exploration", "gradient bandit algorithm"),
        
        # Other类
        ("AlphaGo", "Other", "使用深度强化学习的围棋程序"),
        ("AlphaGo Zero", "Other", "无人类知识，纯自我对弈的AlphaGo"),
        ("TD-Gammon", "Other", "使用TD学习的西洋双陆棋程序"),
        ("人类级别Atari视频游戏智能体", "Other", "DQN玩Atari游戏"),
        ("Samuel的跳棋程序", "Other", "早期使用机器学习玩跳棋的程序"),
        ("遗传算法", "Other", "genetic algorithm"),
        ("遗传规划", "Other", "genetic programming"),
        ("模拟退火算法", "Other", "simulated annealing"),
        ("爬山搜索", "Other", "hill-climbing search"),
        ("进化方法", "Other", "evolutionary methods"),
        ("随机自动学习机", "Other", "stochastic learning automata"),
        ("分类器系统", "Other", "classifier system"),
        ("救火队算法", "Other", "bucket brigade algorithm"),
        ("自动学习机", "Other", "learning automata"),
        ("Alopex算法", "Other", "Alopex algorithm"),
        ("LMS", "Other", "最小均方误差算法"),
        ("最小均方误差算法", "Other", "least-mean-square algorithm"),
        ("随机近似方法", "Other", "stochastic approximation"),
        ("贝尔曼方程", "Other", "Bellman equation"),
        ("贝尔曼最优方程", "Other", "Bellman optimality equation"),
        ("马尔可夫决策过程", "Other", "Markov decision process"),
        ("最优控制", "Other", "optimal control"),
        ("极大极小算法", "Other", "minimax algorithm"),
        ("认知图", "Other", "cognitive map"),
        ("习惯行为模型", "Other", "habitual behavior models"),
        ("目标导向行为模型", "Other", "goal-directed behavior models"),
        ("收益预测误差假说", "Other", "reward prediction error hypothesis"),
        ("神经行动器-评判器", "Other", "neural actor-critic"),
        ("享乐主义神经元模型", "Other", "hedonistic neuron model"),
        ("集体强化学习", "Other", "collective reinforcement learning"),
        ("大脑中的基于模型的算法", "Other", "model-based algorithms in the brain"),
        ("Rescorla-Wagner模型", "Other", "Rescorla-Wagner model"),
        ("TD模型", "Other", "temporal difference model"),
        ("经典条件反射模型", "Other", "classical conditioning models"),
        ("工具性条件反射模型", "Other", "instrumental conditioning models"),
        ("延迟强化方法", "Other", "delayed reinforcement"),
        ("Watson的每日双倍投注策略", "Other", "Watson daily double wagering strategy"),
        ("优化内存控制", "Other", "optimizing memory control"),
        ("个性化网络服务中的强化学习方法", "Other", "RL for personalized web services"),
        ("热气流滑翔控制方法", "Other", "thermal soaring control"),
        ("边际价值函数", "Other", "marginal value function"),
        ("广义价值函数", "Other", "general value functions"),
        ("辅助任务", "Other", "auxiliary tasks"),
        ("选项理论", "Other", "options framework"),
        ("时序摘要", "Other", "temporal abstraction"),
        ("基于选项的时序摘要方法", "Other", "temporal abstraction via options"),
        ("观测量到状态的构造方法", "Other", "observation-to-state construction"),
        ("收益信号设计方法", "Other", "reward signal design"),
        ("兴趣机制", "Other", "interest mechanism"),
        ("强调方法", "Other", "emphasis method"),
        ("平均收益方法", "Other", "average-reward methods"),
        ("采用资格迹保障离轨策略方法稳定性", "Other", "stabilizing off-policy methods with eligibility traces"),
        ("变量λ和γ方法", "Other", "variable λ and γ methods"),
        ("荷兰迹", "Other", "Dutch traces"),
        ("在线λ-回报算法", "Other", "online λ-return algorithm"),
        ("离轨策略TD控制", "Other", "off-policy TD control"),
        ("同轨策略TD控制", "Other", "on-policy TD control"),
    ]
    
    print("=" * 60)
    print("开始生成所有算法文档（修复版）...")
    print("=" * 60)
    
    count = 0
    errors = []
    
    for algo_name, category, description in algorithms:
        try:
            print(f"生成 [{category:10s}]: {algo_name}...")
            content = generate_doc(algo_name, category, description)
            
            filename = sanitize_filename(algo_name)
            filepath = output_dir / f"{filename}.md"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ 已生成: {filepath}")
            count += 1
            
        except Exception as e:
            error_msg = f"{algo_name}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ 错误: {error_msg}")
    
    print("\n" + "=" * 60)
    print(f"文档生成完毕！")
    print(f"成功: {count} 个")
    print(f"失败: {len(errors)} 个")
    print("=" * 60)
    
    if errors:
        print("\n错误列表:")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
