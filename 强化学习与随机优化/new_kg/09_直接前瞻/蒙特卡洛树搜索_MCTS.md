# 蒙特卡洛树搜索(MCTS) 学习文档

> MCTS通过有策略地构建搜索树并结合随机模拟，在不完整信息下做出高质量决策，是AlphaGo的核心算法。

> 来源线索：本节内容根据原书中关于"Monte Carlo Tree Search for Discrete Decisions"的相关章节(Ch 19.8)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：蒙特卡洛树搜索（MCTS）是一种启发式搜索算法，通过反复执行"选择-扩展-模拟-回传"四步循环，逐步构建搜索树来找到最优决策。

**直觉类比**：想象你在一个未知的棋局中选下一步。你不能穷举所有可能（太多了），所以你先试几步看起来有希望的走法，对每一步快速"模拟"到终局看结果。走法好的就多探索几次，走法差的就少花时间。随着探索次数增加，你越来越清楚哪步最优。

**历史背景**：MCTS由Rémi Coulom(2006)和Kocsis & Szepesvári(2006)独立发展。UCB1应用于树搜索（即UCT）是关键突破。2016年AlphaGo（Silver et al.）将MCTS与深度神经网络结合，击败世界围棋冠军，使MCTS广为人知。

**算法定位**：直接前瞻策略(DLA)/搜索方法。在原书四类策略中属于DLA——通过构建前瞻搜索树来近似最优决策。

**前置知识**：决策树、UCB策略、蒙特卡洛方法、博弈论基础。

## 2. 核心原理

**核心思想**：MCTS通过非对称地构建搜索树，将计算资源集中在最有希望的分支。它不需要完整的游戏知识（如评估函数），只需要能模拟到终局并判断胜负的能力。

**四步循环**：

1. **选择（Selection）**：从根节点开始，用树策略（如UCB）选择子节点，直到到达一个未完全展开的节点
2. **扩展（Expansion）**：为选中的节点添加一个或多个子节点
3. **模拟（Simulation/Rollout）**：从新节点开始，用默认策略（通常随机）快速模拟到终局
4. **回传（Backpropagation）**：将模拟结果沿路径回传，更新路径上所有节点的统计信息

```
        根(S₀)  ← 选择: 用UCB选最优子节点
       ╱    ╲
    节点A   节点B  ← A被选中(UCB最高)
    ╱╲      ╱╲
  C  D  [E]  F   ← 扩展: E是新展开的节点
              │
           Rollout  ← 模拟: 从E随机走到终局
              │
           结果=+1  ← 回传: 更新A,C,E的胜率
```

**关键概念**：

- **树策略（Tree Policy）**：在选择阶段选择子节点的策略，平衡探索和利用
- **默认策略（Default Policy）**：在模拟阶段选择动作的策略，通常随机
- **UCB公式**：$UCB(s,a) = \frac{W(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$
- **UCT**：Upper Confidence Bound for Trees，将UCB应用于树搜索
- **乐观初始化**：未访问的节点被视为无限潜力

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $N(s)$ | 状态$s$被访问的次数 |
| $N(s,a)$ | 在状态$s$执行动作$a$的次数 |
| $W(s,a)$ | 动作$(s,a)$的累积奖励 |
| $Q(s,a)$ | 动作$(s,a)$的平均奖励，$W/N$ |
| $c$ | 探索常数 |
| $\Delta$ | 模拟结果（胜负） |

### UCB选择公式

在选择阶段，MCTS用UCB1公式选择子节点：

$$a^* = \arg\max_{a} \left[\frac{W(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right]$$

- 第一项$\frac{W}{N}$：已知的平均回报（利用项）
- 第二项$c\sqrt{\frac{\ln N(s)}{N(s,a)}}$：不确定性 bonuses（探索项）
- $N(s,a)$越小，探索项越大，鼓励尝试未充分探索的动作

### 为什么UCB有效

UCB基于Hoeffding不等式。对于$K$个臂，选择使置信上界最大的动作，可以证明累积遗憾的期望为$O(\sqrt{KT \ln T})$，接近理论下界。

### 节点更新

模拟完成后，回传结果$\Delta$沿路径更新：

$$N(s,a) \leftarrow N(s,a) + 1$$
$$W(s,a) \leftarrow W(s,a) + \Delta$$
$$Q(s,a) \leftarrow W(s,a) / N(s,a)$$

### 乐观MCTS

乐观MCTS（原书Ch 19.8.4）对未访问的子节点使用乐观初始值，加速早期探索。

### 与Minimax的区别

Minimax需要完整搜索树和评估函数。MCTS只需要模拟能力，通过采样自动"发现"好的走法。

## 4. 训练过程讲解

### 参数初始化
- 根节点：$N=0$, $W=0$
- 探索常数$c$：通常$\sqrt{2}$或$\approx 1.41$
- 每个决策点的模拟次数：$n_{sim}$

### 迭代过程
1. 收到当前状态$s$
2. 循环$n_{sim}$次：
   a. 从$s$开始选择到叶节点
   b. 扩展叶节点
   c. 随机模拟到终局
   d. 回传结果
3. 选择$Q$值最高的动作

### 收敛条件
- 固定模拟次数（如1000次/每步）
- 或时间限制
- 模拟次数越多，决策质量越高

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $n_{sim}$ | 模拟次数 | [100, 100000] | 1000 |
| $c$ | 探索常数 | [0.5, 2.0] | 1.41 |
| max_depth | 树最大深度 | [10, 100] | 50 |

## 5. 应用场景

### 1. 棋类游戏AI
为什么适合：围棋等游戏的分支因子巨大（约250），Minimax不可行。MCTS通过采样找到好的走法。

### 2. 规划问题
为什么适合：机器人路径规划、调度等问题可以建模为搜索树，MCTS能高效找到近似最优解。

### 3. 推荐系统
为什么适合：可以建模用户交互序列，MCTS搜索最优推荐策略。

### 不适用场景
- 确定性环境且分支因子小（用Minimax更高效）
- 实时性要求极高（MCTS需要一定计算时间）
- 无法模拟到终局（需要估计中间状态价值）

## 6. 优缺点分析

### 优点
1. **不需要领域知识**：只需模拟能力（不需要评估函数）
2. **非对称搜索**：集中计算资源在有希望的分支
3. **任何时间算法**：可以在任意时间点停止并返回当前最佳
4. **理论收敛**：模拟次数趋于无穷时收敛到最优

### 缺点
1. **计算密集**：需要大量模拟
2. **依赖模拟质量**：默认策略太随机可能错过关键走法
3. **高分支因子时效率下降**：动作太多时搜索树太宽
4. **不擅长长期策略**：纯随机模拟难以发现深层策略

### 算法对比

| 特性 | MCTS | Minimax | Q-Learning |
|------|------|---------|------------|
| 需要模型 | 只需模拟 | 需要完整模型 | 不需要 |
| 领域知识 | 最少 | 需要评估函数 | 不需要 |
| 分支因子容忍 | 高 | 低 | 任意 |
| 在线决策 | 是 | 是 | 是（查表） |
| 计算量/每步 | 可调 | 固定 | 极低 |

## 7. 调库实现

```python
"""
使用自定义MCTS实现Tic-Tac-Toe
"""
import numpy as np
import math
import time

class TicTacToe:
    """井字棋环境"""
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0=空, 1=X, -1=O
        self.current_player = 1

    def get_valid_actions(self):
        return np.where(self.board == 0)[0]

    def step(self, action):
        self.board[action] = self.current_player
        winner = self._check_winner()
        self.current_player *= -1
        return winner

    def _check_winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a,b,c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return 0 if 0 in self.board else None  # 0=继续, None=平局

    def clone(self):
        g = TicTacToe()
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g


class MCTSNode:
    """MCTS节点"""
    def __init__(self, game, action=None, parent=None):
        self.game = game
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried = list(game.get_valid_actions())

    def ucb(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda x: x.ucb())

    def expand(self):
        action = self.untried.pop()
        new_game = self.game.clone()
        result = new_game.step(action)
        child = MCTSNode(new_game, action, self)
        self.children.append(child)
        return child, result

    def update(self, result, player):
        self.visits += 1
        if result == player:
            self.wins += 1.0
        elif result is None:
            self.wins += 0.5


def mcts_search(game, n_simulations=1000, time_limit=None):
    """MCTS搜索：返回最佳动作"""
    root = MCTSNode(game)
    start = time.time()

    for _ in range(n_simulations):
        if time_limit and time.time() - start > time_limit:
            break

        node = root
        player = game.current_player

        # 1. 选择：沿UCB最大路径下行
        while not node.untried and node.children:
            node = node.select_child()

        # 2. 扩展：添加一个新子节点
        if node.untried:
            node, result = node.expand()
            if result == 0:  # 游戏继续
                # 3. 模拟：随机走到终局
                sim_game = node.game.clone()
                while True:
                    valid = sim_game.get_valid_actions()
                    if len(valid) == 0:
                        break
                    result = sim_game.step(np.random.choice(valid))
                    if result != 0:
                        break
        # result此时是终局结果

        # 4. 回传
        while node is not None:
            node.update(result, player)
            node = node.parent

    # 选择访问次数最多的动作
    return max(root.children, key=lambda x: x.visits).action


# 测试
if __name__ == "__main__":
    game = TicTacToe()
    while True:
        valid = game.get_valid_actions()
        if len(valid) == 0:
            print("平局!")
            break
        if game.current_player == 1:
            action = mcts_search(game, n_simulations=500)
        else:
            action = np.random.choice(valid)
        result = game.step(action)
        print(f"玩家{'X' if action else 'O'}下在位置{action}")
        print(game.board.reshape(3,3))
        print()
        if result != 0:
            print(f"{'X' if result==1 else 'O'}获胜!")
            break
        if result is None:
            print("平局!")
            break
```

## 8. 手工代码实现

```python
"""
从零实现MCTS（纯NumPy）
通用框架，可适配不同环境
"""
import numpy as np
import math

class MCTS:
    """通用蒙特卡洛树搜索"""

    def __init__(self, env, simulate_fn, c=1.41):
        """
        参数：
            env: 环境对象（需提供clone/get_actions/step方法）
            simulate_fn: 模拟函数，接受环境返回结果
            c: UCB探索常数
        """
        self.env = env
        self.simulate_fn = simulate_fn
        self.c = c

    class Node:
        __slots__ = ['action', 'parent', 'children', 'visits', 'value', 'untried']
        def __init__(self, action=None, parent=None, n_actions=0):
            self.action = action
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.untried = list(range(n_actions))

        def ucb(self, c):
            if self.visits == 0:
                return float('inf')
            exploit = self.value / self.visits
            explore = c * math.sqrt(math.log(self.parent.visits) / max(self.visits, 1))
            return exploit + explore

    def search(self, state, n_simulations=1000):
        """执行MCTS搜索"""
        root = self.Node(n_actions=len(state.get_actions()))

        for _ in range(n_simulations):
            # 复制环境状态
            sim_state = state.clone()
            node = root

            # 选择：UCB下行
            while not node.untried and node.children:
                node = max(node.children, key=lambda n: n.ucb(self.c))
                sim_state.step(node.action)

            # 扩展
            if node.untried:
                action = node.untried.pop(np.random.randint(len(node.untried)))
                sim_state.step(action)
                child = self.Node(action=action, parent=node,
                                n_actions=len(sim_state.get_actions()))
                node.children.append(child)
                node = child

            # 模拟
            result = self.simulate_fn(sim_state)

            # 回传
            while node is not None:
                node.visits += 1
                node.value += result
                node = node.parent

        # 返回访问次数最多的动作
        if not root.children:
            return None
        return max(root.children, key=lambda n: n.visits).action
```

## 9. 可视化与结果理解

```python
"""MCTS可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_mcts_winrate(n_simulations_list, win_rates):
    """MCTS性能随模拟次数的变化"""
    plt.figure(figsize=(8, 5))
    plt.plot(n_simulations_list, win_rates, 'o-')
    plt.xlabel('每步模拟次数')
    plt.ylabel('胜率 vs 随机对手')
    plt.title('MCTS性能与计算量关系')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('mcts_performance.png', dpi=150)
    plt.show()
```

**结果解读**：模拟次数越多，MCTS的决策质量越高。通常呈对数增长关系——模拟次数翻倍，性能提升递减。

## 10. 模型评估

```python
"""评估MCTS策略"""
def evaluate_mcts(env_class, mcts_simulations=1000, n_games=100):
    wins = 0
    for _ in range(n_games):
        env = env_class()
        while True:
            valid = env.get_valid_actions()
            if len(valid) == 0: break
            action = mcts_search(env, n_simulations=mcts_simulations)
            if action is None: break
            result = env.step(action)
            if result is not None and result != 0:
                if result == 1: wins += 1
                break
    print(f"MCTS({mcts_simulations}次模拟)胜率: {wins/n_games:.1%}")
```

## 11. 常见问题与易错点

### 数据层面

1. **环境模拟不一致**
   - 现象：MCTS找到的"最优"动作实际效果差
   - 原因：模拟环境的规则与真实环境不完全一致
   - 解决方案：确保模拟环境精确反映真实环境

2. **状态表示遗漏信息**
   - 现象：相同局面MCTS做出不同选择
   - 原因：状态表示缺少关键信息导致不同局面看起来相同
   - 解决方案：状态编码必须包含所有必要信息

### 模型层面

3. **探索常数c设置不当**
   - 现象：搜索太集中（c太小）或太分散（c太大）
   - 原因：c控制探索-利用平衡
   - 解决方案：通常从$c=\sqrt{2}$开始调整

4. **模拟策略太随机**
   - 现象：大量模拟浪费在明显差的走法上
   - 原因：默认策略纯随机，忽略领域知识
   - 解决方案：使用启发式默认策略或学习型Rollout

### 调参层面

5. **模拟次数不足**
   - 现象：MCTS表现不如预期
   - 原因：模拟次数太少，搜索树不够深
   - 解决方案：增加模拟次数或使用时间限制而非次数限制

## 12. 学习总结

MCTS的核心创新在于将蒙特卡洛模拟与选择性搜索树结合。它不需要领域专家评估函数，只需要能模拟到终局。通过UCB策略，MCTS非对称地将计算资源集中在有希望的分支。

**关键公式**：
1. UCB选择：$a^* = \arg\max_a [Q(s,a)/N(s,a) + c\sqrt{\ln N(s)/N(s,a)}]$
2. 回传更新：$N(s,a) \leftarrow N(s,a)+1$, $W(s,a) \leftarrow W(s,a)+\Delta$
3. 最终选择：$a^* = \arg\max_a N(s,a)$

在原书框架中，MCTS属于直接前瞻策略(DLA)的一种实现。它与其他搜索方法的区别在于：不需要完整搜索树（vs Minimax），不需要评估函数（vs Alpha-Beta），通过采样自动发现好的策略。AlphaGo将MCTS与深度学习结合（用网络替代模拟），是这一思路的巅峰应用。

## 13. 练习题与思考题

### 基础题

**题目1**：在MCTS中，如果一个节点被访问了100次，其中60次获胜，其父节点被访问了1000次。UCB值是多少？（设$c=\sqrt{2}$）

**参考答案**：
$UCB = \frac{60}{100} + \sqrt{2}\sqrt{\frac{\ln 1000}{100}} = 0.6 + 1.414 \times \sqrt{\frac{6.908}{100}} = 0.6 + 1.414 \times 0.2628 = 0.6 + 0.371 = 0.971$

### 进阶题

**题目2**：为什么MCTS最终选择访问次数最多的动作而非Q值最高的动作？

**参考答案**：
访问次数$N(s,a)$比Q值更稳定可靠。Q值可能因少量极端结果而偏离（如一次偶然的大胜），但访问次数反映了算法对该动作的总体信心——只有在多次选择后仍然保持高胜率的动作才会被持续访问。此外，UCB的探索项会降低高频访问节点的UCB值，所以能积累最多访问次数的节点通常是真正最优的。

### 开放思考题

**题目3**：原书将MCTS归类为"直接前瞻策略(DLA)"。请对比MCTS与"值函数近似(VFA)"方法（如Q-Learning）的优劣。什么情况下MCTS更合适？

**参考答案方向**：
- MCTS优势：不需要大量训练数据，任何时间可停止，适合一次性决策
- VFA优势：决策速度快（查表），训练后可反复使用
- MCTS适合：(1)可以快速模拟的环境；(2)需要可解释决策；(3)离线无法预先训练的场景
- VFA适合：(1)环境模拟代价高；(2)需要实时决策；(3)可以离线大量训练的场景

## 14. 学习路径建议

**前置算法**：决策树搜索、UCB/多臂赌博机、蒙特卡洛方法

**平行算法**：Minimax/Alpha-Beta剪枝、Q-Learning

**进阶算法**：AlphaGo（MCTS+深度学习）、乐观MCTS、POMCP（部分可观测MCTS）

**推荐资源**：
1. 原书Ch 19.8 "Monte Carlo Tree Search for Discrete Decisions"
2. Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods"
3. Silver et al. (2016) "Mastering the Game of Go with Deep Neural Networks and Tree Search" (AlphaGo论文)
