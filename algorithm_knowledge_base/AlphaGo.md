# AlphaGo 学习文档

## 1. 算法基础认知

### 1.1 研究背景

AlphaGo是由DeepMind在2014-2016年间开发的围棋AI，其击败了世界顶级围棋选手李世石（4:1）和柯洁（3:0），成为第一个在完全信息博弈中击败人类冠军的计算机程序。AlphaGo代表了深度强化学习和蒙特卡洛树搜索的突破性结合。

### 1.2 核心思想

AlphaGo的核心创新是结合深度神经网络（策略网络和价值网络）与蒙特卡洛树搜索（MCTS）。策略网络用于选择下一步走法，价值网络用于评估局面，而MCTS用于规划长期策略。

### 1.3 技术定位

AlphaGo属于**深度强化学习+博弈树搜索**的混合方法，是强化学习在完美信息博弈中的成功应用典范。

---

## 2. 核心原理

### 2.1 策略网络

从人类棋谱学习策略：

$$\pi(a|s) = \text{NeuralNetwork}(s)$$

预测在给定局面$s$下，走法$a$的概率。

### 2.2 价值网络

评估局面的胜率：

$$v(s) = \text{ValueNetwork}(s)$$

预测在给定局面$s$下，黑/白方获胜的概率。

### 2.3 MCTS

使用MCTS进行规划：

```
MCTS
├── 选择：UCB选择最有希望的节点
├── 扩展：添加新节点
├── 模拟： rollout到游戏结束
└── 回传：更新节点统计
```

### 2.4 搜索公式

节点选择使用UCB公式：

$$UCB = \frac{Q(s,a)}{N(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

---

## 3. 数学公式与推导

### 3.1 策略网络损失

从人类棋谱学习：

$$\mathcal{L}_{SL} = -\sum \pi(a|s) \log \pi_{target}(a|s)$$

### 3.2 价值网络损失

从自对弈学习：

$$\mathcal{L}_{value} = (v(s) - z)^2$$

其中$z$是实际游戏结果。

### 3.3 搜索输出

MCTS输出的策略：

$$\pi(a|s) \propto \frac{1}{N(s,a)^{1/c}}$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
AlphaGo训练
├── 阶段1：SL策略网络
│   └── 从人类棋谱学习
├── 阶段2：RL策略网络
│   └── 自对弈强化学习
├── 阶段3：价值网络
│   └── 从棋局结果学习
└── 阶段4：MCTS搜索
    └── 组合策略和价值网络
```

### 4.2 超参数设置

| 参数 | 推荐值 |
|------|--------|
| 策略网络层数 | 13 |
| 价值网络层数 | 13 |
| 搜索模拟次数 | 1600 |
| 温度参数 | 自适应 |

### 4.3 自对弈

使用当前最强策略自对弈，收集棋局数据用于价值网络训练。

---

## 5. 应用场景

### 5.1 围棋

击败人类冠军，证明了深度强化学习在复杂博弈中的能力。

### 5.2 其他棋类

国际象棋、日本将棋等。

### 5.3 战略决策

长期规划的战略问题。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 超越人类 | 击败顶级选手 |
| 优雅设计 | 神经网络+MCTS |
| 可扩展 | 可应用到其他领域 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 需要大量计算 | 训练成本高 |
| 专业硬件 | 需要GPU/TPU |

---

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, board_size=19, hidden_channels=128):
        super().__init__()
        
        self.board_size = board_size
        
        self.conv1 = nn.Conv2d(4, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * board_size * board_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, board_size * board_size + 1),
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.head(x)
        
        return F.log_softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, board_size=19, hidden_channels=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(4, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * board_size * board_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.head(x)
        
        return x


class AlphaGo:
    """
    AlphaGo: Mastering the Game of Go without Human Knowledge
    Reference: https://www.nature.com/articles/nature24270
    """
    
    def __init__(self, board_size=19, device="cuda"):
        self.device = device
        self.board_size = board_size
        
        self.policy = PolicyNetwork(board_size).to(device)
        self.value = ValueNetwork(board_size).to(device)
        
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=0.001)
        
    def mcts_search(self, state, num_simulations=100):
        """简化的MCTS"""
        
        policy_log_probs = self.policy(state)
        
        actions = policy_log_probs.exp()
        return actions
    
    def get_action(self, state):
        """获取动作"""
        
        probs = self.mcts_search(state)
        action = probs.argmax(dim=-1)
        
        return action.item()
    
    def train_policy(self, states, actions):
        """训练策略网络"""
        
        log_probs = self.policy(states)
        
        loss = F.nll_loss(log_probs, actions)
        
        self.opt_policy.zero_grad()
        loss.backward()
        self.opt_policy.step()
        
        return loss.item()
    
    def train_value(self, states, winners):
        """训练价值网络"""
        
        pred_values = self.value(states)
        
        loss = F.mse_loss(pred_values.squeeze(), winners)
        
        self.opt_value.zero_grad()
        loss.backward()
        self.opt_value.step()
        
        return loss.item()


def main():
    """AlphaGo示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphago = AlphaGo(board_size=9, device=device)
    
    state = torch.randn(8, 4, 9, 9).to(device)
    actions = torch.randint(0, 82, (8,))
    winners = torch.randint(0, 2, (8,)).float().to(device)
    
    p_loss = alphago.train_policy(state, actions)
    print(f"Policy loss: {p_loss:.4f}")
    
    v_loss = alphago.train_value(state, winners)
    print(f"Value loss: {v_loss:.4f}")


if __name__ == "__main__":
    main()
```

---

## 8. 手工代码实现

```python
import torch
import torch.nn as nn


class SimplePolicyNet(nn.Module):
    """简化策略网络"""
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(81 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 82),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SimpleValueNet(nn.Module):
    """简化价值网络"""
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(81 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SimpleAlphaGo:
    """简化AlphaGo"""
    
    def __init__(self):
        self.policy = SimplePolicyNet()
        self.value = SimpleValueNet()
        
    def get_action(self, state):
        return self.policy(state).argmax()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphago = SimpleAlphaGo().to(device)
    
    state = torch.randn(1, 4, 9, 9).to(device)
    action = alphago.get_action(state)
    print(f"Action: {action.item()}")


if __name__ == "__main__":
    main()
```

---

## 9. 可视化与结果理解

AlphaGo展示了远超人类的围棋水平，其走法常被专业棋手研究学习。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 胜率 | 对人类选手 |
| ELO | 围棋等级分 |

### 10.2 性能

- 击败李世石：4:1
- 击败柯洁：3:0

---

## 11. 常见问题与易错点

训练需要大量GPU资源。

---

## 12. 学习总结

AlphaGo证明了深度强化学习在复杂博弈中的能力，开创了AI+人类竞技的新时代。

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题

**1. AlphaGo使用哪些网络？**
A. 策略网络
B. 策略网络+价值网络
C. 只有判别器

答案：B

**2. AlphaGo使用什么搜索方法？**
A. 随机搜索
B. MCTS
C. 树搜索

答案：B

---


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：AlphaGo的[核心概念]是什么？

**答案**：**答案是[B]**。

**解析**：
AlphaGo的核心机制是[机制描述]。根据算法的数学定义，有：
$$核心公式$$
代入[具体值]后，验证可得正确答案为[B]。

选项分析：
- A：这是对[另一概念]的描述，与AlphaGo不符
- B：✓ 正确，这是[核心概念]的准确定义
- C：虽然有一定关联，但不是AlphaGo的主要特性
- D：这是[另一算法]的特征，在AlphaGo中不适用

#### 练习2：手动计算

**问题**：给定以下数据，请手动计算AlphaGo的[参数/结果]：
- 输入：$X = [x_1, x_2, ...]$
- 标签：$y = [y_1, y_2, ...]$

**答案**：**计算结果为[具体值]**

**解析**：
**步骤1**：根据AlphaGo的定义，计算[第一中间量]
$$第一计算 = [公式]$$
代入数据：$第一计算 = [代入数值] = [结果1]$

**步骤2**：继续计算[第二中间量]
$$第二计算 = [公式]$$
代入数据：$第二计算 = [结果2]$

**步骤3**：得到最终结果
$$最终结果 = f(第一计算, 第二计算) = [最终值]$$

**步骤4**：验证
将结果带回原式检验：$[验证过程]$，确认符合约束条件。

#### 思考题：改进分析

**问题**：AlphaGo在[特定场景]下效果不佳，请分析原因并提出改进方案。

**答案**：

**问题分析**：
1. [局限性1]：具体表现是[现象]，原因是[原因]
2. [局限性2]：具体表现是[现象]，原因是[原因]

**改进方案**：

**方案1：[改进方法名称]**
- **原理**：[解释改进的核心思想]
- **优势**：[改进后带来的好处]
- **实现**：[简要实现说明]

**方案2：[改进方法名称]**
- **原理**：[解释核心思想]
- **��价**：[需要付出的额外计算或复杂度]
- **适用场景**：[何时使用该改进]

## 14. 学习路径建议建议

学习深度学习、RL基础，理解MCTS，实现AlphaGo。