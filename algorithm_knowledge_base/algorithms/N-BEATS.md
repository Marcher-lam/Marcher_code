# N-BEATS 学习文档

> 神经基扩展自回归时间序列预测模型，无需任何假设的端到端预测

---

## 1. 算法基础认知

**一句话定义**：N-BEATS是Neural Basis Expansion Autoregressive Time Series的缩写，是一个无需任何时间序列假设的深度学习预测模型，通过神经网络自动学习数据中的趋势和周期模式。

**直觉类比**：N-BEATS就像一个"智能作曲家"——它不需要知道音乐是什么风格的（趋势、周期），自己就能从乐谱中学习。它把复杂的旋律分解成基础旋律（趋势部分）和装饰音（周期部分），然后分别处理。

**历史背景**：2020年，Oreshkin等人在论文"N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"中提出，在M4竞赛中取得领先成绩。

**算法定位**：
- 类型：深度学习 → 时间序列预测
- 输出：多步预测
- 模型类型：神经网络

**前置知识**：
- [必备]：时间序列基础
- [必备]：深度学习基础
- [扩展]：LSTM、GRU

---

## 2. 核心原理

### 2.1 核心思想

N-BEATS的核心思想是**双栈分解**：
- **趋势栈**：学习低频趋势
- **周期栈**：学习高频周期

每个栈包含多个blocks，每个block负责预测一部分。

### 2.2 架构

```
输入：历史序列
  ↓
[趋势栈] → 预测趋势
  ↓
[周期栈] → 预测周期
  ↓
输出：叠加预测
```

### 2.3 关键概念

- **Basis Expansion**：用基函数展开表示预测
- **Interpretable**：可解释的分解
- **Double Stack**：趋势+周期双栈

---

## 3. 数学公式

### 3.1 预测

$$\hat{y}_{t+1:t+H} = \sum_{b=1}^{B} \theta_b \cdot basis_b$$

其中H是预测步长，B是blocks数量。

### 3.2 趋势学习

使用多项式基：
$$basis_b(t) = [1, t, t^2, ..., t^{deg}]$$

---

## 4. 实现

```python
import torch
import torch.nn as nn


class NBEATSBlock(nn.Module):
    """N-BEATS Block"""
    
    def __init__(self, units, basis_dim, degree=3):
        super().__init__()
        self.fc1 = nn.Linear(units, units)
        self.fc2 = nn.Linear(units, basis_dim)
        self.fc3 = nn.Linear(basis_dim, units)
        
        # 基函数
        self.register_buffer('basis', self._compute_polynomial_basis(basis_dim, degree))
    
    def _compute_polynomial_basis(self, dim, degree):
        t = torch.arange(degree + 1).float()
        return t
    
    def forward(self, x, backcast):
        x = F.relu(self.fc1(x))
        theta = self.fc2(x)
        
        # 基函数展开
        backcast_pred = torch.matmul(theta, self.basis)
        
        forecast = self.fc3(theta)
        
        return backcast_pred, forecast


class NBEATS(nn.Module):
    """N-BEATS模型"""
    
    def __init__(self, input_dim=1, output_dim=1, hidden=128, n_blocks=3):
        super().__init__()
        
        self.fc_in = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([NBEATSBlock(hidden, 32) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = self.fc_in(x)
        backcast = x
        
        for block in self.blocks:
            bc_pred, fc_pred = block(x, backcast)
            x = x - bc_pred
        
        return self.fc_out(x)
```

---

## 5. 应用

### 5.1 适用场景

- 时间序列预测
- M4/M5竞赛
- 业务预测

### 5.2 优点

- 无假设：适用于任意数据
- 可解释：分解趋势和周期
- 高性能：M4竞赛最佳

---

## 6. 练习

**问题**：N-BEATS和ARIMA的主要区别？

答案：ARIMA需要假设数据是平稳的，N-BEATS不需要任何假设。

---

## 7. 学习路径

### 7.1 前置

- [ ] 时间序列

### 7.2 进阶

- [ ] Transformer时序
- [ ] LSTM

---

## 附录

### A. 代码

见第4节。

### B. 参考文献

1. Oreshkin et al., "N-BEATS", 2020

---

**文档结束**