# RNN 学习文档

## 1. 算法基础认知

RNN（循环神经网络）是一类专门处理序列数据的神经网络。与全连接网络不同，RNN 通过隐藏状态在时间步之间传递信息，使模型具备"记忆"能力。在广告系统中，RNN 曾广泛用于用户行为序列建模，现多被 GRU/Transformer 替代，但仍是理解序列模型的基础。

## 2. 核心原理

RNN 的核心思想是：当前时刻的输出不仅依赖当前输入，还依赖历史信息。通过共享参数的循环结构，RNN 对任意长度序列进行建模。信息沿时间轴流动，每个时间步更新隐藏状态，实现动态上下文积累。

## 3. 数学公式与推导

隐藏状态更新：

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

输出计算：

$$
y_t = W_{hy} h_t + b_y
$$

损失函数（以序列预测为例）：

$$
L = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

BPTT 梯度沿时间展开：

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial h_t} \prod_{k=t}^{s+1} \frac{\partial h_k}{\partial h_{k-1}} \frac{\partial h_s}{\partial W}
$$

当序列较长时，$\prod \frac{\partial h_k}{\partial h_{k-1}}$ 中的 $\tanh'$ 导数连乘导致梯度消失或爆炸。

## 4. 训练过程讲解

1. 输入序列 $(x_1, x_2, ..., x_T)$，初始化 $h_0 = 0$
2. 前向传播：依次计算 $h_t$ 和 $y_t$
3. 计算损失 $L$
4. BPTT 反向传播：沿时间轴展开，计算各时间步梯度
5. 梯度裁剪（防止梯度爆炸）：$\|\nabla\| > \text{threshold}$ 时缩放
6. 更新参数 $W_{hh}, W_{xh}, W_{hy}$

## 5. 应用场景

- 广告用户行为序列建模（已被 Transformer 替代）
- 文本特征编码（搜索广告 query 理解）
- 时间序列预测（流量预估）
- 语言模型基础架构

## 6. 优缺点分析

**优点：**
- 天然处理变长序列
- 参数共享，模型紧凑
- 理论上能捕捉任意长度依赖

**缺点：**
- 梯度消失/爆炸问题严重
- 无法并行计算，训练速度慢
- 长程依赖建模能力弱
- 实际有效记忆长度有限

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = RNNModel(input_dim=64, hidden_dim=128, output_dim=1)
x = torch.randn(32, 10, 64)
pred = model(x)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn.functional as F

class CustomRNN:
    def __init__(self, input_dim, hidden_dim):
        scale = 0.01
        self.W_xh = torch.randn(hidden_dim, input_dim) * scale
        self.W_hh = torch.randn(hidden_dim, hidden_dim) * scale
        self.b_h = torch.zeros(hidden_dim)

    def forward(self, x_seq):
        h = torch.zeros(self.W_hh.shape[0])
        outputs = []
        for x_t in x_seq:
            h = torch.tanh(self.W_hh @ h + self.W_xh @ x_t + self.b_h)
            outputs.append(h.clone())
        return torch.stack(outputs), h

rnn = CustomRNN(input_dim=64, hidden_dim=128)
x_seq = [torch.randn(64) for _ in range(10)]
outputs, h_final = rnn.forward(x_seq)
```

## 9. 可视化与结果理解

- 绘制隐藏状态 $h_t$ 随时间步的变化曲线，观察信息衰减
- 使用梯度范数图验证 BPTT 中梯度消失现象
- 对比不同序列长度下模型的预测精度下降趋势
- 热力图展示隐藏状态各维度对不同输入的激活程度

## 10. 模型评估

- 序列预测任务：MSE / RMSE
- 分类任务：Accuracy / F1 / AUC
- 关键指标：对比不同序列长度（T=10, 50, 100）下的性能衰减程度
- 广告场景：AUC + LogLoss 联合评估

## 11. 常见问题与易错点

- **梯度爆炸**：必须使用梯度裁剪，阈值通常设为 1~5
- **隐藏状态初始化**：全零初始化是标准做法，非零初始化可能加速收敛
- **序列过长**：实际使用 truncated BPTT，截断长度通常 20~50
- **双向 RNN**：拼接前向和后向隐藏状态，但并非所有场景都适用
- **batch_first 参数**：PyTorch 中注意 `batch_first=True` 的输入形状为 (B, T, D)

## 12. 学习总结

RNN 是序列建模的基石，核心思想是通过循环连接在时间轴传递信息。其梯度消失问题直接催生了 LSTM 和 GRU。在现代广告系统中，RNN 本身已较少直接使用，但理解其原理对学习 GRU（DIEN 兴趣演化）和 Transformer（BST 序列建模）至关重要。

## 13. 练习题与思考题（含答案）

**Q1：为什么 RNN 会产生梯度消失？**
A1：BPTT 中梯度沿时间步连乘，$\tanh'$ 导数值域为 $(0, 0.25]$，长序列中连乘趋近于零。

**Q2：梯度裁剪的具体做法是什么？**
A2：当梯度范数 $\|\nabla\|$ 超过阈值 $\theta$ 时，缩放梯度为 $\frac{\theta}{\|\nabla\|} \cdot \nabla$。

**Q3：RNN 和全连接网络的核心区别是什么？**
A3：参数共享和循环连接。RNN 在所有时间步共享同一组权重，并通过隐藏状态传递历史信息。

**Q4：为什么广告系统中 RNN 被 Transformer 替代？**
A4：Transformer 支持并行训练、长程依赖更强、Self-Attention 能直接建模任意距离的依赖关系。

## 14. 学习路径建议

1. 先掌握 RNN 前向传播和 BPTT 原理
2. 理解梯度消失问题及其成因
3. 学习 LSTM（门控机制如何解决梯度消失）
4. 学习 GRU（DIEN 中的实际应用）
5. 进阶 Transformer（BST / OneTrans）
