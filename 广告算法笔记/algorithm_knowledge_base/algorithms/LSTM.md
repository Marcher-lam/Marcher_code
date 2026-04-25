# LSTM 学习文档

## 1. 算法基础认知

LSTM（Long Short-Term Memory）是 RNN 的改进版本，通过遗忘门、输入门和输出门三个门控机制控制信息流动，有效解决了标准 RNN 的梯度消失问题。在广告系统中，LSTM 可用于用户行为序列建模和文本特征编码，是理解 GRU 和 Transformer 的重要中间步骤。

## 2. 核心原理

LSTM 的核心创新是引入细胞状态 $C_t$——一条贯穿时间轴的"信息高速公路"。细胞状态通过加法而非乘法更新，梯度可以几乎无损地流动。三个门控分别决定：遗忘多少旧信息、接收多少新信息、输出哪些信息。

## 3. 数学公式与推导

**遗忘门**——决定丢弃哪些旧信息：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**输入门**——决定接收哪些新信息：

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**候选细胞状态**：

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**细胞状态更新**（核心：加法操作避免梯度消失）：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**输出门**：

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**隐藏状态**：

$$
h_t = o_t \odot \tanh(C_t)
$$

LSTM 缓解梯度消失的关键在于细胞状态的更新方式。对 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ 求梯度，$\frac{\partial C_t}{\partial C_{t-1}} = f_t$。与标准 RNN 中 $\frac{\partial h_t}{\partial h_{t-1}}$ 包含 $\tanh'$ (始终 <1) 不同，$f_t$ 是一个可学习的 sigmoid 门控，训练过程中可以趋近于 1，从而让梯度近乎无损地沿 $C_{t-1} \to C_t$ 路径传播。加法操作避免了标准 RNN 中反复乘以小于 1 的导数导致的连乘消失。

## 4. 训练过程讲解

1. 初始化 $h_0 = 0, C_0 = 0$
2. 输入序列 $(x_1, ..., x_T)$，逐步计算 $f_t, i_t, \tilde{C}_t, C_t, o_t, h_t$
3. 最终取 $h_T$ 或所有 $h_t$ 作为序列表示
4. 计算 Loss，BPTT 反向传播
5. 梯度通过细胞状态的加法路径流动，缓解消失问题
6. 参数更新：所有门的权重矩阵 $W_f, W_i, W_C, W_o$

## 5. 应用场景

- 广告用户行为序列建模（长行为兴趣捕捉）
- 搜索广告 query 理解与文本编码
- 时间序列预测（流量/库存预估）
- NLP 任务（情感分析、文本分类）

## 6. 优缺点分析

**优点：**
- 有效解决长程依赖问题
- 细胞状态提供梯度直通路径
- 门控机制灵活控制信息流

**缺点：**
- 参数量是 RNN 的 4 倍（四个门/候选）
- 仍需顺序计算，无法并行
- 训练速度较慢
- 在超长序列上仍弱于 Transformer

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel(input_dim=64, hidden_dim=128, output_dim=1)
x = torch.randn(32, 10, 64)
pred = model(x)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn.functional as F

class CustomLSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.W_f = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.W_i = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.W_C = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.W_o = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.b_f = torch.ones(hidden_dim)
        self.b_i = torch.zeros(hidden_dim)
        self.b_C = torch.zeros(hidden_dim)
        self.b_o = torch.zeros(hidden_dim)

    def forward(self, x_t, h_prev, C_prev):
        combined = torch.cat([h_prev, x_t])
        f_t = torch.sigmoid(self.W_f @ combined + self.b_f)
        i_t = torch.sigmoid(self.W_i @ combined + self.b_i)
        C_bar = torch.tanh(self.W_C @ combined + self.b_C)
        C_t = f_t * C_prev + i_t * C_bar
        o_t = torch.sigmoid(self.W_o @ combined + self.b_o)
        h_t = o_t * torch.tanh(C_t)
        return h_t, C_t
```

## 9. 可视化与结果理解

- 绘制各门的激活值热力图，观察遗忘门如何选择性遗忘
- 绘制细胞状态 $C_t$ 随时间变化曲线，验证长程信息保持能力
- 对比 RNN 和 LSTM 的梯度范数随序列长度的变化
- 可视化 LSTM 隐藏状态对不同类型输入的注意力分布

## 10. 模型评估

- 序列分类：Accuracy / F1 / AUC
- 广告 CTR 预估：AUC + LogLoss
- 对比指标：相同序列长度下 LSTM vs RNN 的性能差异
- 关键验证：长序列（T>50）时 LSTM 是否保持性能

## 11. 常见问题与易错点

- **遗忘门偏置初始化**：b_f 初始化为 1（而非 0），鼓励初期保留信息
- **peephole connections**：部分变体让门控直接观察细胞状态，标准 LSTM 不包含
- **双向 LSTM**：拼接前向和后向输出，适用于需要完整上下文的任务
- **多层堆叠**：第一层输出作为第二层输入，层间可加 Dropout
- **hidden vs output**：`nn.LSTM` 返回的 output 包含所有时间步，h_n 仅包含最后一步

## 12. 学习总结

LSTM 通过门控机制和细胞状态的加法更新，从根本上解决了 RNN 的梯度消失问题。在广告系统中，LSTM 可作为用户行为序列建模的 baseline，但现代系统更多使用 GRU（DIEN 中参数更高效）或 Transformer（BST 中支持并行计算）。

## 13. 练习题与思考题（含答案）

**Q1：LSTM 为什么能解决梯度消失？**
A1：细胞状态 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ 通过加法更新，梯度可通过加法路径近似无损传播。

**Q2：遗忘门偏置为什么初始化为 1？**
A2：初始化为正值使 sigmoid 输出接近 1，即初始时倾向于保留旧信息，避免训练初期遗忘过多。

**Q3：LSTM 参数量是 RNN 的几倍？**
A3：约 4 倍。LSTM 有遗忘门、输入门、候选状态、输出门四组权重，RNN 只有一组。

**Q4：LSTM 和 GRU 的核心区别？**
A4：LSTM 有独立的遗忘门和输入门，以及显式细胞状态；GRU 将两者合并为更新门，无独立细胞状态，参数更少。

## 14. 学习路径建议

1. 确保已理解 RNN 和 BPTT
2. 重点理解细胞状态的加法更新为什么能缓解梯度消失
3. 动手实现单个 LSTM Cell 的前向传播
4. 学习 GRU（简化的门控机制）
5. 进阶 Attention 机制和 Transformer
