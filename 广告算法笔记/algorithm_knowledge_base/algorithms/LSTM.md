# LSTM 学习文档

## 1. 算法基础认知

LSTM（Long Short-Term Memory）是一种门控循环神经网络，通过遗忘门、输入门和输出门控制信息的流动，解决标准 RNN 的梯度消失问题。

## 2. 核心原理

### LSTM 门控结构

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(遗忘门)}
$$
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(输入门)}
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(输出门)}
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

## 3. 应用场景

- 序列行为建模
- 时间序列预测
- 自然语言处理

## 4. 代码实现

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

## 5. 学习总结

LSTM 是序列建模的经典方法，在广告系统中可用作行为序列建模的 baseline。现代系统中更多使用 GRU（参数更少）或 Transformer（并行计算）。
