# GRU（Gated Recurrent Unit）学习文档

## 1. 算法基础认知

GRU 是一种门控循环神经网络，用于序列数据建模。在广告系统中，GRU 被用于 DIEN 的兴趣演化建模和重排中的上下文依赖建模。

## 2. 核心原理

### GRU 门控结构

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$
$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 在 DIEN 中的应用

- 兴趣抽取层：使用 GRU 提取用户兴趣序列
- 兴趣演化层：AUGRU 捕捉与目标广告相关的兴趣演化过程

### 在重排中的应用

DLCM（Deep List Context Model）使用 GRU 显式建模上下文依赖——当前位置的最优选择取决于前面已展示的广告。

## 3. 与其他方法对比

| 特性 | PRM（Self-Attention） | DLCM（GRU） |
|------|----------------------|------------|
| 上下文建模 | 隐式 | 显式 |
| 位置依赖 | 弱 | 强 |
| 推理方式 | 并行 | 顺序 |
| 计算复杂度 | O(n²) | O(n)但串行 |

## 4. 代码实现

```python
import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, h0=None):
        output, hn = self.gru(x, h0)
        return output, hn
```

## 5. 学习总结

GRU 在广告系统中用于序列建模，特别是 DIEN 的兴趣演化和重排的上下文建模。相比 LSTM，GRU 参数更少，训练更快。
