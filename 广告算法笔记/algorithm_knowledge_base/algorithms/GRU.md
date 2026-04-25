# GRU 学习文档

## 1. 算法基础认知

GRU（Gated Recurrent Unit）是 LSTM 的简化版本，将遗忘门和输入门合并为更新门，参数更少但性能相当。在广告系统中，GRU 是 DIEN（兴趣演化建模）和 DLCM（重排上下文建模）的核心组件。

## 2. 核心原理

GRU 通过两个门（重置门和更新门）控制信息流。重置门决定忽略多少过去信息来计算候选状态，更新门决定在新旧隐藏状态之间做多少混合。更新公式 $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ 是线性插值，梯度通过 $(1-z_t)$ 路径直接流动。

## 3. 数学公式与推导

**重置门**——控制计算候选状态时忽略多少过去：

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

**更新门**——控制新旧状态的混合比例：

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**候选隐藏状态**：

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
$$

**最终隐藏状态**：

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

当 $z_t \approx 0$ 时，$h_t \approx h_{t-1}$，信息直接传递；当 $z_t \approx 1$ 时，$h_t \approx \tilde{h}_t$，完全更新。这种线性插值使梯度通过 $(1-z_t)$ 路径保持流动。

## 4. 训练过程讲解

1. 初始化 $h_0 = 0$
2. 依次计算 $r_t, z_t, \tilde{h}_t, h_t$
3. 最终取 $h_T$ 或全部 $h_t$ 用于下游任务
4. BPTT 反向传播，梯度通过更新门的线性路径流动
5. 梯度裁剪（阈值 1~5）
6. 参数更新：$W_r, W_z, W_h$

## 5. 应用场景

- **DIEN 兴趣演化层**：AUGRU 用注意力调制更新门，捕捉与目标广告相关的兴趣演化
- **DLCM 重排**：GRU 显式建模已展示广告序列的上下文依赖
- 用户行为序列建模
- 文本分类与情感分析

## 6. 优缺点分析

**优点：**
- 参数量约为 LSTM 的 2/3，训练更快
- 性能与 LSTM 相当
- 线性更新路径缓解梯度消失

**缺点：**
- 仍需顺序计算，无法并行
- 超长序列不如 Transformer
- 缺少独立细胞状态，部分场景灵活性不如 LSTM

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        return self.fc(gru_out[:, -1, :])

model = GRUModel(input_dim=64, hidden_dim=128, output_dim=1)
x = torch.randn(32, 10, 64)
pred = model(x)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch

class CustomGRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.W_z = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.W_r = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.W_h = torch.randn(hidden_dim, input_dim + hidden_dim) * 0.01

    def forward(self, x_t, h_prev):
        combined = torch.cat([h_prev, x_t])
        z_t = torch.sigmoid(self.W_z @ combined)
        r_t = torch.sigmoid(self.W_r @ combined)
        h_bar = torch.tanh(self.W_h @ torch.cat([r_t * h_prev, x_t]))
        h_t = (1 - z_t) * h_prev + z_t * h_bar
        return h_t

cell = CustomGRUCell(input_dim=64, hidden_dim=128)
h = torch.zeros(128)
for x_t in [torch.randn(64) for _ in range(10)]:
    h = cell.forward(x_t, h)
```

## 9. 可视化与结果理解

- 绘制更新门 $z_t$ 随时间步变化，观察哪些时间步触发状态更新
- 重置门 $r_t$ 热力图：哪些历史信息被忽略
- 对比 LSTM vs GRU 在相同任务上的隐藏状态动态
- DIEN 中 AUGRU 的注意力权重可视化：兴趣演化轨迹

## 10. 模型评估

- 广告 CTR 预估：AUC + LogLoss
- DIEN 场景：对比无兴趣演化的 DIN baseline
- DLCM 场景：对比无上下文建模的 pointwise 排序
- 关键指标：序列长度对模型性能的影响

## 11. 常见问题与易错点

- **GRU vs LSTM 选择**：数据量小时选 GRU（参数少不易过拟合），需要灵活控制时选 LSTM
- **AUGRU 理解**：DIEN 中注意力不是加在输出上，而是调制更新门 $z_t$，即 $z_t' = a_t \cdot z_t$
- **DLCM 顺序性**：GRU 顺序处理已展示广告，推理时需逐步更新
- **初始化**：更新门偏置初始化为负值，使初始 $z_t$ 偏小，倾向保留旧信息

## 12. 学习总结

GRU 是 LSTM 的高效替代，通过重置门和更新门实现更简洁的门控机制。在广告系统中，GRU 是 DIEN 兴趣演化建模和 DLCM 重排上下文建模的核心。相比 LSTM 参数更少，相比 Transformer 更适合顺序依赖强的场景。

## 13. 练习题与思考题（含答案）

**Q1：GRU 的更新门 $z_t$ 如何缓解梯度消失？**
A1：$h_t = (1-z_t) h_{t-1} + z_t \tilde{h}_t$ 中 $(1-z_t)$ 提供梯度直通路径，类似 LSTM 的细胞状态加法路径。

**Q2：DIEN 中 AUGRU 和标准 GRU 的区别是什么？**
A2：AUGRU 用目标广告相关的注意力 $a_t$ 调制更新门：$z_t' = a_t \cdot z_t$，使兴趣演化聚焦于与目标相关的行为。

**Q3：DLCM 为什么用 GRU 而不用 Self-Attention？**
A3：重排中已展示广告存在严格的顺序依赖（先展示的影响后展示），GRU 显式建模位置顺序，Self-Attention 的位置编码是隐式的。

**Q4：GRU 参数量相比 LSTM 减少了多少？**
A4：LSTM 有 4 组权重矩阵，GRU 有 3 组，参数量约为 LSTM 的 3/4。

## 14. 学习路径建议

1. 先掌握 LSTM 的门控机制
2. 理解 GRU 如何通过 2 个门替代 LSTM 的 3 个门
3. 学习 DIEN 论文，理解 AUGRU 的注意力调制
4. 学习 DLCM，理解 GRU 在重排中的应用
5. 进阶 Transformer（BST / PRM）
