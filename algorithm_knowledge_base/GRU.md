# 门控循环单元 (GRU) 学习文档

> LSTM 的轻量化替代，用更少的门控实现相当的性能。

> 来源线索：本节内容根据原书中关于"GRU"的相关章节（第2章2.5.4节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** GRU 通过更新门和重置门两个门控机制，在保持 LSTM 长距离依赖建模能力的同时简化了结构，提高了计算效率。

**直觉类比：** LSTM 像一个有三个保险柜的办公室（遗忘、新增、输出各一个保险柜）。GRU 觉得太复杂了，合并成两个：一个"更新保险柜"（决定多少旧信息保留、多少新信息写入）和一个"重置保险柜"（决定计算新信息时参考多少旧记忆）。效果差不多，但管理更简洁。

**历史背景：** GRU 由 Cho 等人于 2014 年提出（论文 "Learning Phrase Representations using RNN Encoder-Decoder"），作为 LSTM 的简化替代方案。实验表明在多数任务上 GRU 与 LSTM 性能相当，但训练更快。

**算法定位：** 序列建模、RNN 改进架构、LSTM 的轻量替代。

**前置知识：** RNN、LSTM、sigmoid/tanh、PyTorch。

---

## 2. 核心原理

### 核心思想

GRU 对 LSTM 做了两大简化：
1. **合并门控**：将 LSTM 的遗忘门和输入门合并为一个**更新门** $z_t$
2. **合并状态**：将 LSTM 的细胞状态 $C_t$ 和隐状态 $h_t$ 合并为单一状态 $h_t$

### 工作流程

1. **重置门**：$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$ — 控制计算候选状态时参考多少旧记忆
2. **更新门**：$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$ — 控制保留多少旧状态 vs 写入新状态
3. **候选状态**：$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b)$ — 重置门决定是否忽略旧记忆
4. **状态更新**：$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ — 更新门在旧状态和新候选间插值

### 关键概念

- **更新门（Update Gate）**$z_t$：值接近 1 时使用新候选状态，接近 0 时保留旧状态
- **重置门（Reset Gate）**$r_t$：值接近 0 时忽略旧记忆（"重新开始"），接近 1 时利用旧记忆
- **线性跳跃连接**：$(1-z_t) \odot h_{t-1}$ 项使梯度可以直接回传，类似 ResNet 的残差连接

---

## 3. 数学公式与推导

### 重置门

$$r_t = \sigma(W_{rh} h_{t-1} + W_{rx} x_t + b_r)$$

重置门决定计算候选隐状态时是否使用之前的隐状态。如果 $r_t$ 接近 0，则候选状态只依赖当前输入（"遗忘过去"）。

### 更新门

$$z_t = \sigma(W_{zh} h_{t-1} + W_{zx} x_t + b_z)$$

更新门决定了新旧信息的混合比例。

### 候选隐状态

$$\tilde{h}_t = \tanh(W_h (r_t \odot h_{t-1}) + W_x x_t + b)$$

注意重置门 $r_t$ 的作用：它"门控"了旧隐状态，决定在计算新候选时参考多少旧信息。

### 最终隐状态

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

这是 GRU 的核心——在旧状态 $h_{t-1}$ 和新候选 $\tilde{h}_t$ 之间做加权插值。

### 与 LSTM 的对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门控数量 | 3（遗忘、输入、输出） | 2（重置、更新） |
| 状态数量 | 2（细胞状态 + 隐状态） | 1（隐状态） |
| 参数量 | $4h(d+h+1)$ | $3h(d+h+1)$ |
| 更新方式 | 加法更新细胞状态 | 插值更新隐状态 |

### 参数量

$$\text{参数量} = 3 \times [(d + h + 1) \times h] = 3h(d + h + 1)$$

约为 LSTM 的 75%。

---

## 4. 训练过程讲解

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| hidden_size | 64 ~ 512 | 256 |
| num_layers | 1 ~ 4 | 2 |
| lr | 1e-4 ~ 1e-2 | 1e-3 |
| dropout | 0.1 ~ 0.5 | 0.2 |

### 训练技巧
- **选择 LSTM 还是 GRU**：数据量少时 GRU 可能更好（参数少，不易过拟合）；需要精细控制时 LSTM 更有优势
- **与 LSTM 混用**：编码器用 GRU（快速），解码器用 LSTM（精细）

---

## 5. 应用场景

与 LSTM 基本相同：
1. **机器翻译**：Google 早期的神经机器翻译使用 GRU
2. **文本分类**：情感分析、主题分类
3. **时间序列预测**：比 LSTM 训练更快
4. **语音识别**：端到端语音识别系统

---

## 6. 优缺点分析

### 优点
1. **结构简洁**：只有 2 个门，比 LSTM 少 1/4 参数
2. **训练更快**：计算量小，收敛速度通常比 LSTM 快
3. **性能相当**：在多数任务上与 LSTM 表现接近

### 缺点
1. **灵活性低**：缺少独立的细胞状态，无法像 LSTM 那样精确控制信息流
2. **某些任务不如 LSTM**：在需要精细门控的复杂序列任务上 LSTM 可能更好

---

## 7. 调库实现

```python
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_size=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=2,
                          batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        output, h_n = self.gru(embeds)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)

# 测试
model = GRUClassifier()
x = torch.randint(0, 10000, (32, 100))
logits = model(x)
print(f"输入: {x.shape}, 输出: {logits.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# 对比 LSTM 和 GRU 的参数量
lstm = nn.LSTM(128, 256, num_layers=2, bidirectional=True)
gru = nn.GRU(128, 256, num_layers=2, bidirectional=True)
print(f"LSTM 参数量: {sum(p.numel() for p in lstm.parameters()):,}")
print(f"GRU 参数量: {sum(p.numel() for p in gru.parameters()):,}")
```

---

## 8. 手工代码实现

```python
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """手工实现 GRU 单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 三个门：重置门(r)、更新门(z)、候选状态(n)
        self.W_ih = nn.Linear(input_size, 3 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, h):
        # x: (batch, input_size), h: (batch, hidden_size)
        gates_ih = self.W_ih(x)
        gates_hh = self.W_hh(h)
        r_i, z_i, n_i = gates_ih.chunk(3, dim=1)
        r_h, z_h, n_h = gates_hh.chunk(3, dim=1)

        r = torch.sigmoid(r_i + r_h)   # 重置门
        z = torch.sigmoid(z_i + z_h)   # 更新门
        n = torch.tanh(n_i + r * n_h)  # 候选状态（注意重置门的作用）

        h_new = (1 - z) * h + z * n    # 插值更新
        return h_new

class ManualGRU(nn.Module):
    """手工实现多层 GRU"""
    def __init__(self, input_size=32, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(GRUCell(in_size, hidden_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l])
                inp = h[l]
            outputs.append(h[-1])
        return torch.stack(outputs, dim=1), h[-1].unsqueeze(0)

# 测试
model = ManualGRU(input_size=32, hidden_size=64, num_layers=2)
x = torch.randn(4, 20, 32)
output, h_n = model(x)
print(f"输出: {output.shape}, h_n: {h_n.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 9-14. 可视化/评估/问题/总结/练习/路径

### 可视化
GRU 的更新门 $z_t$ 可视化：当处理到句子的关键位置（如转折词"但是"）时，更新门值会显著变化，表示需要更新隐状态。

### 评估指标
同 LSTM：困惑度、准确率、F1 等。

### 常见问题
1. **GRU vs LSTM 选择**：无定论，建议两者都尝试。数据量小、序列短时偏好 GRU
2. **重置门全为 1**：退化为普通 RNN → 检查初始化和学习率

### 练习题

**题1：** GRU 的更新门 $z_t$ 与 LSTM 的遗忘门 $f_t$ 有什么关系？

**参考答案：** GRU 的更新门 $z_t$ 实际上融合了 LSTM 遗忘门和输入门的功能。LSTM 中 $f_t$ 控制旧信息保留，$i_t$ 控制新信息写入，两者独立。GRU 中 $z_t$ 同时控制两者：$(1-z_t)$ 对应遗忘门（保留旧信息），$z_t$ 对应输入门（写入新信息）。由于 $(1-z_t) + z_t = 1$，保留旧信息和写入新信息的比例互补。

**题2（开放）：** GRU 的插值更新 $h_t = (1-z_t) h_{t-1} + z_t \tilde{h}_t$ 与 ResNet 的残差连接有何相似之处？

**参考答案思路：** 两者都提供了"捷径"让信息/梯度直接通过。当 $z_t = 0$ 时，$h_t = h_{t-1}$，信息直接传递，类似 ResNet 的恒等映射。这种设计缓解了梯度消失问题，是深度网络训练的关键技巧。

### 学习路径
- 前置：RNN、LSTM
- 平行：SRU（Simple Recurrent Unit）、QRNN（准 RNN）
- 进阶：双向 GRU、注意力 + GRU、Transformer
- 推荐：Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
