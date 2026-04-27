# 长短期记忆网络 (LSTM) 学习文档

> 通过门控机制解决 RNN 的梯度消失问题，有效捕捉长距离依赖。

> 来源线索：本节内容根据原书中关于"LSTM"的相关章节（第2章2.5.4节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** LSTM 通过遗忘门、输入门和输出门三个门控结构，精确控制信息的遗忘、记忆和输出，有效解决了 RNN 的长距离依赖问题。

**直觉类比：** RNN 像一个"健忘的人"——新信息不断覆盖旧记忆。LSTM 则像一个"有条理的学生"——用三个笔记本管理知识：遗忘本（决定丢弃哪些旧知识）、更新本（决定记录哪些新知识）、输出本（决定现在使用哪些知识）。这种精细的信息管理使 LSTM 能记住几百步之前的重要信息。

**历史背景：** LSTM 由 Hochreiter 和 Schmidhuber 于 1997 年提出，专门解决 RNN 的梯度消失问题。经过多年改进（加入遗忘门、peephole 连接等），LSTM 成为序列建模的主流架构，直到 Transformer 出现。

**算法定位：** 序列建模、RNN 改进架构、长距离依赖建模。

**前置知识：** RNN、梯度消失/爆炸、sigmoid/tanh 激活函数、PyTorch。

---

## 2. 核心原理

### 核心思想

LSTM 的核心创新是**细胞状态（Cell State）**$C_t$——一条贯穿整个链的信息"传送带"。信息在这条传送带上流动时只有少量的线性交互，使梯度可以顺畅回传。三个门（gate）精确控制信息的流动：

1. **遗忘门（Forget Gate）**$f_t$：决定从细胞状态中丢弃什么信息
2. **输入门（Input Gate）**$i_t$：决定什么新信息写入细胞状态
3. **输出门（Output Gate）**$o_t$：决定基于细胞状态输出什么

### 工作流程

1. 遗忘门：$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
2. 候选状态：$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
3. 输入门：$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
4. 更新细胞状态：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
5. 输出门：$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
6. 更新隐状态：$h_t = o_t \odot \tanh(C_t)$

### 关键概念

- **细胞状态（Cell State）**：长期记忆，线性传递，不易丢失信息
- **门（Gate）**：sigmoid 输出 0~1，0 表示完全阻挡，1 表示完全通过
- **点积操作 $\odot$**：逐元素相乘，实现精细的信息筛选

---

## 3. 数学公式与推导

### 遗忘门

$$f_t = \sigma(W_{fh} h_{t-1} + W_{fx} x_t + b_f)$$

决定 $C_{t-1}$ 中每个维度保留多少（1=全保留，0=全丢弃）。

### 输入门与候选状态

$$i_t = \sigma(W_{ih} h_{t-1} + W_{ix} x_t + b_i)$$

$$\tilde{C}_t = \tanh(W_{Ch} h_{t-1} + W_{Cx} x_t + b_C)$$

输入门控制哪些新信息写入，候选状态提供可能写入的新内容。

### 细胞状态更新

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

旧记忆（经遗忘门筛选）+ 新信息（经输入门筛选）。

### 输出门

$$o_t = \sigma(W_{oh} h_{t-1} + W_{ox} x_t + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

### 为什么 LSTM 缓解梯度消失？

关键在细胞状态的更新是**加法**操作：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

反向传播时：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

只要遗忘门 $f_t$ 接近 1（不遗忘），梯度就能无衰减地向前传递。而 RNN 的梯度传递要经过 $\tanh$ 导数（最大值 1），必然衰减。

### 参数量

$$\text{参数量} = 4 \times [(d + h + 1) \times h] = 4h(d + h + 1)$$

其中 $d$ 是输入维度，$h$ 是隐状态维度。4 倍是因为有 4 组权重（遗忘门、输入门、候选状态、输出门）。

---

## 4. 训练过程讲解

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| hidden_size | 64 ~ 512 | 256 |
| num_layers | 1 ~ 4 | 2 |
| lr | 1e-4 ~ 1e-2 | 1e-3 |
| dropout | 0.1 ~ 0.5 | 0.2 |
| gradient_clip | 1.0 ~ 5.0 | 5.0 |

### 训练技巧
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`
- **双向 LSTM**：前向+后向两个 LSTM 拼接，适合需要完整上下文的任务
- **多层 LSTM**：堆叠多层增加表达能力

---

## 5. 应用场景

1. **机器翻译**：编码源语言，解码目标语言（Seq2Seq）
2. **文本生成**：基于前文生成后续文本
3. **语音识别**：将音频序列转换为文字
4. **时间序列预测**：股价、气象、流量等预测
5. **视频分析**：处理视频帧序列

---

## 6. 优缺点分析

### 优点
1. **长距离依赖**：能有效记忆数百步前的信息
2. **训练稳定**：门控机制缓解了梯度消失
3. **广泛验证**：在多种序列任务上表现优异

### 缺点
1. **计算复杂**：每个时间步有 4 组矩阵运算，比 RNN 慢约 4 倍
2. **串行瓶颈**：仍需逐步处理，无法充分并行
3. **参数量大**：参数量是同规模 RNN 的 4 倍

### 与同类对比

| 特性 | Vanilla RNN | LSTM | GRU | Transformer |
|------|------------|------|-----|-------------|
| 长距离依赖 | 差 | 好 | 好 | 极好 |
| 参数量 | 1x | 4x | 3x | 更大 |
| 训练速度 | 快 | 中 | 中 | 并行快 |
| 门控数量 | 0 | 3 | 2 | N/A |

---

## 7. 调库实现

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_size=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向需乘2

    def forward(self, x):
        embeds = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embeds)
        # 双向 LSTM：拼接最后一步的前向和后向隐状态
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)

# 测试
model = LSTMClassifier()
x = torch.randint(0, 10000, (32, 100))
logits = model(x)
print(f"输入: {x.shape}, 输出: {logits.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 8. 手工代码实现

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """手工实现 LSTM 单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 将四个门的权重合并为一个大矩阵，效率更高
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state  # h: (batch, hidden), c: (batch, hidden)
        gates = self.W_ih(x) + self.W_hh(h)  # (batch, 4*hidden)
        # 分割为四个门
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # 输入门
        f = torch.sigmoid(f)   # 遗忘门
        g = torch.tanh(g)      # 候选状态
        o = torch.sigmoid(o)   # 输出门
        # 更新
        c_new = f * c + i * g           # 细胞状态
        h_new = o * torch.tanh(c_new)   # 隐状态
        return h_new, c_new

class ManualLSTM(nn.Module):
    """手工实现多层 LSTM"""
    def __init__(self, input_size=32, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        device = x.device
        # 初始化所有层的隐状态和细胞状态
        h = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, (h[l], c[l]))
                inp = h[l]
            outputs.append(h[-1])
        return torch.stack(outputs, dim=1), (h[-1].unsqueeze(0), c[-1].unsqueeze(0))

# 测试
model = ManualLSTM(input_size=32, hidden_size=64, num_layers=2)
x = torch.randn(4, 20, 32)
output, (h_n, c_n) = model(x)
print(f"输出: {output.shape}, h_n: {h_n.shape}, c_n: {c_n.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 9-14. 可视化/评估/问题/总结/练习/路径

### 可视化
LSTM 的门控值可视化：遗忘门 $f_t$ 接近 1 表示保留旧记忆，接近 0 表示丢弃。在语言模型中，遇到句号等标点时遗忘门值通常会降低（"遗忘"之前的句子信息）。

### 评估指标
- 语言模型：困惑度（PPL）
- 分类：准确率、F1
- 序列标注：token 级准确率、实体级 F1

### 常见问题
1. **遗忘门偏向 0**：模型过度遗忘，学不到长期依赖 → 被称为"初始化偏置"问题，可将遗忘门偏置初始化为 1
2. **双向 LSTM 不能用于生成**：需要未来信息，只能用于编码

### 练习题

**题1：** LSTM 如何保证梯度不消失？

**参考答案：** 细胞状态 $C_t$ 的更新是加法操作 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$。反向传播时，$\partial C_t / \partial C_{t-1} = f_t$。只要遗忘门 $f_t$ 学到接近 1 的值，梯度就能无衰减传递。这是 LSTM 比 RNN 能处理更长序列的关键。

**题2：** 为什么 LSTM 用 sigmoid 作为门的激活函数而非 ReLU？

**参考答案：** sigmoid 输出范围为 (0, 1)，天然适合表示"通过比例"（0=完全阻挡，1=完全通过）。ReLU 输出范围是 $[0, +\infty)$，无法表示"部分通过"的概念，也不适合做门控。

### 学习路径
- 前置：RNN、梯度消失/爆炸
- 平行：GRU（LSTM 的简化版）
- 进阶：双向 LSTM、注意力机制 + LSTM（Seq2Seq with Attention）
- 推荐：Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
