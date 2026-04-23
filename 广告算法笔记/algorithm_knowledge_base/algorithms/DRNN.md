# DRNN 学习文档

## 1. 算法基础认知

深度循环神经网络（Deep Recurrent Neural Network, DRNN）是指通过在时间维度（水平方向）和层数维度（垂直方向）同时扩展 RNN 而得到的深层架构。单层 RNN 的表达能力有限，DRNN 通过堆叠多个 RNN 层来增强模型的表示能力，类似于前馈网络中增加隐藏层的效果。

DRNN 主要有两种扩展方式：**堆叠 RNN（Stacked RNN）** 和**双向 RNN（Bidirectional RNN）**，两者可组合使用。堆叠 RNN 让低层捕获局部模式、高层捕获抽象语义；双向 RNN 则让模型同时看到过去和未来的上下文信息。

## 2. 核心原理

**堆叠 RNN（Stacked RNN）：**

将多个 RNN 层纵向堆叠。第一层 RNN 接收原始输入序列，产生隐藏状态序列作为第二层的输入，以此类推。第 $l$ 层在时刻 $t$ 的隐藏状态为：

$$h_t^{(l)} = f(W^{(l)} h_t^{(l-1)} + U^{(l)} h_{t-1}^{(l)} + b^{(l)})$$

其中 $h_t^{(l-1)}$ 是下一层在时刻 $t$ 的输出（跨层连接），$h_{t-1}^{(l)}$ 是同层上一时刻的隐藏状态（时间连接）。

**双向 RNN（Bidirectional RNN）：**

对同一输入序列分别运行前向 RNN 和后向 RNN，在每个时刻拼接两个方向的隐藏状态：

$$\overrightarrow{h}_t = f(\overrightarrow{W} x_t + \overrightarrow{U} \overrightarrow{h}_{t-1} + \overrightarrow{b})$$

$$\overleftarrow{h}_t = f(\overleftarrow{W} x_t + \overleftarrow{U} \overleftarrow{h}_{t+1} + \overleftarrow{b})$$

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

## 3. 数学公式与推导

**堆叠双向 RNN（综合形式）：**

第 $l$ 层的前向隐藏状态：

$$\overrightarrow{h}_t^{(l)} = \sigma\left(\overrightarrow{W}^{(l)} \overrightarrow{h}_t^{(l-1)} + \overrightarrow{U}^{(l)} \overrightarrow{h}_{t-1}^{(l)} + \overrightarrow{b}^{(l)}\right)$$

第 $l$ 层的后向隐藏状态：

$$\overleftarrow{h}_t^{(l)} = \sigma\left(\overleftarrow{W}^{(l)} \overleftarrow{h}_t^{(l-1)} + \overleftarrow{U}^{(l)} \overleftarrow{h}_{t+1}^{(l)} + \overleftarrow{b}^{(l)}\right)$$

层输出：$h_t^{(l)} = [\overrightarrow{h}_t^{(l)}; \overleftarrow{h}_t^{(l)}]$

最终输出：$y_t = \text{softmax}(V \cdot h_t^{(L)} + c)$

**参数量分析：**

- 单层单向 RNN：$W \in \mathbb{R}^{d_h \times d_x}$，$U \in \mathbb{R}^{d_h \times d_h}$
- 堆叠 L 层双向 RNN：第 $l>1$ 层的输入维度为 $2d_h$（拼接双向），参数约 $2L \times (2d_h \cdot 2d_h + 2d_h \cdot 2d_h) = 8L d_h^2$

## 4. 训练过程讲解

1. **前向传播**：输入序列首先经过第一层 RNN（前向和后向分别计算），得到的双向隐藏状态拼接后传给第二层，逐层向上直至顶层。
2. **损失计算**：可对所有时刻的输出计算损失（序列标注），也可只对最后时刻计算损失（序列分类）。总损失为各时刻损失之和。
3. **反向传播（BPTT）**：沿两个维度反向传播梯度——时间维度（BPTT）和层数维度。双向 RNN 中后向链路的梯度从右向左流动。
4. **梯度问题**：深层堆叠增加了层数维度的梯度路径，可能加剧梯度消失。实践中常用 LSTM/GRU 替代原始 RNN 作为基本单元。

## 5. 应用场景

- 语音识别（声学模型通常使用 3-5 层堆叠双向 LSTM）
- 机器翻译（编码器使用堆叠双向 RNN）
- 命名实体识别（NER，双向捕获上下文）
- 情感分析（深层 RNN 提取复杂语义）
- 广告系统中的用户行为序列建模、query 意图理解
- 时间序列预测（多层捕获多尺度时间模式）

## 6. 优缺点分析

**优点：**
- 堆叠多层增强表达能力，高层可学习更抽象的特征
- 双向结构同时利用过去和未来上下文，提升序列理解能力
- 灵活可扩展，层数和方向可自由组合

**缺点：**
- 训练时间随层数线性增长
- 层数过多时梯度仍可能不稳定（虽然 LSTM 缓解了时间维度的梯度问题）
- 双向 RNN 不能用于实时/流式场景（需等待完整序列）
- 参数量较大，容易在小数据集上过拟合

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class StackedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

vocab_size, embed_dim, hidden_dim, num_layers, num_classes = 5000, 128, 256, 3, 4
model = StackedBiLSTM(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

batch_x = torch.randint(0, vocab_size, (32, 50))
batch_y = torch.randint(0, num_classes, (32,))

for epoch in range(5):
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleDRNN:
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.W = [np.random.randn(hidden_dim, input_dim if l == 0 else hidden_dim) * 0.01 for l in range(num_layers)]
        self.U = [np.random.randn(hidden_dim, hidden_dim) * 0.01 for _ in range(num_layers)]
        self.b = [np.zeros(hidden_dim) for _ in range(num_layers)]

    def forward(self, x_seq):
        T = len(x_seq)
        H = [[np.zeros(self.hidden_dim) for _ in range(T + 1)] for _ in range(self.num_layers)]
        for t in range(T):
            for l in range(self.num_layers):
                inp = x_seq[t] if l == 0 else H[l-1][t]
                H[l][t] = np.tanh(self.W[l] @ inp + self.U[l] @ H[l][t-1] + self.b[l])
        return [[H[l][t] for t in range(T)] for l in range(self.num_layers)]

d_in, d_h = 10, 16
drnn = SimpleDRNN(d_in, d_h, num_layers=3)
x_seq = [np.random.randn(d_in) for _ in range(8)]
layer_outputs = drnn.forward(x_seq)
print(f"Top layer output shape per step: {layer_outputs[-1][0].shape}")
```

## 9. 可视化与结果理解

- **隐藏状态热力图**：将各层各时刻的隐藏状态绘制为热力图，观察低层关注局部特征、高层关注全局语义的层次化表示
- **双向隐藏状态对比**：分别可视化前向和后向隐藏状态，前向状态从左到右累积信息，后向状态从右到左累积
- **层数影响曲线**：绘制层数从 1 到 6 时验证集准确率的变化，通常 2-3 层效果最佳，层数过多可能过拟合
- **梯度范数随层数变化**：观察梯度在层数维度的衰减情况

## 10. 模型评估

- **序列标注任务**：使用 token-level 准确率、F1 值（如 NER 中的实体级 F1）
- **分类任务**：准确率、F1-macro
- **困惑度（Perplexity）**：语言模型评估指标，$PPL = \exp(\frac{1}{N}\sum -\log p(w_t))$
- **推理速度**：堆叠层数直接影响推理延迟，需在精度和速度间权衡

## 11. 常见问题与易错点

- **层数选择**：2-3 层通常足够，层数过多收益递减且易过拟合，不要盲目加深
- **Dropout 位置**：应在 RNN 层之间（embedding 层后和层间）添加 Dropout，而非在时间步之间
- **双向 RNN 的实时性**：双向结构需要完整序列才能运行，不适合在线/流式推理
- **隐藏状态初始化**：多层 RNN 的每层都需要独立初始化隐藏状态，PyTorch 中 `h_0` 形状为 `(num_layers * num_directions, batch, hidden_dim)`
- **梯度裁剪**：深层 RNN 的 BPTT 更容易梯度爆炸，通常需要梯度裁剪（`torch.nn.utils.clip_grad_norm_`）

## 12. 学习总结

DRNN 通过在垂直方向堆叠多层和在时间方向双向展开，显著增强了 RNN 的表达能力。堆叠结构让网络具备层次化特征提取能力，双向结构让模型充分利用完整上下文。实践中几乎总是使用 LSTM/GRU 作为 DRNN 的基本单元而非原始 RNN，以缓解梯度问题。理解 DRNN 是掌握序列建模的关键一步。

## 13. 练习题与思考题（含答案）

**Q1：3 层堆叠 RNN，输入维度 100，隐藏维度 256，求总参数量（不含输出层）。**

A1：第一层：$256×100 + 256×256 + 256 = 91136$；第二层和第三层相同：$256×256 + 256×256 + 256 = 131328$。总计：$91136 + 2×131328 = 353792$。

**Q2：双向 RNN 为什么不能用于实时语音识别？**

A2：后向 RNN 需要从序列末尾开始计算，即需要未来时刻的输入，因此必须等待完整序列才能输出。实时场景只能使用单向 RNN。

**Q3：为什么 DRNN 通常使用 LSTM 而非原始 RNN 作为基本单元？**

A3：原始 RNN 存在严重的梯度消失问题，堆叠多层后在层数维度和时间维度都会加剧该问题。LSTM 的门控机制有效缓解了梯度消失，使深层堆叠成为可能。

## 14. 学习路径建议

1. 复习基础 RNN 和 LSTM 的原理
2. 实现单层双向 LSTM，观察前后向隐藏状态的区别
3. 堆叠 2-3 层 LSTM，在文本分类任务上与单层对比
4. 理解 BPTT 在多层结构中的梯度流动
5. 阅读 "Bidirectional LSTM CRF" 论文（序列标注经典架构）
6. 进阶：了解 Pyramid RNN、Tree-LSTM 等变体，以及 Transformer 如何替代 DRNN
