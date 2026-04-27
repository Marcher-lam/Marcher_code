# TCN（时序卷积网络）学习文档

> 基于卷积神经网络处理序列数据的模型，利用因果卷积和膨胀卷积实现长时依赖建模

---

## 1. 算法基础认知

### 1.1 一句话定义

TCN（Temporal Convolutional Network，时序卷积网络）是一种使用一维卷积处理序列数据的神经网络，通过**因果卷积**确保时间顺序、**膨胀卷积**扩大感受野，兼具RNN的序列建模能力和CNN的高效并行计算。

### 1.2 直觉类比

TCN就像一个带"放大镜"的扫描仪！它从序列开头逐步往后扫描，每一层通过膨胀卷积能看到更远的历史信息，同时保证不回头看未来的数据。

想象你在看一部很长的电影：
- **普通卷积**：只能看到前后几秒钟的内容
- **TCN**：第1层看最近10秒，第2层看最近1分钟，第3层看最近10分钟...
- **原因**：膨胀因子是1, 2, 4, 8...指数级增长！

### 1.3 发展背景

- 2017年，Lea等人在论文" Temporal Convolutional Networks for Action Segmentation"中首次提出TCN
- 后续在时序预测、语音识别等任务中广泛使用
- TCN在保持RNN序列建模能力的同时，利用卷积的并行计算优势

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 深度学习 → 序列建模 |
| 输出 | 序列标签/预测 |
| 模型类型 | 卷积神经网络 |
| 核心创新 | 因果卷积+膨胀卷积 |

### 1.5 前置知识

- [必备]：卷积神经网络基础
- [必备]：序列数据处理
- [扩展]：RNN、LSTM

---

## 2. 核心原理

### 2.1 核心思想

TCN的核心创新是**因果卷积 + 膨胀卷积**：
1. **因果卷积**：确保输出只依赖历史输入，不看未来
2. **膨胀卷积**：指数级扩大感受野，覆盖长时依赖

核心思想概括为：**用卷积的结构+RNN的思路，实现高效序列建模**

### 2.2 工作流程

```
1. 输入序列：x_1, x_2, ..., x_t
2. 因果卷积：每一层只看之前的输入
3. 膨胀卷积：按指数膨胀：d = 1, 2, 4, 8, ...
4. 残差连接：稳定深层网络训练
5. 输出：序列预测/分类
```

### 2.3 关键概念

| 概念 | 说明 |
|------|------|
| **Causal Conv** | 当前输出只依赖之前输入 |
| **Dilated Conv** | d=dilation时，跳过d-1个点 |
| **Receptive Field** | 感受野大小 |

**感受野计算**：
$$R_k = 1 + 2 \sum_{i=1}^{k} (d_i - 1)$$

### 2.4 vs RNN对比

| 方面 | RNN | TCN |
|------|-----|-----|
| 并行化 | 难 | 易 |
| 梯度 | 容易消失 | 稳定 |
| 感受野 | 线性增长 | 指数增长 |
| 内存 | O(seq_len) | O(kernel * layers) |

---

## 3. 数学公式与推导

### 3.1 因果卷积

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

其中卷积核大小K，确保只看$t, t-1, ..., t-K+1$的历史输入。

```
时刻 t:  y_t = w_0*x_t + w_1*x_{t-1} + w_2*x_{t-2}
         不看 x_{t+1}, x_{t+2}...
```

### 3.2 膨胀卷积

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

d为膨胀因子，控制跳跃间隔：

- d=1：普通卷积，x_t, x_{t-1}, x_{t-2}...
- d=2：跳跃1个，x_t, x_{t-2}, x_{t-4}...
- d=4：跳跃3个，x_t, x_{t-4}, x_{t-8}...

### 3.3 残差块

TCN包含多个残差块，每块结构：
```
Dilated Conv → BatchNorm → ReLU → Dropout
    ↓
Dilated Conv → BatchNorm → ReLU → Dropout
    ↓
残差连接 → ReLU
```

残差公式：
$$y = F(x) + x$$

### 3.4 感受野推导

设TCN有k层，膨胀因子为$[d_1, d_2, ..., d_k]$，卷积核为K：

第i层的有效感受野：
$$R_i = 1 + (K-1) \cdot d_i$$

总感受野（近似）：
$$R \approx \prod_{i=1}^{k} d_i \cdot (K-1)$$

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
# 序列数据格式
X = torch.randn(batch, seq_len, features)
# batch: 批量大小
# seq_len: 序列长度
# features: 特征维度

# 示例：时间序列预测
X = torch.randn(32, 100, 10)  # 32样本，100时刻，10维特征
y = torch.randn(32, 100, 1)   # 100步预测
```

### 4.2 模型配置

| 参数 | 说明 | 典型值 |
|------|------|--------|
| input_size | 输入维度 | 特征数 |
| num_channels | 每层通道数 | [32, 32, 32] |
| kernel_size | 卷积核大小 | 3 |
| dropout | Dropout率 | 0.2 |

### 4.3 训练配置

```python
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for x, y_batch in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 5. 应用场景

### 5.1 时序预测

```python
# 股票预测
model = TCN(input_size=5, num_channels=[64, 64, 64], kernel_size=3)
# 输入：[open, high, low, volume, close]
# 输出：下一天价格预测
```

### 5.2 动作分割

```python
# 视频动作识别
model = TCN(input_size=2048, num_channels=[256, 256], kernel_size=5)
# 输入：视频帧特征
# 输出：每帧动作标签
```

### 5.3 语音识别

```python
# 语音增强
model = TCN(input_size=257, num_channels=[128, 128], kernel_size=3)
# 输入：频谱特征
# 输出：增强后的频谱
```

### 5.4 对比其他方法

| 场景 | TCN | LSTM | Transformer |
|------|-----|-----|-------|
| 序列建模 | ✓ | ✓ | ✓ |
| 并行效率 | 高 | 低 | 中 |
| 长依赖 | 指数 | 线性 | 中 |
| 内存 | 小 | 大 | 大 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **并行高效** | 卷积可并行计算 |
| **梯度稳定** | 残差连接 |
| **长时依赖** | 指数级感受野 |
| **内存效率** | 不需要序列内存 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **历史局限** | 需要足够的历史数据 |
| **固定感受野** | 不如注意力灵活 |
| **边界处理** | 需要padding |

### 6.3 注意事项

- 需要足够的历史窗口长度
- 膨胀因子不宜太大
- 注意padding导致的边界效应

---

## 7. 调库实现（Python）

### 7.1 PyTorch实现

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """移除padding，保持因果性"""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
    
    def forward(self, x):
        return x[:, :, :-self.padding].contiguous()


class TemporalBlock(nn.Module):
    """时序残差块"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 padding, dilation, dropout=0.2):
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时序卷积网络"""
    
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    padding=(kernel_size-1) * dilation,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 7.2 TCN训练示例

```python
from torch.utils.data import DataLoader, TensorDataset

# 数据
X = torch.randn(100, 50, 10)
y = torch.randint(0, 2, (100, 50))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16)

# 模型
model = TemporalConvNet(
    input_size=10,
    num_channels=[32, 32, 32],
    kernel_size=3,
    dropout=0.2
)

# 训练
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):
    total_loss = 0
    for x, y_batch in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred[:, -1, :], y_batch.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

---

## 8. 手工代码实现（理解原理）

### 8.1 简化版TCN

```python
import numpy as np

class SimpleTCN:
    """简化版TCN - 理解原理"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # 简化权重
        self.W = np.random.randn(kernel_size, input_dim, hidden_dim) * 0.01
        self.b = np.zeros(hidden_dim)
    
    def causal_conv(self, x, W):
        """因果卷积"""
        seq_len = x.shape[0]
        out = np.zeros((seq_len, W.shape[2]))
        
        for t in range(seq_len):
            for k in range(self.kernel_size):
                if t - k >= 0:
                    out[t] += np.dot(x[t-k], W[k])
        
        return out + self.b
    
    def dilated_conv(self, x, W, dilation):
        """膨胀卷积"""
        seq_len = x.shape[0]
        out = np.zeros((seq_len, W.shape[2]))
        
        for t in range(seq_len):
            for k in range(self.kernel_size):
                idx = t - dilation * k
                if idx >= 0:
                    out[t] += np.dot(x[idx], W[k])
        
        return out + self.b
    
    def forward(self, x):
        """前向传播"""
        # 第一层：dilation=1
        h1 = self.causal_conv(x, self.W)
        h1 = np.maximum(0, h1)  # ReLU
        
        # 第二层：dilation=2
        h2 = self.dilated_conv(h1, self.W, 2)
        h2 = np.maximum(0, h2)
        
        return h2


def receptive_field_size(num_layers, kernel_size, dilation_base):
    """计算感受野大小"""
    return 1 + (kernel_size - 1) * sum([dilation_base**i for i in range(num_layers)])


if __name__ == "__main__":
    np.random.seed(42)
    
    # 测试
    n = 100
    input_dim = 10
    x = np.random.randn(n, input_dim)
    
    model = SimpleTCN(input_dim, hidden_dim=16, kernel_size=3)
    out = model.forward(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    
    # 感受野
    rf = receptive_field_size(4, 3, 2)
    print(f"感受野: {rf}")
```

---

## 9. 可视化与结果理解

### 9.1 感受野可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_receptive_field():
    """可视化感受野增长"""
    
    num_layers = 6
    kernel_size = 3
    
    layers = list(range(1, num_layers + 1))
    dilations = [2 ** i for i in range(num_layers)]
    receptive_fields = [
        1 + 2 * sum([(2**j - 1) for j in range(i)])
        for i in range(1, num_layers + 1)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 膨胀因子
    axes[0].bar(layers, dilations)
    axes[0].set_xlabel('TCN层数')
    axes[0].set_ylabel('膨胀因子')
    axes[0].set_title('膨胀因子 (2^i)')
    axes[0].set_yscale('log')
    
    # 感受野
    axes[1].bar(layers, receptive_fields)
    axes[1].set_xlabel('TCN层数')
    axes[1].set_ylabel('感受野大小')
    axes[1].set_title('感受野增长')
    
    plt.tight_layout()
    plt.savefig('tcn_receptive_field.png', dpi=100)
    plt.show()


def visualize_dilated_conv():
    """可视化膨胀卷积"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    dilation = [1, 2, 4, 8]
    seq_len = 16
    
    for i, d in enumerate(dilation):
        ax = axes[i // 2, i % 2]
        
        # 可视化连接
        connections = []
        for t in range(seq_len):
            for k in range(3):
                if t - d * k >= 0:
                    connections.append((t, t - d * k))
        
        ax.set_xlim(-1, seq_len + 1)
        ax.set_ylim(-1, seq_len + 1)
        ax.set_title(f'Dilation = {d}')
        
        # 绘制连接线
        for target, source in connections[:20]:
            ax.plot([source, target], [0, 1], 'b-', alpha=0.3)
        
        ax.set_xlabel('Input Index')
        ax.set_ylabel('Output Index')
    
    plt.tight_layout()
    plt.savefig('tcn_dilation.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    visualize_receptive_field()
    visualize_dilated_conv()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| MAE | 平均绝对误差 |
| Accuracy | 分类准确率 |
| F1 | F1分数 |

### 10.2 评估代码

```python
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def evaluate_tcn(model, X_test, y_test):
    """评估TCN"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        
    if len(pred.shape) > 2:
        pred = pred[:, -1, :]
    
    # 回归
    mse = mean_squared_error(y_test, pred)
    
    # 分类
    pred_class = (pred > 0.5).int()
    acc = accuracy_score(y_test, pred_class)
    f1 = f1_score(y_test, pred_class)
    
    print(f"MSE: {mse:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    
    return {'MSE': mse, 'Accuracy': acc, 'F1': f1}
```

---

## 11. 常见问题与易错点

### Q1: 感受野不足？

**答案**：增加TCN层数或增大膨胀因子。

### Q2: 梯度消失？

**答案**：使用残差连接，或增加残差块。

### Q3: 边界问题？

**答案**：适当增加padding，或截断边界输出。

### Q4: 训练太慢？

**答案**：减少层数，或用weight_norm。

### Q5: 过拟合？

**答案**：增加Dropout，或减少参数。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 因果卷积 | 只看历史 |
| 膨胀卷积 | 指数感受野 |
| 残差连接 | 稳定训练 |

### 12.2 公式汇总

因果卷积：
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

膨胀卷积：
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

感受野：
$$R = 1 + 2 \sum_{i=1}^{K} (d_i - 1)$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. TCN的因果卷积确保：
   - A) 输出只看未来
   - B) 输出只看历史
   - C) 可以看任意时刻

2. 膨胀因子d=4时，卷积核跳过几个点？
   - A) 1
   - B) 3
   - C) 4

3. TCN相比RNN的优点是：
   - A) 梯度更稳定
   - B) 可以并行计算
   - C) 两者都是

### 13.2 简答题

1. 解释因果卷积和普通卷积的区别？
2. 为什么膨胀卷积可以指数级扩大感受野？
3. 比较TCN和LSTM的优劣？

### 13.3 编程题

1. 实现一个4层TCN。
2. 计算感受野大小。
3. 在时间序列数据上测试。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
CNN基础
    ↓
一维卷积
    ↓
因果卷积
    ↓
膨胀卷积
    ↓
TCN
    ↓
TCN+Attention
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| WaveNet | TCN前身 |
| LSTM | 序列建模 |
| Transformer | 注意力机制 |
| ConvLSTM | CNN+LSTM |

### 14.3 扩展阅读

1. Lea et al. (2017). Temporal Convolutional Networks for Action Segmentation
2. Bai et al. (2018). An Empirical Evaluation of Generic Convolutional Networks for Sequence Modeling

---

## 附录

### A. 参数速查

| 参数 | 推荐值 |
|------|--------|
| kernel_size | 3 |
| num_channels | [32, 32, 32] |
| dropout | 0.2 |

### B. 参考

1. Bai et al. (2018). TCN. arXiv:1803.01271

---

**文档结束**