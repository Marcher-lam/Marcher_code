# TCN（时序卷积网络）学习文档

> 基于卷积神经网络处理序列数据的模型，利用因果卷积和膨胀卷积实现长时依赖建模

---

## 1. 算法基础认知

**一句话定义**：TCN是一种使用一维卷积处理序列数据的神经网络，通过因果卷积确保时间顺序、膨胀卷积扩大感受野，兼具RNN的序列建模能力和CNN的高效并行计算。

**直觉类比**：TCN就像一个带"放大镜"的扫描仪——从序列开头逐步往后扫描，每一层通过膨胀卷积能看到更远的历史信息，同时保证不回头看未来的数据。

**历史背景**：2017年，Lea等人提出TCN用于视频动作分段，后续在时序预测任务中广泛使用。TCN在保持RNN序列建模能力的同时，利用卷积的并行计算优势。

**算法定位**：
- 类型：深度学习 → 序列建模
- 输出：序列标签/预测
- 模型类型：卷积神经网络

**前置知识**：
- [必备]：卷积神经网络基础
- [必备]：序列数据处理
- [扩展]：RNN、LSTM

---

## 2. 核心原理

### 2.1 核心思想

TCN的核心创新是**因果卷积 + 膨胀卷积**：
1. **因果卷积**：确保输出只依赖历史输入
2. **膨胀卷积**：指数级扩大感受野

核心思想可以概括为：**用卷积的结构+RNN的思路，实现高效序列建模**。

### 2.2 工作流程

1. **输入层**：接收序列数据
2. **因果卷积**：按时间顺序卷积
3. **膨胀卷积**：扩大感受野
4. **残差连接**：稳定训练
5. **输出**：序列预测/分类

### 2.3 关键概念

- **Causal Conv**：当前输出只依赖之前输入
- **Dilated Conv**：dilation=d时，跳过d-1个点
- **Receptive Field**：$R_i = 1 + 2\sum_{k=1}^{K} (d_k-1)$

---

## 3. 数学公式

### 3.1 因果卷积

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

其中卷积核大小K，确保只看$t, t-1, ..., t-K+1$。

### 3.2 膨胀卷积

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

d为膨胀因子，控制跳跃间隔。

### 3.3 整体架构

TCN包含多个残差块，每块：
- Dilated Conv → ReLU → Dilated Conv → ReLU → 残差
- 残差：Conv(1, filters) + Conv(1, filters)

---

## 4. 训练方法

### 4.1 PyTorch实现

```python
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """移除padding"""
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
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                           stride=stride, padding=padding,
                           dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                           stride=stride, padding=padding,
                           dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """TCN网络"""
    
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
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
        self.fc = nn.Conv1d(num_channels[-1], 1, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)
        y = self.network(x)
        y = self.fc(y)
        return y.transpose(1, 2)
```

---

## 5. 应用场景

### 5.1 典型应用

- **时序分类**：动作识别、语音识别
- **预测**：时间序列预测
- **分割**：视频动作分段
- **NLP**：文本分类

### 5.2 适用数据

- 序列数据
- 需要长时依赖
- 并行计算资源

---

## 6. 优缺点

### 6.1 优点

1. **感受野大**：膨胀卷积指数扩展
2. **并行计算**：GPU高效
3. **梯度稳定**：残差连接
4. **可变长度**：任意长度输入

### 6.2 缺点

1. **内存**：需要保存激活
2. **调参**：膨胀因子设置
3. **边界**：需要padding

---

## 7. 调库实现

```python
"""
TCN 训练示例
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 数据
X = torch.randn(100, 50, 10)  # seq_len=50, features=10
y = torch.randint(0, 2, (100,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16)

# 模型
model = TCN(input_size=10, num_channels=[16, 16, 16], kernel_size=3)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
for epoch in range(10):
    for x, y_batch in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred.squeeze(), y_batch.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 8. 可视化

```python
import matplotlib.pyplot as plt


def visualize_receptive_field():
    """可视化感受野增长"""
    layers = [1, 2, 3, 4]
    receptive_fields = [2**i * 3 for i in layers]
    
    plt.figure(figsize=(8, 4))
    plt.bar(layers, receptive_fields)
    plt.xlabel('TCN层数')
    plt.ylabel('感受野')
    plt.title('TCN感受野增长')
    plt.savefig('tcn_rf.png')
    plt.show()
```

---

## 9. 评估

```python
from sklearn.metrics import accuracy_score, f1_score

def evaluate_tcn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).squeeze()
        pred_class = (pred > 0.5).int()
    
    acc = accuracy_score(y_test, pred_class)
    f1 = f1_score(y_test, pred_class)
    print(f"Acc: {acc:.4f}, F1: {f1:.4f}")
```

---

## 10. 问题

### 10.1 常见问题

- **感受野不足**：增加TCN层数或膨胀因子
- **梯度消失**：使用残差连接

---

## 11. 学习总结

### 11.1 核心

✓ 因果卷积 + 膨胀卷积 + 残差

### 11.2 算法联系

- 前置：CNN、RNN
- 相关：WaveNet、Transformer
- 进阶：TCN+Attention

---

## 12. 练习

**问题**：TCN和LSTM的主要区别？

答案：TCN并行高效但感受野有限，LSTM序列建模但计算慢。

---

**文档结束**