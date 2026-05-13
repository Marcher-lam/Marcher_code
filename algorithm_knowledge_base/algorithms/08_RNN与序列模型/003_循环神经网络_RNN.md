# 循环神经网络 (RNN) 学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

循环神经网络是一种能处理序列数据并保存"记忆"的神经网络架构。

## 1. 算法基础认知

### 1.1 什么是RNN

循环神经网络（Recurrent Neural Network, RNN）是一种专门用于处理序列数据的神经网络。与传统前馈网络不同，RNN有"内部状态"（隐藏状态），能保存之前的信息并影响当前输出。

### 1.2 直觉类比

想象你读一本书：你不会每次只读当前这句话，而是会记住之前的内容来理解现在的意思。RNN正是这样——它有一个"记忆"（隐藏状态），把之前的信息传递给现在。

### 1.3 历史背景

RNN的概念在1980年代提出。1997年，Sepp Hochreiter和Jürgen Schmidhuber提出了LSTM，解决了RNN的梯度消失问题。如今RNN及其变体在NLP、语音识别等领域有广泛应用。

### 1.4 算法定位

RNN是**监督学习**的**序列模型**，属于深度学习范畴。

### 1.5 前置知识

- 神经网络基础
- 梯度下降
- Python / PyTorch

## 2. 核心原理

### 2.1 核心思想

RNN的核心公式：
$$h_t = f(W \cdot h_{t-1} + U \cdot x_t + b)$$
$$y_t = g(V \cdot h_t + c)$$

每个时间步的输出依赖于当前输入和之前的隐藏状态。

### 2.2 工作流程

1. 初始化隐藏状态
2. 对序列每个元素：更新隐藏状态 → 计算输出
3. 返回完整输出序列或最终状态

### 2.3 关键概念

- 隐藏状态：网络的"记忆"
- 时间步：序列的每个位置
- BPTT：时间反向传播

## 3. 数学公式

### 3.1 前向传播

对于时间步t：
- 输入门：$x_t$ (当前) + $h_{t-1}$ (之前)
- 候选隐藏：$\tanh(W_h h_{t-1} + W_x x_t + b)$
- 激活：$h_t = \tanh(\tilde{h}_t)$

### 3.2 反向传播（BPTT）

沿时间展开网络，计算梯度。

## 4. 训练过程

### 4.1 超参数

| 超参数 | 作用 | 推荐范围 |
|--------|------|---------|
| hidden_size | 隐藏维度 | 128-512 |
| num_layers | 层数 | 1-3 |
| learning_rate | 学习率 | 0.001 |

## 5. 应用场景

1. 语言模型
2. 机器翻译
3. 序列生成
4. 情感分析

## 6. 优缺点

### 6.1 优点

- 能处理变长序列
- 共享参数

### 6.2 缺点

- 梯度消失/爆炸
- 难以长距离依赖

## 7. 调库实现

```python
"""
循环神经网络(RNN) - 使用PyTorch
"""

import torch
import torch.nn as nn

print("=" * 60)
print("循环神经网络(RNN)示例")
print("=" * 60)

# 1. 定义RNN模型
print("\n定义RNN模型...")

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # 取最后一个时间步
        out = self.fc(out[:, -1, :])
        return out

# 参数
input_size = 10
hidden_size = 32
output_size = 2

model = SimpleRNN(input_size, hidden_size, output_size)
print(f"RNN参数: {sum(p.numel() for p in model.parameters())}")

# 2. 测试前向传播
print("\n测试前向传播...")

batch_size = 4
seq_len = 5

x = torch.randn(batch_size, seq_len, input_size)
output = model(x)

print(f"输入: {x.shape}")
print(f"输出: {output.shape}")

# 3. 完整训练示例
print("\n" + "-" * 40)
print("序列分类训练示���")
print("-" * 40)

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 准备数据
n_samples = 100
X = torch.randn(n_samples, 10, input_size)
y = torch.randint(0, output_size, (n_samples,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

print("\n示例完成")
## 8. 手工代码实现（核心算法纯代码实现）

以下是一个完整的RNN手写实现，包含前向传播、梯度计算和参数更新：

```python
"""
循环神经网络(RNN) - 纯手写实现
核心：前向传播、BPTT梯度计算、时间序列处理
"""

import numpy as np

class SimpleRNN:
    """
    简单RNN手写实现
    
    核心公式：
    h_t = tanh(W_{xh} * x_t + W_{hh} * h_{t-1} + b_h)
    y_t = W_{hy} * h_t + b_y
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        参数：
        - input_size: 输入维度
        - hidden_size: 隐藏状态维度
        - output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化参数（Xavier初始化）
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + output_size))
        
        np.random.seed(42)
        self.W_xh = np.random.randn(input_size, hidden_size) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.W_hy = np.random.randn(hidden_size, output_size) * scale_hy
        
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
    def forward(self, X):
        """
        前向传播
        
        X: (batch_size, seq_len, input_size)
        返回: (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = X.shape
        
        # 初始化隐藏状态为0
        h = np.zeros((batch_size, self.hidden_size))
        
        # 存储中间值用于反向传播
        self.hidden_states = [h.copy()]
        self.inputs = X
        
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, input_size)
            
            # RNN核心公式
            h = np.tanh(np.matmul(x_t, self.W_xh) + 
                       np.matmul(h, self.W_hh) + 
                       self.b_h)
            
            y = np.matmul(h, self.W_hy) + self.b_y
            
            self.hidden_states.append(h.copy())
            outputs.append(y)
        
        # 堆叠所有时间步的输出
        outputs = np.stack(outputs, axis=1)
        
        return outputs
    
    def backward(self, X, y_true, learning_rate=0.01):
        """
        BPTT（Backpropagation Through Time）
        
        时间反向传播核心：
        1. 计算每个时间步的输出梯度
        2. 沿时间展开，梯度反向传回
        3. 累加各时间步的参数梯度
        """
        batch_size, seq_len, _ = X.shape
        
        # 前向传播
        y_pred = self.forward(X)
        
        # 计算输出梯度：dL/dy
        # 假设使用MSE损失
        dL_dy = 2 * (y_pred - y_true) / batch_size
        
        # 初始化梯度累加器
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 将隐藏状态从后向前传播
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(seq_len)):
            # 输出层梯度
            dL_dh = np.matmul(dL_dy[:, t, :], self.W_hy.T)
            dL_dh += dh_next  # 加上来自后续的梯度
            
            # tanh的梯度: d(tanh)/dx = 1 - tanh^2
            h_t = self.hidden_states[t + 1]  # 已保存的隐藏状态
            dL_dh_raw = dL_dh * (1 - h_t ** 2)
            
            # 参数梯度累加
            x_t = X[:, t, :]
            h_prev = self.hidden_states[t]
            
            dW_xh += np.matmul(x_t.T, dL_dh_raw)
            dW_hh += np.matmul(h_prev.T, dL_dh_raw)
            dW_hy += np.matmul(h_t.T, dL_dy[:, t, :])
            db_h += dL_dh_raw.sum(axis=0, keepdims=True)
            db_y += dL_dy[:, t, :].sum(axis=0, keepdims=True)
            
            # 梯度传回给之前的隐藏状态
            dh_next = np.matmul(dL_dh_raw, self.W_hh.T)
        
        # 梯度裁剪（防止梯度爆炸）
        clip_value = 1.0
        dW_xh = np.clip(dW_xh, -clip_value, clip_value)
        dW_hh = np.clip(dW_hh, -clip_value, clip_value)
        dW_hy = np.clip(dW_hy, -clip_value, clip_value)
        
        # 梯度下降更新
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
        
    def train_step(self, X, y_true, learning_rate=0.01):
        """一步训练"""
        # 前向传播计算损失
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y_true) ** 2)
        
        # 反向传播更新
        self.backward(X, y_true, learning_rate)
        
        return loss

# 训练示例
def main():
    print("=" * 60)
    print("RNN - 纯手写实现")
    print("=" * 60)
    
    # 参数
    input_size = 10
    hidden_size = 20
    output_size = 2
    seq_len = 5
    batch_size = 4
    
    # 创建模型
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    print(f"\n模型参数:")
    print(f"  W_xh: {rnn.W_xh.shape}")
    print(f"  W_hh: {rnn.W_hh.shape}")
    print(f"  W_hy: {rnn.W_hy.shape}")
    
    # 生成训练数据
    np.random.seed(0)
    X = np.random.randn(batch_size, seq_len, input_size)
    y_true = np.random.randn(batch_size, seq_len, output_size)
    
    print(f"\n输入: {X.shape}")
    print(f"目标: {y_true.shape}")
    
    # 训练
    print("\n训练过程:")
    for epoch in range(20):
        loss = rnn.train_step(X, y_true, learning_rate=0.01)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")
    
    # 测试
    print("\n预测结果:")
    y_pred = rnn.forward(X)
    print(f"预测: {y_pred.shape}")
    print(f"最后一时间步: {y_pred[0, -1, :]}")

if __name__ == "__main__":
    main()
```

**代码核心要点**：

1. **前向传播**：利用历史隐藏状态计算当前隐藏状态
2. **BPTT**：沿时间展开网络，梯度反向传播
3. **tanh激活**：将值压缩到(-1,1)
4. **梯度裁剪**：防止梯度爆炸

---

## 9. 可视化与结果理解

### 9.1 隐藏状态可视化

```python
"""
RNN隐藏状态可视化 - 观察网络"记忆"
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_hidden_states(model, X, save_path='rnn_hidden_states.png'):
    """
    可视化RNN隐藏状态的演化
    
    观察：
    1. 隐藏状态随时间的变化
    2. 不同样本的隐藏状态差异
    """
    # 前向传播
    model.forward(X)
    hidden_states = np.array(model.hidden_states[1:])  # (seq_len, batch, hidden)
    
    # 只取第一个样本
    hs = hidden_states[:, 0, :]  # (seq_len, hidden)
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: 热力图
    ax1 = axes[0]
    im = ax1.imshow(hs.T, aspect='auto', cmap='RdBu_r')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Hidden Unit')
    ax1.set_title('Hidden States Over Time')
    plt.colorbar(im, ax=ax1)
    
    # 图2: 各个隐藏单元的时序变化
    ax2 = axes[1]
    for i in range(min(5, hs.shape[1])):
        ax2.plot(hs[:, i], label=f'Unit {i}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Activation')
    ax2.set_title('Hidden Unit Activations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"可视化已保存为 {save_path}")
    plt.close()

# 测试
print("\n" + "-" * 40)
print("RNN隐藏状态可视化")
print("-" * 40)

np.random.seed(0)
X_test = np.random.randn(1, 5, 10)
model_test = SimpleRNN(10, 8, 2)

try:
    visualize_hidden_states(model_test, X_test)
except Exception as e:
    print(f"可视化失败: {e}")
```

### 9.2 结果解读

**隐藏状态解读**：

1. **时序变化**：不同时间步的隐藏状态反映了对序列不同位置的"记忆"
2. **单元特异性**：某些单元对特定时间步敏感
3. **信息保留**：后期隐藏状态包含早期信息

---

## 10. 模型评估

### 10.1 评估指标

```python
"""
RNN模型评估
"""

def evaluate_rnn(model, X_test, y_test):
    """评估RNN模型"""
    # 预测
    y_pred = model.forward(X_test)
    
    # MSE
    mse = np.mean((y_pred - y_test) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_test))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

# 评估示例
print("\n" + "-" * 40)
print("RNN模型评估")
print("-" * 40)

np.random.seed(0)
X_val = np.random.randn(2, 5, 10)
y_val = np.random.randn(2, 5, 2)

metrics = evaluate_rnn(rnn, X_val, y_val)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### 10.2 评估指标说明

| 指标 | 说明 |
|------|------|
| MSE | 均方误差，越小越好 |
| RMSE | MSE的平方根 |
| MAE | 绝对误差，平均偏离 |

---

## 11. 常见问题与易错点

### 11.1 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 梯度消失 | 时间序列太长 | 使用LSTM/GRU |
| 梯度爆炸 | 权重太大 | 梯度裁剪 |
| 难以收敛 | 隐藏层太小 | 增大隐藏层 |

### 11.2 使用问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 长序列效果差 | 早期信息丢失 | 使用双向RNN |
| 内存占用大 | 存储所有隐藏状态 | Truncated BPTT |

### 11.3 典型易错点

1. **维度不匹配**：W_xh: (input, hidden)，不是(hidden, input)
2. **隐藏状态未初始化**：每次forward要清空hidden_states
3. **梯度未累加**：dh_next要累加，不能直接赋值
4. **tanh梯度**：是(1-h²)，不是(1+h²)

---

## 12. 学习总结

### 12.1 核心思想

RNN的核心是**共享参数 + 时间展开**：用同一套权重处理序列的不同时刻。

### 12.2 关键公式

**前向传播**：
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

**BPTT**：
$$\frac{\partial L}{\partial W} = \sum_t \frac{\partial L}{\partial h_t}\frac{\partial h_t}{\partial W}$$

### 12.3 后续学习

1. **LSTM**：门控机制解决梯度消失
2. **GRU**：简化版LSTM
3. **双向RNN**：增强信息保留

---

## 13. 练习题与思考题

### 13.1 基础题

**问题**：RNN的"循环"指的是什么？

**答案**：隐藏状态从上一时间步传递到当前时间步，形成"循环"。

### 13.2 进��题

**问题**：为什么RNN难以捕捉长距离依赖？

**答案**：梯度沿时间反向传播时会指数衰减（消失）或指数增长（爆炸）。

### 13.3 开放题

**问题**：如何判断RNN训练后的隐藏状态是否"记住"了有用信息？

**答案可包含**：
1. 可视化隐藏状态演化
2. 用隐藏状态做下游任务
3. 分析不同时间步的注意力分布

---

## 14. 学习路径建议

### 14.1 前置算法

1. **神经网络基础**：全连接网络
2. **梯度下降**：SGD
3. **反向传播**：链式法则

### 14.2 平行算法

1. **反向传播时序BPTT**
2. **Truncated BPTT**

### 14.3 进阶算法

1. **LSTM**：门控RNN
2. **GRU**：简化LSTM
3. **Transformer**：注意力RNN

### 14.4 推荐资源

| 资源 | 类型 |
|------|------|
| RNN原始论文 | Werbos, 1990 |
| LSTM原始论文 | Hochreiter & Schmidhuber, 1997 |
| CS224N课程 | Stanford NLP |

---

*第8-14章内容添加完成*
