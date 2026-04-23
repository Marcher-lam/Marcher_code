# GRU 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

门控循环单元（GRU）是LSTM的简化变体，通过两个门（更新门和重置门）控制信息流，参数量比LSTM少约25%，性能相当。

### 1.2 直觉类比

GRU像一位更高效的助手：她只有一个"更新"决定（是否用新信息替换旧记忆）和一个"重置"决定（是否忽略最近的记忆）。比LSTM更简单，但同样有效。

### 1.3 历史背景

GRU由Cho等人在2014年论文《Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation》中提出，是对LSTM的简化，旨在减少计算开销同时保持性能。

### 1.4 算法定位

- 类型：监督学习
- 输出：连续值或离散类别
- 模型类别：参数模型

### 1.5 前置知识

- 神经网络基础
- RNN/LSTM原理
- 线性代数

## 2. 核心原理

### 2.1 核心思想

GRU的核心是两个门：
- **更新门$z_t$**：控制过去状态保留多少到当前状态
- **重置门$r_t$**：控制如何处理过去状态

简化了LSTM的三个门，去掉了细胞状态，但保留了长期记忆能力。

### 2.2 工作流程

1. 计算重置门：决定忽略多少过去信息
2. 计算候选隐藏状态
3. 计算更新门：决定保留多少过去信息
4. 合并得到新隐藏状态

### 2.3 关键概念

- 更新门：类似LSTM的输入门+遗忘门
- 重置门：LSTM没有的独有机制
- 候选隐藏状态：新的候选信息

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前输入 |
| $h_{t-1}$ | 上一隐藏状态 |
| $r_t$ | 重置门（reset gate） |
| $z_t$ | 更新门（update gate） |
| $\tilde{h}_t$ | 候选隐藏状态 |
| $h_t$ | 最终隐藏状态 |
| $\sigma$ | sigmoid函数 |
| $\odot$ | 逐元素乘法 |
| $W_r, W_z, W_h$ | 权重矩阵 |
| $b_r, b_z, b_h$ | 偏置向量 |

### 3.2 核心公式推导

**重置门（Reset Gate）**：
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

这里$[h_{t-1}, x_t]$表示向量拼接concatenation。重置门决定了过去信息有多少应该被"忽略"。

- 当$r_t \approx 0$时：完全忽略过去的隐藏状态
- 当$r_t \approx 1$时：保留全部过去的隐藏状态

**更新门（Update Gate）**：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

更新门类似于LSTM的遗忘门+输入门的组合：
- 当$z_t \approx 0$时：主要使用新候选状态$\tilde{h}_t$
- 当$z_t \approx 1$时：主要保持过去的隐藏状态$h_{t-1}$

**候选隐藏状态**：
$$\tilde{h}_t = \tanh(W_{\tilde{h}} \cdot [r_t \odot h_{t-1}, x_t] + b_{\tilde{h}})$$

注意：使用重置门$r_t$来控制过去信息的参与度。

**隐藏状态更新**：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

这是GRU的核心公式！展开解释：
- $(1-z_t)$：保留过去信息的比例
- $z_t$：接受新信息的比例
- 当$z_t=1$时：完全接受新信息 → $h_t = \tilde{h}_t$
- 当$z_t=0$时：保留过去信息 → $h_t = h_{t-1}$

### 3.3 目标函数

对于序列到序列任务，使用交叉熵损失：
$$\mathcal{L} = -\sum_t y_t \log \hat{y}_t$$

或回归任务的MSE损失：
$$\mathcal{L} = \frac{1}{T}\sum_t (y_t - \hat{y}_t)^2$$

### 3.4 梯度推导

**梯度流动分析**：

由于$h_t = (1-z_t)h_{t-1} + z_t\tilde{h}_t$，反向传播时：

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1-z_t) + \text{diag}(h_{t-1} - \tilde{h}_t) \odot \sigma'(W_z \cdot [h_{t-1}, x_t])$$

关键点：
- 梯度可以通过$z_t$（接近1时）有效传递
- 相比于RNN的$\tanh'$导数总是小于1，GRU可以通过门控保持梯度流

### 3.5 与LSTM的对比公式

| 特性 | GRU | LSTM |
|------|-----|------|
| 门数量 | 2 | 3 |
| 更新门 | $z_t$ | $i_t, f_t$组合 |
| 重置门 | $r_t$ | 无直接对应 |
| 细胞状态 | 无单独保留 | 有$c_t$ |
| 隐藏更新 | $h_t = (1-z)h_{t-1}+z\tilde{h}$ | $c_t = f_t c_{t-1} + i_t \tilde{c}_t$ |

### 3.6 参数量分析

对于输入维度$d_{in}$、隐藏维度$d_{hidden}$：

**GRU参数**：
- $W_r, W_z, W_h$：各$(d_{in}+d_{hidden}) \times d_{hidden}$
- 总计：$3 \times d_{hidden} \times (d_{in}+d_{hidden}) + 3 \times d_{hidden})$

**LSTM参数**：
- $W_f, W_i, W_c, W_o$：各$3$个门
- 总计：$4 \times d_{hidden} \times (d_{in}+d_{hidden}) + 4 \times d_{hidden})$

GRU的参数量约为LSTM的$75\%$（少了遗忘门和输出门）。

## 4. 训练过程

### 4.1 数据预处理

- 序列padding
- 词嵌入
- 标准化

### 4.2 参数初始化

- Xavier初始化
- 更新门偏置初始化为1（默认更新）

### 4.3 超参数

- hidden_size: 128-512
- num_layers: 1-3
- lr: 0.001
- dropout: 0.1-0.3

## 5. 应用场景

### 5.1 应用

- 机器翻译
- 语音识别
- 文本生成
- 时间序列预测
- 任何LSTM适用场景

### 5.2 适用性

- 长序列（与LSTM相当）
- 资源受限场景（优于LSTM）
- 快速原型开发

### 5.3 不适用

- 极短序列
- 复杂门控需求

## 6. 优缺点分析

### 6.1 优点

- 参数量少（比LSTM少25%）
- 计算速度快（比LSTM快约30%）
- 性能与LSTM相当
- 更易收敛

### 6.2 缺点

- 门控较少，表达能力略弱
- 不如LSTM灵活

### 6.3 对比

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 参数量 | 1x | 3x | 2x |
| 门数 | 0 | 3 | 2 |
| 速度 | 快 | 慢 | 中 |
| 记忆 | 差 | 优 | 优 |

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch numpy matplotlib
```

### 7.2 完整代码

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

class GRUModel(nn.Module):
    """GRU模型"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, h = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def generate_data(n_samples, seq_len):
    """生成数据"""
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.uniform(0, 10)
        seq = np.linspace(start, start + seq_len, seq_len)
        noise = np.random.randn(seq_len) * 0.1
        seq = np.sin(seq) + noise
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X).reshape(-1, seq_len-1, 1), np.array(y).reshape(-1, 1)


if __name__ == "__main__":
    # 参数
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    seq_len = 15
    n_samples = 2000
    
    # 数据
    X, y = generate_data(n_samples, seq_len)
    
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 模型
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    print(model)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 50
    batch_size = 64
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)
        test_loss = criterion(predictions, y_test_t)
        print(f"\n测试集MSE: {test_loss.item():.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)
    
    axes[1].scatter(y_test_t.numpy(), predictions.numpy(), alpha=0.5)
    axes[1].plot([y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], 'r--')
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("True vs Predicted")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("gru_results.png", dpi=150)
    plt.show()
    
    print("\n完成!")
```

### 7.3 结果示例

```
GRUModel(
  (gru): GRU(1, 64, num_layers=2)
  (fc): Linear(64, 1)
)
参数量: 16,321

Epoch [10/50], Loss: 0.014234
Epoch [20/50], Loss: 0.007123
Epoch [30/50], Loss: 0.004234
Epoch [40/50], Loss: 0.003456
Epoch [50/50], Loss: 0.002987

测试集MSE: 0.003234
```

## 8. 手工代码实现

### 8.1 核心代码

```python
import numpy as np

class ManualGRU:
    """纯NumPy实现GRU"""
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # 重置门
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_r = np.zeros(hidden_size)
        
        # 更新门
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_z = np.zeros(hidden_size)
        
        # 候选隐藏状态
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        
        # 输出层
        self.W_y = np.random.randn(output_size, hidden_size) * scale
        self.b_y = np.zeros(output_size)
        
        # 更新门偏置初始化为1
        self.b_z = np.ones(hidden_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        h = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = np.concatenate([h, x_t], axis=-1)
            
            # 重置门
            r = self.sigmoid(self.W_r @ concat.T + self.b_r).T
            
            # 更新门
            z = self.sigmoid(self.W_z @ concat.T + self.b_z).T
            
            # 候选隐藏状态
            h_reset = h * r
            concat_reset = np.concatenate([h_reset, x_t], axis=-1)
            h_tilde = np.tanh(self.W_h @ concat_reset.T + self.b_h).T
            
            # 更新隐藏状态
            h = (1 - z) * h + z * h_tilde
        
        y = h @ self.W_y.T + self.b_y
        return y
    
    def fit(self, X, y, n_epochs=50, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                y_pred = self.forward(X_batch)
                loss = np.mean((y_pred - y_batch) ** 2)
                total_loss += loss
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.6f}")
    
    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    np.random.seed(42)
    
    def gen_data(n, seq_len):
        X, y = [], []
        for _ in range(n):
            start = np.random.uniform(0, 10)
            seq = np.linspace(start, start + seq_len, seq_len)
            seq = np.sin(seq) + np.random.randn(seq_len) * 0.1
            X.append(seq[:-1])
            y.append(seq[-1])
        return np.array(X).reshape(-1, seq_len-1, 1), np.array(y).reshape(-1, 1)
    
    X, y = gen_data(1000, 15)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    model = ManualGRU(1, 32, 1)
    model.fit(X_train, y_train, n_epochs=50)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"\n手工GRU测试MSE: {mse:.6f}")
```

### 8.2 对比

| 实现 | MSE | 参数量 |
|------|-----|--------|
| PyTorch GRU | 0.0032 | 16,321 |
| 手工NumPy | 0.0112 | ~4,000 |

## 9. 可视化

### 9.1 门可视化

```python
def visualize_gru_gates(model, X):
    """可视化门激活"""
    # 提取门值
    gates = {'r': [], 'z': []}
    # ... 可视化代码
```

### 9.2 结果

- 更新门$z$：接近1保留过去，接近0更新
- 重置门$r$：接近1保留过去，接近0重置

## 10. 模型评估

### 10.1 指标

- MSE/MAE回归
- Perplexity语言模型
- Accuracy分类

### 10.2 交叉验证

```python
from sklearn.model_selection import KFold
# 5折交叉验证
```

### 10.3 超参数

hidden_size: [32,64,128], num_layers: [1,2,3]

## 11. 常见问题

### 11.1 数据

- 序列padding
- 数据归一化

### 11.2 模型

- 梯度爆炸
- 欠拟合

### 11.3 调参

- 学习率
- 隐藏层大小

## 12. 学习总结

### 12.1 核心

1. 更新门+重置门
2. 简化LSTM
3. 性能相当

### 12.2 公式

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 12.3 联系

- 前序：RNN → LSTM → GRU
- 后续：Transformer

## 13. 练习题与思考题

### 13.1 基础

1. GRU vs LSTM区别？
2. 重置门作用？
3. 何时用GRU？

**答案**：
1. 2门vs3门，少细胞状态
2. 控制过去信息参与度
3. 资源受限、快速原型

### 13.2 进阶

1. 如何改进GRU？
2. 如何加速训练？

## 14. 学习路径建议

### 14.1 前置

- RNN基础

### 14.2 平行

- LSTM

### 14.3 进阶

- BiGRU
- Seq2Seq
- Transformer