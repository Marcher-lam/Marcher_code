# LSTM 学习文档

> 解决RNN梯度消失的利器——门控机制精讲

---

## 1. 算法基础认知

### 1.1 什么是LSTM

**LSTM（Long Short-Term Memory）** 是RNN的改进版本，通过门控机制有效解决了RNN的梯度消失问题，能够学习长距离依赖关系。

### 1.2 为什么需要LSTM

```
RNN的问题:
序列: [用户5分钟前点了科技文章, ..., 现在的行为]
                              ↑
RNN: 梯度消失 → 早期信息丢失 → 无法利用长距离信息

LSTM的解决方案:
通过"细胞状态"(Cell State)直接传递信息，梯度可以无损流动
```

### 1.3 在推荐系统中的应用

| 应用 | 说明 |
|------|------|
| **DIEN** | 阿里深度兴趣演化网络，用GRU建模兴趣变化 |
| **GRU4Rec** | 序列推荐经典模型 |
| **用户行为建模** | 捕捉用户兴趣的长期变化 |
| **会话推荐** | 基于会话内行为预测下一步 |

---

## 2. 核心原理

### 2.1 LSTM的结构

```
                    细胞状态 Cₜ (信息高速公路)
                    ↓──────────────────────↓
                    │                      │
hₜ₋₁ → [遗忘门] → [×] → [输入门] → [+] → [×] → [输出门] → hₜ
         ↑         ↑      ↑         ↑      ↑
        hₜ₋₁      Cₜ₋₁   hₜ₋₁     C̃ₜ    hₜ₋₁
                          ↑
                         xₜ
```

### 2.2 三个门

| 门 | 作用 | 决定什么 |
|----|------|---------|
| **遗忘门** fₜ | 控制丢弃哪些旧信息 | 从细胞状态中丢弃什么 |
| **输入门** iₜ | 控制写入哪些新信息 | 向细胞状态中添加什么 |
| **输出门** oₜ | 控制输出哪些信息 | 从细胞状态中输出什么 |

---

## 3. 数学公式与推导

### 3.1 完整公式

**遗忘门**：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**：

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**候选细胞状态**：

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**更新细胞状态**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**：

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**隐藏状态**：

$$h_t = o_t \odot \tanh(C_t)$$

### 3.2 为什么能缓解梯度消失

细胞状态的梯度：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

> 如果遗忘门 $f_t ≈ 1$，梯度可以无损传递；由网络**自己学习**何时保留/遗忘信息。

### 3.3 参数量计算

对于 hidden_size = h, input_size = d：

每个门有参数: $W$ 维度 $(h, h+d)$，$b$ 维度 $(h,)$

$$\text{总参数} = 4 \times [(h+d) \times h + h] = 4[h(h+d) + h]$$

**示例**：input_size=128, hidden_size=256

$$4 \times [256 \times (256+128) + 256] = 4 \times 98560 = 394,240$$

---

## 4. 训练过程讲解

```
1. 初始化: h₀ = 0, C₀ = 0
2. 对序列中每个时间步:
   a. 计算遗忘门 fₜ (决定遗忘多少旧信息)
   b. 计算输入门 iₜ 和候选 C̃ₜ (决定加入多少新信息)
   c. 更新细胞状态 Cₜ = fₜ·Cₜ₋₁ + iₜ·C̃ₜ
   d. 计算输出门 oₜ 和隐藏状态 hₜ
3. 用最终隐藏状态做预测
4. BPTT计算梯度
```

---

## 5-6. 应用与优缺点

### 应用场景
- DIEN中的兴趣演化层（GRU变体）
- 序列推荐（GRU4Rec）
- 用户行为序列建模

### 优缺点

| 优点 | 缺点 |
|------|------|
| 解决梯度消失 | 参数量4倍于RNN |
| 学习长距离依赖 | 仍然无法并行 |
| 门控机制灵活 | 训练比GRU慢 |

---

## 7. 调库实现

```python
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM序列分类模型"""
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后一个时间步的隐藏状态
        out = self.fc(lstm_out[:, -1, :])
        return out


# 使用示例
if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = LSTMModel(input_size=32, hidden_size=64, num_classes=5)
    x = torch.randn(16, 10, 32)  # (batch, seq_len, input_size)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
```

---

## 8. 手工代码实现

```python
"""
LSTM Cell 纯 NumPy 实现
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    """手写LSTM单元"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 合并的权重矩阵 [h_{t-1}, x_t] → 门控
        concat_size = hidden_size + input_size
        
        # 四组参数: 遗忘门, 输入门, 候选状态, 输出门
        self.W_f = np.random.randn(concat_size, hidden_size) * 0.1
        self.b_f = np.ones((1, hidden_size))  # 遗忘门偏置初始化为1（默认记住）
        
        self.W_i = np.random.randn(concat_size, hidden_size) * 0.1
        self.b_i = np.zeros((1, hidden_size))
        
        self.W_c = np.random.randn(concat_size, hidden_size) * 0.1
        self.b_c = np.zeros((1, hidden_size))
        
        self.W_o = np.random.randn(concat_size, hidden_size) * 0.1
        self.b_o = np.zeros((1, hidden_size))
    
    def forward(self, x_t, h_prev, c_prev):
        """
        单步前向传播
        
        返回: h_t, c_t
        """
        # 拼接 [h_{t-1}, x_t]
        concat = np.concatenate([h_prev, x_t], axis=1)
        
        # 遗忘门: 决定丢弃哪些旧信息
        f_t = sigmoid(concat @ self.W_f + self.b_f)
        
        # 输入门: 决定写入哪些新信息
        i_t = sigmoid(concat @ self.W_i + self.b_i)
        
        # 候选细胞状态
        c_hat_t = np.tanh(concat @ self.W_c + self.b_c)
        
        # 更新细胞状态
        c_t = f_t * c_prev + i_t * c_hat_t
        
        # 输出门
        o_t = sigmoid(concat @ self.W_o + self.b_o)
        
        # 隐藏状态
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    def forward_sequence(self, X):
        """处理整个序列"""
        batch_size, seq_len, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        hidden_states = []
        for t in range(seq_len):
            h, c = self.forward(X[:, t, :], h, c)
            hidden_states.append(h.copy())
        
        return np.array(hidden_states), h, c


if __name__ == "__main__":
    np.random.seed(42)
    
    lstm = LSTMCell(input_size=8, hidden_size=16)
    
    # 模拟用户行为序列
    X = np.random.randn(4, 10, 8)  # 4个样本, 10步序列, 8维特征
    
    hidden_states, h_final, c_final = lstm.forward_sequence(X)
    print(f"序列输入: {X.shape}")
    print(f"隐藏状态: {hidden_states.shape}")
    print(f"最终隐藏状态: {h_final.shape}")
```

---

## 12. 学习总结

1. **LSTM核心**：细胞状态 + 三个门（遗忘/输入/输出）
2. **解决梯度消失**：细胞状态提供梯度直通路径
3. **门控机制**：网络自主学习何时遗忘/记忆/输出
4. **vs GRU**：LSTM参数更多，GRU更简洁，效果通常相近

---

## 14. 学习路径

```
RNN → [当前: LSTM] → GRU → Attention → Transformer → GRU4Rec/DIEN
```
