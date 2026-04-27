# 位置编码（Positional Encoding）学习文档

> 为序列注入位置信息，让注意力机制能够感知元素的顺序。

## 1. 算法基础认知

### 一句话定义

位置编码是一种给序列中的每个元素添加位置信息的机制，让基于自注意力的模型能够区分不同位置的元素。

### 直觉类比

"我打你"和"你打我"由相同的字组成，但意思完全不同。位置编码就像给每个字发一张"身份证"，标明它出现在句子的哪个位置。

### 历史背景

- **2017年**：Transformer论文首次提出正弦余弦位置编码
- **2019年**：BERT使用可学习的位置编码
- **2020年**：Relative Positional Encoding提出改进

### 算法定位

位置编码是**Transformer的必备组件**，解决注意力机制无法感知位置的问题。

---

## 2. 核心原理

### 核心思想

自注意力本身是**排列等变**的——改变输入顺序，输出只会改变对应位置，不会改变内容。位置编码通过添加与位置相关的信息来解决这个问题。

### 工作流程

1. 输入元素通过Embedding得到向量 $x_i$
2. 计算位置编码 $PE_i$
3. 合并：$x'_i = x_i + PE_i$
4. 送入Transformer处理

---

## 3. 数学公式与推导

### 正弦余弦位置编码

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：位置，0到序列长度-1
- $i$：维度索引，0到$d_{model}$-1
- $d_{model}$：模型维度

### 性质

1. **周期性**：不同维度有不同周期（$2\pi, 2\pi/10000, ...$）
2. **线性可分**：$PE_{pos+k}$可以表示为$PE_{pos}$的线性组合
3. **相对位置**：可以通过线性变换恢复相对距离

### 可学习位置编码

$$PE_{pos} = \text{LearnableEmbedding}(pos)$$

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算除数
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)

# 测试
if __name__ == "__main__":
    pe = PositionalEncoding(512, max_len=100)
    x = torch.randn(32, 50, 512)
    out = pe(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
    print(f"位置编码示例:\n{pe.pe[0, :5, :10]}")
```

---

## 5. 手工代码实现

```python
import numpy as np

def get_positional_encoding(max_len, d_model):
    """计算正弦余弦位置编码矩阵"""
    pe = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            # 偶数维度
            pe[pos, i] = np.sin(pos / np.power(10000, 2*i / d_model))
            # 奇数维度
            if i + 1 < d_model:
                pe[pos, i+1] = np.cos(pos / np.power(10000, 2*i / d_model))
    
    return pe

# 测试
if __name__ == "__main__":
    pe = get_positional_encoding(100, 64)
    print(f"位置编码形状: {pe.shape}")
    print(f"位置0的编码前5维: {pe[0, :5]}")
    print(f"位置10的编码前5维: {pe[10, :5]}")
```

---

## 6. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_positional_encoding(d_model=64, max_len=100):
    """可视化位置编码"""
    pe = get_positional_encoding(max_len, d_model)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding')
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150)
    plt.show()
    
    # 绘制特定维度的变化曲线
    plt.figure(figsize=(10, 4))
    for dim in [0, 1, 32, 63]:
        plt.plot(range(max_len), pe[:, dim], label=f'dim={dim}')
    plt.xlabel('Position')
    plt.value('Encoding Value')
    plt.legend()
    plt.title('Positional Encoding across different dimensions')
    plt.savefig('pe_curves.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_positional_encoding()
```

---

## 7. 优缺点

| 方法 | 优点 | 缺点 |
|------|------|------|
| 正弦余弦 | 不需要训练，可外推 | 周期可能造成混淆 |
| 可学习 | 更灵活 | 难以外推过长序列 |
| 相对位置 | 更好地建模相对距离 | 实现复杂 |

---

## 8. 练习题

1. **基础**：为什么Transformer需要位置编码，而RNN不需要？
2. **进阶**：位置编码的周期对模型有什么影响？

---

## 9. 学习路径

- 前置：Transformer基础
- 平行：相对位置编码、旋转位置编码（RoPE）
- 进阶：ALIBi（太长序列处理）