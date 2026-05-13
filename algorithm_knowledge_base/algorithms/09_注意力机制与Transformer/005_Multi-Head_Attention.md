# Multi-Head Attention 多头注意力机制

> 多头注意力通过并行运行多个注意力头，让模型同时学习不同类型的关系，是Transformer的核心组件。

## 1. 算法基础认知

### 1.1 什么是多头注意力

多头注意力（Multi-Head Attention）同时运行多个独立的注意力机制（"头"），每个头可以学习不同的关联模式，然后将结果拼接起来。

### 1.2 直觉类比

想象你在分析一段对话。有的人关注说的内容（语义），有的人关注谁在说话（说话人身份），有的人关注说话的语气（情感）。多头注意力的道理类似——每个头关注不同类型的关系。

### 1.3 历史背景

- **2017年**：Transformer论文中首次提出
- **原因**：单个注意力只能学习一种关系，多头可以学习多种

### 1.4 算法定位

- **所属类别**：注意力机制扩展
- **前置**：基础注意力、自注意力

## 2. 核心原理

### 2.1 核心思想

每个"头"有自己独立的Query、Key、Value投影，可以学习不同的关联模式。所有头的输出拼接后，再通过一个线性变换得到最终结果。

### 2.2 工作流程

```
输入X
  ↓
【并行h个注意力头】
  ↓  头1：学习位置关系
  ↓  头2：学习语法关系
  ↓  ……
  ↓  头h：学习语义关系
  ↓
拼接所有头的输出
  ↓
线性变换输出
```

### 2.3 关键参数

- $h$：注意力头数量（标准是8）
- $d_k$：每个头的维度（$d_{model}/h$）

## 3. 数学公式与推导

### 3.1 核心公式

#### 单头注意力

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^KVW_i^V)$$

其中$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$

#### 多头拼接

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中$W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### 3.2 维度关系

$$d_k = d_v = d_{model} / h$$

标准配置（$d_{model}=512, h=8$）：
$$d_k = d_v = 64$$

## 4. 应用场景

### 4.1 典型应用

1. **Transformer**：每个层都有多头注意力
2. **BERT**：12-16个头
3. **GPT**：12-96个头

### 4.2 每个头可能学习的内容

不同头可能关注：
- 邻近词关系（局部依赖）
- 语法结构
- 指代关系
- 语义相似

## 5. 优缺点分析

### 5.1 优点

| 优点 | 说明 |
|------|------|
| 多样性 | 同时学习多种关系 |
| 表达能力 | 丰富的特征组合 |
| 可并行 | 计算可并行 |

### 5.2 缺点

| 缺点 | 说明 |
|------|------|
| 复杂度 | h倍计算量 |
| 调参 | 头数需要实验 |

## 6. 调库实现

```python
import torch
from torch.nn import MultiheadAttention

# PyTorch内置实现
attention = MultiheadAttention(
    embed_dim=512,  # d_model
    num_heads=8,    # 头数
    dropout=0.1
)

# 准备输入
batch = 2
seq_len = 10
embed_dim = 512

x = torch.randn(seq_len, batch, embed_dim)

# 自注意力
output, weight = attention(x, x, x)

print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"权重: {weight.shape}")
```

## 7. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力实现"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 投影矩阵（合并为���个矩阵提高效率）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 投影并分头
        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        q = split_heads(self.W_q(query))
        k = split_heads(self.W_k(key))
        v = split_heads(self.W_v(value))
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


# 测试
d_model = 512
num_heads = 8
batch = 2
seq_len = 10

x = torch.randn(batch, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)

output, weights = mha(x, x, x)
print(f"输出形状: {output.shape}")
print(f"注意力形状: {weights.shape}")
```

## 8. 可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_all_heads(weights, words):
    """可视化所有注意力头"""
    num_heads = weights.size(1)
    seq_len = weights.size(2)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_heads):
        attn = weights[0, i].cpu().numpy()
        sns.heatmap(attn, ax=axes[i], cmap='viridis', 
                   xticklabels=[], yticklabels=[])
        axes[i].set_title(f'头 {i+1}')
    
    plt.tight_layout()
    plt.show()

# 示例
num_heads = 8
seq_len = 10
weights = torch.rand(1, num_heads, seq_len, seq_len)
words = ['I', 'love', 'this', 'movie', 'very', 'much', '.']

visualize_all_heads(weights, words)
```

## 9. 评估方式

多头注意力通过：
1. 分析不同头学到的模式
2. 下游任务表现
3. 消融实验（移除某些头）

## 10. 学习总结

多头注意力的核心是**并行多视角**——每个头可以学习不同的关联模式，然后组合起来得到更丰富的表示。这是Transformer强大的关键。

## 11. 练习题

**题目**：为什么需要多个注意力头而不是一个？

**答案**：单个注意力只能学习一种固定的关联模式。多头注意力允许模型同时学习语法、语义、位置等多种关系，显著增强表达能力。

## 12. 学习路径建议

- **前置**：自注意力
- **进阶**：Transformer各层堆叠
- **资源**：Transformer论文

## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估

