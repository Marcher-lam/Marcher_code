# Self-Attention 自注意力机制

> 自注意力让序列中每个位置能够直接关注所有其他位置，实现真正的全局上下文建模。

## 1. 算法基础认知

### 1.1 什么是自注意力

自注意力（Self-Attention）也称为内部注意力，是注意力机制的一种特殊形式——Query、Key、Value都来自同一个序列。这使得序列中的每个元素可以直接"关注"序列中的所有其他元素。

### 1.2 直觉类比

当你读一句话时，你会同时理解每个词在上下文中的含义。"bank"这个词到底指"银行"还是"河岸"，需要看它在句子中的上下文。自注意力就是让模型具备这种能力。

### 1.3 历史背景

- **2016年**：Cheng等人提出"Long Memory Document Modeling"
- **2017年**：Transformer论文中正式确立为核心组件

### 1.4 算法定位

- **所属类别**：注意力机制的特例
- **前置知识**：基础注意力机制

## 2. 核心原理

### 2.1 核心思想

自注意力的核心是**同源比较**——用同一个序列的表示来计算相互之间的关联程度。

### 2.2 工作流程

```
输入序列X 
  ↓
转换为Q、K、V（使用不同的线性变换）
  ↓
计算每对位置之间的相似度
  ↓
加权聚合所有位置的信息
  ↓
输出序列（每个位置都包含了全局上下文）
```

### 2.3 与普通注意力的区别

| 特性 | 普通注意力 | 自注意力 |
|------|------------|----------|
| Q来源 | Decoder状态 | 输入序列 |
| K来源 | Encoder状态 | 输入序列 |
| V来源 | Encoder状态 | 输入序列 |
| 应用 | Seq2Seq | 单序列建模 |

## 3. 数学公式与推导

### 3.1 核心公式

给定输入序列 $X = (x_1, x_2, ..., x_n)$

$$ Q = XW^Q $$
$$ K = XW^K $$
$$ V = XW^V $$

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.2 输出特性

输出序列的每个位置 $y_i$ 包含：
$$ y_i = \sum_{j=1}^{n} \alpha_{ij} v_j $$

其中 $\alpha_{ij}$ 是位置 $i$ 对位置 $j$ 的注意力权重。

## 4. 应用场景

### 4.1 典型应用

1. **Transformer Encoder**：每个词看整个句子
2. **Transformer Decoder**：已生成的部分看之前的内容
3. **BERT**：双向自注意力

### 4.2 适用数据

- 需要理解上下文关系的序列
- 变长序列处理
- 需要长距离依赖建模

## 5. 优缺点分析

### 5.1 优点

| 优点 | 说明 |
|------|------|
| 全局视野 | 每个位置看到所有位置 |
| 路径最短 | 任意位置间只需一步 |
| 并行计算 | 所有位置同时计算 |

### 5.2 缺点

| 缺点 | 说明 |
|------|------|
| O(n²)复杂度 | 序列长度平方 |
| 无位置信息 | 需要额外的位置编码 |

## 6. 调库实现

```python
import torch
from torch.nn import MultiheadAttention

# PyTorch内置的MultiheadAttention
attention = MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1
)

# 准备输入
batch_size = 2
seq_len = 10
embed_dim = 512

x = torch.randn(seq_len, batch_size, embed_dim)

# 自注意力（Q=K=V）
output, attn_weights = attention(x, x, x)

print(f"输出: {output.shape}")
print(f"权重: {attn_weights.shape}")
```

## 7. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """自注意力实现"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        # Q、K、V的投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 线性投影
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 分成多个头
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
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
attn = SelfAttention(d_model, num_heads)

output, weights = attn(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"注意力: {weights.shape}")
```

## 8. 可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_self_attention(attn_weights, words):
    """可视化自注意力矩阵"""
    
    # 取第一个样本的第一个头
    attn = attn_weights[0, 0].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, 
               xticklabels=words,
               yticklabels=words,
               cmap='viridis')
    plt.title('自注意力权重')
    plt.show()

# 示例
words = ['I', 'love', 'this', 'movie']
# 假设的注意力权重
weights = torch.rand(1, 1, 4, 4)
weights = F.softmax(weights, dim=-1)

visualize_self_attention(weights, words)
```

## 9. 评估与问题

### 9.1 评估方式

自注意力通过下游任务评估：
- 分类准确率
- 翻译质量(BLEU)
- 困惑度(Perplexity)

### 9.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 注意力分散 | 权重太均匀 | 增加d_k缩放 |
| 注意力不收敛 | 学习率不对 | 调整学习率 |

## 10. 学习总结

自注意力的本质是**让序列自己告诉自己每个位置与其他位置的关系**。这是Transformer能够并行处理序列且能建模长距离依赖的关键。

## 11. 练习题

**题目**：自注意力和普通注意力有什么区别？

**答案**：自注意力的Q、K、V都来自同一个序列，而普通注意力的Q来自一个序列（Decoder），K、V来自另一个序列（Encoder）。

## 12. 学习路径建议

- **前置**：注意力基础
- **进阶**：Multi-Head Attention → Transformer
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

