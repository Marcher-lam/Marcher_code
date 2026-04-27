# SwiGLU 门控前馈网络 学习文档

> 用门控机制增强前馈网络，以更少参数实现更好的表达能力。

> 来源线索：本节内容根据原书附录C C.2节关于feed forward module/SwiGLU的讲解整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
SwiGLU 是一种使用 SiLU 激活 + 门控线性单元的增强型前馈网络设计。

### 直觉类比
普通的FFN像一个工人单独工作，SwiGLU像两个工人配合——一个处理数据，另一个做质检（门控），只让高质量信息通过。

### 历史背景
GLU (Gated Linear Unit) 由 Dauphin 等人在 2017 年提出。Shazeer (2020) 将 SwiGLU 引入 Transformer 架构。2023 年 Llama 模型采用了 SwiGLU，此后成为几乎所有新一代 LLM 的标准配置。

### 算法定位
- **类型**：神经网络架构组件 / 前馈网络变体
- **性质**：模型的一部分，训练和推理均使用

### 前置知识
- 了解 Transformer 中的前馈网络 (FFN)
- 了解激活函数的概念
- 了解 element-wise 操作

## 2. 核心原理

### 核心思想
标准FFN只有两条线性层 + 一个激活：FFN(x) = W2(σ(W1(x)))。SwiGLU使用三条线性层：两个并行的W1/W2（W2不经过激活，充当门控信号），一个W3用于最终投影。SwiGLU(x) = (SiLU(xW1) ⊙ xW2) W3。

### 门控机制的数学直觉
W1 路径（经过SiLU）：提取真正的特征信息
W2 路径（无激活）：学习"哪些信息重要"——门控强度
两者逐元素相乘 (⊙)：信息 × 门控 = 选择性信息传递

这与 LSTM 中的门控机制有相同的数学精神——让网络学会"忽略什么、保留什么"。

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $x \in \mathbb{R}^{d}$ | 输入向量 |
| $W_1, W_2 \in \mathbb{R}^{d \times h}$ | 扩展投影矩阵 |
| $W_3 \in \mathbb{R}^{h \times d}$ | 收缩投影矩阵 |
| $\odot$ | 逐元素相乘 |

### 标准FFN
$$\text{FFN}(x) = \text{ReLU}(xW_1)W_2$$

### SwiGLU
$$\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_2)W_3$$

其中 SiLU(x) = x · σ(x)，σ 是 sigmoid 函数。

### 参数比较（Qwen3 0.6B 为例）
- d=1024, h=3072
- SwiGLU参数: 1024×3072×3 ≈ 9.4M
- 同等效果的标准FFN(W1宽加倍): 1024×6144×2 ≈ 12.6M
- SwiGLU少约25%参数，效果相当或更好

## 4. 训练过程讲解

SwiGLU作为模型的一部分，不需要单独训练。关键实现细节：

- **隐藏维度**：h 通常取 emb_dim 的 8/3 倍（或类似比例），这是从实践中得出的经验值
- **无偏置**：原书 Qwen3 的SwiGLU中所有linear层都设 bias=False，节省参数同时不损害性能
- **训练稳定**：门控机制的乘法交互会在反向传播中自动学习最佳的门控比例

### 关键超参数
| 参数 | 作用 | Qwen3 0.6B的默认值 |
|------|------|----------|
| hidden_dim/emb_dim 比率 | 控制FFN容量 | 3.0 (3072/1024) |
| 是否使用bias | 减少参数 | False |

## 5. 应用场景

所有使用Transformer架构的最新一代LLM：Llama 3、Mistral、Qwen3、Gemma等。SwiGLU已成为事实上的标准FFN设计。

## 6. 优缺点分析

| 优点 | 说明 |
|------|------|
| 参数效率更高 | 相同效果下参数更少（~25%） |
| 表达能力更强 | 门控增加乘法交互，超越线性叠加 |
| 训练更稳定 | 门控自然地过滤掉噪声信号 |

| 缺点 | 说明 |
|------|------|
| 多一个矩阵乘法 | 需要3个linear层而非2个 |
| 调参经验少 | 标准FFN有更多历史实践经验 |

## 7. 调库实现

```python
"""SwiGLU调库实现"""
import torch
import torch.nn as nn
# PyTorch 2.7+ 内置 GatedSiLU
# ffn = nn.GatedSiLU(1024, 3072)  # 直接使用
```

## 8. 手工代码实现

```python
"""SwiGLU手工实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardSwiGLU(nn.Module):
    """Qwen3风格的SwiGLU前馈网络"""
    def __init__(self, emb_dim, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        # 三个全连接层，简化参数量的关键：hidden_dim不用加倍
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x):
        # 门控路径(fc2无激活) × 特征路径(fc1+SiLU)
        gate = self.fc2(x)
        feature = F.silu(self.fc1(x))  # SiLU = x * sigmoid(x)
        return self.fc3(feature * gate)

# 测试
ffn = FeedForwardSwiGLU(1024, 3072)
x = torch.randn(2, 128, 1024)
y = ffn(x)
print(f"SwiGLU: input {x.shape} → output {y.shape}")
print(f"参数量: {sum(p.numel() for p in ffn.parameters()):,}")
```

## 9-14. 评估、问题、总结、练习、路径

### 常见问题
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| OOM(out of memory) | 显存不够 | hidden_dim太大 | 降低hidden_dim或使用Gradient Checkpointing |

### 学习总结
SwiGLU通过"特征提取 + 门控过滤"的双路径设计，以更少的参数达到了比传统FFN更好的效果——是Transformer架构设计从简单到精致演进的代表性案例。

### 练习题
**题1**：为什么SwiGLU参数更少但效果更好？

**参考答案**：核心在于乘法交互（feature ⊙ gate）比加法交互更有表达力——它允许网络学习"某些维度要保留，某些要抑制"的非线性门控决策，而不仅是线性组合。这类似注意力中的softmax——通过乘法引入非线性的选择性。

### 学习路径
- **前置**：Transformer基础、FFN原理
- **进阶**：MoE(Mixture of Experts)中门控机制的设计
