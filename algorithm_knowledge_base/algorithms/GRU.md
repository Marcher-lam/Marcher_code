# GRU 学习文档

> 门控循环单元（Gated Recurrent Unit），通过更新门与重置门两个门控机制控制信息流动，以更少的参数实现与LSTM相当的长期依赖建模能力

---

## 1. 算法基础认知

### 1.1 一句话定义

GRU（Gated Recurrent Unit）是一种通过更新门（Update Gate）和重置门（Reset Gate）控制信息流动的循环神经网络变体，由Cho等人于2014年提出，能够在保持较低计算复杂度的同时有效学习长期依赖关系。

### 1.2 直觉类比：记笔记

想象你在听一节课并做笔记。你需要不断做出两个决策：

- **重置门 -- "擦除旧笔记"**：当老师开始讲一个全新的知识点时，你需要擦掉之前不相关的笔记，以全新的视角来记录新内容。重置门 $r_t$ 就控制"擦掉多少旧笔记"。当 $r_t \approx 0$ 时，之前的笔记被完全擦除；当 $r_t \approx 1$ 时，旧笔记被完整保留。
- **更新门 -- "合并新旧笔记"**：当你听完老师的新讲解后，你需要决定最终笔记怎么写 -- 是基本保留旧笔记，还是以新内容为主重写？更新门 $z_t$ 就控制"保留多少旧笔记 vs 采纳多少新笔记"。

对比之下：
- **GRU 用两本笔记**（一本是正在写的，一本是候选笔记），通过重置门和更新门来管理。
- **LSTM 用三本笔记**（遗忘门、输入门、输出门各管一本），功能更细致，但记笔记的速度更慢（计算量更大）。

### 1.3 历史背景

- **1997年**：Hochreiter和Schmidhuber提出LSTM，用3个门解决RNN梯度消失问题
- **2014年**：Cho等人在论文《Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation》中提出GRU，作为LSTM的简化版本，最初用于机器翻译的编码器-解码器架构
- **2014-2016年**：GRU在机器翻译、语音识别、文本分类等领域广泛应用，被证明在多数任务上与LSTM性能相当
- **2017年至今**：Transformer兴起后，GRU逐渐在大型NLP任务中被替代，但在计算资源受限、小数据量、边缘部署等场景中仍然是首选

### 1.4 算法定位

- **类型**：监督学习 -- 序列建模 -- 循环神经网络
- **输出**：序列（可担任编码器或解码器角色）
- **模型类型**：参数模型、判别模型/生成模型
- **核心创新**：用2个门（更新门+重置门）替代LSTM的3个门（遗忘门+输入门+输出门），减少约25%参数量的同时保持相当的表达能力

### 1.5 前置知识

- 基础神经网络（前馈网络、激活函数）
- 梯度下降和反向传播
- 传统RNN原理及梯度消失问题
- 矩阵运算和微分
- LSTM基本结构（便于对比理解）

---

## 2. 核心原理

### 2.1 核心思想

GRU的核心思想是**通过两个门控机制控制信息流动，使网络能够自适应地决定何时遗忘历史、何时保留记忆**：

1. **重置门（Reset Gate）$r_t$**：决定是否忽略之前的隐藏状态。当检测到当前输入与过去关联不大时（例如话题切换），将 $r_t$ 设为接近0，从而"擦除"历史信息，使候选隐藏状态只基于当前输入计算。
2. **更新门（Update Gate）$z_t$**：决定保留多少之前的隐藏状态。当检测到当前输入不需要太多更新时（例如填充词、停用词），将 $z_t$ 设为接近0，从而直接传递之前的隐藏状态，信息不受损失。

这种设计使GRU能够自适应地学习长期依赖，同时缓解梯度消失问题。关键洞察在于：**GRU的隐藏状态本身兼任了LSTM中记忆单元和隐藏状态两个角色**，通过巧妙的门控设计实现了信息流动的精确控制。

### 2.2 工作流程

```
输入序列: x = (x_1, x_2, ..., x_T)
隐藏状态: h = (h_0, h_1, ..., h_T)

对每个时间步 t:
  输入: x_t (当前输入向量), h_{t-1} (上一时刻隐藏状态)

  Step 1: 计算重置门
    r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)
    --> 决定遗忘多少过去信息

  Step 2: 计算更新门
    z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
    --> 决定保留多少过去 vs 接受多少新信息

  Step 3: 计算候选隐藏状态
    h_tilde_t = tanh(W * x_t + U * (r_t * h_{t-1}) + b)
    --> 基于(被重置门过滤的)过去信息 + 当前输入生成候选

  Step 4: 更新隐藏状态
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    --> 旧状态和新候选的加权组合

  输出: h_t (传递给下一个时间步)
```

### 2.3 关键概念解释

| 概念 | 符号 | 解释 |
|------|------|------|
| 重置门 | $r_t$ | 控制过去信息在候选状态中的贡献。$r_t \approx 0$ 时忽略之前内容，使候选状态只关注当前输入 |
| 更新门 | $z_t$ | 控制过去隐藏状态保留到当前的比例。$z_t \approx 0$ 时完全保留旧状态（长期记忆），$z_t \approx 1$ 时完全更新为新候选 |
| 候选状态 | $\tilde{h}_t$ | 可能成为新隐藏状态的候选值，基于当前输入和（被重置门过滤的）历史信息生成 |
| 隐藏状态 | $h_t$ | 传递给下一个时间步的上下文向量，同时也是GRU的输出 |

### 2.4 GRU如何用2个门实现LSTM的3个门功能

这是理解GRU设计哲学的关键。LSTM有遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$ 三个门，以及记忆单元 $c_t$ 和隐藏状态 $h_t$ 两个状态。GRU将它们巧妙地合并：

**对应关系分析**：

| LSTM | GRU | 说明 |
|------|-----|------|
| 遗忘门 $f_t$ + 输入门 $i_t$ | 更新门 $z_t$ | LSTM中 $f_t$ 控制遗忘、$i_t$ 控制输入，且 $f_t + i_t = 1$（实际不完全如此但趋势如此）。GRU直接用 $z_t$ 统一：$(1-z_t)$ 对应遗忘，$z_t$ 对应输入 |
| 记忆单元 $c_t$ + 隐藏状态 $h_t$ | 隐藏状态 $h_t$ | LSTM把长期记忆（$c_t$）和短期输出（$h_t$）分开存储。GRU合二为一，$h_t$ 同时承担两种角色 |
| 输出门 $o_t$ | （隐含在更新门中） | LSTM的输出门控制 $c_t \to h_t$ 的信息流。GRU没有显式输出门，隐藏状态本身就是输出 |
| （无直接对应） | 重置门 $r_t$ | GRU独有的设计，控制候选状态中历史信息的比例，这是LSTM没有的机制 |

**更直观的理解**：

LSTM的更新公式为：
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

GRU的更新公式为：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

可以看到，GRU的最终公式在结构上与LSTM的 $c_t$ 更新公式几乎完全一致。区别在于：
1. GRU没有独立的记忆单元，$h_t$ 直接扮演了 $c_t$ 的角色
2. GRU通过 $1-z_t$ 和 $z_t$ 的互补关系，隐式实现了遗忘和输入的平衡
3. GRU去掉了输出门，直接输出隐藏状态
4. GRU增加了重置门 $r_t$，让候选状态可以"从头开始"计算

### 2.5 GRU vs LSTM 对比

| 特性 | GRU | LSTM |
|------|-----|------|
| 门数量 | 2个（更新门、重置门） | 3个（输入门、遗忘门、输出门） |
| 状态数量 | 1个（隐藏状态 $h_t$） | 2个（记忆单元 $c_t$ + 隐藏状态 $h_t$） |
| 参数量 | $3d_h(k+d_h+1)$ | $4d_h(k+d_h+1)$ |
| 计算复杂度 | 较低（少一个门操作） | 较高 |
| 表达能力 | 稍弱（但实测差距很小） | 稍强 |
| 训练速度 | 较快 | 较慢 |
| 内存效率 | 较高 | 较低 |
| 长期依赖 | 能学习 | 能学习 |
| 重置机制 | 有显式重置门 | 无显式重置机制 |
| 序列长度 | 中等长度表现好 | 更长序列可能更优 |

从实际效果来看，GRU和LSTM在大多数任务上性能相当。多项研究表明（如Chung et al., 2014; Jozefowicz et al., 2015），两者的优劣因任务和数据集而异，没有一个在所有场景下都更优的选择。当数据量小或计算资源受限时，GRU通常是不二之选。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | $t$ 时刻的输入向量 | $(d_{in},)$ |
| $h_{t-1}$ | $t-1$ 时刻的隐藏状态 | $(d_h,)$ |
| $h_t$ | $t$ 时刻的隐藏状态 | $(d_h,)$ |
| $r_t$ | $t$ 时刻的重置门 | $(d_h,)$ |
| $z_t$ | $t$ 时刻的更新门 | $(d_h,)$ |
| $\tilde{h}_t$ | $t$ 时刻的候选隐藏状态 | $(d_h,)$ |
| $W_r, W_z, W_n$ | 输入权重矩阵 | $(d_h, d_{in})$ |
| $U_r, U_z, U_n$ | 隐藏状态权重矩阵 | $(d_h, d_h)$ |
| $b_r, b_z, b_n$ | 偏置向量 | $(d_h,)$ |
| $\sigma$ | Sigmoid函数 | - |
| $\tanh$ | 双曲正切函数 | - |
| $\odot$ | 逐元素乘法（Hadamard积） | - |

### 3.2 GRU前向传播公式

**Step 1: 重置门（Reset Gate）**

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

重置门接收当前输入 $x_t$ 和前一隐藏状态 $h_{t-1}$ 作为输入，通过sigmoid函数输出一个0到1之间的向量。它决定了在计算候选隐藏状态时，应该忽略多少之前的历史信息。

**Step 2: 更新门（Update Gate）**

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

更新门与重置门的结构完全相同，但参数独立。它决定了最终隐藏状态中，应该保留多少旧状态、采纳多少新候选。

**Step 3: 候选隐藏状态（Candidate Hidden State）**

$$\tilde{h}_t = \tanh(W_n x_t + U_n(r_t \odot h_{t-1}) + b_n)$$

候选隐藏状态的关键在于 $r_t \odot h_{t-1}$ 这一项：重置门对前一隐藏状态进行逐元素过滤，然后再参与候选状态的计算。当 $r_t \approx 0$ 时，候选状态几乎只依赖当前输入 $x_t$，实现了对历史的"重置"。

**Step 4: 隐藏状态更新（Hidden State Update）**

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

最终隐藏状态是旧状态和新候选的凸组合（加权平均）。由于 $z_t \in (0,1)$，$(1-z_t) + z_t = 1$，确保了状态值的稳定范围。

### 3.3 公式推导与直观理解

#### 3.3.1 为什么需要重置门？

考虑候选状态的计算：
$$\tilde{h}_t = \tanh(W_n x_t + U_n(r_t \odot h_{t-1}) + b_n)$$

**当 $r_t \approx 0$ 时**：
$$r_t \odot h_{t-1} \approx \mathbf{0}$$
$$\tilde{h}_t \approx \tanh(W_n x_t + b_n)$$

这意味着重置门关闭时，候选状态只基于当前输入计算，完全忽略了之前的历史。这在以下场景非常有用：
- 话题切换：前文在讨论体育，现在转到天气，旧信息不再相关
- 句子边界：新句子的开头，之前句子的信息可能不需要
- 异常检测：检测到异常输入时，忽略之前可能的噪声累积

**当 $r_t \approx 1$ 时**：
$$r_t \odot h_{t-1} \approx h_{t-1}$$
$$\tilde{h}_t \approx \tanh(W_n x_t + U_n h_{t-1} + b_n)$$

完全保留历史信息，等价于标准RNN的计算方式。这在以下场景有用：
- 同一话题的延续：句子内部词语之间
- 依赖关系：当前词依赖前文信息（如代词消解）

#### 3.3.2 为什么需要更新门？

考虑隐藏状态的更新：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

这是两个状态的加权平均：
- $(1-z_t) \odot h_{t-1}$：保留多少旧状态
- $z_t \odot \tilde{h}_t$：纳入多少新候选

**当 $z_t \approx 0$ 时**：
$$h_t \approx h_{t-1}$$

几乎完全保留之前的状态。这意味着信息可以无损失地在时间步之间传递，这正是解决梯度消失问题的关键。

**当 $z_t \approx 1$ 时**：
$$h_t \approx \tilde{h}_t$$

几乎完全更新为新的候选状态。这意味着当前输入携带了重要信息，需要大幅更新记忆。

#### 3.3.3 为什么更新门等价于LSTM的遗忘门+输入门？

在LSTM中：
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

对比GRU：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

形式上完全一致。映射关系为：
- $f_t \leftrightarrow (1 - z_t)$（遗忘比例 = 保留旧状态的比例）
- $i_t \leftrightarrow z_t$（输入比例 = 采纳新候选的比例）

LSTM中 $f_t$ 和 $i_t$ 是独立学习的，理论上可以 $f_t + i_t \neq 1$。而GRU通过设计强制 $(1-z_t) + z_t = 1$，这是一种简化但也是一种正则化。

### 3.4 梯度传播分析

#### 3.4.1 为什么GRU能缓解梯度消失？

关键在于隐藏状态的更新公式：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

对 $h_{t-1}$ 求偏导：
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + \text{diag}\left(\frac{\partial(z_t \odot \tilde{h}_t)}{\partial h_{t-1}}\right)$$

第一项 $\text{diag}(1 - z_t)$ 是一个对角矩阵，当 $z_t \approx 0$ 时，该项接近单位矩阵 $I$，梯度可以直接流过。这与LSTM中遗忘门 $f_t \approx 1$ 时梯度直接流过记忆单元的原理完全一致。

#### 3.4.2 梯度流的具体推导

考虑经过 $T$ 个时间步的梯度传播。设损失函数为 $L$，则：

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

对于标准RNN：
$$h_t = \tanh(W x_t + U h_{t-1} + b)$$
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) \cdot U$$

由于 $\tanh$ 的导数 $\text{diag}(1 - h_t^2)$ 的最大值是1（在 $h_t = 0$ 时），通常远小于1，经过连乘后梯度指数级衰减，导致梯度消失。

对于GRU：
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + z_t \odot \text{diag}(1 - \tilde{h}_t^2) \cdot U_n \cdot \text{diag}(r_t) + \text{（关于门的梯度项）}$$

第一项 $\text{diag}(1 - z_t)$ 提供了一条梯度直通路径。当 $z_t \approx 0$ 时：
$$\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t) \approx I$$

此时梯度可以直接无衰减地传播到很远的时间步，这正是解决梯度消失的关键。

#### 3.4.3 与LSTM梯度流的对比

| 梯度流特性 | GRU | LSTM |
|-----------|-----|------|
| 直通路径 | 通过 $h_t$ 的加法连接 | 通过 $c_t$ 的加法连接 |
| 控制机制 | 更新门 $z_t$ | 遗忘门 $f_t$ |
| 路径数量 | 1条主路径 | 1条主路径（$c_t$） |
| 门控复杂度 | 2个门 | 3个门 |
| 实际效果 | 在中等长度序列上效果良好 | 在超长序列上可能更稳定 |

### 3.5 参数量计算

设输入维度为 $d_{in}$，隐藏维度为 $d_h$。

**GRU参数量**：

每个门/候选状态都有输入权重、隐藏权重和偏置：

- 重置门：$W_r \in \mathbb{R}^{d_h \times d_{in}}$，$U_r \in \mathbb{R}^{d_h \times d_h}$，$b_r \in \mathbb{R}^{d_h}$，共 $d_h(d_{in} + d_h) + d_h$ 个参数
- 更新门：$W_z \in \mathbb{R}^{d_h \times d_{in}}$，$U_z \in \mathbb{R}^{d_h \times d_h}$，$b_z \in \mathbb{R}^{d_h}$，共 $d_h(d_{in} + d_h) + d_h$ 个参数
- 候选状态：$W_n \in \mathbb{R}^{d_h \times d_{in}}$，$U_n \in \mathbb{R}^{d_h \times d_h}$，$b_n \in \mathbb{R}^{d_h}$，共 $d_h(d_{in} + d_h) + d_h$ 个参数

**总参数量**：$3 \times [d_h(d_{in} + d_h) + d_h] = 3d_h(d_{in} + d_h + 1)$

**LSTM参数量**：

LSTM有4组参数（遗忘门、输入门、候选记忆、输出门）：

**总参数量**：$4 \times [d_h(d_{in} + d_h) + d_h] = 4d_h(d_{in} + d_h + 1)$

**参数量比**：GRU 参数量 = $\frac{3}{4}$ LSTM 参数量 = 75% LSTM参数量

例如，当 $d_{in} = 256$，$d_h = 512$ 时：
- GRU参数量：$3 \times 512 \times (256 + 512 + 1) = 3 \times 512 \times 769 = 1,180,416$
- LSTM参数量：$4 \times 512 \times 769 = 1,573,888$
- 差异：$393,472$ 个参数（约25%的减少）

### 3.6 优化目标

GRU的参数通常通过最大化对数似然估计来学习：

$$\hat{\theta} = \arg\max_{\theta} \sum_{t=1}^{T} \log P(y_t | x_1, \ldots, x_t; \theta)$$

等价于最小化交叉熵损失（分类任务）：

$$L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P_{\theta}(y_t^{(i)} | x_1^{(i)}, \ldots, x_t^{(i)})$$

或者均方误差（回归任务）：

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \|y_t^{(i)} - \hat{y}_t^{(i)}\|_2^2$$

其中 $\theta = \{W_r, U_r, b_r, W_z, U_z, b_z, W_n, U_n, b_n\}$ 包含所有可学习参数。

---

## 4. 训练过程讲解

### 4.1 序列数据处理

GRU处理序列数据的一般流程：

```
原始文本: "I love natural language processing"
    |
    v
分词: ["I", "love", "natural", "language", "processing"]
    |
    v
数值化: [23, 891, 412, 2034, 5682]
    |
    v
嵌入层: x_1, x_2, x_3, x_4, x_5 (每个是d_in维向量)
    |
    v
GRU层: h_1, h_2, h_3, h_4, h_5 (每个是d_h维向量)
    |
    v
输出层: 根据任务进行分类/回归/生成
```

### 4.2 数据预处理

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TextSequenceDataset(Dataset):
    # 文本序列数据集
    def __init__(self, texts, labels, vocab, max_len=128):
        # texts: 文本列表
        # labels: 标签列表
        # vocab: 词表字典 {word: idx}
        # max_len: 最大序列长度
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.get("<PAD>", 0)
        self.unk_idx = vocab.get("<UNK>", 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 将文本转换为索引序列
        indices = [self.vocab.get(w, self.unk_idx) for w in text.split()]
        # 截断或填充
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def create_padding_mask(seq, pad_idx=0):
    # 创建填充掩码，用于忽略padding位置的损失计算
    return (seq != pad_idx).float()  # (batch, seq_len), 1表示有效位置
```

### 4.3 参数初始化策略

GRU权重使用合理的初始化策略对训练效果至关重要：

```python
import torch.nn as nn

def init_gru_weights(module):
    # GRU参数初始化策略
    for name, param in module.named_parameters():
        if 'weight_ih' in name:  # 输入到隐藏的权重
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:  # 隐藏到隐藏的权重
            nn.init.orthogonal_(param)
            # 正交初始化有助于梯度在RNN中的稳定传播
            # 正交矩阵的范数为1，避免权重过大或过小
        elif 'bias' in name:
            nn.init.zeros_(param)
            # 对更新门偏置进行特殊初始化，促进信息流动
            # PyTorch的GRU中bias被分为3份：[r, z, n]
            n = param.size(0)
            start = n // 3
            mid = 2 * n // 3
            # 更新门偏置设为负值，使sigmoid输出偏向0（保留更多信息）
            param.data[start:mid].fill_(-1.0)
            # 重置门偏置设为正值，使sigmoid输出偏向1（保留更多历史）
            param.data[:start].fill_(1.0)
```

### 4.4 完整训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GRUTextModel(nn.Module):
    # GRU文本分类模型
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # GRU层
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 输出维度
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.gru(embedded)
        # output: (batch, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_dim)

        if self.bidirectional:
            # 合并双向最后隐藏状态
            forward_h = hidden[-2]
            backward_h = hidden[-1]
            last_hidden = torch.cat([forward_h, backward_h], dim=1)
        else:
            last_hidden = hidden[-1]

        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    # 训练一个epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # 梯度裁剪（对RNN训练非常重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    # 模型评估
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(dataloader), correct / total
```

### 4.5 BPTT（随时间反向传播）

GRU使用BPTT算法进行训练。与标准反向传播不同，BPTT需要沿时间维度展开计算图：

**前向传播**：按时间步 $t=1, 2, \ldots, T$ 顺序计算

**反向传播**：按时间步 $t=T, T-1, \ldots, 1$ 逆序计算梯度

每个时间步的梯度计算需要依赖下一个时间步传回的梯度：
$$\frac{\partial L}{\partial h_{t-1}} = \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_{t-1}}$$

这就是为什么RNN的训练速度较慢 -- 必须顺序计算，无法并行化。

### 4.6 梯度裁剪

梯度裁剪对GRU训练尤为重要，原因：
1. 序列展开后计算图很深，容易产生梯度爆炸
2. 不同时间步的梯度可能差异巨大

```python
# 方法1：按范数裁剪（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法2：按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 4.7 双向GRU

双向GRU同时从左到右和从右到左处理序列，能捕获前后两个方向的上下文信息：

```python
class BidirectionalGRU(nn.Module):
    # 双向GRU
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, hidden = self.gru(x)
        # output: (batch, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim)
        return output, hidden
```

### 4.8 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| hidden_dim | 隐藏状态维度 | 64-1024 | 256 |
| num_layers | GRU层数 | 1-4 | 2 |
| dropout | Dropout率 | 0.0-0.5 | 0.2 |
| bidirectional | 是否双向 | True/False | 视任务而定 |
| learning_rate | 学习率 | 1e-4-3e-3 | 1e-3 |
| clip_grad | 梯度裁剪阈值 | 0.5-5.0 | 1.0 |
| embed_dim | 词嵌入维度 | 64-512 | 128-256 |
| batch_size | 批大小 | 16-128 | 32-64 |
| max_seq_len | 最大序列长度 | 32-512 | 128 |

### 4.9 变长序列处理

实际应用中序列长度通常不固定，需要使用pack操作高效处理：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def process_variable_length(gru, embeddings, lengths):
    # 处理变长序列
    # gru: GRU模型
    # embeddings: (batch, max_len, embed_dim) 已填充的嵌入
    # lengths: (batch,) 每个序列的实际长度

    # 按长度降序排列
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    sorted_embeddings = embeddings[sorted_idx]

    # Pack: 去除padding部分，只保留有效内容
    packed = pack_padded_sequence(sorted_embeddings, sorted_lengths.cpu(),
                                   batch_first=True, enforce_sorted=True)

    # 通过GRU处理
    packed_output, hidden = gru(packed)

    # Unpack: 恢复为填充后的tensor
    output, _ = pad_packed_sequence(packed_output, batch_first=True)

    # 恢复原始顺序
    _, original_idx = sorted_idx.sort()
    output = output[original_idx]
    hidden = hidden[:, original_idx]

    return output, hidden
```

---

## 5. 应用场景

### 5.1 机器翻译

GRU最初就是为了机器翻译任务而设计的。Cho等人2014年的论文提出了基于GRU的编码器-解码器架构：

- **编码器**：双向GRU将源语言句子编码为固定长度的上下文向量
- **解码器**：单向GRU基于上下文向量逐步生成目标语言单词

GRU在中小规模翻译数据集上表现优秀，且训练速度比LSTM快约20-30%。在实际应用中，结合注意力机制的GRU翻译模型（如Bahdanau Attention + GRU）曾是工业界的主流选择。

### 5.2 文本分类

GRU可以捕获文本中的长距离依赖，适合情感分析、新闻分类、垃圾邮件检测等任务：

- **多对一模式**：将整个序列编码为一个向量（取最后隐藏状态或所有隐藏状态的均值/最大值），然后分类
- **双向GRU**：同时捕获前后文信息，分类效果更好
- **优势**：相比CNN（只能捕获局部n-gram特征），GRU能建模更长的依赖关系

### 5.3 命名实体识别（NER）

NER是一个序列标注任务，每个词都需要一个标签（如人名、地名、机构名等）：

- **多对多模式**：每个时间步都输出一个标签
- **双向GRU + CRF**：双向GRU提取特征，CRF层进行标签约束（如"I-PER"不能出现在"B-LOC"之后）
- GRU在CoNLL-2003等标准NER数据集上表现优秀

### 5.4 时间序列预测

GRU在时间序列任务中的应用同样广泛：

- **金融预测**：股票价格、交易量预测
- **工业应用**：传感器故障检测、设备剩余寿命预测
- **气象预测**：温度、降水量等气象要素预测
- **交通预测**：交通流量、出行时间预测
- GRU的优势在于能够捕捉时间序列中的趋势和周期性模式

### 5.5 语音处理

- **语音识别**：GRU作为声学模型，将声学特征序列映射为音素序列
- **语音合成**：GRU作为解码器，从语言特征生成声学特征
- **说话人识别**：GRU提取语音的时序特征

### 5.6 不适用场景

1. **超长序列**：当序列长度超过数百甚至数千时，Transformer的注意力机制更高效
2. **高并行计算需求**：RNN的顺序计算特性限制了并行化能力
3. **超大规模预训练**：在GPT/BERT规模的任务中，Transformer已成为标准
4. **需要精确的长距离依赖**：某些需要跨越数千个token的依赖的任务，Transformer效果更好

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 详细说明 |
|------|----------|
| 参数量少 | 比LSTM减少约25%的参数量（$3d_h(d_{in}+d_h+1)$ vs $4d_h(d_{in}+d_h+1)$），降低过拟合风险 |
| 训练速度快 | 少一个门操作意味着每个时间步的计算量更少，收敛通常更快20-30% |
| 推理延迟低 | 参数少意味着推理时的内存访问更少，适合实时应用 |
| 效果相当 | 在多数NLP和时序任务上与LSTM性能接近，某些任务上甚至更好 |
| 长期依赖 | 通过更新门的加法连接，能有效学习长期依赖关系 |
| 梯度稳定 | 门控机制缓解梯度消失，比标准RNN稳定得多 |
| 实现简洁 | 只有两个门，代码更简洁，调试更容易 |
| 显式重置 | 重置门允许网络"从头开始"计算，这在LSTM中没有直接对应 |

### 6.2 缺点

| 缺点 | 详细说明 |
|------|----------|
| 顺序计算 | 必须按时间步顺序计算，无法像Transformer那样并行处理整个序列 |
| 长序列瓶颈 | 当序列很长时（数百步以上），梯度传播路径长，信息衰减仍然存在 |
| 表达能力 | 理论上比LSTM稍弱（少一个输出门），某些复杂任务上LSTM可能更优 |
| 信息瓶颈 | 隐藏状态需要同时承担记忆和输出两种角色，可能产生冲突 |
| 缺乏显式记忆 | 没有独立的记忆单元（不像LSTM的cell state），信息存储和读取不够灵活 |

### 6.3 GRU vs LSTM 详细对比表

| 维度 | GRU | LSTM | 胜出 |
|------|-----|------|------|
| 门数量 | 2（更新门+重置门） | 3（遗忘门+输入门+输出门） | GRU（更简洁） |
| 状态数量 | 1（$h_t$） | 2（$c_t$ + $h_t$） | GRU（简洁）/ LSTM（灵活） |
| 参数量 | $3d_h(d_{in}+d_h+1)$ | $4d_h(d_{in}+d_h+1)$ | GRU（少25%） |
| 训练速度 | 快20-30% | 基准 | GRU |
| 小数据集 | 更好（不易过拟合） | 可能过拟合 | GRU |
| 大数据集 | 可能饱和 | 表达能力更强 | LSTM |
| 超长序列 | 稍弱 | 更稳定 | LSTM |
| 实时推理 | 更快 | 较慢 | GRU |
| 代码复杂度 | 低 | 中等 | GRU |
| 调试难度 | 简单 | 较复杂 | GRU |
| 理论分析 | 较少研究 | 大量研究 | LSTM |
| 工业应用 | 广泛 | 广泛 | 平手 |

### 6.4 选择建议

**选择GRU的场景**：
- 数据量有限（<100万样本）
- 计算资源受限（边缘设备、移动端部署）
- 对训练速度和推理延迟有要求
- 序列长度中等（<200步）
- 快速原型验证

**选择LSTM的场景**：
- 数据量充足
- 序列很长（>200步）
- 任务对表达能力要求高
- 需要更精细的记忆控制
- 有充足的计算资源

**选择Transformer的场景**：
- 序列非常长（>500步）
- 需要高度并行化
- 预训练+微调范式
- 大规模数据和算力

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 PyTorch实现：双向GRU文本分类 + 注意力池化

以下是一个完整的GRU文本分类实现，包含双向GRU、注意力池化机制和详细的训练流程：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class AttentionPooling(nn.Module):
    # 注意力池化层：为序列中每个位置计算注意力权重，加权求和得到句子表示
    def __init__(self, hidden_dim):
        super().__init__()
        # 注意力打分函数：将隐藏状态映射为标量分数
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, gru_output, mask=None):
        # gru_output: (batch, seq_len, hidden_dim * num_directions)
        # mask: (batch, seq_len), 1表示有效位置，0表示padding
        # 返回: (batch, hidden_dim * num_directions)

        # 计算注意力分数: (batch, seq_len, 1)
        scores = self.attention(gru_output)

        if mask is not None:
            # 将padding位置的分数设为负无穷
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax得到注意力权重: (batch, seq_len, 1)
        attention_weights = F.softmax(scores, dim=1)

        # 加权求和: (batch, hidden_dim * num_directions)
        weighted_output = torch.sum(attention_weights * gru_output, dim=1)

        return weighted_output, attention_weights.squeeze(-1)


class BiGRUTextClassifier(nn.Module):
    # 双向GRU文本分类器，带注意力池化
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, pretrained_embeddings=None,
                 freeze_embeddings=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2  # 双向

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # 双向GRU编码器
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力池化
        self.attention_pooling = AttentionPooling(hidden_dim * 2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len) 词索引序列
        # mask: (batch, seq_len) 有效位置掩码

        # 词嵌入: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 双向GRU编码: (batch, seq_len, hidden_dim * 2)
        gru_output, _ = self.gru(embedded)

        # 注意力池化: (batch, hidden_dim * 2)
        pooled, attn_weights = self.attention_pooling(gru_output, mask)

        # 分类: (batch, num_classes)
        logits = self.classifier(pooled)

        return logits, attn_weights


def prepare_synthetic_data(num_samples=2000, vocab_size=5000,
                           max_len=64, num_classes=4):
    # 生成合成数据用于演示
    np.random.seed(42)
    # 随机生成词索引序列
    sequences = np.random.randint(1, vocab_size, size=(num_samples, max_len))
    # 随机生成标签
    labels = np.random.randint(0, num_classes, size=num_samples)
    # 构建掩码（模拟变长序列）
    lengths = np.random.randint(10, max_len, size=num_samples)
    masks = np.zeros_like(sequences, dtype=np.float32)
    for i in range(num_samples):
        masks[i, :lengths[i]] = 1.0

    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    m = torch.tensor(masks, dtype=torch.float32)
    return TensorDataset(X, m, y)


def train_gru_classifier():
    # 训练双向GRU文本分类器的完整流程

    # 超参数
    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 128
    num_classes = 4
    num_layers = 2
    dropout = 0.3
    batch_size = 64
    epochs = 15
    learning_rate = 1e-3
    clip_grad = 1.0

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    dataset = prepare_synthetic_data(
        vocab_size=vocab_size, num_classes=num_classes
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 创建模型
    model = BiGRUTextClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # 输出模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_mask, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 前向传播
            logits, attn_weights = model(batch_x, batch_mask)

            # 计算损失
            loss = criterion(logits, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            # 参数更新
            optimizer.step()

            # 统计
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_mask, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_mask = batch_mask.to(device)
                batch_y = batch_y.to(device)

                logits, _ = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = logits.max(1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bigru_classifier.pt')

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model

# 运行训练
# model = train_gru_classifier()
```

### 7.2 GRU序列预测实现

```python
class GRUForecaster(nn.Module):
    # GRU时间序列预测模型
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, hidden = self.gru(x)
        # 取最后一层的隐藏状态
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        prediction = self.fc(last_hidden)  # (batch, output_dim)
        return prediction


def generate_sine_data(num_samples=1000, seq_len=50, freq=1.0):
    # 生成正弦波时间序列数据
    np.random.seed(42)
    t = np.linspace(0, 50 * np.pi, num_samples + seq_len)
    data = np.sin(freq * t) + 0.1 * np.random.randn(len(t))

    X, y = [], []
    for i in range(num_samples):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X.astype(np.float32), y.astype(np.float32)


def train_forecaster():
    # 训练时间序列预测模型
    X, y = generate_sine_data()
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = GRUForecaster(
        input_dim=1, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.1
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train MSE: {total_loss/len(train_loader):.6f} "
                  f"| Val MSE: {val_loss/len(val_loader):.6f}")

# train_forecaster()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 NumPy从零实现GRU单元（含反向传播）

```python
import numpy as np


class GRUCell:
    # 单个GRU单元的纯NumPy实现

    def __init__(self, input_dim, hidden_dim):
        # input_dim: 输入特征维度
        # hidden_dim: 隐藏状态维度
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier初始化
        scale_in = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hid = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        # 重置门参数: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)
        self.W_r = np.random.randn(hidden_dim, input_dim) * scale_in
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * scale_hid
        self.b_r = np.zeros(hidden_dim)

        # 更新门参数: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
        self.W_z = np.random.randn(hidden_dim, input_dim) * scale_in
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * scale_hid
        self.b_z = np.zeros(hidden_dim)

        # 候选隐藏状态参数
        self.W_n = np.random.randn(hidden_dim, input_dim) * scale_in
        self.U_n = np.random.randn(hidden_dim, hidden_dim) * scale_hid
        self.b_n = np.zeros(hidden_dim)

    def sigmoid(self, x):
        # sigmoid激活函数，带数值截断防止溢出
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_t, h_prev):
        # 单步前向传播
        # x_t: (input_dim,) 当前输入
        # h_prev: (hidden_dim,) 前一时刻隐藏状态
        # 返回: h_t, cache（用于反向传播）

        # Step 1: 计算重置门
        r_pre = self.W_r @ x_t + self.U_r @ h_prev + self.b_r
        r_t = self.sigmoid(r_pre)

        # Step 2: 计算更新门
        z_pre = self.W_z @ x_t + self.U_z @ h_prev + self.b_z
        z_t = self.sigmoid(z_pre)

        # Step 3: 计算候选隐藏状态
        n_pre = self.W_n @ x_t + self.U_n @ (r_t * h_prev) + self.b_n
        h_tilde = np.tanh(n_pre)

        # Step 4: 更新隐藏状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        # 缓存前向传播的中间结果，供反向传播使用
        cache = {
            'x_t': x_t, 'h_prev': h_prev, 'h_t': h_t,
            'r_t': r_t, 'z_t': z_t, 'h_tilde': h_tilde,
            'r_pre': r_pre, 'z_pre': z_pre, 'n_pre': n_pre
        }
        return h_t, cache

    def backward(self, dh_t, cache):
        # 单步反向传播
        # dh_t: (hidden_dim,) 从后传来的h_t梯度
        # cache: 前向传播的缓存
        # 返回: dx_t, dh_prev, grads（参数梯度）

        x_t = cache['x_t']
        h_prev = cache['h_prev']
        r_t = cache['r_t']
        z_t = cache['z_t']
        h_tilde = cache['h_tilde']

        # 对 h_t = (1-z_t)*h_prev + z_t*h_tilde 求导
        dz_t = dh_t * (h_tilde - h_prev)
        dh_prev = dh_t * (1 - z_t)
        dh_tilde = dh_t * z_t

        # 对 h_tilde = tanh(n_pre) 求导
        dn_pre = dh_tilde * (1 - h_tilde ** 2)

        # 对 n_pre = W_n @ x_t + U_n @ (r_t * h_prev) + b_n 求导
        dx_t = self.W_n.T @ dn_pre
        d_rh = self.U_n.T @ dn_pre
        dr_t = d_rh * h_prev
        dh_prev += d_rh * r_t

        # 更新门的梯度
        dz_pre = dz_t * z_t * (1 - z_t)
        dx_t += self.W_z.T @ dz_pre
        dh_prev += self.U_z.T @ dz_pre

        # 重置门的梯度
        dr_pre = dr_t * r_t * (1 - r_t)
        dx_t += self.W_r.T @ dr_pre
        dh_prev += self.U_r.T @ dr_pre

        # 参数梯度
        grads = {
            'W_r': np.outer(dr_pre, x_t),
            'U_r': np.outer(dr_pre, h_prev),
            'b_r': dr_pre,
            'W_z': np.outer(dz_pre, x_t),
            'U_z': np.outer(dz_pre, h_prev),
            'b_z': dz_pre,
            'W_n': np.outer(dn_pre, x_t),
            'U_n': np.outer(dn_pre, r_t * h_prev),
            'b_n': dn_pre
        }

        return dx_t, dh_prev, grads
```

### 8.2 NumPy实现多层GRU（含训练）

```python
class MultiLayerGRU:
    # 多层GRU的纯NumPy实现，支持训练

    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=None):
        # input_dim: 输入特征维度
        # hidden_dim: 隐藏状态维度
        # num_layers: GRU层数
        # output_dim: 输出维度（用于分类/回归，None则只返回隐藏状态）
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 创建多层GRU单元
        self.layers = []
        for l in range(num_layers):
            layer_input_dim = input_dim if l == 0 else hidden_dim
            self.layers.append(GRUCell(layer_input_dim, hidden_dim))

        # 输出层（可选）
        if output_dim is not None:
            scale = np.sqrt(2.0 / (hidden_dim + output_dim))
            self.W_out = np.random.randn(output_dim, hidden_dim) * scale
            self.b_out = np.zeros(output_dim)
        else:
            self.W_out = None
            self.b_out = None

    def forward(self, sequence):
        # 完整序列的前向传播
        # sequence: (seq_len, input_dim) 输入序列
        # 返回: outputs, all_hidden, all_caches
        seq_len = len(sequence)

        # 存储每层的隐藏状态和缓存
        all_hidden = np.zeros((self.num_layers, seq_len, self.hidden_dim))
        all_caches = [[] for _ in range(self.num_layers)]

        # 逐层处理
        for l in range(self.num_layers):
            h = np.zeros(self.hidden_dim)

            for t in range(seq_len):
                if l == 0:
                    x_t = sequence[t]
                else:
                    x_t = all_hidden[l - 1, t]  # 上一层的输出作为输入

                h, cache = self.layers[l].forward(x_t, h)
                all_hidden[l, t] = h
                all_caches[l].append(cache)

        # 输出层
        outputs = None
        if self.W_out is not None:
            last_hidden = all_hidden[-1, -1]
            outputs = self.W_out @ last_hidden + self.b_out

        return outputs, all_hidden, all_caches

    def softmax(self, x):
        # softmax函数
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, sequence):
        # 预测类别
        outputs, _, _ = self.forward(sequence)
        if self.W_out is not None:
            probs = self.softmax(outputs)
            return np.argmax(probs), probs
        return None, None

    def compute_loss(self, outputs, targets):
        # 计算交叉熵损失
        probs = self.softmax(outputs)
        loss = -np.log(probs[targets] + 1e-10)
        return loss, probs

    def backward(self, probs, target, all_caches, all_hidden):
        # 完整的反向传播
        seq_len = all_hidden.shape[1]

        # softmax梯度
        d_logits = probs.copy()
        d_logits[target] -= 1.0

        # 输出层梯度
        last_hidden = all_hidden[-1, -1]
        dW_out = np.outer(d_logits, last_hidden)
        db_out = d_logits
        dh_last = self.W_out.T @ d_logits

        # 初始化参数梯度累加器
        param_grads = {}
        for key in ['W_r', 'U_r', 'b_r', 'W_z', 'U_z', 'b_z', 'W_n', 'U_n', 'b_n']:
            param_grads[key] = [np.zeros_like(getattr(self.layers[l], key))
                                for l in range(self.num_layers)]

        # 从最后一层开始反向传播
        for l in reversed(range(self.num_layers)):
            dh = dh_last.copy() if l == self.num_layers - 1 else np.zeros(self.hidden_dim)

            for t in reversed(range(seq_len)):
                cache = all_caches[l][t]
                _, dh_prev, grads = self.layers[l].backward(dh, cache)

                for key in grads:
                    param_grads[key][l] += grads[key]

                dh = dh_prev

        param_grads['W_out'] = dW_out
        param_grads['b_out'] = db_out

        return param_grads

    def update_params(self, grads, lr=0.01):
        # 使用梯度下降更新参数
        for l in range(self.num_layers):
            for key in ['W_r', 'U_r', 'b_r', 'W_z', 'U_z', 'b_z', 'W_n', 'U_n', 'b_n']:
                param = getattr(self.layers[l], key)
                param -= lr * grads[key][l]

        if self.W_out is not None:
            self.W_out -= lr * grads['W_out']
            self.b_out -= lr * grads['b_out']


def train_manual_gru():
    # 使用手动实现的GRU进行训练演示
    np.random.seed(42)

    # 超参数
    input_dim = 8
    hidden_dim = 16
    num_layers = 2
    output_dim = 3
    seq_len = 10
    num_samples = 500
    epochs = 20
    lr = 0.01
    clip_value = 5.0

    # 生成模拟数据
    X_data = np.random.randn(num_samples, seq_len, input_dim).astype(np.float32)
    y_data = np.random.randint(0, output_dim, num_samples)

    # 创建模型
    model = MultiLayerGRU(input_dim, hidden_dim, num_layers, output_dim)

    # 训练循环
    print("开始训练手动实现的GRU...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        indices = np.random.permutation(num_samples)

        for idx in indices:
            sequence = X_data[idx]
            target = y_data[idx]

            # 前向传播
            outputs, all_hidden, all_caches = model.forward(sequence)
            loss, probs = model.compute_loss(outputs, target)

            # 反向传播
            param_grads = model.backward(probs, target, all_caches, all_hidden)

            # 梯度裁剪
            for key in param_grads:
                if isinstance(param_grads[key], list):
                    for g in param_grads[key]:
                        np.clip(g, -clip_value, clip_value, out=g)
                else:
                    np.clip(param_grads[key], -clip_value, clip_value,
                            out=param_grads[key])

            # 参数更新
            model.update_params(param_grads, lr)

            total_loss += loss
            pred = np.argmax(probs)
            if pred == target:
                correct += 1

        avg_loss = total_loss / num_samples
        accuracy = correct / num_samples

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    # 测试前向传播
    test_seq = np.random.randn(seq_len, input_dim).astype(np.float32)
    outputs, hidden_states, _ = model.forward(test_seq)
    pred_class, pred_probs = model.predict(test_seq)

    print(f"\n测试样本预测:")
    print(f"  隐藏状态形状: {hidden_states.shape}")
    print(f"  预测类别: {pred_class}")
    print(f"  预测概率: {pred_probs}")

    return model

# 运行训练
# trained_model = train_manual_gru()
```

### 8.3 手动GRU前向传播演示（带具体数值）

```python
def manual_gru_forward_demo():
    # 手动计算GRU的前向传播，展示每一步的具体数值
    np.random.seed(42)

    input_dim = 3
    hidden_dim = 4

    # 手动设定参数以便验证
    W_r = np.array([[0.1, 0.2, 0.3],
                     [0.4, 0.5, 0.6],
                     [0.7, 0.8, 0.9],
                     [1.0, 1.1, 1.2]])
    U_r = np.eye(4) * 0.1
    b_r = np.zeros(4)

    W_z = np.array([[0.2, 0.3, 0.4],
                     [0.5, 0.6, 0.7],
                     [0.8, 0.9, 1.0],
                     [1.1, 1.2, 1.3]])
    U_z = np.eye(4) * 0.2
    b_z = np.zeros(4)

    W_n = np.array([[0.3, 0.4, 0.5],
                     [0.6, 0.7, 0.8],
                     [0.9, 1.0, 1.1],
                     [1.2, 1.3, 1.4]])
    U_n = np.eye(4) * 0.3
    b_n = np.zeros(4)

    # 输入和初始隐藏状态
    x_t = np.array([0.5, -0.3, 0.8])
    h_prev = np.array([0.1, 0.2, 0.1, 0.0])

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Step 1: 重置门
    r_pre = W_r @ x_t + U_r @ h_prev + b_r
    r_t = sigmoid(r_pre)
    print(f"=== GRU前向传播演示 ===")
    print(f"输入 x_t = {x_t}")
    print(f"前隐藏状态 h_prev = {h_prev}")
    print(f"\nStep 1: 重置门")
    print(f"  r_pre = W_r @ x_t + U_r @ h_prev = {r_pre}")
    print(f"  r_t = sigmoid(r_pre) = {r_t}")

    # Step 2: 更新门
    z_pre = W_z @ x_t + U_z @ h_prev + b_z
    z_t = sigmoid(z_pre)
    print(f"\nStep 2: 更新门")
    print(f"  z_pre = W_z @ x_t + U_z @ h_prev = {z_pre}")
    print(f"  z_t = sigmoid(z_pre) = {z_t}")

    # Step 3: 候选隐藏状态
    r_times_h = r_t * h_prev
    n_pre = W_n @ x_t + U_n @ r_times_h + b_n
    h_tilde = np.tanh(n_pre)
    print(f"\nStep 3: 候选隐藏状态")
    print(f"  r_t * h_prev = {r_times_h}")
    print(f"  n_pre = W_n @ x_t + U_n @ (r_t * h_prev) = {n_pre}")
    print(f"  h_tilde = tanh(n_pre) = {h_tilde}")

    # Step 4: 隐藏状态更新
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    print(f"\nStep 4: 隐藏状态更新")
    print(f"  (1 - z_t) * h_prev = {(1 - z_t) * h_prev}")
    print(f"  z_t * h_tilde = {z_t * h_tilde}")
    print(f"  h_t = {h_t}")
    print(f"\n最终隐藏状态: {h_t}")

# manual_gru_forward_demo()
```

---

## 9. 可视化与结果理解

### 9.1 门控值变化可视化

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_gate_activations():
    # 可视化GRU各门的激活模式
    x = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Sigmoid函数 -- 门控函数
    sigmoid = 1 / (1 + np.exp(-x))
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2, label=r'$\sigma(x)$')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].fill_between(x, 0, sigmoid, alpha=0.1, color='blue')
    axes[0, 0].annotate('z=0: 完全关闭(遗忘)', xy=(-3, 0.05),
                        fontsize=9, color='red', fontweight='bold')
    axes[0, 0].annotate('z=1: 完全打开(保留)', xy=(3, 0.95),
                        fontsize=9, color='green', fontweight='bold')
    axes[0, 0].set_xlabel('输入值 z')
    axes[0, 0].set_ylabel(r'$\sigma(z)$')
    axes[0, 0].set_title('(a) Sigmoid函数 -- GRU门控的基础')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # (b) Tanh函数 -- 候选状态
    tanh = np.tanh(x)
    axes[0, 1].plot(x, tanh, 'r-', linewidth=2, label=r'$\tanh(x)$')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(x, 0, tanh, where=(tanh > 0), alpha=0.1, color='red')
    axes[0, 1].fill_between(x, 0, tanh, where=(tanh < 0), alpha=0.1, color='blue')
    axes[0, 1].set_xlabel('输入值 z')
    axes[0, 1].set_ylabel(r'$\tanh(z)$')
    axes[0, 1].set_title('(b) Tanh函数 -- 候选状态激活')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # (c) 更新门在不同z值下的信息保留比例
    z_values = np.linspace(0, 1, 100)
    old_ratio = 1 - z_values
    new_ratio = z_values
    axes[1, 0].plot(z_values, old_ratio, 'b-', linewidth=2,
                    label='旧状态比例 $(1-z_t)$')
    axes[1, 0].plot(z_values, new_ratio, 'r-', linewidth=2,
                    label='新候选比例 $z_t$')
    axes[1, 0].fill_between(z_values, 0, old_ratio, alpha=0.1, color='blue')
    axes[1, 0].fill_between(z_values, 0, new_ratio, alpha=0.1, color='red')
    axes[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].annotate('长期记忆\n(保留旧状态)', xy=(0.1, 0.9),
                        fontsize=9, color='blue', fontweight='bold')
    axes[1, 0].annotate('快速更新\n(采纳新信息)', xy=(0.7, 0.8),
                        fontsize=9, color='red', fontweight='bold')
    axes[1, 0].set_xlabel(r'更新门 $z_t$')
    axes[1, 0].set_ylabel('信息比例')
    axes[1, 0].set_title(r'(c) 更新门控制的信息流动')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # (d) 重置门对候选状态的影响
    r_values = np.linspace(0, 1, 100)
    history_contribution = r_values
    axes[1, 1].plot(r_values, history_contribution, 'g-', linewidth=2,
                    label=r'历史贡献 $r_t \cdot h_{t-1}$')
    axes[1, 1].plot(r_values, np.ones_like(r_values) * 0.8, 'purple',
                    linewidth=2, linestyle='--',
                    label='当前输入贡献 (恒定)')
    axes[1, 1].fill_between(r_values, 0, history_contribution,
                             alpha=0.1, color='green')
    axes[1, 1].annotate('r_t=0: 完全忽略历史\n候选只基于当前输入',
                        xy=(0.05, 0.2), fontsize=9, color='red')
    axes[1, 1].annotate('r_t=1: 完全使用历史\n候选结合历史+输入',
                        xy=(0.7, 0.9), fontsize=9, color='green')
    axes[1, 1].set_xlabel(r'重置门 $r_t$')
    axes[1, 1].set_ylabel('贡献度')
    axes[1, 1].set_title(r'(d) 重置门对候选状态的影响')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('gru_gate_activations.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_gate_activations()
```

### 9.2 隐藏状态轨迹可视化

```python
def visualize_hidden_state_trajectory():
    # 可视化GRU隐藏状态随时间的演变轨迹
    np.random.seed(42)

    time_steps = 60
    hidden_dim = 32

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 场景1: 稳定记忆（更新门z_t接近0）
    h_stable = np.zeros((time_steps, hidden_dim))
    h = np.random.randn(hidden_dim) * 0.5
    for t in range(time_steps):
        z = 0.05 + 0.02 * np.random.randn()
        h = (1 - z) * h + z * np.random.randn(hidden_dim) * 0.1
        h_stable[t] = h

    im1 = axes[0].imshow(h_stable.T, aspect='auto', cmap='RdBu_r',
                          interpolation='nearest', vmin=-2, vmax=2)
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('隐藏维度')
    axes[0].set_title('场景1: 稳定记忆模式 (更新门z_t接近0)')
    plt.colorbar(im1, ax=axes[0], label='激活值')

    # 场景2: 快速更新（更新门z_t接近1）
    h_fast = np.zeros((time_steps, hidden_dim))
    h = np.zeros(hidden_dim)
    for t in range(time_steps):
        z = 0.9 + 0.05 * np.random.randn()
        h = (1 - z) * h + z * np.random.randn(hidden_dim) * 0.5
        h_fast[t] = h

    im2 = axes[1].imshow(h_fast.T, aspect='auto', cmap='RdBu_r',
                          interpolation='nearest', vmin=-2, vmax=2)
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('隐藏维度')
    axes[1].set_title('场景2: 快速更新模式 (更新门z_t接近1)')
    plt.colorbar(im2, ax=axes[1], label='激活值')

    # 场景3: 自适应（更新门根据内容动态变化）
    h_adaptive = np.zeros((time_steps, hidden_dim))
    h = np.zeros(hidden_dim)
    for t in range(time_steps):
        if t < 20:
            z = 0.1
        elif t < 30:
            z = 0.9
        else:
            z = 0.15
        z += 0.03 * np.random.randn()
        z = np.clip(z, 0, 1)
        h = (1 - z) * h + z * np.random.randn(hidden_dim) * 0.5
        h_adaptive[t] = h

    im3 = axes[2].imshow(h_adaptive.T, aspect='auto', cmap='RdBu_r',
                          interpolation='nearest', vmin=-2, vmax=2)
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('隐藏维度')
    axes[2].set_title('场景3: 自适应模式 (更新门动态变化)')
    plt.colorbar(im3, ax=axes[2], label='激活值')
    axes[2].axvspan(0, 20, alpha=0.1, color='blue', label='稳定记忆')
    axes[2].axvspan(20, 30, alpha=0.1, color='red', label='快速更新')
    axes[2].axvspan(30, 60, alpha=0.1, color='blue')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('gru_hidden_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_hidden_state_trajectory()
```

### 9.3 结果理解指南

理解GRU的可视化结果时，需要关注以下几点：

1. **门激活图（图a-b）**：Sigmoid将输入映射到(0,1)，0代表完全关闭（遗忘/忽略），1代表完全打开（保留/通过）。Tanh将候选状态限制在(-1,1)范围内，防止数值爆炸。

2. **更新门信息流动（图c）**：当更新门 $z_t$ 接近0时，旧状态被完整保留（长期记忆模式）；当 $z_t$ 接近1时，新候选被完整采纳（快速更新模式）。

3. **重置门影响（图d）**：重置门控制候选状态中历史信息的贡献度。$r_t = 0$ 时，候选状态完全基于当前输入重新计算，实现"话题切换"效果。

4. **隐藏状态轨迹（三个场景）**：颜色深浅代表不同维度在不同时刻的激活程度。稳定模式下颜色均匀（信息保持不变），快速更新模式下颜色频繁变化。

---

## 10. 模型评估

### 10.1 分类任务评估

```python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_classifier(model, dataloader, device, num_classes=None):
    # 全面评估分类模型
    # 返回: 指标字典
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, *rest in dataloader:
            batch_y = rest[-1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            mask = rest[0].to(device) if len(rest) > 1 else None

            if mask is not None:
                logits, _ = model(batch_x, mask)
            else:
                logits = model(batch_x)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds,
                                           average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds,
                                     average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds,
                             average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds,
                                average='weighted', zero_division=0),
    }

    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

    if num_classes:
        target_names = [f'Class {i}' for i in range(num_classes)]
        metrics['classification_report'] = classification_report(
            all_labels, all_preds, target_names=target_names, zero_division=0
        )

    print("=" * 50)
    print("模型评估结果")
    print("=" * 50)
    print(f"准确率 (Accuracy):        {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision-macro):  {metrics['precision_macro']:.4f}")
    print(f"召回率 (Recall-macro):     {metrics['recall_macro']:.4f}")
    print(f"F1分数 (F1-macro):         {metrics['f1_macro']:.4f}")
    print(f"F1分数 (F1-weighted):      {metrics['f1_weighted']:.4f}")
    if 'classification_report' in metrics:
        print(f"\n详细分类报告:")
        print(metrics['classification_report'])
    print(f"\n混淆矩阵:")
    print(metrics['confusion_matrix'])

    return metrics
```

### 10.2 序列生成任务评估

```python
def compute_perplexity(model, dataloader, device):
    # 计算语言模型的困惑度（Perplexity）
    # PPL = exp(average cross-entropy loss)
    # PPL越低表示模型越好
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = model(batch_x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1)
            )
            total_loss += loss.item()
            total_tokens += batch_y.numel()

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    print(f"平均交叉熵损失: {avg_loss:.4f}")
    print(f"困惑度 (PPL): {ppl:.2f}")
    return {'loss': avg_loss, 'perplexity': ppl}
```

### 10.3 回归任务评估

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regressor(model, dataloader, device):
    # 评估时间序列回归模型
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print("=" * 50)
    print("回归模型评估结果")
    print("=" * 50)
    print(f"MSE  (均方误差):     {mse:.6f}")
    print(f"RMSE (均方根误差):   {rmse:.6f}")
    print(f"MAE  (平均绝对误差): {mae:.6f}")
    print(f"R2   (决定系数):     {r2:.6f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
```

---

## 11. 常见问题与易错点

### 11.1 GRU什么时候不如LSTM？

以下场景中LSTM可能更优：

1. **超长序列建模**：当序列长度超过数百步时，LSTM的独立记忆单元能提供更稳定的信息存储。
2. **需要精细记忆控制**：LSTM的输出门可以精确控制从记忆单元输出多少信息。
3. **需要精确计数能力**：某些需要精确计数的任务，LSTM通常表现更好。
4. **大数据集场景**：数据量非常大时，LSTM更多的参数反而能提供更强的拟合能力。

### 11.2 多层GRU的层数选择

| 层数 | 适用场景 | 注意事项 |
|------|----------|----------|
| 1层 | 简单任务、小数据集 | 容易训练，不易过拟合 |
| 2层 | 大多数NLP任务 | 推荐作为默认选择 |
| 3层 | 复杂任务、大数据集 | 需要更多数据和正则化 |
| 4层+ | 极少使用 | 通常不如增加hidden_dim |

增加层数的一般原则：
- 每增加一层，hidden_dim可以减半（如2层256 vs 4层128）
- 层数增加时必须使用dropout
- 超过3层后收益递减，但训练难度增加

### 11.3 隐藏状态初始化问题

```python
# 方案1: 零初始化（最常用，PyTorch默认）
hidden = torch.zeros(num_layers, batch_size, hidden_dim).to(device)

# 方案2: 可学习初始化
class GRUWithLearnableInit(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def get_initial_hidden(self, batch_size):
        return self.h0.expand(-1, batch_size, -1).contiguous()
```

### 11.4 梯度爆炸与梯度裁剪

```python
# 方法1：按范数裁剪（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法2：按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 11.5 双向GRU的隐藏状态合并

```python
# PyTorch中双向GRU的hidden排列:
# hidden[0]: 第1层 前向
# hidden[1]: 第1层 后向
# hidden[2]: 第2层 前向
# hidden[3]: 第2层 后向

# 正确合并方式: 取最后一层的前向和后向
last_forward = hidden[-2]   # 最后一层前向
last_backward = hidden[-1]  # 最后一层后向
combined = torch.cat([last_forward, last_backward], dim=1)
```

### 11.6 变长序列的Padding处理

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def handle_variable_length(gru, embeddings, lengths):
    # 正确处理变长序列
    packed = pack_padded_sequence(embeddings, lengths.cpu(),
                                   batch_first=True, enforce_sorted=True)
    packed_output, hidden = gru(packed)
    output, _ = pad_packed_sequence(packed_output, batch_first=True)
    return output, hidden
```

### 11.7 过拟合与欠拟合

**过拟合症状**：训练损失下降但验证损失上升。

**解决方案**：
- 增加Dropout（GRU层间+嵌入层）
- 减小模型规模（hidden_dim、num_layers）
- 使用权重 decay
- 数据增强（随机删除词、同义词替换）
- 早停（Early Stopping）

**欠拟合症状**：训练和验证损失都很高。

**解决方案**：
- 增加模型容量（增大hidden_dim、增加层数）
- 减小正则化（减小dropout和weight_decay）
- 使用预训练词向量
- 增加训练轮数
- 调整学习率

---

## 12. 学习总结

### 12.1 核心要点回顾

1. **两个门控机制**：GRU通过重置门 $r_t$ 和更新门 $z_t$ 控制信息流动，重置门决定遗忘多少过去，更新门决定保留多少过去 vs 接受多少新信息。

2. **候选隐藏状态**：$\tilde{h}_t$ 基于当前输入和（被重置门过滤的）历史信息生成，是可能成为新隐藏状态的候选值。

3. **凸组合更新**：$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$，旧状态和新候选的加权平均，确保数值稳定。

4. **与LSTM的等价性**：GRU本质上用2个门和1个状态实现了LSTM的3个门和2个状态的功能，参数量减少25%，但表达力损失很小。

5. **梯度直通路径**：当 $z_t \approx 0$ 时，$\frac{\partial h_t}{\partial h_{t-1}} \approx I$，梯度可以直接流过，有效缓解梯度消失。

6. **重置门的独特价值**：GRU的重置门允许候选状态"从头开始"计算，这是LSTM所没有的机制。

### 12.2 GRU公式汇总

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

$$\tilde{h}_t = \tanh(W_n x_t + U_n(r_t \odot h_{t-1}) + b_n)$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 12.3 关键洞察

1. GRU的设计哲学是"足够好而非最好" -- 用最少的门控实现足够的信息控制能力。
2. 更新门的互补性 $(1-z_t) + z_t = 1$ 是一种隐式的归一化，有助于训练稳定性。
3. 重置门让GRU能够处理"不连续"的序列模式（如话题切换）。
4. 在Transformer出现之前，GRU是序列建模的最佳平衡点之一。

---

## 13. 练习题与思考题（含答案）

### 练习题1：GRU门控机制边界分析

**题目**：分析GRU在以下四种极端情况下的行为：
- (a) $z_t = \mathbf{0}$，$r_t = \mathbf{1}$
- (b) $z_t = \mathbf{1}$，$r_t = \mathbf{1}$
- (c) $z_t = \mathbf{0}$，$r_t = \mathbf{0}$
- (d) $z_t = \mathbf{1}$，$r_t = \mathbf{0}$

<details>
<summary>答案</summary>

**(a) $z_t = \mathbf{0}$，$r_t = \mathbf{1}$**：

$$h_t = (1-\mathbf{0}) \odot h_{t-1} + \mathbf{0} \odot \tilde{h}_t = h_{t-1}$$

含义：完全保留旧状态（长期记忆模式），信息无损失地传递到未来。

**(b) $z_t = \mathbf{1}$，$r_t = \mathbf{1}$**：

$$h_t = \tilde{h}_t = \tanh(W_n x_t + U_n h_{t-1} + b_n)$$

含义：完全更新为新候选，且候选状态同时使用了当前输入和完整历史。等价于标准RNN。

**(c) $z_t = \mathbf{0}$，$r_t = \mathbf{0}$**：

$$h_t = h_{t-1}$$
$$\tilde{h}_t = \tanh(W_n x_t + b_n)$$

含义：状态不变，候选状态只基于当前输入。模型完全"记住"之前的状态。

**(d) $z_t = \mathbf{1}$，$r_t = \mathbf{0}$**：

$$h_t = \tilde{h}_t = \tanh(W_n x_t + b_n)$$

含义：完全更新，且新状态只基于当前输入（完全忽略历史）。"重新开始"模式，适合话题切换。

</details>

### 练习题2：参数量计算

**题目**：给定 $d_{in} = 300$，$d_h = 512$，计算：
- (a) 单层单向GRU的参数量
- (b) 单层双向GRU的参数量
- (c) 3层单向GRU的参数量
- (d) 对比同等配置下的LSTM参数量

<details>
<summary>答案</summary>

**(a) 单层单向GRU**：
$$3 \times 512 \times (300 + 512 + 1) = 3 \times 512 \times 813 = 1,248,768$$

**(b) 单层双向GRU**：
$$2 \times 1,248,768 = 2,497,536$$

**(c) 3层单向GRU**：
$$1,248,768 + 2 \times 3 \times 512 \times 1025 = 1,248,768 + 3,145,728 = 4,394,496$$

**(d) LSTM对比**：
- 单层单向LSTM：$4 \times 512 \times 813 = 1,665,024$（比GRU多约33.3%）
- 3层单向LSTM：$1,665,024 + 2 \times 4 \times 512 \times 1025 = 5,859,328$（比GRU多 $1,464,832$）

</details>

### 练习题3：手动GRU前向计算

**题目**：给定以下参数和输入，手动计算一个时间步的GRU前向传播：

- $x_t = [1, 0, -1]$，$h_{t-1} = [0.5, -0.5, 0, 0.3]$（$d_{in}=3, d_h=4$）
- $W_r = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.2 & 0.1 & 0.2 \\ 0.3 & 0.3 & 0.1 \\ 0.1 & 0.2 & 0.3 \end{pmatrix}$，$U_r = 0.1 I_4$，$b_r = \mathbf{0}$
- $W_z = \begin{pmatrix} 0.2 & 0.1 & 0.2 \\ 0.1 & 0.2 & 0.1 \\ 0.2 & 0.1 & 0.2 \\ 0.1 & 0.2 & 0.1 \end{pmatrix}$，$U_z = 0.1 I_4$，$b_z = \mathbf{0}$
- $W_n = \begin{pmatrix} 0.3 & 0.2 & 0.1 \\ 0.2 & 0.3 & 0.2 \\ 0.1 & 0.2 & 0.3 \\ 0.3 & 0.1 & 0.2 \end{pmatrix}$，$U_n = 0.2 I_4$，$b_n = \mathbf{0}$

<details>
<summary>答案</summary>

**Step 1: 重置门**

$$W_r x_t = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.2 & 0.1 & 0.2 \\ 0.3 & 0.3 & 0.1 \\ 0.1 & 0.2 & 0.3 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} = \begin{pmatrix} -0.2 \\ 0.0 \\ 0.2 \\ -0.2 \end{pmatrix}$$

$$U_r h_{t-1} = 0.1 \times \begin{pmatrix} 0.5 \\ -0.5 \\ 0 \\ 0.3 \end{pmatrix} = \begin{pmatrix} 0.05 \\ -0.05 \\ 0 \\ 0.03 \end{pmatrix}$$

$$r_{pre} = W_r x_t + U_r h_{t-1} = \begin{pmatrix} -0.15 \\ -0.05 \\ 0.2 \\ -0.17 \end{pmatrix}$$

$$r_t = \sigma(r_{pre}) \approx \begin{pmatrix} 0.463 \\ 0.488 \\ 0.550 \\ 0.458 \end{pmatrix}$$

**Step 2: 更新门**

$$W_z x_t = \begin{pmatrix} 0.2 & 0.1 & 0.2 \\ 0.1 & 0.2 & 0.1 \\ 0.2 & 0.1 & 0.2 \\ 0.1 & 0.2 & 0.1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 0.0 \\ 0.0 \\ 0.0 \\ 0.0 \end{pmatrix}$$

$$z_{pre} = W_z x_t + U_z h_{t-1} = \begin{pmatrix} 0.05 \\ -0.05 \\ 0 \\ 0.03 \end{pmatrix}$$

$$z_t = \sigma(z_{pre}) \approx \begin{pmatrix} 0.512 \\ 0.488 \\ 0.500 \\ 0.507 \end{pmatrix}$$

**Step 3: 候选隐藏状态**

$$r_t \odot h_{t-1} \approx \begin{pmatrix} 0.232 \\ -0.244 \\ 0 \\ 0.137 \end{pmatrix}$$

$$n_{pre} = W_n x_t + U_n(r_t \odot h_{t-1}) = \begin{pmatrix} 0.246 \\ -0.049 \\ -0.2 \\ 0.127 \end{pmatrix}$$

$$\tilde{h}_t = \tanh(n_{pre}) \approx \begin{pmatrix} 0.241 \\ -0.049 \\ -0.197 \\ 0.126 \end{pmatrix}$$

**Step 4: 隐藏状态更新**

$$(1-z_t) \odot h_{t-1} \approx \begin{pmatrix} 0.244 \\ -0.256 \\ 0 \\ 0.148 \end{pmatrix}$$

$$z_t \odot \tilde{h}_t \approx \begin{pmatrix} 0.123 \\ -0.024 \\ -0.099 \\ 0.064 \end{pmatrix}$$

$$h_t \approx \begin{pmatrix} 0.367 \\ -0.280 \\ -0.099 \\ 0.212 \end{pmatrix}$$

</details>

### 练习题4：梯度传播分析

**题目**：假设一个2层GRU，序列长度 $T=10$，输入维度 $d_{in}=64$，隐藏维度 $d_h=128$。

- (a) 画出从损失 $L$ 到输入 $x_1$ 的梯度传播路径
- (b) 为什么GRU的梯度消失问题比标准RNN轻？
- (c) 如果在第5个时间步 $z_5 \approx 0$，解释这对梯度从 $h_{10}$ 到 $h_4$ 的传播有什么影响

<details>
<summary>答案</summary>

**(a) 梯度传播路径**：

$$L \to h_T^{(2)} \to h_{T-1}^{(2)} \to \cdots \to h_1^{(2)} \to h_T^{(1)} \to h_{T-1}^{(1)} \to \cdots \to h_1^{(1)} \to x_1$$

总路径长度 = $T + T = 20$ 步（两个10步的链）。

**(b) GRU梯度消失更轻的原因**：

关键在于 $\frac{\partial h_t}{\partial h_{t-1}}$ 中包含 $\text{diag}(1 - z_t)$。当 $z_t \approx 0$ 时，该项接近单位矩阵 $I$，梯度可以直接流过。标准RNN中 $(1-h_t^2) < 1$ 导致连乘后梯度指数衰减。

**(c) $z_5 \approx 0$ 的影响**：

当 $z_5 \approx 0$ 时，$h_5 \approx h_4$，$\frac{\partial h_5}{\partial h_4} \approx I$。从 $h_{10}$ 到 $h_5$ 的梯度在经过第5步时几乎无衰减地传递到 $h_4$，即使 $h_6, \ldots, h_{10}$ 的时间步有些衰减，$h_5$ 作为一个"梯度保持节点"确保了更早信息的梯度不会完全消失。

</details>

### 练习题5：实现带有残差连接的GRU

**题目**：实现一个带有残差连接的GRU层，当输入维度等于隐藏维度时直接相加，当维度不同时使用线性投影。

<details>
<summary>答案</summary>

```python
class ResidualGRU(nn.Module):
    # 带残差连接的GRU层
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 残差投影（当输入维度 != 隐藏维度时）
        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        gru_output, hidden = self.gru(x)
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        output = self.layer_norm(gru_output + residual)
        return output, hidden


class DeepResidualGRU(nn.Module):
    # 多层残差GRU
    def __init__(self, input_dim, hidden_dim, num_blocks=3,
                 num_layers_per_block=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_input = input_dim if i == 0 else hidden_dim
            self.blocks.append(
                ResidualGRU(block_input, hidden_dim,
                            num_layers_per_block, dropout)
            )

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x
```

</details>

### 思考题1：GRU能否完全替代LSTM？

<details>
<summary>答案</summary>

不能完全替代。虽然在多数任务上性能相当，但以下方面LSTM仍有优势：

1. **理论表达能力**：LSTM有独立的状态单元和输出门，理论上能实现更复杂的信息控制策略。
2. **超长序列**：LSTM的cell state提供了一条更稳定的梯度通路。
3. **计数能力**：LSTM的记忆单元可以在不经过tanh的情况下直接传递，能更稳定地保持计数值。
4. **研究生态**：LSTM有更多变体（如Peephole LSTM、Grid LSTM等）。
5. **历史原因**：LSTM有近20年的研究积累。

但从实用角度：如果数据量和计算资源不是特别充裕，或者任务不是特别复杂，GRU通常是更好的默认选择。

</details>

### 思考题2：残差形式的更新公式

**题目**：如果将GRU的更新门公式改为 $h_t = h_{t-1} + z_t \odot (\tilde{h}_t - h_{t-1})$，这与原公式等价吗？

<details>
<summary>答案</summary>

**等价性证明**：

$$h_t = h_{t-1} - z_t \odot h_{t-1} + z_t \odot \tilde{h}_t = h_{t-1} + z_t \odot (\tilde{h}_t - h_{t-1})$$

两者完全等价。

**这种形式的优势**：

1. **直觉更清晰**：理解为"在旧状态基础上加上门控变化量"，与残差网络思想一致。
2. **梯度分析更直观**：$\frac{\partial h_t}{\partial h_{t-1}} = I - z_t + \text{（其他项）}$，当 $z_t = 0$ 时梯度完全通过。
3. **数值稳定性**：残差形式基于差值计算，数值误差更小。
4. **与ResNet的联系**：揭示了GRU与Highway Network、ResNet之间的深层联系。

</details>

---

## 14. 学习路径建议

### 14.1 总体学习路径

```
RNN基础
  |-- 理解循环结构、隐藏状态、时间展开
  |-- 理解梯度消失/爆炸问题
  |
  v
LSTM
  |-- 理解3个门 + 记忆单元
  |-- 掌握LSTM的训练和调参
  |
  v
GRU (当前)
  |-- 理解2个门如何实现3个门的功能
  |-- 掌握GRU vs LSTM的选择策略
  |
  v
进阶RNN架构
  |-- 双向RNN / GRU
  |-- 深层RNN / 残差连接
  |-- Attention + GRU
  |
  v
Transformer
  |-- 自注意力机制
  |-- 多头注意力
  |-- 位置编码
  |-- 理解为什么Transformer替代了RNN
```

### 14.2 初级阶段（1-2周）

**目标**：理解GRU的基本原理，能用PyTorch实现简单任务。

1. 复习RNN基础和梯度消失问题（1天）
2. 学习GRU的两个门机制（1天）
3. 手动计算一个GRU时间步的前向传播（1天）
4. 使用PyTorch nn.GRU实现文本分类（2天）
5. 理解BPTT和梯度裁剪（1天）
6. 完成一个小项目：IMDb情感分析（2天）

**推荐资源**：
- Cho et al. (2014) 原始论文
- PyTorch官方GRU文档和教程
- Christopher Olah的经典博客文章

### 14.3 中级阶段（2-3周）

**目标**：深入理解GRU的数学原理，能手动实现并调优。

1. 手动实现GRU单元的前向传播（2天）
2. 实现GRU的反向传播（BPTT）（3天）
3. 深入分析GRU vs LSTM的梯度传播差异（2天）
4. 实现双向GRU + 注意力池化（2天）
5. 实现GRU Seq2Seq模型（3天）
6. 学习变长序列处理（pack_padded_sequence）（1天）
7. 完成一个项目：GRU机器翻译（3天）

**推荐资源**：
- Chung et al. (2014) "Empirical Evaluation of Gated Recurrent Neural Networks"
- Jozefowicz et al. (2015) "An Empirical Exploration of Recurrent Network Architectures"
- Goodfellow et al.《Deep Learning》第10章

### 14.4 高级阶段（3-4周）

**目标**：掌握GRU的各种变体和高级应用。

1. 实现GRU语言模型 + 困惑度评估（3天）
2. 实现Attention + GRU的编码器-解码器（3天）
3. 实现多层GRU + 残差连接（2天）
4. 研究GRU变体（MinimalRNN、JANET等）（3天）
5. 将GRU与Transformer进行对比实验（3天）
6. 完成一个挑战项目：带注意力的GRU文本摘要（5天）

### 14.5 实践项目建议

| 项目 | 难度 | 涉及技能 | 预计时间 |
|------|------|----------|----------|
| IMDb情感分析 | 入门 | 单向GRU + 分类 | 1-2天 |
| 新闻分类（多分类） | 入门 | 双向GRU + 注意力 | 2-3天 |
| 正弦波预测 | 入门 | GRU回归 + 时间序列 | 1-2天 |
| 命名实体识别 | 中级 | 双向GRU + CRF | 3-5天 |
| 机器翻译 | 中级 | GRU Seq2Seq + Attention | 5-7天 |
| 文本生成 | 中级 | GRU语言模型 + 采样策略 | 3-5天 |
| 语音命令识别 | 中级 | GRU + 音频特征 | 5-7天 |
| 带注意力的文本摘要 | 高级 | GRU编码器-解码器 + 全局注意力 | 7-10天 |
| 对话生成 | 高级 | GRU Seq2Seq + 上下文管理 | 7-10天 |

### 14.6 推荐资源

**论文**：
- Cho, K., Van Merrienboer, B., Gulcehre, C., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). "An Empirical Exploration of Recurrent Network Architectures"

**教程与代码**：
- PyTorch官方文档：torch.nn.GRU
- Andrej Karpathy的"Minimal Character-Level RNN"（用GRU扩展）
- Hugging Face Transformers中的GRU相关实现

**书籍**：
- Goodfellow, Bengio, Courville《Deep Learning》第10章（Recurrent Neural Networks）
- Zhang et al.《Dive into Deep Learning》RNN章节

**延伸学习**：
- 从GRU到Transformer：理解自注意力机制如何替代循环结构
- 现代RNN变体：RWKV、Mamba等线性复杂度的序列模型
- GRU在非NLP领域的应用：视频理解、图神经网络等
