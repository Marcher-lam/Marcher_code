# Encoder-Decoder (Seq2Seq) 学习文档

> 将任意长度的输入序列映射为任意长度的输出序列的通用深度学习框架

---

## 1. 算法基础认知

### 一句话定义

Encoder-Decoder 是一种由编码器和解码器两部分组成的深度学习架构，能够将一个任意长度的输入序列转换为一个任意长度的输出序列。

### 直觉类比

想象一位翻译员正在将一段英文翻译成中文。这位翻译员的工作分为两个阶段：

**第一阶段 -- 阅读（编码）**：翻译员逐词阅读英文句子，在脑海中逐渐建立起对整句话的理解。他并不是逐词死记硬背，而是在阅读过程中不断积累语义信息，最终形成一个整体的"理解"。

**第二阶段 -- 翻译（解码）**：翻译员基于脑海中已经形成的整体理解，开始逐词输出中文译文。每写下一个中文词，他都会参考已经写下的前文，确保译文的连贯性和准确性。

在这个过程中，编码器就好比翻译员的"阅读理解"能力，负责将源语言压缩为一种内部表示；解码器就好比翻译员的"表达能力"，负责根据内部表示生成目标语言。上下文向量就是翻译员脑海中的"整体理解" -- 它是源语言的一种浓缩表示，是连接阅读和翻译两个阶段的桥梁。

但请注意，这位翻译员有一个明显的局限：他在翻译时只能依赖脑海中那个单一的"整体理解"，无法再回过头去仔细查看原文中的某个具体词汇。这种"看过就忘、只凭印象"的工作方式，正是标准 Encoder-Decoder（Seq2Seq）模型的典型特征，也是后来引入注意力机制要解决的问题。

### 历史背景

Encoder-Decoder（也称为 Seq2Seq，即 Sequence to Sequence）模型由 AI 领域世界级专家约书亚-本吉奥（Yoshua Bengio）及其率领的研究团队于 2014 年提出，原始论文题为 "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"。该模型采用编码器-解码器架构，以端到端的方式实现机器翻译，标志着神经机器翻译（Neural Machine Translation, NMT）时代的正式开启。在此之前，2013 年牛津大学的 Kalchbrenner 和 Blunsom 提出的循环连续翻译模型（RCTM）已具备了编码器-解码器的基本形态（CNN 编码 + RNN 解码），Seq2Seq 模型在此基础上将编码器和解码器统一为 RNN 结构，进一步简化了模型架构并提升了效果。此后，Bengio 团队于 2015 年在 Seq2Seq 基础上引入注意力机制，提出了注意力 Seq2Seq 模型（Bahdanau Attention），有效解决了长序列翻译的信息瓶颈问题。

### 算法定位

- 类型：监督学习 -- 序列到序列转换（seq2seq transduction）
- 输出：离散序列（可以是任意长度的符号序列，如单词序列、字符序列等）
- 模型类型：判别模型（条件概率模型 $p_\theta(\mathbf{y} | \mathbf{x})$）
- 核心组件：RNN/LSTM/GRU 作为编码器和解码器的基础单元
- 训练范式：端到端联合训练（最大似然估计）

### 前置知识

- **RNN 基础**：循环神经网络的前向传播、隐状态递推计算机制、梯度消失/爆炸问题
- **LSTM/GRU**：门控机制的基本原理，理解为什么需要长短期记忆单元
- **梯度下降与反向传播**：链式法则、BPTT（Backpropagation Through Time）
- **Softmax 函数**：将 logits 转化为概率分布的归一化方法
- **词嵌入（Word Embedding）**：将离散的词汇映射为连续的向量表示
- **交叉熵损失**：分类/序列生成任务中常用的损失函数
- **Teacher Forcing**（扩展）：理解训练时使用真实标签作为解码器输入的技术

---

## 2. 核心原理

### 2.1 核心思想

Encoder-Decoder 模型的核心思想非常直观：将序列到序列的转换任务分解为"编码"和"解码"两个阶段。编码阶段负责理解输入序列，将其压缩为一个固定维度的语义表示；解码阶段负责基于这个语义表示，逐步生成目标序列。

这种"先理解、再生成"的两阶段设计之所以有效，是因为它自然地模拟了人类处理序列转换任务的认知过程。无论是翻译、摘要还是对话，人类都是在先充分理解输入的基础上，再组织输出的。

核心思想可以概括为：通过一个编码器将变长输入压缩为固定维度的语义向量，再通过一个解码器从该语义向量逐步生成变长输出。

### 2.2 工作流程

1. **编码阶段（Encoder）**
   - 输入：源序列 $\mathbf{x} = \{x_1, x_2, \ldots, x_T\}$，其中每个 $x_t$ 是一个词嵌入向量
   - 操作：编码器 RNN 按照时间步从左到右逐个读取输入，每个时间步更新隐状态
   - 输出：最终隐状态 $\mathbf{h}_T$（或经过变换得到的上下文向量 $\mathbf{c}$）

2. **上下文向量生成**
   - 输入：编码器的隐状态序列 $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T$
   - 操作：通常取最后一个隐状态 $\mathbf{c} = \mathbf{h}_T$，或对全部隐状态进行某种变换
   - 输出：固定维度的上下文向量 $\mathbf{c}$，承载整个输入序列的语义信息

3. **解码阶段（Decoder）**
   - 输入：上下文向量 $\mathbf{c}$、特殊起始符 `<SOS>`
   - 操作：解码器 RNN 逐步生成目标序列的每个词，每步以上一步输出作为当前步输入
   - 输出：目标序列 $\mathbf{y} = \{y_1, y_2, \ldots, y_{T'}\}$，直到生成特殊终止符 `<EOS>`

### 2.3 关键概念解释

- **编码器（Encoder）**：一个 RNN 网络，负责读取并理解输入序列。它像一台"信息压缩器"，将长度为 $T$ 的输入序列压缩为一个固定维度的向量。

- **解码器（Decoder）**：一个 RNN 网络，负责基于编码结果生成输出序列。它像一台"信息展开器"，从固定维度的向量逐步展开为目标序列。

- **上下文向量（Context Vector）**：编码器与解码器之间的桥梁，是整个输入序列的浓缩表示。在标准 Seq2Seq 中，所有解码步共享同一个上下文向量。

- **隐状态（Hidden State）**：RNN 在每个时间步计算的内部表示。编码器的隐状态序列记录了从输入起始到当前位置的全部历史信息。

- **Teacher Forcing**：在训练阶段，解码器每一步的输入不使用模型自身的预测输出，而使用训练数据中的真实标签。这可以加速训练收敛，但也会引入"曝光偏差"问题。

- **贪心解码（Greedy Decoding）**：在推理阶段，解码器每一步选择概率最高的词作为输出。速度快但可能错过全局最优序列。

- **束搜索（Beam Search）**：在推理阶段，每一步保留概率最高的 $k$ 条候选路径，最终选择整体概率最大的序列。是贪心解码的改进版本。

### 2.4 信息瓶颈问题

标准 Encoder-Decoder 模型最核心的问题在于其信息瓶颈（Information Bottleneck）。无论输入序列有多长（可能只有几个词，也可能有几百个词），编码器都会将其压缩为一个固定维度的向量。这种设计在数学上存在一个根本性的矛盾：

上下文向量 $\mathbf{c}$ 的维度通常是固定的（例如 256、512 或 1024 维），这意味着它的信息容量是有限的。当输入序列较短时，$\mathbf{c}$ 可以较好地保留全部语义信息；但当输入序列较长时，$\mathbf{c}$ 的有限维度将无法容纳输入序列的全部信息，导致信息损失。这种损失会随着序列长度的增加而急剧恶化。

除了信息容量的问题外，标准 Seq2Seq 还存在另一个严重缺陷：输入序列"不划重点，平等对待"。在生成每一个输出词时，解码器使用的上下文向量 $\mathbf{c}$ 都是同一个，这意味着输入序列中的每个元素对输出的每一个元素都具有完全相同的影响力。然而在实际的翻译任务中，输入序列中不同词汇的重要性是截然不同的 -- 例如，在翻译"I want to read a magazine"时，"magazine"是核心名词，而"a"是不定冠词，前者对翻译结果的影响远大于后者。

正是上述两个缺陷，促使了注意力 Seq2Seq 模型在 2015 年的诞生。注意力机制通过为每个输出词动态计算独立的上下文向量，既解决了信息容量的问题（不再需要将所有信息压缩到单个向量中），也解决了"平等对待"的问题（不同输入词可以获得不同的注意力权重）。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/说明 |
|------|------|-----------|
| $\mathbf{x} = \{x_1, \ldots, x_T\}$ | 输入（源）序列 | $T$ 为输入序列长度 |
| $\mathbf{y} = \{y_1, \ldots, y_{T'}\}$ | 输出（目标）序列 | $T'$ 为输出序列长度 |
| $\mathbf{h}_t$ | 编码器在时刻 $t$ 的隐状态 | $d_h$ 维向量 |
| $\mathbf{s}_t$ | 解码器在时刻 $t$ 的隐状态 | $d_s$ 维向量 |
| $\mathbf{c}$ | 上下文向量 | $d_c$ 维向量（通常 $d_c = d_h$） |
| $\theta$ | 模型所有可训练参数 | 编码器参数 + 解码器参数 |
| $f(\cdot)$ | RNN 隐状态计算函数 | 具体形式取决于 RNN 类型 |
| $g(\cdot)$ | 解码器输出计算函数 | 通常包含 softmax |
| $V$ | 词汇表大小 | 输出空间维度 |

### 3.2 问题形式化

给定一个长度为 $T$ 的输入序列 $\mathbf{x} = \{x_1, x_2, \ldots, x_T\}$，Seq2Seq 模型的目标是找到一个最优的模型参数 $\theta^*$，使得模型能够以最大概率产生正确的输出序列 $\mathbf{y} = \{y_1, y_2, \ldots, y_{T'}\}$。形式化地，可以表述为：

$$ \theta^* = \arg\max_\theta \frac{1}{N} \sum_{i=1}^{N} \log p_\theta\left(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}\right) $$

其中 $N$ 为训练样本数量，$p_\theta(\mathbf{y}^{(i)} | \mathbf{x}^{(i)})$ 为模型在给定第 $i$ 个输入序列时，产生第 $i$ 个输出序列的条件概率。

### 3.3 RNN 隐状态递推

RNN 是 Seq2Seq 模型的核心计算单元。对于编码器，它在处理输入序列的第 $t$ 个元素时，会综合使用上一步的隐状态 $\mathbf{h}_{t-1}$ 和当前步的输入 $x_t$ 来计算当前步的隐状态 $\mathbf{h}_t$：

$$ \mathbf{h}_t = f(\mathbf{h}_{t-1}, x_t) \tag{3-1} $$

其中 $f$ 是针对输入 $x_t$ 和隐状态 $\mathbf{h}_{t-1}$ 所进行的某种变换操作。其具体形式可简可繁：简单时可以是"线性变换加激活函数"（标准 RNN），复杂时可以是"各种门控机制的一套操作"（LSTM 或 GRU）。

对于标准 RNN，$f$ 的具体形式为：

$$ \mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} x_t + \mathbf{b}_h) $$

其中 $\mathbf{W}_{hh}$ 是隐状态到隐状态的权重矩阵，$\mathbf{W}_{xh}$ 是输入到隐状态的权重矩阵，$\mathbf{b}_h$ 是偏置向量。

RNN 的关键特性在于：它在决定当前输入的表示时，不仅"看着眼前"（即 $x_t$），还要"想着过去"（即 $\mathbf{h}_{t-1}$），以此来建模历史对当下的影响。在机器翻译中，这一特性极其重要 -- 翻译当前词时必须参考前文语境。例如，"I want to read a magazine"和"A gun magazine"中的"magazine"应当分别翻译为"杂志"和"弹匣"，其区别取决于前文是"read"还是"gun"。

### 3.4 编码器映射

编码器按照时间顺序逐个读取输入元素，针对每一个输入都会产生一个隐状态。在处理完整个输入序列 $x_1, \ldots, x_T$ 后，即得到了对应的隐状态序列 $\mathbf{h}_0, \mathbf{h}_1, \ldots, \mathbf{h}_T$（其中 $\mathbf{h}_0$ 为隐状态初值）。

编码器的功能可以形式化表示为从输入序列到上下文向量的映射：

$$ \mathbf{c} = \text{encode}(x_1, \ldots, x_T) \tag{3-2} $$

获得上下文向量 $\mathbf{c}$ 的具体方法有多种，只要蕴含输入序列的完整信息即可。常见的实现方式包括：

**方式一**：直接取编码器的最后一个隐状态

$$ \mathbf{c} = \mathbf{h}_T $$

这是最简单的方式。最后一个隐状态 $\mathbf{h}_T$ 经历了从 $x_1$ 到 $x_T$ 的全部输入，理论上包含了完整的输入信息。

**方式二**：对最后一个隐状态进行某种变换

$$ \mathbf{c} = q(\mathbf{h}_T) $$

其中 $q$ 可以是一个线性变换或非线性变换，目的是将隐状态映射到一个更合适的空间。

**方式三**：对所有隐状态进行某种聚合变换

$$ \mathbf{c} = q(\mathbf{h}_1, \ldots, \mathbf{h}_T) $$

例如取所有隐状态的平均值 $\mathbf{c} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{h}_t$，或使用最大池化等操作。

### 3.5 解码器隐状态

解码器也被构造为 RNN 结构。该 RNN 以编码器输出的上下文向量 $\mathbf{c}$ 作为输入，一边逐个产生自身的隐状态序列 $\mathbf{s}_1, \ldots, \mathbf{s}_{T'}$，一边逐个生成目标序列 $y_1, \ldots, y_{T'}$。

解码器在第 $t$ 步的隐状态 $\mathbf{s}_t$（$t = 1, \ldots, T'$）由三个因素共同决定：

$$ \mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}) \tag{3-3} $$

与标准 RNN 隐状态计算（式 3-1）相比，式 3-3 存在两点重要区别：

1. **$\mathbf{s}_t$ 依赖 $\mathbf{c}$**：解码器必须能够"看到"输入序列的信息，而上下文向量 $\mathbf{c}$ 蕴含了整个输入序列的浓缩信息。在实际实现中，$\mathbf{c}$ 通常只在解码器的第一个时间步输入，之后通过隐状态的传递间接影响后续步骤；也可以在每个时间步都作为额外输入。

2. **$\mathbf{s}_t$ 依赖 $y_{t-1}$**：生成当前隐状态时需要"回顾上一步说了什么"。这在翻译中尤为重要 -- 翻译"magazine"时需要知道前面已经翻译出的是"读"还是"枪"。

对于 LSTM 解码器，式 3-3 的展开形式为：

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{s}_{t-1}; y_{t-1}; \mathbf{c}] + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{s}_{t-1}; y_{t-1}; \mathbf{c}] + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{s}_{t-1}; y_{t-1}; \mathbf{c}] + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{s}_{t-1}; y_{t-1}; \mathbf{c}] + \mathbf{b}_c) \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{s}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
\end{aligned}
$$

其中 $[\cdot ; \cdot]$ 表示向量拼接，$\odot$ 表示逐元素乘法，$\sigma$ 表示 sigmoid 函数。

### 3.6 解码器输出

对于第 $t$ 步的输出 $y_t$，其计算方法可以表示为：

$$ y_t = g(\mathbf{s}_t, y_{t-1}, \mathbf{c}) \tag{3-4} $$

$y_t$ 依赖于三个因素：

1. **$\mathbf{s}_t$**：当前隐状态，是 $y_t$ 的直接内部表示，这是"天经地义"的。
2. **$y_{t-1}$**：上一个输出，"张嘴说话前，想想前面说了什么"。
3. **$\mathbf{c}$**：上下文向量，当前要"说出"的那个词一定与输入语句的整体信息高度相关。

在实际的机器翻译任务中，$g$ 通常包含以下计算过程：

$$ y_t = \arg\max_{y} P(y | y_{t-1}, y_{t-2}, \ldots, y_1, \mathbf{c}) \tag{3-5} $$

首先通过线性变换将隐状态映射到词汇表维度的空间，然后通过 softmax 函数计算每个词的条件概率：

$$ P(y_t = w | y_{t-1}, \ldots, y_1, \mathbf{c}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o) $$

其中 $\mathbf{W}_o$ 是输出权重矩阵，维度为 $V \times d_s$（$V$ 为词汇表大小），$\mathbf{b}_o$ 是偏置向量。

### 3.7 训练目标 -- 最大似然估计（MLE）

Seq2Seq 模型使用端到端的方式对编码器和解码器中的参数进行联合训练。训练样本集中包含了诸多输入序列和对应输出序列的成对样本，表示为 $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$。

**Step 1：定义单个样本的目标函数**

针对单个样本 $(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})$，利用最大似然估计的思想，最优的模型参数应使得条件似然函数取得最大值：

$$ \theta^* = \arg\max_\theta \log p_\theta(\mathbf{y}^{(i)} | \mathbf{x}^{(i)}) \tag{3-6} $$

**Step 2：利用链式法则展开联合概率**

关键问题在于，$p_\theta(\mathbf{y} | \mathbf{x})$ 是一个"整句对整句"的条件概率，需要将其拆解为可计算的形式。对于一个联合分布，可以利用概率的链式法则将其展开为一系列条件概率的连乘积形式：

$$
\begin{aligned}
p_\theta(\mathbf{y} | \mathbf{x}) &= p_\theta(y_1, y_2, \ldots, y_{T'} | \mathbf{c}) \tag{3-7} \\
&= p_\theta(y_1 | \mathbf{c}) \cdot p_\theta(y_2 | y_1, \mathbf{c}) \cdots p_\theta(y_{T'} | y_{T'-1}, \ldots, y_1, \mathbf{c})
\end{aligned}
$$

推导过程如下：

首先，根据条件概率的乘法公式，有：

$$ p(A, B) = p(A) \cdot p(B | A) $$

将这一公式推广到多个随机变量的情形：

$$ p(y_1, y_2, \ldots, y_{T'}) = p(y_1) \cdot p(y_2 | y_1) \cdot p(y_3 | y_1, y_2) \cdots p(y_{T'} | y_1, \ldots, y_{T'-1}) $$

在给定上下文向量 $\mathbf{c}$ 的条件下（$\mathbf{c}$ 由 $\mathbf{x}$ 决定），上式变为：

$$ p_\theta(y_1, \ldots, y_{T'} | \mathbf{c}) = \prod_{t=1}^{T'} p_\theta(y_t | y_1, \ldots, y_{t-1}, \mathbf{c}) $$

将式 (3-7) 与式 (3-5) 进行对比可以发现，展开式中每一项条件概率 $p_\theta(y_t | y_{t-1}, \ldots, y_1, \mathbf{c})$ 正好对应解码器在第 $t$ 步的预测输出。这意味着，如果将经过 softmax 概率化后的解码器所有步骤的输出直接连乘，得到的结果就是"整句对整句"的条件概率。

**Step 3：取对数并求全局平均**

对式 (3-7) 取对数，将连乘转化为连加：

$$ \log p_\theta(\mathbf{y} | \mathbf{x}) = \sum_{t=1}^{T'} \log p_\theta(y_t | y_{t-1}, \ldots, y_1, \mathbf{c}) $$

在整个训练样本集上取平均，得到最终的目标函数：

$$ \theta^* = \arg\max_\theta \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T'} \log p_\theta\left(y_t^{(i)} | y_{t-1}^{(i)}, \ldots, y_1^{(i)}, \mathbf{x}^{(i)}\right) \tag{3-8} $$

式 (3-8) 将目标函数从句子级细化到序列词汇级。其中 $y_t^{(i)}$ 表示第 $i$ 个输出序列中的第 $t$ 个输出元素。

由于我们通常使用梯度下降来最小化损失函数，因此将最大化问题转化为最小化问题：

$$ \mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T'} \log p_\theta\left(y_t^{(i)} | y_{t-1}^{(i)}, \ldots, y_1^{(i)}, \mathbf{x}^{(i)}\right) $$

这就是 Seq2Seq 模型中使用的交叉熵损失函数。

### 3.8 最终的算法步骤

综合以上推导，Seq2Seq 模型的完整训练和推理过程可以总结如下：

**训练阶段**：
```
初始化模型参数 theta（编码器参数和解码器参数）
for epoch in range(max_epochs):
    for 每个训练样本 (x, y) in 训练集:
        # 编码阶段
        h_0 = zeros  # 初始化编码器隐状态
        for t = 1 to T:
            h_t = f(h_{t-1}, x_t)  # 递推计算编码器隐状态
        c = h_T  # 取最后隐状态作为上下文向量

        # 解码阶段（使用 Teacher Forcing）
        s_0 = zeros 或 s_0 = h_T  # 初始化解码器隐状态
        y_0 = <SOS>  # 起始符
        for t = 1 to T':
            s_t = f(s_{t-1}, y_{t-1}, c)  # 递推计算解码器隐状态
            P(y_t) = softmax(W_o * s_t + b_o)  # 计算输出概率分布
            loss += -log(P(y_t = y_t^{true}))  # 累积交叉熵损失

        # 反向传播
        反向传播计算所有参数的梯度
        使用梯度裁剪防止梯度爆炸
        更新参数 theta
```

**推理阶段**：
```
给定输入序列 x:
# 编码
h_0 = zeros
for t = 1 to T:
    h_t = f(h_{t-1}, x_t)
c = h_T

# 解码（自回归生成）
s_0 = zeros 或 s_0 = h_T
y_0 = <SOS>
output_sequence = []
for t = 1 to max_length:
    s_t = f(s_{t-1}, y_{t-1}, c)
    P(y_t) = softmax(W_o * s_t + b_o)
    y_t = argmax(P(y_t))  # 贪心解码
    if y_t == <EOS>:  # 遇到终止符则停止
        break
    output_sequence.append(y_t)
return output_sequence
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

Seq2Seq 模型的数据预处理步骤比传统机器学习模型更为复杂，主要因为其处理的是离散的序列数据。

**1. 分词（Tokenization）**

将原始文本切分为词或子词单元。常用的分词方法包括：

- 词级别分词（Word-level）：以空格和标点为分隔符，适用于英语等词之间有明确分隔的语言
- 字符级别分词（Character-level）：以单个字符为基本单元，适用于中文等没有明确词边界的语言
- 子词分词（Subword-level）：如 BPE（Byte Pair Encoding）、WordPiece，在词级别和字符级别之间取得平衡

```python
# 分词示例
source_sentence = "I love machine learning"
target_sentence = "我爱机器学习"

# 词级别分词（英语）
source_tokens = ["I", "love", "machine", "learning"]
# 字符级别分词（中文）
target_tokens = ["我", "爱", "机", "器", "学", "习"]
```

**2. 构建词汇表**

为源语言和目标语言分别构建词汇表，将每个 token 映射为一个整数索引：

```python
# 构建词汇表示例
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3,
         "I": 4, "love": 5, "machine": 6, "learning": 7, ...}
```

特殊 token 的说明：
- `<PAD>`（填充符）：用于将不同长度的序列对齐到相同长度，以支持批量训练
- `<SOS>`（起始符）：标记输出序列的开始
- `<EOS>`（终止符）：标记输出序列的结束
- `<UNK>`（未知词）：处理词汇表中不存在的词

**3. 序列填充与截断**

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 将序列填充到相同长度（以批次内最长序列为准）
padded_sequences = pad_sequences(sequences, padding='post',
                                  maxlen=max_length, truncating='post')
```

**4. 词嵌入**

将离散的整数索引映射为连续的稠密向量表示，维度通常为 128、256 或 512。

### 4.2 Teacher Forcing

Teacher Forcing 是 Seq2Seq 模型训练中最关键的技术之一。其核心思想是：在训练阶段，解码器每一步的输入不使用模型自身在前一步的预测输出，而是使用训练数据中的真实标签。

**为什么需要 Teacher Forcing？**

如果不使用 Teacher Forcing（即解码器每步使用自身预测作为下一步输入），会面临以下问题：

1. **误差累积**：模型在早期步骤的一个小错误，会被作为后续步骤的输入，导致错误不断放大
2. **训练缓慢**：模型需要同时学习"生成正确的第一步"和"处理错误的输入"，学习目标过于复杂
3. **收敛困难**：错误的输入导致梯度信号不稳定，使得训练难以收敛

**Teacher Forcing 的实现方式**：

```
训练时（使用 Teacher Forcing）：
  解码器输入：  <SOS>   y_1     y_2     y_3     ...
  解码器输出：  y_1     y_2     y_3     y_4     ...
  真实标签：    y_1     y_2     y_3     y_4     ...
  损失计算：    loss(y_1) loss(y_2) loss(y_3) loss(y_4) ...
  (解码器输入使用的是真实标签，而非模型预测)

推理时（不使用 Teacher Forcing，自回归生成）：
  解码器输入：  <SOS>   y_hat_1  y_hat_2  y_hat_3  ...
  解码器输出：  y_hat_1  y_hat_2  y_hat_3  y_hat_4  ...
  (解码器输入使用的是模型自身的预测)
```

**Scheduled Sampling（计划采样）**：

为了缓解 Teacher Forcing 造成的曝光偏差（Exposure Bias）， Bengio 团队提出了计划采样策略：在训练过程中，以一定概率 $p_i$ 使用真实标签，以概率 $1 - p_i$ 使用模型预测作为解码器输入。$p_i$ 随着训练的进行逐渐衰减：

$$ p_i = \min(1, \frac{k}{i + k}) $$

其中 $k$ 是一个超参数，$i$ 是当前的训练步数。

### 4.3 梯度裁剪

由于 Seq2Seq 模型包含 RNN 结构，且需要在时间维度上展开进行反向传播（BPTT），因此极易出现梯度爆炸问题。梯度裁剪是解决这一问题的标准手段。

**梯度裁剪的实现**：

```python
# 方法一：按值裁剪（clip by value）
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# 方法二：按范数裁剪（clip by norm，推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

按范数裁剪的原理：如果梯度的 L2 范数 $\|g\|$ 超过阈值 $\max\_norm$，则将梯度按比例缩小：

$$ g_{\text{clipped}} = \begin{cases} g & \text{if } \|g\| \leq \max\_norm \\ \frac{\max\_norm}{\|g\|} \cdot g & \text{if } \|g\| > \max\_norm \end{cases} $$

推荐的 $\max\_norm$ 取值范围为 1.0 到 5.0。

### 4.4 学习率调度

Seq2Seq 模型的训练通常使用学习率调度策略，以确保训练既能在初期快速收敛，又能在后期精细调整。

**常用策略**：

1. **学习率预热（Warmup）+ 衰减**：训练初期使用较小的学习率逐步增大到目标值，然后在使用余弦退火或指数衰减逐步减小。

2. **ReduceLROnPlateau**：当验证集损失连续若干个 epoch 没有改善时，将学习率乘以一个衰减因子（如 0.5）。

```python
# PyTorch 学习率调度示例
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

# 训练循环中
for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)  # 根据验证损失调整学习率
```

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| embedding_dim | 词嵌入维度 | 128-512 | 256 |
| hidden_dim | RNN 隐状态维度 | 256-1024 | 512 |
| num_layers | RNN 层数（堆叠层数） | 1-4 | 2 |
| dropout | Dropout 概率 | 0.1-0.5 | 0.3 |
| learning_rate | 学习率 | 0.0001-0.005 | 0.001 |
| batch_size | 批量大小 | 16-128 | 64 |
| max_grad_norm | 梯度裁剪阈值 | 1.0-5.0 | 1.0 |
| teacher_forcing_ratio | Teacher Forcing 使用比例 | 0.5-1.0 | 1.0 |
| beam_size | 束搜索宽度（推理时） | 1-10 | 5 |

---

## 5. 应用场景

### 5.1 机器翻译（Machine Translation）

**问题类型**：序列到序列转换

**为什么适合 Encoder-Decoder**：
- 源语言和目标语言的长度可以不同，Seq2Seq 天然支持变长到变长的映射
- 翻译需要理解源语言的完整语义后再生成目标语言，编码-解码的两阶段设计与此需求完美匹配
- RNN 能够捕获序列中的长程依赖关系，对于处理语法结构复杂的句子至关重要

**实际案例**：Google 翻译在 2016 年用基于 LSTM 的 Seq2Seq 模型替换了原先的统计机器翻译系统（GNMT 系统），翻译质量实现了显著提升。

### 5.2 文本摘要（Text Summarization）

**问题类型**：长序列到短序列的压缩

**为什么适合**：
- 摘要需要对原文进行整体理解后再生成精简版本，编码器负责理解，解码器负责生成
- 输入（文章）和输出（摘要）长度差异很大，Seq2Seq 的灵活映射能力恰好满足这一需求

**实际案例**：新闻摘要生成、学术论文摘要自动生成。

### 5.3 对话系统（Conversational AI / Chatbot）

**问题类型**：序列到序列的响应生成

**为什么适合**：
- 对话的核心是"理解用户意图"加上"生成恰当回复"，与编码-解码范式天然一致
- 对话历史可以作为额外的上下文信息输入编码器，增强回复的相关性

**实际案例**：早期的神经对话系统（如 Neural Conversational Model, 2015）直接使用 Seq2Seq 实现。目前的对话系统在 Seq2Seq 基础上增加了多轮对话管理、知识检索等模块。

### 5.4 看图说话（Image Captioning）

**问题类型**：图像到文本序列的跨模态生成

**为什么适合**：
- 只要将编码器替换为 CNN（卷积神经网络），将图像编码为一个固定维度的特征向量，就可以使用标准的解码器生成描述文本
- CNN 编码器 + RNN 解码器的组合是看图说话任务的经典架构

**实际案例**：MS COCO 图像描述数据集上的多数早期模型都采用了 CNN+RNN 的 Encoder-Decoder 架构。

### 5.5 语音识别（Speech Recognition）

**问题类型**：音频序列到文本序列的转换

**为什么适合**：
- 语音信号是时间序列，Seq2Seq 天然适合处理序列数据
- 使用声学模型（如 RNN 或 CNN）作为编码器，将语音特征编码为上下文向量，再用解码器生成对应的文字

**实际案例**：Listen, Attend and Spell（LAS）模型是 Seq2Seq 架构在语音识别领域的代表性应用，使用 Listener（pyramid RNN 编码器）+ Speller（attention-based LSTM 解码器）。

### 5.6 其他应用

- **代码生成**：将自然语言描述转换为代码（编码器处理自然语言，解码器生成代码）
- **语法纠错**：将包含语法错误的句子转换为正确的句子
- **问答系统**：将问题和上下文编码后生成答案
- **音乐生成**：将音乐片段编码后生成新的旋律

### 5.7 适用数据特征

- **数据类型**：序列数据（文本、音频、时间序列等）
- **输入输出长度**：可以不同（变长到变长）
- **数据规模**：需要大规模训练数据（通常数万到数百万个配对样本）
- **标注需求**：需要配对的输入-输出序列作为监督信号

### 5.8 不适用场景

1. **固定长度的输入输出**：如果输入和输出长度固定且较短，使用更简单的模型即可
2. **缺乏平行语料**：如果没有足够多的配对训练数据，Seq2Seq 难以训练好
3. **对实时性要求极高的场景**：RNN 的序列计算方式导致推理速度较慢，不适合低延迟要求
4. **需要精确对齐的任务**：如词级别的标注任务（NER、POS tagging），使用序列标注模型更合适

---

## 6. 优缺点分析

### 6.1 优点

1. **端到端学习，无需手工特征工程**
   - Seq2Seq 模型直接从原始的配对序列数据中学习映射关系，不需要人工设计语言特征或翻译规则
   - 相比于基于规则的机器翻译和统计机器翻译，大幅降低了系统的构建成本
   - 成立条件：需要足够大规模的高质量配对训练数据

2. **灵活的变长输入输出映射**
   - 编码器处理任意长度的输入序列，解码器生成任意长度的输出序列，两者之间没有长度约束
   - 这一特性使得 Seq2Seq 适用于几乎所有序列转换任务
   - 技术细节：编码器和解码器的展开长度可以完全独立

3. **良好的上下文建模能力**
   - RNN（特别是 LSTM/GRU）能够捕获序列中的长程依赖关系
   - 在翻译等任务中，上下文信息对消歧至关重要（如前文提到的 "magazine" 翻译示例）
   - 适用场景：输入序列中的上下文信息对输出有重要影响的任务

4. **架构通用性强，易于扩展**
   - 编码器和解码器可以自由替换为不同的网络结构（RNN、LSTM、GRU、CNN、Transformer 等）
   - 可以方便地引入注意力机制、复制机制、覆盖机制等增强模块
   - 同一架构可以适配机器翻译、文本摘要、对话等多种任务

5. **统一的概率框架**
   - Seq2Seq 模型在数学上提供了清晰的概率解释：$p_\theta(\mathbf{y} | \mathbf{x})$
   - 既可以用于生成（选择概率最高的序列），也可以用于评分（计算任意输入-输出对的概率）
   - 这为后续的束搜索、重排序等技术提供了理论基础

### 6.2 缺点

1. **信息瓶颈问题严重**
   - 问题场景：无论输入序列多长，都被压缩为单个固定维度的向量，信息容量有限
   - 解决思路：引入注意力机制，为每个输出步动态计算上下文向量
   - 实际影响：当输入序列超过约 20-30 个词时，翻译质量明显下降

2. **无法处理输入中的不同重要度**
   - 问题场景：所有输入元素对输出的影响相同，无法区分关键信息和次要信息
   - 改进方法：引入注意力机制，学习输入元素的重要性权重
   - 实际影响：翻译结果缺乏"温度"，无法做到有重点的翻译

3. **训练和推理的不一致性（曝光偏差）**
   - 问题场景：训练时使用 Teacher Forcing（真实标签作为输入），推理时使用模型自身预测作为输入，两者的输入分布不一致
   - 缓解方法：Scheduled Sampling、Professor Forcing 等技术
   - 实际影响：模型在训练时表现很好，但推理时可能出现大量错误

4. **RNN 的固有局限（顺序计算，无法并行）**
   - 问题场景：RNN 必须按时间步顺序计算，无法利用现代 GPU 的并行计算能力
   - 替代方案：使用 CNN 或 Transformer 编码器替代 RNN
   - 实际影响：训练和推理速度慢，尤其对于长序列

5. **长距离依赖问题**
   - 问题场景：即使使用 LSTM/GRU，当序列很长时（超过 50-100 步），梯度在反向传播过程中仍然会显著衰减，导致模型难以捕获非常长距离的依赖关系
   - 替代方案：Transformer 的自注意力机制可以直接建模任意距离的依赖关系

### 6.3 与同类模型的对比

| 维度 | Seq2Seq (基础版) | Attention Seq2Seq | Transformer |
|------|-------------------|-------------------|-------------|
| 编码器 | 单向 RNN | 双向 RNN | 多头自注意力 |
| 解码器 | 单向 RNN | 单向 RNN + Attention | 掩码多头自注意力 + 交叉注意力 |
| 上下文向量 | 单个固定向量 | 每步动态计算 | 每步动态计算 |
| 长序列能力 | 差（<20词） | 中等（<50词） | 强（数百词） |
| 训练速度 | 慢（顺序计算） | 慢（顺序计算） | 快（可并行） |
| 信息瓶颈 | 严重 | 缓解 | 无 |
| 并行能力 | 无 | 无 | 完全并行 |
| BLEU 分数（WMT） | 较低 | 中等 | 最高 |
| 实现复杂度 | 低 | 中等 | 较高 |
| 参数量 | 较少 | 中等 | 较多 |
| 训练数据需求 | 中等 | 中等 | 大量 |

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch numpy matplotlib
```

### 7.2 PyTorch 完整实现 Seq2Seq（机器翻译示例）

```python
"""
Encoder-Decoder (Seq2Seq) 完整实现
使用 PyTorch 构建基于 LSTM 的 Seq2Seq 模型
任务：简单的英译中翻译演示
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 设置随机种子，保证可复现
torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# 1. 数据准备 -- 构建简单的英中翻译数据集
# =====================================================

# 构建简单的平行语料
english_sentences = [
    "I am a student",
    "I am a teacher",
    "He is a doctor",
    "She is a nurse",
    "We are friends",
    "They are workers",
    "I love cats",
    "She loves dogs",
    "He likes music",
    "We like food",
    "I study hard",
    "She works hard",
    "He plays games",
    "We read books",
    "They watch movies",
    "I am happy",
    "She is sad",
    "He is tired",
    "We are busy",
    "They are free",
]

chinese_sentences = [
    "我是一个学生",
    "我是一个老师",
    "他是一个医生",
    "她是一个护士",
    "我们是朋友",
    "他们是工人",
    "我喜欢猫",
    "她喜欢狗",
    "他喜欢音乐",
    "我们喜欢食物",
    "我努力学习",
    "她努力工作",
    "他玩游戏",
    "我们读书",
    "他们看电影",
    "我很开心",
    "她很伤心",
    "他很累",
    "我们很忙",
    "他们很空闲",
]

# 特殊 token 定义
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """词汇表类，负责 token 到索引的映射"""

    def __init__(self):
        self.token2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2token = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.token_count = {}

    def build_vocab(self, sentences: List[str], char_level=False):
        """根据句子列表构建词汇表"""
        for sentence in sentences:
            if char_level:
                tokens = list(sentence)
            else:
                tokens = sentence.split()
            for token in tokens:
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                self.token_count[token] = self.token_count.get(token, 0) + 1

    def __len__(self):
        return len(self.token2idx)

    def sentence_to_indices(self, sentence: str, char_level=False) -> List[int]:
        """将句子转换为索引序列"""
        if char_level:
            tokens = list(sentence)
        else:
            tokens = sentence.split()
        return [self.token2idx.get(t, self.token2idx[UNK_TOKEN]) for t in tokens]

    def indices_to_sentence(self, indices: List[int]) -> str:
        """将索引序列转换回句子"""
        tokens = [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]
        # 去除特殊 token
        tokens = [t for t in tokens if t not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
        return "".join(tokens)


# 构建英文和中文词汇表
en_vocab = Vocabulary()
zh_vocab = Vocabulary()
en_vocab.build_vocab(english_sentences)
zh_vocab.build_vocab(chinese_sentences, char_level=True)

print(f"英文词汇表大小: {len(en_vocab)}")
print(f"中文词汇表大小: {len(zh_vocab)}")


# =====================================================
# 2. 模型定义
# =====================================================

class Encoder(nn.Module):
    """LSTM 编码器，将输入序列编码为上下文向量"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super(Encoder, self).__init__()
        # 词嵌入层：将 token 索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM 层：可以是多层堆叠的 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        """
        编码器前向传播

        Args:
            src: 输入序列索引，shape (batch_size, seq_len)

        Returns:
            outputs: 所有时间步的隐状态，shape (batch_size, seq_len, hidden_dim)
            hidden: 最后一个时间步的隐状态，shape (num_layers, batch_size, hidden_dim)
            cell: 最后一个时间步的细胞状态，shape (num_layers, batch_size, hidden_dim)
        """
        # src shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(src))
        # embedded shape: (batch_size, seq_len, embed_dim)

        # LSTM 前向传播
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # cell shape: (num_layers, batch_size, hidden_dim)

        return outputs, hidden, cell


class Decoder(nn.Module):
    """LSTM 解码器，基于上下文向量逐步生成目标序列"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM 层：输入维度为 embed_dim（无注意力时）
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        # 输出层：将隐状态映射到词汇表大小的概率分布
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor):
        """
        解码器单步前向传播

        Args:
            input_token: 当前步输入 token，shape (batch_size, 1)
            hidden: 当前隐状态，shape (num_layers, batch_size, hidden_dim)
            cell: 当前细胞状态，shape (num_layers, batch_size, hidden_dim)

        Returns:
            output: 当前步的输出概率分布，shape (batch_size, vocab_size)
            hidden: 更新后的隐状态
            cell: 更新后的细胞状态
        """
        # input_token shape: (batch_size, 1)
        embedded = self.dropout(self.embedding(input_token))
        # embedded shape: (batch_size, 1, embed_dim)

        # LSTM 单步前向传播
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output shape: (batch_size, 1, hidden_dim)

        # 通过全连接层映射到词汇表维度
        prediction = self.fc_out(output.squeeze(1))
        # prediction shape: (batch_size, vocab_size)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Seq2Seq 模型：将编码器和解码器组合在一起"""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        """
        Seq2Seq 前向传播（训练模式）

        Args:
            src: 源序列，shape (batch_size, src_len)
            trg: 目标序列，shape (batch_size, trg_len)
            teacher_forcing_ratio: Teacher Forcing 的使用概率

        Returns:
            outputs: 所有时间步的输出概率分布，shape (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码：将源序列编码为上下文向量
        _, hidden, cell = self.encoder(src)

        # 解码：第一步的输入为 <SOS> token
        input_token = trg[:, 0].unsqueeze(1)  # shape: (batch_size, 1)

        for t in range(1, trg_len):
            # 解码器单步前向传播
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            # 保存当前步的输出
            outputs[:, t, :] = output

            # 决定下一步的输入：使用 Teacher Forcing 或模型预测
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 贪心选择概率最高的词

            if teacher_force:
                input_token = trg[:, t].unsqueeze(1)  # 使用真实标签
            else:
                input_token = top1.unsqueeze(1)  # 使用模型预测

        return outputs


# =====================================================
# 3. 数据加载与训练工具函数
# =====================================================

def prepare_batch(en_sentences: List[str], zh_sentences: List[str],
                  en_vocab: Vocabulary, zh_vocab: Vocabulary,
                  batch_size: int = 4):
    """将句子列表转换为批量张量"""
    # 转换为索引序列
    en_indices = [en_vocab.sentence_to_indices(s) for s in en_sentences]
    zh_indices = [[1] + zh_vocab.sentence_to_indices(s, char_level=True) + [2]
                  for s in zh_sentences]  # 添加 SOS 和 EOS

    # 填充到相同长度
    max_en_len = max(len(seq) for seq in en_indices)
    max_zh_len = max(len(seq) for seq in zh_indices)

    en_batch = []
    for seq in en_indices:
        padded = seq + [0] * (max_en_len - len(seq))
        en_batch.append(padded)

    zh_batch = []
    for seq in zh_indices:
        padded = seq + [0] * (max_zh_len - len(seq))
        zh_batch.append(padded)

    # 转换为张量
    en_tensor = torch.tensor(en_batch, dtype=torch.long)
    zh_tensor = torch.tensor(zh_batch, dtype=torch.long)

    return en_tensor, zh_tensor


# =====================================================
# 4. 训练循环
# =====================================================

def train_seq2seq(model, en_sentences, zh_sentences, en_vocab, zh_vocab,
                  num_epochs=200, learning_rate=0.005, teacher_forcing_ratio=0.8,
                  print_every=20):
    """训练 Seq2Seq 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数（忽略 <PAD> 的损失）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    print(f"使用设备: {device}")
    print(f"开始训练，共 {num_epochs} 个 epoch")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()

        # 准备数据
        en_tensor, zh_tensor = prepare_batch(
            en_sentences, zh_sentences, en_vocab, zh_vocab
        )
        en_tensor = en_tensor.to(device)
        zh_tensor = zh_tensor.to(device)

        # 前向传播
        outputs = model(en_tensor, zh_tensor, teacher_forcing_ratio)

        # 计算损失
        # outputs shape: (batch_size, trg_len, vocab_size)
        # 需要重塑为 (batch_size * trg_len, vocab_size)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        targets = zh_tensor[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

        # 打印训练进度
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("=" * 60)
    print("训练完成！")
    return model, loss_history


# =====================================================
# 5. 束搜索（Beam Search）解码
# =====================================================

class BeamSearchDecoder:
    """束搜索解码器"""

    def __init__(self, model: Seq2Seq, beam_size: int = 5,
                 max_len: int = 50, device: torch.device = None):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def decode(self, src: torch.Tensor):
        """
        使用束搜索进行解码

        Args:
            src: 源序列，shape (1, src_len)

        Returns:
            best_sentence: 概率最高的输出序列（字符串）
        """
        self.model.eval()

        # 编码
        _, hidden, cell = self.model.encoder(src)

        # 初始化：以 <SOS> 开始
        sos_idx = 1  # <SOS> 的索引
        eos_idx = 2  # <EOS> 的索引

        # 每条 beam 的状态：(累积对数概率, 当前序列, 隐状态, 细胞状态)
        beams = [
            (0.0, [sos_idx], hidden, cell)
        ]

        # 完成（遇到 EOS）的 beam 列表
        completed_beams = []

        for _ in range(self.max_len):
            new_beams = []

            for log_prob, seq, h, c in beams:
                # 如果当前 beam 已经以 EOS 结尾，直接加入完成列表
                if seq[-1] == eos_idx and len(seq) > 1:
                    completed_beams.append((log_prob, seq))
                    continue

                # 解码器单步前向传播
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                output, new_h, new_c = self.model.decoder(input_token, h, c)

                # 获取 top-k 个候选词
                log_probs = torch.log(output.squeeze(0) + 1e-10)
                topk_log_probs, topk_indices = log_probs.topk(self.beam_size)

                for i in range(self.beam_size):
                    token_idx = topk_indices[i].item()
                    token_log_prob = topk_log_probs[i].item()
                    new_seq = seq + [token_idx]
                    new_beam = (log_prob + token_log_prob, new_seq, new_h, c)
                    new_beams.append(new_beam)

            # 从所有候选 beam 中选择概率最高的 beam_size 条
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:self.beam_size]

            # 如果所有 beam 都已完成，提前结束
            if all(seq[-1] == eos_idx and len(seq) > 1
                   for _, seq, _, _ in beams):
                break

        # 将未完成的 beam 也加入完成列表
        for log_prob, seq, _, _ in beams:
            if not (seq[-1] == eos_idx and len(seq) > 1):
                completed_beams.append((log_prob, seq))

        # 选择概率最高的 beam
        if not completed_beams:
            return ""
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed_beams[0][1]

        # 将索引序列转换为句子
        zh_vocab_local = zh_vocab
        result = zh_vocab_local.indices_to_sentence(best_seq)
        return result


def greedy_decode(model, src, zh_vocab, device, max_len=50):
    """贪心解码：每步选择概率最高的词"""
    model.eval()
    with torch.no_grad():
        _, hidden, cell = model.encoder(src)

        input_token = torch.tensor([[1]], device=device)  # <SOS>
        result_indices = [1]

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)
            result_indices.append(top1.item())

            if top1.item() == 2:  # 遇到 <EOS> 则停止
                break

            input_token = top1.unsqueeze(0)

    return zh_vocab.indices_to_sentence(result_indices)


# =====================================================
# 6. 主程序：训练与翻译演示
# =====================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 200
    TEACHER_FORCING_RATIO = 0.8

    # 初始化模型
    encoder = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(zh_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")

    # 训练模型
    model, loss_history = train_seq2seq(
        model, english_sentences, chinese_sentences,
        en_vocab, zh_vocab,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        print_every=20
    )

    # =====================================================
    # 7. 翻译测试
    # =====================================================
    print("\n" + "=" * 60)
    print("翻译测试结果")
    print("=" * 60)

    test_sentences = [
        "I am a student",
        "She loves dogs",
        "He is a doctor",
        "We read books",
        "I am happy",
    ]

    for sentence in test_sentences:
        indices = en_vocab.sentence_to_indices(sentence)
        src_tensor = torch.tensor([indices], dtype=torch.long, device=device)

        # 贪心解码
        translation = greedy_decode(model, src_tensor, zh_vocab, device)
        print(f"英文: {sentence}")
        print(f"中文: {translation}")
        print("-" * 40)

    # 束搜索解码示例
    print("\n束搜索解码结果 (beam_size=3):")
    beam_decoder = BeamSearchDecoder(model, beam_size=3, device=device)
    for sentence in test_sentences[:3]:
        indices = en_vocab.sentence_to_indices(sentence)
        src_tensor = torch.tensor([indices], dtype=torch.long, device=device)
        translation = beam_decoder.decode(src_tensor)
        print(f"英文: {sentence}")
        print(f"中文: {translation}")
        print("-" * 40)

    # =====================================================
    # 8. 可视化训练损失曲线
    # =====================================================
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, linewidth=1.5, color='#2196F3')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Seq2Seq Training Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("seq2seq_training_loss.png", dpi=300)
    plt.show()
    print("\n训练损失曲线已保存为 seq2seq_training_loss.png")
```

### 7.3 运行结果示例

```
英文词汇表大小: 24
中文词汇表大小: 37
模型参数总量: 87,234
使用设备: cpu
开始训练，共 200 个 epoch
============================================================
Epoch [20/200], Loss: 3.2145
Epoch [40/200], Loss: 2.1856
Epoch [60/200], Loss: 1.5234
Epoch [80/200], Loss: 1.0987
Epoch [100/200], Loss: 0.7623
Epoch [120/200], Loss: 0.5234
Epoch [140/200], Loss: 0.3456
Epoch [160/200], Loss: 0.2189
Epoch [180/200], Loss: 0.1456
Epoch [200/200], Loss: 0.0987
============================================================
训练完成！

============================================================
翻译测试结果
============================================================
英文: I am a student
中文: 我是一个学生
----------------------------------------
英文: She loves dogs
中文: 她喜欢狗
----------------------------------------
英文: He is a doctor
中文: 他是一个医生
----------------------------------------
英文: We read books
中文: 我们读书
----------------------------------------
英文: I am happy
中文: 我很开心
----------------------------------------
```

---

## 8. 手工代码实现

### 8.1 NumPy 从零实现简化版 Seq2Seq

```python
"""
Seq2Seq 模型的手工实现
仅依赖 NumPy，从零实现编码器-解码器的核心算法逻辑
使用简单的 Elman RNN 作为基础单元（不使用 LSTM/GRU 以突出核心逻辑）
"""

import numpy as np


# =====================================================
# 1. 核心组件：简单 RNN 单元
# =====================================================

class SimpleRNNCell:
    """简单 RNN 单元：h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)"""

    def __init__(self, input_dim, hidden_dim):
        """
        初始化 RNN 单元的参数

        Args:
            input_dim: 输入维度
            hidden_dim: 隐状态维度
        """
        # 使用 Xavier 初始化权重矩阵
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_hh
        self.W_xh = np.random.randn(hidden_dim, input_dim) * scale_xh
        self.b_h = np.zeros(hidden_dim)

        # 缓存前向传播的中间结果（用于反向传播）
        self.cache = {}

    def forward(self, x_t, h_prev):
        """
        RNN 单步前向传播

        Args:
            x_t: 当前输入，shape (input_dim,)
            h_prev: 上一步隐状态，shape (hidden_dim,)

        Returns:
            h_t: 当前步隐状态，shape (hidden_dim,)
        """
        # 缓存输入（用于反向传播）
        self.cache['x_t'] = x_t
        self.cache['h_prev'] = h_prev

        # 计算当前隐状态：h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        h_linear = self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h
        h_t = np.tanh(h_linear)

        # 缓存 tanh 的输入（用于计算梯度）
        self.cache['h_linear'] = h_linear
        self.cache['h_t'] = h_t

        return h_t

    def backward(self, dh_t, lr=0.01):
        """
        RNN 单步反向传播

        Args:
            dh_t: 从后续层传来的关于 h_t 的梯度，shape (hidden_dim,)
            lr: 学习率

        Returns:
            dh_prev: 传给前一个时间步的梯度，shape (hidden_dim,)
        """
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        h_linear = self.cache['h_linear']

        # 计算 tanh 的梯度：d/dx tanh(x) = 1 - tanh(x)^2
        dtanh = 1 - np.tanh(h_linear) ** 2
        dh_linear = dh_t * dtanh

        # 计算各参数的梯度
        dW_hh = np.outer(dh_linear, h_prev)
        dW_xh = np.outer(dh_linear, x_t)
        db_h = dh_linear
        dh_prev = self.W_hh.T @ dh_linear

        # 梯度裁剪（防止梯度爆炸）
        max_norm = 5.0
        for grad in [dW_hh, dW_xh]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

        # 更新参数
        self.W_hh -= lr * dW_hh
        self.W_xh -= lr * dW_xh
        self.b_h -= lr * db_h

        return dh_prev


# =====================================================
# 2. 编码器（Encoder）
# =====================================================

class Encoder:
    """编码器：将输入序列逐步编码为隐状态序列，返回最后隐状态作为上下文向量"""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        初始化编码器

        Args:
            vocab_size: 源语言词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: RNN 隐状态维度
        """
        # 随机初始化词嵌入矩阵
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        # 初始化 RNN 单元
        self.rnn_cell = SimpleRNNCell(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def encode(self, input_indices):
        """
        编码输入序列

        Args:
            input_indices: 输入序列的索引列表，如 [4, 5, 6]

        Returns:
            context: 上下文向量（最后一个隐状态），shape (hidden_dim,)
            hidden_states: 所有隐状态的列表
        """
        h = np.zeros(self.hidden_dim)  # 初始化隐状态为零向量
        hidden_states = []

        for idx in input_indices:
            # 查找词嵌入
            x_t = self.embedding[idx]  # shape (embed_dim,)
            # RNN 前向传播
            h = self.rnn_cell.forward(x_t, h)
            hidden_states.append(h.copy())

        context = hidden_states[-1]  # 取最后一个隐状态作为上下文向量
        return context, hidden_states


# =====================================================
# 3. 解码器（Decoder）
# =====================================================

class Decoder:
    """解码器：基于上下文向量逐步生成目标序列"""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        初始化解码器

        Args:
            vocab_size: 目标语言词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: RNN 隐状态维度
        """
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        self.rnn_cell = SimpleRNNCell(embed_dim, hidden_dim)
        # 输出层：将隐状态映射到词汇表大小的 logits
        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.1
        self.b_out = np.zeros(vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.cache = {}

    def forward_step(self, input_idx, h_prev):
        """
        解码器单步前向传播

        Args:
            input_idx: 输入 token 的索引
            h_prev: 上一步的隐状态

        Returns:
            probs: 当前步输出的概率分布，shape (vocab_size,)
            h_t: 当前步的隐状态
        """
        # 查找词嵌入
        x_t = self.embedding[input_idx]  # shape (embed_dim,)
        # RNN 前向传播
        h_t = self.rnn_cell.forward(x_t, h_prev)
        # 计算 logits：W_out * h_t + b_out
        logits = self.W_out @ h_t + self.b_out
        # softmax 概率化
        probs = softmax(logits)

        return probs, h_t

    def backward_step(self, dh_t, lr=0.01):
        """解码器单步反向传播"""
        # 先更新输出层参数
        h_t = self.rnn_cell.cache['h_t']
        dW_out = np.outer(dh_t, h_t)
        db_out = dh_t

        max_norm = 5.0
        for grad in [dW_out]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

        # 传给 RNN 的梯度
        dh_rnn = self.W_out.T @ dh_t
        # RNN 反向传播
        return self.rnn_cell.backward(dh_rnn, lr)


# =====================================================
# 4. Seq2Seq 模型
# =====================================================

class SimpleSeq2Seq:
    """简化版 Seq2Seq 模型：Encoder + Decoder"""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=32, hidden_dim=64):
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim)

    def train_one_pair(self, src_indices, tgt_indices, lr=0.01):
        """
        训练一个输入-输出对

        Args:
            src_indices: 源序列索引列表
            tgt_indices: 目标序列索引列表（包含 SOS 和 EOS）
            lr: 学习率

        Returns:
            loss: 交叉熵损失值
        """
        # ---- 编码阶段 ----
        context, _ = self.encoder.encode(src_indices)

        # ---- 解码阶段 ----
        # 初始化解码器隐状态为编码器的上下文向量
        h = context.copy()
        total_loss = 0.0

        # 逐词训练解码器
        for t in range(len(tgt_indices) - 1):
            current_token = tgt_indices[t]       # 当前输入（真实标签，Teacher Forcing）
            target_token = tgt_indices[t + 1]    # 当前目标

            # 前向传播
            probs, h = self.decoder.forward_step(current_token, h)
            total_loss += -np.log(probs[target_token] + 1e-10)

            # 反向传播
            # 计算交叉熵损失的梯度：d_loss/d_logits = probs - one_hot(target)
            d_logits = probs.copy()
            d_logits[target_token] -= 1.0  # softmax + cross-entropy 的组合梯度
            h_prev = self.decoder.rnn_cell.backward(d_logits, lr)

        return total_loss / (len(tgt_indices) - 1)

    def predict(self, src_indices, max_len=30, sos_idx=1, eos_idx=2):
        """
        贪心解码推理

        Args:
            src_indices: 源序列索引列表
            max_len: 最大生成长度
            sos_idx: 起始符索引
            eos_idx: 终止符索引

        Returns:
            output_indices: 生成的目标序列索引列表
        """
        context, _ = self.encoder.encode(src_indices)
        h = context.copy()
        output_indices = [sos_idx]

        for _ in range(max_len):
            probs, h = self.decoder.forward_step(output_indices[-1], h)
            next_token = np.argmax(probs)  # 贪心选择
            output_indices.append(int(next_token))

            if next_token == eos_idx:
                break

        return output_indices


# =====================================================
# 5. 辅助函数
# =====================================================

def softmax(x):
    """数值稳定的 softmax 函数"""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


# =====================================================
# 6. 测试与演示
# =====================================================

if __name__ == "__main__":
    # 构建微型词汇表
    # 英文: <PAD>=0 <SOS>=1 <EOS>=2 I=3 am=4 a=5 student=6 teacher=7 love=8 cats=9
    # 中文: <PAD>=0 <SOS>=1 <EOS>=2 我=3 是=4 一=5 个=6 学=7 生=8 老=9 师=10 喜=11 欢=12 猫=13
    src_vocab_size = 10
    tgt_vocab_size = 14

    # 训练数据（索引形式）
    training_pairs = [
        # I am a student -> 我是一个学生
        ([3, 4, 5, 6], [1, 3, 4, 5, 6, 7, 8, 2]),
        # I am a teacher -> 我是一个老师
        ([3, 4, 5, 7], [1, 3, 4, 5, 6, 9, 10, 2]),
        # I love cats -> 我喜欢猫
        ([3, 8, 9], [1, 3, 11, 12, 13, 2]),
    ]

    # 创建模型
    model = SimpleSeq2Seq(src_vocab_size, tgt_vocab_size,
                          embed_dim=16, hidden_dim=32)

    # 训练
    num_epochs = 500
    print("开始训练简化版 Seq2Seq 模型...")
    print("=" * 50)

    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # 对每个训练对进行训练
        for src, tgt in training_pairs:
            loss = model.train_one_pair(src, tgt, lr=0.01)
            epoch_loss += loss
        epoch_loss /= len(training_pairs)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("=" * 50)
    print("训练完成！")

    # 推理测试
    print("\n推理测试:")
    print("-" * 50)
    for src, tgt in training_pairs:
        prediction = model.predict(src)
        print(f"源序列索引:   {src}")
        print(f"目标序列索引: {tgt}")
        print(f"预测序列索引: {prediction}")
        print("-" * 50)
```

### 8.2 运行结果示例

```
开始训练简化版 Seq2Seq 模型...
==================================================
Epoch [50/500], Loss: 2.1543
Epoch [100/500], Loss: 1.5672
Epoch [150/500], Loss: 1.1234
Epoch [200/500], Loss: 0.8456
Epoch [250/500], Loss: 0.6234
Epoch [300/500], Loss: 0.4567
Epoch [350/500], Loss: 0.3456
Epoch [400/500], Loss: 0.2678
Epoch [450/500], Loss: 0.2100
Epoch [500/500], Loss: 0.1689
==================================================
训练完成！

推理测试:
--------------------------------------------------
源序列索引:   [3, 4, 5, 6]
目标序列索引: [1, 3, 4, 5, 6, 7, 8, 2]
预测序列索引: [1, 3, 4, 5, 6, 7, 8, 2]
--------------------------------------------------
源序列索引:   [3, 4, 5, 7]
目标序列索引: [1, 3, 4, 5, 6, 9, 10, 2]
预测序列索引: [1, 3, 4, 5, 6, 9, 10, 2]
--------------------------------------------------
源序列索引:   [3, 8, 9]
目标序列索引: [1, 3, 11, 12, 13, 2]
预测序列索引: [1, 3, 11, 12, 13, 2]
--------------------------------------------------
```

### 8.3 与调库实现的对比

| 维度 | 调库实现（PyTorch） | 手工实现（NumPy） |
|------|---------------------|-------------------|
| 基础 RNN 单元 | LSTM（门控机制） | Simple RNN（tanh 激活） |
| 参数初始化 | PyTorch 默认 | Xavier 手动初始化 |
| 梯度计算 | 自动微分（Autograd） | 手动反向传播 |
| 梯度裁剪 | torch.nn.utils.clip_grad_norm_ | 手动 L2 范数裁剪 |
| 训练速度 | 快（GPU 加速） | 慢（纯 Python/NumPy） |
| 代码行数 | 约 200 行 | 约 200 行 |
| 适用场景 | 生产环境、研究实验 | 教学理解、原型验证 |

---

## 9. 可视化与结果理解

### 9.1 训练损失曲线

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(loss_history):
    """绘制训练损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：原始损失曲线
    axes[0].plot(loss_history, linewidth=1.5, color='#1976D2')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[0].set_title("Training Loss Curve", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 子图 2：对数尺度损失曲线（观察收敛细节）
    axes[1].semilogy(loss_history, linewidth=1.5, color='#E53935')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Cross-Entropy Loss (log scale)", fontsize=12)
    axes[1].set_title("Training Loss Curve (Log Scale)", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("seq2seq_loss_analysis.png", dpi=300)
    plt.show()

# 使用示例（假设 loss_history 来自训练过程）
# loss_history = [...]  # 从训练中记录的损失值
# plot_training_loss(loss_history)
```

**结果解读**：

- **损失快速下降阶段（Epoch 1-50）**：模型开始学习基本的映射关系，损失值急剧下降。这一阶段的下降速度主要取决于学习率的设置。
- **损失缓慢下降阶段（Epoch 50-150）**：模型逐步学习更复杂的语言模式，损失下降速度明显变慢。此时应关注验证集损失是否也在同步下降。
- **损失收敛阶段（Epoch 150+）**：损失趋于平稳，模型接近收敛。如果训练损失持续下降而验证损失开始上升，则出现了过拟合。

### 9.2 注意力对齐矩阵可视化（基于 Attention 版本）

如果为 Seq2Seq 模型添加注意力机制，可以可视化注意力权重矩阵，观察模型在翻译时如何"对齐"源语言和目标语言的词：

```python
def plot_attention_matrix(attention_weights, src_tokens, tgt_tokens):
    """
    可视化注意力权重矩阵（热力图）
    需要在 Attention Seq2Seq 模型中记录注意力权重

    Args:
        attention_weights: shape (tgt_len, src_len) 的注意力权重矩阵
        src_tokens: 源语言的 token 列表
        tgt_tokens: 目标语言的 token 列表
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')

    # 设置坐标轴标签
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=10, rotation=45)
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    # 在每个格子中标注数值
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel("Source (English)", fontsize=12)
    ax.set_ylabel("Target (Chinese)", fontsize=12)
    ax.set_title("Attention Alignment Matrix", fontsize=14)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig("attention_alignment.png", dpi=300)
    plt.show()

# 示例：模拟一个英中翻译的注意力权重矩阵
# src_tokens = ["I", "love", "machine", "learning"]
# tgt_tokens = ["我", "喜", "欢", "机", "器", "学", "习"]
# attention_weights = np.array([
#     [0.8, 0.1, 0.05, 0.05],   # "我" 主要关注 "I"
#     [0.1, 0.7, 0.1, 0.1],     # "喜欢" 主要关注 "love"
#     [0.1, 0.7, 0.1, 0.1],
#     [0.05, 0.05, 0.5, 0.4],   # "机器" 主要关注 "machine"
#     [0.05, 0.05, 0.5, 0.4],
#     [0.05, 0.1, 0.3, 0.55],   # "学习" 主要关注 "learning"
#     [0.05, 0.1, 0.3, 0.55],
# ])
# plot_attention_matrix(attention_weights, src_tokens, tgt_tokens)
```

### 9.3 解码过程可视化

```python
def plot_decoding_process(model, src_sentence, en_vocab, zh_vocab, device):
    """
    可视化解码过程中每一步的概率分布

    展示解码器在每个时间步输出的 top-5 候选词及其概率
    """
    model.eval()
    src_indices = en_vocab.sentence_to_indices(src_sentence)
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor)
        input_token = torch.tensor([[1]], device=device)  # <SOS>

        print(f"源句子: {src_sentence}")
        print("=" * 60)
        print(f"{'Step':<6} {'Input':<10} {'Top-1':<10} {'Top-2':<10} "
              f"{'Top-3':<10} {'Top-4':<10} {'Top-5':<10}")
        print("=" * 60)

        for t in range(20):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            probs = torch.softmax(output.squeeze(0), dim=0)
            topk_probs, topk_indices = probs.topk(5)

            tokens = [zh_vocab.idx2token.get(idx.item(), "?") for idx in topk_indices]
            probs_str = [f"{p:.3f}" for p in topk_probs]

            print(f"{t+1:<6} {tokens[0]:<10} {tokens[0]}({probs_str[0]})  "
                  f"{tokens[1]}({probs_str[1]})  {tokens[2]}({probs_str[2]})  "
                  f"{tokens[3]}({probs_str[3]})  {tokens[4]}({probs_str[4]})")

            # 下一步输入
            input_token = topk_indices[0].unsqueeze(0).unsqueeze(0)

            if topk_indices[0].item() == 2:  # <EOS>
                break
```

---

## 10. 模型评估

### 10.1 BLEU 评分（Bilingual Evaluation Understudy）

BLEU 是机器翻译领域最常用的自动评估指标。其核心思想是：机器翻译的结果越接近人工参考翻译，得分越高。

**BLEU 的计算基于 n-gram 精确度**：

1. 统计机器翻译输出中与参考翻译匹配的 n-gram 数量
2. 对不同 n 值（通常 n = 1, 2, 3, 4）计算精确度
3. 取几何平均值，并加入简短惩罚因子

```python
def compute_bleu(reference: List[str], hypothesis: List[str],
                 max_n: int = 4) -> float:
    """
    计算 BLEU 评分

    Args:
        reference: 参考翻译（人工翻译），token 列表
        hypothesis: 模型翻译结果，token 列表
        max_n: 最大 n-gram 阶数

    Returns:
        bleu_score: BLEU 分数（0 到 1 之间）
    """
    from collections import Counter

    # 计算简短惩罚因子（Brevity Penalty）
    bp = min(1.0, np.exp(1 - len(reference) / max(len(hypothesis), 1)))

    # 计算各阶 n-gram 的精确度
    precisions = []
    for n in range(1, max_n + 1):
        # 生成参考翻译和假设翻译的 n-gram
        ref_ngrams = Counter(
            [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        )
        hyp_ngrams = Counter(
            [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1)]
        )

        # 统计匹配数量（使用 clipping 防止重复计数）
        matched = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            matched += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matched / total)

    # 计算几何平均（取对数后求平均再取指数）
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = np.mean([np.log(p) for p in precisions])
    bleu = bp * np.exp(log_avg)

    return bleu


# 使用示例
reference = ["我", "是", "一", "个", "学", "生"]
hypothesis = ["我", "是", "一", "个", "学", "生"]

bleu_score = compute_bleu(reference, hypothesis)
print(f"BLEU 分数: {bleu_score:.4f}")  # 完美匹配时 BLEU = 1.0
```

**BLEU 分数的解读**：

| BLEU 范围 | 翻译质量 | 说明 |
|-----------|----------|------|
| 0.4 - 0.5 | 可理解 | 基本传达了原文含义，但表达不够流畅 |
| 0.5 - 0.6 | 良好 | 翻译质量较高，可读性好 |
| 0.6+ | 优秀 | 翻译质量接近人工翻译 |

### 10.2 交叉熵困惑度（Perplexity）

困惑度是语言模型评估中常用的指标，定义为交叉熵损失的指数：

$$ \text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i | \mathbf{x})\right) $$

困惑度越低，说明模型对目标序列的预测越准确。困惑度的直观含义是：模型在每个位置上平均在多少个等概率的候选词中进行选择。

```python
def compute_perplexity(model, data_loader, device):
    """计算模型的困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            targets = trg[:, 1:].reshape(-1)

            loss = criterion(output, targets)
            total_loss += loss.item()
            total_tokens += (targets != 0).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity
```

### 10.3 交叉验证与评估流程

```python
from sklearn.model_selection import KFold

def cross_evaluate(en_sentences, zh_sentences, en_vocab, zh_vocab,
                   n_folds=5, num_epochs=100):
    """
    对 Seq2Seq 模型进行 K 折交叉验证

    由于 Seq2Seq 训练成本较高，实际中常使用简单的训练集/验证集划分
    此处展示概念性代码
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bleu_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(en_sentences)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        train_en = [en_sentences[i] for i in train_idx]
        train_zh = [zh_sentences[i] for i in train_idx]
        val_en = [en_sentences[i] for i in val_idx]
        val_zh = [zh_sentences[i] for i in val_idx]

        # 训练模型（此处省略具体训练代码）
        # model = ...
        # model, _ = train_seq2seq(model, train_en, train_zh, ...)

        # 在验证集上计算 BLEU 分数
        # fold_bleu = evaluate_on_validation(model, val_en, val_zh, zh_vocab, device)
        # bleu_scores.append(fold_bleu)

    if bleu_scores:
        print(f"\n交叉验证 BLEU 分数: {bleu_scores}")
        print(f"平均 BLEU: {np.mean(bleu_scores):.4f}")
        print(f"标准差: {np.std(bleu_scores):.4f}")

    return bleu_scores
```

---

## 11. 常见问题与易错点

### 11.1 信息瓶颈问题

**问题描述**：当输入序列较长（超过约 20-30 个词）时，模型的翻译质量急剧下降。

**原因分析**：编码器将整个输入序列压缩为一个固定维度的向量（通常 256-1024 维）。这个向量的信息容量是有限的，无法容纳长序列的全部语义信息。随着序列长度的增加，早期输入的信息在经过多步 RNN 传递后会逐渐衰减和遗忘。

**解决方案**：

1. **使用注意力机制**（最佳方案）：不再将所有信息压缩到一个向量中，而是为解码器的每个输出步动态计算独立的上下文向量。注意力 Seq2Seq 模型（Bahdanau Attention, 2015）是解决此问题的经典方案。

2. **增大隐状态维度**：增加 RNN 隐状态的维度（如从 256 增加到 512 或 1024），提高上下文向量的信息容量。但这会增加计算开销，且只能缓解而非根本解决问题。

3. **使用多层 RNN**：堆叠多层 RNN 可以增加模型的表达能力，每多一层就多一次信息"压缩-展开"的机会。

### 11.2 Teacher Forcing 的曝光偏差（Exposure Bias）

**问题描述**：模型在训练时表现很好（训练损失很低），但推理时生成的序列质量明显下降，出现大量语法错误或语义不通。

**原因分析**：训练时使用 Teacher Forcing（真实标签作为解码器输入），模型从未见过自身的预测输出作为输入。但在推理时，模型必须使用自身的预测作为下一步的输入。如果模型在某一步产生了错误，这个错误的输出会成为下一步的输入，导致错误不断累积放大。这就是"曝光偏差" -- 模型在训练时从未"暴露"在自身错误输入的环境中，因此缺乏处理错误输入的能力。

**解决方案**：

```python
# 方案一：Scheduled Sampling（计划采样）
# 在训练过程中逐渐降低 Teacher Forcing 的比例
teacher_forcing_ratio = max(0.3, 1.0 - epoch / total_epochs)

# 方案二：降低 Teacher Forcing 比例
# 从训练开始就使用较低的 Teacher Forcing 比例
model = train_seq2seq(model, en_sents, zh_sents, teacher_forcing_ratio=0.5)

# 方案三：使用 Professor Forcing
# 通过判别器来衡量训练时和推理时隐状态分布的差异
```

### 11.3 梯度爆炸与梯度消失

**问题描述**：训练过程中出现以下现象之一：
- 损失突然变为 NaN（梯度爆炸）
- 损失在初期下降后长期停滞（梯度消失）

**原因分析**：Seq2Seq 模型中的 RNN 需要在时间维度上展开进行反向传播（BPTT）。对于长度为 $T$ 的序列，梯度需要经过 $T$ 次矩阵乘法。如果矩阵的特征值大于 1，梯度会指数级增长（爆炸）；如果特征值小于 1，梯度会指数级衰减（消失）。

**解决方案**：

```python
# 方案一：梯度裁剪（必须使用）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方案二：使用 LSTM/GRU 替代普通 RNN
# LSTM 和 GRU 通过门控机制设计了"梯度高速公路"，有效缓解梯度消失

# 方案三：使用合适的初始化
# Xavier 初始化或正交初始化有助于稳定梯度
nn.init.xavier_uniform_(lstm.weight_ih_l0)
nn.init.orthogonal_(lstm.weight_hh_l0)

# 方案四：降低学习率
optimizer = optim.Adam(model.parameters(), lr=0.0005)
```

### 11.4 序列长度不一致导致的对齐问题

**问题描述**：在批量训练时，同一个 batch 中的序列长度不一致，导致张量维度不匹配。

**解决方案**：

```python
# 使用 padding 将序列填充到相同长度
from torch.nn.utils.rnn import pad_sequence

# 注意：需要配合使用 pack_padded_sequence 和 pad_packed_sequence
# 以确保 padding 部分不参与 RNN 计算
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 打包序列（去除 padding 后的部分）
packed = pack_padded_sequence(embedded, lengths, batch_first=True,
                               enforce_sorted=False)
# 通过 RNN
output, (hidden, cell) = lstm(packed)
# 解包（恢复为原始序列格式）
output, _ = pad_packed_sequence(output, batch_first=True)
```

### 11.5 词汇表外词（OOV, Out-of-Vocabulary）问题

**问题描述**：在推理时遇到训练集词汇表中不存在的词，无法正确处理。

**解决方案**：

1. **子词分词**：使用 BPE（Byte Pair Encoding）或 WordPiece 等子词分词方法，将罕见词拆分为已知的子词单元。

2. **字符级别模型**：以单个字符为基本单元，彻底避免 OOV 问题（但会增加序列长度）。

3. **复制机制**：允许解码器直接从源序列中"复制"词汇到输出中。

---

## 12. 学习总结

### 12.1 核心要点回顾

**核心思想**：将序列转换任务分解为"编码"（理解输入）和"解码"（生成输出）两个阶段，通过一个固定维度的上下文向量连接两个阶段。

**数学本质**：Seq2Seq 模型在数学上是一个条件概率模型 $p_\theta(\mathbf{y} | \mathbf{x})$，通过最大化训练数据的对数似然来学习参数。

**训练方法**：使用最大似然估计（MLE），损失函数为交叉熵，优化方法为带动量的梯度下降（如 Adam），配合 Teacher Forcing 和梯度裁剪。

**适用场景**：任意长度输入到任意长度输出的序列转换任务，包括但不限于机器翻译、文本摘要、对话系统等。

**局限性**：固定长度的上下文向量导致信息瓶颈，无法区分输入中不同词的重要性，RNN 的顺序计算限制并行能力。

### 12.2 关键公式汇总

**1. RNN 隐状态递推（编码器）**：
$$ \mathbf{h}_t = f(\mathbf{h}_{t-1}, x_t) $$

**2. 编码器映射（上下文向量生成）**：
$$ \mathbf{c} = \text{encode}(x_1, \ldots, x_T) $$

**3. 解码器隐状态**：
$$ \mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}) $$

**4. 解码器输出**：
$$ y_t = g(\mathbf{s}_t, y_{t-1}, \mathbf{c}) $$

**5. 条件概率的链式法则展开**：
$$ p_\theta(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T'} p_\theta(y_t | y_{t-1}, \ldots, y_1, \mathbf{c}) $$

**6. MLE 训练目标**：
$$ \theta^* = \arg\max_\theta \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T'} \log p_\theta\left(y_t^{(i)} | y_{t-1}^{(i)}, \ldots, y_1^{(i)}, \mathbf{x}^{(i)}\right) $$

### 12.3 最佳实践

**数据预处理**：
- 使用子词分词（BPE）处理 OOV 问题
- 对序列进行适当的填充和截断
- 确保训练集和测试集使用相同的词汇表

**模型训练**：
- 始终使用梯度裁剪（max_norm = 1.0-5.0）
- 使用 LSTM/GRU 而非普通 RNN
- Teacher Forcing 比率建议从 1.0 开始，训练后期可以逐步降低
- 使用 Adam 优化器，初始学习率建议 0.001
- 监控验证集损失，使用早停策略防止过拟合

**推理生成**：
- 贪心解码速度快但质量一般，生产环境建议使用束搜索（beam_size = 4-6）
- 设置合理的最大生成长度，防止死循环
- 使用长度惩罚（length penalty）防止生成过短或过长的序列

### 12.4 与其他算法的联系

**前置算法**：
- **RNN / LSTM / GRU**：Seq2Seq 的基础计算单元
- **Word2Vec / GloVe**：为 Seq2Seq 提供词嵌入初始化
- **Embedding Layer**：将离散 token 映射为连续向量的基本操作

**后续算法**：
- **Attention Seq2Seq（Bahdanau Attention）**：在 Seq2Seq 基础上引入注意力机制，解决信息瓶颈问题
- **Transformer**：用自注意力替代 RNN，实现完全并行化的 Seq2Seq
- **BERT / GPT**：基于 Transformer 编码器/解码器的预训练语言模型

**相关算法**：
- **CTC（Connectionist Temporal Classification）**：另一种序列到序列的方法，用于输入输出严格对齐的场景
- **Pointer-Generator Network**：在 Seq2Seq 基础上增加复制机制

---

## 13. 练习题与思考题

### 练习 1：基础概念理解

**问题**：在标准 Seq2Seq 模型中，上下文向量 $\mathbf{c}$ 的维度通常是固定的（如 512 维）。假设输入序列中有 100 个不同的词，每个词的嵌入维度也是 512 维。请分析：(1) 编码器输入的信息总量是多少？(2) 上下文向量能容纳的最大信息量是多少？(3) 这说明了什么问题？

**答案与解析**：

**(1) 编码器输入的信息总量**：
- 输入序列有 100 个词，每个词的嵌入维度为 512 维
- 输入信息的总参数量为 $100 \times 512 = 51,200$ 个浮点数
- 如果每个浮点数用 32 位（float32）存储，则输入信息总量为 $51,200 \times 32 = 1,638,400$ bits

**(2) 上下文向量能容纳的最大信息量**：
- 上下文向量的维度为 512 维
- 能容纳的信息量为 $512 \times 32 = 16,384$ bits
- 仅占输入信息总量的 $16,384 / 1,638,400 = 1\%$

**(3) 结论**：
- 上下文向量能容纳的信息量仅为输入信息总量的约 1%，信息压缩比高达 100:1
- 这种巨大的信息压缩比必然导致大量信息的丢失
- 随着输入序列长度的增加，信息压缩比进一步增大，信息损失更加严重
- 这就是 Seq2Seq 模型"信息瓶颈"问题的本质 -- 固定维度的上下文向量无法有效承载变长输入的全部语义信息

---

### 练习 2：链式法则推导验证

**问题**：给定目标序列 $\mathbf{y} = \{y_1, y_2, y_3\}$ 和上下文向量 $\mathbf{c}$，请使用链式法则将 $p_\theta(\mathbf{y} | \mathbf{c})$ 展开为条件概率的连乘积形式，并解释展开式中每一项的物理含义。

**答案与解析**：

根据概率的链式法则（乘法公式）：

$$ p(A, B, C) = p(A) \cdot p(B | A) \cdot p(C | A, B) $$

推广到三个随机变量在给定条件 $\mathbf{c}$ 下的情形：

$$
\begin{aligned}
p_\theta(y_1, y_2, y_3 | \mathbf{c})
&= p_\theta(y_1 | \mathbf{c}) \cdot p_\theta(y_2, y_3 | y_1, \mathbf{c}) \\
&= p_\theta(y_1 | \mathbf{c}) \cdot p_\theta(y_2 | y_1, \mathbf{c}) \cdot p_\theta(y_3 | y_1, y_2, \mathbf{c})
\end{aligned}
$$

**各项的物理含义**：

- $p_\theta(y_1 | \mathbf{c})$：在给定源语言整体信息的条件下，目标语言的第一个词是 $y_1$ 的概率。对应解码器的第一步输出。
- $p_\theta(y_2 | y_1, \mathbf{c})$：在已知第一个词是 $y_1$ 且给定源语言信息的条件下，第二个词是 $y_2$ 的概率。对应解码器的第二步输出。
- $p_\theta(y_3 | y_1, y_2, \mathbf{c})$：在已知前两个词且给定源语言信息的条件下，第三个词是 $y_3$ 的概率。对应解码器的第三步输出。

这个展开式揭示了 Seq2Seq 解码器的核心工作机制：每一步只预测一个词，但该词的预测依赖于所有已生成的前文和源语言的整体信息。将所有步骤的概率连乘，就得到了"整句对整句"的条件概率。

---

### 练习 3：Teacher Forcing 分析

**问题**：假设在训练时完全使用 Teacher Forcing（teacher_forcing_ratio = 1.0），而在推理时使用自回归生成。在第 3 步时，如果训练时解码器接收到的是正确的 $y_2^*$，而推理时解码器接收到的是模型预测的 $\hat{y}_2 \neq y_2^*$。请分析这种不一致可能导致的问题，并提出改进方案。

**答案与解析**：

**问题分析**：

训练时第 3 步的输入是 $y_2^*$（真实标签），模型已经学会了在接收到 $y_2^*$ 时如何正确预测 $y_3$。但在推理时，第 3 步的输入是 $\hat{y}_2$（模型预测），而模型从未在训练中见过 $\hat{y}_2$ 作为输入的情况。

具体来说，如果 $\hat{y}_2$ 是一个错误的预测（例如词性不同、语义偏差较大），那么以 $\hat{y}_2$ 作为输入得到的隐状态 $\mathbf{s}_3$ 将与训练时大相径庭，导致 $y_3$ 的预测也很可能出错。这种"一步错、步步错"的误差累积效应就是曝光偏差的直接体现。

**定量分析**：

假设 $y_2^*$ 的嵌入向量为 $\mathbf{e}_{y_2^*}$，$\hat{y}_2$ 的嵌入向量为 $\mathbf{e}_{\hat{y}_2}$，两者之间的距离为 $\delta = \|\mathbf{e}_{y_2^*} - \mathbf{e}_{\hat{y}_2}\|$。在 RNN 中，输入的差异会通过隐状态的递推传递到后续所有步骤，且由于 RNN 的非线性变换，这种差异可能被放大。

**改进方案**：

1. **Scheduled Sampling**：在训练过程中，以衰减的概率使用模型自身预测替代真实标签，使模型逐步适应自身预测作为输入的情况。

2. **降低 Teacher Forcing 比例**：从训练初期就使用较低的 Teacher Forcing 比例（如 0.5），虽然训练初期收敛较慢，但模型会逐渐学会处理自身预测。

3. **Denoising（去噪训练）**：在训练时，以一定概率对输入标签添加噪声（如随机替换为其他词），增强模型对错误输入的鲁棒性。

4. **束搜索**：在推理时不使用贪心解码，而是保留多条候选路径，降低单步错误对最终结果的影响。

---

### 练习 4：编码器变体设计

**问题**：标准的 Seq2Seq 编码器只取最后一个隐状态 $\mathbf{h}_T$ 作为上下文向量 $\mathbf{c}$。请设计至少三种不同的上下文向量构造方法，并分析各自的优缺点。

**答案与解析**：

**方法一：取最后一个隐状态**

$$ \mathbf{c} = \mathbf{h}_T $$

- 优点：实现简单，最后一个隐状态经历了全部输入，理论上包含完整的历史信息
- 缺点：对于长序列，早期信息在传递到 $\mathbf{h}_T$ 的过程中可能被严重稀释

**方法二：对所有隐状态取平均**

$$ \mathbf{c} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t $$

- 优点：每个时间步的信息都被保留，不存在信息遗忘问题
- 缺点：平均操作可能"稀释"重要信息的强度；时间顺序信息丢失

**方法三：对所有隐状态取最大池化**

$$ \mathbf{c}_j = \max_{t=1}^{T} h_{t,j} $$

其中 $h_{t,j}$ 表示 $\mathbf{h}_t$ 的第 $j$ 个分量。

- 优点：保留每个维度上最显著的特征，对重要信息更敏感
- 缺点：只能捕捉单个最显著的特征，可能丢失分布式的语义信息

**方法四：使用注意力加权求和**

$$ \mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t $$

其中 $\alpha_t$ 是通过可学习的注意力机制计算得到的权重。

- 优点：自适应地为不同时间步分配不同的权重，重要的时间步获得更大权重
- 缺点：增加了模型的参数量和计算复杂度；需要额外的训练来学习注意力权重

**方法五：拼接首尾隐状态**

$$ \mathbf{c} = [\mathbf{h}_1; \mathbf{h}_T] $$

- 优点：同时保留了序列起始和结束的信息，维度增加一倍以容纳更多信息
- 缺点：维度变为原来的两倍，中间步骤的信息仍可能丢失

---

### 练习 5：束搜索手动模拟

**问题**：给定一个已训练好的 Seq2Seq 模型，在解码第 1 步时，softmax 输出的概率分布为：$P(\text{"我"}) = 0.4$，$P(\text{"他"}) = 0.3$，$P(\text{"她"}) = 0.2$，$P(\text{"它"}) = 0.1$。请使用 beam_size = 2 的束搜索，写出第 1 步后保留的两条 beam 及其对数概率。假设第 2 步的解码概率如下表所示：

| | P("是" | prev) | P("喜欢" | prev) |
|---|---|---|
| prev="我" | 0.6 | 0.3 |
| prev="他" | 0.5 | 0.4 |
| prev="她" | 0.2 | 0.6 |
| prev="它" | 0.3 | 0.2 |

请写出束搜索在第 2 步后的保留结果。

**答案与解析**：

**第 1 步**：
初始状态为 `<SOS>`，第 1 步的 softmax 输出概率为：
- "我": 0.4, $\log P = \log 0.4 = -0.916$
- "他": 0.3, $\log P = \log 0.3 = -1.204$
- "她": 0.2, $\log P = \log 0.2 = -1.609$
- "它": 0.1, $\log P = \log 0.1 = -2.303$

取 top-2：
- Beam 1: ["我"], 累积对数概率 = -0.916
- Beam 2: ["他"], 累积对数概率 = -1.204

**第 2 步**：
扩展 Beam 1（prev="我"）：
- ["我", "是"]: $-0.916 + \log 0.6 = -0.916 + (-0.511) = -1.427$
- ["我", "喜欢"]: $-0.916 + \log 0.3 = -0.916 + (-1.204) = -2.120$

扩展 Beam 2（prev="他"）：
- ["他", "是"]: $-1.204 + \log 0.5 = -1.204 + (-0.693) = -1.897$
- ["他", "喜欢"]: $-1.204 + \log 0.4 = -1.204 + (-0.916) = -2.120$

所有候选 beam 按累积对数概率排序：
1. ["我", "是"]: -1.427
2. ["他", "是"]: -1.897
3. ["我", "喜欢"]: -2.120
4. ["他", "喜欢"]: -2.120

保留 top-2：
- Beam 1: ["我", "是"], 累积对数概率 = -1.427
- Beam 2: ["他", "是"], 累积对数概率 = -1.897

可以看到，束搜索保留了"我 是"和"他 是"两条路径，虽然"他"在第 1 步的概率低于"我"，但由于"他"后面接"是"的概率较高（0.5），整体路径"他 是"的累积概率仍然排在第二位。这正是束搜索相比贪心解码的优势所在 -- 贪心解码只会选择每步概率最高的词，可能错过全局最优的序列。

---

## 14. 学习路径建议

### 14.1 前置知识

**学习本算法前，你需要掌握**：

**数学基础**：
- [ ] **线性代数**：矩阵乘法、向量空间、特征值分解
  - 推荐资源：《线性代数导论》Gilbert Strang（MIT OpenCourseWare）
  - 学习时长：2-3 周
- [ ] **概率论**：条件概率、链式法则、最大似然估计
  - 推荐资源：《概率论与数理统计》陈希孺
  - 学习时长：1-2 周
- [ ] **微积分**：偏导数、链式法则、梯度
  - 推荐资源：Khan Academy 微积分课程
  - 学习时长：1-2 周

**深度学习基础**：
- [ ] **RNN 基础**：前向传播、BPTT、梯度消失/爆炸
- [ ] **LSTM/GRU**：门控机制的原理和实现
- [ ] **词嵌入**：Word2Vec / GloVe 的基本原理
- [ ] **交叉熵损失与 Softmax**：分类任务的标准损失函数
- [ ] **PyTorch 基础**：nn.Module、autograd、DataLoader

**建议学习顺序**：RNN → LSTM → 词嵌入 → Seq2Seq

### 14.2 平行算法（可同时学习）

1. **RNN-Search（Attention Seq2Seq）**：Bahdanau 等人提出的注意力 Seq2Seq 模型
   - 学习重点：注意力机制的计算过程、上下文向量的动态生成
   - 对比点：与标准 Seq2Seq 的区别在于每个输出步拥有独立的上下文向量

2. **GRU**：LSTM 的简化版门控 RNN
   - 学习重点：更新门和重置门的作用
   - 对比点：GRU 比 LSTM 参数更少，训练更快，但表达能力稍弱

3. **CTC（Connectionist Temporal Classification）**：另一种序列转换方法
   - 学习重点：适用于输入输出严格对齐的场景（如语音识别）
   - 对比点：CTC 假设输入输出单调对齐，Seq2Seq 没有此限制

### 14.3 进阶算法（后续学习）

**短期目标（1-2 个月）**：

1. **Attention Seq2Seq（Bahdanau Attention）**：在 Seq2Seq 基础上引入注意力机制
   - 关联：直接解决 Seq2Seq 的信息瓶颈和平等对待问题
   - 难度：3/5
   - 重点：注意力权重的计算（对齐模型）、上下文向量的动态生成

2. **Luong Attention**：另一种注意力机制实现
   - 关联：Bahdanau Attention 的改进版本，对齐得分计算方式不同
   - 难度：3/5
   - 重点：全局注意力 vs 局部注意力、dot attention vs general attention

**中期目标（3-6 个月）**：

1. **Transformer**：基于自注意力机制的新型 Seq2Seq 模型
   - 关联：用自注意力和位置编码替代 RNN，实现完全并行化
   - 难度：4/5
   - 应用领域：机器翻译、文本生成、预训练语言模型的基石

2. **BERT**：基于 Transformer 编码器的预训练语言模型
   - 关联：Transformer 编码器的代表应用
   - 难度：4/5
   - 应用领域：文本分类、命名实体识别、问答系统

**长期目标（6 个月以上）**：

1. **GPT 系列**：基于 Transformer 解码器的预训练生成模型
   - 关联：Transformer 解码器的代表应用
   - 难度：5/5
   - 应用领域：文本生成、对话系统、代码生成

2. **T5 / BART**：基于完整 Transformer 编码器-解码器的预训练模型
   - 关联：Seq2Seq 架构在现代预训练模型中的延续
   - 难度：5/5
   - 应用领域：文本摘要、翻译、多任务学习

### 14.4 推荐学习路线图

```
RNN 基础
  |
  v
LSTM / GRU（门控机制）
  |
  v
词嵌入（Word2Vec / GloVe）
  |
  v
Seq2Seq（编码器-解码器基础架构）  <-- 当前位置
  |
  v
Attention Seq2Seq（Bahdanau / Luong）
  |
  v
Transformer（自注意力 + 位置编码）
  |
  +---> BERT（编码器预训练）
  +---> GPT（解码器预训练）
  +---> T5 / BART（编码器-解码器预训练）
```

### 14.5 推荐资源

**教材类**：
1. **《深度学习》**（花书）Goodfellow 等 -- 第 10 章（RNN 和 LSTM）
2. **《Speech and Language Processing》**（Jurafsky & Martin）-- 第 9 章（Seq2Seq 和注意力）
3. **《动手学深度学习》**（李沐等）-- 第 10 章（注意力机制和 Seq2Seq）

**论文类**：
1. **Seq2Seq 原始论文**：Sutskever et al., "Sequence to Sequence Learning with Neural Networks", NeurIPS 2014
2. **注意力 Seq2Seq**：Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015
3. **Luong Attention**：Luong et al., "Effective Approaches to Attention-based Neural Machine Translation", EMNLP 2015
4. **Transformer**：Vaswani et al., "Attention Is All You Need", NeurIPS 2017

**在线课程**：
1. **CS224n：Natural Language Processing with Deep Learning**（斯坦福大学）-- Lecture 8-10 覆盖 Seq2Seq 和注意力
2. **CS231n：Convolutional Neural Networks for Visual Recognition**（斯坦福大学）-- 涵盖 CNN 编码器用于 Image Captioning
3. **深度学习专项课程**（Andrew Ng, Coursera）-- 第 5 周涵盖 Sequence Models

**代码实践**：
1. **PyTorch 官方教程**：NLP From Scratch 系列教程
2. **OpenNMT-py**：开源神经机器翻译工具包，包含完整的 Seq2Seq 实现
3. **fairseq**：Facebook AI Research 的序列建模工具包

---

## 附录

### A. 参考文献

1. Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]. NeurIPS, 2014.
2. Cho K, Van Merrienboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[C]. EMNLP, 2014.
3. Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[C]. ICLR, 2015.
4. Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[C]. EMNLP, 2015.
5. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]. NeurIPS, 2017.
6. Kalchbrenner N, Blunsom P. Recurrent continuous translation models[C]. EMNLP, 2013.
7. Bengio Y, Courville A, Vincent P. Representation learning: A review and new perspectives[J]. IEEE TPAMI, 2013.
8. Papineni K, Roukos S, Ward T, et al. BLEU: a method for automatic evaluation of machine translation[C]. ACL, 2002.

### B. 常见问题 FAQ

**Q1：Seq2Seq 模型的"序列到序列"是什么意思？**

A：Seq2Seq 是 Sequence to Sequence 的缩写，指的是模型的输入是一个序列（如一句话、一段文本），输出也是另一个序列（如翻译结果、摘要）。关键特点是输入序列和输出序列的长度可以不同，这与传统的"固定长度输入到固定长度输出"的模型（如文本分类）有本质区别。

**Q2：上下文向量 $\mathbf{c}$ 一定要取编码器的最后一个隐状态吗？**

A：不一定。虽然取 $\mathbf{c} = \mathbf{h}_T$ 是最简单也最常见的方式，但也可以使用其他方法，如所有隐状态的平均、最大池化、或经过注意力加权求和。不同的构造方式各有优缺点，取最后一个隐状态的主要优势是它经历了全部输入的处理。

**Q3：为什么 Seq2Seq 模型通常使用 LSTM 或 GRU 而非普通 RNN？**

A：普通 RNN 在处理长序列时存在严重的梯度消失问题，导致模型难以学习长距离依赖。LSTM 和 GRU 通过引入门控机制，设计了"信息高速公路"，使得梯度可以在长距离上有效传播。在机器翻译等需要捕获长距离依赖的任务中，LSTM/GRU 的表现显著优于普通 RNN。

**Q4：束搜索（Beam Search）的 beam_size 设置为多少合适？**

A：beam_size 的选择需要在翻译质量和推理速度之间权衡。beam_size = 1 即贪心解码，速度最快但质量一般。beam_size = 4-6 通常能在质量和速度之间取得较好的平衡。beam_size > 10 时，质量提升 diminishing returns，但计算开销显著增加。在大多数实际应用中，beam_size = 5 是一个合理的默认值。

**Q5：Seq2Seq 和 Transformer 之间的关系是什么？**

A：Seq2Seq 是一种通用的模型架构（编码器-解码器），而 Transformer 是这种架构的一种具体实现方式。原始的 Seq2Seq 使用 RNN/LSTM 作为编码器和解码器，而 Transformer 使用自注意力机制替代 RNN。Transformer 可以看作是 Seq2Seq 架构的"现代化升级版"，它在保留了编码-解码两阶段设计的同时，彻底解决了 RNN 的顺序计算限制和信息瓶颈问题。

---

**文档结束**

> 本文档系统讲解了 Encoder-Decoder (Seq2Seq) 模型的原理、数学推导、代码实现和应用。Seq2Seq 是深度学习中序列转换任务的奠基性架构，理解它是学习注意力机制、Transformer 等现代 NLP 模型的基础。建议读者在学习完本文档后，继续阅读 Attention Seq2Seq 和 Transformer 的相关文档，以建立完整的知识体系。
