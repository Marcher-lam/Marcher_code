# SwiGLU 激活函数 学习文档

> 来源线索：本节内容根据原书中关于"SwiGLU激活函数"（第4章 4.1.3节）的相关章节整理、扩展与教学化改写。

> 门控机制+Swish激活，让网络选择性传递信息，实现更灵活的非线性表达。

## 1. 算法基础认知

**一句话定义**：SwiGLU（Swish-Gated Linear Unit）是一种基于门控机制的激活函数，它将输入线性变换后的两部分分别用作"信息通道"和"门控信号"，其中门控信号经过 Swish（SiLU）激活，二者逐元素相乘得到最终输出。

**直觉类比**：想象你家的水龙头。SwiGLU 就像一个带有精密阀门的智能水龙头——信息（水流）不是简单地被 ReLU 一刀切（要么全开要么关死），而是通过 Swish 激活形成的"柔性阀门"来平滑控制流量大小。阀门开度不是简单的 0/1，而是介于 0 到输入值之间的连续值，允许网络"部分打开"某些信息通道。更妙的是，这个阀门不是根据水流本身来调节的（那是自门控），而是通过一支独立的控制管路来调节——这就是"学习到的独立门控"。

**历史背景**：门控线性单元（GLU, Gated Linear Unit）最初由 Dauphin 等人在 2017 年的自然语言处理任务中提出，使用 sigmoid 作为门控函数。2020 年，Google 的研究员 Noam Shazeer 在论文《GLU Variants Improve Transformer》中系统比较了多种 GLU 变体，发现使用 Swish 作为门控函数的 SwiGLU 在 Transformer 模型上表现最优。随后，Google 的 PaLM 大模型、Meta 的 LLaMA 系列以及 DeepSeek 系列模型均采用 SwiGLU 作为前馈网络（FFN）的激活函数，使其成为现代大语言模型的标准组件之一。

**算法定位**：SwiGLU 属于深度学习基础组件中的激活函数类别，但它不是一个单纯的逐元素激活函数，而是一个带有可学习参数的门控模块。它通常替代 Transformer 中前馈网络（FFN）的 ReLU/GeLU，配合两个（实际上是三个）线性变换矩阵构成完整的 FFN 层。

**前置知识**：
- 对 ReLU（Rectified Linear Unit, `max(0, x)`）的基本理解，因为 Swish 是 ReLU 的平滑泛化
- 对 GeLU（Gaussian Error Linear Unit）的了解，它是 BERT/GPT 早期常用的激活函数
- sigmoid 函数 `σ(x) = 1/(1+e^{-x})` 的基本概念，因为 Swish 正是 `x * σ(x)`
- 门控机制的基本概念：通过一个 [0,1] 范围的门控信号控制信息流动，类似 LSTM/GRU 中的门

## 2. 核心原理

**核心思想**：SwiGLU 的核心思想是"选择性信息传递"——不是所有输入信息都同等重要。通过引入一个独立的门控分支，网络可以学习"什么时候让信息通过、什么时候抑制信息"。门控函数使用 Swish（自门控的平滑激活），使得门控信号的过渡更加连续和平滑，梯度流动更好。

传统激活函数（如 ReLU、GeLU）是"自门控"的——它们用自己的值决定是否激活。例如 ReLU 中，`x > 0` 就输出 `x`，`x <= 0` 就输出 `0`，这个"开关"决策完全取决于 `x` 本身。SwiGLU 打破了这种约束：门控信号来自另一个独立的线性投影，网络可以学到"即使某个维度的值很大，如果它在这个上下文中不重要，也应该被抑制"。

**工作流程**：

1. **输入分路**：假设输入向量维度为 `d`，首先通过两个独立的线性变换将其投影为两个维度为 `d_ff` 的中间向量。记输入为 `x`（shape: `[batch, d]`），权重矩阵为 `W_gate`（shape: `[d, d_ff]`）和 `W_up`（shape: `[d, d_ff]`）。计算得到 `gate_input = x @ W_gate` 和 `up_input = x @ W_up`。注意这两个投影从相同输入出发，但学到了不同的映射方向。

2. **门控激活**：`gate_input` 通过 Swish（SiLU）激活函数得到门控信号 `gate = silu(gate_input)`。Swish 的输出范围大约在 `[-0.278, +∞)`——注意它不像 sigmoid 那样被限制在 [0,1]，而是可以取更大的正值，也能取较小的负值。这使得门控不仅可以选择"通过多少"，甚至可以在某些情况下"反相"（轻微抑制）信号。

3. **逐元素门控**：将门控信号与 `up_input` 逐元素相乘（Hadamard 积），得到 `output = gate ⊙ up_input`。这是 SwiGLU 的核心操作——信息流被门控选择性调制。

4. **最终投影**（在 FFN 的完整实现中）：`output` 经过第三个线性变换 `W_down` 投影回原始维度 `d`，即为 FFN 的最终输出。

**关键概念深度解析**：

- **GLU（门控线性单元）**：最基础的门控机制，`GLU(x) = (xW_1 + b_1) ⊙ σ(xW_2 + b_2)`，门控函数是 sigmoid，输出范围被限制在 (0, 1)。可以理解为"软开关"——门控值接近 0 时关闭通道，接近 1 时完全打开。

- **Swish / SiLU**：`silu(x) = x * σ(x) = x / (1 + e^{-x})`。由 Google Brain（Ramachandran et al., 2017）提出，利用自动搜索发现的激活函数。同时期 Elfwing et al. (2017) 也独立发现了该函数并将其命名为 SiLU（Sigmoid Linear Unit）。它的巧妙之处在于：对于大的正值，行为接近线性（梯度接近 1，不消失）；对于大的负值，输出接近 0（具有饱和性）；在整个定义域上处处可微、导数连续。

- **SwiGLU**：SiLU-GLU 的组合，用 Swish 替代 sigmoid 作为门控函数。Shazeer（2020）发现这是所有 GLU 变体中表现最好的组合。

**与其它激活函数的本质对比**：

| 维度 | ReLU | GeLU | SwiGLU |
|------|------|------|--------|
| 计算方式 | `max(0, x)` | `x * Φ(x)` | `silu(xW_g) ⊙ (xW_u)` |
| 门控信息 | 自门控（硬） | 自门控（软） | 学习到的独立门控 |
| 参数 | 无 | 无 | 有（两个权重矩阵） |
| 负值区域 | 完全为0 | 轻微非零 | 可正可负（取决于门控） |
| 表达能力 | 基础 | 中等 | 强 |
| 梯度平滑度 | 在0处不连续 | 处处光滑 | 处处光滑 |

关键区别在于：ReLU 和 GeLU 是"自门控"激活——它们用自己的值决定是否激活；而 SwiGLU 拥有独立学习的门控路径，门控信号来自另一个线性变换的结果，使得网络可以学习更复杂的激活模式。

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| `x` | 输入向量，维度 `d` |
| `W_gate, W_up, W_down` | 可学习的权重矩阵 |
| `⊙` | 逐元素乘法（Hadamard 积） |
| `σ(x)` | sigmoid 函数: `σ(x) = 1 / (1 + e^{-x})` |
| `silu(x)` | SiLU/Swish 激活函数: `x · σ(x)` |
| `d_ff` | 前馈网络中间层维度 |
| `d` | 模型隐藏层维度（d_model） |

### 3.2 GLU（门控线性单元）

**公式**：

$$GLU(x) = (xW_1 + b_1) \odot \sigma(xW_2 + b_2)$$

**推导思路**：GLU 的思想来源于 LSTM 中的门控机制。输入 `x` 被两个线性变换 `W_1` 和 `W_2` 投影到相同的维度空间。第一个变换产生"值"（这是实际要传递的信息内容），第二个变换经过 sigmoid 压缩到 (0,1) 区间作为"门控信号"（决定信息通过的比例）。当 `σ(xW_2 + b_2) ≈ 0` 时，输出几乎为零（通道关闭）；当 `σ(xW_2 + b_2) ≈ 1` 时，输出等于原值（通道全开）。介于之间的值则实现"部分通过"。

**问题**：sigmoid 函数在远离原点时梯度趋近于 0（饱和），导致门控分支在深层网络中出现梯度消失。这意味着网络无法有效地学习"何时应该打开或关闭通道"。

### 3.3 Swish / SiLU

**公式**：

$$silu(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**函数图像特性分析**：

- 当 `x → +∞`：`σ(x) → 1`，所以 `silu(x) → x`——在大正值区，SiLU 表现为近似恒等函数。这意味着对于强激活的神经元，梯度接近 1，不会像 sigmoid 那样梯度消失。

- 当 `x → -∞`：`σ(x) → 0`，所以 `silu(x) → 0⁻`（从负方向趋近于 0）。输出虽然接近 0，但不是严格为零，且从负值方向逼近——这提供了一个非零（虽然很小）的梯度。

- 当 `x = 0`：`silu(0) = 0 · 0.5 = 0`。

- 最小值在 `x ≈ -1.278` 处，最小值 `≈ -0.278`。这个"谷底"意味着 SiLU 在轻微负值区域有一个非单调的凹陷——函数先下降再回升。这个特性在 ReLU 和 GeLU 中都不存在。

**与 ReLU 的数学关系**：ReLU 是硬门控 `max(0, x)`——x > 0 时输出 x（门控=1），x <= 0 时输出 0（门控=0）。Swish 则是 "软 ReLU"：用平滑的 sigmoid 门控替代硬截断。实际上，如果用阶跃函数替代 sigmoid（`H(x) = I(x > 0)`），`x · H(x)` 就精确等于 ReLU。因此 Swish 可以理解为 ReLU 的"处处可微的平滑版本"。

### 3.4 SwiGLU

**完整公式**：

$$SwiGLU(x) = silu(xW_{gate}) \odot (xW_{up})$$

**展开为分步计算**：

1. 计算门控输入：$g = xW_{gate}$
2. 计算值输入：$v = xW_{up}$
3. 应用 Swish：$gate = silu(g) = g \odot \sigma(g)$
4. 门控乘法：$output = gate \odot v$

**在 FFN 中的完整形态**（现代大模型的标准写法）：

$$FFN_{SwiGLU}(x) = (silu(xW_{gate}) \odot (xW_{up}))W_{down}$$

**参数量分析**：
- 标准 ReLU FFN：两个权重矩阵 `W_up [d, d_ff]` 和 `W_down [d_ff, d]`，参数量 = `2 · d · d_ff`
- SwiGLU FFN：三个权重矩阵 `W_gate [d, d_ff]`、`W_up [d, d_ff]` 和 `W_down [d_ff, d]`，参数量 = `3 · d · d_ff`
- 结论：参数增加了约 50%。实践中为保持总参数量不变，通常将 `d_ff` 缩小为原来的 2/3（即 `d_ff = 8/3 · d`），使参数量大致持平。

### 3.5 导数推导

**sigmoid 的导数**（基础准备）：

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**证明**：
$$\sigma'(x) = \frac{d}{dx}\frac{1}{1+e^{-x}} = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

**SiLU 的导数**（SwiGLU 的关键组成部分）：

$$\begin{aligned}
silu'(x) &= \frac{d}{dx}[x \cdot \sigma(x)] \\
&= 1 \cdot \sigma(x) + x \cdot \sigma'(x) \quad \text{(乘法求导法则)} \\
&= \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) \\
&= \sigma(x)[1 + x(1 - \sigma(x))]
\end{aligned}$$

**SiLU 导数的重要性质**：
- 当 `x → +∞`：`σ(x) → 1`，`1 - σ(x) → 0`，所以 `silu'(x) → 1`——深层网络不会梯度消失
- 当 `x → -∞`：`σ(x) → 0`，`1 - σ(x) → 1`，所以 `silu'(x) → 0`——与期望一致，大负值梯度趋于 0
- 在 x=0 处：`silu'(0) = 0.5 · (1 + 0) = 0.5`——不为零，梯度可以顺利通过

**SwiGLU 的反向传播**：SwiGLU 的两个可学习权重矩阵 `W_gate` 和 `W_up` 的梯度计算（通过链式法则）：

$$\frac{\partial L}{\partial W_{gate}} = x^T \cdot \left[\frac{\partial L}{\partial output} \odot xW_{up} \odot silu'(xW_{gate})\right]$$

$$\frac{\partial L}{\partial W_{up}} = x^T \cdot \left[\frac{\partial L}{\partial output} \odot silu(xW_{gate})\right]$$

这体现了门控机制的梯度分流特性：`W_gate` 的梯度接收来自 `silu'` 和 `up` 的调制；`W_up` 的梯度接收来自 `gate`（silu 后的门控信号）的调制。

### 3.6 为什么 SwiGLU 在 Transformer 中表现好

1. **门控提供信号选择性**：与 ReLU 的"一刀切"不同，SwiGLU 允许网络学习哪些特征维度应该被放大或抑制。在 Transformer 的 FFN 中，这意味着模型可以更精细地控制知识存储和信息加工——它不仅仅是"记忆"模式，还能根据上下文决定"这个模式现在适用吗？"

2. **非线性更平滑**：Swish 是处处光滑的函数，梯度在整个实数域上连续。相比之下，ReLU 在 x=0 处梯度不连续（左侧为 0，右侧为 1），这可能导致优化路径在该点附近振荡。

3. **实证优越性**：Shazeer（2020）的实验表明，在相同训练条件下，SwiGLU 的困惑度显著低于 ReLU-GLU 和 GeLU-GLU 变体。这种优势随着模型规模的增大而更加明显。

4. **与残差连接的协同**：Transformer 的残差连接（`output = x + FFN(x)`）要求 FFN 输出与输入在相似的数值量级。SwiGLU 的门控机制天然提供了输出幅度的自适应调节，不容易产生数值爆炸。

## 4. 训练过程讲解

**前馈层中使用 SwiGLU 的标准配置**：

在 Transformer 中，每个 decoder（或 encoder）层的标准结构是：

```
x -> [Self-Attention + residual + LayerNorm] -> [FFN + residual + LayerNorm]
```

SwiGLU 仅替换 FFN 内部的激活函数部分。标准 ReLU FFN 为：

$$FFN_{ReLU}(x) = ReLU(xW_{up})W_{down}$$

SwiGLU FFN 变为：

$$FFN_{SwiGLU}(x) = (silu(xW_{gate}) \odot (xW_{up}))W_{down}$$

其中新增的 `W_gate` 与 `W_up` 维度相同，都是 `[d, d_ff]`。

**参数量考虑**：相比于标准 ReLU FFN（两个线性层 `d → d_ff → d`），SwiGLU FFN 有三个线性层（`d → d_ff`、`d → d_ff`、`d_ff → d`）。如果保持相同的 `d_ff`，参数量增加约 50%。

**实践中的补偿策略**：LLaMA 等模型将 SwiGLU 的中间维度设为 `d_ff ≈ 8/3 · d` 而不是常见的 `4 · d`。这样三个矩阵的总参数量约为 `3 · d · (8/3 · d) = 8 · d²`，与标准 FFN 的 `2 · d · 4 · d = 8 · d²` 相等。计算开销主要来自矩阵乘法（O(d · d_ff)），逐元素乘法和 SiLU 的 O(d_ff) 开销几乎可以忽略。

**典型超参数配置**：

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| `d_ff / d` | 2.67 (8/3) | 实践中最常见的比例，与标准 FFN 参数量持平 |
| 线性层 bias | False | 大模型中通常省略偏置，减少参数且不影响性能 |
| 权重初始化 | 正态分布 `N(0, 0.02)` | 同 Transformer 标准初始化 |
| SiLU 计算精度 | FP32 | 门控的 sigmoid 含指数运算，FP16 下易出现精度问题 |
| 混合精度训练 | BF16 + FP32 gate | 推荐在门控路径保留 FP32 精度 |
| 学习率 warmup | 是 | SiLU 门控在训练初期可能饱和，warmup 帮助平稳启动 |

**训练时的注意点**：
- SiLU 的计算涉及指数运算（sigmoid 中的 `e^{-x}`），在 FP16 下可能出现精度问题。建议在门控分支使用 FP32 的中间计算，或直接使用 BF16 训练。
- 门控信号在训练初期可能饱和（sigmoid 接近 0 或 1），此时门控梯度接近 0。使用合适的学习率 warmup 和较小的初始化方差可以缓解。
- 在分布式训练的张量并行中，`W_gate` 和 `W_up` 按列切分到不同设备。由于门控乘法 `gate ⊙ up` 在各设备本地完成，SwiGLU 实际上比 ReLU FFN 少一次 All-Reduce 通信。

## 5. 应用场景

**主要应用**：

1. **LLaMA 系列 / DeepSeek 中的 FFN 替代**：从 LLaMA 第一版开始，Meta 就选择 SwiGLU 作为 Transformer 前馈网络的激活函数，并在 LLaMA 2、LLaMA 3 中沿用。DeepSeek-V2/V3 同样采用 SwiGLU FFN，且在其 MoE（混合专家）架构中，每个专家网络内部都使用 SwiGLU 作为激活函数。这使得模型在保持训练稳定性的同时，获得了更强的非线性表达能力。

2. **PaLM 大模型**：Google 的 PaLM（540B 参数）首次大规模验证了 SwiGLU 的优势。论文报告，在同等训练 token 数下，SwiGLU 的困惑度显著低于 GeLU，且随着模型规模增大，优势更加明显。

3. **需要高质量非线性变换的 Transformer 层**：任何需要 FFN 进行深层信息加工的 Transformer 场景——语言模型、翻译、代码生成、多模态模型——都从 SwiGLU 受益。门控机制使得每个 token 的 FFN 输出更好地适配其上下文需求。

**适用场景**（推荐使用 SwiGLU）：
- 大规模 Transformer 语言模型（>1B 参数）
- 需要高表达能力的深度网络（>24 层）
- 对困惑度 / 下游任务精度有较高要求的预训练
- MoE（混合专家）架构中的专家网络

**不适用场景**（使用 ReLU/GeLU 更合适）：
- 极低延迟的在线推理（SwiGLU 多一次矩阵乘法和指数计算）
- 移动端 / 边缘设备部署（三个权重矩阵占更多内存带宽和存储）
- 参数量非常小的模型（<10M，SwiGLU 的增益不明显）
- CNN 架构（SwiGLU 的优势主要在 Transformer 中得到验证，CNN 场景仍以 ReLU 为主流）
- 需要极致简化的教学/演示代码

## 6. 优缺点分析

### SwiGLU 的优缺点

| 优点 | 缺点 |
|------|------|
| 非线性表达更丰富——独立门控分支提供额外自由度 | 计算量稍大——多了一个矩阵乘法和逐元素 SiLU |
| 梯度更平滑——SiLU 处处可微，无梯度不连续点 | 参数更多——三个权重矩阵，比标准 FFN 多约 50% 参数 |
| 选择性信息传递——门控机制允许网络抑制噪声维度 | 内存占用更大——额外的权重矩阵和激活值需要更多 GPU 显存 |
| 训练更稳定——平滑梯度使得大 batch 训练时优化轨迹更平稳 | 门控有负值区域（最低约 -0.278），可能与 LayerNorm 产生非预期交互 |
| 在大规模模型上经过充分验证——LLaMA/PaLM/DeepSeek 等均采用 | 与传统 ReLU 硬件优化不兼容——需要专门的 SiLU 算子适配 |
| 张量并行中通信更少——门控乘法在本地完成，少一次 All-Reduce | 经验调参成本——对于不同规模的模型，d_ff/d 的最优比例需要实验确定 |

### 与其它激活函数的详细对比

| 特性 | ReLU | GeLU | SiLU (Swish) | SwiGLU |
|------|------|------|-------------|--------|
| 公式 | `max(0, x)` | `x · Φ(x)` | `x · σ(x)` | `silu(xW_g) ⊙ (xW_u)` |
| 参数量（FFN） | `2 · d · d_ff` | `2 · d · d_ff` | `2 · d · d_ff` | `3 · d · d_ff` |
| 计算复杂度 | O(d·d_ff) | O(d·d_ff) | O(d·d_ff) | O(d·d_ff) (常数因子更大) |
| 可学习门控 | 否 | 否 | 否（自门控） | 是（独立学习） |
| 负值输出 | 否 | 微小负值 | 最小约 -0.278 | 取决于门控值 |
| 梯度连续性 | x=0处不连续 | 连续 | 连续 | 连续 |
| 大模型验证 | BERT时代 | BERT/GPT-2/3 | EfficientNet | LLaMA/PaLM/DeepSeek |
| 代表性模型 | ResNet, VGG | BERT, GPT-2 | EfficientNet | LLaMA3, DeepSeek-V2/V3 |

**为什么选择 SwiGLU 而非 GeLU**：
- Shazeer（2020）系统实验表明：在相同训练条件下，SwiGLU 的困惑度比 GeGLU 低约 2-3%，比标准 GeLU FFN 低约 5-7%。
- 这种优势随着模型规模从 100M 扩展到 1B+ 变得更加显著，说明 SwiGLU 的额外表达能力在大模型中能有效转化为性能提升。

## 7. 调库实现

```python
"""
SwiGLU 激活函数的调库实现
使用 PyTorch 的 F.silu (即 Swish/SiLU) 构建 SwiGLU 前馈网络
Python 3.9+, PyTorch 2.0+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU_FFN(nn.Module):
    """
    使用 PyTorch 内置 F.silu 实现的 SwiGLU 前馈网络。
    
    这是 LLaMA / DeepSeek 等现代大模型中 FFN 层的标准实现模式。
    与标准的两层 ReLU FFN 不同，SwiGLU FFN 有三个线性层，
    其中两个（w_gate 和 w_up）将输入投影到相同的中间维度，
    gate 分支的输出经过 SiLU 激活后，与 up 分支逐元素相乘，
    结果再通过 w_down 投影回原始维度。
    
    注意：为保持与标准 FFN 相同的参数量，建议设置
    d_ff = int(8/3 * d_model) 而非常见的 4 * d_model。
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        """
        Args:
            d_model: 输入/输出维度（模型隐藏层维度）
            d_ff:    前馈层中间维度（gate 和 up 分支的维度）
                     推荐值: int(8/3 * d_model) 以匹配标准 FFN 参数量
            dropout: Dropout 比率，大模型训练中常设为 0.0
        """
        super().__init__()
        # 门控投影：将输入映射到门控信号空间
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        # 值投影：将输入映射到信息值空间
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        # 输出投影：将门控后的结果映射回原始维度
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        # Dropout（大模型中常不使用）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行 SwiGLU 门控计算。
        
        Args:
            x: 输入张量, shape (batch_size, seq_len, d_model)
        Returns:
            输出张量, shape (batch_size, seq_len, d_model)
        """
        # 步骤1: 两个并行的线性投影
        # gate_input: 将用于生成门控信号
        # up_input:   将作为被调制的信息值
        gate_input = self.w_gate(x)  # (batch, seq_len, d_ff)
        up_input   = self.w_up(x)    # (batch, seq_len, d_ff)
        
        # 步骤2: 门控信号通过 SiLU / Swish 激活
        # F.silu 内部实现: x * sigmoid(x)，已做了算子融合优化
        gate = F.silu(gate_input)    # (batch, seq_len, d_ff)
        
        # 步骤3: 逐元素门控乘法（Hadamard 积）
        # gate 的每个元素控制 up 对应元素通过的比例
        gated = gate * up_input      # (batch, seq_len, d_ff)
        
        # 步骤4: 投影回原始维度
        output = self.w_down(gated)  # (batch, seq_len, d_model)
        
        return self.dropout(output)


# ============================================================
# 测试代码：验证 SwiGLU FFN 的张量维度和梯度正确性
# ============================================================
def test_swiglu_ffn():
    """
    验证 SwiGLU FFN 模块：
    1. 前向传播输出维度正确
    2. 梯度正常回传到所有权重矩阵
    3. 打印参数量信息
    """
    # 测试配置
    batch_size = 2
    seq_len = 8
    d_model = 64
    # 使用 8/3 比例以匹配标准 FFN 参数量
    d_ff = int(8 / 3 * d_model)  # ≈ 170
    
    # 创建模型和模拟输入
    model = SwiGLU_FFN(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = model(x)
    
    # 验证输出维度
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"输出维度错误: 期望 {expected_shape}, 实际 {output.shape}"
    print(f"[维度验证] 通过 - 输出维度: {output.shape}")
    
    # 验证输出不含 NaN 或 Inf
    assert not torch.isnan(output).any(), "输出包含 NaN!"
    assert not torch.isinf(output).any(), "输出包含 Inf!"
    print("[数值验证] 通过 - 输出无 NaN/Inf")
    
    # 验证梯度回传
    loss = output.sum()
    loss.backward()
    
    # 检查所有权重矩阵都有梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 '{name}' 没有梯度!"
        assert not torch.isnan(param.grad).any(), f"参数 '{name}' 梯度包含 NaN!"
        print(f"[梯度验证] 通过 - '{name}': shape={list(param.grad.shape)}")
    
    # 参数量统计
    gate_params = d_model * d_ff
    up_params = d_model * d_ff
    down_params = d_ff * d_model
    total = gate_params + up_params + down_params
    
    print(f"\n{'='*50}")
    print(f"参数量统计 (d_model={d_model}, d_ff={d_ff})")
    print(f"  w_gate: {gate_params:,}  ({d_model} x {d_ff})")
    print(f"  w_up:   {up_params:,}  ({d_model} x {d_ff})")
    print(f"  w_down: {down_params:,}  ({d_ff} x {d_model})")
    print(f"  总计:   {total:,}")
    print(f"  对比标准 FFN (2*{d_model}*{4*d_model}): {2*d_model*4*d_model:,}")
    print(f"{'='*50}")
    print("\n全部测试通过! SwiGLU FFN 正常工作。")
    
    return output


if __name__ == "__main__":
    test_swiglu_ffn()
```

## 8. 手工代码实现

```python
"""
SwiGLU 激活函数的手工实现
从零实现 SiLU (Swish) 和 SwiGLU，并在简单回归任务上对比 ReLU
Python 3.9+, PyTorch 2.0+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


# ============================================================
# 1. 从零实现 SiLU (Swish) 激活函数
# ============================================================
def silu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    完全手工实现 SiLU / Swish 激活函数。
    
    公式: silu(x) = x * sigmoid(x)
          = x / (1 + exp(-x))
    
    实现说明：
    - 使用 torch.sigmoid 而非 F.silu，以展示函数的内部原理
    - torch.sigmoid 保证了数值稳定性（在内部做了 clamp 操作）
    - 实际工程中使用 F.silu 更高效（支持算子融合）
    
    Args:
        x: 任意形状的输入张量
    Returns:
        与 x 相同形状的输出张量
    """
    # 计算 sigmoid: σ(x) = 1 / (1 + e^{-x})
    sigmoid_x = torch.sigmoid(x)
    # SiLU = x * sigmoid(x)
    return x * sigmoid_x


# ============================================================
# 2. 从零实现 SwiGLU 类
# ============================================================
class SwiGLU_Manual(nn.Module):
    """
    手工实现的 SwiGLU 激活模块。
    
    该类实现了 SwiGLU 的核心操作：将输入通过两个独立的线性
    变换投影后，一个经过 SiLU 激活作为门控信号，与另一个逐
    元素相乘。
    
    公式: SwiGLU(x) = silu_manual(x @ W_gate) ⊙ (x @ W_up)
    
    注意：这个类只做 SwiGLU 的核心门控计算，不包含 down 投影。
    完整 FFN 还需要一个后续的线性层将输出投影回原始维度。
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Args:
            d_in:  输入特征维度
            d_out: 输出维度（即中间维度 d_ff）
        """
        super().__init__()
        # 门控分支的权重矩阵：将输入投影到门控信号空间
        # 使用较小的初始化方差，避免门控信号在训练初期饱和
        self.W_gate = nn.Parameter(torch.randn(d_in, d_out) * 0.02)
        # 值分支的权重矩阵：将输入投影到信息值空间
        self.W_up = nn.Parameter(torch.randn(d_in, d_out) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 计算 SwiGLU 门控输出。
        
        Args:
            x: 输入张量 (batch_size, d_in)
        Returns:
            output: 门控后的输出张量 (batch_size, d_out)
        """
        # 线性投影: 将输入映射到门控信号空间
        gate_input = x @ self.W_gate   # (batch, d_out)
        # 线性投影: 将输入映射到信息值空间
        up_input   = x @ self.W_up     # (batch, d_out)
        
        # 手工 SiLU 激活: gate = gate_input * sigmoid(gate_input)
        gate = silu_manual(gate_input) # (batch, d_out)
        
        # 门控乘法: output = gate ⊙ up
        return gate * up_input         # (batch, d_out)


# ============================================================
# 3. 对比模型定义：ReLU FFN vs SwiGLU FFN
# ============================================================
class ReLU_FFN(nn.Module):
    """
    标准 ReLU 前馈网络，用于与 SwiGLU 对比。
    两层结构: d_in -> d_hidden (ReLU) -> d_out
    """
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SwiGLU_FFN_Manual(nn.Module):
    """
    完整的手工 SwiGLU 前馈网络。
    
    结构: d_in -> SwiGLU (d_in -> d_hidden) -> d_out
    
    注意：SwiGLU 内部有三个权重矩阵（W_gate, W_up, W_down），
    而 ReLU FFN 只有两个（fc1, fc2）。为了公平对比，我们将
    SwiGLU 的中间维度调整为 ReLU 的 2/3，使总参数量保持一致。
    """
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        # SwiGLU 有 3 个矩阵，ReLU 有 2 个矩阵
        # 为使参数量可比：adjusted = d_hidden * 2/3
        adjusted_hidden = int(d_hidden * 2 / 3)
        # 手工 SwiGLU 核心模块 (W_gate 和 W_up)
        self.swiglu = SwiGLU_Manual(d_in, adjusted_hidden)
        # 输出投影 (W_down)
        self.down = nn.Linear(adjusted_hidden, d_out)
    
    def forward(self, x):
        # SwiGLU 门控计算
        gated = self.swiglu(x)     # (batch, adjusted_hidden)
        # 输出投影回目标维度
        return self.down(gated)    # (batch, d_out)


# ============================================================
# 4. 合成数据生成
# ============================================================
def generate_synthetic_data(n_samples=2000, n_features=20):
    """
    生成合成回归数据：y = 3 * sin(X @ w_true) + noise
    
    选择 sin 函数是因为：
    - 它是非线性的，需要激活函数来拟合
    - 既不是 ReLU 优势区（纯粹的分段线性），也不是 SwiGLU 优势区
    - 可以提供公平的比较基准
    
    Args:
        n_samples:  样本数
        n_features: 特征维度
    Returns:
        X_train, y_train, X_test, y_test (均为 torch.Tensor)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成随机特征
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 生成随机的真实权重
    w_true = np.random.randn(n_features, 1).astype(np.float32) * 0.5
    # 目标值: 非线性 sin 函数 + 小噪声
    y = np.sin(X @ w_true) * 3 + np.random.randn(n_samples, 1).astype(np.float32) * 0.1
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # 划分训练集/测试集 (80/20)
    split = int(0.8 * n_samples)
    return (X_tensor[:split], y_tensor[:split],
            X_tensor[split:], y_tensor[split:])


# ============================================================
# 5. 训练与评估函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    """评估模型，返回平均损失。"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


# ============================================================
# 6. 对比实验主函数
# ============================================================
def compare_relu_vs_swiglu():
    """
    在相同合成数据上对比 ReLU FFN 和 SwiGLU FFN 的表现。
    
    对比维度：
    1. 训练损失下降速度（是否收敛更快）
    2. 最终测试损失（泛化能力）
    3. 参数量（效率）
    4. 训练时间（计算开销）
    """
    # 超参数
    d_in = 20       # 输入特征维度
    d_hidden = 128  # 中间层维度
    d_out = 1       # 输出维度（回归）
    batch_size = 64
    epochs = 200
    lr = 0.001
    
    # 准备数据
    X_train, y_train, X_test, y_test = generate_synthetic_data()
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )
    
    # 创建两个模型
    relu_model = ReLU_FFN(d_in, d_hidden, d_out)
    swiglu_model = SwiGLU_FFN_Manual(d_in, d_hidden, d_out)
    
    criterion = nn.MSELoss()
    
    relu_opt = optim.Adam(relu_model.parameters(), lr=lr)
    swiglu_opt = optim.Adam(swiglu_model.parameters(), lr=lr)
    
    # 记录训练历史
    history = {
        "relu":   {"train": [], "test": []},
        "swiglu": {"train": [], "test": []}
    }
    
    print("="*60)
    print("开始对比训练: ReLU FFN vs SwiGLU FFN")
    print(f"ReLU    参数量: {sum(p.numel() for p in relu_model.parameters()):,}")
    print(f"SwiGLU  参数量: {sum(p.numel() for p in swiglu_model.parameters()):,}")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ReLU 训练和评估
        relu_train = train_one_epoch(relu_model, train_loader, relu_opt, criterion)
        relu_test  = evaluate(relu_model, test_loader, criterion)
        history["relu"]["train"].append(relu_train)
        history["relu"]["test"].append(relu_test)
        
        # SwiGLU 训练和评估
        swiglu_train = train_one_epoch(swiglu_model, train_loader, swiglu_opt, criterion)
        swiglu_test  = evaluate(swiglu_model, test_loader, criterion)
        history["swiglu"]["train"].append(swiglu_train)
        history["swiglu"]["test"].append(swiglu_test)
        
        # 每 40 epochs 打印一次进度
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"ReLU  train={relu_train:.5f}  test={relu_test:.5f} | "
                  f"SwiGLU train={swiglu_train:.5f} test={swiglu_test:.5f}")
    
    elapsed = time.time() - start_time
    
    # 汇总结果
    print("\n" + "="*60)
    print(f"训练完成! 耗时 {elapsed:.1f} 秒")
    print("-"*60)
    print(f"{'指标':<25} {'ReLU FFN':<18} {'SwiGLU FFN':<18}")
    print("-"*60)
    print(f"{'参数量':<25} {sum(p.numel() for p in relu_model.parameters()):<18,} "
          f"{sum(p.numel() for p in swiglu_model.parameters()):<18,}")
    print(f"{'最终训练 Loss':<25} {history['relu']['train'][-1]:<18.6f} "
          f"{history['swiglu']['train'][-1]:<18.6f}")
    print(f"{'最终测试 Loss':<25} {history['relu']['test'][-1]:<18.6f} "
          f"{history['swiglu']['test'][-1]:<18.6f}")
    print(f"{'最佳测试 Loss':<25} {min(history['relu']['test']):<18.6f} "
          f"{min(history['swiglu']['test']):<18.6f}")
    print("="*60)
    
    # 分析
    relu_final = history['relu']['test'][-1]
    swiglu_final = history['swiglu']['test'][-1]
    if swiglu_final < relu_final:
        improvement = (relu_final - swiglu_final) / relu_final * 100
        print(f"SwiGLU 测试 Loss 比 ReLU 低 {improvement:.1f}%")
    
    return history


# ============================================================
# 7. 正确性验证
# ============================================================
if __name__ == "__main__":
    # 验证手工 SiLU 与 PyTorch 内置 F.silu 一致
    print("验证手工 SiLU 实现的正确性...")
    x_test = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    manual_result = silu_manual(x_test)
    builtin_result = F.silu(x_test)
    
    print(f"  输入 x:          {x_test.tolist()}")
    print(f"  手工 SiLU:       {[f'{v:.6f}' for v in manual_result.tolist()]}")
    print(f"  PyTorch F.silu:  {[f'{v:.6f}' for v in builtin_result.tolist()]}")
    
    is_close = torch.allclose(manual_result, builtin_result, atol=1e-6)
    assert is_close, "手工 SiLU 与 F.silu 不一致!"
    print("  [通过] 手工 SiLU 与 PyTorch F.silu 完全一致\n")
    
    # 运行对比实验
    compare_relu_vs_swiglu()
```

## 9. 可视化与结果理解

```python
"""
SwiGLU 相关函数的全面可视化
包括：Swish/ReLU/GeLU 曲线对比、Swish 导数分析、门控信号变化
Python 3.9+, matplotlib 3.5+, numpy
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 全局绘图设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ============================================================
# 基础函数定义
# ============================================================
def sigmoid(x):
    """sigmoid 函数: σ(x) = 1 / (1 + e^{-x})"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def swish(x):
    """Swish / SiLU: silu(x) = x * σ(x)"""
    return x * sigmoid(x)


def swish_derivative(x):
    """
    SiLU 的导数。
    推导: silu'(x) = σ(x) + x * σ(x) * (1 - σ(x))
                    = σ(x) * [1 + x * (1 - σ(x))]
    """
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def gelu_approximate(x):
    """
    GeLU 的 tanh 近似（BERT/GPT 中使用的高效版本）。
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))


def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0.0, x)


# ============================================================
# 图1: Swish / ReLU / GeLU 函数曲线对比
# ============================================================
def plot_activation_comparison():
    """
    三种常用激活函数的全面对比。
    
    关键观察点:
    1. ReLU 在 x=0 处的硬截断——左侧完全为 0
    2. GeLU 在负值区域有微小非零输出（约 -0.17 到 0 之间）
    3. Swish 在 x≈-1.28 处有一个独特的负值谷底（约 -0.28）
    4. 三者在 x>0 时行为相似，但 Swish 最平滑
    """
    x = np.linspace(-4, 4, 500)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制三种激活函数
    ax.plot(x, relu(x), 'b-', linewidth=2, label='ReLU: max(0, x)', alpha=0.75)
    ax.plot(x, gelu_approximate(x), 'g-', linewidth=2, label='GeLU (tanh approx)', alpha=0.75)
    ax.plot(x, swish(x), 'r-', linewidth=2.5, label='Swish / SiLU: x * σ(x)', alpha=0.9)
    
    # 坐标轴参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    
    # 标注 Swish 的最小值位置
    min_idx = np.argmin(swish(x))
    min_x, min_y = x[min_idx], swish(x)[min_idx]
    ax.plot(min_x, min_y, 'ro', markersize=6)
    ax.annotate(
        f'最小值 ≈ ({min_x:.2f}, {min_y:.3f})',
        xy=(min_x, min_y),
        xytext=(min_x + 1.5, min_y + 0.35),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    
    # 标注 Swish = 0.5 附近的 point
    ax.annotate(
        'x=0 时 Swish=0',
        xy=(0, 0),
        xytext=(0.6, -0.2),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        fontsize=9, color='gray'
    )
    
    ax.set_xlabel('x (输入值)', fontsize=13)
    ax.set_ylabel('激活输出 f(x)', fontsize=13)
    ax.set_title('Swish vs ReLU vs GeLU 激活函数曲线对比', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 4.5)
    
    plt.tight_layout()
    plt.savefig('swiglu_01_activation_comparison.png', bbox_inches='tight')
    plt.show()
    print("[图1] 激活函数对比图已保存: swiglu_01_activation_comparison.png")


# ============================================================
# 图2: Swish / SiLU 的函数与导数
# ============================================================
def plot_swish_derivative():
    """
    同时展示 SiLU 函数和其导数。
    
    关键观察:
    1. 导数在整个定义域上连续（vs ReLU 在 0 处跳变）
    2. x>0 时导数从 0.5 平滑上升到 1.0
    3. x<0 时导数平滑下降到 0（不是突然断崖）
    4. 导数的"隆起"在 x≈1.5 附近——此时导数值 > 1，
       意味着 SiLU 在某个区间比恒等函数传播梯度更多
    """
    x = np.linspace(-4, 4, 500)
    f_x = swish(x)
    deriv = swish_derivative(x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # ---- 左子图: SiLU 函数 ----
    ax1.plot(x, f_x, 'r-', linewidth=2.5, label='silu(x) = x * σ(x)')
    ax1.plot(x, relu(x), 'b--', linewidth=1.2, label='ReLU (参考)', alpha=0.45)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
    
    # 填充负值区域以强调"非零"
    mask_neg = x < 0
    ax1.fill_between(x[mask_neg], 0, f_x[mask_neg], alpha=0.08, color='red')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('SiLU (Swish) 函数', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)
    
    # ---- 右子图: SiLU 导数 ----
    ax2.plot(x, deriv, 'purple', linewidth=2.5,
             label="silu'(x) = σ(x)·[1 + x·(1-σ(x))]")
    
    # 参考线
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.35, label='导数上限 ≈ 1')
    ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.4)
    ax2.axvline(x=0.0, color='gray', linestyle=':', alpha=0.4)
    
    # 标注关键导数值
    ax2.annotate(
        "silu'(0) = 0.5",
        xy=(0, 0.5), xytext=(0.7, 0.35),
        arrowprops=dict(arrowstyle='->', color='purple', lw=1.2),
        fontsize=10, color='darkviolet'
    )
    
    # 标注导数大于 1 的区域（梯度放大区）
    peak_idx = np.argmax(deriv)
    ax2.annotate(
        f'max ≈ {deriv[peak_idx]:.3f}',
        xy=(x[peak_idx], deriv[peak_idx]),
        xytext=(x[peak_idx] + 1.0, deriv[peak_idx] - 0.05),
        arrowprops=dict(arrowstyle='->', color='purple', lw=1.2),
        fontsize=9, color='darkviolet'
    )
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("f'(x)", fontsize=12)
    ax2.set_title("SiLU 的导数曲线", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(-0.05, max(deriv) * 1.08)
    
    plt.tight_layout()
    plt.savefig('swiglu_02_silu_derivative.png', bbox_inches='tight')
    plt.show()
    print("[图2] SiLU 函数与导数图已保存: swiglu_02_silu_derivative.png")


# ============================================================
# 图3: SwiGLU 门控信号可视化
# ============================================================
def plot_swiglu_gating():
    """
    使用二维热力图可视化 SwiGLU 门控机制的工作方式。
    
    模拟一个简化的场景：2D 输入通过两个不同的权重投影，
    展示门控信号如何随输入空间的分布变化。
    
    解读方法：
    - 红色: 正值区域（信息通过/放大）
    - 蓝色: 负值区域（信息抑制/反转）
    - 白色: 接近零区域（信息被阻断）
    """
    np.random.seed(42)
    
    # 生成 2D 网格
    resolution = 200
    x_range = np.linspace(-3, 3, resolution)
    X1, X2 = np.meshgrid(x_range, x_range)
    points = np.stack([X1.ravel(), X2.ravel()], axis=1)  # (40000, 2)
    
    # 模拟两个不同的权重投影方向
    # W_gate: 主要关注 x1 (第一维)
    # W_up:   主要关注 x2 (第二维)，但有负权重
    W_gate = np.array([[1.2], [0.3]])   # gate 投影：偏好 x1
    W_up   = np.array([[0.4], [-0.9]])  # value 投影：偏好負 x2
    
    gate_input = points @ W_gate       # (40000, 1)
    up_input   = points @ W_up         # (40000, 1)
    
    gate_signal = swish(gate_input)    # SiLU 门控
    output = gate_signal * up_input    # 最终输出
    
    # 重塑为网格格式
    gate_grid = gate_signal.reshape(resolution, resolution)
    output_grid = output.reshape(resolution, resolution)
    up_grid = up_input.reshape(resolution, resolution)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # ---- 子图1: 值输入 (up_input) ----
    im1 = axes[0].contourf(X1, X2, up_grid, levels=30, cmap='RdBu_r')
    axes[0].set_title('值输入: up = x @ W_up', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x1 (第一个特征)', fontsize=11)
    axes[0].set_ylabel('x2 (第二个特征)', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('值', fontsize=9)
    
    # ---- 子图2: Swish 门控信号 ----
    im2 = axes[1].contourf(X1, X2, gate_grid, levels=30, cmap='RdBu_r')
    axes[1].set_title('SiLU 门控: gate = silu(x @ W_gate)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x1 (第一个特征)', fontsize=11)
    axes[1].set_ylabel('x2 (第二个特征)', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('门控值', fontsize=9)
    
    # ---- 子图3: 最终输出 = gate * up ----
    im3 = axes[2].contourf(X1, X2, output_grid, levels=30, cmap='RdBu_r')
    axes[2].set_title('SwiGLU 输出: gate × up', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x1 (第一个特征)', fontsize=11)
    axes[2].set_ylabel('x2 (第二个特征)', fontsize=11)
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('输出值', fontsize=9)
    
    fig.suptitle('SwiGLU 门控机制: 输入空间 -> 门控调制 -> 输出', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('swiglu_03_gating_heatmap.png', bbox_inches='tight')
    plt.show()
    print("[图3] 门控热力图已保存: swiglu_03_gating_heatmap.png")


# ============================================================
# 图4: SiLU 的组成分解
# ============================================================
def plot_silu_decomposition():
    """
    将 SiLU 分解为"线性成分 x"和"门控成分 σ(x)"。
    帮助初学者理解 SiLU 内部的乘法门控机理。
    """
    x = np.linspace(-4, 4, 500)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # sigmoid 门控
    ax.plot(x, sigmoid(x), 'b-', linewidth=2.2, label='门控: σ(x) = sigmoid(x)', alpha=0.75)
    # 恒等线(参考)
    ax.plot(x, x, 'gray', linestyle='--', linewidth=1.2, label='线性部分: x (参考)', alpha=0.5)
    # SiLU
    ax.plot(x, swish(x), 'r-', linewidth=2.8, label='SiLU: silu(x) = x * σ(x)', alpha=0.9)
    
    # 关键阈值线
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(y=0.5, color='blue', linestyle=':', alpha=0.25)
    ax.axhline(y=1.0, color='blue', linestyle=':', alpha=0.25, label='σ(x) 上限 = 1')
    
    # 标注区域的注释
    ax.annotate(
        'x>0: σ(x)>0.5\n门控偏"开"\n信息通过',
        xy=(2.0, swish(2.0)),
        xytext=(2.2, 0.8),
        fontsize=9, color='darkred',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    ax.annotate(
        'x<0: σ(x)<0.5\n门控偏"关"\n信息抑制',
        xy=(-2.0, swish(-2.0)),
        xytext=(-3.8, 0.6),
        fontsize=9, color='darkblue',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    ax.set_xlabel('x (输入值)', fontsize=13)
    ax.set_ylabel('值', fontsize=13)
    ax.set_title('SiLU 的组成分解: sigmoid 门控 × 线性输入 x', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 4.5)
    
    plt.tight_layout()
    plt.savefig('swiglu_04_silu_decomposition.png', bbox_inches='tight')
    plt.show()
    print("[图4] SiLU 分解图已保存: swiglu_04_silu_decomposition.png")


# ============================================================
# 图5: 门控信号与输入的交叉影响
# ============================================================
def plot_gate_input_relationship():
    """
    展示 SiLU 门控信号随输入的一维变化。
    对比 sigmoid 门控(传统GLU) vs SiLU 门控(SwiGLU) 的区别。
    """
    x = np.linspace(-4, 4, 500)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---- 左图: 门控函数对比 ----
    ax1.plot(x, sigmoid(x), 'b-', linewidth=2, label='sigmoid 门控 (GLU)', alpha=0.8)
    ax1.plot(x, swish(x), 'r-', linewidth=2.5, label='SiLU 门控 (SwiGLU)', alpha=0.9)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
    
    # 标注 sigmoid 的饱和区域
    ax1.axhspan(-0.1, 0.1, xmin=0.7, alpha=0.1, color='blue', label='sigmoid 饱和 (梯度≈0)')
    
    ax1.set_xlabel('x (门控输入)', fontsize=12)
    ax1.set_ylabel('门控输出', fontsize=12)
    ax1.set_title('门控函数对比: sigmoid vs SiLU', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(-0.5, 4.5)
    
    # ---- 右图: 对数尺度 ----
    x_pos = np.linspace(0.1, 4, 200)
    ax2.semilogy(x_pos, sigmoid(x_pos), 'b-', linewidth=2, label='sigmoid', alpha=0.8)
    ax2.semilogy(x_pos, swish(x_pos), 'r-', linewidth=2.5, label='SiLU', alpha=0.9)
    ax2.set_xlabel('x (门控输入)', fontsize=12)
    ax2.set_ylabel('门控输出 (log 尺度)', fontsize=12)
    ax2.set_title('门控函数 (对数 y 轴) -- SiLU 远大于 sigmoid', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25, which='both')
    
    plt.tight_layout()
    plt.savefig('swiglu_05_gate_comparison.png', bbox_inches='tight')
    plt.show()
    print("[图5] 门控函数对比图已保存: swiglu_05_gate_comparison.png")


# ============================================================
# 主执行
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("SwiGLU 激活函数 -- 可视化分析")
    print("=" * 55)
    
    plot_activation_comparison()
    plot_swish_derivative()
    plot_swiglu_gating()
    plot_silu_decomposition()
    plot_gate_input_relationship()
    
    print("\n全部 5 张可视化图表已生成完毕!")
    print("文件列表:")
    print("  - swiglu_01_activation_comparison.png")
    print("  - swiglu_02_silu_derivative.png")
    print("  - swiglu_03_gating_heatmap.png")
    print("  - swiglu_04_silu_decomposition.png")
    print("  - swiglu_05_gate_comparison.png")
```

## 10. 模型评估

```python
"""
使用 SwiGLU 和 ReLU 前馈层的小型 Transformer 对比实验
在 AG_NEWS 文本分类任务上评估（4 分类，~120k 训练样本）
Python 3.9+, PyTorch 2.0+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


# ============================================================
# 1. 前馈网络定义
# ============================================================
class FFN_ReLU(nn.Module):
    """标准两层 ReLU 前馈网络。"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class FFN_SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络。
    
    结构: x -> [gate: d->d_ff, silu] x [up: d->d_ff] -> [down: d_ff->d]
    
    与 FFN_ReLU 的区别：
    1. 使用三个线性层而非两个
    2. 中间的激活从 ReLU 改为 SwiGLU 门控
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 两条并行路径
        gate = F.silu(self.w_gate(x))  # SiLU 门控
        up   = self.w_up(x)            # 信息值
        # 门控调制 + 输出投影
        return self.w_down(self.dropout(gate * up))


# ============================================================
# 2. Transformer 模块
# ============================================================
class TransformerBlock(nn.Module):
    """
    单个 Transformer 块 (Pre-LN 风格，与 LLaMA 一致)。
    
    流程: x -> LayerNorm -> Attention(+residual) -> LayerNorm -> FFN(+residual)
    """
    def __init__(self, d_model, n_heads, d_ff, ffn_type='relu', dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # 根据 ffn_type 选择不同的 FFN
        if ffn_type == 'swiglu':
            self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        else:
            self.ffn = FFN_ReLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None):
        # Pre-LN Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x,
                                      key_padding_mask=padding_mask,
                                      need_weights=False)
        x = residual + self.dropout(attn_out)
        
        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x


class TextClassifier(nn.Module):
    """
    小型 Transformer 文本分类器。
    
    Args:
        ffn_type: 'relu' 或 'swiglu'，控制使用哪种前馈网络
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 d_ff=512, num_classes=4, ffn_type='relu',
                 max_length=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, ffn_type, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model
    
    def forward(self, tokens):
        # tokens: (batch, seq_len)
        padding_mask = (tokens == 1)  # <pad> token id = 1
        
        # Token Embedding + Positional Encoding
        x = self.embedding(tokens) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer 层堆叠
        for block in self.blocks:
            x = block(x, padding_mask)
        
        # 简单的平均池化 + 分类头
        # 对于 padding 位置，我们不参与平均
        if padding_mask is not None:
            # 将 padding 位置设为 0，然后对非 padding 位置求均值
            mask_expanded = (~padding_mask).float().unsqueeze(-1)  # (B, S, 1)
            x = x * mask_expanded
            seq_lens = mask_expanded.sum(dim=1).clamp(min=1)       # (B, 1)
            pooled = x.sum(dim=1) / seq_lens
        else:
            pooled = x.mean(dim=1)
        
        pooled = self.norm_final(pooled)
        return self.classifier(pooled)


# ============================================================
# 3. 合成文本分类数据（避免依赖外部数据集）
# ============================================================
def create_synthetic_text_data(num_samples=5000, vocab_size=5000,
                                max_length=64, num_classes=4):
    """
    创建合成文本分类数据用于快速评估。
    
    这样避免了依赖 torchtext 下载外部数据集，
    可以在任何环境中直接运行评估实验。
    
    虽然这不是真实验证 SwiGLU 优势的最佳方式（合成数据过于简单），
    但足以验证代码正确性和观察基本趋势。
    """
    torch.manual_seed(42)
    
    # 生成随机的 token 序列
    # 为每个类别创建略有不同的 token 分布
    tokens_list = []
    labels_list = []
    
    for c in range(num_classes):
        n = num_samples // num_classes
        # 每个类别有不同的 token 倾向分布
        class_bias = torch.randn(vocab_size) * 0.5
        class_bias = F.softmax(class_bias, dim=0)
        
        for _ in range(n):
            # 生成长度不一的序列
            seq_len = torch.randint(max_length // 2, max_length + 1, (1,)).item()
            # 从此类别的分布中采样 token
            tokens = torch.multinomial(class_bias, seq_len, replacement=True)
            # 填充到固定长度
            if seq_len < max_length:
                padding = torch.ones(max_length - seq_len, dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            tokens_list.append(tokens.unsqueeze(0))
            labels_list.append(c)
    
    tokens = torch.cat(tokens_list, dim=0)
    labels = torch.tensor(labels_list)
    
    # 打乱
    shuffle_idx = torch.randperm(len(labels))
    tokens = tokens[shuffle_idx]
    labels = labels[shuffle_idx]
    
    # 划分训练集/测试集 (80/20)
    split = int(0.8 * len(labels))
    return (tokens[:split], labels[:split],
            tokens[split:], labels[split:])


# ============================================================
# 4. 训练与评估循环
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for tokens_batch, labels_batch in loader:
        tokens_batch = tokens_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(tokens_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels_batch.size(0)
        correct += (logits.argmax(dim=1) == labels_batch).sum().item()
        total += labels_batch.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """评估一个 epoch。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for tokens_batch, labels_batch in loader:
        tokens_batch = tokens_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        logits = model(tokens_batch)
        loss = criterion(logits, labels_batch)
        
        total_loss += loss.item() * labels_batch.size(0)
        correct += (logits.argmax(dim=1) == labels_batch).sum().item()
        total += labels_batch.size(0)
    
    return total_loss / total, correct / total


# ============================================================
# 5. 主实验
# ============================================================
def run_evaluation():
    """
    主评估实验：对比 ReLU FFN 和 SwiGLU FFN 的 Transformer。
    """
    # 超参数
    d_model = 128
    n_heads = 4
    n_layers = 3
    d_ff = 512
    num_classes = 4
    batch_size = 64
    epochs = 10
    lr = 0.001
    dropout = 0.1
    vocab_size = 5000
    max_length = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"模型配置: d_model={d_model}, n_heads={n_heads}, "
          f"n_layers={n_layers}, d_ff={d_ff}")
    
    # 准备数据
    print("\n创建合成文本分类数据...")
    X_train, y_train, X_test, y_test = create_synthetic_text_data(
        num_samples=4000, vocab_size=vocab_size,
        max_length=max_length, num_classes=num_classes
    )
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    
    # 创建两个模型
    relu_model = TextClassifier(
        vocab_size, d_model, n_heads, n_layers, d_ff,
        num_classes, ffn_type='relu', max_length=max_length, dropout=dropout
    ).to(device)
    
    swiglu_model = TextClassifier(
        vocab_size, d_model, n_heads, n_layers, d_ff,
        num_classes, ffn_type='swiglu', max_length=max_length, dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    relu_opt = torch.optim.AdamW(relu_model.parameters(), lr=lr)
    swiglu_opt = torch.optim.AdamW(swiglu_model.parameters(), lr=lr)
    
    # 记录训练历史
    history = {
        "relu":   {"train_loss": [], "train_acc": [],
                   "test_loss": [],  "test_acc": []},
        "swiglu": {"train_loss": [], "train_acc": [],
                   "test_loss": [],  "test_acc": []}
    }
    
    # 训练
    print("\n" + "=" * 75)
    header = (f"{'Epoch':<7} {'ReLU TrLoss':<13} {'ReLU TeAcc':<12} "
              f"{'SwiGLU TrLoss':<15} {'SwiGLU TeAcc':<14}")
    print(header)
    print("=" * 75)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ReLU
        r_tr_loss, r_tr_acc = train_epoch(
            relu_model, train_loader, relu_opt, criterion, device)
        r_te_loss, r_te_acc = evaluate_epoch(
            relu_model, test_loader, criterion, device)
        
        history["relu"]["train_loss"].append(r_tr_loss)
        history["relu"]["train_acc"].append(r_tr_acc)
        history["relu"]["test_loss"].append(r_te_loss)
        history["relu"]["test_acc"].append(r_te_acc)
        
        # SwiGLU
        s_tr_loss, s_tr_acc = train_epoch(
            swiglu_model, train_loader, swiglu_opt, criterion, device)
        s_te_loss, s_te_acc = evaluate_epoch(
            swiglu_model, test_loader, criterion, device)
        
        history["swiglu"]["train_loss"].append(s_tr_loss)
        history["swiglu"]["train_acc"].append(s_tr_acc)
        history["swiglu"]["test_loss"].append(s_te_loss)
        history["swiglu"]["test_acc"].append(s_te_acc)
        
        print(f"{epoch+1:<7} {r_tr_loss:<13.4f} {r_te_acc:<12.4f} "
              f"{s_tr_loss:<15.4f} {s_te_acc:<14.4f}")
    
    elapsed = time.time() - start_time
    print("=" * 75)
    
    # 汇总
    relu_params = sum(p.numel() for p in relu_model.parameters())
    swiglu_params = sum(p.numel() for p in swiglu_model.parameters())
    
    print(f"\n训练耗时: {elapsed:.1f} 秒")
    print(f"\n{'='*55}")
    print(f"最终结果汇总:")
    print(f"  {'':<20} {'ReLU FFN':<18} {'SwiGLU FFN':<18}")
    print(f"  {'-'*55}")
    print(f"  {'参数量':<20} {relu_params:<18,} {swiglu_params:<18,}")
    print(f"  {'最终训练 Loss':<20} {history['relu']['train_loss'][-1]:<18.4f} "
          f"{history['swiglu']['train_loss'][-1]:<18.4f}")
    print(f"  {'最终训练 Acc':<20} {history['relu']['train_acc'][-1]:<18.4f} "
          f"{history['swiglu']['train_acc'][-1]:<18.4f}")
    print(f"  {'最终测试 Loss':<20} {history['relu']['test_loss'][-1]:<18.4f} "
          f"{history['swiglu']['test_loss'][-1]:<18.4f}")
    print(f"  {'最终测试 Acc':<20} {history['relu']['test_acc'][-1]:<18.4f} "
          f"{history['swiglu']['test_acc'][-1]:<18.4f}")
    print(f"{'='*55}")
    
    # 绘制训练曲线
    plot_comparison_curves(history)
    
    return history


def plot_comparison_curves(history):
    """绘制 ReLU vs SwiGLU 训练曲线对比。"""
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = range(1, len(history["relu"]["test_acc"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---- Test Accuracy ----
    axes[0].plot(epochs, history["relu"]["test_acc"], 'b-o',
                 linewidth=2, markersize=5, label='ReLU FFN')
    axes[0].plot(epochs, history["swiglu"]["test_acc"], 'r-s',
                 linewidth=2, markersize=5, label='SwiGLU FFN')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Test Accuracy: ReLU vs SwiGLU', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # ---- Train Loss ----
    axes[1].plot(epochs, history["relu"]["train_loss"], 'b-o',
                 linewidth=2, markersize=5, label='ReLU FFN')
    axes[1].plot(epochs, history["swiglu"]["train_loss"], 'r-s',
                 linewidth=2, markersize=5, label='SwiGLU FFN')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Train Loss', fontsize=12)
    axes[1].set_title('Train Loss: ReLU vs SwiGLU', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('swiglu_evaluation_curves.png', bbox_inches='tight')
    plt.show()
    print("[评估] 训练曲线图已保存: swiglu_evaluation_curves.png")


if __name__ == "__main__":
    run_evaluation()
```

## 11. 常见问题与易错点

### 数据层面

**问题1：小 batch size 下 SiLU 门控的统计不稳定**

- **现象**：当 batch size 较小时（如 < 16），门控信号在不同 step 间剧烈波动，导致训练损失出现 high-variance 的噪声。尤其在序列较短的文本上更为明显。
- **原因**：SiLU 门控值是通过 `silu(xW_gate)` 生成的，其统计特性（均值、方差）对 batch 中样本的分布敏感。小 batch 意味着每个训练步的统计估计不可靠，门控信号的有效范围不稳定，间接导致 FFN 输出层的梯度估计方差过大。
- **解决方案**：(1) 增大 batch size 到至少 32 以上，让门控的统计更稳定；(2) 在门控投影后、SiLU 之前加入 LayerNorm，归一化门控输入空间，降低 batch 敏感性；(3) 使用梯度累积，在多个小 batch 上累积梯度后再更新，等效增大有效 batch size。

**问题2：长文本中部分维度门控持续接近饱和**

- **现象**：在超长序列（如 4096+ token）上训练时，SwiGLU 的门控信号中某些维度始终接近 0（通道关闭），而另一些维度始终很大（通道全开且放大），存在明显的维度利用率不均。
- **原因**：长序列中 token 的分布差异极大——前面的 token 和后面的 token 可能需要完全不同的信息处理模式。如果门控权重 `W_gate` 在训练早期学到了某种偏向（例如更倾向于处理序列前半部分的信息），后续优化可能无法纠正，导致部分门控维度"锁死"在极端值。
- **解决方案**：(1) 在 FFN 的 gate 分支也加入 dropout，增加门控的随机性，防止早熟收敛；(2) 监控门控信号的 per-dimension 统计（均值、使用率），如果发现某些维度长期接近 0，可以针对性地重置这些维度；(3) 使用 LLaMA 风格的 pre-LayerNorm 设计，在 FFN 输入前做归一化，减少输入分布漂移对门控的影响。

### 模型层面

**问题3：SwiGLU 参数量膨胀导致的小规模模型过拟合**

- **现象**：在小模型（<50M 参数）上将 ReLU FFN 替换为 SwiGLU 后，训练损失比 ReLU 更低，但验证损失反而更高（明显过拟合），最终测试指标不如 ReLU 基准。
- **原因**：SwiGLU FFN 的三个权重矩阵比 ReLU FFN 多出约 50% 的参数。在小模型上，ReLU FFN 可能已经接近"参数量刚好够"的状态，新增的参数没有在足够的数据约束下学到有意义的模式，而是记住了训练数据中的噪声。
- **解决方案**：(1) 缩小 SwiGLU 的 `d_ff` 到原来的 2/3，使参数量与 ReLU FFN 持平（这是 LLaMA 的标准做法）；(2) 增加 dropout rate 或在 gate 分支也加 dropout；(3) 使用更强的权重衰减（weight decay）；(4) 在小模型上考虑使用 SiLU 逐元素激活（不引入门控）而不是完整的 SwiGLU，这是一个更轻量的替代方案。

**问题4：Gate 分支和 Up 分支权重的初始化不对称导致的训练初期不稳定**

- **现象**：训练刚开始的几个 step，loss 出现 spike（突然飙升到很高的值），然后逐步回落。重启训练可能完全消失，说明与初始化有关。
- **原因**：如果 `W_gate` 和 `W_up` 的初始化使得 `gate * up` 的幅值远大于正常范围，门控输出在训练早期会产生数值爆炸。由于两个矩阵各自独立初始化（通常都是 `N(0, 0.02)`），它们的乘积 `silu(xW_gate) * (xW_up)` 可能和合理的量级相差很远。
- **解决方案**：(1) 使用更小的初始化方差（如 `std=0.006` 或 `std=0.02/sqrt(2)`）来补偿两个分支相乘带来的幅值放大；(2) 在 FFN 的输入处使用 LayerNorm（Pre-LN 设计），确保输入 x 的方差稳定在 1 附近，这样两个分支的中间值也在可控范围；(3) 使用学习率 warmup，在训练初期逐步提高学习率，避免大梯度更新放大不稳定性。

### 调参层面

**问题5：d_ff/d 比例选择不当导致性能不达预期**

- **现象**：使用了 SwiGLU 但保持 `d_ff = 4 * d`（和 ReLU FFN 一样的比例），结果参数量增加了 50% 但困惑度只下降了不到 1%，投入产出比很低。
- **原因**：SwiGLU 的额外参数量如果没有被有效利用（例如数据量不足以训练这些参数，或优化过程没有充分探索参数空间），就会成为"死参数"——占用了计算和内存但不贡献性能提升。特别是在中等规模模型上（100M-1B），直接增加 50% 参数可能导致欠拟合。
- **解决方案**：(1) 使用 `d_ff = 8/3 * d` 的比例，这是 LLaMA 在实践中验证的最优折中——参数量持平但表达力更强；(2) 如果希望保留更大的 `d_ff`，确保训练数据量和训练步数足够，让额外参数有机会被充分训练；(3) 在训练过程中监控每个门控维度的激活频率和梯度范数，判断额外参数是否"活着"。

**问题6：在 FP16 混合精度下 SiLU 计算的数值不稳定**

- **现象**：使用 FP16 自动混合精度（AMP）训练时，偶尔出现 NaN 梯度，且 NaN 总是在 FFN 层的 gate 分支中首先出现。切换到 FP32 后问题完全消失。
- **原因**：`silu(x) = x / (1 + e^{-x})` 在 FP16 下有两个数值陷阱：(1) 当 `x` 较大正值时，`e^{-x}` 下溢出到 0，分母变为 1，计算仍然正确；(2) 当 `x` 为较大负值（如 -20 以下）时，`e^{-x}` 上溢出到 FP16 的最大值（65504），导致 Inf；(3) 即使未溢出，FP16 的精度（约 3 位有效十进制数字）在计算 `1 + e^{-x}` 时可能出现严重的舍入误差。
- **解决方案**：(1) 使用 BF16 替代 FP16——BF16 的动态范围与 FP32 相同，不会出现指数上溢/下溢；(2) 在 AMP 中，将 SwiGLU 的 gate 计算标记为需要在 FP32 下执行；(3) 使用 PyTorch 的 `F.silu` 而非手动 `x * sigmoid(x)`——PyTorch 的 SiLU 有专门的 CUDA kernel，内部处理了数值稳定性。

## 12. 学习总结

SwiGLU 是现代大语言模型中最关键的激活函数创新之一。它巧妙地将"门控机制"与"Swish（SiLU）激活"结合，用独立的可学习权重矩阵生成门控信号，实现了比传统 ReLU/GeLU 更丰富的非线性表达能力。

理解 SwiGLU 的关键在于把握三个层次：

**第一层 -- 门控的本质**：不同于 ReLU 的"自门控"（用自己的值决定是否通过），SwiGLU 使用独立学习的分支 `W_gate` 来决定信息通过量。这意味着某个特征维度即使数值很大，如果从另一个"视角"（经由不同的线性变换）判断认为不重要，也可以被抑制。这赋予了网络一种"多角度评估信息重要性"的能力。

**第二层 -- Swish 的优越性**：`silu(x) = x · σ(x)` 是一个处处可微、梯度连续的平滑函数。它在正值区域接近线性（梯度约等于 1），在负值区域平滑衰减（非硬截断），且在 x≈-1.28 处有一个独特的微小负值谷底。这些性质在深层网络中协同累积，带来比 ReLU 更稳定的优化轨迹和更好的梯度流动。

**第三层 -- 实践价值**：SwiGLU 已经从学术好奇变成工业标准。LLaMA 系列、DeepSeek-V2/V3、PaLM 等旗舰大模型都将其作为 FFN 的默认激活函数。在使用 SwiGLU 时，关键实践要点包括：将中间维度设为 `8/3 * d` 以保持参数量持平、使用 Pre-LayerNorm 稳定门控输入、使用 BF16 或 FP32 确保 SiLU 计算的数值精度。

掌握 SwiGLU 不仅是理解现代大模型代码的关键，更是深入 MoE 架构、下一代激活函数设计的重要基础。

## 13. 练习题与思考题

### 基础题

**题目1**：写出 SwiGLU 在 Transformer FFN 中的完整前向计算公式，并标注每一步的输入输出维度。假设输入 `x` 维度为 `(batch, seq_len, d_model)`，中间维度为 `d_ff`。

**参考答案**：

SwiGLU FFN 的完整前向计算分为以下步骤：

1. **门控投影**：`gate_input = x @ W_gate`，其中 `W_gate` 的维度为 `(d_model, d_ff)`，输出维度为 `(batch, seq_len, d_ff)`
2. **值投影**：`up_input = x @ W_up`，其中 `W_up` 的维度为 `(d_model, d_ff)`，输出维度为 `(batch, seq_len, d_ff)`
3. **SiLU 激活**：`gate = silu(gate_input)`，其中 `silu(z) = z · σ(z)`，输出维度为 `(batch, seq_len, d_ff)`
4. **逐元素门控**：`gated = gate ⊙ up_input`（Hadamard 积），输出维度为 `(batch, seq_len, d_ff)`
5. **输出投影**：`output = gated @ W_down`，其中 `W_down` 的维度为 `(d_ff, d_model)`，输出维度为 `(batch, seq_len, d_model)`

公式总结：
$$SwiGLU\_FFN(x) = (silu(x \cdot W_{gate}) \odot (x \cdot W_{up})) \cdot W_{down}$$

---

**题目2**：如果标准 ReLU FFN 的参数量为 `P_relu = 2 * d_model * d_ff`，那么 SwiGLU FFN 的参数量是多少？如果想让两个模型的参数量相等，且原始 ReLU FFN 中 `d_ff = 4 * d_model`，SwiGLU 的 `d_ff'` 应该设为多少？

**参考答案**：

SwiGLU FFN 有三个权重矩阵：`W_gate (d_model × d_ff')`、`W_up (d_model × d_ff')`、`W_down (d_ff' × d_model)`。

参数量为：
$$P_{swiglu} = d\_model \times d\_ff' + d\_model \times d\_ff' + d\_ff' \times d\_model = 3 \times d\_model \times d\_ff'$$

当原始 ReLU FFN 采用 `d_ff = 4 * d_model` 时：
$$P_{relu} = 2 \times d\_model \times 4 \times d\_model = 8 \times d\_model^2$$

令两者相等：
$$3 \times d\_model \times d\_ff' = 8 \times d\_model^2$$
$$d\_ff' = \frac{8}{3} \times d\_model \approx 2.67 \times d\_model$$

因此 SwiGLU 的 `d_ff'` 应设为原来的 `2.67 / 4 = 2/3` 倍。这正是 LLaMA 等模型采用 `d_ff = 8/3 * d_model` 的原因。

### 进阶题

**题目3**：在 SwiGLU 中，门控函数 SiLU 的输出范围大致是 `[-0.278, +∞)`，这不像 sigmoid 那样被限制在 [0,1]。负的门控值会导致什么效果？这可能是 bug 还是 feature？请结合梯度和网络行为进行分析。

**参考答案**：

**负门控值的效果**：当 `gate < 0` 时，`gate ⊙ up` 的结果符号与 `up` 相反。这意味着门控不仅可以选择"通过多少"信息，还能实现**符号翻转**（phase inversion）。具体场景例如：
- 如果某个特征维度在处理否定语义时需要输出与正面语义相反的方向，负门控就提供了这个能力
- 作为一种"减性"交互——如果两个专家分别输出了相反方向的信号，它们的加权组合可以产生更丰富的函数族

**这是 feature 而非 bug**，理由如下：

1. **表达能力增强**：允许门控在 `[-0.278, 0]` 范围内取负值相当于给了网络一个额外的自由度。这类似于线性层的负权重——限制权重只能为正会严重削弱网络表示能力。

2. **梯度行为良好**：SiLU 在最小值附近（`x ≈ -1.278`）的导数并非为零。计算 `silu'(-1.278) ≈ 0.05`，虽然小但非零，梯度仍然可以回传。如果门控在这个区域，它仍然能接收到梯度信号并调整。

3. **与残差连接的兼容性**：Transformer 使用 `output = x + FFN(x)`。如果 FFN 输出有正有负，它可以作为"修正信号"——不仅增加某些方向，也能抵消某些方向。纯粹的 ReLU（输出恒 ≥ 0）只能做正向修正，所有特征方向只能被增强不能被抑制。

4. **负值的幅度有限**：最负也仅约 -0.278，且仅在很小的输入区间内取负值。在大多数实际激活状态下（尤其是经过 LayerNorm 的输入），门控值通常为正。

**注意点**：在 SFT（有监督微调）或 RLHF 场景中，如果分布发生 shift，更多的 token 可能落入负门控区间，需要在推理时监控数值。

---

**题目4**：在分布式训练中使用张量并行（Tensor Parallelism）时，SwiGLU FFN 和 ReLU FFN 的跨设备通信次数有什么不同？请结合 SwiGLU 的门控计算过程进行分析。

**参考答案**：

假设使用张量并行，将 FFN 的中间维度（`d_ff`）切分到 N 个设备上。每个设备持有权重矩阵的部分列/行：

#### ReLU FFN 的通信
1. `W_up [d_model, d_ff]` 按列切分：设备 i 持有 `W_up_i [d_model, d_ff/N]`
2. 各设备计算 `h_i = ReLU(x @ W_up_i)` → **无需通信**
3. All-Reduce：需将各设备的 `h_i` 合并得到完整的 `h` → **第 1 次通信**
4. `W_down [d_ff, d_model]` 按行切分：设备 i 持有 `W_down_i [d_ff/N, d_model]`
5. 各设备计算 `out_i = h @ W_down_i`
6. All-Reduce：合并 `out_i` → **第 2 次通信**
7. **总计：2 次 All-Reduce**

#### SwiGLU FFN 的通信
1. `W_gate [d_model, d_ff]` 按列切分：设备 i 计算 `gate_i = silu(x @ W_gate_i)` → **无需通信**
2. `W_up [d_model, d_ff]` 按列切分：设备 i 计算 `up_i = x @ W_up_i` → **无需通信**
3. 逐元素乘法 `gated_i = gate_i ⊙ up_i` → **无需通信**（关键！门控乘法在本地分片内完成，不需要跨设备协同）
4. `W_down [d_ff, d_model]` 按行切分：设备 i 持有 `W_down_i [d_ff/N, d_model]`
5. 各设备计算 `out_i = gated_i @ W_down_i`
6. All-Reduce：合并 `out_i` → **仅 1 次通信**
7. **总计：1 次 All-Reduce**

**结论**：SwiGLU 在张量并行下比 ReLU FFN **少了一次 All-Reduce**！原因是门控激活（SiLU）和门控乘法（`gate ⊙ up`）都在每个设备的本地分片上独立完成。在大规模分布式训练中（数百个 GPU），All-Reduce 通信往往是瓶颈，SwiGLU 节省了一半的 FFN 通信量——这是一个在架构设计时可能没有预见到、但实际上非常重要的额外优势。这也是为什么现代大模型几乎一致地选择了 SwiGLU。

### 开放思考题

**题目5**：SwiGLU 的成功催生了大量 GLU 变体（如 ReGLU、GeGLU、SwiGLU）。请提出一种你自己的 GLU 变体设计，阐述你的设计动机（解决了 SwiGLU 的什么不足），并设计一个简单的对比实验方案来验证你的假设。不需要实际跑实验，但要写清楚实验设置、预期结果、以及可能的失败模式。

**参考答案**（这是一个开放题，以下是示例分析思路，任何合理的创新思路都应该被认可）：

**候选设计思路 1："Adaptive-GLU"**
- **动机**：SwiGLU 的门控程度对所有维度一视同仁（都使用 SiLU），但在推理时，不同维度的"信息重要性"可能随输入变化。例如，对于句子末尾的 token，某些 FFN 维度可能需要更强的门控。
- **设计**：在 SiLU 门控之前，引入一个可学习的"门控温度"向量 `τ`（维度 `d_ff`），计算 `gate = silu(gate_input / τ)`。当 `τ > 1`，门控变"暖和"（更像线性），当 `τ < 1`，门控变"尖锐"（更像开关）。允许每个维度自适应。
- **实验方案**：
  - 基准：标准 SwiGLU（d_ff = 8/3 d）
  - 实验组：Adaptive-GLU，d_ff 设为与基准参数量相等
  - 任务：语言模型预训练 + 下游任务评估
- **预期结果**：自适应温度参数使得模型在不同层有不同的门控敏感度——浅层可能需要更"暖和"的门控（信息保留更多），深层需要更"尖锐"的门控（选择性更强）。预期训练困惑度略优于标准 SwiGLU。
- **可能的失败模式**：
  - `τ` 可能在学习过程中坍塌到 1（退化回标准 SwiGLU）或发散到极端值，需要正则化约束
  - 额外的可学习参数在各层独立，可能导致层间不协调
  - 在 FP16 下除法可能带来额外精度损失

**候选设计思路 2："Sparse-GLU"**
- **动机**：SwiGLU 的所有 `d_ff` 个门控维度都参与计算，但可能大部分维度对当前 token 没有实质影响。可以在门控后引入稀疏性。
- **设计**：`gate_thresholded = gate * I(|gate| > threshold)` ——只保留门控值超过阈值的维度。阈值可设为门控值分布的第 k 百分位数。
- **可能的失败模式**：稀疏选择不可微，训练阶段需要用 STE（Straight-Through Estimator）近似梯度，可能导致训练不稳定。

**评判标准**：合理的创新应满足：1. 有一个明确的洞察；2. 与现有方法有清晰的区别；3. 对可能的失败有预见。

## 14. 学习路径建议

**前置知识（必须先牢固掌握）**：

- **前馈神经网络（FFN）在 Transformer 中的角色**：理解每个 Transformer 层的 FFN 负责对每个位置独立地进行非线性变换。这是 SwiGLU 发挥作用的具体位置。推荐阅读《Attention Is All You You Need》中关于 Position-wise Feed-Forward Networks 的部分。
- **ReLU 及其局限性**：理解 `max(0, x)` 的"死神经元"问题——当某个神经元对所有输入都输出 0 时，它的梯度永远为 0，参数无法更新。这为理解 SiLU 的"平滑门控"为什么更好提供了直接动机。
- **Transformer 架构完整理解**：Multi-Head Attention、Layer Normalization、残差连接的原理和交互。特别要理解 LayerNorm 如何影响 FFN 的输入分布——这直接关系 SwiGLU 门控信号的稳定性。
- **sigmoid 函数与门控概念**：`σ(x) = 1/(1+e^{-x})` 的输出范围 (0,1) 如何自然地成为一个"软开关"——这是 GLU 的本质出发点。理解 LSTM/GRU 中的门控也有助于建立直觉。

**平行学习（建议同时或交替学习）**：

- **GeLU（Gaussian Error Linear Unit）**：BERT/GPT-2/GPT-3 使用的激活函数，`x · Φ(x)`（Φ 是高斯 CDF）。它与 SiLU 共享"x 乘以累积分布函数"的形式——SiLU 用 sigmoid (logistic CDF)，GeLU 用 Gaussian CDF。理解它们的共性（自门控，处处可微）和区别（近似的线性区间位置、负值区域行为）。
- **GLU 变体家族**：将 ReGLU（ReLU-GLU）、GeGLU（GeLU-GLU）、SwiGLU 放在统一框架 "GLU(x) = activate(xW₁) ⊙ (xW₂)" 中对比。理解"为什么 SiLU 作为门控最好"——这可能因为 SiLU 的梯度最连续、负值区域的行为最合理。推荐阅读 Shazeer (2020) 的论文。
- **GQA（Grouped-Query Attention）**：GQA 也是注意力机制的工程优化——它将 Query 分组，组内 Query 共享 K/V。这与 SwiGLU 共享 K/V 的思想有形式上的相似——都是"不完全独立，分组/全局共享"来减少参数量/缓存。理解这两种"共享"的动机和技术细节的异同。

**进阶方向（深入探索的推荐路线）**：

1. **MLA（Multi-head Latent Attention）--DeepSeek 的极致优化**：DeepSeek-V2/V3 提出的 MLA 通过将 K/V 压缩到一个低维的"潜在空间"（latent space）来进一步减少 KV Cache。理解 MLA 如何将 SwiGLU 的"门控优化"思路延伸到注意力层——通过对 KV 的低秩分解代替直接压缩头数。推荐阅读 DeepSeek-V2 技术报告。

2. **SwiGLU 在 MoE（混合专家）中的运用**：在 MoE 架构中，每个专家都是一个独立的 SwiGLU FFN。理解"稀疏激活的 SwiGLU 专家群体"如何协同工作——专家的路由选择与门控内部的维度抑制是两个层级的"选择性"。

3. **DeepSeekMoE（细粒度专家 + 共享专家）**：这是 SwiGLU + MoE 的顶级工程实践。将标准 SwiGLU 专家进一步细化为更小粒度的单元，并引入一个被所有 token 共享的"共享专家"。推荐阅读《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》。

4. **Flash Attention 与 SwiGLU 的协同优化**：Flash Attention 通过 IO-aware 的 tiling 技术优化了注意力层的显存和速度，但它不直接优化 FFN 的 SwiGLU 部分。理解如何用类似的"显存感知"思路优化 SwiGLU 的门控计算——例如 gate 投影与 SiLU 的 kernel fusion。

**推荐论文阅读顺序**（由浅入深）：
1. Ramachandran et al. (2017) "Searching for Activation Functions"——自动搜索发现 Swish，理解它的来源
2. Shazeer (2020) "GLU Variants Improve Transformer"——系统比较 GLU 变体，确立 SwiGLU 优势（必读）
3. Touvron et al. (2023) "LLaMA: Open and Efficient Foundation Language Models"——SwiGLU 在开源大模型中的实践，注意第 2.4 节关于 SwiGLU 的参数设置
4. DeepSeek-AI (2024) "DeepSeek-V2"——从 SwiGLU 出发，进入 MLA + DeepSeekMoE 的更广阔世界
