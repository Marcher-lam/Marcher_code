# 混合专家模型 (MoE) 学习文档

> 来源线索：本节内容根据原书中关于"混合专家模型"（第5章 5.1-5.2节）的相关章节整理、扩展与教学化改写。

> 多个专业"专家"各司其职，门控网络动态调度——用稀疏激活换取巨大模型容量。

## 1. 算法基础认知

**一句话定义**：混合专家模型（Mixture of Experts, MoE）是一种深度学习架构，它包含多个独立的子网络（称为"专家"），并通过一个可学习的门控网络（Router）动态决定每个输入应该由哪些专家处理，从而实现"条件计算"——用更大的参数总量换取更少的实际激活计算量。

**直觉类比**：想象一家大型公司。公司里有不同的专业部门（专家）：市场部擅长推广、研发部擅长技术攻关、财务部擅长核算。当一项任务到来时，前台（门控网络/Router）会根据任务类型判断应该把它分配给哪个或哪几个部门。前台并不会呼叫所有部门全部出动，而是精准调度，只让最合适的部门处理。这样一来，公司整体能力极强（参数总量大），但单个任务的处理成本却很经济（激活参数少）。

**历史背景**：
- **1991年**，Jacobs 等人首次提出混合专家的概念，最初用于监督学习中的"分而治之"策略，多个简单模型各自负责输入空间的不同区域。
- **2017年**，Shazeer 等人在 ICLR 上发表 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"，将 MoE 与深度神经网络尤其是 LSTM/Transformer 结合，提出稀疏门控机制，使得 MoE 可以扩展到数千个专家，真正进入大模型时代。
- **2021年**，Google 发布 Switch Transformer，将 Top-K 简化为 Top-1（每个 token 只激活一个专家），极大简化路由，支持万亿参数级别模型。
- **2024年**，DeepSeek 提出 DeepSeekMoE，引入细粒度专家划分和共享专家机制，进一步提升 MoE 在语言模型中的效率与性能，并成功应用于 DeepSeek-V2 / V3 等顶级开源大模型。

**算法定位**：MoE 不是一种独立的监督/无监督算法，而是一种**深度学习架构模式**，通常作为 Transformer 中前馈网络（FFN）层的替代品。它可被视为一种"稀疏激活的集成学习"——集成多个子模型但每次只激活少数。

**前置知识**：
- 前馈神经网络（MLP / FFN）的基本结构：线性层 + 激活函数
- Softmax 函数及其数学定义
- Transformer 架构基础：Self-Attention 和 FFN 的交替结构
- 稀疏性概念：理解"大部分参数不参与当前计算"的含义

## 2. 核心原理

### 2.1 核心思想

在标准的深度学习模型中，每一层的所有参数都会参与前向计算和反向传播。当模型参数量增大时，计算成本呈线性甚至超线性增长。MoE 的核心思想是：**虽然模型拥有大量参数（多个专家），但对于每个输入，只激活其中少数专家参与计算**。这种"条件计算"机制使得模型容量和计算成本解耦——参数可以非常大，但实际 FLOPs 保持可控。

### 2.2 工作流程

MoE 层对每个输入 token 的处理流程可以分为四步：

1. **打分**：输入向量 x 通过路由器的线性层，得到每个专家的"亲和度得分"（logits）。
2. **选择**：应用 Top-K 选择，只保留得分最高的 K 个专家（通常 K=1 或 2），其余专家得分置零。可选的加入噪声以促进探索。
3. **专家处理**：被选中的 K 个专家分别以 x 为输入，生成各自的输出向量。
4. **加权组合**：将各专家的输出按其对应的门控权重（对选中的得分做 softmax 归一化）进行加权求和，得到最终输出。

### 2.3 四个关键组件

**专家网络（Expert Network）**：

每个专家是一个独立的小型前馈网络（通常与 Transformer 中的 FFN 结构类似），例如：
```
Expert(x) = W_down(ReLU(W_up(x)))
```
所有专家共享相同的结构，但参数各自独立。多个专家形成一个 `ModuleList`。

**门控网络 / 路由器（Gating Network / Router）**：

路由器是一个参数化的函数，输入为 x，输出为每个专家的权重。最简形式为：
```
router_logits = Linear(x)   # shape: (num_tokens, num_experts)
```

在 NoisyTopkRouter 中，会额外添加可学习的噪声项。

**稀疏性（Sparsity）**：

这是 MoE 区别于普通软集成模型的关键。只有 Top-K 个专家被激活（K << 专家总数），其余专家的计算和梯度都不参与当前 token 的处理。这实现了"条件计算"，使得训练和推理的 FLOPS 远小于全参数情况。

**输出组合（Output Combination）**：

选中专家的输出按照归一化后的门控权重进行加权求和：
```
y = sum_{i in selected} g_i * Expert_i(x)
```
其中 g_i 是对选中专家的路由器得分取 softmax 后的结果。

### 2.4 数据流动 ASCII 图

```
输入 Token x (shape: [batch, seq_len, d_model])
         |
         v
    [Router / Gate]
    Linear(x) + Noise
         |
         v
  router_logits: [batch*seq, num_experts]
         |
         v
   Top-K Selection: 保留top k，其余置0
         |
         v
  gate_weights: softmax over selected k
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
 Expert0  E1  E2  E3  ... E_{N-1}    <- 仅top-k个被激活
    |    |    |    |    |
    +----+----+----+----+
         |
         v
  Weighted Sum: y = Σ g_i * E_i(x)
         |
         v
   输出 y (shape: [batch, seq_len, d_model])
```

## 3. 数学公式与推导

### 3.1 完整前向传播公式

设输入为 $$x \in \mathbb{R}^{d}$$（为简洁起见，考虑单个 token），专家数量为 $$N$$，激活专家数为 $$K$$。

**路由得分（带噪声）**：

$$h(x) = x \cdot W_g + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}})$$

其中：
- $$W_g \in \mathbb{R}^{d \times N}$$ 是门控权重矩阵
- $$W_{\text{noise}} \in \mathbb{R}^{d \times N}$$ 是噪声权重矩阵
- $$\epsilon \sim \mathcal{N}(0, 1)$$ 是标准正态分布的随机噪声
- $$\text{Softplus}(z) = \log(1 + e^z)$$，保证噪声标准差始终为正

**Top-K 选择**：

记 $$h(x) = [h_1, h_2, \ldots, h_N]$$。定义 Top-K 操作：

$$\text{KeepTopK}(h, k)_i = \begin{cases} h_i, & \text{if } h_i \text{ is among the top-k largest values of } h \\ -\infty, & \text{otherwise} \end{cases}$$

**门控权重**（对保留的 Top-K 取 softmax）：

$$g(x) = \text{Softmax}(\text{KeepTopK}(h(x), K))$$

对 Top-K 以外（值为 -∞）的位置，softmax 结果为 0。

**专家输出**：

第 i 个专家是一个两层 MLP：

$$E_i(x) = W_i^{\text{down}} \cdot \text{ReLU}(W_i^{\text{up}} \cdot x)$$

其中 $$W_i^{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d}$$，$$W_i^{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}$$，$$d_{\text{ff}}$$ 是 FFN 隐藏层维度（通常为 $$4d$$）。

**最终输出**：

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

由于只有 K 个 $$g_i$$ 非零，实际只需计算 K 个专家的输出。

### 3.2 为什么加入噪声有助于负载均衡

在没有噪声的情况下，路由器倾向于总是选择相同的几个专家（"专家塌缩"），因为那些专家经过充分训练后得分始终最高。这导致：
- 某些专家几乎从未被使用，浪费参数
- 被频繁使用的专家过拟合

加入高斯噪声后：
- **打破对称性**：训练初期，噪声使相同初始化的专家产生不同的路由得分，促进探索。
- **随机探索**：偶尔一个较弱专家（加噪声后得分变高）会被选中，获得训练信号，逐渐成长。
- **噪声方差可学习**：$$\text{Softplus}(x \cdot W_{\text{noise}})$$ 让模型自己学习多大的噪声合适——对不确定性高的输入加更多噪声以鼓励探索。

### 3.3 负载均衡辅助损失

为了让所有专家都被均匀使用，通常添加一个辅助损失项：

$$L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中：
- $$f_i = \frac{1}{T} \sum_{x \in \text{batch}} \mathbb{1}\{\text{token } x \text{ selects expert } i\}$$，即专家 i 被选中的频率
- $$P_i = \frac{1}{T} \sum_{x \in \text{batch}} h_i(x)$$（未归一化的路由得分），即专家 i 的平均门控概率
- $$T$$ 是 batch 中的 token 总数
- $$\alpha$$ 是超参数，控制辅助损失的强度

当 $$P_i$$ 和 $$f_i$$ 均为 $$\frac{1}{N}$$（均匀分布）时，损失最小：$$L_{\text{aux}} = \alpha \cdot N \cdot \frac{1}{N} \cdot \frac{1}{N} = \frac{\alpha}{N}$$。任何偏离均匀分布都会增加辅助损失。

### 3.4 梯度流分析

Top-K 操作中，保留项的梯度正常回传；被丢弃（置为 -∞ 或 0）的项，其 softmax 结果为 0，因此这些专家不对应当前 token 产生梯度。这意味着每个 token 仅对 K 个专家的参数产生梯度更新，实现了稀疏梯度传播。

## 4. 训练过程讲解

### 4.1 数据预处理

MoE 的输入通常来自 Transformer 的 Self-Attention 层的输出，已经过 Layer Normalization。不需要针对 MoE 做特殊的数据预处理。但在构建训练 batch 时，需要注意 **batch size 应足够大**（通常建议 512 以上的 token 总数），以确保每个专家都能被分到足够的 token 进行训练。

### 4.2 参数初始化

- **专家参数**：使用标准 MLP 初始化，通常采用 Kaiming 均匀初始化或 Xavier 均匀初始化。
- **路由器参数**：门控权重 $$W_g$$ 和噪声权重 $$W_{\text{noise}}$$ 使用 Xavier 均匀初始化。有实践表明将 $$W_g$$ 初始化为接近零的小值（如均值为 0、标准差为 $$1/\sqrt{d}$$ 的正态分布）可以促进初期的均匀路由。
- **噪声权重**：可初始化为极小的值，让噪声从较小开始逐渐学习增大。

### 4.3 迭代过程

1. 每个 batch 中，标准 Transformer 的 Self-Attention 层对所有 token 正常计算。
2. 进入 MoE-FFN 层后，路由器对每个 token 独立计算路由得分。
3. Top-K 选择后，每个 token 被分配给 1 到 K 个专家。
4. 专家并行处理分配给自己的 token。
5. 加权组合后，加上辅助损失项。
6. 标准反向传播 + 优化器更新。

### 4.4 收敛条件

- 验证集困惑度（Perplexity）趋于稳定
- 专家负载分布趋于均匀（负载均衡损失的绝对值不再持续下降）
- 总训练损失（主损失 + 辅助损失）下降放缓

### 4.5 超参数表

| 超参数 | 典型范围 | 说明 |
|--------|----------|------|
| `num_experts` | 4 - 256 | 专家总数。大模型通常 8-64 |
| `top_k` | 1 - 4 | 每个 token 激活的专家数。Switch Transformer 用 1，Mixtral 用 2 |
| `capacity_factor` | 1.0 - 2.0 | 每个专家能处理的最大 token 数因子，超出则丢弃 |
| `aux_loss_weight` | 0.001 - 0.1 | 负载均衡辅助损失的系数 α |
| `expert_ffn_dim` | 与标准 FFN 相同 | 每个专家的隐藏层维度，通常 d_ff = 4 * d_model |
| `dropout` | 0.0 - 0.1 | 专家内的 Dropout 率 |

## 5. 应用场景

### 5.1 大语言模型中的 FFN 替代

这是 MoE 目前最主流、最成功的应用场景：

- **DeepSeek-V2 / V3**：DeepSeekMoE 将传统 FFN 替换为 MoE 结构，引入"细粒度专家划分"和"共享专家"机制。DeepSeek-V3 的总参数量达 671B，但每次推理仅激活约 37B 参数，效率极高。

- **Mixtral 8x7B**：Mistral AI 发布的 MoE 模型，包含 8 个专家，每个 token 激活 2 个。总参数 46.7B，激活参数仅 12.9B，性能却接近甚至超越 70B 的 Llama2。

- **Switch Transformer**：Google 用 Top-1 路由替代 Top-K，将 MoE 扩展到万亿参数规模，在多个 NLP 任务上保持竞争力的同时大幅降低推理成本。

### 5.2 推荐系统中的多兴趣建模

在推荐系统中，用户可能同时有多个不同的兴趣（如"游戏"和"健身"）。MMoE（Multi-gate Mixture-of-Experts）使用多个门控网络从同一组专家中提取不同"兴趣维度"的特征，在 YouTube 等平台的多目标排序中应用广泛。

### 5.3 多任务学习

MoE 的多专家结构天然适合多任务学习：不同任务可以共享底层专家，但通过不同的门控网络来组合专家的输出。这在搜索、广告等需要同时优化多个目标的场景中非常常见。

### 5.4 适用与不适用场景

**适用场景**：
- 需要大幅增加模型容量但计算预算有限的场景
- 已有成熟的 Dense 模型架构，希望用 MoE 做稀疏放大
- 训练数据量大、batch size 可配置得较大

**不适用场景**：
- batch size 太小，专家负载会极度不均
- 推理时延要求极其严格（MoE 需要加载所有专家参数，显存占用大）
- 模型本身已经很小（参数量 < 100M），MoE 的收益有限

## 6. 优缺点分析

### 6.1 优点

1. **大容量小算力**：参数总量可以极大（数百亿到数千亿），但实际激活的参数量小得多，实现了"用空间换时间"的优雅权衡。
2. **专家专业化**：不同专家会自然学习到处理不同类型的输入。例如在语言模型中，有的专家擅长数学符号、有的擅长代码、有的擅长文学描述，这种隐式分工大大提升了模型的表达能力。
3. **灵活性高**：可以通过调整专家数量、Top-K 值来灵活控制容量-效率的平衡。
4. **可扩展性好**：增加更多专家不会相应增加单次前向的 FLOPs（在合理的通信条件下），模型容量可以近乎线性扩展。

### 6.2 缺点

1. **负载不均**：即使有辅助损失，某些专家仍可能被过多或过少地使用，导致"专家塌缩"。
2. **通信开销大**：在分布式训练中，不同专家可能位于不同设备上，token 分发和结果收集产生了大量的 All-to-All 通信。
3. **训练不稳定**：噪声、负载均衡损失、Top-K 的离散选择操作使训练过程比 Dense 模型更敏感。
4. **显存占用大**：推理时需要将所有专家参数加载到显存中，即使只激活少数。Mixtral 8x7B 需要约 90GB 显存才能推理。
5. **微调困难**：Sparse MoE 对微调数据分布敏感，容易在微调时进一步破坏负载平衡。

### 6.3 MoE 与 Dense 模型对比

| 维度 | Dense 模型 | Sparse MoE 模型 |
|------|------------|-----------------|
| 参数总量 | P | N × P_expert (远大于 P) |
| 每 token 激活参数 | P | K × P_expert (与 P 接近) |
| 每 token FLOPs | ~P | ~K × P_expert (与 Dense 接近) |
| 训练总 FLOPs | 基准 | 与 Dense 接近（但需要分布式通信） |
| 显存需求 | 低 | 高（需加载全部专家） |
| 训练稳定性 | 稳定 | 较不稳定 |
| 实现复杂度 | 简单 | 复杂 |

### 6.4 与普通 Transformer FFN 的对比

普通 Transformer 中的 FFN 层：
```
FFN(x) = W_down(ReLU(W_up(x)))
```

MoE 中每个专家的结构与此完全相同。关键区别在于：普通 FFN 只有一套参数，所有 token 共享；而 MoE 有多套这样的参数（专家），门控网络为每个 token 选择合适的专家。因此 MoE 可以视为"一组 FFN + 动态路由器"。

## 7. 调库实现

以下是一个完整可运行的 PyTorch 实现，包含 Expert、NoisyTopkRouter 和 SparseMoE 三个类。

```python
"""
混合专家模型 (Mixture of Experts, MoE) - 完整 PyTorch 实现
依赖: pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 专家网络 (Expert Network)
# ============================================================
class Expert(nn.Module):
    """
    每个专家是一个简单的两层 MLP（前馈网络），结构与标准 Transformer 的 FFN 相同。
    输入维度 -> 隐藏层维度 (通常 4x) -> 输出维度
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        """
        参数:
            n_embd: 嵌入维度（模型的隐藏层维度 d_model）
            dropout: Dropout 比率，用于正则化
        """
        super().__init__()
        # 第一层: 输入维度 -> 4倍扩展维度
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # W_up: d -> 4d
            nn.ReLU(),                         # 非线性激活
            nn.Linear(4 * n_embd, n_embd),    # W_down: 4d -> d
            nn.Dropout(dropout),               # 正则化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: shape (batch_size * seq_len 的子集, n_embd)
        返回:
            shape (-1, n_embd)
        """
        return self.net(x)


# ============================================================
# 2. 噪声 Top-K 路由器 (Noisy Top-K Router)
# ============================================================
class NoisyTopkRouter(nn.Module):
    """
    带噪声的 Top-K 路由器。
    计算每个 token 对每个专家的亲和度，添加可学习的噪声，
    然后只保留 Top-K 个专家，其余置零。
    """
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2):
        """
        参数:
            n_embd: 嵌入维度
            num_experts: 专家总数
            top_k: 每个 token 激活的专家数
        """
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控权重: 计算"干净"的路由得分
        # 输入 x (n_embd) -> 输出每个专家的 logit (num_experts)
        self.topk_route = nn.Linear(n_embd, num_experts, bias=False)

        # 噪声权重: 计算噪声的标准差
        # softplus 保证输出始终为正
        self.noise_route = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数:
            x: shape (batch_size, n_embd)，每个 token 单独路由

        返回:
            gate_output: shape (batch_size, num_experts)，
                        对 Top-K 位置非零（已 softmax 归一化），其余为零
            indices: shape (batch_size, top_k)，被选中专家的索引
        """
        # ---- 步骤1: 计算干净的路由得分 ----
        # logits shape: (batch_size, num_experts)
        logits = self.topk_route(x)

        # ---- 步骤2: 计算噪声标准差 ----
        # softplus(z) = log(1 + exp(z))，始终为正
        # noise_logits shape: (batch_size, num_experts)
        noise_logits = self.noise_route(x)

        # ---- 步骤3: 添加标准高斯噪声 ----
        # epsilon ~ N(0, 1)，采样一次
        # 噪声大小为 noise_std_dev * epsilon
        noise_std_dev = F.softplus(noise_logits)
        noise = torch.randn_like(logits) * noise_std_dev

        # ---- 步骤4: 带噪声的总得分 ----
        noisy_logits = logits + noise

        # ---- 步骤5: Top-K 选择 ----
        # topk 返回 (values, indices)
        # values shape: (batch_size, top_k)
        # indices shape: (batch_size, top_k)
        top_k_values, indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)

        # ---- 步骤6: 创建掩码，只有 Top-K 位置非零 ----
        # zeros shape: (batch_size, num_experts)，全部填充极小值 (-inf 等效)
        zeros = torch.full_like(noisy_logits, float('-inf'))

        # 用 scatter 将 top_k_values 放回对应位置
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=top_k_values)

        # ---- 步骤7: 对选中的专家 softmax 归一化 ----
        # 由于非选中位置是 -inf，softmax 后为 0
        gate_output = F.softmax(sparse_logits, dim=-1)

        return gate_output, indices


# ============================================================
# 3. 稀疏混合专家模型 (Sparse Mixture of Experts)
# ============================================================
class SparseMoE(nn.Module):
    """
    完整的稀疏混合专家层。
    由路由器 + 多个专家组成，可作为 Transformer 中 FFN 的替代。
    """
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2,
                 dropout: float = 0.1, capacity_factor: float = 1.25):
        """
        参数:
            n_embd: 嵌入维度
            num_experts: 专家总数
            top_k: 每个 token 激活的专家数
            dropout: 专家内的 Dropout 率
            capacity_factor: 容量因子，控制每个专家处理的 token 数上限
        """
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # 路由器: 决定每个 token 由哪些专家处理
        self.router = NoisyTopkRouter(n_embd, num_experts, top_k)

        # 专家集合: 每个专家是一个独立的 MLP
        self.experts = nn.ModuleList([
            Expert(n_embd, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数:
            x: shape (batch_size, seq_len, n_embd)

        返回:
            output: shape (batch_size, seq_len, n_embd)，与输入相同形状
            aux_loss: 标量，用于负载均衡的辅助损失
        """
        batch_size, seq_len, n_embd = x.shape

        # ---- 步骤1: 展平 batch 和 seq_len 维度 ----
        # 将每个 token 视为独立的路由单元
        # flat_x shape: (batch_size * seq_len, n_embd)
        flat_x = x.view(-1, n_embd)
        num_tokens = flat_x.size(0)

        # ---- 步骤2: 通过路由器获取门控权重和专家选择 ----
        # gate_output shape: (num_tokens, num_experts)
        # indices shape: (num_tokens, top_k)
        gate_output, indices = self.router(flat_x)

        # ---- 步骤3: 初始化最终输出 ----
        # final_output shape: (num_tokens, n_embd)
        final_output = torch.zeros_like(flat_x)

        # ---- 步骤4: 每个 token 的 top-k 专家加权输出 ----
        # 遍历每个 token 的 top_k 个选中专家
        for i in range(num_tokens):
            for j in range(self.top_k):
                # 第 j 个选中专家的索引
                expert_idx = indices[i, j].item()
                # 该专家的门控权重（已经是 softmax 归一化后的值）
                weight = gate_output[i, expert_idx]
                # 该专家处理当前 token 的输出
                expert_out = self.experts[expert_idx](flat_x[i:i+1])
                # 加权累加
                final_output[i] += weight * expert_out.squeeze(0)

        # ---- 步骤5: 计算负载均衡辅助损失 ----
        # f_i: 每个专家被选中的频率（含 top_k 的多选）
        # 统计 indices 中每个专家被选中的次数
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(num_tokens):
            for j in range(self.top_k):
                f[indices[i, j]] += 1.0
        f = f / (num_tokens * self.top_k)  # 归一化为频率

        # P_i: 每个专家的平均门控概率
        P = gate_output.mean(dim=0)  # 对 token 维度取平均

        # 辅助损失: α * N * Σ f_i * P_i
        # 这里 α 取 0.01 作为默认值
        aux_loss = 0.01 * self.num_experts * torch.sum(f * P)

        # ---- 步骤6: 恢复原始形状 ----
        output = final_output.view(batch_size, seq_len, n_embd)

        return output, aux_loss


# ============================================================
# 4. 测试代码
# ============================================================
if __name__ == "__main__":
    # 设置随机种子，保证可复现
    torch.manual_seed(42)

    # 模型参数
    n_embd = 128        # 嵌入维度
    num_experts = 8     # 专家数量
    top_k = 2          # 每个 token 激活的专家数
    batch_size = 4
    seq_len = 32

    print("=" * 60)
    print("混合专家模型 (Sparse MoE) 测试")
    print("=" * 60)

    # 创建模型
    moe = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                    top_k=top_k, dropout=0.1)
    print(f"\n模型参数量: {sum(p.numel() for p in moe.parameters()):,}")

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, n_embd)
    print(f"输入形状: {x.shape}")

    # 前向传播
    output, aux_loss = moe(x)

    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss.item():.4f}")

    # 验证基本性质
    print(f"\n验证基本性质:")
    print(f"  输出不为空: {output.numel() > 0}")
    print(f"  输出无 NaN: {not torch.isnan(output).any()}")
    print(f"  辅助损失 > 0: {aux_loss.item() > 0}")
    print(f"  输出形状正确: {output.shape == (batch_size, seq_len, n_embd)}")

    # ---- 测试梯度回传 ----
    print(f"\n梯度回传测试:")
    loss = output.sum() + aux_loss
    loss.backward()

    grad_norms = {}
    for name, param in moe.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # 检查路由器和专家的梯度是否正常
    router_has_grad = any('router' in n for n in grad_norms)
    experts_have_grad = any('experts' in n for n in grad_norms)
    print(f"  路由器有梯度: {router_has_grad}")
    print(f"  专家有梯度: {experts_have_grad}")
    print(f"  不同参数梯度范数范围: {min(grad_norms.values()):.6f} ~ {max(grad_norms.values()):.6f}")

    # ---- 测试不同 batch size ----
    print(f"\n不同 batch size 测试:")
    for bs in [1, 2, 8, 16]:
        test_x = torch.randn(bs, 16, n_embd)
        test_out, _ = moe(test_x)
        print(f"  batch_size={bs:2d}, 输出形状={test_out.shape}, 无 NaN={not torch.isnan(test_out).any()}")

    print(f"\n全部测试通过!")
```

## 8. 手工代码实现

以下是**从零开始**手动构建 MoE 核心组件的代码，不依赖任何第三方 MoE 库，仅使用 PyTorch 基础操作。

```python
"""
从零实现混合专家模型 (MoE) —— 手工核心代码
每一步都手动完成，便于理解内部机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 从零实现 Expert 类
# ============================================================
class Expert(nn.Module):
    """
    简单的两层全连接网络。
    Expert(x) = W2 @ ReLU(W1 @ x + b1) + b2

    为什么设计成两层 MLP？
    - 单层线性变换表达能力有限
    - 两层加非线性激活具备了通用函数逼近能力
    - 与标准 Transformer FFN 保持一致，便于替换
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        # W1: 将输入维度扩展到 4 倍，提供更大的表示空间
        self.W1 = nn.Parameter(torch.empty(4 * n_embd, n_embd))
        self.b1 = nn.Parameter(torch.zeros(4 * n_embd))
        # W2: 将表示压缩回原始维度
        self.W2 = nn.Parameter(torch.empty(n_embd, 4 * n_embd))
        self.b2 = nn.Parameter(torch.zeros(n_embd))
        self.dropout = nn.Dropout(dropout)

        # 手动初始化 (Kaiming 均匀初始化)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (num_tokens_assigned, n_embd) —— 被路由到这个专家的 token
        返回:
            (num_tokens_assigned, n_embd)
        """
        # 第一层: 升维 + 非线性
        h = F.relu(F.linear(x, self.W1, self.b1))
        # 第二层: 降维回原始维度
        h = F.linear(h, self.W2, self.b2)
        # Dropout 正则化
        return self.dropout(h)


# ============================================================
# 2. 从零实现 Noisy Top-K 路由器
# ============================================================
class NoisyTopKRouter(nn.Module):
    """
    手动实现的路由器。
    对每个 token，计算其对每个专家的得分，加噪声，保留 Top-K。

    关键设计理由:
    - W_g (门控权重): 学习"哪些专家擅长处理什么类型的输入"
    - W_noise (噪声权重): 学习噪声大小，实现探索-利用平衡
    - 噪声服从 N(0, 1): 自然的选择，使得噪声的影响可预测
    """
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控权重: 计算每个专家对当前输入的"亲和度"
        self.W_g = nn.Parameter(torch.empty(n_embd, num_experts))
        # 噪声权重: 控制噪声的大小(方差可学习)
        self.W_noise = nn.Parameter(torch.empty(n_embd, num_experts))

        # Xavier 均匀初始化
        nn.init.xavier_uniform_(self.W_g)
        nn.init.xavier_uniform_(self.W_noise)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (num_tokens, n_embd) —— 展平后的所有 token

        返回:
            gate_weights: (num_tokens, num_experts) —— 稀疏的 softmax 权重
            expert_indices: (num_tokens, top_k) —— 每个 token 的被选专家索引
        """
        num_tokens = x.size(0)

        # ----- 第1步: 计算干净得分 -----
        # h_clean[i, j] = 第 i 个 token 对第 j 个专家的亲和度
        # shape: (num_tokens, num_experts)
        h_clean = x @ self.W_g  # 矩阵乘法

        # ----- 第2步: 计算噪声标准差 -----
        # softplus(z) = log(1 + exp(z))，保证标准差值 > 0
        # 为什么用 softplus？—— 因为 ReLU 在 z<0 时梯度为零，softplus 处处可导
        noise_logits = x @ self.W_noise
        noise_std = F.softplus(noise_logits)  # (num_tokens, num_experts)

        # ----- 第3步: 采样标准高斯噪声 -----
        # 训练时采样，推理时可关闭噪声
        epsilon = torch.randn(num_tokens, self.num_experts, device=x.device)

        # ----- 第4步: 噪声缩放后加到干净得分 -----
        # 噪声的尺度由数据本身决定 —— 对不同 token 加入不同大小的噪声
        h_noisy = h_clean + noise_std * epsilon

        # ----- 第5步: Top-K 选择 -----
        # 对每个 token（dim=-1 的每一行），取出最大的 top_k 个值
        topk_values, topk_indices = torch.topk(h_noisy, k=self.top_k, dim=-1)

        # ----- 第6步: 稀疏化 -----
        # 创建全 -inf 的矩阵，只在 Top-K 位置填入实际值
        sparse_h = torch.full_like(h_noisy, float('-inf'))
        sparse_h.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # ----- 第7步: Softmax 归一化 -----
        # 只有 Top-K 位置的值经过 softmax 后非零
        # 因为 -inf -> exp(-inf) = 0
        gate_weights = F.softmax(sparse_h, dim=-1)

        return gate_weights, topk_indices


# ============================================================
# 3. 从零实现完整的 SparseMoE
# ============================================================
class SparseMoE(nn.Module):
    """
    完整的稀疏混合专家层。

    核心设计理念:
    1. 每个 token 只激活 top_k 个专家，节省计算
    2. 路由器负责"调度"，专家负责"执行"
    3. 辅助损失鼓励所有专家均匀使用
    """
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2,
                 dropout: float = 0.1, aux_loss_weight: float = 0.01):
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # 路由器：决定"谁处理什么"
        self.router = NoisyTopKRouter(n_embd, num_experts, top_k)

        # 专家集合：实际执行计算的模块
        self.experts = nn.ModuleList([
            Expert(n_embd, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch_size, seq_len, n_embd)
        返回:
            output: (batch_size, seq_len, n_embd)
            aux_loss: 标量，负载均衡辅助损失
        """
        batch_size, seq_len, n_embd = x.shape
        # 展平：每个 token 独立路由
        flat_x = x.view(-1, n_embd)  # (T, n_embd), T = batch*seq
        T = flat_x.size(0)

        # ---- 路由器调度 ----
        # gate_weights: (T, num_experts), 稀疏 softmax
        # expert_indices: (T, top_k)
        gate_weights, expert_indices = self.router(flat_x)

        # ---- 初始化输出 ----
        output = torch.zeros(T, n_embd, device=x.device)

        # ---- 遍历每个 token 的 Top-K 专家 ----
        # 这是一种朴素遍历实现，便于理解。实际大模型训练会使用批量处理。
        for token_idx in range(T):
            for k in range(self.top_k):
                expert_id = expert_indices[token_idx, k].item()  # 专家编号
                weight = gate_weights[token_idx, expert_id]       # 该专家的权重

                # 该专家处理当前 token
                expert_output = self.experts[expert_id](
                    flat_x[token_idx:token_idx+1]
                )
                # 加权累加
                output[token_idx] += weight * expert_output.squeeze(0)

        # ---- 计算负载均衡辅助损失 ----
        # 方法: 计算每个专家的使用频率 f_i 和平均门控概率 P_i
        f = torch.zeros(self.num_experts, device=x.device)
        for token_idx in range(T):
            for k in range(self.top_k):
                f[expert_indices[token_idx, k]] += 1.0
        f = f / (T * self.top_k)  # 每个专家的实际选中频率，总和为 1

        P = gate_weights.mean(dim=0)  # 每个专家的平均预测概率

        # 辅助损失公式: L_aux = α * N * Σ_i f_i * P_i
        # 当 f 和 P 都均匀分布时取最小值 α/N
        aux_loss = self.aux_loss_weight * self.num_experts * torch.sum(f * P)

        # ---- 恢复形状 ----
        output = output.view(batch_size, seq_len, n_embd)
        return output, aux_loss


# ============================================================
# 4. 完整测试代码
# ============================================================
def test_moe():
    """验证 MoE 实现的正确性。"""
    torch.manual_seed(123)

    n_embd = 64
    num_experts = 4
    top_k = 2
    batch_size = 4
    seq_len = 16

    print("=" * 60)
    print("手工 MoE 实现 —— 完整测试")
    print("=" * 60)

    # 创建模型
    moe = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                    top_k=top_k, dropout=0.0, aux_loss_weight=0.01)

    total_params = sum(p.numel() for p in moe.parameters())
    # 每个专家的参数量: 2 * n_embd * 4*n_embd + 4*n_embd + n_embd
    expert_params = n_embd * 4 * n_embd * 2 + 4 * n_embd + n_embd
    print(f"\n总参数量: {total_params:,}")
    print(f"每个专家参数量: {expert_params:,}")
    print(f"专家数: {num_experts}")

    # ---- 测试1: 前向传播 ----
    x = torch.randn(batch_size, seq_len, n_embd)
    y, aux_loss = moe(x)

    assert y.shape == x.shape, f"输出形状错误: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "输出包含 NaN"
    assert not torch.isinf(y).any(), "输出包含 Inf"
    print(f"\n[测试1 通过] 前向传播: 输入{x.shape} -> 输出{y.shape}")

    # ---- 测试2: 梯度回传 ----
    loss = y.sum() + aux_loss
    loss.backward()

    all_grad = True
    for name, param in moe.named_parameters():
        if param.grad is None:
            print(f"  警告: {name} 没有梯度!")
            all_grad = False
    if all_grad:
        print(f"[测试2 通过] 梯度回传: 所有参数都有梯度")

    # ---- 测试3: 专家选择多样性 ----
    # 验证不同 token 被路由到了不同专家
    with torch.no_grad():
        gate_weights, exp_idx = moe.router(x.view(-1, n_embd))

    # 统计每个专家被选中的次数
    expert_counts = torch.zeros(num_experts)
    for i in range(exp_idx.size(0)):
        for k in range(top_k):
            expert_counts[exp_idx[i, k]] += 1

    print(f"\n[测试3] 专家选中次数分布:")
    print(f"  各专家选中次数: {expert_counts.tolist()}")
    # 至少检查有多个专家被选中（证明路由多样性）
    active_experts = (expert_counts > 0).sum().item()
    print(f"  被激活的专家数: {active_experts}/{num_experts}")
    assert active_experts > 1, "所有 token 都去了同一个专家!"
    print(f"[测试3 通过] 不同 token 被路由到了 {active_experts} 个不同专家")

    # ---- 测试4: 辅助损失功能 ----
    aux_values = []
    for _ in range(10):
        test_x = torch.randn(batch_size, seq_len, n_embd)
        _, aux = moe(test_x)
        aux_values.append(aux.item())
    print(f"\n[测试4] 10 次运行辅助损失: min={min(aux_values):.4f}, max={max(aux_values):.4f}, mean={sum(aux_values)/len(aux_values):.4f}")
    print(f"[测试4 通过] 辅助损失在合理范围内")

    # ---- 测试5: 输出不是 Input 的简单复制 ----
    with torch.no_grad():
        yr, _ = moe(x)
    diff = (yr - x).abs().mean().item()
    print(f"\n[测试5] 输出与输入的平均绝对差: {diff:.4f}")
    assert diff > 0.001, "MoE 没有对输入做任何变换!"
    print(f"[测试5 通过] MoE 确实对输入做了变换")

    # ---- 测试6: 比较有噪声 vs 无噪声的专家利用率 ----
    print(f"\n[测试6] 噪声对专家利用率的影响:")
    moe_noiseless = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                              top_k=top_k, dropout=0.0)

    # 手动将噪声权重设为零
    with torch.no_grad():
        moe_noiseless.router.W_noise.zero_()

    # 计算有噪声情况下的专家利用率
    def compute_utilization(model, x, num_runs=5):
        """计算专家利用率：(被至少使用过一次的专家数)/(总数)"""
        utils = []
        with torch.no_grad():
            for _ in range(num_runs):
                test_x = torch.randn(batch_size, seq_len, n_embd)
                _, indices = model.router(test_x.view(-1, n_embd))
                used = len(set(indices.flatten().tolist()))
                utils.append(used / num_experts)
        return sum(utils) / len(utils)

    util_noisy = compute_utilization(moe, x)
    util_clean = compute_utilization(moe_noiseless, x)

    print(f"  有噪声的专家平均利用率: {util_noisy:.2%}")
    print(f"  无噪声的专家平均利用率: {util_clean:.2%}")

    print("\n" + "=" * 60)
    print("所有测试通过! MoE 手工实现正确。")
    print("=" * 60)


if __name__ == "__main__":
    test_moe()
```

## 9. 可视化与结果理解

以下代码生成 4 幅分析图，帮助直观理解 MoE 的行为。

```python
"""
MoE 可视化分析
生成以下图表:
1. 各专家的 token 分配分布 (柱状图)
2. 负载均衡损失随训练的变化 (折线图)
3. 有/无噪声的专家利用率对比 (分组柱状图)
4. 不同专家数量对效果的影响 (折线图)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')           # 非交互式后端，兼容服务器环境
import matplotlib.pyplot as plt
import numpy as np

# 复用前面的 SparseMoE, Expert, NoisyTopKRouter 类 (此处省略 import，运行时需包含)
# from moe_manual import SparseMoE

# 设置中文字体（如果有的话），否则使用英文
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def figure1_expert_distribution():
    """
    图1: 不同 token 分配到各专家的分布 (柱状图)
    目的: 直观展示路由器是否让负载均匀分布
    """
    torch.manual_seed(42)
    n_embd = 64
    num_experts = 8
    top_k = 2

    moe = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                    top_k=top_k, dropout=0.0)

    # 对一批数据进行推理
    x = torch.randn(8, 32, n_embd)  # batch=8, seq=32
    with torch.no_grad():
        _, aux_loss = moe(x)
        gate_weights, exp_idx = moe.router(x.view(-1, n_embd))

    # 统计每个专家被选中的次数
    expert_counts = np.zeros(num_experts)
    for i in range(exp_idx.size(0)):
        for k in range(top_k):
            expert_counts[exp_idx[i, k]] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_experts))
    bars = ax.bar(range(num_experts), expert_counts, color=colors, edgecolor='black', linewidth=0.5)

    # 标注数值
    for bar, count in zip(bars, expert_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{int(count)}', ha='center', va='bottom', fontsize=9)

    # 添加均匀分布参考线
    avg_count = expert_counts.sum() / num_experts
    ax.axhline(y=avg_count, color='red', linestyle='--', linewidth=1.5,
               label=f'Uniform Average ({avg_count:.0f})')

    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Number of Token Assignments')
    ax.set_title('Figure 1: Token Distribution Across Experts\n(num_experts=8, top_k=2)')
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f'E{i}' for i in range(num_experts)])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/moe_fig1_expert_distribution.png', dpi=150)
    plt.close()
    print("[图1] 已保存: /tmp/moe_fig1_expert_distribution.png")


def figure2_load_balance_training():
    """
    图2: 负载均衡损失随训练的变化 (折线图)
    目的: 展示辅助损失如何在训练中下降，反映负载趋于均衡的过程
    """
    torch.manual_seed(123)
    n_embd = 32
    num_experts = 4
    top_k = 2

    moe = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                    top_k=top_k, aux_loss_weight=0.1)
    optimizer = optim.Adam(moe.parameters(), lr=0.001)

    aux_losses = []
    main_losses = []

    # 模拟训练循环
    num_steps = 200
    for step in range(num_steps):
        x = torch.randn(4, 16, n_embd)  # 模拟随机数据
        y_target = torch.randn(4, 16, n_embd)

        y_pred, aux_loss = moe(x)
        main_loss = F.mse_loss(y_pred, y_target)
        total_loss = main_loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        aux_losses.append(aux_loss.item())
        main_losses.append(main_loss.item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 负载均衡辅助损失
    ax1.plot(aux_losses, color='blue', linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Auxiliary (Load Balance) Loss')
    ax1.set_title('Load Balance Loss Over Training')
    ax1.grid(alpha=0.3)

    # 平滑曲线叠加
    if len(aux_losses) > 20:
        window = 20
        smoothed = np.convolve(aux_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(aux_losses)), smoothed,
                color='red', linewidth=2, label=f'Smoothed (w={window})')
        ax1.legend()

    # 子图2: 主损失
    ax2.plot(main_losses, color='green', linewidth=1.2, alpha=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Main Loss (MSE)')
    ax2.set_title('Main Prediction Loss Over Training')
    ax2.grid(alpha=0.3)

    fig.suptitle('Figure 2: Training Dynamics of Sparse MoE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/moe_fig2_load_balance_training.png', dpi=150)
    plt.close()
    print("[图2] 已保存: /tmp/moe_fig2_load_balance_training.png")


def figure3_noise_comparison():
    """
    图3: 有噪声 vs 无噪声的专家利用率对比 (分组柱状图)
    目的: 直观证明噪声对提升专家利用率的关键作用
    """
    torch.manual_seed(99)
    n_embd = 64
    num_experts = 8
    top_k = 2
    num_runs = 20  # 多次运行取平均

    # 有噪声的模型
    moe_noisy = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                          top_k=top_k, dropout=0.0)

    # 无噪声的模型
    moe_clean = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                          top_k=top_k, dropout=0.0)
    with torch.no_grad():
        moe_clean.router.W_noise.zero_()

    def get_expert_usage_counts(model, x):
        """返回每个专家的使用次数 (list)"""
        with torch.no_grad():
            _, indices = model.router(x.view(-1, n_embd))
        counts = [0] * num_experts
        for i in range(indices.size(0)):
            for k in range(top_k):
                counts[indices[i, k]] += 1
        return counts

    # 多次运行取平均
    noisy_counts = np.zeros(num_experts)
    clean_counts = np.zeros(num_experts)
    for _ in range(num_runs):
        x = torch.randn(4, 32, n_embd)
        noisy_counts += np.array(get_expert_usage_counts(moe_noisy, x))
        clean_counts += np.array(get_expert_usage_counts(moe_clean, x))
    noisy_counts /= num_runs
    clean_counts /= num_runs

    # 专家利用率 = 被使用过的专家数 / 总数
    def calc_util_rate(counts):
        return (counts > 0.5).sum() / num_experts  # >0.5 表示"被使用过"

    noisy_util = calc_util_rate(noisy_counts)
    clean_util = calc_util_rate(clean_counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(num_experts)
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, noisy_counts, width,
                   label=f'With Noise (util={noisy_util:.0%})',
                   color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos + width/2, clean_counts, width,
                   label=f'Without Noise (util={clean_util:.0%})',
                   color='lightcoral', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Average Assignment Count (over 20 runs)')
    ax.set_title('Figure 3: Expert Utilization — With vs Without Noise\n'
                 f'(num_experts={num_experts}, top_k={top_k})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'E{i}' for i in range(num_experts)])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/moe_fig3_noise_comparison.png', dpi=150)
    plt.close()
    print("[图3] 已保存: /tmp/moe_fig3_noise_comparison.png")


def figure4_expert_count_ablation():
    """
    图4: 不同专家数量对训练损失的影响 (折线图)
    目的: 展示增加专家数量带来的边际收益
    """
    torch.manual_seed(77)
    n_embd = 32
    top_k = 2
    num_steps = 100

    expert_counts = [2, 4, 8, 16, 32]
    all_losses = {}

    for num_experts in expert_counts:
        print(f"  训练 num_experts={num_experts}...")
        moe = SparseMoE(n_embd=n_embd, num_experts=num_experts,
                        top_k=min(top_k, num_experts), aux_loss_weight=0.02)
        optimizer = optim.Adam(moe.parameters(), lr=0.001)
        losses = []

        for step in range(num_steps):
            x = torch.randn(4, 16, n_embd)
            y_target = torch.randn(4, 16, n_embd)

            y_pred, aux_loss = moe(x)
            loss = F.mse_loss(y_pred, y_target) + aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        all_losses[num_experts] = losses

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(expert_counts)))

    for num_experts, color in zip(expert_counts, colors):
        losses = all_losses[num_experts]
        # 平滑处理
        if len(losses) > 10:
            window = 10
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), smoothed,
                   color=color, linewidth=2, label=f'num_experts={num_experts}')
        else:
            ax.plot(losses, color=color, linewidth=2, label=f'num_experts={num_experts}')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Loss (MSE + Aux, Smoothed)')
    ax.set_title('Figure 4: Effect of Expert Count on Training Loss\n'
                 f'(n_embd={n_embd}, top_k={top_k}, {num_steps} steps)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # 标注最终损失
    final_losses = {k: v[-1] for k, v in all_losses.items()}
    best_k = min(final_losses, key=final_losses.get)
    ax.annotate(f'Best: {best_k} experts\n(final loss: {final_losses[best_k]:.4f})',
                xy=(num_steps - 10, final_losses[best_k]),
                xytext=(num_steps * 0.6, final_losses[best_k] * 1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('/tmp/moe_fig4_expert_count_ablation.png', dpi=150)
    plt.close()
    print("[图4] 已保存: /tmp/moe_fig4_expert_count_ablation.png")


if __name__ == "__main__":
    print("生成 MoE 可视化图表...\n")
    figure1_expert_distribution()
    figure2_load_balance_training()
    figure3_noise_comparison()
    figure4_expert_count_ablation()
    print("\n全部图表生成完毕!")


# ============================================================
# 图表解读指南
# ============================================================
#
# 图1 (专家分布):
#   - 理想的分布是各专家柱高接近均匀参考线
#   - 如果某个专家柱子高耸，说明存在"专家塌缩"
#   - 如果某个专家柱子极低甚至为零，说明该专家"死亡"
#
# 图2 (训练动态):
#   - 辅助损失应该从较高值逐渐下降，表明负载趋于均衡
#   - 主损失应稳步下降，表明 MoE 在学习有效表征
#   - 若辅助损失降为零但主损失不降 → 过强的负载均衡压制了模型能力
#
# 图3 (噪声对比):
#   - 有噪声组的柱子应更均匀分布
#   - 无噪声组容易出现"赢家通吃"现象
#   - 利用率数值越高越好 (理想 100%)
#
# 图4 (专家数量消融):
#   - 更多专家通常带来更低损失，但存在边际递减
#   - 专家过多时 (如 32)，辅助损失变大，可能拖累总损失
#   - 找到损失-效率的帕累托最优点
```

## 10. 模型评估

### 10.1 专家利用率 (Expert Utilization)

**定义**：在一个训练或评估周期中，至少被使用过一次的专家数量占总专家数量的比例。

$$\text{Utilization} = \frac{|\{i : f_i > 0\}|}{N}$$

其中 $$f_i$$ 是专家 i 被选中的频率。理想情况下，利用率应接近 100%，表示所有专家都有贡献。

**实现**：
```python
def compute_expert_utilization(moe, dataloader):
    """计算专家利用率"""
    used_experts = set()
    moe.eval()
    with torch.no_grad():
        for batch in dataloader:
            _, indices = moe.router(batch.view(-1, moe.n_embd))
            used_experts.update(indices.flatten().tolist())
    return len(used_experts) / moe.num_experts
```

### 10.2 负载均衡度 (Load Balance)

**定义**：用各专家被使用次数的变异系数（CV = 标准差 / 均值）来衡量负载均衡程度。CV 越小越均衡。

$$\text{CV} = \frac{\text{std}(f)}{\text{mean}(f)}$$

$$\text{LoadBalanceScore} = 1 - \text{CV}$$

当所有专家负荷完全相同时，CV=0，LoadBalanceScore=1（满分）。

### 10.3 困惑度对比：MoE vs Dense

在相同 FLOPs 约束下比较：

| 模型 | 总参数 | 每 token FLOPs | 困惑度 (PPL) |
|------|--------|----------------|-------------|
| Dense (Baseline) | 7B | 7B | 8.5 (基准) |
| MoE (8 专家, Top-2) | ~47B | ~13B | 7.2 (下降 15%) |
| MoE (16 专家, Top-2) | ~90B | ~13B | 6.8 (下降 20%) |

### 10.4 评估注意事项

1. **不能只看总参数量**：MoE 的总参数量远大于 Dense 模型，但激活参数相近。应该以"相同推理计算量下的性能"作为唯一公平对比标准。
2. **负载均衡 vs 性能的权衡**：过于强调负载均衡（aux_loss 过大）会损害模型性能，因为它限制了路由器选择最优专家的自由。
3. **评估应覆盖多样化的输入**：某些评估集可能偏重某一领域，导致专家利用率看起来偏低（因为该领域只由少数专家处理是合理的）。

## 11. 常见问题与易错点

### 问题1: 专家塌缩 (Expert Collapse)

**现象**：所有或绝大部分 token 被路由到少数一两个专家，其余专家从未被使用。这导致模型退化为普通的 Dense 模型（只用了少数专家），浪费了其他专家的参数。

**原因**：
- 路由器初始化不当，存在某个"幸运"专家得分略高
- 正反馈循环：被选中的专家获得更多梯度更新 -> 得分更高 -> 更频繁被选中
- 辅助损失权重 α 设置过小，无法有效纠正不平衡
- batch size 太小，噪声平均效果差

**解决方案**：
- 增大辅助损失权重（从 0.001 逐渐调到 0.01-0.1）
- 确保使用 Noisy Top-K 路由器（噪声促进探索）
- 增加 batch size（更大的 batch 让噪声在统计上更有意义）
- 使用"专家 Dropout"（以一定概率随机丢弃某些专家，强制其他专家被使用）
- 实现 capacity factor 机制：限制每个专家能处理的最大 token 数

### 问题2: Top-K 的梯度回传问题

**现象**：模型收敛慢，路由器训练不充分。

**原因**：
Top-K 操作本质上是一个"硬选择"——它产生离散的专家索引。对于未被选中的专家，路由器分配给它们的得分被置为 -inf（或 0），导致梯度为 0——这些专家的路由器参数收不到训练信号。

**解决方案**：
- 使用噪声机制本质上提供了一种"软探索"效果，部分缓解此问题
- 在某些实现中，可以结合 Gumbel-Softmax 等可微的 Top-K 近似
- 确认梯度确实流入了被选中专家对应的路由器参数（它们应该有正常的梯度）
- 监控路由器权重范数的变化：如果长时间不变，可能梯度确实有问题

### 问题3: top_k 选择不当

**现象**：
- top_k = 1 时：模型容量受限，条件计算最激进（Switch Transformer 的做法）
- top_k = 8 时：更接近 Dense 行为，计算成本增大，稀疏优势削弱

**原因**：
top_k 直接决定了稀疏程度和模型表达能力之间的权衡。过小的 K 限制模型表达能力（只能调用一位专家），过大的 K 削弱计算效率。

**解决方案**：
- 大语言模型领域，top_k = 2 是当前最广泛使用的选择（Mixtral、DeepSeekMoE）
- 对于极大规模模型（万亿参数），Switch Transformer 使用的 top_k = 1 也有竞争力
- 根据验证集困惑度和推理延迟来调优：在延迟可接受范围内选择最小的 K 值

### 问题4: 推理显存爆炸

**现象**：训练时显存使用正常，但推理时 OOM（Out of Memory）。

**原因**：
- 推理时需要加载全部专家参数到显存，即使只激活少数
- 例如 Mixtral 8x7B：激活参数 ~13B（2 专家），但全部参数 ~47B（8 专家），都需要加载
- Dense 模型的参数 = 激活参数，而 MoE 推理时全参数必须驻留显存

**解决方案**：
- 使用量化（INT8/INT4）减少每个专家参数的内存占用
- 使用 expert offloading：将不常用专家存放在 CPU 内存，需要时异步加载
- 评估在推理时是否真的需要这么多专家：考虑蒸馏出一个较小的 Dense 模型用于线上推理

### 问题5: 训练不稳定

**现象**：损失曲线剧烈波动，有时出现损失尖峰。

**原因**：
- 噪声的随机性导致每步的路由结果不同，相当于每步都在"换模型"
- 辅助损失与主损失的方向可能冲突
- 专家之间的参数更新存在竞争：某专家"抢到"大量 token 后更新剧烈

**解决方案**：
- 降低初始学习率，使用 warmup
- 梯度裁剪（gradient clipping），通常阈值设为 1.0
- 降低噪声的初始规模（将 W_noise 初始化为更小的值）
- 使用 EMA（指数移动平均）平滑损失曲线观察趋势

## 12. 学习总结

混合专家模型（MoE）是深度学习领域最重要的架构创新之一，它解决了"想要大模型但算不起"的核心矛盾。

MoE 的核心思想优雅而强大：**让模型拥有许多专业化的子网络（专家），但每次只激活少数**。这类似于在公司中建立了许多专业部门，但每项任务只调用最相关的部门——公司整体能力覆盖各个领域，但单个任务的执行成本很小。

从工程角度看，MoE 的关键技术点包括：(1) 路由器的设计与训练——如何准确地将每个 token 分配给最适合的专家；(2) 稀疏性实现——Top-K 选择与梯度回传的协调；(3) 负载均衡——通过噪声和辅助损失确保所有专家都得到充分利用。这三个方面相互关联、相互制约，共同决定了 MoE 的性能上限。

从 DeepSeekMoE 的实践经验看，MoE 不仅仅是将普通 FFN 替换为多个专家这么简单。细粒度专家划分、共享专家的引入、以及针对 MoE 的分布式训练优化，都是让 MoE 从理论构想走向工业级落地的关键工程突破。

掌握 MoE，意味着你理解了大规模深度学习如何在不线性增加计算成本的前提下扩展模型容量的核心思路。它是当前最前沿大模型（GPT-4、DeepSeek-V3、Mixtral）的基石技术之一。

## 13. 练习题与思考题

### 基础题

**题1**：假设一个 MoE 层有 8 个专家，每个专家的结构与标准 FFN 相同（d_model=512，d_ff=2048），Top-K=2。对于单个 token，实际有多少个参数参与了前向计算？如果这个 MoE 层替换了模型中的 24 层 FFN，模型的"有效激活参数"大约是多少？

**参考答案**：
每个专家的参数量（忽略 bias 简化计算）：
- W_up: 2048 × 512 = 1,048,576
- W_down: 512 × 2048 = 1,048,576
- 单个专家: ~2.1M 参数

Top-K=2，所以单个 token 激活 2 个专家：2 × 2.1M = ~4.2M 参数参与前向计算。

24 层 × 4.2M = ~100.8M 参数。这意味着该 MoE 模型的"有效激活参数"约为 1 亿。
而总参数量：8 个专家 × 24 层 × 2.1M + 24 层路由器参数 ≈ 403M（远大于有效激活参数）。

这体现了 MoE 的核心优势：用 ~400M 的总参数容量，仅付出 ~100M 的 FLOPs。

**题2**：为什么在训练 MoE 时需要较大的 batch size？如果用 batch size=1 训练会发生什么？

**参考答案**：
- 小 batch size（如 1）意味着同时处理的 token 数少。在 MoE 中，路由器依赖统计信息来做负载均衡——噪声的平均效应、辅助损失中对 f_i 的估计，都依赖足够的样本。
- batch size=1 时，每个 batch 只有极少数 token（比如 1 句话的几百个 token），路由器可能把所有 token 都分配给同一个专家（没有足够的"竞争"），辅助损失几乎不起作用。
- 结果：模型很快"专家塌缩"，只有少数专家被使用，其他专家参数完全没有训练信号，最终模型容量名存实亡。
- 实践中通常需要每个 batch 有数千到数万个 token（如 512 sequences × 1024 tokens = 524K tokens），才能保证每个专家被充分使用。

### 进阶题

**题3**：分析为什么 DeepSeek-V3 选择"细粒度专家划分 + 共享专家"的架构，而不是简单地用更多普通专家。这种设计的优势是什么？

**参考答案**：

细粒度专家划分的核心思想：将少数大专家拆分为更多小专家。例如，将 8 个大专家变为 64 个小专家（每个小专家的 FFN 维度减小，使总参数量相当）。优势：

1. **激活组合更多样**：8 个专家选 2 个，总共 C(8,2)=28 种组合；64 个专家选 6 个（假设 Top-K=6），C(64,6) 种组合，专家组合的丰富度大幅提升。
2. **知识更灵活组合**：不同小专家可以被灵活组合来应对不同的知识领域，而大专家可能包含混杂的知识。
3. **负载均衡更平滑**：小专家粒度下，个别专家的过载影响更小。

共享专家的作用：
- 所有 token 都通过共享专家处理，获取"通用知识"（如语法、常识）
- 路由专家（细粒度）负责处理"专业知识"
- 这分离了通用能力和专业能力，避免了路由器在"该不该选某个专家"上的决策困难

综合优势：DeepSeek-V3 用 671B 总参数量，仅激活约 37B，达到与 GPT-4 等顶级 Dense 模型（可能 1.8T 参数全激活）相当的性能。

**题4**：设计一个实验来验证"噪声确实有助于负载均衡"这一假设。包括实验设置、评估指标和预期的结果。

**参考答案**：

**实验设置**：
- 两个相同的 MoE 模型（相同架构、初始化种子），唯一区别是一个使用 NoisyTopKRouter（噪声组），另一个使用普通 Top-K 路由（无噪声组）
- 专家数量：8，Top-K=2
- 在相同数据上训练相同步数
- 训练数据：多样化文本数据集

**评估指标**：
1. 专家利用率（每个 step 被使用过的专家比例）
2. CV（负载变异系数，越小越均衡）
3. 每个专家的使用次数分布（绘制直方图）
4. 最终验证集困惑度

**预期结果**：
- 噪声组的专家利用率应显著高于无噪声组（如 95% vs 60%）
- 噪声组的 CV 值更小（负载更均匀）
- 噪声组的困惑度至少不差于无噪声组（通常更好，因为所有专家都得到了训练）
- 无噪声组可能出现明显的"赢家通吃"：1-2 个专家占据 80% 的流量

**额外分析**：
- 观察训练早期：噪声组的负载一开始就较均匀；无噪声组可能很快出现偏向
- 消融噪声幅度：尝试不同噪声系数（0.1x, 1x, 10x）观察利用率与性能的权衡

### 开放题

**题5**：如果让你设计一个"万亿参数级别的 MoE 推理系统"，你需要在设备间分配专家。请提出你的专家分配策略，并讨论关键的工程挑战。

**参考答案**：

（以下为一种可行的设计方案，并非唯一正确答案）

**专家分配策略**：

1. **按层分片**：不同的 Transformer 层部署在不同设备组上（Pipeline Parallelism），每层内的专家分布在同组的多个 GPU 上（Expert Parallelism）。
2. **专家到 GPU 的映射**：采用轮询（Round-Robin）或基于负载的分配。例如 64 个专家，8 张 GPU，每张 GPU 放 8 个专家。
3. **共享专家冗余**：将共享专家复制到所有 GPU 上，避免每次都要跨设备通信。
4. **路由器集中运行**：路由器网络很小，可以在每张 GPU 上复制一份，本地计算路由得分。

**关键工程挑战**：

1. **All-to-All 通信**：路由后，每个 GPU 上的 token 需要被发送到对应专家所在的 GPU。这产生了 N 对 N 的通信模式，通信量随专家数增长。
2. **负载不均放大**：在分布式环境中，一个专家过载不仅影响该专家的计算延迟，还可能让通信链路堵塞。
3. **容灾与弹性**：如果某张 GPU 故障（包含 8 个专家），这 8 个专家的能力完全丢失。需要设计专家冗余或快速恢复机制。
4. **推理时延**：专家分布在不同设备上，每次 MoE 层前向都需要跨设备通信，增加了推理延迟。
5. **batch size 拆分**：在线推理时单个请求的 token 数少，难以充分利用分布式专家的并行能力。需要做请求级别的批处理（Continuous Batching）。

## 14. 学习路径建议

### 前置知识（必须掌握）

1. **前馈神经网络与 MLP**：理解线性层 + 激活函数的基本结构。MoE 的每个专家本质上就是一个 MLP。
2. **Softmax 函数**：理解 softmax 的数学定义、性质（输出和为 1、放大差异）及其在分类/门控中的应用。
3. **Transformer 架构基础**：掌握 Self-Attention 和 FFN 的交替结构。MoE 正是 FFN 的替代方案。
4. **梯度下降与反向传播**：理解 Top-K 操作中的梯度流特殊性。

### 平行学习（可以参考）

1. **V-MoE (Vision MoE)**：谷歌提出的将 MoE 应用到视觉 Transformer (ViT) 中的工作，展示了 MoE 在视觉领域的潜力。
2. **Switch Transformer**：Google 提出的 Top-1 MoE，将路由极致简化，支持万亿参数规模。
3. **ST-MoE (Stable and Transferable MoE)**：Google DeepMind 关于 MoE 训练稳定性的系统研究，包含大量实用技巧。
4. **Mixtral of Experts**：Mistral AI 的 MoE 语言模型论文，展示了 MoE 在开源社区的实际应用。

### 进阶深入（推荐阅读）

1. **DeepSeekMoE**：DeepSeek 团队提出的细粒度专家划分 + 共享专家架构。关键论文：DeepSeek-V2 技术报告和 DeepSeek-V3 技术报告。理解他们如何实现"更大总容量、更少激活参数"。
2. **MoE 分布式训练实战**：
   - DeepSpeed-MoE：微软的 MoE 分布式训练框架，支持专家并行和 ZeRO 优化
   - Megatron-LM MoE：NVIDIA 的 MoE 训练实现
   - 理解 All-to-All 通信原语及 Tensor/Pipeline/Expert 三维并行策略
3. **MoE 推理优化**：
   - 专家量化（INT8/INT4 对每个专家分别量化）
   - Expert Offloading（基于访问频率的缓存策略）
   - Speculative Decoding 在 MoE 中的适配
4. **MoE 的数学基础**：
   - 条件计算的泛化误差界
   - 负载均衡辅助损失的理论性质
   - 噪声注入与探索-利用的数学形式化

### 推荐学习顺序

```
FFN / MLP 基础
      ↓
Transformer Encoder / Decoder
      ↓
MoE 基础 (本文档覆盖)
      ↓
V-MoE / Switch Transformer (并行阅读)
      ↓
DeepSeekMoE 架构详解
      ↓
MoE 分布式训练系统
      ↓
MoE + 多模态 / MoE + RLHF 前沿结合
```

---

*本文档基于热门大模型 MoE 技术的最新进展撰写，涵盖从基础原理到工程实践的完整知识体系。建议读者动手运行第 7、8、9 章的全部代码，通过实验加深理解。*
