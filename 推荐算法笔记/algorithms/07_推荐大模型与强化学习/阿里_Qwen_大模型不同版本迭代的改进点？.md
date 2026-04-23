# 面试题：阿里 Qwen 大模型不同版本迭代的改进点？

面试题：阿里 Qwen 大模型不同版本迭代的改进点？

# 一、 Qwen 不同版本迭代详解

# 1. Qwen1.5（2024 年初）

# 基础架构：

- 纯 Decoder 结构，采用 Rotary Positional Embeddings（RoPE）增强位置感知
- 首次引入分组查询注意力（GQA），仅在 32B/110B 模型应用，平衡 MHA 质量与 MQA 效率
- MoE 版（14B-A2.7B）采用共享专家+专属专家混合路由

**GQA 的核心思想：** GQA 是 Multi-Head Attention（MHA）和 Multi-Query Attention（MQA）的折中方案。MHA 中每个注意力头都有独立的 Key/Value，MQA 中所有头共享同一组 Key/Value，而 GQA 将注意力头分组，组内共享 Key/Value。数学表达为：

$$
\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_G)W_O
$$

其中每组 $g$ 内的 head 共享 $K_g$ 和 $V_g$。这使得 KV 缓存减少为原来的 $1/G$，在推理时显著降低显存占用和延迟。

**RoPE 的位置编码原理：** RoPE 通过将位置信息融入 Query 和 Key 的点积计算中，使得注意力分数自然包含相对位置信息：

$$
q_m \cdot k_n^T = (R_{\Theta,m} W_Q x_m) \cdot (R_{\Theta,n} W_K x_n)^T
$$

其中 $R_{\Theta,m}$ 是旋转矩阵，使得 $q_m \cdot k_n^T$ 仅依赖相对位置 $m-n$，从而支持长度外推。

# 改进局限：

- MoE 层的专家负载不均衡，部分专家利用率低
- 上下文窗口仅 32K，长文本处理弱于竞品（如 GPT-4-128K）

# 2. Qwen2.5（2024 年 9 月）

# 架构升级：

- 全系列 GQA 覆盖：从 0.5B 到 72B 均应用 GQA，KV 缓存减少 $40\%$，推理吞吐提升 $30\%$
- 上下文扩展至 128K：通过 Dual Chunk Attention（DCA）分块处理长序列，捕获块间依赖
- MoE 路由优化：细粒度专家分割，引入任务感知的门控网络

**Dual Chunk Attention（DCA）原理：** DCA 将长序列分成多个 chunk，在 chunk 内计算完整的自注意力，同时通过特殊的 chunk 间注意力机制捕获跨 chunk 的依赖关系。其核心创新在于将绝对位置编码转换为 chunk 内相对位置编码，从而支持超长序列的高效注意力计算。

# 训练策略革新：

# 三阶段预训练：

- S1：通用语料奠基（4K 上下文）
- S2：注入数学/代码数据，通过课程学习逐步提升难度
- S3：动态 NTK 扩展至 32K→128K，缓解长序列训练不稳定

**动态 NTK 扩展：** 这是一种无需微调即可扩展 RoPE 上下文长度的技术。通过动态调整 RoPE 的基频 $\theta$：

$$
\theta' = \theta \cdot \left(\frac{s}{L}\right)^{d/(d-2)}
$$

其中 $s$ 为目标序列长度，$L$ 为训练时的序列长度。这使得模型在推理时能够处理比训练时更长的序列。

# 两阶段 RLHF：

- 离线RL：基于 DPO 优化数学/代码等确定性任务
- 在线RL：实时奖励模型对齐人类偏好（如无害性、简洁性）

**DPO（Direct Preference Optimization）原理：** DPO 绕过了奖励模型的训练，直接通过偏好数据优化策略模型。其损失函数为：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

其中 $y_w$ 和 $y_l$ 分别为偏好数据中的优选和劣选回复。

# 性能表现：

- 72B 模型在 MMLU、GSM8K、HumanEval 全面超越 Llama3-70B

# 3. Qwen3（2025 年 4 月）

# 架构突破：

- QK-Norm 替代QKV-bias：归一化Query-Key矩阵，缓解注意力头标准差问题，提升训练稳定性
- MoE 专家独立化：取消共享专家，引入 Global-Batch Load Balancing Loss，均衡专家负载
- 动态思维模式：单模型支持思考模式（深度推理）与非思考模式（高效响应）动态切换

**QK-Norm 的数学原理：** 在标准 Transformer 中，注意力分数为 $QK^T/\sqrt{d_k}$。当 $Q$ 或 $K$ 的范数增长时，注意力分数的方差会增大，导致训练不稳定。QK-Norm 通过对 $Q$ 和 $K$ 分别进行 Layer Norm 或 RMS Norm 来约束其范数：

$$
\hat{Q} = \text{RMSNorm}(W_Q x), \quad \hat{K} = \text{RMSNorm}(W_K x)
$$

这样可以防止注意力分数爆炸，显著提升训练稳定性。

**Global-Batch Load Balancing Loss：** 传统 MoE 的负载均衡损失在单个 batch 内计算，容易导致局部均衡但全局不均衡的问题。Global-Batch 方法在全局 batch 级别统计专家负载分布，确保专家在整个训练数据上均匀激活：

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 为专家 $i$ 在全局 batch 中被选中的频率，$P_i$ 为专家 $i$ 的平均路由概率。

# 训练规模跃迁：

- 预训练数据达 36T tokens（Qwen2.5 的 2 倍），覆盖 119 种语言
- 推出超大规模 MoE 模型：
  - Qwen3-235B-A22B：总参 235B，激活 22B
  - Qwen3-Coder-480B-A35B：专精代码，总参 480B

# 关键能力提升：

- 数学推理能力提升 $30\%$，代码生成准确率提高 $25\%$
- 支持 1M Token 上下文（通过 YaRN 扩展）

**YaRN 扩展原理：** YaRN（Yet another RoPE extensioN）结合了 NTK 扩展和注意力温度调整。通过插值因子 $\alpha$ 将 RoPE 的频率缩放：

$$
\theta_i' = \begin{cases} \theta_i & \text{if } i < d_{\text{low}} \\ \frac{\alpha \cdot s}{L} \cdot \theta_i & \text{if } i \geq d_{\text{low}} \end{cases}$$

同时对注意力分数乘以温度因子 $\sqrt{t}$ 来补偿扩展后的注意力分布偏移。

# 4. Qwen3-2507（2025 年 7 月）

# 架构分化：

- 双模型独立部署（非动态切换）：
  - Thinking 版（Qwen3-235B-A22B-Thinking-2507）：深度逻辑链推理，适用数学/科学/伪科学辨析
  - Non-thinking 版（Qwen3-235B-A22B-Instruct-2507）：FP8 量化，响应速度优先，适用信息提取/格式化生成

- 长文本再升级：支持 256K 上下文，超越 Claude 3（200K）

# 垂直模型发布：

- Qwen3-Coder：针对代码任务优化，GitHub 任务解决率超 DeepSeek-V3
- Qwen-MT：低参数量高精度机器翻译模型

# 对齐能力强化：

- Arena-Hard 评测超越 Claude Opus4，人类偏好对齐提升显著

# 二、核心技术对比

| 技术点 | Qwen1.5 | Qwen2.5 | Qwen3 | Qwen3-2507 |
|--------|---------|---------|-------|-----------|
| 注意力机制 | GQA 部分应用 | 全系列 GQA | QK-Norm 稳定训练 | 继承 QK-Norm |
| 上下文长度 | 32K | 128K (DCA) | 1M (YaRN) | 256K |
| MoE 架构 | 共享专家 | 细粒度专家 | 独立专家+均衡损失 | 独立专家+均衡损失 |
| 思维模式 | 无 | 无 | 动态切换 | 双模型分立 |
| 训练数据量 | 18T tokens | 18T tokens | 36T tokens | 未公开(增量) |
| RLHF 方式 | 标准 PPO | DPO+在线RL | 改进DPO | 改进DPO |

# 三、面试常见追问

1. **Qwen 和 LLaMA 的核心架构差异？** Qwen 从 2.5 开始全系列使用 GQA，LLaMA 从 LLaMA2 开始在部分模型引入 GQA；Qwen3 引入了 QK-Norm，LLaMA 系列未采用；两者的 RoPE 实现细节也有差异。

2. **为什么 Qwen3 取消了共享专家？** 共享专家虽然能提供通用的知识基础，但会限制专家的特化程度。取消共享专家后，通过更强的负载均衡损失（Global-Batch Load Balancing Loss）来保证训练稳定性，同时让每个专家都能更加特化，提升了 MoE 模型的整体效果。

3. **MoE 模型的推理效率如何？** MoE 模型虽然总参数量大，但推理时只激活部分专家（如 Qwen3-235B 只激活 22B），实际计算量远小于同等参数量的稠密模型。但 MoE 模型需要将所有专家参数加载到显存中，对显存容量要求较高。

4. **Qwen3 的动态思维模式如何实现？** 通过在训练时混合"有推理过程"和"无推理过程"的数据，让模型学会根据输入自动判断是否需要深度推理。在推理时通过特殊的 system prompt 或 token 控制切换。
