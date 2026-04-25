# 面试题：快手生成式推荐 OneRec V2 技术原理介绍

# 面试题：快手生成式推荐 OneRec V2 技术原理介绍

标题：OneRec-V2 Technical Report

链接：https://arxiv.org/pdf/2508.20900

# 1 提出背景：解决 OneRec V1的扩展性与效率瓶颈

OneRec-V1 作为快手端到端生成式推荐系统的初步尝试，采用了 Encoder-Decoder 架构，虽然相比传统级联推荐系统有显著改进，但在实际工业部署中仍面临两个核心挑战：

 计算资源分配严重低效：在 Encoder-Decoder 架构中，高达 97.66%的计算资源被用于处理非常长的用户行为序列（编码阶段），而非直接用于生成目标推荐项（解码阶段）。  
 强化学习方法的固有局限：V1 依赖一个额外的奖励模型来提供强化学习信号，这带来了两方面问题：

 采样效率有限（Sampling Efficiency），因为计算奖励需要额外开销，只能对部分用户样本进行近似；  
 奖励黑客问题（Reward Hacking），即模型可能学会利用奖励函数的设计缺陷来获得高分，而非真正学习到符合用户偏好的行为。

# 2 核心创新点

![](images/1cc637ca81f05d0b11b8511047c487d4ca78cb0b087a642d9c1d04f9e1b0253b.jpg)

# 2.1 Lazy Decoder-Only 架构

V2 彻底移除了独立的 Encoder 部分，将其重构为一个 Lazy Decoder-Only 架构。其核心思想在于将用户的历史行为序列视为静态的上下文条件（Context），直接输入给 Decoder，而无需先经过一个庞大的 Encoder 进行编码。

 样本组织：采用 New Impression Only 方式。按曝光组织样本，只在 Target Item 上进行 next token prediction，避免了信息泄漏，并支持流式更新。  
 Context Processor（上下文处理器）：

 这是一个轻量化的模块，负责将异构的用户特征（静态特征、短期行为、长期行为）处理成统一的表示。  
 它使用分组共享策略（Group-Sharing）和分组查询注意力（GQA）来极大减少 Key-Value（KV）缓存的数量和计算量。

 Lazy Decoder Block：

 其"Lazy"（惰性）体现在对上下文 Key-Value 对的极致复用上。  
 传统的Cross-Attention中，K和V需要每层通过线性变换从上下文序列中投影得到。Lazy Decoder 移除了这些投影层，直接使用 Context Processor产生的统一 KV对，供所有 Decoder层共享。这意味着上下文只需计算一次，后续所有层和所有生成步骤都复用这一结果，避免了重复计算。

![](images/1f80ddac593d93a2c1140e905bf75a1877c9d3c33ba9e4bd979186ee1599b3f2.jpg)

# 2.2 基于真实用户交互的偏好对齐

V2 摒弃了依赖奖励模型代理信号的做法，转向直接利用真实世界的用户反馈信号来进行偏好对齐。

 时长感知奖励塑造（Duration-Aware Reward Shaping）：

 直接使用播放时长作为奖励信号存在偏差（长视频天然时长更长）。  
 V2的解决方案是按视频时长分桶，对于一个视频，只有其播放时长在其所属的时长分桶中排名前 $25\%$ ，才被认定为正样本。这样能更好剥离时长偏差，反映内容质量。

 自适应比率裁剪与 GBPO 算法 ：

 V2 提出了梯度有界策略优化（Gradient-Bounded Policy Optimization, GBPO）算法。  
 GBPO 不再使用粗暴的梯度裁剪（Clip），而是引入二元交叉熵（BCE）损失的梯度来动态约束和稳定 RL 训练的梯度，特别是在处理负样本（低奖励样本）时，能有效防止梯度爆炸和训练不稳定。

# 3 OneRec V2 与 V1 的对比  

<table><tr><td>对比维度</td><td>OneRec-V1</td><td>OneRec-V2</td><td>改进点总结</td></tr><tr><td>模型架构</td><td>Encoder-Decoder</td><td>Lazy Decoder-Only</td><td>移除 Encoder，计算集中于 Target Decoding</td></tr><tr><td>Scaling能力</td><td>受编码器瓶颈限制，难以扩展</td><td>支持扩展至 8B 参数（MoE 版 4B/0.5B 激活）</td><td>参数规模大幅提升，更遵循 Scaling Law</td></tr><tr><td>Cross-Attention</td><td>标准 Cross-Attention（每层计算 KV）</td><td>Lazy Cross-Attention（共享静态 KV）</td><td>移除 KV 投影层，复用 KV，内存和计算开销大幅降低</td></tr><tr><td>RL信号来源</td><td>依赖奖励模型（Reward Model）</td><td>直接使用真实用户反馈（如播放时长）</td><td>避免 Reward Hacking，信号更直接、稳定。</td></tr><tr><td>Reward设计</td><td>代理奖励（Proxy Reward）</td><td>时长感知奖励塑造（分位数归一化）</td><td>消除视频时长偏差，奖励更准确反映内容质量。</td></tr><tr><td>RL算法</td><td>ECPO（早期梯度裁剪）</td><td>GBPO（梯度有界，全样本利用）</td><td>训练更稳定，不丢弃负样本，鼓励多样化探索。</td></tr><tr><td>线上效果(主站)</td><td>停留时长 +0.269%</td><td>停留时长+0.467%，LT7+0.069%</td><td>核心用户指标提升显著。</td></tr></table>

# 4 Lazy Decoder 架构代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContextProcessor(nn.Module):
    def __init__(self, static_dim, short_dim, long_dim, hidden_dim, n_groups=4):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.short_proj = nn.Linear(short_dim, hidden_dim)
        self.long_proj = nn.Linear(long_dim, hidden_dim)
        self.n_groups = n_groups
        self.group_k_proj = nn.Linear(hidden_dim, hidden_dim // n_groups)
        self.group_v_proj = nn.Linear(hidden_dim, hidden_dim // n_groups)

    def forward(self, static_feat, short_seq, long_seq):
        static_emb = self.static_proj(static_feat).unsqueeze(1)
        short_emb = self.short_proj(short_seq)
        long_emb = self.long_proj(long_seq)
        context = torch.cat([static_emb, short_emb, long_emb], dim=1)
        k = self.group_k_proj(context)
        v = self.group_v_proj(context)
        return context, k, v

class LazyCrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, cached_k, cached_v, causal_mask=None):
        batch_size = query.shape[0]
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = cached_k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v = cached_v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        return self.out_proj(out)

class LazyDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads=8, ffn_dim=None):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.cross_attn = LazyCrossAttention(hidden_dim, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, cached_k, cached_v, self_mask=None):
        residual = x
        x = self.ln1(x)
        x2, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        x = residual + x2

        residual = x
        x = self.ln2(x)
        x2 = self.cross_attn(x, cached_k, cached_v)
        x = residual + x2

        residual = x
        x = self.ln3(x)
        x = residual + self.ffn(x)
        return x

class OneRecV2(nn.Module):
    def __init__(self, vocab_size, static_dim, short_dim, long_dim,
                 hidden_dim=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.context_proc = ContextProcessor(static_dim, short_dim, long_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            LazyDecoderBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_ids, static_feat, short_seq, long_seq):
        _, cached_k, cached_v = self.context_proc(static_feat, short_seq, long_seq)
        x = self.token_emb(target_ids)
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for block in self.decoder_blocks:
            x = block(x, cached_k, cached_v, self_mask=~causal_mask)
        logits = self.output_head(x)
        return logits
```

# 5 GBPO 算法详解

GBPO（梯度有界策略优化）是 OneRec V2 的核心 RL 算法，其核心思想是用 BCE 梯度作为 RL 梯度的动态上界：

```python
class GBPOLoss(nn.Module):
    def __init__(self, clip_ratio=0.2, entropy_coef=0.01):
        super().__init__()
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

    def forward(self, log_probs, old_log_probs, advantages, rewards):
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        bce_loss = F.binary_cross_entropy_with_logits(advantages, (rewards > 0).float())
        grad_bound = bce_loss.detach()

        total_loss = policy_loss + self.entropy_coef * (-log_probs.mean())
        return total_loss, grad_bound
```

GBPO 与 PPO/ECPO 的对比：

| 算法 | 梯度裁剪方式 | 负样本利用 | 训练稳定性 | 样本效率 |
|------|------------|----------|----------|---------|
| PPO | 固定 clip_ratio | 正常利用 | 中 | 中 |
| ECPO (V1) | 早期梯度裁剪 | 部分丢弃 | 中高 | 低 |
| GBPO (V2) | BCE梯度动态约束 | 全量利用 | 高 | 高 |

# 5.5 核心数学公式推导

## 5.5.1 GBPO 损失函数推导

GBPO 的核心思想是用 BCE 梯度作为 RL 策略梯度的动态上界。设策略为 $\pi_\theta$，奖励为 $r$，优势函数为 $A$。

**PPO 的裁剪目标**：

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中重要性采样比率：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

**BCE 梯度上界**：GBPO 引入 BCE 损失 $L_{\text{BCE}}$ 的梯度范数作为策略梯度的动态约束：

$$\left\|\nabla_\theta \mathcal{L}_{\text{RL}}\right\| \leq \left\|\nabla_\theta \mathcal{L}_{\text{BCE}}\right\|$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} \left[y_i \log \sigma(A_i) + (1 - y_i) \log(1 - \sigma(A_i))\right]$$

其中 $y_i = \mathbb{1}[r_i > 0]$，$\sigma$ 为 sigmoid 函数。

**GBPO 总损失**：

$$\mathcal{L}_{\text{GBPO}} = \mathcal{L}_{\text{PPO}} + \beta \cdot \mathcal{H}(\pi_\theta)$$

其中 $\mathcal{H}(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ 为策略熵，$\beta$ 为熵正则系数。

## 5.5.2 时长感知奖励公式

为消除视频时长偏差，V2 采用分桶分位数奖励：

$$r_{\text{duration}} = \mathbb{1}\left[\text{play\_time}(v) \geq Q_{0.75}(\text{play\_time} \mid \text{bucket}(v))\right]$$

其中 $Q_{0.75}$ 表示该时长分桶内播放时长的 75% 分位数。分桶函数为：

$$\text{bucket}(v) = \left\lfloor \frac{\text{duration}(v)}{\Delta t} \right\rfloor$$

$\Delta t$ 为分桶宽度。归一化后的奖励为：

$$\tilde{r}_v = \frac{r_{\text{duration}}(v) - \mu_{\text{bucket}}}{\sigma_{\text{bucket}} + \epsilon}$$

## 5.5.3 Lazy Cross-Attention 计算公式

Lazy Decoder 的核心是复用静态 KV 缓存。Context Processor 一次性计算：

$$K_{\text{ctx}} = \text{GQA}_K(\text{Concat}[e_{\text{static}}, e_{\text{short}}, e_{\text{long}}])$$

$$V_{\text{ctx}} = \text{GQA}_V(\text{Concat}[e_{\text{static}}, e_{\text{short}}, e_{\text{long}}])$$

其中 GQA 为分组查询注意力，将 $h$ 个注意力头分成 $g$ 组共享 KV：

$$\text{GQA}(Q, K, V) = \text{Concat}\left[\text{head}_1, \ldots, \text{head}_h\right] W^O$$

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_{\lceil i/g \rceil}^T}{\sqrt{d_k}}\right) V_{\lceil i/g \rceil}$$

Lazy Cross-Attention 在第 $l$ 层直接复用 $K_{\text{ctx}}, V_{\text{ctx}}$：

$$\text{LazyAttn}^l(X^l) = \text{softmax}\left(\frac{X^l W_Q^l (K_{\text{ctx}})^T}{\sqrt{d_k}}\right) V_{\text{ctx}}$$

## 5.5.4 序列生成概率

OneRec V2 以自回归方式生成推荐项，目标项 $y = (t_1, t_2, \ldots, t_T)$ 的生成概率：

$$P_\theta(y \mid \text{ctx}) = \prod_{t=1}^{T} P_\theta(t_t \mid t_{<t}, \text{ctx})$$

其中上下文 ctx 为用户的静态特征、短期行为和长期行为的拼接。训练目标为负对数似然：

$$\mathcal{L}_{\text{NLL}} = -\sum_{t=1}^{T} \log P_\theta(t_t \mid t_{<t}, \text{ctx})$$

**RL + NLL 联合训练**：

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{NLL}} + (1-\alpha) \cdot \mathcal{L}_{\text{GBPO}}$$

## 5.5.5 计算复杂度分析

| 模块 | 计算复杂度 | 说明 |
|------|-----------|------|
| Context Processor | $O(L_{\text{ctx}} \cdot d^2 / g)$ | GQA 将 KV 计算量降至 $1/g$ |
| Lazy Cross-Attention | $O(T \cdot d^2)$ per layer | $T$ 为生成序列长度，$K_{\text{ctx}}$ 复用 |
| V1 Encoder-Decoder | $O(L_{\text{ctx}}^2 \cdot d)$ | Encoder 需完整自注意力 |
| V2 总体 | $O(T \cdot d^2 \cdot L_{\text{layers}})$ | 计算集中于 Target Decoding |

# 6 部署架构与优化

```
用户请求
  ↓
特征服务（拉取静态特征 + 短期行为 + 长期行为）
  ↓
Context Processor（一次性计算 KV 缓存）
  ↓
Lazy Decoder（多步自回归生成推荐项）
  ↓          ↑
  └── 复用静态 KV 缓存 ──┘
  ↓
后处理（去重、多样性、业务过滤）
  ↓
推荐结果展示
```

关键部署优化：
1. **KV 缓存复用**：同一用户的上下文 KV 只计算一次，多次生成复用
2. **MoE 稀疏激活**：4B 参数模型仅激活 0.5B，降低推理成本
3. **流式训练**：New Impression Only 支持实时流式样本更新
4. **分桶奖励归一化**：按时长分桶消除偏差，使奖励更公平

# 7 常见问题与易错点

1. **Lazy KV 的更新策略**：用户行为更新后需重新计算 KV 缓存，需设计增量更新机制
2. **生成多样性**：自回归生成容易重复，需配合 temperature、top-k、nucleus sampling
3. **RL 训练的稳定性**：GBPO 虽然比 PPO 更稳定，但仍需注意 reward scale 和 advantage 归一化
4. **序列长度与延迟**：生成 token 数越多延迟越高，需权衡推荐精度和实时性

# 8 学习路径建议

1. 理解传统级联推荐系统的架构
2. 学习 Encoder-Decoder 和 Decoder-Only 架构差异
3. 掌握 Cross-Attention 和 KV 缓存优化
4. 研究 RLHF/RL 在推荐系统中的应用
5. 探索生成式推荐的大规模部署实践
