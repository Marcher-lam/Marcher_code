# 面试题：字节 RankMixer 模型介绍

# 面试题：字节 RankMixer 模型介绍

五分钟了解字节推荐大模型 RankMixer，大幅提升业务效果，且推理成本不变~

ByteDance 提出的 RankMixer 是一个面向工业级推荐系统的排序模型架构，它通过一系列创新设计，成功将模型参数量提升至十亿级别，同时保证了推理效率。

论文：RankMixer: Scaling Up Ranking Models in Industrial Recommenders

![](images/b51d54d1ce4a52bc4dcfb1d17f5eaaf8955672c95823b85fb7879942da1b7c0c.jpg)

# 1. 特征令牌化（Feature Tokenization）

RankMixer 首先将传统的特征输入转换为类似于 Transformer 的令牌（Token）序列，以解决推荐系统中特征异构、维度不一的问题。

 输入特征分组：基于业务先验知识，将数百个特征（用户画像、视频属性、行为序列等）按语义划分为若干组，每组特征拼接成一个长向量：

$e _ { \mathrm { i n p u t } } = [ e _ { 1 } ; e _ { 2 } ; \ldots ; e _ { N } ]$ ，其中 $e _ { i }$ 代表第 $j$ 个特征组的嵌入表示。

 维度对齐与切片：将拼接后的超长向量通过线性投影或等距切分为 T 个固定维度 D 的 Token：

$$
x _ {i} = \operatorname {P r o j} \left(e _ {\text {i n p u t}} [ d \cdot (i - 1): d \cdot i ]\right), \quad i = 1, \dots , T
$$

其中，每个 token 代表一个语义一致的特征子空间，便于后续并行处理。

# 2. Token 混合模块（Token Mixing）

![](images/d96bae2b77cecc87c1666bcbe649ca18793ca4c599e9de6b55c9120cf4c4507d.jpg)

该模块替代了 Transformer 中的自注意力机制，实现无参数的特征交互，显著提升计算效率。

 多头拆分与重组：将每个令牌的 $D$ 维向量拆分为 $H$ 个头（head），每个头维度为 D/H。随后，将不同令牌在相同头位置上的子向量拼接，形成新的混合 Token：

$$
\operatorname {T o k e n M i x} (X) = \operatorname {C o n c a t} _ {\text {h e a d} = 1} ^ {H} \left(\operatorname {C o n c a t} _ {t = 1} ^ {T} \left(x _ {t} ^ {\text {h e a d}}\right)\right)
$$

这一操作类似张量的重排，实现跨特征的信息交换。最后输出是一个[H, T*D/H]的 tensor。

 残差连接与归一化：将混合后的结果与原始 Token 相加，并通过 LayerNorm 稳定训练：

$$
X _ {\text {o u t}} = \operatorname {L a y e r N o r m} (X + \operatorname {T o k e n M i x} (X))
$$

与自注意力相比，Token Mixing 避免了计算二次复杂度的注意力矩阵，更适合异构特征空间。

# 3. Per-Token 前馈网络（Per-Token FFN）

![](images/dd7f68fa0bbf29268e8955d45cb21a2ea2b623a929e075d738c4274d80c77a7e.jpg)

为每个 Token 分配独立的前馈网络（FFN），增强模型容量并避免高频特征主导。

 独立参数设计 ：每个令牌 $x _ { i }$ 经过其专属的 FFN 进行非线性变换：

$$
y _ {i} = \sigma \left(W _ {i} ^ {(2)} \cdot \sigma \left(W _ {i} ^ {(1)} x _ {i} + b _ {i} ^ {(1)}\right) + b _ {i} ^ {(2)}\right)
$$

其中 $\sigma$ 是激活函数（如 Gelu）， Wk） $W _ { i } ^ { ( k ) }$ 和 b(𝑘） $b _ { i } ^ { ( k ) }$ 是第 i个Token的私有参数。

 扩展为稀疏 MoE：为进一步提升参数规模，将 FFN 替换为稀疏混合专家（Sparse MoE）结构。通过门控机制动态选

择专家：

$$
y _ {i} = \sum_ {j = 1} ^ {E} G \left(x _ {i}\right) _ {j} \cdot \operatorname {E x p e r t} _ {j} \left(x _ {i}\right)
$$

其中门控权重 $G ( x _ { i } )$ 通过 ReLU 路由实现稀疏激活，训练时采用密集路由（Dense Training），推理时转为稀疏（SparseInference）以提升效率。

# 4. 整体架构与输出

RankMixer 由多个上述模块堆叠而成（L 层），最终输出通过mean-pooling 聚合所有令牌，并输入到多目标预测层（如完播率、快滑率、点赞率等）。

核心创新总结：  

<table><tr><td>模块</td><td>传统方法问题</td><td>RankMixer 解决方案</td></tr><tr><td>特征输入</td><td>特征异构、维度不一，处理碎片化</td><td>语义分组+Token 化，统一维度并行处理</td></tr><tr><td>特征交互（Token Mixing）</td><td>自注意力计算复杂度高，不适于异构特征</td><td>无参数 Token 混合，高效实现跨特征信息交换</td></tr><tr><td>非线性变换（FFN）</td><td>共享参数导致高频特征主导，长尾信号丢失</td><td>每 Token 独立 FFN/MoE，提升容量与泛化能力</td></tr></table>

# 效果：

 模型效率：参数量从 16M 扩展到 1B（70倍），但通过优化 GPU 利用率（MFU 从 $4 . 5 \%$ 提升至 $45 \%$ ），推理延迟保持稳定（14ms）。  
 业务指标：在抖音推荐场景中，用户日均活跃天数提升 $0 . 3 \%$ ，使用时长增长 $1 . 0 8 \%$ ；广告场景 AUC 提升 $0 . 7 3 \%$ ，广告主价值 advv $+ 3 . 9 \%$ 。

---

# 五、RankMixer 与其他排序模型的对比

## 1. 与传统推荐模型对比

| 维度 | DCN-v2 | DeepFM | DIN/DIEN | RankMixer |
|------|--------|--------|----------|-----------|
| 参数规模 | 百万级 | 百万级 | 百万级 | 十亿级 |
| 特征交互方式 | 显式交叉网络 | FM 隐式交叉 | 注意力序列 | 无参数 Token 混合 |
| 扩展性 | 受限于交叉网络层数 | 受限于 FM 阶数 | 受限于序列长度 | 可通过 MoE 线性扩展 |
| GPU 利用率 | 中 | 中 | 低 | 高（MFU 45%） |

## 2. 与 Transformer 排序模型对比

| 维度 | Transformer 排序 | RankMixer |
|------|-----------------|-----------|
| 特征交互 | 自注意力 $O(T^2)$ | 无参数混合 $O(T)$ |
| 推理延迟 | 随序列长度二次增长 | 恒定 |
| 参数效率 | 共享 FFN | Per-Token 独立 FFN |
| 长尾特征 | 容易被高频特征压制 | 独立 FFN 保护长尾信号 |
| 工程部署 | 需优化注意力计算 | 直接部署，无特殊优化需求 |

## 3. 关键设计哲学

RankMixer 的核心思想是**"用参数量换精度，用架构设计换效率"**：

1. **Token Mixing 替代 Self-Attention**：避免了 $O(T^2)$ 的注意力计算，将复杂度降至 $O(T)$，且无需学习注意力参数
2. **Per-Token FFN 替代 Shared FFN**：增加了模型容量（参数量与 Token 数成正比），同时保护了低频特征的表达能力
3. **Sparse MoE 进一步扩展**：在保持推理成本不变的前提下，将参数量扩展到十亿级

---

# 六、Token Mixing 的数学直觉

## 1. 为什么无参数混合有效？

Token Mixing 本质上是一种**通道混洗（Channel Shuffle）**操作。它将不同 Token 在相同头位置的子向量拼接在一起，相当于在特征维度上做了信息交换。

类比理解：
- **Self-Attention**：通过注意力权重动态决定信息交换的强度（$O(T^2)$）
- **Token Mixing**：通过固定的维度重排实现信息交换（$O(T)$），效率更高但灵活性较低

在推荐场景中，特征之间的交互模式相对稳定（不像 NLP 中的语义关系那么复杂），因此无参数混合已经足够捕捉主要的交叉信号。

## 2. 残差连接的必要性

没有残差连接时，多层 Token Mixing 会退化为简单的线性变换（因为混合操作本身是线性的）。残差连接确保了每层的输出保留原始输入信息，使模型可以通过堆叠多层来逐步精炼特征表示。

---

# 七、代码实现（简化版 RankMixer）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenMixing(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x = x.reshape(B, T, self.num_heads, self.d_head)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_heads, T * self.d_head)
        x = x.reshape(B, self.num_heads, T, self.d_head).permute(0, 2, 1, 3)
        x = x.reshape(B, T, D)
        return self.norm(x + residual)

class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens, d_model, d_ff, use_moe=False, num_experts=4, top_k=2):
        super().__init__()
        self.use_moe = use_moe
        self.num_tokens = num_tokens
        if use_moe:
            self.gate = nn.ModuleList([nn.Linear(d_model, num_experts) for _ in range(num_tokens)])
            self.experts = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
                    for _ in range(num_experts)
                ])
                for _ in range(num_tokens)
            ])
            self.top_k = top_k
        else:
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                )
                for _ in range(num_tokens)
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            if self.use_moe:
                gate_logits = self.gate[t](x_t)
                topk_vals, topk_idx = torch.topk(F.relu(gate_logits), self.top_k)
                gate_weights = F.softmax(topk_vals, dim=-1)
                out_t = torch.zeros(B, D, device=x.device)
                for k in range(self.top_k):
                    expert_idx = topk_idx[:, k]
                    for e in range(len(self.experts[t])):
                        mask = (expert_idx == e)
                        if mask.any():
                            expert_out = self.experts[t][e](x_t[mask])
                            out_t[mask] += gate_weights[mask, k].unsqueeze(-1) * expert_out
                outputs.append(out_t)
            else:
                outputs.append(self.ffn[t](x_t))
        x = torch.stack(outputs, dim=1)
        return self.norm(x + residual)

class RankMixerBlock(nn.Module):
    def __init__(self, num_tokens, d_model, d_ff, num_heads, use_moe=False):
        super().__init__()
        self.token_mix = TokenMixing(num_tokens, d_model, num_heads)
        self.ffn = PerTokenFFN(num_tokens, d_model, d_ff, use_moe)

    def forward(self, x):
        x = self.token_mix(x)
        x = self.ffn(x)
        return x

class RankMixer(nn.Module):
    def __init__(self, feature_groups, d_model=64, d_ff=128, num_heads=4,
                 num_blocks=3, num_tasks=3, use_moe=False):
        super().__init__()
        self.token_projections = nn.ModuleList([
            nn.Linear(group_dim, d_model)
            for group_dim in feature_groups
        ])
        self.num_tokens = len(feature_groups)
        self.blocks = nn.ModuleList([
            RankMixerBlock(self.num_tokens, d_model, d_ff, num_heads, use_moe)
            for _ in range(num_blocks)
        ])
        self.output_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
            for _ in range(num_tasks)
        ])

    def forward(self, feature_groups_list):
        tokens = []
        for i, (proj, fg) in enumerate(zip(self.token_projections, feature_groups_list)):
            tokens.append(proj(fg))
        x = torch.stack(tokens, dim=1)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        outputs = [head(x) for head in self.output_layers]
        return outputs

feature_groups = [32, 64, 16, 48]
model = RankMixer(feature_groups, d_model=64, d_ff=128, num_heads=4, num_blocks=2)
inputs = [torch.randn(8, dim) for dim in feature_groups]
outputs = model(inputs)
for i, out in enumerate(outputs):
    print(f"任务 {i+1} 预测形状: {out.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

moe_model = RankMixer(feature_groups, d_model=64, d_ff=128, num_heads=4, num_blocks=2, use_moe=True)
moe_params = sum(p.numel() for p in moe_model.parameters())
print(f"MoE版本参数量: {moe_params:,}")
```

---

# 八、工程部署与优化

## 1. 推理优化策略

- **TensorRT 部署**：将 Per-Token FFN 融合为批量矩阵运算，减少 kernel launch 开销
- **模型量化**：支持 INT8/FP16 量化，推理速度提升 2-3 倍，精度损失 < 0.1%
- **特征预计算**：将 Embedding 查表放在特征工程阶段，减少在线计算开销

## 2. 训练优化策略

- **数据并行 + 模型并行**：Per-Token FFN 的参数独立，天然适合模型并行
- **梯度累积**：大 batch 训练（batch size > 10000）时使用梯度累积
- **混合精度训练**：BF16 训练 + FP32 梯度更新，减少显存占用

## 3. 常见问题

1. **Token 数量选择**：Token 数量（特征组数）建议 8-32 个。过少信息损失大，过多 Token Mixing 效率下降
2. **MoE 专家数量**：建议 4-16 个专家，Top-K 取 2。专家数量过多会导致门控网络训练不稳定
3. **特征分组策略**：分组应基于业务语义（如用户特征一组、物品特征一组、上下文特征一组），而非随机分组
