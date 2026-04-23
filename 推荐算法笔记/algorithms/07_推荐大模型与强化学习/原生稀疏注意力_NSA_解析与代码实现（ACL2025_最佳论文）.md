# 面试题：原生稀疏注意力 NSA 解析与代码实现（ACL2025 最佳论文）

面试题：原生稀疏注意力 NSA 解析与代码实现（ACL2025 最佳论文）

以下是关于 ACL2025 最佳论文《Native Sparse Attention（NSA）》技术解析。

 文章题目：Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention   
 原文链接：https://arxiv.org/pdf/2502.11089  
 源码链接：https://github.com/fla-org/native-sparse-attention

# 一、背景：长文本处理的算力瓶颈

传统 Transformer 的注意力机制计算复杂度为 $\scriptstyle \alpha ( n \pmb { \wedge } 2 )$ （n 为序列长度），在处理长文本时面临严重效率问题：

 64K Token 序列中，注意力计算占总延迟的 $70 \% { \sim } 8 0 \%$ ；  
 现有稀疏注意力方案（如局部窗口、KV 缓存淘汰）存在局限：

 硬件不友好：理论计算量减少 ≠ 实际加速（内存访问成瓶颈）；  
 训练不可行：仅优化推理阶段，预训练仍需全注意力计算。

 行业需求：深度推理、库级代码生成、医疗长文本分析等场景需处理百万 Token 上下文 。

**标准注意力的计算瓶颈分析**：

标准自注意力的计算过程为 $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，其时间和空间复杂度均为 $O(n^2 d)$。当序列长度 $n = 128K$ 时，仅注意力矩阵就需要 $128K \times 128K \times 2 \text{ bytes} \approx 32\text{GB}$ 的显存，远超单卡容量。

现有加速方案的局限性：
- **Flash Attention**：通过分块计算减少显存占用，但计算量仍为 $O(n^2)$，仅解决空间瓶颈
- **滑动窗口注意力**：限制注意力范围至局部窗口，但丢失全局信息
- **KV Cache 淘汰**（如 H2O, Scissorhands）：推理阶段丢弃不重要的KV对，但仅适用于推理，且淘汰策略可能损失关键信息
- **Sparse Transformer**：手动设计稀疏模式，缺乏自适应性

# 二、NSA 的核心创新

NSA 通过动态分层稀疏策略 $^ +$ 硬件协同优化，实现性能与效率的双重突破：

1. 三重注意力分支协同   

<table><tr><td>分支</td><td>功能</td><td>计算复杂度</td><td>类比说明</td></tr><tr><td>压缩注意力</td><td>块级聚合（粗粒度全局语义）</td><td>O(n□m)</td><td>略读章节标题</td></tr><tr><td>选择注意力</td><td>动态筛选关键块（细粒度信息）</td><td>O(n□k)</td><td>精读核心段落</td></tr><tr><td>滑动窗口注意力</td><td>局部上下文保留</td><td>O(n□w)</td><td>细读当前句子</td></tr></table>

# 2. 硬件对齐的极速内核

 连续块内存访问：将随机索引转为 DMA 批量传输，内存带宽利用率提升 3.2 倍；  
 GQA 组共享加载：同组 Query 共享 KV 块，解码阶段内存占用降低 $64 \%$ ；  
 Triton 定制内核：通过网格调度器优化 GPU 计算流，算术强度逼近理论最优值。

**硬件对齐的设计哲学**：

GPU的计算瓶颈通常不是算力（FLOPS），而是内存带宽（Memory Bandwidth）。标准注意力中，随机访问KV对导致大量不连续内存访问（Cache Miss），实际吞吐远低于理论峰值。NSA 的硬件对齐策略包括：

1. **块状内存布局**：将KV存储为连续的块（Block），每次加载一个完整的块到SRAM，减少全局内存访问次数
2. **GQA感知的KV共享**：在Grouped Query Attention中，同一组内的多个Query头共享KV块，避免重复加载
3. **Kernel Fusion**：将压缩、选择、滑窗三个分支的计算融合到一个Triton Kernel中，减少中间结果的显存写入

# 3. 端到端可训练架构

 原生稀疏预训练：从预训练开始应用稀疏注意力，避免推理时剪枝导致的性能崩塌（传统方法保留 Top $20 \%$ 注意力仅恢复 $70 \%$ 性能）；  
 可微分稀疏操作：压缩（MLP）、选择（Top-K）、门控（Softmax）全程可导，支持梯度回传。

**为什么"原生稀疏预训练"很重要**：

传统方法在全注意力上训练，推理时才做稀疏化（Post-hoc Sparsification）。这种做法的问题在于：
1. 模型在训练时"学会"了依赖所有Token的注意力，推理时突然去掉部分注意力会导致性能骤降
2. 模型无法学会"在稀疏注意力下如何更好地分配注意力权重"
3. 实验表明，保留Top 20%注意力仅能恢复约70%的全注意力性能

NSA 从预训练开始就用稀疏注意力，模型从头开始学习如何在稀疏模式下高效分配注意力，因此能在大幅减少计算量的同时保持甚至超越全注意力性能。

# 三、 NSA 的算法原理详解

![](images/454d87254eb214607ef8e8eca7cc0ccbfe3f07d6afa90cfff29682469ba853a9.jpg)

![](images/6106e30ec854f4a3dda4fbe35eb4199955c9941a9fcf3988b6b09bd5eb2b2e60.jpg)

# 1. 压缩注意力（Compressed Attention）

 输入序列分块：将序列划分为长度为 的块，步长 $d$ （例 $_ { I = 3 2 , d = 1 6 }$ ）；  
 块级语义压缩： $\begin{array} { r } { K _ { \mathrm { c m p } } ^ { j } = \phi \Big ( \{ k _ { i } \} _ { i = ( j - 1 ) \cdot d } ^ { ( j - 1 ) \cdot d + l } \Big ) } \end{array}$ ，ϕ 为可学习 MLP，将块内所有 Key 压缩为 1 个向量。 $\phi$

**压缩注意力的详细计算流程**：

1. 将Key序列 $K = [k_1, k_2, \ldots, k_n]$ 划分为 $\lceil n/l \rceil$ 个块
2. 对每个块 $j$，使用可学习MLP $\phi$ 将块内所有Key压缩为单个向量：

$$K_{cmp}^j = \phi(\text{MeanPool}(\{k_i\}_{i=(j-1)\cdot d}^{(j-1)\cdot d + l}))$$

3. 计算Query与压缩Key的注意力分数：

$$A_{cmp} = \text{Softmax}\left(\frac{Q \cdot K_{cmp}^T}{\sqrt{d_k}}\right)$$

4. 用压缩Value计算输出：$O_{cmp} = A_{cmp} \cdot V_{cmp}$

压缩注意力的复杂度从 $O(n^2)$ 降低到 $O(n \cdot n/l)$，实现了粗粒度的全局信息捕获。

# 2. 选择注意力（Selected Attention）

 块重要性评分：基于压缩注意力分数 $\boldsymbol { p } _ { t } ^ {\mathrm {c m p}}$ 选择 Top-N 块：

$$
p _ {t} ^ {\mathrm {c m p}} = \operatorname {S o f t m a x} \left(\frac {Q _ {t} \cdot K _ {\mathrm {c m p}}}{\sqrt {d _ {k}}}\right)
$$

$$
I _ {\text {t o p}} = \operatorname {T o p K - I n d i c e s} \left(p _ {t} ^ {\mathrm {c m p}}, N\right)
$$

 细粒度 token 保留：从选中块中提取原始 Key/Value。

**选择注意力的设计思路**：

压缩注意力提供了粗粒度的全局视图，但信息损失不可避免。选择注意力通过"先粗筛、再精读"的策略弥补这一缺陷：
1. 利用压缩注意力的分数作为块级重要性指标
2. 选出Top-N个最重要的块（N通常为总块数的5%-20%）
3. 对选中块内的所有原始Token进行完整的注意力计算

这种两阶段策略既保证了全局信息的捕获，又实现了对关键信息的精细处理。

**Top-K选择的可微分实现**：

标准的Top-K操作不可导，NSA使用松弛的近似方法：

$$\text{SoftTopK}(s, N) = \text{Softmax}(s / \tau) \cdot \text{Mask}_N(s)$$

其中 $\tau$ 是温度参数，$\text{Mask}_N$ 保留分数最高的N个位置。在训练中使用松弛版本保证梯度流通，推理时使用硬Top-K。

# 3. 滑动窗口注意力（Sliding Attention）

 固定局部窗口 ：保留当前 Token 前后各 w/2 的上下文（w=512）。

**滑动窗口的必要性**：

局部信息是语言理解的基础——相邻词之间的语法关系、短语结构都需要精确建模。滑动窗口注意力保证了：
1. 局部语法关系的精确捕获
2. 当前Token周围的上下文完整性
3. 作为压缩注意力和选择注意力的补充，确保近处信息不丢失

# 4. 门控聚合机制

三个分支的输出通过可学习的门控机制聚合：

$$O = g_{cmp} \odot O_{cmp} + g_{sel} \odot O_{sel} + g_{win} \odot O_{win}$$

其中 $g_{cmp}, g_{sel}, g_{win}$ 是通过sigmoid激活函数生成的门控权重，模型可以自动学习在不同位置对不同类型信息的依赖程度。

# 四、实际效果：性能与效率双突破

 性能无损：通用任务超越全注意力，长文本任务显著领先；  
 效率革命：11.6 倍解码加速，使百万 Token 上下文成为可能；

![](images/d8015e5fb2a01ca941f26822d4a5543225057a7c13def64c3ca5ee9833043526.jpg)

![](images/1bb634a0d29fc67b94b2f881f7f2c436b164bbfdf26be48d7ca2becd3a239551.jpg)

<table><tr><td>Model</td><td>MMLU Acc. 5-shot</td><td>MMLU-PRO Acc. 5-shot</td><td>CMMLU Acc. 5-shot</td><td>BBH Acc. 3-shot</td><td>GSM8K Acc. 8-shot</td><td>MATH Acc. 4-shot</td><td>DROP F1 1-shot</td><td>MBPP Pass@1 3-shot</td><td>HumanEval Pass@1 0-shot</td><td>Avg.</td></tr><tr><td>Full Attn</td><td>0.567</td><td>0.279</td><td>0.576</td><td>0.497</td><td>0.486</td><td>0.263</td><td>0.503</td><td>0.482</td><td>0.335</td><td>0.443</td></tr><tr><td>NSA</td><td>0.565</td><td>0.286</td><td>0.587</td><td>0.521</td><td>0.520</td><td>0.264</td><td>0.545</td><td>0.466</td><td>0.348</td><td>0.456</td></tr></table>

# 五、与相关方法的对比

| 方法 | 稀疏策略 | 训练支持 | 硬件优化 | 长文本性能 |
|------|---------|---------|---------|-----------|
| Sparse Transformer | 手动模式 | 部分 | 无 | 中等 |
| Longformer | 局部+全局 | 是 | 有限 | 中等 |
| Flash Attention | 无稀疏 | 是 | 是 | 受限于计算量 |
| Mamba (SSM) | 状态压缩 | 是 | 是 | 良好 |
| NSA | 动态分层 | 原生支持 | 深度优化 | 优秀 |

# 六、Python 代码实现（简化版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressedAttention(nn.Module):
    def __init__(self, d_model, block_size=32, stride=16):
        super().__init__()
        self.block_size = block_size
        self.stride = stride
        self.compress_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(self, Q, K, V):
        n = K.size(1)
        num_blocks = (n - self.block_size) // self.stride + 1
        K_compressed = []
        V_compressed = []
        for i in range(num_blocks):
            start = i * self.stride
            end = min(start + self.block_size, n)
            block_k = K[:, start:end, :]
            block_v = V[:, start:end, :]
            K_compressed.append(self.compress_mlp(block_k.mean(dim=1, keepdim=True)))
            V_compressed.append(self.compress_mlp(block_v.mean(dim=1, keepdim=True)))
        K_cmp = torch.cat(K_compressed, dim=1)
        V_cmp = torch.cat(V_compressed, dim=1)
        attn = torch.matmul(Q, K_cmp.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, V_cmp)

class SelectedAttention(nn.Module):
    def __init__(self, d_model, num_selected_blocks=8, block_size=32, stride=16):
        super().__init__()
        self.num_selected = num_selected_blocks
        self.block_size = block_size
        self.stride = stride
    
    def forward(self, Q, K, V, block_scores):
        n = K.size(1)
        num_blocks = (n - self.block_size) // self.stride + 1
        top_indices = block_scores.topk(
            min(self.num_selected, num_blocks), dim=-1
        ).indices
        selected_k = []
        selected_v = []
        for i in range(top_indices.size(-1)):
            idx = top_indices[0, 0, i].item()
            start = idx * self.stride
            end = min(start + self.block_size, n)
            selected_k.append(K[:, start:end, :])
            selected_v.append(V[:, start:end, :])
        if selected_k:
            K_sel = torch.cat(selected_k, dim=1)
            V_sel = torch.cat(selected_v, dim=1)
            attn = torch.matmul(Q, K_sel.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            attn = F.softmax(attn, dim=-1)
            return torch.matmul(attn, V_sel)
        return torch.zeros_like(Q)

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, window_size=512):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
    
    def forward(self, Q, K, V):
        n = Q.size(1)
        half_w = self.window_size // 2
        outputs = []
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            q_i = Q[:, i:i+1, :]
            k_local = K[:, start:end, :]
            v_local = V[:, start:end, :]
            attn = torch.matmul(q_i, k_local.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = F.softmax(attn, dim=-1)
            outputs.append(torch.matmul(attn, v_local))
        return torch.cat(outputs, dim=1)

class NSABlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size=32, stride=16, 
                 num_selected=8, window_size=512):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.comp_attn = CompressedAttention(self.d_head, block_size, stride)
        self.sel_attn = SelectedAttention(self.d_head, num_selected, block_size, stride)
        self.win_attn = SlidingWindowAttention(self.d_head, window_size)
        self.gate = nn.Linear(d_model * 3, 3)
    
    def forward(self, x):
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        o_cmp = self.comp_attn(Q, K, V)
        comp_scores = torch.matmul(Q, self.comp_attn.compress_mlp(
            K.mean(dim=2, keepdim=True)
        ).transpose(-2, -1)) / (self.d_head ** 0.5)
        o_sel = self.sel_attn(Q, K, V, comp_scores)
        o_win = self.win_attn(Q, K, V)
        o_cmp = o_cmp.transpose(1, 2).contiguous().view(B, N, D)
        o_sel = o_sel.transpose(1, 2).contiguous().view(B, N, D)
        o_win = o_win.transpose(1, 2).contiguous().view(B, N, D)
        gate_input = torch.cat([o_cmp, o_sel, o_win], dim=-1)
        g = F.softmax(self.gate(gate_input), dim=-1)
        output = (g[:, :, 0:1] * o_cmp + 
                  g[:, :, 1:2] * o_sel + 
                  g[:, :, 2:3] * o_win)
        return self.out_proj(output)

if __name__ == "__main__":
    batch_size = 2
    seq_len = 256
    d_model = 64
    nsa = NSABlock(d_model, n_heads=4, block_size=16, stride=8,
                   num_selected=4, window_size=64)
    x = torch.randn(batch_size, seq_len, d_model)
    out = nsa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"NSA block test passed!")
```

# 七、应用前景与局限

**应用场景**：
- 长文档问答与摘要（法律、医疗、金融报告）
- 代码仓库级别的代码生成与理解
- 多轮长对话系统
- 科学文献综合分析

**当前局限**：
- 块大小和步长等超参数需要根据任务调整
- 压缩MLP可能丢失细粒度信息
- 短序列场景下优势不明显，反而增加额外开销
- 依赖Triton等GPU编程框架，硬件适配门槛较高
