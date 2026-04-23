# 面试题：多模态 Embedding 特征融合方法介绍

# 面试题：多模态 Embedding 特征融合方法介绍

多模态 Embedding 特征融合介绍以下方法：

# 一、基础代数融合

通过简单运算实现特征交互，计算高效但表达能力有限。

# 1、向量拼接（Concatenation）

原理：将不同模态的 Embedding 向量首尾连接，形成高维联合特征。

公式：$z = [v_{text}; v_{image}]$，后续接入全连接层进行分类或预测。

局限：忽略模态间交互，特征维度爆炸。

**拼接融合的维度问题：** 假设文本 Embedding 维度为 $d_t$，图像 Embedding 维度为 $d_i$，则拼接后的维度为 $d_t + d_i$。当模态数量增多时（如文本+图像+音频+视频），拼接后的特征维度急剧增长，导致后续全连接层的参数量爆炸。例如，4 个模态各 512 维，拼接后为 2048 维，接入一个 1024 维的全连接层就需要 $2048 \times 1024 \approx 200$ 万个参数。

# 2、加权平均（Weighted Sum）

原理：对多模态 Embedding 加权求和，权重可学习或固定。

$$
\mathbf{z} = \sum_{i} w_i \cdot \mathbf{v}_i, \quad \sum w_i = 1
$$

适合各模态重要性明确的场景（如广告文本权重大于背景图）。

**加权平均的局限：** 要求各模态的 Embedding 维度相同，且处于同一语义空间。直接对不同模态的 Embedding 取平均可能没有物理意义（文本的 512 维向量与图像的 512 维向量可能编码完全不同的信息）。因此通常需要先将各模态投影到共享语义空间，再进行加权平均。

# 3、逐元素运算（Element-wise Operations）

- 加法：$\mathbf{z} = \mathbf{v}_{text} + \mathbf{v}_{image}$
- 乘法（Hadamard 积）：$\mathbf{z} = \mathbf{v}_{text} \odot \mathbf{v}_{image}$
- 差值：$\mathbf{z} = |\mathbf{v}_{text} - \mathbf{v}_{image}|$

逐元素乘法可以捕捉两个模态在各个维度上的"共振"（共同激活的维度），常用于视觉问答（VQA）等任务。

# 二、注意力机制融合

动态学习不同模态的重要性权重，解决特征贡献不平衡问题。

# 1、跨模态注意力（Cross-Attention）

原理：以 Query 模态为基准，计算其对 Key 模态的注意力权重。

公式（以文本-图像为例）：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

$$
\mathbf{Q} = \mathbf{W}_q \mathbf{v}_{\text{text}}, \quad \mathbf{K} = \mathbf{W}_k \mathbf{v}_{\text{image}}, \quad \mathbf{V} = \mathbf{v}_{\text{image}}
$$

效果：增强相关特征（如广告文案文本中的"运动鞋"与图片中的鞋款像素特征对齐）。

**Cross-Attention 的计算复杂度：** 设 Query 序列长度为 $L_q$，Key/Value 序列长度为 $L_k$，维度为 $d$，则 Cross-Attention 的计算复杂度为 $O(L_q \cdot L_k \cdot d)$。在图像-文本融合中，如果图像的 patch 数量较多（如 $14 \times 14 = 196$），计算量会显著增加。

**多头跨模态注意力：** 类似 Transformer 的多头机制，将 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 分别投影到 $h$ 个子空间，独立计算注意力后拼接：

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O
$$

这使得模型可以在不同的语义子空间中捕捉不同类型的跨模态关系。

# 2、门控多模态单元（Gated Multimodal Unit, GMU）

原理：学习不同模态的选择门控 Gate，通过 Gate 抑制噪声模态的贡献。

公式：$\mathbf{z} = g \cdot \mathbf{v}_{\text{text}} + (1 - g) \cdot \mathbf{v}_{\text{image}}, \quad g = \sigma(\mathbf{W}_g[\mathbf{v}_{\text{text}}; \mathbf{v}_{\text{image}}])$

$\sigma$ 为 Sigmoid 函数，$g$ 控制文本与图像的融合比例。

**GMU 的直觉理解：** 当文本信息更可靠（如描述详细的商品文案）时，门控 $g$ 倾向于接近 1，主要使用文本特征；当图像信息更可靠（如品牌 Logo 清晰的商品图）时，$g$ 接近 0，主要使用图像特征。这种自适应机制使得融合结果在不同样本上能够动态调整。

# 三、动态自适应融合

# 1、FusionMamba（2025 SOTA）

FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba

![](images/694c2406b18e007f8163482bb281cebc790b06d2d4591d09938fb17e3d18812a.jpg)

![](images/9f17475247d2780c5b4971e324a91c01480a48141e882648c17bee0d3e896572.jpg)

原理：在状态空间模型（SSM）中引入跨模态门控，实现隐空间动态融合。

- 动态视觉状态空间模块（DVSS）：将图像分块映射为状态序列
- 门控机制：调节文本对图像特征的增强强度

公式：$\mathbf{h}_{t+1} = \mathbf{A}\mathbf{h}_t + \mathbf{B}(\mathbf{v}_{\text{image}} \odot \sigma(\mathbf{W}_c \mathbf{v}_{\text{text}}))$

其中，$\mathbf{A}$ 为状态转移矩阵，$\odot$ 为逐元素乘。

优势：在 RGB-IR 目标检测任务中超越 Transformer，适合广告素材中的跨模态对齐（如商品图与描述文本）。

**Mamba/SSM 的核心优势：** 状态空间模型（State Space Model, SSM）通过线性递推处理序列，计算复杂度为 $O(N)$（Transformer 为 $O(N^2)$），在长序列建模中具有显著优势。FusionMamba 将跨模态门控融入 SSM 的状态转移过程，实现了高效的多模态融合。

# 四、双线性池化（Bilinear Pooling）

捕捉模态间高阶交互，提升细粒度特征融合。

# 1、多模态紧凑双线性池化（MCB）Multimodal Compact Bilinear

![](images/1731f9ecf44c27d1232604de9aa0556761585f2490cfada1e5afd96e363c404d.jpg)

原理：将特征投影到高维空间后做外积，再通过 FFT 加速计算。

$\mathbf{z} = \text{FFT}^{-1}(\text{FFT}(\phi(\mathbf{v}_{\text{text}})) \odot \text{FFT}(\phi(\mathbf{v}_{\text{image}})))$，$\phi$ 为随机投影矩阵。

**双线性池化的数学本质：** 标准双线性池化计算两个向量的外积：

$$
\mathbf{z} = \mathbf{v}_{\text{text}} \otimes \mathbf{v}_{\text{image}} \in \mathbb{R}^{d_t \times d_i}
$$

这会产生 $d_t \times d_i$ 维的特征，维度爆炸。MCB 通过 FFT 在频域中高效近似外积运算，将计算复杂度从 $O(d_t \times d_i)$ 降低到 $O((d_t + d_i)\log(d_t + d_i))$。

改进：MFB（多模态因子分解双线性池化），引入低秩分解降低计算量：

$$
\mathbf{z} = \mathbf{U}^T(\mathbf{v}_{\text{text}} \otimes \mathbf{v}_{\text{image}}), \mathbf{U} \text{为低秩投影矩阵}
$$

# 五、融合方法对比

| 方法 | 计算复杂度 | 交互能力 | 可解释性 | 适用场景 |
|------|----------|---------|---------|---------|
| 拼接 | 低 | 无（依赖后续网络） | 低 | 基线方法、快速验证 |
| 加权平均 | 低 | 弱（线性组合） | 中 | 模态重要性已知的场景 |
| Cross-Attention | 高 | 强（全局交互） | 中 | 复杂跨模态推理任务 |
| GMU | 中 | 中（门控选择） | 高 | 模态质量不稳定的场景 |
| 双线性池化 | 高 | 强（高阶交互） | 低 | 细粒度特征融合 |
| FusionMamba | 低 | 强（动态门控） | 中 | 长序列多模态任务 |

# 六、Python 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatenationFusion(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_feat, image_feat):
        fused = torch.cat([text_feat, image_feat], dim=-1)
        return self.fc(fused)

class GatedMultimodalUnit(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

    def forward(self, text_feat, image_feat):
        gate = self.gate(torch.cat([text_feat, image_feat], dim=-1))
        z = gate * self.text_proj(text_feat) + (1 - gate) * self.image_proj(image_feat)
        return z

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, image_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        self.q_proj = nn.Linear(text_dim, text_dim)
        self.k_proj = nn.Linear(image_dim, text_dim)
        self.v_proj = nn.Linear(image_dim, text_dim)
        self.out_proj = nn.Linear(text_dim, text_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, text_feat, image_feat):
        B = text_feat.size(0)
        Q = self.q_proj(text_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, -1)
        return self.out_proj(out)

text_feat = torch.randn(4, 256)
image_feat = torch.randn(4, 256)

concat = ConcatenationFusion(256, 256, 128, 10)
gmu = GatedMultimodalUnit(256, 256, 128)
cross_attn = CrossAttentionFusion(256, 256)

print("拼接融合:", concat(text_feat, image_feat).shape)
print("GMU融合:", gmu(text_feat, image_feat).shape)
print("Cross-Attention融合:", cross_attn(text_feat, image_feat).shape)
```

# 七、面试常见追问

1. **多模态融合的 Early/Middle/Late Fusion 有什么区别？** Early Fusion 在输入层融合原始特征，Middle Fusion 在中间表征层融合，Late Fusion 在决策层融合各自的预测结果。Middle Fusion（如 Cross-Attention）是当前主流，因为它允许模型在合适的抽象层次上建模跨模态交互。

2. **如何处理模态缺失问题？** 常见策略包括：使用默认值（如零向量）填充缺失模态、训练模态特定的缺失指示器、使用注意力机制自动降低缺失模态的权重。GMU 的门控机制天然适合处理模态缺失，因为它可以学习将缺失模态的权重设为 0。

3. **推荐系统中多模态融合的实际应用？** 在电商推荐中，商品有文本描述、图片、视频等多模态信息。多模态融合可以帮助：冷启动（新商品无交互数据，但有多模态内容）、跨模态检索（以文搜图）、多模态相似度计算（综合文本和视觉相似性）。
