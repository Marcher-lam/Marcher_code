# 面试题：谷歌生成式推荐 TIGER 模型介绍

# 面试题：谷歌生成式推荐 TIGER 模型介绍

以下是谷歌生成式推荐模型 TIGER（Transformer Index for Generative Recommenders）的原理详解，综合其核心创新、技术实现及优势：

论文链接：https://arxiv.org/pdf/2305.05065

# 一、核心范式突破

TIGER提出了一种全新的生成式检索推荐范式，取代了传统推荐系统中"双塔模型+近似最近邻（ANN）搜索"的两阶段流程。

其核心思想是：通过自回归解码直接生成候选物品的语义 ID，而非依赖向量空间相似度匹配。这种范式将 Transformer 模型的参数视为隐式索引，实现端到端的推荐系统架构。

![](images/a6a5425594bf097bbabfc0279458c3fb0032e745994cbb6e40fc9accd3daa446.jpg)

**传统范式的局限性：**
- 双塔模型分别编码用户和物品，通过内积/余弦相似度匹配，无法捕捉复杂的用户-物品交互
- ANN 检索（如 FAISS）在十亿级物品库上的延迟和内存开销巨大
- 新物品需要先训练 Embedding 才能被检索，冷启动效果差

**生成式范式的核心优势：**
- 将推荐问题转化为序列到序列（Seq2Seq）的生成问题
- 模型参数本身就是索引，无需额外的向量检索系统
- 新物品可通过内容特征直接生成语义 ID，天然支持冷启动

# 二、关键技术实现

# 1. 语义 ID 生成（Semantic ID）

目标：将物品内容信息（如文本描述）转化为层次化、可解释的标识符序列，使语义相似的物品具有重叠 ID 结构。

![](images/a8f1db04c9ddc71e4b5b6b7c3e5e1fdb70e5ea053a1adbd5c045097351ab9395.jpg)

实现步骤：

1. 内容编码：使用预训练文本编码器（如 Sentence-T5 或 BERT）将物品文本描述映射为稠密向量 $x \in \mathbb{R}^d$

2. 残差量化（RQ-VAE）：通过多级残差量化生成离散码字序列：

- 编码与残差计算：编码器将 $x$ 映射为潜在表示 $z$，初始残差 $r_0 = z$
- 逐级量化：在每级 $d$（共 $m$ 级）：从码本 $C_d$ 中选取最邻近码字 $c_d = \arg\min_k ||r_d - e_{k,d}||^2$
- 更新残差 $r_{d+1} = r_d - e_{c_d, d}$

3. 重构与训练：量化后的表示

$$
\hat{z} = \sum_{d=0}^{m-1} e_{c_d, d}
$$

输入解码器重构 $x$，损失函数包括重构损失和量化损失：

$$
L = \|x - \text{Decoder}(\hat{z})\|^2 + \beta \sum_{d=0}^{m-1} \|\text{stop\_gradient}(r_d) - e_{c_d, d}\|^2 \quad (\beta=0.25)
$$

4. 碰撞处理：若多个物品映射到同一语义 ID，则在末尾追加唯一标识符（如哈希值）。

**RQ-VAE 与 VQ-VAE 的区别：** VQ-VAE 只进行一级量化，码本大小限制了可表示的物品数量。RQ-VAE 通过多级残差量化，每级使用一个独立的码本，逐级逼近原始表示。$m$ 级量化、每级码本大小为 $K$，可表示 $K^m$ 个不同的语义 ID，大大扩展了表示能力。

**残差量化的数学直觉：** 第一级量化捕捉粗粒度信息（如商品大类），第二级量化捕捉第一级残差（更细粒度的特征），依此类推。每级的量化误差就是下一级的输入，实现了从粗到细的层次化编码。

# 特点：

- 层次化结构：高层码字表示粗粒度类别（如"美妆"），底层细化到子类（如"口红"）
- 语义泛化：相似物品 ID 前缀重叠，支持知识迁移

# 2. 生成式推荐模型

![](images/20045661d6c7ac1f5125321d29d6b0c269c0e9c18569cacb28962a0c28528298.jpg)
(a) Semantic ID generation for items using quantization of content embeddings.
(b) Transformer based encoder-decoder setup for building the sequence-to-sequence model used for generative retrieval.

# 模型架构：

- 输入：用户历史交互序列 $\{i_1, i_2, \ldots, i_t\}$，每个物品 $i$ 的语义 ID 展开为序列 $(c_0^{(i)}, c_1^{(i)}, \ldots, c_{m-1}^{(i)})$

结构：基于 T5 的编码器-解码器 Transformer：

- 编码器：处理用户历史序列，捕捉行为模式
- 解码器：自回归生成目标物品的语义 ID 码字序列

**序列构建细节：** 用户历史 $[i_1, i_2, i_3]$ 的输入序列展开为 $[c_0^{(i_1)}, c_1^{(i_1)}, c_0^{(i_2)}, c_1^{(i_2)}, c_0^{(i_3)}, c_1^{(i_3)}]$（假设 $m=2$）。每个物品的语义 ID 码字之间通过特殊的分隔符连接。解码器逐步生成目标物品的码字 $[c_0^{(target)}, c_1^{(target)}]$。

# 训练与推理：

- 训练目标：最大化目标码字的对数似然（交叉熵 Loss）

$$
\mathcal{L} = -\sum_{d=0}^{m-1} \log P(c_d^{(target)} | c_{<d}^{(target)}, \text{context})
$$

- 推理优化：
  - Beam Search：生成 Top-K 候选 ID 序列
  - 有效性过滤：剔除未注册的无效 ID

**Beam Search 的必要性：** 由于语义 ID 空间 $K^m$ 远大于实际物品数量，贪心解码可能生成不存在的 ID。Beam Search 保留多个候选路径，最后通过有效性过滤（检查生成的 ID 是否对应真实物品）确保输出合法。

# 三、核心优势

# 1. 内存效率：

- 传统双塔模型需存储十亿级物品嵌入表（约 TB 级），TIGER 仅需维护小型码本（MB 级）
- 语义 ID 空间为 $K^m$（$K$ 为码本大小，$m$ 为层级数），可覆盖百亿物品

**内存效率的定量分析：** 假设有 10 亿物品，传统方法需要存储 $10^9 \times 256 \times 4$ bytes $\approx 1$ TB 的 Embedding 表。TIGER 使用 3 级量化，每级码本大小 4096，码本参数仅 $3 \times 4096 \times 256 \times 4$ bytes $\approx 12$ MB，降低约 5 个数量级。

# 2. 冷启动优化：

- 新物品通过内容特征生成语义 ID，无需交互数据即可被推荐
- 语义碰撞（不同物品共享部分 ID）具有意义，缓解长尾问题

**冷启动的工作原理：** 新上架的商品通过文本编码器生成内容向量，再通过 RQ-VAE 量化为语义 ID。由于语义 ID 基于内容特征，与已有商品的语义相似度已经编码在 ID 结构中。即使用户从未与新商品交互，模型也能根据新商品的语义 ID 与用户偏好的相似性来推荐。

# 3. 性能优势：

- 在 Amazon 数据集上，Recall@10 和 NDCG@10 指标超越 SOTA 模型（如 P5、Caser）$20-30\%$
- 支持多样性控制：通过调整 Beam Search 宽度和温度参数平衡相关性与多样性

# 4. 可解释性：语义 ID 的层次结构提供推荐理由（如"运动鞋 > 跑步鞋 > 缓震系列"）

# 四、与传统范式的对比

| 维度 | 传统双塔+ANN | TIGER生成式检索 |
|------|------------|---------------|
| 索引方式 | 显式嵌入索引 | Transformer参数隐式索引 |
| 检索逻辑 | 内积/余弦相似度 | 自回归语义ID生成 |
| 可扩展性 | 新增物品需重训练 | 动态生成新物品ID |
| 内存消耗 | 高（TB级嵌入表） | 低（MB级码本） |
| 冷启动 | 依赖哈希或随机初始化 | 基于内容语义自然融入 |
| 多样性控制 | 需要MMR等后处理 | 通过温度和beam宽度控制 |
| 可解释性 | 低（黑盒Embedding） | 高（层次化语义ID） |

# 五、TIGER 的局限性与挑战

1. **语义 ID 碰撞问题：** 多个物品可能映射到相同的语义 ID（尤其是长尾物品），需要额外的唯一标识符处理，增加了系统复杂度。

2. **Beam Search 的推理延迟：** 虽然避免了 ANN 检索，但自回归生成的多步解码和 Beam Search 的多路径探索可能引入额外的推理延迟。

3. **码本学习的稳定性：** RQ-VAE 的码本在训练初期可能不稳定（码本坍塌问题），需要使用 EMA 更新和码本重置等技巧。

4. **对内容特征的依赖：** 语义 ID 完全基于内容特征生成，如果物品的文本描述质量差或缺失，会影响推荐效果。

5. **长序列建模的限制：** 用户历史交互序列较长时，Transformer 的计算复杂度为 $O(n^2)$，需要截断或使用高效注意力机制。

# 六、Python 代码实现（简化版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class SimpleRQVAE(nn.Module):
    def __init__(self, input_dim, codebook_size=256, num_levels=3, codebook_dim=64):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, codebook_dim)
        )
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.1)
            for _ in range(num_levels)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def quantize(self, z):
        codes = []
        residual = z
        for level in range(self.num_levels):
            dist = torch.cdist(residual, self.codebooks[level])
            code = dist.argmin(dim=-1)
            codes.append(code)
            quantized = self.codebooks[level][code]
            residual = residual - quantized
        return codes

    def reconstruct(self, codes):
        z_hat = torch.zeros(codes[0].shape[0], self.codebooks[0].shape[1])
        if next(self.parameters()).is_cuda:
            z_hat = z_hat.cuda()
        for level, code in enumerate(codes):
            z_hat = z_hat + self.codebooks[level][code]
        return self.decoder(z_hat)

    def forward(self, x):
        z = self.encoder(x)
        codes = self.quantize(z)
        x_hat = self.reconstruct(codes)
        return x_hat, codes

class SimpleGenerativeRecommender(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_items=1000, codebook_size=64, num_levels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        output_vocab = codebook_size
        self.heads = nn.ModuleList([
            nn.Linear(d_model, output_vocab) for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, input_ids, target_codes=None):
        emb = self.embedding(input_ids)
        hidden = self.transformer(emb)
        last_hidden = hidden[:, -1, :]
        logits = [head(last_hidden) for head in self.heads]
        if target_codes is not None:
            loss = 0
            for level in range(self.num_levels):
                loss += F.cross_entropy(logits[level], target_codes[:, level])
            return loss
        return logits

rq_vae = SimpleRQVAE(input_dim=50, codebook_size=64, num_levels=3)
item_features = torch.randn(100, 50)
x_hat, codes = rq_vae(item_features[:5])
print(f"重构输出形状: {x_hat.shape}")
print(f"语义ID (3级码字): {[[c.item() for c in code] for code in codes]}")

recommender = SimpleGenerativeRecommender(vocab_size=64, num_levels=2, codebook_size=64)
input_ids = torch.randint(0, 64, (4, 10))
target_codes = torch.randint(0, 64, (4, 2))
loss = recommender(input_ids, target_codes)
print(f"训练损失: {loss.item():.4f}")
```

# 七、面试常见追问

1. **TIGER 与 P5 的区别？** P5 也是生成式推荐模型，但使用自然语言作为物品标识（如商品标题），而 TIGER 使用 RQ-VAE 生成的语义 ID。语义 ID 相比自然语言更加紧凑，且层次化结构有利于 Beam Search 的效率和准确性。

2. **TIGER 如何处理物品更新？** 新物品通过内容编码器和 RQ-VAE 直接生成语义 ID，无需重新训练生成模型。但如果大量新物品导致码本分布偏移，可能需要重新训练 RQ-VAE。

3. **RQ-VAE 的码本大小如何选择？** 码本大小 $K$ 和层级数 $m$ 需要联合考虑。总容量 $K^m$ 应大于物品总数。常见配置为 $K=1024 \sim 4096$，$m=3 \sim 4$，可覆盖十亿到千亿级物品。

4. **TIGER 的推理延迟如何？** 生成 $m$ 个码字需要 $m$ 步自回归解码，加上 Beam Search 的多路径探索，推理延迟约为传统 ANN 检索的 2-5 倍。但在内存占用上有 100-1000 倍的优势，适合内存受限的场景。
