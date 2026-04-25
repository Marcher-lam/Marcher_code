# 面试题：阿里长序列建模 SIM 方案原理介绍

# 面试题：阿里长序列建模 SIM 方案原理介绍

SIM 论文链接：Search-based User Interest Modeling with Lifelong Sequential Behavior Data

# 1. 核心背景

在推荐系统中，用户行为序列长度直接影响兴趣建模的准确性。传统方法（如 DIN、DIEN）仅能处理数百量级长度的行为数据，而用户全生命周期行为可能长达数万次。直接建模全序列会导致计算复杂度爆炸（如 DIN 的注意力机制复杂度为O(BLd)，L 为序列长度），且在线服务时延无法满足实时性要求。

SIM（Search-based Interest Model）通过两阶段搜索范式，将序列长度从万级压缩至百级，同时精准捕捉候选 Item 相关的兴趣。

# 2. 两阶段架构设计

![](images/d69ddbe9e3f64dcb9c2d369a3cdfc0d9136507e8f94300cfb15c9c33a352431e.jpg)

# 2.1 第一阶段：通用搜索单元（GSU）

目标：从原始长序列中快速筛选与候选 Item 相关的子序列（Top-K），将序列长度从万级降至百级。

实现方式：

 Hard Search（硬搜索）

基于规则的非参数化方法，通过类目匹配筛选行为。例如，候选 Item 为“连衣裙”，则仅保留用户历史中同类目的行为。优势是速度快、易部署，但可能损失跨类目相关性信息。

 Soft Search（软搜索）

参数化方法，通过 Embedding 内积相似度筛选。关键点包括：

 Embedding 优化：为避免长/短期行为分布差异，引入辅助 CTR 任务训练长期行为 Embedding，确保相似度与点击相关性一致。  
 近似检索：采用 ALSH（非对称局部敏感哈希）算法，实现次线性时间检索，支持大规模行为库快速匹配。

 索引结构：用户行为树（UBT）采用 Key-Key-Value 存储（一级 Key 为用户 ID，二级 Key 为类目），分布式部署支持高并发查询。

# 2.2 第二阶段：精确搜索单元（ESU）

目标：对GSU筛选的子序列进行精细化兴趣建模，支持复杂模型（如 DIN、DIEN）的深度计算。

关键技术：

 动态注意力机制：引入候选 Item与子序列的时间间隔特征，增强时间衰减效应。例如，近期行为赋予更高权重。  
 多头注意力优化：通过多组独立注意力头捕捉多样化兴趣，防止单一注意力头的信息偏置。  
 特征融合：将候选 Item 的 Embedding 与行为 Embedding 拼接，输入 MLP 层进行高阶特征交互。

# 3. 损失函数与训练策略

 联合训练：模型损失包括 GSU 和 ESU 两部分，通过超参数加权（公式：L = αL_GSU $^ +$ βL_ESU）。其中，Hard Search模式下 $\mathtt { q } = 0$ （无监督筛选），Soft Search 需同步优化辅助 CTR 任务的 Embedding 参数。  
 采样策略：训练时对原始长序列随机采样，保持数据分布一致性，降低计算开销。

# 4. 技术优势与效果

 效率突破：在线服务时延仅增加 5ms，支持最大 54,000 长度的行为序列（较 MIMN 提升 54 倍）。  
 效果提升：在阿里广告场景中，CTR 提升 $7 . 7 \%$ 、PRM 提升 $4 . 4 \%$ ，主要得益于噪声过滤与精准兴趣捕捉。  
 工程友好性：GSU 的索引结构可离线预计算，在线仅需百级序列的实时计算，降低存储与通信成本。

# 5. 局限性

 目标不一致性：GSU索引依赖预训练Embedding或类目标签，可能偏离实际CTR任务目标。后续ETA模型引入SimHash统一 Embedding 空间，缓解此问题。  
更新延迟：离线索引更新频率低于在线模型，动态兴趣捕捉受限。部分方案尝试结合增量更新与在线索引。

小结：

SIM 通过“粗筛+精算”的两阶段架构，平衡了长序列建模的效率与精度，成为工业级推荐系统的标杆方案。其核心思想——以候选Item 为锚点的相关性搜索——为后续长序列模型（如ETA、SDIM）提供了重要范式。

论文地址：Temporal Interest Network for User Response Prediction

# 一、论文背景

传统推荐模型（如 DIN、DIEN、SASRec）仅单独建模用户行为的语义相关性 （例如商品类别匹配）或时间相关性 （例如行为顺序），但未能有效结合两者。例如：

 语义相关性不足：用户近期点击的同类商品可能因时间间隔过长而失效。  
 时间建模粗粒度：仅依赖位置编码或简单时间衰减函数，无法捕捉真实场景中的复杂时序模式。

TIN 提出语义-时间四向交互，通过联合建模行为与目标的语义关联及动态时间衰减，解决上述问题。

# 二、 模型架构详解

![](images/e101439d9df1cd12c779d12ddb8f49003d096b31542282001a01433c42a479ff.jpg)  
(a) TIN Architecture

![](images/f36c8e1b4db1c10e7bcb6a6254bde679845419e73b4e24ae2d735ac588755b26.jpg)  
(b) Temporal Interest Module

# 1. 核心模块设计

TIN 的核心是时间兴趣模块（Temporal Interest Module, TIM） ，包含以下关键组件：

 目标感知时间编码（Target-aware Temporal Encoding, TTE）

TTE-P（相对位置编码）：根据行为在序列中的位置（如倒数第 5 次点击）编码时间信息。

TTE-T（时间间隔编码）：基于行为与目标的时间差（如点击广告前 3 天）动态调整权重。

公式： $e _ { i } = { \mathrm { E m b e d } } ( x _ { i } ) + { \mathrm { T T E } } ( t _ { i } )$ ，其中， $x _ { i }$ 为行为语义特征， $t _ { i }$ 为时间特征。

 目标感知注意力（Target-aware Attention, TA）

使用缩放点积注意力（Scaled Dot-Product Attention）计算行为与目标的语义-时间相关性：

$$
\alpha_ {i} = \operatorname {S o f t m a x} \left(\frac {Q \cdot K _ {i}}{\sqrt {d}}\right)
$$

Q 为目标嵌入， $K _ { i }$ 为用户历史行为嵌入。

 目标感知表示（Target-aware Representation, TR）

通过元素级乘法显式融合行为与目标的嵌入：

$$
v _ {i} = e _ {i} \odot \operatorname {E m b e d} (y)
$$

其中 y 为候选目标特征。

#  四向交互

将 TA 的注意力权重与 TR 的融合表征相乘，实现语义-时间联合建模：

$$
\text {O u t p u t} = \sum \left(\alpha_ {i} \cdot v _ {i}\right)
$$

该操作同时捕捉了“行为语义×目标语义×行为时间×目标时间”的高阶交叉。

# 2. 模型优势

 动态时间衰减：在广告场景中，用户点击行为稀疏，时间间隔编码（TTE-T）比相对位置编码（TTE-P）更有效。  
 噪声过滤：通过硬搜索（Hard-Search）从万级长序列中筛选百级相关子序列，提升计算效率。

# 三、实验结果与落地效果

# 1. 离线实验

数据集：Amazon（商品评论）和 Alibaba（广告点击日志）。  
指标：GAUC（全局 AUC）和 LogLoss。

结果：TIN 相比最佳基线提升 $0 . 4 3 \%$ （Amazon）和 $0 . 5 1 \%$ （Alibaba）。

# 2. 在线应用

 腾讯微信朋友圈广告中，TIN 带来 $1 . 9 3 \%$ 的 GMV 提升，时间间隔嵌入的衰减效应显著强于相对位置。  
 支持最大 54,000 长度的行为序列，在线时延仅增加 5ms。

# 四、代码实现

1. 代码：GitHub 开源地址：https://github.com/zhouxy1003/TIN

# 2. 工程优化技巧

 长序列处理：采用类目分层采样（Category Stratified Sampling）保证稀疏行为的覆盖率。  
异构序列解耦：使用多组 TIN 分别建模广告域与内容域行为，通过门控机制融合。

# 一、模型原理

KuaiFormer 是快手提出的基于 Transformer 架构的召回模型，旨在通过 Next Action Prediction 范式重构短视频推荐系统的检索流程。其核心原理包括以下部分：

# 1、序列化用户行为建模

将用户历史交互行为（如观看、点赞、分享等）转化为序列数据，每条记录包含视频 ID 及附加属性（如观看时长、分类标签等）。通过离散特征嵌入 （视频 ID、标签）和连续特征分桶嵌入 （时长统计）。将用户行为序列编码为稠密向量，并输入 Transformer 骨干网络（基于 Llama 架构改进）进行序列建模。

# 2、层次化序列压缩机制

针对长序列计算效率问题，提出自适应序列压缩策略：将用户行为序列按时间划分为早、中、晚三部分，分别以 64和 16 的粒度进行分组聚合。早期序列通过单层无掩码 Transformer 压缩为单个表征，保留核心兴趣信息，最终将输入长度从 256 压缩至可处理范围，计算资源消耗降低至原方案的 $10 \%$ 。

# 3、多兴趣提取与生成式预测

引入多 Query Token 机制：在序列头部添加多个可学习的特殊 Token（类似 BERT 的[CLS]），通过自注意力机制生成用户的多维兴趣表征。预测阶段取多兴趣与候选视频的最大内积作为得分，实现多兴趣解耦与动态融合。

# 4、高效训练优化

 In-batch Softmax加速：采用批次内负采样替代全局 Softmax，解决数十亿候选视频的计算瓶颈。  
 LogQ 校正：对采样偏差进行修正，缓解热门视频作为负样本的过拟合问题。  
 标签平滑：因用户行为存在模糊性（如划走视频不代表不感兴趣），将硬标签 0/1 替换为平滑概率分布。

![](images/f7da0ad3f2f9f0f77a60325bf7fbd9d99337fd662f4991d7703b2ec6e1a56d0e.jpg)

# 二、解决的痛点问题

KuaiFormer 针对工业级短视频推荐系统的三大痛点提出了解决方案：

# 1、动态候选库与计算效率矛盾

传统召回模型（如双塔结构）需维护数十亿视频的 Embedding 表，更新成本高。KuaiFormer 通过 Next Action Prediction范式直接生成候选表征，结合 GPU暴力检索（替代 ANN索引），实现分钟级在线更新与实时反馈。

# 2、长序列建模资源瓶颈

Transformer的复杂度（O(N²)）限制了长序列处理能力。通过层次化压缩策略，在保持 256长度序列建模能力的同时，将计算资源消耗降低至原方案的 $10 \%$ 。

# 3、用户兴趣多样性与实时性挑战

短视频场景中用户兴趣快速变化且呈现多样性。多 Query Token 机制可同时捕捉实时兴趣（近期行为）与长期偏好（压缩后的早期行为），相比传统多兴趣模型（如 ComiRec）NDCG $@ 1 0 0$ 提升 $2 5 \%$ 。

# 三、核心创新点

# 1. Next Action Prediction 范式重构

将传统 CTR 预估转化为序列生成任务，实现召回与排序目标一致性，同时支持端到端训练。

# 2. 自适应序列压缩

基于"早期行为记忆模糊"假设设计的分级压缩策略，兼顾长序列建模与计算效率，相比未压缩方案在 256长度下资源消耗减少 $8 3 \%$ 。

# 3. 多兴趣动态融合机制

通过可学习的多Query Token实现兴趣解耦，结合最大内积得分策略，在离线测试中相比单兴趣模型 $\mathsf { H R @ 1 0 0 }$ 提升$30 \%$ 。

# 4. 工业级训练优化组合

LogQ 校正+标签平滑的联合训练方案，缓解采样偏差与行为模糊性，线上 A/B 实验观看时长提升 $0 . 3 6 \% { - 0 . 4 1 \% }$ 。

# 五、长序列建模方案对比与选型指南

## 5.1 SIM 与其他长序列建模方案对比

| 方案 | 最大序列长度 | 核心思想 | 计算复杂度 | 在线延迟增量 | 工程复杂度 | 代表来源 |
|------|-----------|---------|-----------|------------|-----------|---------|
| MIMN | ~1000 | 记忆网络（Memory Network）存储长期兴趣 | O(BMd) M为记忆槽 | <5ms | 中 | 阿里 (2019) |
| SIM (Hard) | ~54000 | 粗筛（类目匹配）+ 精算（注意力） | O(BKd) K为筛选后长度 | ~5ms | 中 | 阿里 (2020) |
| SIM (Soft) | ~54000 | 粗筛（Embedding 相似度）+ 精算 | O(BNd) N为原始长度 | ~15ms | 高 | 阿里 (2020) |
| ETA | ~54000 | SimHash 替代 Embedding 相似度筛选 | O(BKd) + Hash | ~8ms | 中高 | 阿里 (2021) |
| TWIN | ~32000 | 基于时间感知的注意力 + KV Cache | O(BKd) | ~10ms | 高 | 快手 (2023) |
| LONGER | ~100000 | 大语言模型作为兴趣编码器 | O(BL²d) | >50ms | 极高 | 学术 (2024) |
| SDIM | ~54000 | 随机哈希近似注意力 | O(BKd) | ~3ms | 中 | 阿里 (2022) |

## 5.2 SIM (Hard) vs SIM (Soft) 详细对比

| 维度 | SIM Hard Search | SIM Soft Search |
|------|----------------|-----------------|
| 筛选依据 | 类目/属性精确匹配 | Embedding 内积相似度 |
| 召回率 | 低（可能漏掉跨类目相关行为） | 高（语义相似即召回） |
| 计算速度 | 快（O(1) 查表） | 慢（需计算 Embedding 相似度） |
| 在线延迟 | ~5ms | ~15ms |
| 索引构建 | UBT 树（Key-Key-Value） | ALSH 近似检索 |
| 跨域相关性 | 不支持 | 支持（语义空间跨域） |
| Embedding 一致性 | 不依赖 Embedding | 需要辅助任务训练 Embedding |
| 工程实现 | 简单 | 复杂（需维护 Embedding 索引） |
| 生产推荐 | ★★★★★（阿里主推） | ★★★☆☆（效果更好但延迟高） |

## 5.3 各方案优缺点分析

### SIM 的优势
1. **两阶段解耦**：粗筛与精算分离，各自可独立优化和部署
2. **工程友好**：Hard Search 模式仅增加 5ms 延迟，适合实时推荐
3. **序列长度突破**：支持 54000 长度，是 MIMN 的 54 倍

### SIM 的劣势
1. **目标不一致**：GSU 的筛选目标与 ESU 的 CTR 预估目标不完全一致
2. **Hard Search 信息损失**：类目匹配可能漏掉语义相关但类目不同的行为
3. **Soft Search 延迟高**：Embedding 相似度检索增加 10ms 延迟

### TWIN 的优势
1. **时间感知注意力**：显式建模行为时间间隔，更适合短视频等时序敏感场景
2. **KV Cache 优化**：利用 KV Cache 加速注意力计算

### LONGER 的优势与劣势
1. **优势**：利用 LLM 的通用知识，覆盖超长序列（10万+）
2. **劣势**：延迟过高（>50ms），难以用于实时推荐

## 5.4 长序列建模选型决策树

```
序列长度需求 → 
  < 1000: DIN/DIEN（直接注意力）
  1000-54000:
    ├── 延迟要求 < 10ms → SIM Hard Search 或 SDIM
    ├── 延迟要求 < 20ms → SIM Soft Search 或 ETA
    └── 时序敏感场景 → TWIN
  > 54000:
    ├── 有 LLM 资源 → LONGER
    └── 无 LLM 资源 → 分层压缩 + SIM
```

## 5.5 常见落地陷阱

| 陷阱 | 描述 | 建议 |
|------|------|------|
| 忽略 GSU-ESU 目标不一致 | GSU 筛选行为与 CTR 预估目标偏差 | 定期用 CTR 模型特征重要性校准 GSU 筛选规则 |
| 类目体系变更导致 Hard Search 失效 | 类目调整后历史行为的类目标签过期 | 维护类目映射表，新类目兼容旧类目 |
| Soft Search Embedding 漂移 | 长期行为 Embedding 与短期不一致 | 使用辅助任务定期更新长期行为 Embedding |
| UBT 索引更新延迟 | 离线索引更新频率跟不上实时行为 | 增量索引更新 + 定期全量重建 |
| 序列长度配置不当 | 盲目追求超长序列 | 根据业务场景分析行为衰减曲线，选择有效长度 |
| 内存开销过大 | 54000 长度序列的存储和传输成本高 | 分级存储（近期行为热存 + 远期行为冷存） |

## 常见问题与易错点

1. **Hard Search 的类目粒度**：类目过粗则筛选不够精准，过细则候选太少，通常用二级或三级类目
2. **Soft Search 的 Embedding 漂移**：长期行为 Embedding 需独立训练，直接复用短期 Embedding 会因分布差异导致检索不准
3. **时间间隔特征**：ESU 中的时间衰减推荐用 log(1+Δt) 或可学习 embedding，不宜用简单线性函数
4. **GSU 与 ESU 的目标一致性**：Hard Search 无监督信号，Soft Search 的辅助 CTR 任务应与主任务对齐

## 学习总结

SIM 的核心贡献是"两阶段搜索"范式：GSU 粗筛（万级→百级）+ ESU 精算。Hard Search 因实现简单、延迟低而更常用，Soft Search 效果更优但工程复杂度高。SIM 证明长序列信息对推荐效果提升显著（CTR +7.7%），开启了长序列建模的研究热潮。

## SIM 两阶段代码实现

```python
import torch
import torch.nn as nn


class HardSearchGSU:
    """GSU 硬搜索：基于类目匹配筛选"""
    def __init__(self, top_k=100):
        self.top_k = top_k

    def search(self, user_seq_items, user_seq_cats, target_cat):
        mask = (user_seq_cats == target_cat)
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) > self.top_k:
            indices = indices[-self.top_k:]
        if len(indices) == 0:
            indices = torch.arange(min(self.top_k, len(user_seq_items)))
        return user_seq_items[indices], indices


class SoftSearchGSU(nn.Module):
    """GSU 软搜索：基于 Embedding 内积相似度检索"""
    def __init__(self, item_dim, hidden_dim=64, top_k=100):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        self.target_proj = nn.Linear(item_dim, hidden_dim)
        self.top_k = top_k

    def search(self, user_seq_items, target_item):
        seq_emb = self.item_proj(user_seq_items)
        target_emb = self.target_proj(target_item)
        scores = torch.matmul(seq_emb, target_emb.unsqueeze(-1)).squeeze(-1)
        topk_scores, topk_indices = torch.topk(scores, min(self.top_k, len(scores)))
        return user_seq_items[topk_indices], topk_indices


class ESU(nn.Module):
    """精确搜索单元：注意力建模 + 时间衰减"""
    def __init__(self, item_dim, hidden_dim=64, n_heads=4):
        super().__init__()
        self.time_embed = nn.Linear(1, hidden_dim)
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, sub_seq_items, time_intervals, target_item):
        item_emb = self.item_proj(sub_seq_items)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        combined = item_emb + time_emb
        target_emb = self.item_proj(target_item).unsqueeze(1)
        attn_out, attn_w = self.attn(target_emb, combined, combined)
        output = self.mlp(attn_out.squeeze(1))
        return output.squeeze(-1), attn_w


class SIMModel(nn.Module):
    """SIM 完整模型：GSU + ESU"""
    def __init__(self, item_dim, hidden_dim=64, n_heads=4, top_k=100, mode="hard"):
        super().__init__()
        self.mode = mode
        if mode == "hard":
            self.gsu = HardSearchGSU(top_k)
        else:
            self.gsu = SoftSearchGSU(item_dim, hidden_dim, top_k)
        self.esu = ESU(item_dim, hidden_dim, n_heads)

    def forward(self, user_seq_items, target_item, time_intervals=None,
                user_seq_cats=None, target_cat=None):
        if self.mode == "hard":
            sub_seq, indices = self.gsu.search(user_seq_items, user_seq_cats, target_cat)
        else:
            sub_seq, indices = self.gsu.search(user_seq_items, target_item)
        sub_seq = sub_seq.unsqueeze(0)
        sub_time = time_intervals[indices].unsqueeze(0) if time_intervals is not None else torch.ones(1, len(indices), 1)
        score, attn_w = self.esu(sub_seq, sub_time, target_item.unsqueeze(0))
        return score


if __name__ == "__main__":
    seq_len, item_dim = 10000, 32
    user_seq = torch.randn(seq_len, item_dim)
    target = torch.randn(item_dim)
    time_gaps = torch.rand(seq_len) * 100
    seq_cats = torch.randint(0, 50, (seq_len,))
    target_cat = torch.tensor(5)

    sim_hard = SIMModel(item_dim, mode="hard")
    print(f"Hard Search 分数: {sim_hard(user_seq, target, time_gaps, seq_cats, target_cat).item():.4f}")

    sim_soft = SIMModel(item_dim, mode="soft")
    print(f"Soft Search 分数: {sim_soft(user_seq, target, time_gaps).item():.4f}")
```

