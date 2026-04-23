# 面试题：介绍阿里 ESANS 召回负采样方法

# 面试题：介绍阿里 ESANS 召回负采样方法

# 一、提出背景

在推荐系统召回阶段，负采样质量直接影响模型区分用户兴趣的能力。传统方法存在三大痛点

 随机采样（UNS）：易采样到与用户兴趣无关的"简单负样本"（如冷门商品），模型难以学习细粒度差异；  
 启发式规则采样 （如 Airbnb 同城未点击样本）：引入流行度偏差，导致长尾覆盖不足；  
 基于模型的硬负采样 （如 MixGCF）：计算成本高且易生成语义不完整的伪负样本（False Negatives）。

阿里 ESANS 的提出旨在通过多模态语义对齐和动态难度控制，解决上述问题，提升召回模型的语义理解能力和长尾覆盖效果。

论文地址：ESANS: Effective and Semantic-Aware Negative Sampling for Large-Scale Retrieval Systems

# 模型原理

![](images/f5ae3d89d9da58b112898e1a161ea5ef54fb67ee5901fdb5aed7ea87faf6b9b1.jpg)

![](images/c9acbdccff6c57a5e4bd38aea9cb1d1e775e7173a6df66ba9aa09bcf9ed04213.jpg)  
Figure2:Our proposed ESANS framework.a)Multimodal-aligned Technique.b) Vector Quantized Clustering with Cascaded Codebooks.c) Semantic-Aware Negative Sampling & Effective Dense Interpolation Strategy (EDIS).

ESANS 框架包含三个核心模块：

# 1. 多模态对齐与分层聚类

 多模态对齐：融合文本（BERT）、图像（CLIP）、行为（GNN）三种模态特征，通过对比学习对齐语义空间：  
 分层残差量化（RQ）：

 一级码本：粗粒度聚类，基于多模态均值特征进行 K-means 划分；  
 二级码本：细粒度划分，对一级聚类残差（各模态特征与一级中心的差值拼接）再次聚类。

# 2. 语义感知负采样策略

 易负样本（Easy Negatives）：从其他一级簇中按相似度概率采样：

$$
P (C _ {j} | C _ {i}) \propto {\frac {1}{\mathrm {d i s t} (C _ {i} , C _ {j})}}, \quad {\text {归 一 化 为}}   {\frac {e ^ {- \mathrm {d i s t} (C _ {i} , C _ {j})}}{\sum_ {k} e ^ {- \mathrm {d i s t} (C _ {i} , C _ {k})}}}
$$

 硬负样本（Hard Negatives）：在同一一级簇内但不同二级簇中采样，确保语义相近但细节差异。

# 3. 高效密集插值（EDIS）

简单插值：在簇内样本间线性插值生成虚拟样本：

$$
\mathbf {e} _ {\text {v i r t u a l}} = \alpha \mathbf {e} _ {i} + (1 - \alpha) \mathbf {e} _ {j}, \quad \alpha \sim U (0, 1)
$$

困难插值：在正样本与硬负样本间插值，动态调整难度：

$$
\mathbf {e} _ {\text {h a r d - i n t e r p}} = \beta \mathbf {e} _ {\text {p o s}} + (1 - \beta) \mathbf {e} _ {\text {h a r d}}, \quad \beta \in [ - 0. 5, 1. 5 ]
$$

# 三、实验效果

# 1. 离线实验

 数据集：Amazon Electronics、Pixel-Rec 等；  
 指标：Recall@50 平均提升 $1 5 . 3 2 \%$ ，Recall@200 提升 $1 0 . 7 3 \%$ （见表 2）。

# 2. 在线 A/B 测试

 电商场景：广告收入 $+ 2 . 8 3 \%$ ， $\mathsf { C T R } { + } 1 . 1 9 \%$ ， $6 M V + 1 . 9 4 \%$

# 四、小结

 多模态语义对齐：消除单模态偏差，提升负样本语义相关性；  
 动态难度控制：通过插值策略平衡难/易样本比例；  
 长尾覆盖优化：分层聚类减少伪负样本，提升冷门商品召回率。

---

# 五、数学推导补充

## 1. 多模态对比学习对齐损失

对于三种模态特征 $\mathbf{f}_{\text{text}}, \mathbf{f}_{\text{image}}, \mathbf{f}_{\text{behavior}}$，ESANS 通过跨模态对比损失对齐语义空间。以文本-图像对齐为例：

$$
\mathcal{L}_{\text{align}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{f}_i^t, \mathbf{f}_i^v) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{f}_i^t, \mathbf{f}_j^v) / \tau)}
$$

融合后的多模态特征为加权组合：

$$
\mathbf{f}_{\text{fused}} = w_1 \mathbf{f}_{\text{text}} + w_2 \mathbf{f}_{\text{image}} + w_3 \mathbf{f}_{\text{behavior}}
$$

## 2. 残差量化聚类推导

一级聚类将物品按融合特征 $\mathbf{f}_{\text{fused}}$ 进行 K-means 划分，得到聚类中心 $\{\mathbf{c}_1^{(1)}, ..., \mathbf{c}_K^{(1)}\}$。

二级聚类对残差进行细粒度划分：

$$
\mathbf{r}_i = [\mathbf{f}_{\text{text}} - \mathbf{c}_{k_i}^{(1)}, \mathbf{f}_{\text{image}} - \mathbf{c}_{k_i}^{(1)}, \mathbf{f}_{\text{behavior}} - \mathbf{c}_{k_i}^{(1)}]
$$

其中 $k_i$ 为物品 $i$ 所属的一级簇编号，残差向量 $\mathbf{r}_i$ 拼接后再次进行 K-means 聚类。

## 3. 难度插值的边界分析

困难插值中 $\beta \in [-0.5, 1.5]$ 允许生成比硬负样本更难的虚拟样本：

- 当 $\beta < 0$ 时：$\mathbf{e}_{\text{hard-interp}}$ 位于正样本对面的区域，难度极高
- 当 $\beta > 1$ 时：$\mathbf{e}_{\text{hard-interp}}$ 位于正样本远离硬负样本的方向，同样增加难度

这种设计确保了训练过程中难度的多样性。

# 六、与传统负采样方法对比

| 方法 | 语义感知 | 难度控制 | 伪负样本风险 | 计算开销 | 长尾覆盖 |
|------|---------|---------|-------------|---------|---------|
| 随机采样（UNS） | 否 | 无 | 高 | 极低 | 差 |
| 流行度采样 | 部分 | 无 | 中 | 低 | 差 |
| 同城未点击（Airbnb） | 部分 | 弱 | 中 | 低 | 中 |
| MixGCF | 是 | 强 | 中高 | 高 | 中 |
| ESANS | 是（多模态） | 强（动态插值） | 低 | 中 | 强 |

# 七、应用场景

**电商召回**：商品检索场景中，通过语义感知负采样提升长尾商品的曝光机会。

**广告检索**：广告召回阶段需要精确区分相似广告，硬负样本的质量直接影响排序效果。

**内容推荐**：图文、视频等多模态内容场景，多模态对齐能显著提升语义匹配精度。

**搜索引擎**： query-doc 匹配场景中，语义对齐的负采样能提升检索相关性。

# 八、优缺点分析

## 优点

- 多模态融合消除了单模态偏差，负样本语义质量显著提升
- 分层聚类+难度插值实现了从易到难的渐进式训练
- 有效减少伪负样本（False Negatives），保护长尾物品
- 在线 A/B 实验效果显著，广告收入提升 2.83%

## 缺点

- 依赖多模态特征（BERT、CLIP、GNN），离线预处理成本高
- 分层聚类的簇数（K值）需要调优，不同场景最佳参数可能不同
- EDIS 插值生成的虚拟样本可能偏离真实数据分布
- 框架整体较复杂，工程落地需要较多基础设施支持

# 九、Python 代码实现（ESANS 核心逻辑）

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class ESANSNegSampler:
    def __init__(self, n_clusters_l1=100, n_clusters_l2=50, temperature=0.1):
        self.n_clusters_l1 = n_clusters_l1
        self.n_clusters_l2 = n_clusters_l2
        self.temperature = temperature
        self.kmeans_l1 = MiniBatchKMeans(n_clusters=n_clusters_l1, batch_size=1000)
        self.kmeans_l2_dict = {}
        self.cluster_centers_l1 = None
        self.item_embeddings = None
        self.item_cluster_l1 = None
        self.item_cluster_l2 = None

    def fit(self, item_embeddings):
        self.item_embeddings = item_embeddings
        n_items = item_embeddings.shape[0]

        self.kmeans_l1.fit(item_embeddings)
        self.cluster_centers_l1 = self.kmeans_l1.cluster_centers_
        self.item_cluster_l1 = self.kmeans_l1.labels_

        for c in range(self.n_clusters_l1):
            mask = self.item_cluster_l1 == c
            if mask.sum() < self.n_clusters_l2:
                continue
            residuals = item_embeddings[mask] - self.cluster_centers_l1[c]
            kmeans_l2 = MiniBatchKMeans(n_clusters=self.n_clusters_l2, batch_size=500)
            kmeans_l2.fit(residuals)
            self.kmeans_l2_dict[c] = {
                "model": kmeans_l2,
                "labels": kmeans_l2.labels_,
                "indices": np.where(mask)[0]
            }

    def sample_easy_negatives(self, anchor_cluster, n_negatives):
        other_centers = np.delete(self.cluster_centers_l1, anchor_cluster, axis=0)
        other_indices = list(range(self.n_clusters_l1))
        other_indices.remove(anchor_cluster)

        dists = np.linalg.norm(other_centers - self.cluster_centers_l1[anchor_cluster], axis=1)
        probs = np.exp(-dists / self.temperature)
        probs = probs / probs.sum()

        sampled_clusters = np.random.choice(other_indices, size=n_negatives, p=probs, replace=True)
        negatives = []
        for c in sampled_clusters:
            items_in_c = np.where(self.item_cluster_l1 == c)[0]
            neg = np.random.choice(items_in_c)
            negatives.append(neg)
        return np.array(negatives)

    def sample_hard_negatives(self, anchor_cluster_l1, n_negatives):
        if anchor_cluster_l1 not in self.kmeans_l2_dict:
            return self.sample_easy_negatives(anchor_cluster_l1, n_negatives)

        info = self.kmeans_l2_dict[anchor_cluster_l1]
        sub_labels = info["labels"]
        orig_indices = info["indices"]

        n_sub = len(np.unique(sub_labels))
        sampled_sub = np.random.choice(n_sub, size=n_negatives, replace=True)
        negatives = []
        for s in sampled_sub:
            items_in_sub = orig_indices[sub_labels == s]
            if len(items_in_sub) > 0:
                neg = np.random.choice(items_in_sub)
                negatives.append(neg)
        if len(negatives) == 0:
            return self.sample_easy_negatives(anchor_cluster_l1, n_negatives)
        return np.array(negatives[:n_negatives])

    def edis_interpolation(self, pos_embed, hard_neg_embed, n_virtual, hard=True):
        virtual_embeds = []
        for _ in range(n_virtual):
            if hard:
                beta = np.random.uniform(-0.5, 1.5)
                virtual = beta * pos_embed + (1 - beta) * hard_neg_embed
            else:
                alpha = np.random.uniform(0, 1)
                idx_i = np.random.randint(0, len(hard_neg_embed))
                idx_j = np.random.randint(0, len(hard_neg_embed))
                virtual = alpha * hard_neg_embed[idx_i] + (1 - alpha) * hard_neg_embed[idx_j]
            virtual_embeds.append(virtual)
        return np.array(virtual_embeds)


np.random.seed(42)
n_items = 5000
embed_dim = 64
item_embeddings = np.random.randn(n_items, embed_dim).astype(np.float32)
item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

sampler = ESANSNegSampler(n_clusters_l1=50, n_clusters_l2=10)
sampler.fit(item_embeddings)

easy_negs = sampler.sample_easy_negatives(anchor_cluster=0, n_negatives=10)
hard_negs = sampler.sample_hard_negatives(anchor_cluster_l1=0, n_negatives=10)

print(f"易负样本索引: {easy_negs}")
print(f"硬负样本索引: {hard_negs}")
print(f"易负样本与簇0中心平均距离: {np.mean([np.linalg.norm(item_embeddings[i] - sampler.cluster_centers_l1[0]) for i in easy_negs]):.4f}")
print(f"硬负样本与簇0中心平均距离: {np.mean([np.linalg.norm(item_embeddings[i] - sampler.cluster_centers_l1[0]) for i in hard_negs]):.4f}")
```

# 十、常见问题与易错点

## 1. 簇数 K 的选择

K 值过大导致簇内样本过少，难以采样足够的硬负样本；K 值过小则簇间差异不足，硬负样本质量下降。建议根据物品总量选择 $\sqrt{N}$ 附近的值。

## 2. 多模态特征缺失

部分物品可能缺失图像或文本特征。此时需要用零向量填充或训练模态特定的缺失值预测器，避免影响聚类质量。

## 3. 聚类更新频率

物品库动态变化，聚类需要定期更新。但全量重聚类成本高，可采用增量聚类或定时全量+实时增量的混合策略。

## 4. 插值样本的标签问题

EDIS 生成的虚拟样本没有真实的用户交互标签，只能作为负样本使用。若插值区域恰好存在潜在正样本，会引入假负样本噪声。

# 十一、学习路径建议

1. **基础**：掌握召回模型（DSSM、MIND）和负采样基本原理
2. **核心**：理解对比学习中负样本质量对模型效果的影响
3. **进阶**：学习多模态表示学习（CLIP、BLIP）和向量量化方法
4. **拓展**：研究 Graph 负采样（MixGCF）、对抗式负采样等前沿方法
