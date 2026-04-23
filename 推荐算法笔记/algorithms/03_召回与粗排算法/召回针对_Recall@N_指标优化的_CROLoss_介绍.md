# 面试题：召回针对 Recall@N 指标优化的 CROLoss 介绍

# 面试题：召回针对 Recall@N 指标优化的 CROLoss 介绍

# 一、CROLoss 背景

推荐系统中的召回模型常面临指标与损失函数不匹配的问题。传统损失函数（如交叉熵、BPR、Triplet Loss）主要优化分类或排序目标，而非直接针对召回率（Recal $@ \mathbb { N }$ ）这一核心指标。

CROLoss（Customized Recall-Optimized Loss）由 CIKM 2022 提出，旨在通过可定制的损失函数直接优化召回指标，并适配不同业务场景的检索规模需求。

论文地址：CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems

![](images/b30478b06cd1ecfe95a5b32dfad6ca91f03b9fb11e2f130e784456ce67a1a743.jpg)

# 二、核心原理

# 1. 召回指标建模

CROLoss 将 Recall@N 转化为成对比较任务：对每个正样本，确保其与用户的相似度高于负样本。通过引入比较核函数（Comparison Kernel）和权重函数 （Weight Function），动态调整样本对的重要性。

# 2. 定制化能力

 比较核函数：定义正负样本得分差异的惩罚强度，支持 Sigmoid、Softplus、阶跃函数等。  
 权重函数：根据召回规模 N调整样本权重，例如当 N较小时，关注头部样本的区分度。

# 3. 与传统损失的关系

CROLoss 构建了一个统一损失空间，通过选择不同核函数可退化为：

 BPR Loss：选择 Sigmoid 核  
 Triplet Loss：选择 Hinge 核  
 交叉熵：选择 Softmax 核

![](images/2f0a4e2b65de2355f550c63ca9139d09a788657dd368ed9a61e077567b9920e1.jpg)  
Figure 2: Example of optional comparison kernel functions.

# 三、数学公式

# 1. 基本形式

$$
\mathcal {L} = \sum_ {(u, i ^ {+}, i ^ {-})} \phi (s (u, i ^ {-}) - s (u, i ^ {+})) \cdot w (N)
$$

 s(u,i)：用户-物品相似度得分  
 $\phi ( \bigtriangledown )$ ：比较核函数（如 Sigmoid、Softplus）  
 w(N)：权重函数，与 N 相关

# 2. Lambda 梯度优化

引入双核函数机制，分离梯度计算中的排序和权重调整角色：

 核函数 1（如 Sigmoid）：控制样本对的梯度方向  
 核函数 2（如 Softplus）：调整梯度幅值

# 四、实验效果

以下为 CRO Loss 与交叉熵损失、三元组损失、bpr 损失的实验对比结果

<table><tr><td>Datasets</td><td>Methods</td><td>R@50</td><td>R@100</td><td>R@200</td><td>R@500</td></tr><tr><td rowspan="5">Amazon</td><td>cross-entropy loss</td><td>9.68</td><td>13.24</td><td>17.46</td><td>24.26</td></tr><tr><td>triplet loss</td><td>7.53</td><td>11.21</td><td>15.93</td><td>24.11</td></tr><tr><td>BPR loss</td><td>8.24</td><td>12.08</td><td>16.96</td><td>25.21</td></tr><tr><td>CROLoss1</td><td>10.20</td><td>14.03</td><td>18.63</td><td>26.06</td></tr><tr><td>CROLoss-lambda2</td><td>10.17</td><td>14.07</td><td>18.81</td><td>26.20</td></tr><tr><td rowspan="5">Taobao</td><td>cross-entropy loss</td><td>4.71</td><td>6.59</td><td>9.01</td><td>13.13</td></tr><tr><td>triplet loss</td><td>2.46</td><td>3.71</td><td>5.43</td><td>8.84</td></tr><tr><td>BPR loss</td><td>2.89</td><td>4.33</td><td>6.35</td><td>10.25</td></tr><tr><td>CROLoss</td><td>4.75</td><td>6.65</td><td>9.06</td><td>13.13</td></tr><tr><td>CROLoss-lambda4</td><td>5.27</td><td>7.35</td><td>10.01</td><td>14.57</td></tr></table>

1.Use softplus as kernel and set $_ { \alpha }$ to 1.0.   
2. Use sigmoid as kernel 1 and softplus as kernel 2 and set $_ { \alpha }$ to 1.0.   
3.Use exponential askernel and set $_ { \alpha }$ to 1.4.   
4.Use sigmoid as kernel 1 and exponential as kernel 2 and set $_ { \alpha }$ to 1.4.

# 结论：

# 理想情况：一致性越高越好

理论上，粗排的目标是拟合精排的排序逻辑。若精排绝对精准，粗排与其完全一致可确保优质候选不被遗漏，提升系统整体效率。例如，通过模型蒸馏 （精排指导粗排）或特征共享可拉齐两者打分。

# 现实约束：一致性并非绝对要求

1. 精排的局限性：精排受特征稀疏性、模型复杂度限制，预估可能存在偏差（如高估热门、低估长尾）。此时粗排若完全对齐精排，可能放大错误，反而降低效果。  
2. 角色分工差异：两者目标不同，完全一致可能导致粗排过度筛选，牺牲多样性。

 粗排：更关注候选集的覆盖能力（Recall-oriented），需快速区分用户可能喜欢与不喜欢的物品。  
 精排：聚焦头部精准排序（Precision-oriented），深入分析用户-物品交互特征。

# 3. 不同场景的权衡：

 大流量场景：一致性更关键，精排可快速修正粗排输出。  
 小流量/冷启动场景：粗排需与召回配合，通过先验知识补充精排数据不足，此时一致性可适当放宽。

在推荐系统的多级链路（召回 粗排 精排 重排）中，粗排（Pre-Ranking）承担着承上启下的关键角色，其作用与精排（Ranking）的关系既紧密又存在微妙差异。

![](images/4a05ef9cb28ce173cd38e5b2f70bb56b8297e379b61b061e882af1ff28696935.jpg)

# 粗排的核心作用

# 1. 高效过滤与候选集缩减

a. 粗排的核心目标是从召回阶段的海量候选集（通常数千至百万级）中快速筛选出几百到几千条高质量候选，大幅降低精排的计算负担。  
b. 技术实现：采用轻量模型（如双塔 DNN）或规则策略（如热度过滤），单条请求处理时间控制在 n 毫秒以内，确保高并发场景的低延迟。

# 2. 平衡效率与多样性

粗排需兼顾相关性 （保留潜在用户兴趣物品）和多样性 （避免过度依赖热门内容，为冷门物品留机会），为后续精排提供丰富输入。例如，电商平台可能通过品类配额分配（流量池）确保各品类均有曝光机会。

# 3. 缓解样本选择偏差（SSB）

粗排面对的候选集包含大量未曝光样本，而训练数据仅来自精排曝光的子集。通过引入未曝光负样本（如全局随机采样或困难负样本），可减少离线训练与线上预测的分布差异。

# 二、粗精排不一致的正向和负向影响

<table><tr><td>影响类型</td><td>正向效果</td><td>负向风险</td></tr><tr><td>效果优化</td><td>粗排补充精排未覆盖的长尾候选(如高方差物品)</td><td>粗排高分但精排低分的候选挤压优质物品曝光</td></tr><tr><td>系统效率</td><td>粗排简化模型降低延迟，保障实时性</td><td>严重不一致导致精排输入质量下降，整体效果损失</td></tr><tr><td>业务目标</td><td>粗排独立引入多样性策略，打破信息茧房</td><td>商业规则（如广告插入）在精排阶段被破坏</td></tr></table>

# 三、业界的粗精排一致性优化相关实践

# 1. 动态一致性优化

 蒸馏技术：使用精排的 Soft Label 指导粗排训练，既吸收精排知识，又保留粗排灵活性。  
 特征工程：粗排复用精排的 Embedding 层，但限制交叉特征以平衡效果与性能。

# 2. 评估指标创新

淘宝提出 ASH (All-Scenario Hitrate)，用全场景正样本（如跨场景点击/购买）评估粗排的覆盖能力，取代传统 HitRate@K。可参考：https://arxiv.org/pdf/2305.13647

# 3. 样本构造升级

如 ASMOL 框架：训练时同时输入曝光样本、精排未曝光样本、粗排未曝光样本，通过多目标学习（曝光/点击/购买）缓解 SSB 问题。

![](images/b60598ecfd54ca17bde38fb448caef0347f45c3bde8f94a47ee72559fa27a3be.jpg)  
Figure 3: The All-Scenario-based Multi-Objective Learning framework (ASMOL)in Taobao Search

# 五、核函数对比与选择指南

| 核函数 | 公式 | 对应损失 | 特点 |
|--------|------|---------|------|
| Sigmoid | σ(x) | BPR Loss | 平滑梯度，适合一般场景 |
| Softplus | log(1+exp(x)) | Softplus Loss | 对大误差惩罚更强 |
| Hinge | max(0, x+margin) | Triplet Loss | 稀疏梯度，需要设 margin |
| Exponential | exp(x) | Exponential Loss | 对困难样本极其敏感 |
| Softmax | exp(x)/Σexp(x_i) | Cross Entropy | 全局归一化，适合多负样本 |

**Lambda 梯度优化**：使用双核函数机制，核1 控制方向，核2 控制幅值，实现更精细的梯度控制。

# 六、代码实现：CROLoss 及对比实验

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class CROLoss(nn.Module):
    def __init__(self, kernel='softplus', lambda_kernel='sigmoid', alpha=1.0):
        super().__init__()
        self.kernel = kernel
        self.lambda_kernel = lambda_kernel
        self.alpha = alpha

    def apply_kernel(self, x, kernel_type):
        if kernel_type == 'sigmoid':
            return torch.sigmoid(x)
        elif kernel_type == 'softplus':
            return F.softplus(x)
        elif kernel_type == 'hinge':
            return F.relu(x + 1.0)
        elif kernel_type == 'exponential':
            return torch.exp(x.clamp(max=5.0))
        else:
            return F.softplus(x)

    def forward(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb * pos_emb, dim=-1)
        neg_score = torch.sum(user_emb.unsqueeze(1) * neg_emb, dim=-1)
        diff = neg_score - pos_score.unsqueeze(-1)
        if self.lambda_kernel:
            direction = self.apply_kernel(diff, self.kernel)
            magnitude = self.apply_kernel(diff, self.lambda_kernel)
            loss = (direction * magnitude).mean()
        else:
            loss = self.apply_kernel(diff, self.kernel).mean()
        return self.alpha * loss
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim=32, item_dim=64, embed_dim=32):
        super().__init__()
        self.user_proj = nn.Sequential(
            nn.Linear(user_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.item_proj = nn.Sequential(
            nn.Linear(item_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, user_feat, item_feat):
        return (
            F.normalize(self.user_proj(user_feat), p=2, dim=-1),
            F.normalize(self.item_proj(item_feat), p=2, dim=-1),
        )
batch_size, num_neg, embed_dim = 64, 10, 32
user_feat = torch.randn(batch_size, 32)
pos_item_feat = torch.randn(batch_size, 64)
neg_item_feat = torch.randn(batch_size, num_neg, 64)
model = TwoTowerModel()
user_proj = nn.Sequential(nn.Linear(32, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
item_proj = nn.Sequential(nn.Linear(64, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
user_emb = F.normalize(user_proj(user_feat), p=2, dim=-1)
pos_emb = F.normalize(item_proj(pos_item_feat), p=2, dim=-1)
neg_emb = F.normalize(item_proj(neg_item_feat.view(-1, 64)), p=2, dim=-1).view(batch_size, num_neg, -1)
cro_loss = CROLoss(kernel='softplus', lambda_kernel='sigmoid', alpha=1.0)
loss = cro_loss(user_emb, pos_emb, neg_emb)
print(f"CROLoss (softplus+sigmoid): {loss.item():.4f}")
cro_exp = CROLoss(kernel='exponential', lambda_kernel='sigmoid', alpha=1.4)
loss_exp = cro_exp(user_emb, pos_emb, neg_emb)
print(f"CROLoss (exp+sigmoid): {loss_exp.item():.4f}")
```

# 七、召回损失函数对比

| 损失函数 | 优化目标 | 对 Recall@N 的适配性 | 计算复杂度 |
|---------|---------|---------------------|----------|
| BPR Loss | 正负样本相对排序 | 间接优化，无 N 感知 | 低 |
| Triplet Loss | 正负样本边界 | 需手动设 margin | 低 |
| Cross Entropy | 全局分类 | 多负样本时接近 Recall 优化 | 中 |
| Sampled Softmax | 采样子集上的分类 | 负采样策略影响效果 | 中 |
| **CROLoss** | **直接优化 Recall@N** | **可定制化适配不同 N** | **中** |

# 八、常见问题与易错点

| 问题 | 说明 | 建议 |
|------|------|------|
| 核函数选择 | 不同核函数适合不同场景 | 一般推荐 softplus；需精细控制梯度用双核 lambda |
| α 参数调节 | α 控制整体损失缩放 | 从 1.0 开始，Amazon 数据集推荐 1.0，Taobao 推荐 1.4 |
| 负采样策略 | 负样本质量直接影响损失效果 | 结合随机负采样 + 流行度负采样 + 困难负采样 |
| 与 batch size 耦合 | 负样本数量影响损失值 | 固定负样本数或归一化损失 |
| 粗精排一致性过度追求 | 完全一致可能不是最优 | 粗排应保持一定灵活性，覆盖精排盲区 |

# 九、学习总结

1. CROLoss 构建了统一损失框架，通过选择不同核函数可以退化为 BPR、Triplet、CE 等传统损失
2. 核心创新在于直接针对 Recall@N 指标优化，而非间接通过排序或分类目标
3. Lambda 梯度优化使用双核函数分离梯度方向和幅值，实现更精细的训练控制
4. 在 Amazon 和 Taobao 数据集上，CROLoss 相比传统损失 Recall@N 提升 5-12%
5. 粗精排一致性需要权衡：理想情况应一致，但粗排的 Recall-oriented 角色决定了不能完全对齐精排
