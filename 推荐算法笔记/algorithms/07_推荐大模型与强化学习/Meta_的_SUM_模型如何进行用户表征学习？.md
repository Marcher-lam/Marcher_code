# 面试题：Meta 的 SUM 模型如何进行用户表征学习？

面试题：Meta 的 SUM 模型如何进行用户表征学习？

Meta 的 SUM（Scaling User Modeling）模型是一种针对大规模在线广告个性化设计的用户表征框架，其核心目标是通过高效的嵌入学习和实时更新机制，解决传统推荐系统中的特征冗余、数据稀疏及模型定制化等问题。

# 一、提出的背景

# 业务需求与系统复杂性

Meta 的广告系统包含数百个不同规格的排序模型，每天需处理数千亿次用户请求。传统方法中，每个模型独立学习用户表征，导致以下问题：

 次优表征：模型独立学习用户特征时效果较差，难以捕捉全局用户兴趣；  
 特征冗余：不同模型重复处理相似用户特征，造成存储和计算资源浪费（如存储开销增加 $40 \%$ ）；  
 数据稀疏性：小众领域模型因训练数据不足，难以深入理解用户行为。

# 2. 工程约束

在线广告系统对延迟（如 30ms内响应）和吞吐量（千亿级请求）的严苛要求，限制了模型复杂性和实时更新能力。

# 二、解决的问题

SUM框架旨在：

 统一用户表征共享：通过上游模型生成紧凑的用户 embedding 嵌入，供下游数百个广告模型复用，避免重复特征处理；  
 动态特征适应：实时更新用户 embedding 以响应用户行为变化（如新用户 ID 引入）；  
平衡模型性能与效率：在有限延迟预算下，支持复杂用户建模。

# 三、核心创新点

# 1. 分层特征压缩与残差学习

 用户塔金字塔架构：通过多级交互模块（Interaction Module）逐步压缩上千维稀疏特征，结合残差连接保留原始信息（如稀疏 ID 特征压缩为低维密集嵌入）；  
 混合塔轻量化设计：仅接收用户塔输出的嵌入与广告特征交互，避免重复输入原始用户数据。

# 2. 异步在线服务系统（SOAP）

 写入-读取分离：用户请求触发 embedding 实时生成并存储，客户端异步读取历史 embedding，绕过复杂模型的延迟瓶颈；  
 缓存与动态更新：高频用户 embedding 缓存减少重复计算，同时支持用户特征动态更新（如每数小时循环训练）。

# 3. 多任务联合优化

结合点击率（CTR）、转化率（CVR）等多目标损失函数，动态调整任务权重以适配不同业务场景。

# 四、模型原理详解

# 1. 模型架构

SUM 基于双塔 DLRM 架构，分为用户塔（User Tower）和混合塔（Mix Tower）：

![](images/52c8e11e7beef61e14dc9933b194c686ac53b8cf616714619790bba7549b9e25.jpg)

# 用户塔：

 输入特征处理：用户稀疏特征（如 ID、页面访问记录）和密集特征（如点击频率）分别嵌入后融合；  
 交互模块堆叠：通过金字塔结构逐步压缩特征（例如从1000维稀疏特征 $\multimap$ 维密集嵌入），每个模块包含注意力压缩、残差连接和多层感知机（MLP）；  
 输出：生成低维统一用户嵌入（如多个 32 维向量）。

# 混合塔：

 跨模态交互：将用户嵌入与广告特征输入深层交叉网络（DCN）或MLP-Mixer，捕捉高阶特征交互（如用户兴趣与广告内容的匹配度）；  
 监督信号：通过多任务交叉熵损失优化广告点击率等目标。

# 2. 训练与推理机制

增强循环训练：定期用平均池化聚合用户近期行为，更新嵌入以应对数据分布漂移；  
 在线推理（SOAP）：仅部署用户塔进行实时嵌入生成，混合塔离线预计算，确保 30ms 内响

# 总结：

SUM 通过分层特征压缩、异步服务系统和多任务联合优化，解决了大规模广告系统中用户表征共享与动态更新的核心难题，兼顾模型性能与工程效率，成为 Meta 广告生态的核心基础设施

---

# 五、数学公式与核心推导

## 1. 金字塔特征压缩过程

用户塔中，第 $l$ 层交互模块的输出为：

$$
\mathbf{h}^{(l)} = \text{MLP}^{(l)}\left(\text{Attention}^{(l)}(\mathbf{h}^{(l-1)})\right) + \mathbf{h}^{(l-1)}
$$

其中 $\mathbf{h}^{(0)}$ 为原始融合嵌入，残差连接确保低层信息不会在压缩过程中丢失。

## 2. 多任务损失函数

SUM 的总损失为多任务加权组合：

$$
\mathcal{L}_{\text{total}} = \sum_{t \in \{CTR, CVR, ...\}} w_t \cdot \mathcal{L}_t
$$

其中 $\mathcal{L}_t$ 为第 $t$ 个任务的交叉熵损失，$w_t$ 为动态调整的任务权重。

## 3. 注意力压缩机制

对于输入特征集合 $\{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_N\}$，注意力压缩将其聚合成固定维度向量：

$$
\mathbf{c} = \sum_{i=1}^{N} \alpha_i \mathbf{e}_i, \quad \alpha_i = \frac{\exp(\mathbf{q}^T \mathbf{e}_i)}{\sum_{j=1}^{N} \exp(\mathbf{q}^T \mathbf{e}_j)}
$$

其中 $\mathbf{q}$ 为可学习的查询向量，动态决定每个特征的重要性。

# 六、应用场景

**在线广告系统**：Meta 内部数百个广告排序模型共享同一套用户表征，涵盖信息流广告、搜索广告、Stories 广告等不同业务线。

**电商推荐**：统一的用户 embedding 可以同时服务商品推荐、店铺推荐、活动推荐等多个下游任务，避免每个业务独立建设用户画像。

**内容平台**：短视频、图文内容分发场景中，通过共享用户表征实现跨品类的内容推荐。

**实时竞价广告（RTB）**：SOAP 机制确保在毫秒级响应时间内提供高质量用户特征，支撑竞价决策。

# 七、优缺点分析

## 优点

- **表征共享高效**：一次训练，数百个模型复用，大幅降低存储和计算开销
- **实时更新能力**：SOAP 异步架构支持分钟级用户嵌入更新，适应行为变化
- **多任务增益**：联合训练使稀疏任务（如CVR）从密集任务（如CTR）中受益
- **延迟可控**：用户塔与混合塔分离部署，确保在线推理延迟稳定在30ms以内
- **可扩展性强**：新增下游模型无需重新训练用户表征，直接接入即可

## 缺点

- **上游瓶颈风险**：用户表征作为单一上游，若质量下降则影响所有下游模型
- **灵活性受限**：下游模型无法根据自身需求调整用户表征的粒度或维度
- **冷启动挑战**：新用户缺乏历史行为，初始嵌入质量有限
- **训练成本高**：多任务联合训练需要大量标注数据和计算资源
- **缓存一致性**：高频更新场景下，缓存与最新嵌入之间可能存在延迟

# 八、与同类方法对比

| 方法 | 表征共享 | 实时更新 | 多任务支持 | 延迟控制 | 典型场景 |
|------|---------|---------|-----------|---------|---------|
| SUM（Meta） | 支持 | SOAP异步 | 支持 | 30ms | 大规模广告 |
| DLRM（Meta） | 不支持 | 流式训练 | 部分支持 | 中等 | 排序模型 |
| MOUR（Google） | 部分共享 | 定期更新 | 支持 | 较高 | 搜索推荐 |
| PEP（Pinterest） | 有限 | 不支持 | 不支持 | 低 | 图文推荐 |

# 九、Python 代码实现（简化版 SUM 用户塔）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionModule(nn.Module):
    def __init__(self, input_dim, compressed_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, compressed_dim)
        self.key = nn.Linear(input_dim, compressed_dim)
        self.value = nn.Linear(input_dim, compressed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        q = self.query(x).unsqueeze(-2)
        k = self.key(x).unsqueeze(-2)
        v = self.value(x).unsqueeze(-2)
        attn = F.softmax(q @ k.transpose(-2, -1) / (x.size(-1) ** 0.5), dim=-1)
        compressed = (attn @ v).squeeze(-2)
        residual = self.mlp(compressed) + x
        return self.layer_norm(residual)


class UserTower(nn.Module):
    def __init__(self, num_sparse_features, sparse_dim, dense_dim, embedding_dim=32, num_layers=3):
        super().__init__()
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(num_sparse_features[i], sparse_dim) for i in range(len(num_sparse_features))
        ])
        total_dim = len(num_sparse_features) * sparse_dim + dense_dim
        self.interaction_modules = nn.ModuleList([
            InteractionModule(total_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(total_dim, embedding_dim)

    def forward(self, sparse_ids, dense_features):
        sparse_embeds = [emb(sparse_ids[:, i]) for i, emb in enumerate(self.sparse_embeddings)]
        x = torch.cat(sparse_embeds + [dense_features], dim=-1)
        for module in self.interaction_modules:
            x = module(x)
        return self.output_layer(x)


class MixTower(nn.Module):
    def __init__(self, user_dim, ad_dim, hidden_dim=64):
        super().__init__()
        self.cross_layer = nn.Sequential(
            nn.Linear(user_dim + ad_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_embed, ad_features):
        x = torch.cat([user_embed, ad_features], dim=-1)
        return torch.sigmoid(self.cross_layer(x))


class SUMModel(nn.Module):
    def __init__(self, user_tower, mix_tower):
        super().__init__()
        self.user_tower = user_tower
        self.mix_tower = mix_tower

    def forward(self, sparse_ids, dense_features, ad_features):
        user_embed = self.user_tower(sparse_ids, dense_features)
        pctr = self.mix_tower(user_embed, ad_features)
        return pctr, user_embed


num_features = [1000, 500, 200]
user_tower = UserTower(num_features, sparse_dim=16, dense_dim=8, embedding_dim=32, num_layers=3)
mix_tower = MixTower(user_dim=32, ad_dim=16)
model = SUMModel(user_tower, mix_tower)

sparse_ids = torch.randint(0, 100, (4, 3))
dense_features = torch.randn(4, 8)
ad_features = torch.randn(4, 16)

pctr, user_embed = model(sparse_ids, dense_features, ad_features)
print(f"预测CTR: {pctr.squeeze().detach().numpy()}")
print(f"用户嵌入维度: {user_embed.shape}")
```

# 十、常见问题与易错点

## 1. 用户嵌入维度选择

维度过低（如8维）会导致信息瓶颈，过高（如256维）增加存储和计算开销。实践中建议从32维起步，通过消融实验确定最佳维度。

## 2. 残差连接的必要性

去掉残差连接后，深层交互模块容易出现梯度消失，导致底层特征信息丢失。务必保留残差路径。

## 3. SOAP 缓存过期策略

缓存时间过长会导致用户嵌入无法及时反映最新行为，过短则增加计算压力。Meta 实践表明，2-4小时的缓存周期在大多数场景下效果最佳。

## 4. 多任务权重调节

任务权重 $w_t$ 若设置不当，可能导致主导任务压制辅助任务。建议使用 Uncertainty Weighting 或 GradNorm 等自动权重调节方法。

# 十一、学习路径建议

1. **基础**：先掌握双塔模型（Two-Tower）和 DLRM 架构原理
2. **进阶**：学习多任务学习（MMOE、PLE）和注意力机制
3. **深入**：理解在线 Serving 系统（缓存、异步更新）的工程实现
4. **拓展**：研究 Google MOUR、Pinterest PEP 等同类工业界方案
