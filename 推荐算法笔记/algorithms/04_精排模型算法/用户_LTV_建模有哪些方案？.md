# 面试题：用户 LTV 建模有哪些方案？

# 面试题：用户 LTV 建模有哪些方案？

广告推荐中的用户 LTV（生命周期价值）建模旨在预测用户未来可能带来的收益（例如游戏付费金额），以优化广告投放策略。LTV 预估存在着数据稀疏、零膨胀（zero-inflated）和长尾分布（long-tailed distribution）等挑战，下面这个表格汇总了几个业界有代表性的 LTV 建模方案：

<table><tr><td>方案</td><td>机构</td><td>关键创新点</td><td>论文链接</td></tr><tr><td>ZILN (Zero-Inflated Lognormal)</td><td>Google, 2019</td><td>使用零膨胀对数正态分布拟合LTV，DNN输出分布参数，损失函数为负对数似然，端到端建模付费概率与金额。</td><td>https://arxiv.org/pdf/1912.07753</td></tr><tr><td>ODMN &amp; MDME</td><td>Kuaishou, 2022</td><td>ODMN建模多时间跨度LTV间的有序依赖；MDME用分而治之思想（分桶采样）处理极不平衡分布。</td><td>https://arxiv.org/pdf/2208.13358</td></tr><tr><td>ExpLTV</td><td>Tencent, 2023</td><td>创新性将大R识别（Game Whale Detection）作为门控网络，引导不同特质的用户进入专属的LTV专家进行预估。</td><td>https://arxiv.org/pdf/2308.12729</td></tr><tr><td>CMLTV (Contrastive Multi-view)</td><td>Huawei, 2023</td><td>对比学习多视角框架，集成多个异构回归器（分布/对数/分类），提升模型鲁棒性，为即插即用模块。</td><td>https://arxiv.org/pdf/2306.14400</td></tr></table>

# ZILN：概率化建模的开创者

ZILN模型为 LTV预估提供了一种优雅的概率化建模思路，其核心思想是对LTV的真实分布做出合理的概率假设。它认为 LTV数据来源于一个混合过程：大部分用户不付费（产生零值），而付费用户的金额服从对数正态分布。

 模型结构：一个深度神经网络（DNN）同时输出三个参数：付费概率 p、对数正态分布的均值 $\mu$ 和标准差 $\sigma$ 。激活函数通常为 Sigmoid (for p), Identity (for μ), Softplus (for σ)。  
 损失函数：摒弃传统的 MSE 损失，采用基于 ZILN 分布的负对数似然损失。这使得模型训练更稳定，对高 LTV 的异常值不敏感。  
 预估值：预测时，使用付费概率乘以付费金额的期望，即 $p \cdot e ^ { (\mu + \sigma ^ { 2 } / 2) }$   
 适用场景：付费行为相对规范、认可概率化建模的业务。其缺点在于依赖"付费金额服从对数正态分布"的强假设，在真实复杂场景中可能不总是成立。

# ODMN-MDME：工业级的 LTV 预估系统方案

快手的ODMN-MDME框架是针对超大规模用户场景下 LTV分布极度不平衡和多时间跨度预测一致性问题的一套系统性、工业级的解决方案。

 MDME (多分布多专家)：核心是"分而治之"。它将极度不平衡的 LTV分布先按值域切分为几个子分布（例如，零值、低价值、高价值），再在每个子分布内进行分桶，最后在桶内进行偏差回归。这种"分类 + 排序 + 回归"的级联结构极大降低了直接回归高难度长尾分布的复杂度。  
 ODMN (序依赖单调网络)：用于处理多时间跨度（如 ltv7, ltv30, ltv90）的预估。它通过一个单调单元显式地建模不同跨度任务间的有序依赖关系（即保证 $\hat{y}_7 \le \hat{y}_{30} \le \hat{y}_{90}$ ），利用更易预测的短期 LTV辅助长期 LTV的学习，并保证了业务逻辑上的严格一致性。

# ExpLTV：聚焦"大 R 用户"的价值挖掘

腾讯 ExpLTV 的核心洞察在于，极少数的"大 R 用户"贡献了绝大部分收入，而他们的行为模式与普通用户差异显著。传统单一模型难以同时处理好普通用户和大 R 用户的预估。

 专家路由与门控网络：模型创新地设计了一个大 R 用户检测器，该检测器作为一个门控网络，为每个用户计算其属于大R用户的概率。根据这个概率，模型动态地将用户路由到不同的"LTV专家"网络（例如，一个专家擅长处理普通用户，另一个专家专注处理大 R 用户）。  
 解决选择偏差与数据稀疏：通过构建"转化 → 购买 → 大R用户"的行为序列，并引入购买率预测等辅助任务，在全量用户空间进行训练，有效缓解了传统方法只在付费用户上训练带来的样本选择偏差（SSB）和数据稀疏（DS）问题。

# CMLTV：集成与对比学习的视角

华为的 CMLTV 框架更像一个"即插即用"的增强模块，旨在通过模型集成和对比学习来提升基模型的鲁棒性和泛化能力。

 多视角预估：框架集成了三种异构的回归器：基于分布的（如伽马分布）、基于对数的、基于分类分桶的。它们从不同视角对样本的 LTV 进行分析建模，具有很强的互补性。  
 对比学习：在 Batch 样本间实施对比学习。例如，拉近高 LTV 用户与高付费概率用户的表征，拉远低 LTV 用户与高付费概率用户的表征，从而在批次内挖掘样本间的内在相关性，减轻对数据丰富性的依赖。

# LTV 评估指标

在评估模型时，除了通用的 NRMSE（归一化均方根误差），应特别关注排序能力指标：

 基尼系数（Gini）：源于洛伦兹曲线，是评估模型将高价值用户排在低价值用户之前的能力标准。其归一化版本（NormalizedGini）便于跨模型比较，取值为 0-1 之间，与 AUC 的换算关系 (1 + Norm_Gini) / 2 ≈ AUC。  
 互基尼系数（Mutual Gini）：快手提出的新指标，专门衡量预测值与真实值之间的分布差异，更能反映模型拟合不平衡分布的能力。

# 方案选择

 追求理论优雅与快速落地：ZILN 是一个非常好的起点，它提供了概率化框架，易于理解和实现。  
 应对超大规模数据与复杂业务逻辑：如果需要同时预测多个时间跨度且要求严格满足有序性，ODMN-MDME 是经过亿级用户验证的工业级方案之一。  
 业务中"大 R 用户"效应显著：在游戏、在线娱乐等大 R 用户贡献突出的行业，ExpLTV 的思路比较有借鉴意义，可以显著提升对高价值用户的识别和预估精度。  
 提升现有模型的泛化能力：可将 CMLTV 中的多视角和对比学习思路作为增强模块融入现有基线，或尝试模型集成。

# ZILN 模型完整实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZILNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()])
            prev_dim = h_dim

        self.shared_net = nn.Sequential(*layers)
        self.prob_head = nn.Linear(prev_dim, 1)
        self.mu_head = nn.Linear(prev_dim, 1)
        self.sigma_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        shared = self.shared_net(x)
        prob = torch.sigmoid(self.prob_head(shared))
        mu = self.mu_head(shared)
        sigma = F.softplus(self.sigma_head(shared)) + 1e-6
        return prob, mu, sigma

def ziln_loss(prob, mu, sigma, target, epsilon=1e-8):
    is_zero = (target == 0).float()
    is_nonzero = 1.0 - is_zero

    log_prob_zero = torch.log(prob + epsilon) * is_zero
    log_prob_nonzero = torch.log(1.0 - prob + epsilon) * is_nonzero

    log_target = torch.log(target + epsilon)
    lognormal_ll = -0.5 * torch.log(2 * np.pi * sigma ** 2 + epsilon) \
                   - ((log_target - mu) ** 2) / (2 * sigma ** 2 + epsilon) \
                   - log_target

    loss = -(log_prob_zero + (log_prob_nonzero + lognormal_ll * is_nonzero))
    return loss.mean()

def ziln_predict(prob, mu, sigma):
    return prob * torch.exp(mu + sigma ** 2 / 2)
```

# ZILN 训练与评估代码

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_ziln(model, train_loader, optimizer, epochs=50, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            prob, mu, sigma = model(batch_x)
            loss = ziln_loss(prob, mu, sigma, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

def evaluate_ltv(model, test_loader, device='cpu'):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            prob, mu, sigma = model(batch_x)
            pred = ziln_predict(prob, mu, sigma)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    nrmse = rmse / (targets.max() - targets.min() + 1e-8)

    sorted_idx = np.argsort(-targets)
    sorted_targets = targets[sorted_idx]
    sorted_preds = preds[sorted_idx]
    cum_targets = np.cumsum(sorted_targets) / np.sum(sorted_targets)
    cum_random = np.arange(1, len(targets) + 1) / len(targets)
    gini = 2 * np.sum(cum_targets - cum_random) / len(targets)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, Gini: {gini:.4f}")
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "gini": gini}
```

# 各方案详细对比

| 维度 | ZILN | ODMN-MDME | ExpLTV | CMLTV |
|------|------|-----------|--------|-------|
| 建模方式 | 概率分布 | 分而治之+级联 | MoE门控路由 | 多视角集成 |
| 零值处理 | 显式建模（混合分布） | 子分布独立建模 | 全量用户训练 | 分布回归器处理 |
| 长尾处理 | 对数正态假设 | 分桶+偏差回归 | 专家网络 | 对比学习 |
| 多时间跨度 | 不支持 | ODMN有序约束 | 不支持 | 不支持 |
| 实现复杂度 | 低 | 高 | 中 | 中 |
| 训练数据要求 | 中等 | 大规模 | 大规模+标注 | 任意规模 |
| 生产部署难度 | 低 | 高 | 中 | 中 |
| 效果稳定性 | 中 | 高 | 高 | 中高 |

# 生产部署建议

1. **特征选择**：LTV 模型特征应包含付费相关（历史付费金额/频次）、活跃相关（登录天数/时长）和用户属性（年龄/地域/设备）。
2. **样本构建**：正样本为付费用户，但必须包含零付费用户参与训练，否则预估偏高。
3. **在线推理优化**：LTV 模型通常不直接在线推理，而是离线批量预估后写入特征存储，在线直接读取。
4. **模型更新频率**：LTV 是长期累积值，建议按天/周更新模型，不需要实时更新。
5. **业务校准**：预估值需定期与真实值对比校准，偏差超过阈值时触发模型重训练。

# 常见问题与易错点

1. **对数正态假设不成立**：真实 LTV 分布可能更复杂，需先做分布检验（Kolmogorov-Smirnov 检验）。
2. **标签定义时间窗口**：LTV 标签需明确时间窗口（如 LTV7/LTV30/LTV90），窗口选择影响模型效果。
3. **归因问题**：LTV 归因到具体渠道或广告位时，需考虑归因窗口和归因模型（首次/末次/线性）。
4. **零值过多导致欠拟合**：零值比例过高时（如 >95%），建议先做付费预测分类，再对付费用户做金额回归。

# 学习路径建议

1. 理解 LTV 的业务定义和建模挑战
2. 学习概率分布建模（对数正态、伽马分布）
3. 掌握 ZILN 损失函数的推导与实现
4. 研究长尾分布的处理策略（分桶、MoE）
5. 实践 LTV 模型的评估与校准方法
