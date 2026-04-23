# 面试题：电商大促 CVR 预估会出现性能显著下降是什么原因，如何优化?

面试题：电商大促 CVR 预估会出现性能显著下降是什么原因，如何优化?

电商大促期间广告 CVR（转化率）预估模型性能显著下降的现象（AUC 大幅下降，CVR模型出现严重预估偏差），主要源于以下核心原因及对应的优化思路：

# 一、CVR 模型性能下降的原因

# 1. 用户行为突变导致的分布偏移

 大促期间用户购买行为呈现剧烈波动，例如促销前用户转化率骤降（等待折扣生效），促销爆发期转化率激增。  
 传统 CVR 模型基于 i.i.d 假设（训练数据与线上数据独立同分布），但大促期间数据分布突变导致该假设失效，模型难以准确捕捉动态变化。

# 2. 延迟反馈问题加剧

 与点击行为不同，用户转化可能延迟数天甚至数周发生（如预售订单）。在大促周期内，实时训练数据无法及时获取完整转化标签，导致模型短期内的预估严重低估真实 CVR。  
 例如，促销前点击的广告可能在大促正式开始后才转化，但模型无法预知这一未来行为。

# 3. 历史数据与当前场景的分布差异

 大促期间新增广告活动和商品品类可能从未在历史数据中出现，导致传统模型缺乏对新特征的适应能力。

# 二、模型优化方案

# 基于历史数据复用的智能建模 (HDR 算法)

论文链接：https://arxiv.org/pdf/2305.12837

参考链接：KDD'23 | 转化率预估新思路：基于历史数据复用的大促转化率精准预估

核心思路：在大促前，从历史数据中筛选与当前大促分布相似的周期（如往年双 11、618 数据），通过微调（Fine-tuning）提升 CVR 模型对新数据分布的适应能力。规避实时数据延迟问题，同时通过分布校正技术对齐历史与当前数据差异。框架包含以下模块：

# 1. 自动数据检索模块（Automated Data Retrieval）

# 销售日-特征向量化：

将历史大促日表示为特征向量，特征包括：前 3 天的 CVR 均值、当日前 10 小时各大商品的品类曝光占比（动态捕捉用户兴趣迁移）

# 相似度匹配：

使用近似最近邻搜索（ANN），计算当前大促特征向量与历史向量的相似度，选取 Top-K 相似历史大促日数据。

# 2. TransBlock 微调模块

#  分层参数更新：

 基础模型（Main Model）：固定大部分参数，仅用小学习率微调，保留日常模式知识。  
 新增 TransBlock 层：在基础模型顶部叠加轻量 MLP，使用大学习率快速适配大促模式。

#  双学习率策略：

基础模型学习率 LR=1e-6，TransBlock 层 LR=1e-3，平衡稳定性与适应性。

![](images/95bc2114df241defc541f01852ba843e9e20ac4383d71c509c52e1a94653f990.jpg)

# 3. 分布偏移校正模块（Distribution Shift Correction）

# 重要性加权（Importance Weighting）：

基于重要性加权经验风险最小化框架（Importance-Weighted Empirical Risk Minimization 衡量历史样本与当前分布的差异，对检索到的样本重新加权，权重为：

$$
w (x) = \frac {\mathcal {B} _ {h} (y)}{\mathcal {B} _ {h} ^ {\prime} (y)}
$$

其中，$B _ { h } ^ { \prime } ( y )$ 代表历史数据对应当天前 10 小时的 CVR 均值，可以从历史数据中统计获得；而 $\boldsymbol { B } _ { h } ( y )$ 代表大促当天前 10 小时的真实 CVR 均值（不可实时获取），设计了一个简单的无监督预估方案对其进行估计（为了准确性，该估计不是样本级别，而是前 10 小时整体数据的 CVR，即期望）。

# 4. 在线部署

具体来说，保留原本模型的流式训练流程，在其训练完成后叠加一个微调过程，并将微调后的模型推送上线。

![](images/ee29cb092b294ffc734187828e69f9d92ee66138457c526b87b0f49f93eb860b.jpg)

在线效果：双十一大促期间，智能数据复用方案在展示广告信息流主场景全量上线，全周期（10 月 23 日～11 月 11 日）为展示大盘信息流整体带来了 $R P M + 9 \%$ ，$C V R + 1 6 \%$ ，$R O | + 1 1 \%$ 的显著提升，创造可观营收增长的同时，提升了客户体验，达成了客户侧与平台侧的双赢。

---

# 三、数学推导补充

## 1. 分布偏移的形式化定义

设日常训练数据分布为 $P_{\text{daily}}(x, y)$，大促期间数据分布为 $P_{\text{promo}}(x, y)$。当两者差异较大时：

$$
E_{(x,y) \sim P_{\text{daily}}}[\ell(f(x), y)] \neq E_{(x,y) \sim P_{\text{promo}}}[\ell(f(x), y)]
$$

模型在 $P_{\text{daily}}$ 上训练的参数 $\theta^*_{\text{daily}}$ 在 $P_{\text{promo}}$ 上不再是最优解。

## 2. 重要性加权推导

利用重要性采样，将大促分布上的期望转化为历史分布上的加权期望：

$$
E_{P_{\text{promo}}}[\ell(f(x), y)] = E_{P_{\text{hist}}}\left[\frac{P_{\text{promo}}(x)}{P_{\text{hist}}(x)} \ell(f(x), y)\right]
$$

实际中直接估计分布比值困难，HDR 简化为基于 CVR 均值的比率：

$$
w(x) = \frac{P_{\text{promo}}(y|x)}{P_{\text{hist}}(y|x)} \approx \frac{\bar{y}_{\text{promo}}}{\bar{y}_{\text{hist}}}
$$

## 3. TransBlock 的损失函数

$$
\mathcal{L} = \sum_{(x,y) \in \mathcal{D}_{\text{hist}}} w(x) \cdot \ell(f_{\theta_{\text{base}} \circ \theta_{\text{trans}}}(x), y)
$$

其中 $\theta_{\text{base}}$ 用小学习率更新，$\theta_{\text{trans}}$（TransBlock 层参数）用大学习率更新。

## 4. 延迟反馈的数学建模

设转化延迟时间为 $d$，实际观测到的标签为：

$$
y_{\text{obs}}(t) = \begin{cases} 0 & \text{if } t_{\text{click}} + d > t_{\text{observe}} \\ y_{\text{true}} & \text{otherwise} \end{cases}
$$

大促期间 $d$ 的分布右移（用户等待降价后转化），导致短期内 $y_{\text{obs}}$ 大量缺失正标签。

# 四、其他优化方法对比

| 方法 | 核心思路 | 优势 | 局限 | 适用场景 |
|------|---------|------|------|---------|
| HDR（历史数据复用） | 检索相似大促数据微调 | 直接利用历史经验 | 依赖历史数据质量 | 年度大促（双11等） |
| 延迟反馈模型（DFM） | 建模转化延迟时间分布 | 无需历史大促数据 | 需假设延迟分布 | 所有延迟反馈场景 |
| Domain Adaptation | 对齐日常与大促特征分布 | 理论完备 | 训练复杂 | 分布偏移通用场景 |
| 在线学习（FTRL） | 实时更新模型参数 | 即时适应 | 冷启动阶段效果差 | 实时性要求高 |
| 多场景模型（STAR） | 场景特定参数 + 共享参数 | 场景间知识迁移 | 需预定义场景 | 多场景并存 |

# 五、应用场景

**年度大促**：双 11、618 等大型促销活动，用户行为模式与日常差异极大。

**品类促销**：品牌日、品类专场等中大型促销，特定品类转化率波动显著。

**新品首发**：新品上架时缺乏历史转化数据，可复用类似新品的投放数据。

**跨区域投放**：不同地区大促时间不同，可利用先结束区域的数据指导后开始区域。

# 六、优缺点分析

## 优点

- 直接利用历史大促数据，无需等待新数据积累
- TransBlock 轻量微调避免了灾难性遗忘
- 重要性加权缓解分布偏移的理论保证
- 在线效果显著（RPM +9%, CVR +16%）

## 缺点

- 依赖历史大促数据的可用性和质量
- 相似日检索的特征设计需要领域经验
- 重要性加权的估计是近似的（用整体CVR均值代替样本级权重）
- TransBlock 的学习率调优需要实验验证

# 七、Python 代码实现（HDR 简化版）

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


class BaseCVRModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class TransBlock(nn.Module):
    def __init__(self, base_output_dim=1, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, base_hidden, x=None):
        return self.mlp(base_hidden)


class HDRModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        self.base_model = BaseCVRModel(input_dim, hidden_dims)
        self.trans_block = TransBlock(hidden_dim=hidden_dims[-1])
        self.extract_hidden = True

        base_layers = list(self.base_model.network.children())[:-1]
        self.feature_extractor = nn.Sequential(*base_layers)

    def forward(self, x):
        hidden = self.feature_extractor(x)
        base_out = self.base_model.network[-1](hidden)
        base_pred = torch.sigmoid(base_out)
        trans_out = self.trans_block(hidden)
        final_pred = torch.sigmoid(base_out + trans_out)
        return base_pred, final_pred, hidden


def extract_promo_features(cvr_history_3d, category_exposure_ratio):
    features = []
    for i in range(len(cvr_history_3d)):
        feat = np.concatenate([cvr_history_3d[i], category_exposure_ratio[i]])
        features.append(feat)
    return np.array(features)


def retrieve_similar_days(current_features, history_features, top_k=5):
    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(history_features)), metric='cosine').fit(history_features)
    distances, indices = nbrs.kneighbors([current_features])
    return indices[0], 1 - distances[0]


def compute_importance_weights(hist_cvrs, current_estimated_cvr):
    weights = []
    for cvr in hist_cvrs:
        w = current_estimated_cvr / (cvr + 1e-8)
        weights.append(min(w, 5.0))
    return np.array(weights)


def train_hdr():
    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 20
    n_hist = 1000
    n_promo = 200

    X_hist = np.random.randn(n_hist, input_dim).astype(np.float32)
    y_hist = (np.random.random(n_hist) < 0.03).astype(np.float32)

    X_promo = np.random.randn(n_promo, input_dim).astype(np.float32) + 0.5
    y_promo = (np.random.random(n_promo) < 0.08).astype(np.float32)

    X_hist_t = torch.FloatTensor(X_hist)
    y_hist_t = torch.FloatTensor(y_hist).unsqueeze(-1)
    X_promo_t = torch.FloatTensor(X_promo)
    y_promo_t = torch.FloatTensor(y_promo).unsqueeze(-1)

    model = HDRModel(input_dim, hidden_dims=[64, 32])
    optimizer = torch.optim.Adam([
        {"params": model.base_model.parameters(), "lr": 1e-6},
        {"params": model.trans_block.parameters(), "lr": 1e-3}
    ])

    hist_cvr = y_hist.mean()
    promo_cvr_est = 0.06
    sample_weights = compute_importance_weights(
        np.full(n_hist, hist_cvr), promo_cvr_est
    )
    sample_weights_t = torch.FloatTensor(sample_weights).unsqueeze(-1)

    for epoch in range(100):
        model.train()
        base_pred, final_pred, hidden = model(X_hist_t)
        loss = F.binary_cross_entropy(final_pred, y_hist_t, weight=sample_weights_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                _, promo_pred, _ = model(X_promo_t)
                promo_auc = roc_auc_score(y_promo, promo_pred.numpy())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Promo AUC: {promo_auc:.4f}")

    model.eval()
    with torch.no_grad():
        _, final_pred, _ = model(X_promo_t)
        final_auc = roc_auc_score(y_promo, final_pred.numpy())
        print(f"\n最终 Promo AUC: {final_auc:.4f}")
        print(f"预测CVR均值: {final_pred.mean().item():.4f}")
        print(f"实际CVR均值: {y_promo.mean():.4f}")


train_hdr()
```

# 八、常见问题与易错点

## 1. 相似日检索的特征选择

特征选择不当会导致检索到不相关的历史数据。建议包含：CVR趋势、品类分布、价格区间分布、流量规模等多维度特征。

## 2. TransBlock 梯度冲突

基础模型和 TransBlock 使用不同学习率，可能出现梯度方向冲突。如果微调后日常场景效果下降，应减小基础模型学习率或增加正则化。

## 3. 重要性权重的极端值

当历史 CVR 和当前 CVR 差异过大时，权重可能出现极端值（如10以上），导致训练不稳定。建议对权重做截断处理（如限制在 [0.1, 5.0]）。

## 4. 冷启动问题

如果某年大促是首次举办，没有可复用的历史数据。此时可退化为使用上一年最近日常数据的微调方案。

# 九、学习路径建议

1. **基础**：理解 CVR 预估的基本概念和数据偏移问题
2. **核心**：学习迁移学习（Fine-tuning、Domain Adaptation）的理论基础
3. **进阶**：掌握延迟反馈建模（DFM、FSIW）和因果推断方法
4. **拓展**：研究多场景建模（STAR、SAML）和元学习在推荐中的应用
