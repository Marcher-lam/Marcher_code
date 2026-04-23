# Uplift

论文链接：https://arxiv.org/pdf/2207.09920

DESCN（Deep Entire Space Cross Networks）是一种用于个体处理效应（ITE）估计的深度学习模型，由阿里巴巴团队提出，主要应用于电商优惠券发放等因果推断场景。

# 2.1 背景介绍

在因果推断中，个体处理效应（ITE）的准确估计是关键挑战。传统方法（如 T-Learner 或 S-Learner）存在两大问题：

处理偏差（Treatment Bias）：处理组（如收到优惠券的用户）和对照组（未收到优惠券的用户）的分布差异显著，导致模型难以学习无偏表示。  
样本不平衡（Sample Imbalance）：处理组和对照组的样本量可能极度不均衡（例如仅对少量用户发券），影响模型稳定性。

DESCN 通过全空间建模和交叉网络设计同时解决这两个问题。

 全空间网络（Entire Space Network, ESN）

 传统方法仅在处理组或对照组的子空间建模响应函数（如购买率），而 ESN 联合建模处理倾向评分、处理组响应和对照组响应，利用全样本空间的信息缓解处理偏差。  
 关键公式：

$$
\operatorname {E S T R} = P (Y, W = 1 \mid X) = \mu_ {1} (X) \cdot \pi (X),
$$

$$
\operatorname {E S C R} = P (Y, W = 0 \mid X) = \mu_ {0} (X) \cdot (1 - \pi (X)),
$$

其中 $\pi ( X )$ 是倾向评分， $\mu _ { 1 }$ 和 $\mu _ { 0 }$ 分别是处理组和对照组的响应函数。

 交叉网络（X-Network）

 引入伪处理效应（Pseudo Treatment Effect, PTE）作为中间变量，连接处理组和对照组的响应函数通过多任务学习平衡样本不平衡问题。  
 通过交叉计算反事实结果：

$$
\mu_ {1} ^ {\prime} = \sigma \left(\sigma^ {- 1} \left(\mu_ {0}\right) + \sigma^ {- 1} \left(\tau^ {\prime}\right)\right),
$$

$$
\mu_ {0} ^ {\prime} = \sigma \left(\sigma^ {- 1} \left(\mu_ {1}\right) - \sigma^ {- 1} \left(\tau^ {\prime}\right)\right),
$$

其中 $\tau ^ { \prime }$ 是伪处理效应， $\sigma$ 为 Sigmoid 函数。

# 2.2 模型架构

DESCN 由两部分组成：

 ESN 模块：

 输入：用户特征 。

 输出：倾向评分 $\pi ( X )$ 、处理组响应 $\mu _ { 1 } ( X )$ 、对照组响应 $\mu _ { 0 } ( X )$   
 通过乘法节点计算 ESTR 和 ESCR，损失函数包含 $L _ { \pi }$ $L _ { \mathrm { E S T R } }$ $L _ { \mathrm { E S C R } }$ 。

#  X-Network 模块：

 在 ESN 基础上增加 PTE 网络，生成伪处理效应 。 $\tau ^ { \prime } ( X )$   
 通过交叉计算得到 $\mu _ { 1 } ^ { \prime }$ 和 $\mu _ { 0 } ^ { \prime }$ ，损失函数增加 $L _ { \mathrm { C r o s s } \mathrm { T R } }$ 和 $L _ { \mathrm { C r o s s } } \mathrm { C R }$

![](images/f206a91da3a5156d7b5d205e04a4280e6755aa66022d0be012687ccc1afc19bb.jpg)  
(a) Entire Space Network (ESN)

![](images/f8ff56a6916d329dc68001c20705498b3c56a858a288313f4cd598ef80dcdbee.jpg)  
(b) X-network

![](images/b1b0caeb932466233a54b974470a0989bb24fcb4115eed09680df977732d6a04.jpg)  
(c) Deep Entire Space Cross Networks (DESCN)

# 2.3 核心数学公式

 ITE 定义：

$$
\tau (X) = \mathbb {E} [ Y (1) - Y (0) \mid X ] = \mu_ {1} (X) - \mu_ {0} (X)
$$

 损失函数：

DESCN 的总损失为加权和：

$$
\begin{array}{l} L _ {\text {D E S C N}} = \alpha L _ {\pi} + \beta_ {1} L _ {\text {E S T R}} + \beta_ {0} L _ {\text {E S C R}} \\ + \gamma_ {1} L _ {\text {C r o s s T R}} + \gamma_ {0} L _ {\text {C r o s s C R}}. \\ \end{array}
$$

<table><tr><td>倾向得分损失</td><td>Lπ = 1/n ∑i l(wi, π(xi))</td></tr><tr><td>全空间处理组响应损失</td><td>LESTR = 1/n ∑i l(yi&amp;wi, μ1(xi) · π(xi))</td></tr><tr><td>全空间对照组响应损失</td><td>LESCR = 1/n ∑i l(yi&amp;(1-wi), μ0(xi) · (1-π(xi)))</td></tr><tr><td>交叉处理组响应损失</td><td>LCrossTR = 1/T ∑i∈T l(yi, μ1&#x27;(xi))其中, μ1&#x27;(xi) = σ(σ-1(μ0(xi)) + σ-1(τ&#x27;(xi)))</td></tr><tr><td>交叉对照组响应损失</td><td>LCrossCR = 1/C ∑i∈C l(yi, μ0&#x27;(xi))其中, μ0&#x27;(xi) = σ(σ-1(μ1(xi)) - σ-1(τ&#x27;(xi)))</td></tr></table>

#  反事实估计：

通过 PTE连接双响应函数：

$$
\hat {\mu} _ {1} ^ {\prime} = \sigma \left(\sigma^ {- 1} \left(\hat {\mu} _ {0}\right) + \sigma^ {- 1} \left(\hat {\tau} ^ {\prime}\right)\right), \quad \hat {\mu} _ {0} ^ {\prime} = \sigma \left(\sigma^ {- 1} \left(\hat {\mu} _ {1}\right) - \sigma^ {- 1} \left(\hat {\tau} ^ {\prime}\right)\right)
$$

# 2.4 输入输出形式

 输入：用户特征 （如历史购买频次、活跃度）、处理指示 $W \in \{ 0 , 1 \} _ { \L }$ （是否发券）、响应变量 $Y \in \{ 0 , 1 \}$ （是否购买）。  
输出：

 直接输出：倾向评分 ${ \hat { \pi } } ( X )$ 、处理组响应 $\hat { \mu } _ { 1 } ( X )$ 、对照组响应 ${ \hat { \mu } } _ { 0 } ( X )$ 。  
 最终目标：ITE 估计值 $\hat { \tau } ( X ) = \hat { \mu } _ { 1 } ( X ) - \hat { \mu } _ { 0 } ( X ) _ { \circ }$ 。

# 2.5 样本组织形式（电商发券场景为例）

#  训练数据：

1. 处理组（ $\pmb { W } \mathbf { = } \pmb { \mathrm { 1 } }$ ）：被发放优惠券的用户，特征 X 可能包含低活跃度等偏置特征。  
2. 对照组（ $\scriptstyle \pmb { W } = \pmb { 0 }$ ）：未收到优惠券的用户，通常样本量更大。  
3. 响应变量 Y：是否在活动期内购买。

 测试数据：使用随机实验（RCT）数据评估模型，避免选择偏置。  
 关键处理：训练集包含强处理偏置（如仅对不活跃用户发券），测试集为随机样本，模拟现实场景中"训练偏置、测试无偏"的需求。

# 三、DragonNet 和 DESCN 对比

<table><tr><td>对比维度</td><td>DragonNet</td><td>DESCN</td></tr><tr><td>核心创新</td><td>端到端的三头网络结构,联合预测倾向评分、处理组结果和对照组结果</td><td>全空间网络(ESN)+交叉网络(X-Network),通过多任务学习集成倾向评分、响应函数和伪处理效应</td></tr><tr><td>模型架构</td><td>共享底层表示,三个输出头分别对应倾向评分 (π)、处理组响应(μ1)和对照组响应(μ0)</td><td>分ESN和X-Network两部分:ESN联合建模π、μ1、μ0; X-Network引入伪处理效应(τ&#x27;)连接双响应函数</td></tr><tr><td>处理的关键问题</td><td>混淆变量偏置(通过倾向评分调整)、表示学习平衡</td><td>治疗偏差(因非随机分配导致分布差异)、样本不平衡(处理组/对照组大小不均)</td></tr><tr><td>数学基础</td><td>基于倾向加权的损失函数,最小化预测结果与真实结果的误差</td><td>全空间概率分解:ESTR=μ1·π,ESCR=μ0·(1-π);引入伪处理效应τ&#x27;进行反事实计算</td></tr><tr><td>输入形式</td><td>特征向量X,处理指示W,响应Y</td><td>同左</td></tr><tr><td>输出形式</td><td>直接输出π(X)、μ1(X)、μ0(X),ITE推导为τ=μ1-μ0</tdtd></tr></table>

# 四、Uplift 建模方法全景对比

| 方法 | 类型 | 核心思想 | 优势 | 劣势 |
|------|------|---------|------|------|
| T-Learner | 元学习 | 分别对处理组和对照组训练独立模型 | 简单直观 | 模型可能学习差异而非因果效应 |
| S-Learner | 元学习 | 将处理变量作为特征输入单一模型 | 利用全部数据 | 处理效应可能被其他特征淹没 |
| X-Learner | 元学习 | 结合 T-Learner 和倾向评分的混合方法 | 平衡偏差和方差 | 实现复杂，依赖倾向评分质量 |
| DragonNet | 深度学习 | 三头网络联合学习倾向评分和响应函数 | 端到端训练 | 仅关注混淆变量偏置 |
| DESCN | 深度学习 | 全空间建模 + 交叉网络 | 同时解决处理偏差和样本不平衡 | 训练复杂，多损失函数调参困难 |
| Causal Forest | 树模型 | 因果树直接估计异质性处理效应 | 非参数化、可解释 | 大数据集计算慢 |

# 五、代码实现：DESCN 核心模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class SharedBottom(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            prev_dim = h_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class DESCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        self.shared_bottom = SharedBottom(input_dim, hidden_dims)
        bottom_dim = hidden_dims[-1]
        self.propensity_head = nn.Sequential(
            nn.Linear(bottom_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.treatment_response_head = nn.Sequential(
            nn.Linear(bottom_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.control_response_head = nn.Sequential(
            nn.Linear(bottom_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.pte_head = nn.Sequential(
            nn.Linear(bottom_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared_bottom(x)
        pi = self.propensity_head(features)
        mu1 = self.treatment_response_head(features)
        mu0 = self.control_response_head(features)
        tau_prime = self.pte_head(features)
        mu1_cross = torch.sigmoid(
            torch.logit(mu1.clamp(1e-6, 1 - 1e-6))
            - torch.logit(tau_prime.clamp(1e-6, 1 - 1e-6))
        )
        mu0_cross = torch.sigmoid(
            torch.logit(mu0.clamp(1e-6, 1 - 1e-6))
            + torch.logit(tau_prime.clamp(1e-6, 1 - 1e-6))
        )
        return pi, mu1, mu0, tau_prime, mu1_cross, mu0_cross
def descn_loss(pi, mu1, mu0, mu1_cross, mu0_cross, treatment, response,
               alpha=1.0, beta1=1.0, beta0=1.0, gamma1=1.0, gamma0=1.0):
    bce = nn.BCELoss()
    l_pi = bce(pi.squeeze(-1), treatment.float())
    estr = mu1.squeeze(-1) * pi.squeeze(-1)
    escr = mu0.squeeze(-1) * (1 - pi.squeeze(-1))
    l_estr = F.mse_loss(estr, response.float() * treatment.float())
    l_escr = F.mse_loss(escr, response.float() * (1 - treatment.float()))
    treat_mask = treatment.bool()
    control_mask = ~treatment.bool()
    l_cross_tr = torch.tensor(0.0)
    l_cross_cr = torch.tensor(0.0)
    if treat_mask.sum() > 0:
        l_cross_tr = F.mse_loss(
            mu1_cross[treat_mask].squeeze(-1), response[treat_mask].float()
        )
    if control_mask.sum() > 0:
        l_cross_cr = F.mse_loss(
            mu0_cross[control_mask].squeeze(-1), response[control_mask].float()
        )
    total = alpha * l_pi + beta1 * l_estr + beta0 * l_escr
    total = total + gamma1 * l_cross_tr + gamma0 * l_cross_cr
    return total
input_dim = 50
model = DESCNModel(input_dim)
batch_size = 256
x = torch.randn(batch_size, input_dim)
treatment = torch.randint(0, 2, (batch_size,))
response = torch.randint(0, 2, (batch_size,))
pi, mu1, mu0, tau_prime, mu1_cross, mu0_cross = model(x)
loss = descn_loss(pi, mu1, mu0, mu1_cross, mu0_cross, treatment, response)
ite = mu1 - mu0
print(f"倾向评分: {pi[:3].squeeze().tolist()}")
print(f"ITE估计: {ite[:3].squeeze().tolist()}")
print(f"总损失: {loss.item():.4f}")
```

# 六、Uplift 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| AUUC | Uplift 曲线下面积 | 衡量模型排序 uplift 的能力，值越大越好 |
| Qini Curve | 累积 uplift 随人群比例变化的曲线 | 可视化模型在不同干预比例下的增益 |
| ATU | 处理组上的平均处理效应 | 评估模型在处理组上的预测准确性 |

# 七、常见问题与易错点

| 问题 | 说明 | 建议 |
|------|------|------|
| 选择偏置 | 观测数据中处理分配非随机 | 必须使用倾向评分或全空间方法缓解 |
| 反事实无法观测 | 同一用户不可能同时观察到处理和对照结果 | 通过交叉网络近似反事实 |
| Sigmoid 数值不稳定 | logit 变换在 0 和 1 附近不稳定 | 使用 clamp 限制范围 |
| 损失权重调节 | 多损失函数权重敏感 | 先固定 α=1，逐步调节其他权重 |

# 八、学习总结

1. DESCN 通过全空间建模（ESN）解决处理偏差，通过交叉网络（X-Network）解决样本不平衡
2. 伪处理效应（PTE）作为中间变量连接处理组和对照组的响应函数，实现反事实估计
3. 多任务联合损失函数同时优化倾向评分、响应函数和交叉预测
4. ITE 估计值 $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$ 直接用于决策哪些用户值得干预
5. 实际应用中需注意选择偏置、数值稳定性和损失权重调节
