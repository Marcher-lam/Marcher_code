# ESCM²（Entire Space Counterfactual Multi-Task Model）

## 1. 算法基础认知

ESCM² 是阿里在 ESMM 基础上提出的因果推断多任务学习模型，用于解决推荐系统中点击后转化（CVR）预估的样本选择偏差和数据稀疏问题。ESMM 通过全局空间建模部分缓解了这些问题，但存在**固有估计偏差（IEB）**和**潜在独立性优先（PIP）**两大缺陷，ESCM² 引入因果推断中的逆倾向加权和双重稳健估计来彻底消除这些偏差。

## 2. 详细原理

### 2.1 ESMM 的局限

ESMM 将 pCTCVR 分解为 pCTR × pCVR，在全样本空间上训练：

$$pCTCVR = pCTR \times pCVR$$

但存在两个问题：

1. **IEB（Inherent Estimation Bias）**：ESMM 的 pCVR 是通过 pCTCVR / pCTR 间接得到的，该分解假设点击和转化独立，导致 pCVR 估计有偏
2. **PIP（Potential Independence Priority）**：乘法结构使得 pCTR 主导梯度，pCVR 的学习信号被淹没

### 2.2 逆倾向加权（IPW）

核心思想：将点击视为"处理（Treatment）"，对未点击样本赋予反事实权重，实现全空间无偏估计。

通过 CTR 预估值调整点击样本权重：

$$R_{IPS} = \frac{1}{|D|}\sum_{(u,i) \in D} \frac{o_{u,i} \cdot \delta(r_{u,i}, \hat{r}_{u,i})}{\hat{o}_{u,i}}$$

其中：
- $o_{u,i}$ 为实际点击（0/1）
- $\hat{o}_{u,i}$ 为预估 pCTR（倾向分数）
- $\delta$ 为损失函数
- $D$ 为全样本空间

**直觉理解**：若一个样本被点击的概率很低但实际被点击了，说明这是"稀有但重要"的信号，应给予更高权重。

### 2.3 双重稳健估计（DR）

DR 结合了 IPW 与直接建模两种策略的估计结果：

$$\hat{R}_{DR} = \frac{1}{|D|}\sum_{(u,i) \in D} \left[ \hat{e}_{u,i} + \frac{o_{u,i} \cdot (\delta_{u,i} - \hat{e}_{u,i})}{\hat{o}_{u,i}} \right]$$

其中 $\hat{e}_{u,i}$ 是误差纠正模型的预测值。

**稳健性保证**：若倾向分数 $\hat{o}_{u,i}$ 或误差模型 $\hat{e}_{u,i}$ 之一正确，则 DR 估计无偏。两者都正确时方差最小。

## 3. 数学推导

### 3.1 IEB 偏差推导

ESMM 的 CVR 间接估计为：

$$\hat{pCVR} = \frac{\hat{pCTCVR}}{\hat{pCTR}}$$

当 $\hat{pCTR} \to 0$ 时，该比值不稳定。且由于乘法结构：

$$\nabla_{pCVR} \mathcal{L} = \nabla_{pCTCVR} \mathcal{L} \cdot \hat{pCTR}$$

pCTR 较小的样本对 pCVR 的梯度贡献极小，导致高价值但低 CTR 的样本被忽略。

### 3.2 IPW 无偏性证明

在因果框架下，CVR 的期望损失可写为：

$$\mathbb{E}[L] = \mathbb{E}_{X}[\mathbb{E}_{O|X}[\frac{O \cdot L(Y, \hat{Y})}{P(O=1|X)}]]$$

由期望的线性性和迭代期望定律，IPW 估计器无偏。

### 3.3 DR 方差分析

DR 估计器的方差为：

$$Var(\hat{R}_{DR}) = \frac{1}{n^2}\sum \left(\frac{\sigma^2_e}{\hat{o}_{u,i}} + (\hat{e}_{u,i} - e_{u,i})^2 \cdot (1 - \frac{1}{\hat{o}_{u,i}})^2 \right)$$

当 $\hat{e}$ 准确时，第二项趋零，方差小于纯 IPW。

## 4. 模型架构与训练过程

ESCM² 包含三个共享 Embedding 的子网络：

1. **CTR 塔**：预估 $\hat{o}_{u,i}$（倾向分数），在全空间训练
2. **CVR 塔**：预估转化概率，通过 IPW/DR 在全空间训练
3. **CTCVR 塔**：$\hat{pCTCVR} = \hat{pCTR} \times \hat{pCVR}$，辅助监督

**训练流程**：
- Step 1：CTR 塔用全部曝光样本训练，输出 $\hat{o}_{u,i}$
- Step 2：CVR 塔用 IPW 或 DR 损失训练，$\hat{o}_{u,i}$ 作为倾向分数
- Step 3：CTCVR 塔通过乘法约束提供额外监督信号
- 联合优化总损失

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 电商转化预估 | 点击→加购→下单多阶段转化 |
| 广告 oCPX | 按转化出价但按点击计费 |
| 内容推荐 | 曝光→点击→完播/互动 |
 | 职位推荐 | 曝光→点击→投递简历 |

## 6. 优缺点分析

**优点**：
- 理论保证无偏性，消除 IEB 和 PIP
- 全空间建模，利用全部曝光数据
- DR 估计双重稳健，对模型误设鲁棒
- 兼容任意底层网络结构

**缺点**：
- IPW 对极小的倾向分数敏感（方差爆炸），需要截断
- DR 需额外训练误差纠正模型，计算复杂度高
- 倾向分数估计偏差会传播至 CVR 估计
- 工程实现比 ESMM 复杂

## 7. 与相关方法对比

| 方法 | 建模空间 | 偏差处理 | 训练方式 | 复杂度 |
|------|---------|---------|---------|--------|
| ESMM | 全空间 | 隐式（乘法） | 联合训练 | 中 |
| ESM² | 全空间 | 部分改进 | 联合训练 | 中 |
| ESCM²-IPW | 全空间 | IPW 显式去偏 | 联合训练 | 高 |
| ESCM²-DR | 全空间 | 双重稳健去偏 | 联合训练 | 高 |
| DFSM | 全空间 | 对比学习去偏 | 联合训练 | 高 |

## 8. PyTorch 代码实现

```python
import torch
import torch.nn as nn

class ESCM2(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dims=[128, 64]):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        emb_out = embedding_dim
        
        self.ctr_tower = nn.Sequential(
            nn.Linear(emb_out, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        self.cvr_tower = nn.Sequential(
            nn.Linear(emb_out, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        self.error_model = nn.Sequential(
            nn.Linear(emb_out, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)
        pctr = self.ctr_tower(emb)
        pcvr = self.cvr_tower(emb)
        pctcvr = pctr * pcvr
        error_pred = self.error_model(emb)
        return pctr, pcvr, pctcvr, error_pred

def dr_loss(pctr, pcvr, error_pred, click, conversion, eps=1e-6):
    propensity = torch.clamp(pctr, min=eps, max=1.0 - eps)
    sample_weight = click / propensity.detach()
    
    cvr_bce = nn.functional.binary_cross_entropy(pcvr, conversion, reduction='none')
    
    direct_pred = error_pred.squeeze()
    dr_residual = click * (conversion.float() - direct_pred) / propensity.detach()
    dr_estimate = direct_pred + dr_residual
    
    loss_ipw = (sample_weight * cvr_bce).mean()
    loss_dr = nn.functional.mse_loss(dr_estimate.squeeze(), conversion.float())
    
    ctr_loss = nn.functional.binary_cross_entropy(pctr.squeeze(), click.float())
    pctcvr = pctr * pcvr
    pctcvr_loss = nn.functional.binary_cross_entropy(
        pctcvr.squeeze(), (click.float() * conversion.float())
    )
    
    total_loss = ctr_loss + pctcvr_loss + loss_ipw + 0.5 * loss_dr
    return total_loss
```

## 9. 常见问题与易错点

1. **倾向分数截断不当**：$\hat{o}_{u,i}$ 过小导致 IPW 权重爆炸，需 clamp 到 $[0.01, 1.0]$
2. **梯度传播问题**：IPW 中倾向分数需 detach，否则 CTR 和 CVR 梯度耦合导致训练不稳定
3. **误差纠正模型欠拟合**：DR 的误差模型需要单独充分训练，否则 DR 退化为 IPW
4. **线上推理与训练不一致**：线上只用到 CVR 塔，需确保其独立预测质量

## 10. 学习总结

ESCM² 将因果推断中的 IPW 和 DR 估计引入多任务 CVR 预估，从理论上解决了 ESMM 的 IEB 和 PIP 问题。核心洞察是：将"点击"视为因果推断中的"处理"，通过反事实推理在全空间上实现无偏的 CVR 估计。实践中推荐优先使用 DR 变体（更稳健），并注意倾向分数的截断策略。

## 11. 学习路径建议

- **前置知识**：ESMM、多任务学习基础、因果推断入门
- **进阶方向**：DFSFM、Multi-IPW、因果推荐系统
- **推荐论文**：ESMM (SIGIR 2018)、ESCM² (SIGIR 2022)
