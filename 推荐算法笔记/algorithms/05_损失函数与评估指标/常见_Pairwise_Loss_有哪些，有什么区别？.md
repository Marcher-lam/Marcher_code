# 面试题：常见 Pairwise Loss 有哪些，有什么区别？

面试题：常见 Pairwise Loss 有哪些，有什么区别？

推荐算法中常见的 Pairwise Loss 主要包括以下四种核心方法，它们在优化相对排序关系时各有特点：

# 一、BPR Loss (Bayesian Personalized Ranking Loss)

原理：BPR Loss基于贝叶斯后验优化思想，强制正样本（用户交互过的物品）的预测得分高于随机采样的负样本。其核心是最大化正负样本得分差异的概率。

$$
\mathcal {L} _ {\mathrm {B P R}} = - \sum_ {(u, i ^ {+}, i ^ {-})} \log \sigma (s (u, i ^ {+}) - s (u, i ^ {-}))
$$

公式：

其中， $\sigma$ 为 Sigmoid 函数， $s ( u , i )$ 为用户-物品得分函数。

特点：

 隐式反馈优化：适用于点击、购买等隐式反馈数据，强调正样本的相对优先级。  
 高效负采样：通常采用随机负采样，但对困难负样本（Hard Negative）区分能力有限。

# 二、Triplet Loss (Margin Ranking Loss)

原理：通过三元组（anchor 锚样本 、正样本 $p$ 、负样本 $n$ ）引入边界（Margin），强制正样本与锚样本的距离比负样本近至少一个边距m。

$$
\mathcal {L} _ {\text {T r i p l e t}} = \sum_ {(a, p, n)} \max  (0, d (a, p) - d (a, n) + m)
$$

公式：

其中，d 为距离函数（如欧氏距离或余弦相似度）。

特点：

 边界控制：通过 $m$ 调节正负样本区分度，防止模型陷入局部最优。  
 困难样本挖掘：需在线采样困难负样本以提升效果（如 FaceNet 人脸识别）。

![](images/d7db8d9499010ec7a970d57eab313fb509b35a232ee72d2a34a36da0a9a9fe94.jpg)

# 三、RankNet Loss

原理：

将排序问题转化为概率预测，通过交叉熵损失衡量正样本得分高于负样本的概率。其核心是定义两个物品的排序概率：

$$
\begin{array}{l} P _ {i j} = \frac {e ^ {s _ {i}}}{e ^ {s _ {i}} + e ^ {s _ {j}}} \\ \mathcal {L} _ {\text {R a n k N e t}} = - \sum \bar {P} _ {i j} \log P _ {i j} + (1 - \bar {P} _ {i j}) \log (1 - P _ {i j}) \\ \end{array}
$$

公式： ,j)

其中， $\bar { P } _ { i j }$ 为真实排序标签（1 表示 i 排在 j 前）。

特点：

概率化排序：输出具有可解释性的概率值，适用于需要置信度评估的场景。  
梯度平滑：相比 BPR，梯度更新更稳定，但计算复杂度较高。

# 四、Pairwise Logistic Loss

原理：与 RankNet 类似，但简化了概率计算，直接使用得分差异的对数损失。其本质是 RankNet 的一阶近似。

$$
\mathcal {L} _ {\text {L o g i s t i c}} = \sum_ {(i, j)} \log \left(1 + e ^ {s _ {j} - s _ {i}}\right)
$$

特点：

 计算高效：去除了 Sigmoid 函数，适合大规模数据训练。  
鲁棒性：对噪声标签敏感度低于 RankNet。

# 五、核心区别与选型建议

<table><tr><td>损失函数</td><td>优化目标</td><td>计算复杂度</td><td>适用场景</td></tr><tr><td>BPR Loss</td><td>隐式反馈的相对排序</td><td>低</td><td>电商推荐、社交网络</td></tr><tr><td>Triplet Loss</td><td>边界约束的硬样本区分</td><td>中</td><td>图像检索、冷启动用户</td></tr><tr><td>RankNet Loss</td><td>概率化排序关系</td><td>高</td><td>搜索排序、风险评估</td></tr><tr><td>Pairwise Logistic</td><td>高效的大规模排序</td><td>中</td><td>广告CTR、短视频流</td></tr></table>

# 选型建议：

1. 数据规模大且需快速迭代：优先选择 BPR 或 Pairwise Logistic Loss。  
2. 需精细化困难样本区分：使用 Triplet Loss 并配合在线困难样本挖掘。  
3. 要求概率输出或稳定性：选择 RankNet Loss。

# 六、四种 Pairwise Loss 梯度行为对比

<table><tr><td>损失函数</td><td>梯度特点</td><td>对困难样本敏感度</td><td>梯度消失风险</td></tr><tr><td>BPR</td><td>梯度与σ'(·)成正比</td><td>中等（依赖得分差）</td><td>得分差很大时梯度趋零</td></tr><tr><td>Triplet</td><td>Hinge型，margin内才有梯度</td><td>高（hard mining必要）</td><td>已满足margin时无梯度</td></tr><tr><td>RankNet</td><td>平滑梯度，处处可导</td><td>中等</td><td>低</td></tr><tr><td>Logistic</td><td>类似BPR但无sigmoid</td><td>中等</td><td>得分差极大时梯度小</td></tr></table>

# 七、Python 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()

def triplet_loss(anchor, positive, negative, margin=1.0):
    dist_pos = F.pairwise_distance(anchor, positive, p=2)
    dist_neg = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(dist_pos - dist_neg + margin).mean()

def ranknet_loss(score_i, score_j, label_i, label_j):
    s_ij = label_i - label_j
    p_ij = 1.0 / (1.0 + torch.exp(-(score_i - score_j)))
    p_bar_ij = (s_ij + 1.0) / 2.0
    return F.binary_cross_entropy(p_ij, p_bar_ij, reduction='mean')

def pairwise_logistic_loss(pos_scores, neg_scores):
    return torch.log1p(torch.exp(neg_scores - pos_scores)).mean()

batch = 32
pos_scores = torch.randn(batch)
neg_scores = torch.randn(batch)
anchor = torch.randn(batch, 16)
positive = torch.randn(batch, 16)
negative = torch.randn(batch, 16)
labels_i = torch.ones(batch)
labels_j = torch.zeros(batch)

print(f"BPR Loss: {bpr_loss(pos_scores, neg_scores).item():.4f}")
print(f"Triplet Loss: {triplet_loss(anchor, positive, negative, margin=1.0).item():.4f}")
print(f"RankNet Loss: {ranknet_loss(pos_scores, neg_scores, labels_i, labels_j).item():.4f}")
print(f"Logistic Loss: {pairwise_logistic_loss(pos_scores, neg_scores).item():.4f}")

print(f"\n梯度对比（正样本得分变化时）:")
pos = torch.tensor([2.0], requires_grad=True)
neg = torch.tensor([0.5], requires_grad=True)
for name, fn in [("BPR", bpr_loss), ("Logistic", pairwise_logistic_loss)]:
    loss = fn(pos, neg)
    loss.backward()
    print(f"  {name}: loss={loss.item():.4f}, grad_pos={pos.grad.item():.4f}")
    pos.grad = None
    neg.grad = None
```

# 八、推荐系统中 Pairwise Loss 的实践建议

1. 负采样策略：BPR的随机负采样效果有限，工业上常用混合负采样（Popularity-based + Hard Negative Mining）
2. 温度调节：BPR中可引入温度系数τ：$\log\sigma((s^+ - s^-)/\tau)$，τ越小模型越关注困难样本
3. 多目标融合：实际推荐中常将Pairwise Loss与Pointwise Loss（如BCE）加权组合，兼顾绝对精度与相对排序

# 一、Focal Loss 解决的问题

Focal Loss 主要用于解决以下两类问题：

# 1. 类别不平衡问题

在目标检测（尤其是 One-Stage 方法）中，正样本（前景目标）数量远少于负样本（背景），导致模型训练时被大量简单负样本主导，难以有效学习正样本特征。

# 2. 难易样本不均衡问题

易分类样本（如高置信度的背景）占比过高，而难分类样本（如模糊目标）的损失贡献被稀释，模型优化方向偏离实际需求。

Focal loss 论文：Focal Loss for Dense Object Detection

![](images/dac82a2f78973b5161ba43a6ca4c53f137ec0c1bb6915a69f8c0c138e8796e9c.jpg)  
Figure 1. We propose a novel loss we term the Focal Loss that adds a factor $( 1 - p _ { \mathrm { t } } ) ^ { \gamma }$ to the standard cross entropy criterion. Setting $\gamma > 0$ reduces the relative loss for well-classified examples $( p _ { \mathrm { t } } > . 5 )$ , putting more focus on hard, misclassified examples. As our experiments will demonstrate, the proposed focal loss enables training highly accurate dense object detectors in the presence of vast numbers of easy background examples.

# 二、原理与公式推导

Focal Loss 基于标准交叉熵（Cross Entropy, CE）改进，通过引入两个调节因子实现上述目标：

# 1. 标准交叉熵公式

对于二分类问题，交叉熵损失为：

$$
C E \left(p _ {t}\right) = - \log \left(p _ {t}\right)
$$

其中 $p _ { t }$ 表示模型对正确类别的预测概率： $p _ { t } = { \left\{ \begin{array} { l l } { p , } & { { \mathrm { i f ~ } } y = 1 } \\ { 1 - p , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }$

# 2. 引入调节因子

Focal Loss 在 CE 基础上增加两个权重项

 α（类别平衡因子）：控制正负样本权重，通常正样本 $\pmb q$ 较小（如 0.25），负样本 1−α 较大。  
 （调制因子）：降低易分类样本的权重，γ（聚焦参数）越大，简单样本的损失衰减越强。 $( 1 - p _ { t } ) ^ { \gamma }$

最终公式：

$$
F L \left(p _ {t}\right) = - \alpha_ {t} \left(1 - p _ {t}\right) ^ {\gamma} \log \left(p _ {t}\right)
$$

展开形式（二分类）：

$$
F L (y, p) = \left\{ \begin{array}{l l} - \alpha (1 - p) ^ {\gamma} \log (p), & y = 1 \\ - (1 - \alpha) p ^ {\gamma} \log (1 - p), & y = 0 \end{array} \right.
$$

# 三、PyTorch 代码实现

以下是一个完整的 Focal Loss 实现，支持多标签分类

# 代码关键点说明：

1. 输入要求：

 inputs：未归一化的模型输出（Logits），形状为 (N, *)。  
 targets：与 inputs 同形状的 0-1 标签。

2. 调制因子计算：

会使高置信度样本（pt →1）的损失权重降低，反之保留难样本的高权重。 $( 1 - p _ { t } ) ^ { \gamma }$ $p t  1$

3. α 平衡：

对正负样本分别应用 $\pmb q$ 和 $\pmb { 1 - a }$ ，缓解类别数量不平衡。

import torch   
import torch(nn as nn   
import torch(nnfunctional as F   
class FocalLoss(nnModule): def__init__(self, alpha=0.25,gamma=2,reduction='mean'): super(FocalLoss,self).__init_(） self.alpha $=$ alpha #正样本权重（如0.25） self.gamma $=$ gamma #难易样本调节（常用2） self.reduce $=$ reduction #损失聚合方式（mean/sum）   
def forward(self，inputs,targets): #计算二元交叉熵（无需Sigmoid，因含Logits） BCE_loss $=$ F;binary CROSS_entropy_with_logits( inputs,targets,reduction $\coloneqq$ 'none') #计算概率pt（对正确类别的预测概率） pt $=$ torch.exp(-bce_loss)#pt $=$ p_t（公式中的p_t） #计算Focal Loss的核心调制因子 focal_term $\equiv$ (1-pt）\*\*self.gamma #应用 $\alpha$ 平衡：正样本乘 $\alpha$ ，负样本乘 $(1 - \alpha)$ alpha_factor $=$ targets\*self.alpha $+$ (1-tTargets)\* (1-self.alpha) #组合得到最终损失 fl_loss $=$ alpha_factor\*focal_term\*bce_loss return fl_loss.mean()   
labels $=$ torch.randint(0,2,(32,1)) $\ast 1.0$ preds $=$ torch RAND(32,1)   
fl $=$ FocalLoss() (preds,labels)   
print(fl)

# 九、Focal Loss 与 Pairwise Loss 的关系

Focal Loss属于Pointwise Loss，解决的是分类中的类别不平衡问题；Pairwise Loss解决的是排序中的相对序关系问题。在推荐系统中，两者可以互补：

- 精排阶段：常用BCE + Focal Loss处理点击率预估的样本不平衡
- 重排/列表排序阶段：常用BPR或Pairwise Logistic优化相对排序
- 工业实践：可将Focal Loss嵌入Pairwise框架，如Focal BPR：对正负样本对引入难易权重

# 十、常见面试追问

1. Q: BPR Loss和Cross Entropy Loss的区别是什么？
A: BCE是Pointwise Loss，独立优化每个样本的预测值；BPR是Pairwise Loss，优化正负样本对的相对排序。BPR不关心绝对分数只关心相对大小，因此更适合排序任务；BCE适合需要精确概率估计的场景（如CTR预估）。

2. Q: 为什么Triplet Loss需要Hard Negative Mining？
A: 随机负样本通常与锚样本距离很远，Triplet Loss在margin已满足时梯度为零。只有挖掘接近决策边界的困难负样本（距离锚样本近但应为负类），模型才能学到更精细的特征表示。

3. Q: Focal Loss的γ参数如何选择？
A: γ=0退化为标准交叉熵；γ=2是论文推荐值，效果最好；γ越大，简单样本的损失衰减越强，模型越关注困难样本。在推荐系统中，由于正负样本极度不平衡（点击率约1-5%），γ通常取2-3。

# 5.2 评估指标
