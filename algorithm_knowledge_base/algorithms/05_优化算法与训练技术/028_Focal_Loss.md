# Focal Loss 学习文档

## 1. 算法基础认知

Focal Loss（焦点损失）是一种专门设计用于解决**类别不平衡**问题的损失函数，由Lin等人在2017年的RetinaNet论文中首次提出。在目标检测等任务中，前景样本（需要检测的物体）往往远少于背景样本，导致模型训练时容易被大量背景样本主导，忽略了对少数类的学习。Focal Loss通过动态调整交叉熵损失的权重，降低简单样本的贡献，增强困难样本的学习信号，使模型能够更好地处理类别不平衡问题。

## 2. 核心原理

Focal Loss的核心思想是**降低简单样本的损失权重，增加困难样本的损失权重**。在标准交叉熵损失中，无论是简单样本（模型预测概率很高）还是困难样本（模型预测概率很低），其损失值都是-log(p_t)，当类别严重不平衡时，大量简单样本的累积损失会主导梯度更新，使模型倾向于预测多数类。Focal Loss引入一个聚焦参数γ（gamma），当预测概率p_t较高（样本容易分类）时，(1-p_t)^γ项会大幅衰减损失；当p_t较低（样本难以分类）时，该项接近1，损失基本不变。这样模型训练时会自动关注那些难以分类的样本，提高对少数类的检测能力。

## 3. 数学公式与推导

Focal Loss的数学表达式为：

$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

其中p_t表示模型对真实类别的预测概率，对于二分类问题，p_t∈[0,1]，当y=1时p_t=p，当y=-1时p_t=1-p。参数α_t是类别权重，用于平衡不同类的的重要性，通常取α_t=0.25，对应正类的权重为0.75。参数γ（gamma）是聚焦参数，控制对困难样本的关注程度，推荐值为γ=2。

推导过程如下：标准二分类交叉熵损失为CE(p_t)=-log(p_t)，加入类别权重得到CE(p_t)=-α_t log(p_t)，加入聚焦因子得到FL(p_t)=-α_t(1-p_t)^γ log(p_t)。当γ=0时，Focal Loss退化为带权重的交叉熵损失；当γ>0时，简单样本的损失被衰减。设γ=2，对于p_t=0.9的简单样本，损失衰减为原来的(1-0.9)^2=0.01倍；对于p_t=0.5的困难样本，损失衰减为原来的(1-0.5)^2=0.25倍。


### 3.6 补充公式

**Sigmoid函数及其导数**：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
导数形式：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$
可用于Logistic回归输出层的概率解释。

**ReLU激活函数**：
$$ReLU(z) = \max(0, z)$$
导数：$ReLU'(z) = 1$ 当$z > 0$，否则为$0$。

**softmax函数**（多分类输出）：
$$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$
保证输出所有类别的概率和为1。

**交叉熵损失**（softmax输出）：
$$L = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$
其中$y_k$是真实标签（one-hot），$\hat{y}_k$是softmax预测概率。

**参数更新（Adam优化器）**：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{（一阶矩）}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{（二阶矩）}$$
偏差校正：
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
参数更新：
$$\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## 4. 训练过程讲解

Focal Loss的训练过程与标准交叉熵损失类似，但需要在计算损失时引入聚焦因子。具体步骤包括：首先获取模型对batch中每个样本的预测logits，通过sigmoid函数转换为概率p_t；然后根据真实标签y计算每个样本的Focal Loss值；接着使用类别权重α_t对不同类的损失进行加权；最后对batch中所有样本的损失求平均或求和。在反向传播时，聚焦因子的导数(1-p_t)^γ项会自动调整梯度大小，使困难样本获得更大的梯度。在实际训练中，通常将γ设为2，α设为与类别频率成反比的值，例如在COCO数据集上γ=2，α_1=0.25（正类），α_-1=0.75（背景类）效果较好。

## 5. 应用场景

Focal Loss主要用于**目标检测**领域，特别是单阶段目标检测器（One-Stage Detector），如RetinaNet、YOLO系列等。单阶段检测器需要在密集的候选框中进行分类，背景框占绝大多数，使用Focal Loss可以有效解决正负样本不平衡问题。此外，Focal Loss也应用于**语义分割**、**实例分割**、**医学影像分析**等存在严重类别不平衡的任务中。在医学影像中，如肿瘤检测，阳性样本往往远少于阴性样本，使用Focal Loss可以提高对阳性样本的检测灵敏度。在**多目标跟踪**、**人脸检测**等任务中也有广泛应用。

## 6. 优缺点分析

Focal Loss的优点包括：有效解决类别不平衡问题，提高少数类的检测精度；只有一个超参数γ需要调参，使用简单；与现有的检测框架兼容，不需要额外的修改；可以与交叉熵损失无缝切换。缺点包括：当γ值设置不当时，可能导致训练不稳定；当正负样本比例极低时，效果可能不如Focal Loss与其他方法的组合；计算复杂度与标准交叉熵损失相当，需要额外的幂运算；对于多分类问题，需要为每个类单独设置α_t参数。

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: 模型预测的logits，shape为[N, num_classes]或[N, 1]
        targets: 真实标签，shape为[N]
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(-1))
            targets = targets.view(targets.size(0))
        
        if inputs.size(-1) == 1:
            p = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            p_t = p * targets + (1 - p) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_t * focal_weight * ce_loss
            else:
                loss = focal_weight * ce_loss
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            if self.alpha is not None:
                alpha_t = torch.full_like(inputs, self.alpha)
                alpha_t.scatter_(1, targets.unsqueeze(1), 1 - self.alpha)
                loss = alpha_t * focal_weight * ce_loss
            else:
                loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def create_focal_loss(alpha=0.25, gamma=2.0):
    return FocalLoss(alpha=alpha, gamma=gamma)

if __name__ == '__main__':
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    
    criterion = create_focal_loss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Focal Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np

def focal_loss_numpy(predictions, targets, alpha=0.25, gamma=2.0, num_classes=10):
    """
    手工实现Focal Loss（NumPy版本）
    predictions: 模型预测的logits，shape为[N, K]
    targets: 真实标签，shape为[N]
    alpha: 类别权重
    gamma: 聚焦参数
    """
    batch_size = predictions.shape[0]
    predictions = predictions - np.max(predictions, axis=1, keepdims=True)
    exp_predictions = np.exp(predictions)
    probs = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
    
    ce_loss = -np.log(probs[np.arange(batch_size), targets] + 1e-10)
    pt = np.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    
    alpha_t = np.full(num_classes, alpha)
    alpha_t[0] = 1 - alpha
    
    loss = focal_weight * ce_loss
    return np.mean(loss)

def focal_loss_binary(predictions, targets, alpha=0.25, gamma=2.0):
    """
    二分类Focal Loss实现
    predictions: 模型预测的logits（未经过sigmoid）
    targets: 真实标签（0或1）
    """
    p = 1.0 / (1.0 + np.exp(-predictions))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    ce_loss = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce_loss
    
    return np.mean(loss)

if __name__ == '__main__':
    np.random.seed(42)
    logits = np.random.randn(8, 10)
    targets = np.random.randint(0, 10, (8,))
    
    loss = focal_loss_numpy(logits, targets)
    print(f"Multi-class Focal Loss: {loss:.4f}")
    
    binary_logits = np.random.randn(8)
    binary_targets = np.random.randint(0, 2, (8,))
    binary_loss = focal_loss_binary(binary_logits, binary_targets)
    print(f"Binary Focal Loss: {binary_loss:.4f}")
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_focal_loss():
    p_t = np.linspace(0.01, 0.99, 100)
    gamma_values = [0, 1, 2, 3]
    alpha = 0.25
    
    plt.figure(figsize=(10, 6))
    for gamma in gamma_values:
        if gamma == 0:
            fl = -alpha * np.log(p_t)
        else:
            fl = -alpha * (1 - p_t) ** gamma * np.log(p_t)
        plt.plot(p_t, fl, label=f'γ={gamma}')
    
    ce = -alpha * np.log(p_t)
    plt.plot(p_t, ce, 'k--', label='CE (γ=0)', alpha=0.5)
    
    plt.xlabel('p_t (正类预测概率)', fontsize=12)
    plt.ylabel('Focal Loss', fontsize=12)
    plt.title('Focal Loss vs Cross Entropy Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('focal_loss_comparison.png', dpi=150)
    plt.show()

def analyze_effective_weights():
    gamma = 2.0
    p_t = np.linspace(0.01, 0.99, 100)
    effective_weights = (1 - p_t) ** gamma
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_t, effective_weights, 'b-', linewidth=2)
    plt.xlabel('p_t', fontsize=12)
    plt.ylabel('有效权重 (1-p_t)^γ', fontsize=12)
    plt.title(f'Focal Loss有效权重 (γ={gamma})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.25, color='r', linestyle='--', label='25%权重')
    plt.legend()
    plt.tight_layout()
    plt.savefig('focal_weights.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    visualize_focal_loss()
    analyze_effective_weights()
```

运行结果分析：当γ=2时，对于p_t>0.5的简单样本，有效权重迅速衰减；对于p_t<0.5的困难样本，权重接近1。这导致模型训练时自动聚焦于难以分类的样本，提高对少数类的检测能力。在RetinaNet实验中，使用Focal Loss相比标准Cross Entropy在COCO数据集上mAP提升了约3.2个百分点。

## 10. 模型评估

Focal Loss的评估主要关注以下几个方面：首先是**类别不平衡处理效果**，可以使用混淆矩阵、Precision、Recall、F1-score等指标，特别关注少数类的召回率；其次是**训练稳定性**，观察训练过程中损失的变化曲线，确保没有出现NaN或发散的情况；第三是**收敛速度**，对比使用Focal Loss与使用标准交叉熵损失的收敛轮数；第四是**检测精度**，在目标检测任务中使用mAP@IoU=0.5和mAP@IoU=0.5:0.95作为主要指标。实践中推荐使用OHEM（Online Hard Example Mining）与Focal Loss结合，可以进一步提升效果。评估时应注意α和γ的超参数搜索，使用验证集进行调参。

## 11. 常见问题与易错点

常见问题包括：**γ值选择不当**，γ过小（<1）时效果不明显，γ过大（>3）可能导致训练不稳定，推荐从γ=2开始尝试；**α设置错误**，α应该与类别频率相关，正类少时α应该较大；**与Sigmoid/Softmax的配合**，对于二分类使用Sigmoid，多分类使用Softmax，不要混淆；对于多分类Focal Loss，需要正确处理α_t的映射。使用时的易错点包括：忽略背景类的处理，在目标检测中背景类也应该有α权重；忘记���推���时移除Focal Loss，推理只需要argmax或阈值判断；过度调参，Focal Loss通常只需要微调γ值即可。

## 12. 学习总结

Focal Loss是解决类别不平衡问题的经典方法，通过引入聚焦因子动态调整损失权重。核心思想是降低简单样本的损失贡献，增强困难样本的学习信号。公式简洁高效，只需调节γ和α两个超参数。在目标检测领域已成为标准配置，RetinaNet等网络都采用Focal Loss作为损失函数。学习和使用Focal Loss时，重点理解(1-p_t)^γ的物理意义，以及如何根据具体任务调整超参数。推荐的实践路线是：首先在二分类问题上验证Focal Loss的效果，然后迁移到多分类和目标检测任务中。

## 13. 练习题与思考题与思考题（含答案）

**练习题1**：证明当γ=0时，Focal Loss退化为带权重的交叉熵损失。

答案：当γ=0时，(1-p_t)^γ=(1-p_t)^0=1，所以FL(p_t)=-α_t·1·log(p_t)=-α_t log(p_t)，这正是带权重的交叉熵损失（CE）。

**练习题2**：当p_t=0.9，α=0.25，γ=2时，计算Focal Loss相比Cross Entropy损失衰减了多少倍？

答案：Focal Loss的有效权重为(1-0.9)^2=0.01，所以损失衰减为原来的0.01倍，即衰减了100倍。

**练习题3**：为什么Focal Loss二分类版本使用Sigmoid而多分类使用Softmax？

答案：二分类问题只需要一个logit即可表示正类概率（p），负类概率为1-p；多分类问题有K个类别，需要Softmax将K个logits转换为概率分布，确保所有类别的概率之和为1。

**思考题1**：Focal Loss与OHEM（Online Hard Example Mining）有什么异同？

答案：相同点：都试图让模型关注困难样本。不同点：Focal Loss通过数学公式连续地降低简单样本的权重，是隐式的困难样本挖掘；OHEM通过筛选batch中损失最大的样本来显式地关注困难样本。两者可以结合使用。

**思考题2**：如果数据集中正负样本比例为1:1000，普通交叉熵会出现什么问题？Focal Loss如何解决这个问题？

答案：普通交叉熵中负类损失累积会主导梯度，使模型倾向于预测负类以最小化整体损失。Focal Loss通过(1-p_t)^γ降低易分类负类的损失权重，使正类和难分类负类获得相对更大的梯度，从而学习到有区分性的特征。


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：本算法的核心机制是什么？请简述其工作原理。

**答案与解析**：

**步骤1**：识别问题类型
根据算法定义，这是一个[类型：监督/无监督/生成/强化学习]任务。

**步骤2**：应用核心公式
$$核心公式 = [具体公式]$$
该公式的意义是[解释公式含义]。

**步骤3**：验证答案
代入具体数据验证：[计算过程]
最终结果符合预期，说明理解正确。

**答案**：算法的核心是通过[机制]实现[目标]，属于[算法类别]。

---

#### 练习2：手动计算

**问题**：给定数据[X=具体值, y=具体值]，手动计算[算法名]的[参数/结果]。

**答案与解析**：

**步骤1**：准备数据
$X = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$  
$y = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}$

**步骤2**：应用算法步骤
根据[算法名]的定义，计算第一步：
$$第一步 = [具体公式代入] = [数值]$$

**步骤3**：继续计算
$$第二步 = [公式] = [结果]$$

**步骤4**：得到最终答案
$$最终结果 = [综合计算] = [具体数值]$$

**验证**：将结果带回原式检验 $[验证过程]$，确认正确。

---

#### 思考题：改进分析

**问题**：本算法在[特定场景]下存在哪些局限性？请提出改进方案。

**答案与解析**：

**局限性分析**：
1. **局限性1**：[具体表现]，原因是[原因解释]
2. **局限性2**：[具体表现]，原因是[原因解释]

**改进方案对比**：

| 改进方法 | 原理 | 优势 | 代价 |
|---------|------|------|------|
| 方法A | [原理] | [好处] | [额外成本] |
| 方法B | [原理] | [好处] | [额外成本] |
| 方法C | [原理] | [好处] | [额外成本] |

**推荐方案**：在实际应用中优先考虑[方法A]，因为[理由]。
## 14. 学习路径建议建议

学习Focal Loss建议按照以下路径进行：首先理解类别不平衡问题的本质以及它对模型训练的影响；然后学习传统的解决方法如类别加权、欠采样、过采样、OHEM等；接着深入学习Focal Loss的数学公式和物理意义；通过实验对比Focal Loss与标准交叉熵损失的效果差异；在自己的项目中应用Focal Loss，可以从目标检测或医学图像分割任务开始；最后研究Focal Loss的变体如Focal Loss的DFL（Distribution Focal Loss）等扩展工作。

---

## 补充材料：Focal Loss变体与扩展

### A1. Equalized Focal Loss (EFLN)

Equalized Focal Loss针对前景与背景不平衡提出：

$$FL_{EQL}(p_t) = -\alpha_t(1-p_t)^{\gamma+\delta}\log(p_t)$$

其中$\delta$是根据样本困难程度动态调整的参数。

### A2. Quality Focal Loss (QFL)

质量焦损用于检测框质量打分：

$$QFL(p, q) = -|q - q_\tau|^\beta \cdot ((1-p_t)^\gamma \log(p_t) + p_t^\gamma \log(1-p_t))$$

其中$q$是预测质量分数，$q_\tau$是质量阈值。

### A3. Distribution Focal Loss (DFL)

分布焦损用于边界框回归：

$$DFL(\sigma) = -\sum_{i=1}^{n} (p_{c_i} \cdot \log(\sigma_i) + (1-p_{c_i}) \cdot \log(1-\sigma_i))$$

其中$\sigma_i$是softmax输出的离散分布。

### A4. 组合使用Focal Loss的最佳实践

```python
class FocalLoss Combinator(nn.Module):
    """组合使用不同Focal Loss变体"""
    
    def __init__(self, alpha=0.25, gamma=2.0, use_ohem=True, ohem_ratio=3):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
    
    def forward(self, predictions, targets):
        loss = self.focal(predictions, targets)
        
        if self.use_ohem:
            loss = self._ohem_loss(loss, targets, self.ohem_ratio)
        
        return loss
    
    def _ohem_loss(self, loss, targets, ratio):
        """在线难例挖掘"""
        batch_size = loss.size(0)
        
        kept = batch_size // ratio
        
        _, sorted_idx = loss.sort(descending=True)
        selected_idx = sorted_idx[:kept]
        
        return loss[selected_idx].mean()


class MultiTaskFocalLoss(nn.Module):
    """多任务Focal Loss"""
    
    def __init__(self, num_classes=80, gamma=2.0, alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.bce_loss = nn BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets):
        cls_output = outputs[:, :self.num_classes]
        reg_output = outputs[:, self.num_classes:]
        
        cls_loss = self.focal_loss(cls_output, targets[:, 0].long())
        reg_loss = self.bce_loss(reg_output, targets[:, 1]).mean()
        
        total_loss = cls_loss + 0.1 * reg_loss
        
        return total_loss, cls_loss, reg_loss
```

### A5. Focal Loss与医学图像分析

医学影像中的类别不平衡问题尤为突出，Focal Loss在那里有重要应用：

```python
def create_medical_focal_loss(class_weights):
    """医学图像专用的Focal Loss
    
    class_weights: 针对不同类别的权重，如 tumor: 10, normal: 1
    """
    alpha = class_weights / class_weights.sum()
    gamma = 2.0
    
    def loss_fn(predictions, targets):
        probs = torch.sigmoid(predictions)
        
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        
        alpha_t = alpha[targets.long()]
        
        return (alpha_t * focal_weight * ce_loss).mean()
    
    return loss_fn


def evaluate_medical_detection(model, test_loader, threshold=0.5):
    """医学检测评估：关注敏感度和特异度"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).float()
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    tp = ((all_preds == 1) & (all_targets == 1)).sum()
    tn = ((all_preds == 0) & (all_targets == 0)).sum()
    fp = ((all_preds == 1) & (all_targets == 0)).sum()
    fn = ((all_preds == 0) & (all_targets == 1)).sum()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    dice = 2 * tp / (2 * tp + fp + fn)
    
    return {
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'dice': dice.item()
    }
```

### A6. Focal Loss可视化进阶

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_focal_loss_detailed():
    """详细的Focal Loss可视化"""
    p_t = np.linspace(0.001, 0.999, 100)
    gamma_values = [0, 1, 2, 3, 4]
    alpha = 0.25
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Focal Loss曲线
    ax = axes[0, 0]
    for gamma in gamma_values:
        if gamma == 0:
            fl = -alpha * np.log(p_t)
        else:
            fl = -alpha * (1 - p_t) ** gamma * np.log(p_t)
        ax.plot(p_t, fl, label=f'γ={gamma}')
    
    ax.set_xlabel('p_t (正类预测概率)')
    ax.set_ylabel('Focal Loss')
    ax.set_title('Focal Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # 2. 有效权重
    ax = axes[0, 1]
    for gamma in gamma_values:
        weight = (1 - p_t) ** gamma
        ax.plot(p_t, weight, label=f'γ={gamma}')
    
    ax.set_xlabel('p_t')
    ax.set_ylabel('有效权重 (1-p_t)^γ')
    ax.set_title('Effective Weights')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 类别不平衡效果
    ax = axes[1, 0]
    imbalance_ratios = [1, 10, 100, 1000]
    
    for ratio in imbalance_ratios:
        alpha_pos = 1 / (1 + ratio)
        alpha_neg = ratio / (1 + ratio)
        
        fl_pos = -alpha_pos * (1 - p_t) ** 2 * np.log(p_t)
        fl_neg = -alpha_neg * p_t ** 2 * np.log(1 - p_t + 1e-10)
        
        ax.plot(p_t, fl_pos, label=f'正类 (α={alpha_pos:.3f})')
        ax.plot(p_t, fl_neg, linestyle='--', label=f'负类 (α={alpha_neg:.3f})')
    
    ax.set_xlabel('p_t')
    ax.set_ylabel('Loss')
    ax.set_title('Class Imbalance Effect')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 对比不同损失函数
    ax = axes[1, 1]
    
    ce = -np.log(p_t)
    focal = -0.25 * (1 - p_t) ** 2 * np.log(p_t)
    ohem = np.where(p_t > 0.5, ce, focal * 10)
    
    ax.plot(p_t, ce, label='Cross Entropy', linewidth=2)
    ax.plot(p_t, focal, label='Focal Loss', linewidth=2)
    ax.plot(p_t, ohem, label='OHEM + Focal', linewidth=2)
    
    ax.set_xlabel('p_t')
    ax.set_ylabel('Loss')
    ax.set_title('Comparison of Loss Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('focal_loss_detailed.png', dpi=150)
    plt.show()


def analyze_gamma_sensitivity():
    """分析γ参数的敏感性"""
    np.random.seed(42)
    
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    bg_fg_ratios = [1, 10, 50, 100, 500, 1000]
    
    results = np.zeros((len(gammas), len(bg_fg_ratios)))
    
    for i, gamma in enumerate(gammas):
        for j, ratio in enumerate(bg_fg_ratios):
            alpha_pos = 0.25
            alpha_neg = 0.75
            
            p_fg = 0.9
            p_bg = 0.1
            
            loss_fg = -alpha_pos * (1 - p_fg) ** gamma * np.log(p_fg)
            
            ratio_adjusted = ratio / (1 + ratio)
            loss_bg = -alpha_neg * ratio_adjusted * p_bg ** gamma * np.log(1 - p_bg)
            
            results[i, j] = loss_fg + loss_bg
    
    plt.figure(figsize=(10, 6))
    
    for i, gamma in enumerate(gammas):
        plt.plot(bg_fg_ratios, results[i], 'o-', label=f'γ={gamma}', linewidth=2)
    
    plt.xlabel('背景/前景比例 (log scale)')
    plt.ylabel('总损失')
    plt.title('γ Sensitivity Analysis')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gamma_sensitivity.png', dpi=150)
    plt.show()


def show_hard_example_mining_effect():
    """展示OHEM与Focal Loss的组合效果"""
    np.random.seed(42)
    
    difficulties = ['Easy', 'Medium', 'Hard']
    
    methods = {
        'CE': [0.01, 0.05, 0.15],
        'Focal': [0.02, 0.08, 0.25],
        'OHEM': [0.05, 0.12, 0.35],
        'Focal+OHEM': [0.03, 0.10, 0.40]
    }
    
    x = np.arange(len(difficulties))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (method, values) in enumerate(methods.items()):
        ax.bar(x + i*width, values, width, label=method)
    
    ax.set_xlabel('样本难度')
    ax.set_ylabel('损失值')
    ax.set_title('Hard Example Mining Effect')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ohem_effect.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_focal_loss_detailed()
    analyze_gamma_sensitivity()
    show_hard_example_mining_effect()
```

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Focal_Loss的核心思想及适用场景。
<details><summary>参考答案</summary>
Focal_Loss通过数据驱动学习输入到输出的映射，适用于人工智能中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Focal_Loss的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Focal_Loss核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Focal_Loss在什么情况下会失效？
2. 训练数据很少时，Focal_Loss还能有效工作吗？
3. 如何将Focal_Loss与其他方法结合？

