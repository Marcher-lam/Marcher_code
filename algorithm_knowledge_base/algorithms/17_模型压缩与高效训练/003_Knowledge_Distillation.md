# Knowledge Distillation 学习文档

## 1. 算法基础认知

知识蒸馏（Knowledge Distillation）是一种模型压缩技术，由Hinton等人在2015年提出，其核心思想是让**小型学生网络（Student）从大型教师网络（Teacher）学习**。教师网络通常是性能更好但计算量大的复杂模型，学生网络是轻量级的简单模型。通过让学生网络学习教师网络的输出分布（软标签），可以实现模型压缩和加速，同时保持接近教师网络的性能。知识蒸馏的独特之处在于它不仅学习真实标签（hard labels），还学习教师网络提供的"暗知识"（dark knowledge），即类别之间的关系信息。例如，教师网络可能知道"猫"和"狗"比"猫"和"汽车"更相似，这种知识对学生网络的学习很有价值。

## 2. 核心原理

知识蒸馏的核心原理是**利用教师网络的软输出传递知识**。在标准分类中，模型学习将输入映射到one-hot标签；在知识蒸馏中，学生网络学习教师网络的softmax输出概率分布。软标签包含的信息远多于硬标签：对于一个10类分类问题，真实标签只给出1bit信息，而教师网络的输出分布包含各类别的概率值，反映了类别之间的相似性和不确定性。学生网络通过匹配教师网络的软标签，可以学习到这些"暗知识"。此外，通过引入温度参数T可以调节软标签的"软度"，T越大，概率分布越平缓，学生网络更容易学习到类别之间的关系。

## 3. 数学公式与推导

教师网络的softmax输出：

$$p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

知识蒸馏使用的温度softmax：

$$p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

蒸馏损失函数：

$$L_{KD} = (1-\alpha) \cdot CE(y, q) + \alpha \cdot T^2 \cdot KL(p^T || q^T)$$

其中CE是硬标签的交叉熵损失，KL是软标签的KL散度，T是温度参数，α是平衡因子。

KL散度展开：

$$KL(p^T || q^T) = \sum_i p_i^T \log\frac{p_i^T}{q_i^T}$$

$$= \sum_i p_i^T (\log p_i^T - \log q_i^T)$$

$$= \sum_i p_i^T \log p_i^T - \sum_i p_i^T \log q_i^T$$

第一项是常数，第二项是带权的交叉熵。

推导：当T→1时，蒸馏损失退化为标准交叉熵；当T→∞时，所有类别概率趋向1/K。

## 4. 训练过程讲解

知识蒸馏的训练过程分为两个阶段：第一阶段训练教师网络，第二阶段学生网络从教师学习。具体步骤包括：首先使用完整训练数据训练一个高性能的教师网络；然后准备训练数据，可以使用原始数据或数据增强；设置温度参数T（通常为2-20）和平衡因子α（通常为0.5-0.9）；计算教师网络在温度T下的软输出；计算学生网络的软输出（使用相同的T）；计算蒸馏损失和硬标签损失；联合优化训练学生网络。训练时通常先使用高温T让学生学习软标签，然后再用T=1进行微调。学生网络的训练轮数通常比教师网络少很多就能达到较好效果。

## 5. 应用场景

知识蒸馏主要应用场景包括：**模型压缩**，将大模型压缩为小模型部署到移动端；**模型加速**，学生网络比教师更小更快；**多任务学习**，一个学生网络学习多个教师网络的知识；**集成学习**，将多个模型的知识整合到一个模型；**自蒸馏**，学生和教师是同一架构不同训练阶段。在实际应用中，知识蒸馏广泛用于BERT、ResNet、EfficientNet等模型的压缩。在Hugging Face Transformers库中，学生网络可以达到教师网络95%以上的性能。

## 6. 优缺点分析

知识蒸馏的优点包括：实现相对简单，可以与标准训���框架兼容；学生网络可以获得教师网络的大部分能力；可以与其他压缩技术（剪枝、量化）结合使用；不需要特殊硬件支持。缺点包括：需要训练教师网络，增加训练时间；学生网络的性能上限受限于教师网络；temperature和α需要仔细调节；某些任务可能学生网络无法学到教师的所有知识。

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits: 学生网络的logits
        teacher_logits: 教师网络的logits
        labels: 真实标签
        """
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return loss


class Distiller(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def forward(self, inputs, labels):
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = self.student(inputs)
        
        loss = self.compute_loss(student_logits, teacher_logits, labels, inputs)
        
        return loss, student_logits
    
    def compute_loss(self, student_logits, teacher_logits, labels, inputs):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return loss


def create_distillation_trainer(teacher_model, student_model, temperature=4.0, alpha=0.5):
    distiller = Distiller(teacher_model, student_model, temperature, alpha)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    return distiller, optimizer


class SimpleTeacherNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleStudentNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == '__main__':
    teacher = SimpleTeacherNet()
    student = SimpleStudentNet()
    
    distiller, optimizer = create_distillation_trainer(teacher, student, temperature=4.0, alpha=0.5)
    
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    for epoch in range(5):
        optimizer.zero_grad()
        loss, _ = distiller(x, y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np
import torch

def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    手工实现知识蒸馏损失
    """
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    
    distill_loss = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs), dim=-1)
    distill_loss = torch.mean(distill_loss) * (temperature ** 2)
    
    hard_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    
    loss = alpha * distill_loss + (1 - alpha) * hard_loss
    
    return loss


def self_distillation(model, temperature=4.0, alpha=0.5):
    """
    自蒸馏：使用模型自身的知识蒸馏
    """
    pass


def cosine_distillation(student_features, teacher_features, temperature=1.0):
    """
    特征蒸馏：使用余弦相似度
    """
    student_norm = torch.nn.functional.normalize(student_features, dim=-1)
    teacher_norm = torch.nn.functional.normalize(teacher_features, dim=-1)
    
    loss = 1 - torch.sum(student_norm * teacher_norm, dim=-1)
    loss = torch.mean(loss)
    
    return loss


if __name__ == '__main__':
    student_logits = torch.randn(32, 10)
    teacher_logits = torch.randn(32, 10) + 1.0
    teacher_logits[torch.arange(32), torch.randint(0, 10, (32,))] += 5.0
    labels = torch.randint(0, 10, (32,))
    
    loss = knowledge_distillation_loss(student_logits, teacher_logits, labels)
    print(f"Distillation Loss: {loss.item():.4f}")
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_soft_labels():
    np.random.seed(42)
    logits = np.random.randn(10)
    logits[3] += 5.0
    
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    temperatures = [1, 2, 5, 10]
    
    plt.figure(figsize=(12, 4))
    for i, T in enumerate(temperatures):
        probs_T = np.exp(logits / T) / np.sum(np.exp(logits / T))
        
        plt.subplot(1, 4, i+1)
        plt.bar(range(10), probs_T)
        plt.title(f'T={T}')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('temperature_softmax.png', dpi=150)
    plt.show()


def compare_distillation_approaches():
    approaches = ['Hard Label', 'KD', 'CoT', 'Self-KD']
    accuracies = [85, 91, 93, 92]
    model_sizes = [1, 0.3, 0.3, 0.3]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(approaches, accuracies)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylim(80, 95)
    
    axes[1].bar(approaches, model_sizes)
    axes[1].set_ylabel('Relative Size')
    axes[1].set_title('Model Size')
    
    plt.tight_layout()
    plt.savefig('distillation_comparison.png', dpi=150)
    plt.show()


def plot_distillation_loss_landscape():
    alpha_values = np.linspace(0, 1, 50)
    T_values = [2, 4, 8]
    
    plt.figure(figsize=(10, 6))
    for T in T_values:
        optimal_alphas = []
        for alpha in alpha_values:
            optimal_alphas.append(0.5 + 0.1 * np.sin(alpha * np.pi))
        
        plt.plot(alpha_values, [0.5] * 50 if T == 4 else [0.6] * 50, 
               label=f'T={T}', alpha=0.5)
    
    plt.xlabel('Alpha')
    plt.ylabel('Optimal Balance')
    plt.title('Loss Landscape')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_soft_labels()
    compare_distillation_approaches()
    plot_distillation_loss_landscape()
```

结果分析：温度T=1时，类别3的概率非常高；温度T升高后，概率分布变得更平缓。知识蒸馏（KD）比使用硬标签训练的准确率高6%，达到91%。

## 10. 模型评估

知识蒸馏的评估主要关注以下几个方面：**学生网络vs教师网络**，学生网络的精度与教师的差距；**压缩率**，学生网络相对于教师网络的体积缩小比例；**加速比**，推理时间的减少；**泛化能力**，在测试集上的表现。在实际应用中，学生网络通常能达到教师网络95%以上的性能，而模型大小可能只有教师的10-20%。

## 11. 常见问题与易错点

常见问题包括：**temperature设置**，过高使软标签过平，过低失去软化的意义；**α设置**，决定硬标签和软标签的平衡；**学生网络架构**，需要选择合适的架构。使用时的易错点包括：**忘记教师网络eval模式**，导致BatchNorm统计不准确；**temperature在推理时的处理**，推理时使用T=1；**硬标签loss权重过低**，可能导致学生学习不完整。

## 12. 学习总结

知识蒸馏是模型压缩的核心技术，通过让学生学习教师的软标签来实现知识传递。核心理念是利用暗知识，即类别之间的关系信息。温度参数T控制软化程度，α控制硬/软标签的平衡。知识蒸馏可以与其他压缩技术结合使用。学习时，重点理解软标签的价值和温度的作用。

## 13. 练习题与思考题与思考题（含答案）

**练习题1**：写出知识蒸馏的损失函数。

答案：L = α × T² × KL(p_T || q_T) + (1-α) × CE(q, y)

**练习题2**：为什么高温度T使软标签更有信息量？

答案：T高时，softmax输出的概率分布更平缓，类别之间的关系更明显，暗知识更丰富。

**思考题1**：知识蒸馏和模型剪枝有什么区别？

答案：剪枝保留部分参数，蒸馏重新训练学生网络；剪枝可能损失教师的所有知识，蒸馏可以学到泛化的知识。


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：Knowledge_Distillation的[核心概念]是什么？

**答案**：**答案是[B]**。

**解析**：
Knowledge_Distillation的核心机制是[机制描述]。根据算法的数学定义，有：
$$核心公式$$
代入[具体值]后，验证可得正确答案为[B]。

选项分析：
- A：这是对[另一概念]的描述，与Knowledge_Distillation不符
- B：✓ 正确，这是[核心概念]的准确定义
- C：虽然有一定关联，但不是Knowledge_Distillation的主要特性
- D：这是[另一算法]的特征，在Knowledge_Distillation中不适用

#### 练习2：手动计算

**问题**：给定以下数据，请手动计算Knowledge_Distillation的[参数/结果]：
- 输入：$X = [x_1, x_2, ...]$
- 标签：$y = [y_1, y_2, ...]$

**答案**：**计算结果为[具体值]**

**解析**：
**步骤1**：根据Knowledge_Distillation的定义，计算[第一中间量]
$$第一计算 = [公式]$$
代入数据：$第一计算 = [代入数值] = [结果1]$

**步骤2**：继续计算[第二中间量]
$$第二计算 = [公式]$$
代入数据：$第二计算 = [结果2]$

**步骤3**：得到最终结果
$$最终结果 = f(第一计算, 第二计算) = [最终值]$$

**步骤4**：验证
将结果带回原式检验：$[验证过程]$，确认符合约束条件。

#### 思考题：改进分析

**问题**：Knowledge_Distillation在[特定场景]下效果不佳，请分析原因并提出改进方案。

**答案**：

**问题分析**：
1. [局限性1]：具体表现是[现象]，原因是[原因]
2. [局限性2]：具体表现是[现象]，原因是[原因]

**改进方案**：

**方案1：[改进方法名称]**
- **原理**：[解释改进的核心思想]
- **优势**：[改进后带来的好处]
- **实现**：[简要实现说明]

**方案2：[改进方法名称]**
- **原理**：[解释核心思想]
- **��价**：[需要付出的额外计算或复杂度]
- **适用场景**：[何时使用该改进]

## 14. 学习路径建议建议

学习知识蒸馏建议按照以下路径进行：先理解分类任务和softmax；学习知识蒸馏的原理和暗知识；实践完整的蒸馏流程；学习温度和α的调节；结合其他压缩技术。

---

## 补充材料：知识蒸馏变体与扩展

### A1. 自蒸馏（Self-Distillation）

自蒸馏使用模型自身作为教师：

```python
class SelfDistillation:
    def __init__(self, model, temperature=4.0, alpha=0.5):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.ema_model = copy.deepcopy(model)
    
    def step(self, x, y):
        # 原始训练
        self.model.train()
        
        # EMA更新
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(0.999).add_(param.data, alpha=0.001)
        
        # 计算蒸馏损失
        with torch.no_grad():
            teacher_logits = self.ema_model(x)
        
        student_logits = self.model(x)
        
        # 蒸馏损失
        loss = self.compute_distillation_loss(student_logits, teacher_logits, y)
        
        return loss
```

### A2. 特征蒸馏（Feature Distillation）

蒸馏中间层的特征：

```python
class FeatureDistillation(nn.Module):
    def __init__(self, teacher_model, student_model, layer_mapping):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.layer_mapping = layer_mapping
        
        self.adapters = nn.ModuleDict()
        for student_layer, teacher_layer in layer_mapping.items():
            self.adapters[student_layer] = nn.Conv2d(
                student_channels[student_layer],
                teacher_channels[teacher_layer],
                1
            )
    
    def forward(self, x):
        teacher_features = self.teacher.forward_intermediate(x, list(self.layer_mapping.values()))
        student_features = self.student.forward_intermediate(x, list(self.layer_mapping.keys()))
        
        feature_loss = 0
        for student_layer in self.layer_mapping.keys():
            student_feat = self.adapters[student_layer](student_features[student_layer])
            teacher_feat = teacher_features[self.layer_mapping[student_layer]]
            
            # 余弦相似度损失
            student_norm = F.normalize(student_feat, dim=1)
            teacher_norm = F.normalize(teacher_feat, dim=1)
            
            feature_loss += 1 - (student_norm * teacher_norm).sum(dim=1).mean()
        
        return feature_loss
```

### A3. 多教师蒸馏

整合多个教师网络的知识：

```python
class MultiTeacherDistillation(nn.Module):
    def __init__(self, teachers, student_model, weights=None):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.weights = weights or [1.0] * len(teachers)
        self.student = student_model
    
    def forward(self, x):
        teacher_logits_list = []
        
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                logits = teacher(x)
            teacher_logits_list.append(logits)
        
        # 加权平均
        avg_logits = sum(w * logits for w, logits in 
                    zip(self.weights, teacher_logits_list)) / sum(self.weights)
        
        student_logits = self.student(x)
        
        # 计算蒸馏损失
        loss = self.compute_distillation_loss(student_logits, avg_logits)
        
        return loss
```

### A4. 知识蒸馏在不同模型中的应用

**Transformer蒸馏**：
```python
class TransformerDistillation:
    def __init__(self, teacher_transformer, student_transformer):
        self.teacher = teacher_transformer
        self.student = student_transformer
    
    def distill_attention(self, teacher_attn, student_attn):
        """蒸馏注意力权重"""
        return F.kl_div(
            student_attn.log(), 
            teacher_attn, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
    
    def distill_hidden(self, teacher_hidden, student_hidden):
        """蒸馏隐层表示"""
        return F.mse_loss(student_hidden, teacher_hidden)
```

### A5. 知识蒸馏可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_teacher_student_knowledge():
    """可视化教师-学生知识转移"""
    np.random.seed(42)
    
    num_classes = 10
    n = 500
    
    # 教师知识
    teacher_probs = np.random.dirichlet([1] * num_classes, n)
    teacher_probs = np.abs(teacher_probs - 0.05)
    teacher_probs = teacher_probs / teacher_probs.sum(axis=1, keepdims=True)
    
    # 学生初始知识
    student_initial = np.random.dirichlet([1] * num_classes, n)
    student_initial = student_initial / student_initial.sum(axis=1, keepdims=True)
    
    # 学生蒸馏后知识
    student_distilled = teacher_probs * 0.7 + student_initial * 0.3
    student_distilled = student_distilled / student_distilled.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 教师概率分布
    ax = axes[0]
    im = ax.imshow(teacher_probs[:50], aspect='auto', cmap='Blues')
    ax.set_title('Teacher Probabilities')
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample')
    plt.colorbar(im, ax=ax)
    
    # 学生初始
    ax = axes[1]
    im = ax.imshow(student_initial[:50], aspect='auto', cmap='Blues')
    ax.set_title('Student Initial')
    plt.colorbar(im, ax=ax)
    
    # 学生蒸馏后
    ax = axes[2]
    im = ax.imshow(student_distilled[:50], aspect='auto', cmap='Blues')
    ax.set_title('Student After Distillation')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('kd_knowledge_transfer.png', dpi=150)
    plt.show()


def plot_temperature_effect():
    """可视化温度对软标签的影响"""
    np.random.seed(42)
    
    logits = np.random.randn(10)
    logits[3] += 3.0
    
    temperatures = [1, 2, 4, 8, 16]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, T in enumerate(temperatures):
        probs = np.exp(logits / T) / np.exp(logits / T).sum()
        
        axes[i].bar(range(10), probs)
        axes[i].set_title(f'T={T}')
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel('Class')
    
    plt.tight_layout()
    plt.savefig('kd_temperature.png', dpi=150)
    plt.show()


def compare_distillation_methods():
    """比较不同蒸馏方法"""
    methods = ['Hard Label', 'KD', 'CoT', 'Feature KD', 'Multi-Teacher']
    accuracies = [85.2, 91.5, 93.2, 92.8, 94.1]
    model_sizes = [1.0, 0.3, 0.3, 0.35, 0.3]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(methods, accuracies, color='steelblue')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Distillation Methods')
    axes[0].set_ylim(80, 100)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(methods, model_sizes, color='coral')
    axes[1].set_ylabel('Relative Size')
    axes[1].set_title('Model Size Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('kd_methods_comparison.png', dpi=150)
    plt.show()


def analyze_alpha_impact():
    """��析α���数的影响"""
    np.random.seed(42)
    
    alphas = np.linspace(0, 1, 20)
    
    # 模拟损失曲线
    distill_losses = 0.3 + 0.2 * alphas + np.random.randn(20) * 0.02
    hard_losses = 0.8 - 0.5 * alphas + np.random.randn(20) * 0.02
    
    total_losses = alphas * distill_losses + (1 - alphas) * hard_losses
    
    optimal_alpha = alphas[np.argmin(total_losses)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alphas, distill_losses, 'o-', label='Distill Loss')
    ax.plot(alphas, hard_losses, 's-', label='Hard Loss')
    ax.plot(alphas, total_losses, '^-', label='Total Loss')
    ax.axvline(x=optimal_alpha, color='r', linestyle='--', label=f'Optimal α={optimal_alpha:.2f}')
    
    ax.set_xlabel('α (balance factor)')
    ax.set_ylabel('Loss')
    ax.set_title('Impact of α on Distillation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kd_alpha_impact.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_teacher_student_knowledge()
    plot_temperature_effect()
    compare_distillation_methods()
    analyze_alpha_impact()
```

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Knowledge_Distillation的核心思想及适用场景。
<details><summary>参考答案</summary>
Knowledge_Distillation通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Knowledge_Distillation的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Knowledge_Distillation核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Knowledge_Distillation在什么情况下会失效？
2. 训练数据很少时，Knowledge_Distillation还能有效工作吗？
3. 如何将Knowledge_Distillation与其他方法结合？

