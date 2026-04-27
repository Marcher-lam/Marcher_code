# DeiT（Data-efficient Image Transformers）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
DeiT（Data-efficient Image Transformers）是Facebook AI于2020年提出的数据高效视觉Transformer，通过**知识蒸馏**和**强数据增强**策略，使Vision Transformer（ViT）能在中等规模数据集（ImageNet-1K，120万图片）上从零开始训练成功，彻底打破了ViT对JFT-300M等超大数据集的依赖。

### 1.2 直觉类比
DeiT就像给一个学生（ViT）配了一位经验丰富的老师（CNN教师）——学生不仅自己读课本（ImageNet数据），还通过"蒸馏token"从老师那里学习解题技巧。老师教学生"这个任务应该关注图像的哪些区域"，帮助学生更快更好地学习。

### 1.3 历史背景
- **2020年10月**：ViT提出，但需要3亿张图像的JFT-300M预训练
- **2020年12月**：Facebook AI提出DeiT，只需ImageNet-1K
- **2021年**：DeiT成为CV领域重要基线
- **影响**：推动了数据高效ViT的研究（CaiT, T2T-ViT等）

### 1.4 算法定位
DeiT是**数据高效的Vision Transformer**，使用有监督预训练+知识蒸馏，属于ViT的改进变体。

---

## 2. 核心原理

### 2.1 蒸馏token（Distillation Token）
DeiT在ViT的基础上增加一个特殊的蒸馏token：

```
输入: [class_token, dist_token, patch_1, patch_2, ..., patch_N]
```

- **class token**：用于分类预测（同ViT）
- **distillation token**：用于学习教师模型的输出
- 与class token类似，dist token通过自注意力与其他token交互
- 最终用dist token的输出匹配教师的预测

### 2.2 知识蒸馏策略
DeiT使用**硬标签蒸馏（Hard-label Distillation）**：

$$L = \lambda L_{CE}(\psi(Z_s), y) + (1-\lambda) L_{CE}(\psi(Z_s), y_t)$$

其中：
- $\psi(Z_s)$ 是student（DeiT）的预测
- $y$ 是ground truth标签
- $y_t = \arg\max \psi(Z_t)$ 是teacher（CNN）的预测（硬标签）
- $\lambda$ 控制蒸馏强度（DeiT使用0.5）

### 2.3 训练配方
DeiT使用强大的数据增强组合：
1. **RandAugment**：随机应用图像增强操作
2. **CutMix**：混合两张图像的裁切区域
3. **MixUp**：线性插值混合两张图像
4. **Random Erasing**：随机擦除图像区域
5. **更长的训练**：300~600 epoch（vs ViT的90 epoch）

---

## 3. 数学公式与推导

### 3.1 硬标签蒸馏损失
传统软标签蒸馏（KD）：$L = KL(\text{teacher\_softmax}, \text{student\_softmax})$

DeiT的硬标签蒸馏：

$$L = \underbrace{(1-\lambda) H(y, \psi(Z_s))}_{\text{ground truth loss}} + \underbrace{\lambda H(y_t, \psi(Z_s))}_{\text{distillation loss}}$$

其中 $y_t = \arg\max_c \psi_t(Z_t)[c]$ 是教师的硬决策。

### 3.2 蒸馏token的前向传播
class token和distillation token并行计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

$$Z_{\text{class}} = \text{MLP}(\text{Attention}(q_{\text{cls}}, K, V))$$
$$Z_{\text{dist}} = \text{MLP}(\text{Attention}(q_{\text{dist}}, K, V))$$

两个token共享Transformer层，但最后的分类头不同。

### 3.3 训练动量教师
DeiT的teacher模型（RegNetY）是预训练的CNN，参数固定。也可以使用指数移动平均（EMA）的student作为教师：

$$\theta_t = \alpha \theta_t + (1-\alpha) \theta_s$$

### 3.4 数据增强公式
**CutMix**：
$$\tilde{x} = M \odot x_A + (1-M) \odot x_B$$
$$\tilde{y} = \lambda y_A + (1-\lambda) y_B$$

其中 $M \in \{0,1\}^{W \times H}$ 是二值掩码，$\lambda \sim \text{Beta}(1,1)$。

**MixUp**：
$$\tilde{x} = \lambda x_A + (1-\lambda) x_B$$
$$\tilde{y} = \lambda y_A + (1-\lambda) y_B$$

其中 $\lambda \sim \text{Beta}(0.2, 0.2)$。

---

## 4. 训练过程讲解

### 4.1 训练步骤（每个batch）
1. 从ImageNet-1K采样一个batch的图像
2. 应用RandAugment + CutMix/MixUp等数据增强
3. 将图像输入DeiT（student），输出class和dist两个预测
4. 将图像输入预训练的CNN（teacher），获得教师预测
5. 计算硬标签$y_t = \arg\max \text{teacher\_pred}$
6. 计算损失：$L = 0.5*CE(student\_class, y) + 0.5*CE(student\_dist, y_t)$
7. 反向传播更新student参数

### 4.2 推理阶段
- 可使用class token或distillation token的输出
- 或两者平均：$y_{\text{final}} = (\psi_{\text{class}} + \psi_{\text{dist}}) / 2$
- 通常两者性能相近，可以只用一个减少计算

### 4.3 与ViT的训练对比
| 方面 | ViT | DeiT |
|------|-----|------|
| 预训练数据 | JFT-300M (3亿) | ImageNet-1K (120万) |
| Epoch | 90 | 300-600 |
| 数据增强 | 基础 | RandAugment+CutMix+MixUp |
| 蒸馏 | 无 | CNN蒸馏 |
| 训练结果 | 77.9% (300M数据) | 83.1% (1.2M数据) |

---

## 5. 应用场景

| 场景 | 说明 | DeiT优势 |
|------|------|----------|
| ImageNet分类 | 标准图像分类任务 | 数据高效，无需繁重预训练 |
| 迁移学习 | 作为骨干网络 | 预训练权重可直接微调 |
| 目标检测 | 作为backbone | 特征提取效果好 |
| 语义分割 | 像素级分类 | DeiT特征层次丰富 |

---

## 6. 优缺点分析

### 优点
1. **数据高效**：只需ImageNet-1K即可训练
2. **性能优秀**：DeiT-B达83.1%（超越同等规模的CNN）
3. **训练稳定**：强增强和蒸馏防止过拟合
4. **灵活架构**：支持多种ViT变体

### 缺点
1. **依赖CNN教师**：需要预训练的CNN模型
2. **训练时间长**：300-600 epoch，计算量大
3. **蒸馏token冗余**：推理时与class token功能重叠
4. **复杂数据增强**：需要仔细调参

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeiTModel, DeiTImageProcessor, DeiTForImageClassification
import math

class DeiTClassifier(nn.Module):
    """DeiT图像分类器"""
    def __init__(self, model_name='facebook/deit-base-distilled-patch16-224', num_classes=1000):
        super().__init__()
        self.model = DeiTForImageClassification.from_pretrained(model_name)
        self.processor = DeiTImageProcessor.from_pretrained(model_name)
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, 3, 224, 224] 图像张量
        Returns:
            logits: [B, num_classes]
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def predict(self, image):
        """
        单图像预测
        Args:
            image: PIL Image or numpy array
        """
        inputs = self.processor(images=image, return_tensors='pt')
        with torch.no_grad():
            logits = self.forward(inputs.pixel_values)
            probs = F.softmax(logits, dim=-1)
            pred_idx = logits.argmax(dim=-1).item()
            pred_prob = probs[0, pred_idx].item()
        return pred_idx, pred_prob


class DeiTDistillationTrainer:
    """DeiT蒸馏训练器"""
    def __init__(self, student_model='facebook/deit-base-distilled-patch16-224'):
        self.student = DeiTForImageClassification.from_pretrained(student_model)
        self.teacher = None  # 需要在外部加载预训练CNN
        
    def set_teacher(self, teacher_model):
        """设置教师模型（如RegNetY）"""
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0):
        """
        计算蒸馏损失（硬标签版本）
        Args:
            student_logits: [B, C] student的class token输出
            teacher_logits: [B, C] teacher的logits
            labels: [B] ground truth标签
        """
        # Ground truth交叉熵损失
        loss_gt = F.cross_entropy(student_logits, labels)
        
        # 硬标签蒸馏损失（使用教师的argmax作为目标）
        with torch.no_grad():
            teacher_hard_labels = teacher_logits.argmax(dim=-1)
        loss_distill = F.cross_entropy(student_logits, teacher_hard_labels)
        
        # 总损失
        loss = alpha * loss_gt + (1 - alpha) * loss_distill
        return loss


class DeiTTrainingAugmentation:
    """DeiT训练使用的数据增强组合"""
    @staticmethod
    def rand_augment(image, num_ops=2, magnitude=9):
        """
        RandAugment: 随机选择num_ops个图像增强操作
        实际操作列表: rotate, shear_x, shear_y, translate_x, 
                     translate_y, color, brightness, contrast, etc.
        """
        # 简化的增强示例
        # 实际使用torchvision的RandAugment
        pass
    
    @staticmethod
    def cutmix(image1, image2, label1, label2, alpha=1.0):
        """CutMix增强"""
        lam = torch.distributions.Beta(alpha, alpha).sample()
        B, C, H, W = image1.shape
        
        # 裁切区域
        rx = torch.randint(W, (1,))
        ry = torch.randint(H, (1,))
        rw = int(W * torch.sqrt(1 - lam))
        rh = int(H * torch.sqrt(1 - lam))
        
        # 混合
        mixed_image = image1.clone()
        mixed_image[:, :, ry:ry+rh, rx:rx+rw] = image2[:, :, ry:ry+rh, rx:rx+rw]
        
        # 混合标签
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


def test_deit():
    """测试DeiT模型"""
    model = DeiTClassifier()
    
    # 模拟图像输入
    dummy_images = torch.randn(2, 3, 224, 224)
    logits = model(dummy_images)
    print(f"DeiT分类输出: {logits.shape}")  # [2, 1000]
    
    print("DeiT测试通过！")

if __name__ == "__main__":
    test_deit()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenDeiT(nn.Module):
    """DeiT核心逻辑手工实现（含蒸馏token）"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # 图像分块嵌入
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token + Distillation token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置编码（+2是因为两个特殊token）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 2, embed_dim))
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 两个分类头（class和distillation各自独立）
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.dist_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            class_logits: [B, num_classes]
            dist_logits: [B, num_classes]
        """
        B = x.shape[0]
        
        # 图像分块
        x = self.patch_embed(x)  # [B, D, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        
        # 拼接class token和distillation token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)  # [B, 2+num_patches, D]
        
        # 位置编码
        x = x + self.pos_embed
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # 取class token和distillation token的输出
        class_out = x[:, 0]  # [B, D]
        dist_out = x[:, 1]   # [B, D]
        
        class_logits = self.class_head(class_out)
        dist_logits = self.dist_head(dist_out)
        
        return class_logits, dist_logits
    
    def forward_with_distill(self, x):
        """推理时返回蒸馏后的最终预测"""
        class_logits, dist_logits = self.forward(x)
        return (class_logits + dist_logits) / 2


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def test_handwritten_deit():
    """测试手工实现"""
    model = HandwrittenDeiT(embed_dim=256, depth=6, num_heads=8)
    x = torch.randn(2, 3, 224, 224)
    cls_logits, dist_logits = model(x)
    print(f"手工DeiT: class={cls_logits.shape}, dist={dist_logits.shape}")
    
    # 蒸馏损失计算
    teacher_logits = torch.randn(2, 1000)
    labels = torch.randint(0, 1000, (2,))
    
    loss_gt = F.cross_entropy(cls_logits, labels)
    teacher_hard = teacher_logits.argmax(dim=-1)
    loss_distill = F.cross_entropy(cls_logits, teacher_hard)
    total_loss = 0.5 * loss_gt + 0.5 * loss_distill
    print(f"蒸馏损失: {total_loss.item():.4f}")

if __name__ == "__main__":
    test_handwritten_deit()
```

---

## 9. 可视化与结果理解

### 9.1 蒸馏token的注意力图
可视化显示distillation token的关注区域与class token不同：
- class token关注全局判别区域
- dist token关注教师模型认为重要的区域

### 9.2 数据增强的效果
| 增强策略 | DeiT-S (ImageNet) |
|----------|-------------------|
| 无增强 | 72.1% |
| +RandAugment | 76.3% |
| +RandAugment+MixUp | 78.4% |
| +RandAugment+MixUp+CutMix | 79.8% |

### 9.3 蒸馏方式对比
| 蒸馏方式 | DeiT-S |
|----------|--------|
| 无蒸馏 | 78.4% |
| 软标签蒸馏 | 79.5% |
| 硬标签蒸馏 | 79.8% |
| 蒸馏+CNN教师(RegNetY) | 81.8% |

---

## 10. 模型评估

| 模型 | ImageNet top-1 | 预训练数据 | 参数量 |
|------|---------------|-----------|--------|
| DeiT-Ti | 72.2% | ImageNet-1K | 5M |
| DeiT-S | 79.8% | ImageNet-1K | 22M |
| DeiT-B | 81.8% | ImageNet-1K | 86M |
| DeiT-B (蒸馏) | 83.1% | ImageNet-1K | 86M |
| ViT-B/16 | 77.9% | JFT-300M | 86M |

---

## 11. 常见问题

### Q1: DeiT的蒸馏token和class token有什么区别？
A: class token学习预测ground truth标签，dist token学习模仿教师的预测。两者共享Transformer层但最后用不同的线性分类头。

### Q2: DeiT为什么需要300-600 epoch？
A: 强数据增强（CutMix, MixUp等）虽然提升了泛化性，但也让每个epoch的信息量减少（图像被严重修改），需要更多epoch才能充分学习。

### Q3: 为什么硬标签蒸馏比软标签好？
A: DeiT论文的实验发现硬标签蒸馏对ViT更有效。可能是因为ViT的学习方式对确切的标签信号更敏感。

### Q4: DeiT能否不用CNN教师？
A: 可以。DeiT也实验了自蒸馏（student自己作为EMA教师），但效果不如CNN教师。

---

## 12. 学习总结

DeiT证明了**ViT可以在没有海量数据的情况下成功训练**，关键在于：
1. **知识蒸馏**：CNN教师指导学生
2. **强数据增强**：防止过拟合
3. **更长的训练**：充分学习增强后的数据
4. **蒸馏token**：为蒸馏提供专用的表示通道

---

## 13. 练习题与思考题（含答案）

### 习题1：DeiT如何实现数据高效？
**答案**：1）使用强数据增强（RandAugment+CutMix+MixUp）防止过拟合；2）引入CNN蒸馏提升泛化；3）训练更多epoch充分利用数据。

### 习题2：硬标签蒸馏和软标签蒸馏的区别？
**答案**：硬标签蒸馏使用教师预测的 argmax 作为目标（即"教师认为的类别"），软标签使用教师的 softmax 概率分布。硬标签更简单直接。

### 习题3：如果deit的dist_token和class_token输出差异很大，说明什么？
**答案**：说明student在某些样本上无法很好地模仿teacher，可能这些样本对student来说比较困难。此时可以检查蒸馏损失的权重是否需要调整。

### 习题4：思考：DeiT的自蒸馏（self-distillation）怎么做？
**答案**：使用student自身的指数移动平均（EMA）作为teacher。每一轮训练，teacher参数是student历史参数的加权平均，提供更稳定的目标。

---

## 14. 学习路径建议

### 前置
- ViT、Transformer、知识蒸馏

### 平行
- Swin Transformer、T2T-ViT、CaiT、PVT

### 进阶
- BEiT、MAE、DINO（自监督版本）
- DeiT III（改进版）
