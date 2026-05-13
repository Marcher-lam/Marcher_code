# CaiT (Class-Attention in Transformers) 学习文档

## 1. 算法基础认知

### 1.1 算法要解决什么问题

在标准 ViT 中，class token（CLS）从一开始就和图像 patch token 一起参与自注意力计算。这意味着：

- **class token 的注意力被"污染"**：在浅层中，class token 需要关注局部细节来帮助特征提取，但这种"细节关注"的能力到了深层反而变成了干扰——深层中 class token 应该关注全局语义信息，但它仍然受到与其他 patch token 同等地位的自注意力限制。
- **自注意力计算效率低**：class token 和 patch token 的关系在每个注意力层都被重复计算，但 class token 实际上只需要在最后几层"读取"patch token 的全局信息。

CaiT（Class-Attention in Transformers）由 Touvron 等人于 2021 年提出，其核心思想是：**将"自注意力"（Self-Attention）和"类注意力"（Class-Attention）分离开来**。

### 1.2 核心思路概览

CaiT 的设计分为两个阶段：

```
第一阶段: 多个 Self-Attention 块（仅 patch tokens 之间交互）
第二阶段: 多个 Class-Attention 块（class token 与 patch tokens 交互，但 patch tokens 之间不交互）
```

**Self-Attention 块**：
- 只有 patch tokens 参与
- class token 不被引入
- 用于 patch token 之间的信息交换

**Class-Attention 块**：
- class token 作为 query，patch tokens 作为 key 和 value
- patch tokens 之间无交互
- class token "读取" patch tokens 中的全局信息

这种分离设计使得浅层网络专注于 patch 间的局部关系，而 class token 仅在最后阶段进行"一次性"的全局信息聚合。

### 1.3 整体架构

```
输入图像 → Patch Embedding → [SA 块 × L] → class token 插入 → [CA 块 × M] → MLP Head
                                       ↓
SA 块 (Self-Attention): LayerNorm → MHA(仅 patch) → LayerNorm → FFN
CA 块 (Class-Attention): LayerNorm → MHA(cls→patch) → LayerNorm → FFN(仅 cls token)
```

## 2. 核心原理

### 2.1 自注意力 vs 类注意力

**标准自注意力（Self-Attention）**：
- Q, K, V 都来自同一组输入
- 每个 token 既做 query 又做 key
- 计算复杂度: O(N²·D)

**类注意力（Class-Attention）**：
- Q 来自 class token
- K, V 来自 patch tokens
- 只有 class token "关注" patches，patches 之间不互相关注
- 计算复杂度: O(N·D)（因为只有 cls→patches 的交互）

### 2.2 为什么需要分离？

假设我们有一个 24 层的 Transformer：
- **标准 ViT**：24 层都是 full self-attention（cls + patches），共 24 次 O((N+1)²·D)
- **CaiT**：12 层 SA（仅 patches）+ 12 层 CA（cls→patches），patches 之间的交互被限制在前 12 层

这种分离的直观解释是：
1. 浅层中，patches 之间需要充分交互来建立局部和全局关系
2. 深层中，class token 只需要"读出"已经在 patches 中编码好的全局信息
3. class token 不应该反过来"干扰"已经稳定的 patch 表示

### 2.3 CA 块的详细设计

每个 Class-Attention 块包含两个子层：

**子层 1：多头类注意力（MCA - Multi-head Class-Attention）**
```
输入: cls_token (1, D), patch_tokens (N, D)
Q = cls_token · W_Q  (1, D) → (1, d_k)
K = patch_tokens · W_K  (N, D) → (N, d_k)
V = patch_tokens · W_V  (N, D) → (N, d_v)

attn_weights = softmax(Q · K^T / √d_k)  (1, N)
output = attn_weights · V  (1, d_v)

cls_out = output · W_O  (1, D)
```

注意：这里是 class token 作为 query，patches 作为 key/value，所以注意力权重形状为 (1, N)，而不是 (N+1, N+1)。

**子层 2：FFN（仅作用于 class token）**
```
cls_out = cls_out + FFN(LayerNorm(cls_out))
```

关键区别：在 CA 块中，FFN 只更新 class token，patch tokens 保持不变。

### 2.4 与 ViT 的架构对比

| 组件 | ViT | CaiT |
|------|-----|------|
| 浅层 | cls + patches 自注意力 | 仅 patches 自注意力 |
| 深层 | cls + patches 自注意力 | cls→patches 类注意力 |
| class token | 全程参与 | 仅在后半部分引入 |
| FFN | 作用于所有 token | 在 SA 块作用于 patches，在 CA 块作用于 cls |
| 位置编码 | 绝对位置编码 | 相对位置编码（或 LayerScale） |

### 2.5 LayerScale

CaiT 还引入了一个重要的训练技巧——**LayerScale**。LayerScale 在每个残差块的输出上乘以一个可学习的对角矩阵：

```
x = x + diag(λ₁, ..., λ_D) · Sublayer(LayerNorm(x))
```

其中 λ_i 初始化为一个很小的值（如 0.1），并在训练过程中学习。

LayerScale 的作用是：
1. 在训练初期有效抑制大梯度，稳定训练
2. 让深层网络更容易训练（类似 ResNet 中残差连接的 scaling）
3. 与 CaiT 的分离设计相辅相成

## 3. 数学公式与推导

### 3.1 Self-Attention 块

输入：patch tokens P ∈ ℝ^{N×D}（不含 class token）

```
P' = P + MSA(LN(P))
P'' = P' + FFN(LN(P'))
```

其中 MSA 是多头自注意力：
```
MSA(P) = Concat(head₁, ..., head_H) · W_O
head_h = softmax((P·W_Q_h) · (P·W_K_h)^T / √d_h) · (P·W_V_h)
```

### 3.2 Class-Attention 块

输入：class token c ∈ ℝ^{1×D}，patch tokens P ∈ ℝ^{N×D}

```
c' = c + MCA(LN(c), LN(P))
c'' = c' + FFN(LN(c'))
P' = P  (patch tokens 不更新)
```

其中 MCA 是多头类注意力：
```
Q = c · W_Q ∈ ℝ^{1×d_k}
K = P · W_K ∈ ℝ^{N×d_k}
V = P · W_V ∈ ℝ^{N×d_v}

A = softmax(Q · K^T / √d_k) ∈ ℝ^{1×N}
O = A · V ∈ ℝ^{1×d_v}

MCA(c, P) = O · W_O ∈ ℝ^{1×D}
```

注意：class token 的维度变化：
- 输入: (1, D)
- 通过 Q 投影: (1, d_k)  
- 与 K (N, d_k) 点积: (1, N)
- 加权 V (N, d_v): (1, d_v)
- 输出投影: (1, D)

### 3.3 LayerScale

每个子层（MSA/MCA/FFN）的输出乘以可学习的缩放向量：

```
x = x + diag(λ) · Sublayer(LN(x))
```

其中 λ ∈ ℝ^D，初始化为小值。

对于第 l 层的第 s 个子层：
```
λ_l,s,i = α^l · sign(l, s, i)
```

实践中通常全部初始化为 0.1 或更小。

### 3.4 总体计算量分析

假设 N 个 patches，L 个 SA 块，M 个 CA 块：

**ViT 计算量**：
- (L+M) × (N+1)² × D × H (自注意力部分)
- (L+M) × (N+1) × D × FFN_ratio (FFN 部分)

**CaiT 计算量**：
- L × N² × D × H (SA 部分，仅 patches)
- M × N × D × H (CA 部分，cls→patches)
- L × N × D × FFN_ratio (SA 的 FFN)
- M × 1 × D × FFN_ratio (CA 的 FFN)

当 N=196（224²/16²）时，CA 比 SA 节省约 N 倍的计算量。

## 4. 训练过程讲解

### 4.1 训练配置

CaiT 的训练配置与 DeiT 相似，但加入了一些改良：

- **优化器**：AdamW (β₁=0.9, β₂=0.999)
- **学习率**：5e-4（cosine schedule）
- **权重衰减**：0.05
- **热身 epoch**：5
- **批次大小**：1024
- **训练轮数**：400 epoch（比 DeiT 多 100 epoch）
- **标签平滑**：0.1
- **Dropout**：0.0（不使用 Dropout，而是使用 Stochastic Depth）
- **Stochastic Depth**：0.1-0.5（随深度线性增加）

### 4.2 数据增强

- RandomResizedCrop (224×224)
- RandomHorizontalFlip
- 3-Augment (简化版 RandAugment)
- Mixup (α=0.8)
- CutMix (α=1.0)

### 4.3 LayerScale 的初始化策略

LayerScale 的初始化非常重要：

```python
# CaiT 的 LayerScale 初始化
def init_layerscale(layer, dim, init_value=0.1):
    """对每层的输出缩放进行初始化"""
    if hasattr(layer, 'ls1'):
        nn.init.constant_(layer.ls1, init_value)
    if hasattr(layer, 'ls2'):
        nn.init.constant_(layer.ls2, init_value)
```

初始值的选择：
- 对深层（ca 块）：init_value = 1e-5
- 对浅层（sa 块）：init_value = 0.1
- 一般统一使用 0.1 即可

### 4.4 训练流程

```python
for epoch in range(400):
    model.train()
    for images, labels in dataloader:
        # Mixup/CutMix
        images, labels_a, labels_b, lam = mixup_cutmix(images, labels)

        # 前向传播
        outputs = model(images)

        # 损失
        loss = lam * CE(outputs, labels_a) + (1-lam) * CE(outputs, labels_b)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    val_acc = validate(model, val_loader)
    print(f"Epoch {epoch}: {val_acc:.2f}%")
```

## 5. 应用场景

### 5.1 图像分类

CaiT 在 ImageNet 上取得了 SOTA 性能：
- CaiT-S：82.0% top-1（~25M 参数）
- CaiT-M：83.4% top-1（~45M 参数）
- CaiT-L：84.7% top-1（~80M 参数）

### 5.2 迁移学习

CaiT 在迁移学习场景中表现优异：
- 在 CIFAR-100 上达到 95.3%（ViT-B 为 93.8%）
- 在 Oxford Flowers 上达到 99.2%
- 在 Stanford Cars 上达到 94.1%

### 5.3 知识蒸馏

CaiT 天然适合知识蒸馏：
- 教师模型提供 class token 的 soft label
- 学生模型仅需要学习 class token 的输出
- 蒸馏效率高

## 6. 优缺点分析

### 6.1 优点

1. **计算效率高**：CA 块的计算复杂度为 O(N) 而非 O(N²)
2. **训练更稳定**：LayerScale 提高了训练稳定性
3. **分离设计合理**：SA 负责 patches 交互，CA 负责 cls 读取，职责明确
4. **迁移性能好**：在多个下游任务上表现优异

### 6.2 缺点

1. **架构复杂度增加**：需要区分 SA 块和 CA 块
2. **超参数更多**：SA 块和 CA 块的数量比例需要调优
3. **深层 patch 表示不再更新**：在 CA 阶段，patch tokens 不再交互，可能会丢失精细信息
4. **不适合密集预测任务**：由于 class token 设计主要是为分类服务，用于分割/检测时需要修改

## 7. 调库实现

### 7.1 使用 timm 库调用 CaiT

```python
import torch
from timm.models import create_model

# 创建 CaiT 模型
model = create_model(
    'cait_s24_224',  # CaiT-S 24 层版本
    pretrained=True,
    num_classes=1000,
)

# 查看模型结构
print(model)

# 测试前向传播
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"输出形状: {output.shape}")  # (2, 1000)

# 获取注意力权重（CaiT 的 CA 块支持返回注意力）
model.eval()
with torch.no_grad():
    output = model(x)
    predicted = output.argmax(dim=1)
    print(f"预测类别: {predicted}")
```

### 7.2 完整的训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import create_model
from timm.scheduler import CosineLRScheduler
import numpy as np


# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-5
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# CIFAR-100
train_dataset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transform
)
val_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 创建 CaiT 模型
model = create_model(
    'cait_xxs24_224',  # CaiT 超小版本
    pretrained=False,
    num_classes=NUM_CLASSES,
)
model = model.to(DEVICE)

# 损失函数
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

# 学习率调度
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=EPOCHS,
    lr_min=1e-6,
    warmup_t=5,
    warmup_lr_init=1e-6,
)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


# 训练循环
print("开始训练 CaiT...")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    val_loss, val_acc = validate(
        model, val_loader, criterion, DEVICE
    )
    scheduler.step(epoch)

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "cait_best.pth")
        print(f"  → 新最佳模型保存 (acc={val_acc:.2f}%)")

print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
```

## 8. 手工代码实现

### 8.1 CaiT 的完整手工实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerScale(nn.Module):
    """
    LayerScale: 对残差块的输出进行逐元素缩放
    初始化为较小值以稳定深层训练
    """
    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ClassAttention(nn.Module):
    """
    Class-Attention (多头类注意力)
    class token 作为 query，patch tokens 作为 key 和 value
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, cls_token: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            cls_token: (B, 1, D) - class token
            x: (B, N, D) - patch tokens
        返回:
            (B, 1, D) - 更新后的 class token
        """
        B, N, D = x.shape

        # Q: 仅来自 class token
        Q = self.q(cls_token)  # (B, 1, D)
        K = self.k(x)  # (B, N, D)
        V = self.v(x)  # (B, N, D)

        # 分头
        Q = Q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, 1, Dh)
        K = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        V = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, Dh)

        # 注意力: (B, H, 1, Dh) @ (B, H, Dh, N) -> (B, H, 1, N)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和: (B, H, 1, N) @ (B, H, N, Dh) -> (B, H, 1, Dh)
        out = attn @ V
        out = out.permute(0, 2, 1, 3).reshape(B, 1, D)

        # 输出投影
        out = self.proj(out)

        return out


class SelfAttention(nn.Module):
    """
    标准多头自注意力（仅 patch tokens 之间）
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # QKV
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # 各 (B, H, N, Dh)

        # 注意力
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = attn @ V  # (B, H, N, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.proj(out)
        return out


class MLP(nn.Module):
    """
    MLP 前馈网络
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SABlock(nn.Module):
    """
    Self-Attention 块（仅处理 patch tokens）
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, init_values: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + LayerScale + 残差
        x = x + self.ls1(self.attn(self.norm1(x)))
        # MLP + LayerScale + 残差
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class CABlock(nn.Module):
    """
    Class-Attention 块
    class token 读取 patch tokens 的信息
    patch tokens 保持不变
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, init_values: float = 1e-5):
        super().__init__()

        # Class-Attention 子层
        self.norm1_cls = nn.LayerNorm(dim)
        self.norm1_x = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)

        # MLP 子层（仅更新 class token）
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, cls_token: torch.Tensor, x: torch.Tensor) -> tuple:
        """
        参数:
            cls_token: (B, 1, D)
            x: (B, N, D)
        返回:
            cls_token: (B, 1, D) - 更新后的 class token
            x: (B, N, D) - 保持不变
        """
        # Class-Attention + LayerScale + 残差
        cls_token = cls_token + self.ls1(
            self.attn(self.norm1_cls(cls_token), self.norm1_x(x))
        )
        # MLP + LayerScale + 残差（仅 cls token）
        cls_token = cls_token + self.ls2(self.mlp(self.norm2(cls_token)))

        return cls_token, x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: 将图像分割为 patches 并投影到 D 维
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class CaiT(nn.Module):
    """
    CaiT: Class-Attention in Transformers
    分离自注意力和类注意力
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 24,           # 总深度
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sa_depth: int = 12,        # Self-Attention 块数量
        ca_depth: int = 12,        # Class-Attention 块数量
        init_values_sa: float = 0.1,
        init_values_ca: float = 1e-5,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码（仅 patch tokens）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Self-Attention 块
        self.sa_blocks = nn.ModuleList([
            SABlock(embed_dim, num_heads, mlp_ratio, dropout, init_values_sa)
            for _ in range(sa_depth)
        ])

        # Class token（在 SA 阶段之后才引入）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Class-Attention 块
        self.ca_blocks = nn.ModuleList([
            CABlock(embed_dim, num_heads, mlp_ratio, dropout, init_values_ca)
            for _ in range(ca_depth)
        ])

        # 最终归一化和分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Self-Attention 阶段
        for sa_block in self.sa_blocks:
            x = sa_block(x)

        # 插入 class token
        cls_token = self.cls_token.expand(B, -1, -1)

        # Class-Attention 阶段
        for ca_block in self.ca_blocks:
            cls_token, x = ca_block(cls_token, x)

        # 最终 class token 用于分类
        cls_token = self.norm(cls_token)
        out = self.head(cls_token.squeeze(1))

        return out


def test_cait():
    """测试 CaiT 前向传播"""
    model = CaiT(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=384,
        depth=24,
        num_heads=8,
        sa_depth=12,
        ca_depth=12,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_cait()
```

### 8.2 分析 SA/CA 比例的影响

```python
def analyze_sa_ca_ratio():
    """
    分析 Self-Attention 和 Class-Attention 块的比例对模型的影响
    """
    total_depth = 24
    ratios = [
        (12, 12),  # 1:1 - 默认
        (16, 8),   # 2:1 - 更多 SA
        (8, 16),   # 1:2 - 更多 CA
        (20, 4),   # 5:1 - 几乎全是 SA
        (4, 20),   # 1:5 - 几乎全是 CA
    ]

    for sa_d, ca_d in ratios:
        model = CaiT(
            img_size=224,
            patch_size=16,
            num_classes=100,
            embed_dim=384,
            depth=sa_d + ca_d,
            num_heads=8,
            sa_depth=sa_d,
            ca_depth=ca_d,
        )

        params = sum(p.numel() for p in model.parameters())
        print(f"SA={sa_d}, CA={ca_d}: {params:,} parameters")

    print("\n建议: SA 和 CA 比例通常为 1:1 或 2:1")
    print("更多 SA: 更适合需要精细局部特征的任务")
    print("更多 CA: 计算效率更高，适合分类任务")
```

## 9. 可视化与结果理解

### 9.1 注意力图可视化

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_class_attention(model, x, layer_idx=-1):
    """
    可视化 CaiT 中 Class-Attention 块的注意力图
    class token 如何关注不同的 patch 区域
    """
    model.eval()
    attention_maps = []

    # 注册 hook 捕获 ClassAttention 的注意力权重
    def hook_fn(module, input, output):
        # 这里简化处理，实际需要捕获 attention 前的中间值
        pass

    with torch.no_grad():
        output = model(x)

    # 简化版可视化（使用模拟数据）
    # 实际应用中，需要修改 ClassAttention 以返回注意力权重
    H = W = 14  # 224/16 = 14
    num_patches = H * W

    # 模拟 class attention 图
    np.random.seed(42)
    attn_map = np.random.rand(H, W)

    # 模拟一个"关注物体中心"的注意力模式
    center_h, center_w = H // 2, W // 2
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
            attn_map[i, j] = np.exp(-dist / 3.0)

    attn_map /= attn_map.sum()

    plt.figure(figsize=(8, 6))
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Class Token Attention Map')
    plt.axis('off')

    # 标注关注区域
    plt.contour(attn_map, levels=5, colors='white', alpha=0.5, linewidths=0.5)
    plt.tight_layout()
    plt.show()


def visualize_cait_architecture():
    """
    可视化 CaiT 的分离架构
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 标准 ViT 架构
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].set_title('Standard ViT Architecture', fontsize=14)

    # 绘制 layer blocks
    for i in range(12):
        y = 9 - i * 0.7
        rect = plt.Rectangle((1, y-0.25), 8, 0.3, fill=True,
                            facecolor='lightblue', edgecolor='blue', alpha=0.7)
        axes[0].add_patch(rect)
        axes[0].text(5, y, f'SA Block {i+1}\n(cls + patches)', ha='center', va='center', fontsize=8)

    # 标注
    axes[0].arrow(5, 0.5, 0, 0.5, head_width=0.3, head_length=0.3, fc='black', ec='black')
    axes[0].text(5, 0.1, 'CLS token used in ALL layers', ha='center', fontsize=10)
    axes[0].axis('off')

    # 右图: CaiT 架构
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].set_title('CaiT Architecture', fontsize=14)

    # SA blocks
    for i in range(6):
        y = 9 - i * 0.7
        rect = plt.Rectangle((1, y-0.25), 8, 0.3, fill=True,
                            facecolor='lightgreen', edgecolor='green', alpha=0.7)
        axes[1].add_patch(rect)
        axes[1].text(5, y, f'SA Block {i+1}\n(patches only)', ha='center', va='center', fontsize=8)

    # 分隔线
    axes[1].axhline(y=9-6*0.7-0.5, color='red', linestyle='--', linewidth=2)
    axes[1].text(5, 9-6*0.7-0.7, '← Insert CLS token', ha='center', fontsize=10, color='red')

    # CA blocks
    for i in range(6):
        y = 9 - 6*0.7 - 1.2 - i * 0.7
        rect = plt.Rectangle((1, y-0.25), 8, 0.3, fill=True,
                            facecolor='lightsalmon', edgecolor='red', alpha=0.7)
        axes[1].add_patch(rect)
        axes[1].text(5, y, f'CA Block {i+1}\n(cls → patches)', ha='center', va='center', fontsize=8)

    axes[1].arrow(5, y-0.8, 0, 0.5, head_width=0.3, head_length=0.3, fc='black', ec='black')
    axes[1].text(5, y-1.2, 'CLS token used only in CA blocks', ha='center', fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_cait(model, dataloader, device):
    """
    完整的 CaiT 模型评估
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)

    # Top-5 准确率
    all_probs = np.array(all_probs)
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([labels[i] in top5_preds[i] for i in range(len(all_labels))])
    top5_acc = top5_correct.mean()

    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    return {
        'top1': accuracy,
        'top5': top5_acc,
    }
```

### 10.2 与 ViT/DeiT 对比

| 模型 | 参数量 | ImageNet Top-1 | FLOPs | 推理速度 |
|------|--------|----------------|-------|---------|
| ViT-B/16 | 86M | 77.9% | 17.6G | 1x |
| DeiT-B | 86M | 81.8% | 17.6G | 0.95x |
| CaiT-S24 | 25M | 82.0% | 9.2G | 0.85x |
| CaiT-M36 | 45M | 83.4% | 15.5G | 0.75x |
| CaiT-L48 | 80M | 84.7% | 26.4G | 0.6x |

CaiT 以更低的计算量（CA 块效率高）取得了更高的准确率。

## 11. 常见问题与易错点

### 11.1 class token 的插入位置

**问题**：什么时候插入 class token？是在 SA 阶段后立即插入，还是在中间某层插入？

**答案**：CaiT 在所有 SA 块完成后插入 class token。原因是 SA 块负责 patches 之间的充分交互，之后 CA 块用 class token 一次性读出全局信息。

### 11.2 LayerScale 的初始值

**问题**：LayerScale 的初始值太小（如 1e-5）导致训练初期 class token 几乎不被更新。

**原因**：CA 块的初始值很小是为了防止 class token 过早"锁定"到错误的注意力模式。

**解决方法**：保持 CA 块的小初始值，SA 块使用较大的初始值（0.1）。

### 11.3 CA 块中 FFN 的作用范围

**问题**：CA 块中的 FFN 是否作用于所有 token？

**答案**：不，FFN 只作用于 class token。patch tokens 在 CA 阶段保持不变。

### 11.4 与 DeiT 的区别

CaiT 和 DeiT 常被混淆。关键区别：
- **DeiT**：使用知识蒸馏训练 ViT，架构与 ViT 相同（cls+patches 自注意力）
- **CaiT**：修改了架构本身（分离 SA 和 CA），与训练方法无关

## 12. 学习总结

### 12.1 核心贡献

CaiT 的关键创新是：

1. **SA-CA 分离设计**：将自注意力（patches 之间）和类注意力（cls→patches）分离开来
2. **LayerScale**：简单但有效的深层训练稳定技术
3. **计算效率**：CA 块的计算复杂度为 O(N) 而非 O(N²)，大幅降低计算开销

### 12.2 设计哲学

CaiT 的设计体现了"关注分离"（Separation of Concerns）的原则：

- SA 块：关注 patches 之间的关系建模（"位置敏感"任务）
- CA 块：关注全局语义聚合（"内容敏感"任务）
- class token：只在需要时才被引入

### 12.3 与 ViT 的关键区别

| 方面 | ViT | CaiT |
|------|-----|------|
| class token 参与阶段 | 全程 | 后半部分 |
| patches 交互 | 全程全连接 | 前半部分全连接，后半部分互不交互 |
| 残差缩放 | 无 | LayerScale |
| 深层计算效率 | O(N²) | O(N) |
| 训练稳定性 | 一般 | 好 |

## 13. 练习题与思考题

### 13.1 基础题

**题目 1**：CaiT 中 Self-Attention 块和 Class-Attention 块的核心区别是什么？

**答案**：
- SA 块：所有 token（patches）之间进行双向交互，Q、K、V 都来自 patches
- CA 块：class token 作为 query，"读取" patch tokens 的信息，patches 之间不交互

**题目 2**：CaiT 中的 LayerScale 是什么？为什么要使用它？

**答案**：LayerScale 是对每个残差块的输出进行逐元素缩放（乘以一个可学习的向量）。它通过抑制训练初期的梯度大小，使深层网络的训练更加稳定。

**题目 3**：CaiT 为什么将 class token 放在 SA 阶段之后才引入？

**答案**：SA 阶段让 patches 之间充分交互，建立局部和全局关系。然后 class token 一次性"读出"已经编码在 patches 中的全局信息。过早引入 class token 会使它受到 patch 级别细节的干扰。

### 13.2 进阶题

**题目 4**：CaiT 的 CA 块比 ViT 的标准自注意力块节省了多少计算量？请用 N 和 D 表示。

**答案**：
- 标准自注意力：O(3·N·D² + 2·N²·D) = O(N²·D)（主要开销在 N² 项）
- CA 块：O(3·N·D² + 2·N·D) = O(N·D)（没有 N² 项）

具体来说，CA 块中 class token 有 1 个 query，与 N 个 keys 做内积，得到 (1, N) 的注意力矩阵，复杂度为 O(N·D)。而标准注意力有 N 个 queries 与 N 个 keys 做内积，复杂度 O(N²·D)。

**题目 5**：CaiT 能用于语义分割或目标检测等密集预测任务吗？需要如何修改？

**答案**：CaiT 的 class token 设计主要是为分类服务的，用于密集预测需要修改：
1. 移除 CA 块和 class token
2. 将 SA 块改为类 FPN 的多尺度输出
3. 或保留 SA 块做 backbone，将不同层的特征图输出给检测/分割头

### 13.3 思考题

**题目 6**：CaiT 的分离设计是否可能应用于 NLP 中的特定任务？例如，句子级分类任务中是否可以使用类似设计？

**答案**：可以。对于句子级分类（如情感分析），可以设计类似架构：
- SA 阶段：单词之间充分交互
- CA 阶段：[CLS] token 读取所有单词的表示
这比标准 BERT 中将 [CLS] 在所有层都参与交互更高效。

**题目 7**：如果让 CA 块中的 patch tokens 也进行简单的更新（例如通过门控机制），会提升性能吗？

**答案**：这是一个开放问题。直觉上，如果 patch tokens 在 CA 阶段也能得到微弱更新（例如只更新被 class token 高度关注的 patches），可能提升性能。但 CaiT 的作者实验发现，保持 patch tokens 不变效果最好，因为这强制 class token 从"已有的"patch 表示中提取信息，而不是诱导 patches 去适应 class token 的需求。

## 14. 学习路径建议

### 14.1 前置知识

1. **ViT**：理解 patch embedding、class token、位置编码等基本概念
2. **多头注意力机制**：理解 QKV 的计算过程
3. **DeiT**：理解知识蒸馏和数据增强策略
4. **残差网络思想**：理解 ResNet 的残差连接

### 14.2 学习步骤

1. **第一步**：阅读原论文《Going deeper with Image Transformers》，重点理解 SA-CA 分离设计
2. **第二步**：理解 LayerScale 的原理和初始化策略
3. **第三步**：手工实现 CaiT，关注 SA 块和 CA 块的区别
4. **第四步**：在 CIFAR-100 上训练 CaiT，与 ViT/DeiT 对比
5. **第五步**：可视化 CA 块的注意力图，分析 class token 关注哪些区域
6. **第六步**：实验不同的 SA:CA 比例，寻找最优配置

### 14.3 相关论文推荐

- CaiT (Touvron et al., 2021)：Going deeper with Image Transformers
- DeiT (Touvron et al., 2021)：Training data-efficient image transformers
- ViT (Dosovitskiy et al., 2020)：An Image is Worth 16x16 Words
- LayerScale：原 CaiT 论文中提出的技巧

### 14.4 实践建议

1. 在小型数据集（CIFAR-10/100）上对比 ViT、DeiT、CaiT 的收敛速度和最终性能
2. 尝试 SA:CA = 1:1, 2:1, 1:2 等不同比例，观察对性能的影响
3. 将 LayerScale 集成到其他模型中（如 ViT、Swin Transformer）
4. 修改 CaiT 用于多标签分类或多任务学习
