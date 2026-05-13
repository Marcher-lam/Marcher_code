# T2T-ViT (Tokens-to-Token Vision Transformer) 学习文档

## 1. 算法基础认知

### 1.1 算法要解决什么问题

ViT（Vision Transformer）直接将图像切分为固定大小的 patch（如 16×16），然后通过线性投影得到 token 序列。这种方式存在两个根本性问题：

- **局部结构破坏**：ViT 的 patch 化过程破坏了图像内部的局部邻域结构（如边缘、纹理、小物体），使 Transformer 难以捕捉细粒度的视觉信息。
- **冗余 token**：固定大小的 patch 没有考虑图像内容的自适应性——背景区域和平坦区域的 patch 携带的信息量远少于物体边缘的 patch，但 ViT 一视同仁。

T2T-ViT 的出发点是：**是否可以用更渐进、更结构化的方式，将图像逐步转化为 token 序列，就像人类观察图像时先看局部细节，再逐步整合成全局理解？**

### 1.2 核心思路概览

T2T-ViT（Tokens-to-Token Vision Transformer）由 Yuan 等人于 2021 年提出，其核心创新是 **Token-to-Token (T2T) 模块**，该模块通过渐进式地将图像像素聚合成 token，保留了图像的局部结构信息。

T2T 模块的流程可以概括为：

1. **Soft Split**：将输入特征图用滑动窗口方式切分为重叠的 patch
2. **Restructurization**：将 patch 序列重新拼接为特征图
3. **重复以上步骤**：逐步减小序列长度，增加 token 的语义丰富度

这种渐进式的 token 化过程，使 T2T-ViT 在 ImageNet 上以相近参数量取得了比 ViT 高 2-3% 的 top-1 准确率。

### 1.3 整体架构

T2T-ViT 的整体架构分为两大部分：

```
输入图像 → T2T 模块（渐进式 Token 化） → 主干 Transformer 编码器 → 分类头
```

- **T2T 模块**：将图像从像素空间逐步转化为 token 序列，同时保留局部结构
- **主干 Transformer**：使用标准 ViT 的 Transformer 编码器进行全局关系建模
- **分类头**：简单的 MLP 分类器

## 2. 核心原理

### 2.1 ViT 的 patch embedding 有什么问题

ViT 的 patch embedding 过程如下：
1. 将 H×W×3 的图像分割为 N 个 P×P×3 的 patch（N = HW/P²）
2. 将每个 patch 展平为 P²·3 维向量
3. 通过线性投影得到 D 维 token

这个过程有两个本质缺陷：

**缺陷一：信息丢失**
当 patch size = 16 时，一个 patch 包含了 16×16×3 = 768 个像素值，但被压缩成一个 D 维向量。原本像素之间的空间关系（相邻像素的色差、纹理模式）被彻底破坏。例如，在 16×16 的区域内可能存在一个 8×8 的小物体，ViT 的 patch embedding 会将其与背景像素混在一起，无法区分。

**缺陷二：缺乏多层次结构**
人类视觉系统具有层次性——先看边缘（1-2 像素），再看纹理（4-8 像素），再看局部形状（16-32 像素），最后看整体物体。ViT 一次性地将 16×16 的 patch 作为基本单元，跳过了中间的层次结构。

### 2.2 T2T 模块如何解决

T2T 模块通过**多次渐进式聚合**来解决上述问题。每次聚合包含两个步骤：

**Step 1: Soft Split**
将当前的 2D 特征图用滑动窗口切分为重叠的 patch。设当前特征图尺寸为 Hᵢ×Wᵢ×Cᵢ，窗口大小为 k×k，步长为 s，padding 为 p，可以得到：

- 输出序列长度：Hᵢ₊₁ = ⌊(Hᵢ + 2p - k)/s + 1⌋, Wᵢ₊₁ 类似
- 每个 token 的维度：Cᵢ · k²

关键点：**重叠的滑动窗口**（stride < k）使相邻 patch 之间共享像素信息，保留了局部邻域结构。

**Step 2: Restructurization**
将 soft split 得到的 token 序列重新排列回 2D 特征图（Hᵢ₊₁ × Wᵢ₊₁ × (Cᵢ·k²)），作为下一轮 soft split 的输入。

通过多次重复，T2T 模块实现了：
- 序列长度逐步减少（Hᵢ₊₁ < Hᵢ, Wᵢ₊₁ < Wᵢ）
- 每个 token 的语义信息逐步丰富
- 局部结构在每次聚合中都得到保留

### 2.3 深层连接

T2T-ViT 还引入了**深层连接（Deep Narrow Structure）**。与 ViT 使用宽而浅的架构不同，T2T-ViT 使用了深而窄的架构（14-24 层），并在浅层特征与深层特征之间建立跳跃连接，帮助梯度流动。

## 3. 数学公式与推导

### 3.1 Soft Split 操作

设输入特征图为 X ∈ ℝ^{H×W×C}，soft split 使用滑动窗口：

对于位置 (i, j) 处的窗口，其包含的像素区域为：
- 行范围：[i·s - p, i·s - p + k - 1]（被限制在 [0, H-1] 内）
- 列范围：[j·s - p, j·s - p + k - 1]（被限制在 [0, W-1] 内）

其中 s 是步长，p 是 padding。

窗口内的像素被展平为一个向量：
```
t_{i,j} = Flatten(X[i·s-p:i·s-p+k, j·s-p:j·s-p+k, :]) ∈ ℝ^{k²·C}
```

所有窗口的向量组成 token 序列 T ∈ ℝ^{N×k²C}，其中 N = (⌊(H+2p-k)/s⌋+1) · (⌊(W+2p-k)/s⌋+1)。

### 3.2 Restructurization 操作

将 token 序列 T 重新排列为 2D 特征图：
```
X' = Reshape(T, (H', W', k²·C))
```
其中 H' = ⌊(H+2p-k)/s⌋+1, W' = ⌊(W+2p-k)/s⌋+1。

### 3.3 线性投影

在 restructurization 之后，使用线性投影将 token 维度映射到 Transformer 的隐层维度 D：
```
T' = T · W_proj + b_proj
```
其中 W_proj ∈ ℝ^{k²C × D}, b_proj ∈ ℝ^{D}。

### 3.4 T2T 模块整体流程

T2T 模块通常执行两次 soft split：

```
第一次 Soft Split:
输入: H×W×3 → 输出: H₁×W₁×(k₁²·3) → 投影到 D 维 → Transformer 层

第二次 Soft Split:
输入: H₁×W₁×D → 输出: H₂×W₂×(k₂²·D) → 投影到 D 维
```

最终得到固定长度的 token 序列，送入主干 Transformer。

### 3.5 MSRI 模块（可选）

部分 T2T-ViT 变体在每两次 soft split 之间使用 MSRI（Multi-head Self-attention with Relative position encoding）模块，对 token 进行自注意力处理，进一步增强 token 的表示能力：
```
T_out = MSRI(T_in) = Concat(head₁,...,head_h) · W_O
head_i = Attention(Q_i, K_i, V_i) = Softmax(Q_i·K_iᵀ/√d + R) · V_i
```
其中 R 是相对位置编码矩阵。

## 4. 训练过程讲解

### 4.1 数据准备

T2T-ViT 在 ImageNet 上训练，使用标准的数据增强策略：
- RandomResizedCrop（随机裁剪到 224×224）
- RandomHorizontalFlip（随机水平翻转）
- RandAugment（随机数据增强）
- Mixup & CutMix（混合增强）

### 4.2 训练配置

- **优化器**：AdamW（β₁=0.9, β₂=0.999）
- **学习率调度**：Cosine annealing，初始学习率 1e-3
- **权重衰减**：0.05
- **热身阶段**：前 20 个 epoch 线性热身
- **批次大小**：1024
- **训练轮数**：300 epoch
- **标签平滑**：0.1

### 4.3 训练流程

```
for each epoch:
    for each batch:
        1. 前向传播：
           a. T2T 模块渐进式生成 tokens
           b. Transformer 编码器处理 tokens
           c. 分类头输出预测
        2. 计算交叉熵损失
        3. 反向传播计算梯度
        4. AdamW 更新参数
    验证集评估准确率
学习率按照 cosine 调度衰减
```

### 4.4 微调策略

在 ImageNet 上预训练后，可以在下游任务上微调：
- 调整输入分辨率（如 384×384 或 448×448）
- 使用更小的学习率（预训练学习率的 0.1 倍）
- 减少权重衰减

## 5. 应用场景

### 5.1 图像分类（主要场景）

T2T-ViT 在 ImageNet 分类任务上表现优异：
- T2T-ViT-14：81.5% top-1 准确率（~22M 参数）
- T2T-ViT-19：81.9% top-1 准确率（~39M 参数）
- T2T-ViT-24：82.3% top-1 准确率（~64M 参数）

相比同等参数的 ViT，提升约 2-3%。

### 5.2 迁移学习

在多个下游分类任务上表现优异：
- CIFAR-10/100、Oxford Flowers、Stanford Cars 等
- 预训练 T2T-ViT 的迁移效果优于预训练 ViT

### 5.3 作为骨干网络

T2T-ViT 可以作为通用视觉骨干，用于：
- 目标检测（配合 DETR 等检测框架）
- 语义分割（配合 SETR 等分割框架）

## 6. 优缺点分析

### 6.1 优点

1. **更好的局部信息保留**：渐进式 token 化保留了图像的局部结构信息
2. **更高的数据效率**：相比 ViT，T2T-ViT 在更少的数据上就能取得好效果
3. **架构灵活**：T2T 模块可以替换任何需要 patch embedding 的视觉模型
4. **多层次特征**：具有类似 CNN 的层次特征提取能力

### 6.2 缺点

1. **计算开销增加**：多次 soft split 和线性投影增加了计算量
2. **内存占用高**：重叠的滑动窗口产生更多的 token
3. **推理速度较慢**：相比 ViT，T2T 模块的前向时间更长
4. **实现复杂度高**：T2T 模块的实现比 ViT 的 patch embedding 复杂得多

## 7. 调库实现

### 7.1 使用 timm 库调用 T2T-ViT

```python
import torch
import torch.nn as nn
from timm.models.tnt import TNT

# T2T-ViT 在 timm 中对应 TNT (Transformer in Transformer)
# 注意：部分 timm 版本直接提供 t2t_vit 模型
# 如果使用较新版本，可直接调用：
try:
    from timm.models import t2t_vit_14
    model = t2t_vit_14(pretrained=True, num_classes=1000)
except ImportError:
    # 使用 TNT 作为替代（TNT 也使用类似 T2T 的 token 化方式）
    model = TNT(
        img_size=224,        # 输入图像大小
        patch_size=16,       # 最终 patch 大小
        in_chans=3,          # 输入通道数
        num_classes=1000,    # 分类数
        embed_dim=384,       # Transformer 隐层维度
        num_heads=6,         # 注意力头数
        mlp_ratio=4.0,       # MLP 隐藏层倍率
        depth=12,            # Transformer 层数
        rpi=True,            # 是否使用相对位置编码
    )

# 创建随机输入
x = torch.randn(4, 3, 224, 224)

# 前向传播
output = model(x)
print(f"输出形状: {output.shape}")  # (4, 1000)

# 推理模式
model.eval()
with torch.no_grad():
    output = model(x)
    predicted_classes = output.argmax(dim=-1)
    print(f"预测类别索引: {predicted_classes}")
```

### 7.2 完整的训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from timm.models import t2t_vit_14

# 超参数设置
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

# 加载数据集（示例使用 CIFAR-10）
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 创建模型
model = t2t_vit_14(pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 训练函数
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

# 验证函数
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
print("开始训练 T2T-ViT...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# 保存模型
torch.save(model.state_dict(), "t2t_vit_cifar10.pth")
print("模型已保存！")
```

## 8. 手工代码实现

### 8.1 T2T 模块的手工实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftSplit(nn.Module):
    """
    Soft Split: 使用滑动窗口将特征图切分为重叠的 patch token
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x: (B, C, H, W)
        输出: tokens: (B, N, C*k*k), 其中 N = H' * W' 是输出位置数
        """
        # 使用 unfold 实现滑动窗口提取
        # unfold 在 H 维度上滑动: (B, C, H_out, C*k)
        x = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        # x.shape: (B, C*k*k, N)
        # 转置为 (B, N, C*k*k)
        x = x.transpose(1, 2)
        return x


class Restructurization(nn.Module):
    """
    Restructurization: 将 token 序列重新排列回 2D 特征图
    """
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x: (B, N, C*k*k)
        输出: (B, C*k*k, H', W')
        """
        B, N, D = x.shape
        # 确保 N = H' * W'
        assert N == self.height * self.width, \
            f"序列长度 {N} 不匹配 {self.height}x{self.width}"
        # 重新排列为 2D 特征图
        x = x.transpose(1, 2)  # (B, D, N)
        x = x.view(B, D, self.height, self.width)  # (B, D, H', W')
        return x


class T2TModule(nn.Module):
    """
    T2T (Tokens-to-Token) 模块
    通过多次 Soft Split + Restructurization 渐进式生成 tokens
    """
    def __init__(
        self,
        img_size: int = 224,
        token_dim: int = 384,
        in_chans: int = 3,
        soft_split_configs: list = None
    ):
        super().__init__()

        if soft_split_configs is None:
            # 默认的两阶段 T2T 配置
            # 第一阶段: kernel=7, stride=4, padding=2
            # 第二阶段: kernel=3, stride=2, padding=1
            soft_split_configs = [
                {"kernel": 7, "stride": 4, "padding": 2},
                {"kernel": 3, "stride": 2, "padding": 1},
            ]

        # 计算每阶段的输出尺寸
        h, w = img_size, img_size
        self.split_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.restruct_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()

        for i, config in enumerate(soft_split_configs):
            k, s, p = config["kernel"], config["stride"], config["padding"]

            # 计算输出尺寸
            h_out = (h + 2 * p - k) // s + 1
            w_out = (w + 2 * p - k) // s + 1

            # Soft Split 层
            self.split_layers.append(SoftSplit(k, s, p))

            # 投影层（将 token 映射到 token_dim）
            in_dim = (in_chans if i == 0 else token_dim) * k * k
            self.proj_layers.append(
                nn.Linear(in_dim, token_dim)
            )

            # Restructurization 层
            self.restruct_layers.append(
                Restructurization(h_out, w_out)
            )

            # Transformer 编码器层（每阶段后使用）
            if i < len(soft_split_configs) - 1:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=6,
                    dim_feedforward=token_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True
                )
                self.transformer_layers.append(
                    nn.TransformerEncoder(encoder_layer, num_layers=2)
                )

            h, w = h_out, w_out

        self.num_tokens = h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x: (B, 3, H, W)
        输出: tokens: (B, N, D)
        """
        for i in range(len(self.split_layers)):
            # Step 1: Soft Split
            x = self.split_layers[i](x)  # (B, N, C*k*k)

            # Step 2: 线性投影
            x = self.proj_layers[i](x)  # (B, N, D)

            # Step 3: Restructurization（除了最后一次）
            if i < len(self.split_layers) - 1:
                x = self.restruct_layers[i](x)  # (B, D, H', W')
                x = self.transformer_layers[i](x)  # (B, N', D)
                x = x.transpose(1, 2)  # (B, D, N')
                B, D, N = x.shape
                H, W = self.restruct_layers[i].height, self.restruct_layers[i].width
                x = x.view(B, D, H, W)  # (B, D, H', W')

        return x


class T2TViT(nn.Module):
    """
    T2T-ViT: Token-to-Token Vision Transformer
    完整的手工实现
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # T2T 模块
        self.t2t = T2TModule(
            img_size=img_size,
            token_dim=embed_dim,
            in_chans=in_chans,
        )

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        num_tokens = self.t2t.num_tokens + 1  # +1 为 cls token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # LayerNorm 和分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T2T 模块: 生成 tokens
        x = self.t2t(x)  # (B, N, D)

        # 添加 class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer 编码器
        x = self.transformer(x)

        # 取 class token
        x = x[:, 0]

        # LayerNorm 和分类头
        x = self.norm(x)
        x = self.head(x)

        return x


def test_t2t_vit():
    """测试 T2T-ViT 前向传播"""
    model = T2TViT(
        img_size=224,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_t2t_vit()
```

### 8.2 核心组件详解

**为什么 Soft Split 中 stride < kernel 是关键的？**

当 stride < kernel 时，相邻的窗口会重叠。重叠区域确保了局部结构在 token 化过程中被保留。例如，kernel=7, stride=4 时，相邻窗口有 3 个像素的重叠，这些重叠像素在两个窗口中都出现，使得相邻 token 之间共享信息。

**Restructurization 的作用**

将 token 重新排列为 2D 特征图，允许后续的 soft split 在空间维度上继续操作。如果不进行 restructurization，每个 soft split 只是对 1D 序列的操作，无法利用 2D 空间结构。

## 9. 可视化与结果理解

### 9.1 可视化 T2T 过程

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_t2t_process(model, image_path):
    """
    可视化 T2T 模块的各个阶段输出
    """
    from torchvision import transforms

    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 提取中间特征
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())

    # 注册 hook
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, SoftSplit):
            hooks.append(module.register_forward_hook(hook_fn))

    # 前向传播
    model.eval()
    with torch.no_grad():
        model(x)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 可视化
    fig, axes = plt.subplots(1, len(features) + 1, figsize=(5 * (len(features) + 1), 5))

    # 原始图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = x[0].cpu().numpy().transpose(1, 2, 0)
    img_display = img_display * std + mean
    img_display = np.clip(img_display, 0, 1)
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Soft Split 输出
    for i, feat in enumerate(features):
        # 将 token 中的第一个通道 reshape 回 2D
        B, N, D = feat.shape
        H = W = int(np.sqrt(N))
        feat_map = feat[0, :, 0].reshape(H, W).cpu().numpy()
        axes[i + 1].imshow(feat_map, cmap="viridis")
        axes[i + 1].set_title(f"Soft Split {i + 1}\n{H}x{W} tokens")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# 使用示例
# visualize_t2t_process(model, "path/to/image.jpg")
```

### 9.2 注意力图可视化

```python
def visualize_attention_maps(model, image_path):
    """
    可视化 T2T-ViT 最后一层的注意力图
    """
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    # 加载图像
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    # 提取注意力权重
    attention_weights = []

    def attention_hook(module, input, output):
        # TransformerEncoderLayer 的 forward 返回 (x, attn_weights)
        # 但 nn.TransformerEncoder 不直接暴露注意力权重
        # 这里需要更精细的 hook 实现
        pass

    # 简化版：使用最后层的注意力
    model.eval()
    with torch.no_grad():
        # 获取注意力权重（需要修改模型以返回注意力）
        # 作为演示，我们直接可视化 cls token 的注意力
        x = model.t2t(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + model.pos_embed

        # 最后一层的注意力
        for layer in model.transformer.layers:
            x = layer(x)

    # 可视化 cls token 的注意力分布
    # 假设 attn 形状为 (B, num_heads, N, N)
    # 取 cls token（索引为 0）对其他 token 的注意力
    # attn_cls = attn[0, :, 0, 1:].mean(dim=0)  # 所有头平均
    # attn_map = attn_cls.reshape(int(np.sqrt(attn_cls.shape[0])), -1).cpu().numpy()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(attn_map, cmap='jet')
    # plt.show()
    print("需要修改 TransformerEncoderLayer 以返回注意力权重")
```

## 10. 模型评估

### 10.1 评估指标

T2T-ViT 的评估指标与标准分类模型相同：

```python
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return accuracy, precision, recall, f1
```

### 10.2 与 ViT 的性能对比

| 模型 | 参数量 | ImageNet Top-1 | 训练速度 | 推理速度 |
|------|--------|----------------|----------|----------|
| ViT-B/16 | 86M | 77.9% | 1x | 1x |
| T2T-ViT-14 | 22M | 81.5% | 0.8x | 0.7x |
| T2T-ViT-19 | 39M | 81.9% | 0.7x | 0.6x |
| T2T-ViT-24 | 64M | 82.3% | 0.6x | 0.5x |

T2T-ViT 以较少的参数量取得了更好的性能，但训练和推理速度较慢。

## 11. 常见问题与易错点

### 11.1 Soft Split 的 padding 设置

**问题**：soft split 的 padding 设置不当会导致特征图尺寸与预期不符。

**原因**：padding 的大小需要根据 kernel size 和 stride 计算，使得输出尺寸符合预期。

**解决方法**：使用公式 `H_out = (H + 2P - K) // S + 1` 计算，确保尺寸一致。

### 11.2 Restructurization 的维度假定

**问题**：将 token 序列 reshape 回 2D 特征图时，维度不匹配。

**原因**：restructurization 要求 token 序列的长度 N 必须等于 H_out × W_out。

**解决方法**：在初始化阶段计算并保存 H_out 和 W_out，确保 reshape 时维度正确。

### 11.3 位置编码的匹配

**问题**：T2T 模块输出的 token 数量与位置编码的长度不一致。

**原因**：不同输入尺寸下，T2T 模块输出的 token 数量不同，固定长度的位置编码无法适应。

**解决方法**：使用插值调整位置编码，或使用相对位置编码代替绝对位置编码。

### 11.4 内存溢出

**问题**：T2T 模块产生大量 token，导致 GPU 内存溢出。

**原因**：ViT 从图像生成 196 个 token（14×14），而 T2T 可能生成更多的 token。

**解决方法**：减小输入分辨率，或调整 soft split 的 kernel size 和 stride。

## 12. 学习总结

### 12.1 核心贡献

T2T-ViT 的重要贡献在于：

1. **提出了 Tokens-to-Token 模块**：通过渐进式 token 化，解决了 ViT 初期 token 化过程破坏局部结构的问题
2. **验证了渐进式聚合的有效性**：展示了多层次、渐进式的特征提取在视觉 Transformer 中的重要性
3. **更高效的数据利用**：相比 ViT，T2T-ViT 在更少的数据上也能有好的表现

### 12.2 与 ViT 的关键区别

| 方面 | ViT | T2T-ViT |
|------|-----|----------|
| Token 化方式 | 一次性 patch embedding | 渐进式 T2T 模块 |
| 局部结构保留 | 差（patch 内部结构被破坏） | 好（重叠窗口保留邻域信息） |
| 层次性 | 无层次结构 | 多层次特征提取 |
| 参数量效率 | 低 | 高 |
| 数据效率 | 低（需要大量预训练数据） | 高 |

### 12.3 学习要点

1. T2T-ViT 的核心思想是"渐进式"——不是一次性将图像切成 token，而是逐步聚合
2. Soft Split 的关键是重叠窗口（stride < kernel），让相邻 token 共享信息
3. Restructurization 让特征图在 2D 空间中保持，使得后续操作可以继续利用空间结构
4. T2T 模块是一个可插拔的设计，可以用于替换任何视觉模型中的 patch embedding

## 13. 练习题与思考题

### 13.1 基础题

**题目 1**：T2T-ViT 的 T2T 模块中，如果输入图像大小为 224×224，第一次 soft split 使用 kernel=7, stride=4, padding=2，输出特征图的大小是多少？

**答案**：
H_out = (224 + 2×2 - 7) // 4 + 1 = (224 + 4 - 7) // 4 + 1 = 221 // 4 + 1 = 55 + 1 = 56
W_out = 56
所以输出特征图大小为 56×56，token 数量为 56×56 = 3136。

**题目 2**：T2T-ViT 为什么使用重叠的滑动窗口而不是不重叠的？

**答案**：重叠的滑动窗口（stride < kernel）使相邻的 patch 共享部分像素，保留了图像的局部邻域结构。如果不重叠，相邻 patch 之间没有信息共享，会像 ViT 一样破坏局部结构。

**题目 3**：Restructurization 操作的作用是什么？如果不进行 restructurization 会怎样？

**答案**：Restructurization 将 token 序列重新排列为 2D 特征图，使得后续的 soft split 可以在 2D 空间中进行操作。如果不进行 restructurization，后续的 soft split 只能对 1D 序列操作，无法利用 2D 空间结构（如相邻位置的空间关系）。

### 13.2 进阶题

**题目 4**：T2T-ViT 在参数量远小于 ViT 的情况下取得更好的性能，为什么？

**答案**：T2T-ViT 的参数量效率来自两点：
1. T2T 模块在 tokenization 过程中逐步减少序列长度，同时增加每个 token 的信息量，使主干 Transformer 可以用较小的隐层维度处理
2. T2T-ViT 使用"深而窄"的架构（更多层但更小的隐层维度），相比 ViT 的"浅而宽"架构，参数效率更高

**题目 5**：如果要在 T2T-ViT 中引入卷积操作增强局部建模，应该加在哪个位置？

**答案**：可以在 T2T 模块的各阶段之间加入深度可分离卷积（depthwise convolution）。具体来说，在 restructurization 之后、soft split 之前加入 3×3 或 5×5 的 depthwise conv，可以进一步增强局部特征提取能力。这相当于在 T2T 的渐进式 tokenization 过程中加入卷积正则化。

### 13.3 思考题

**题目 6**：T2T-ViT 和金字塔结构的视觉 Transformer（如 PVT、Swin Transformer）有何异同？

**答案**：
相同点：
- 都是多层次结构，输出不同分辨率的特征图
- 都试图在 Transformer 中引入类似 CNN 的层次性

不同点：
- T2T-ViT 只在 tokenization 阶段使用多层次，主干 Transformer 仍使用固定长度
- PVT/Swin 在整个模型中都保持多层次结构（类似 FPN）
- T2T-ViT 更关注 tokenization 过程的信息保留，PVT/Swin 更关注多尺度特征

**题目 7**：T2T-ViT 的设计是否可以应用于其他模态（如文本、音频）？为什么？

**答案**：理论上可以，但需要调整。T2T-ViT 的设计假设输入具有 2D 空间结构（图像），如果应用于文本（1D 序列），需要将 soft split 改为 1D 滑动窗口。对于音频（1D 序列或 2D 频谱图），也可以类似地设计 T2T 模块。关键思想——渐进式聚合和多层次特征提取——是通用的。

## 14. 学习路径建议

### 14.1 前置知识

在学习 T2T-ViT 之前，建议先掌握：
1. **ViT (Vision Transformer)**：理解 patch embedding、class token、位置编码等基本概念
2. **Transformer 原理**：理解 self-attention、multi-head attention、FFN 等
3. **PyTorch 基础**：能够使用 nn.Module 构建模型

### 14.2 学习步骤

1. **第一步**：通读原论文《Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet》
2. **第二步**：理解 T2T 模块的 forward 流程——Soft Split → Projection → Restructurization
3. **第三步**：手工实现 T2T 模块，关注维度变化
4. **第四步**：在小型数据集（CIFAR-10/100）上训练 T2T-ViT
5. **第五步**：分析 T2T 模块各阶段的输出，理解渐进式 tokenization 的效果
6. **第六步**：阅读 T2T-ViT 的改进变体，如 T2T-ViT with MSRI

### 14.3 相关论文推荐

- 原论文：Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet (Yuan et al., 2021)
- ViT：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)
- DeiT：Training data-efficient image transformers & distillation through attention (Touvron et al., 2021)
- PVT：Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions (Wang et al., 2021)

### 14.4 实践建议

1. 使用 timm 库预训练 T2T-ViT 模型测试 ImageNet 性能
2. 在 CIFAR-100 上从零训练 T2T-ViT，与 ViT 对比
3. 尝试修改 T2T 模块的配置（kernel size、stride、padding 组合），观察对性能的影响
4. 将 T2T 模块集成到其他视觉架构中（如作为 DETR 的 backbone）
