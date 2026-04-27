# 正则化技术 (Dropout + 残差连接) 学习文档

> 来源线索：本节内容根据原书中涉及的正则化技术（Dropout和残差连接）整理、扩展与教学化改写。这些技术贯穿全书，是Transformer等模型的基础组件。

> 让模型学会不依赖任何单一神经元——Dropout和残差连接是深度网络训练的两大基石。

## 1. 算法基础认知

**一句话定义**：Dropout通过随机丢弃神经元防止过拟合，残差连接通过跨层直连解决梯度消失。

**直觉类比**：
- **Dropout**：像一个公司不依赖任何一个员工——如果某个员工请假，公司仍能运转。训练时随机让部分神经元"请假"，迫使网络不过度依赖任何单个神经元。
- **残差连接**：像走楼梯时可以坐电梯直达——即使中间楼层有问题，你也可以直接到达目标楼层。信息可以绕过中间层直接传递。

**历史背景**：Dropout由Srivastava等人在2014年提出，迅速成为深度学习最常用的正则化方法。残差连接由He等人在2015年的ResNet中提出，解决了深层网络训练困难的问题。两者都是现代深度学习的标准组件。

**算法定位**：深度学习 / 正则化 / 训练技巧。不是独立算法，而是几乎所有深度模型的基础组件。

**前置知识**：
- 前馈神经网络
- 反向传播和梯度消失问题
- 过拟合的概念

## 2. 核心原理

### Dropout核心思想

训练时以概率 $p$ 随机将神经元的输出置零，测试时使用所有神经元但输出乘以 $(1-p)$ 缩放。效果等价于训练了多个子网络的集成。

**工作流程**：
1. 训练时：每个神经元以概率 $p$ 被丢弃（输出置零）
2. 反向传播时：被丢弃的神经元梯度也为零
3. 测试时：所有神经元都使用，但输出乘以 $(1-p)$ 补偿
4. 实现变体：Inverted Dropout——训练时就除以 $(1-p)$，测试时不变

### 残差连接核心思想

在深度网络中，不是让每一层直接学习目标映射 $\mathcal{H}(x)$，而是学习残差 $\mathcal{F}(x) = \mathcal{H}(x) - x$：

$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

**工作流程**：
1. 输入 $x$ 进入网络层
2. 网络层计算残差映射 $\mathcal{F}(x)$
3. 输出 = $\mathcal{F}(x) + x$（逐元素相加）
4. 梯度可以通过 $x$ 的路径直接回传，避免梯度消失

### 关键概念

- **Dropout率 $p$**：丢弃概率，越大正则化越强但信息损失越多
- **残差块**：包含残差连接的基本单元，如 Conv→BN→ReLU→Conv→BN + skip → ReLU
- **恒等映射**：当 $\mathcal{F}(x) = 0$ 时，残差块退化为恒等映射（信息无损传递）

## 3. 数学公式与推导

### Dropout数学表达

训练时：
$$\tilde{y} = m \odot f(Wx + b), \quad m_i \sim \text{Bernoulli}(1-p)$$

其中 $m$ 是二值掩码向量，$\odot$ 是逐元素乘法。

Inverted Dropout（训练时缩放）：
$$\tilde{y} = \frac{m \odot f(Wx + b)}{1-p}$$

### 残差连接数学表达

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}$ 表示残差映射（1-2个权重层）。

### 梯度分析

对残差连接求导：
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

即使 $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ 很小，由于有恒等项 $\mathbf{I}$，梯度也不会消失。

### Transformer中的残差

Transformer中每个子层的输出为：
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

这是残差连接 + LayerNorm的组合（Post-LN变体）。

## 4. 训练过程讲解

### Dropout使用策略

- 全连接层后：$p = 0.5$（经典设置）
- 嵌入层后：$p = 0.1$
- 注意力权重上：$p = 0.1$
- Transformer FFN中：$p = 0.1$
- 微调预训练模型时：通常减小或去掉Dropout

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| Dropout率 $p$ | 丢弃概率 | 0.1-0.5 | 0.1（Transformer）|
| 残差连接 | 是否使用 | - | 默认使用 |

## 5. 应用场景

1. **Transformer训练**：每个子层（注意力和FFN）都有残差连接，注意力权重和FFN内部使用Dropout。

2. **大模型微调**：LoRA等微调方法中通常不使用Dropout（模型已充分正则化）。

3. **视觉模型**：ResNet的残差连接是CV领域的标准架构，ViT也继承了残差连接。

## 6. 优缺点分析

| 技术 | 优点 | 缺点 |
|------|------|------|
| Dropout | 有效防止过拟合；实现简单；等价于模型集成 | 训练时间增加（需要更多epoch）；不适用于小模型 |
| 残差连接 | 解决梯度消失；允许极深层网络；信息无损传递 | 要求输入输出维度相同（或需要投影） |

## 7. 调库实现

```python
"""Dropout和残差连接的 PyTorch 实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlockWithDropout(nn.Module):
    """带Dropout和残差连接的Transformer块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 注意力层
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True,
                                           dropout=dropout)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),  # FFN中的Dropout
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 注意力Dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out  # 残差连接
        
        # FFN + 残差连接
        x = x + self.ffn(self.norm2(x))  # 残差连接
        
        return x


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    d_model, num_heads, d_ff = 256, 4, 1024
    block = TransformerBlockWithDropout(d_model, num_heads, d_ff, dropout=0.1)
    
    x = torch.randn(2, 16, d_model)
    
    # 训练模式：Dropout生效
    block.train()
    out_train1 = block(x)
    out_train2 = block(x)
    diff = (out_train1 - out_train2).abs().mean()
    print(f"训练模式两次前向差异: {diff:.6f} (Dropout导致随机性)")
    
    # 评估模式：Dropout关闭
    block.eval()
    out_eval1 = block(x)
    out_eval2 = block(x)
    diff = (out_eval1 - out_eval2).abs().mean()
    print(f"评估模式两次前向差异: {diff:.10f} (应为0)")
    
    # PyTorch内置Dropout
    dropout = nn.Dropout(p=0.3)
    x_test = torch.ones(5, 10)
    dropout.train()
    x_dropped = dropout(x_test)
    survived = (x_dropped != 0).float().mean()
    print(f"\nDropout(p=0.3): {survived:.2%}的元素保留")
    print(f"保留元素值: {x_dropped[x_dropped != 0][0]:.4f} (应为1/{1-0.3}={1/(1-0.3):.4f})")
```

## 8. 手工代码实现

```python
"""从零实现Dropout和残差连接"""
import torch
import torch.nn as nn


class ManualDropout(nn.Module):
    """手写Dropout（不使用nn.Dropout）"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # 丢弃概率
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # 生成二值掩码: 1-p的概率保留
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        
        # Inverted Dropout: 训练时缩放
        return x * mask / (1 - self.p)


class ManualResidualBlock(nn.Module):
    """手写残差块"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = ManualDropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x):
        # 残差映射 F(x)
        residual = self.fc2(self.act(self.fc1(self.norm(x))))
        residual = self.dropout(residual)
        # 残差连接: F(x) + x
        return residual + x


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Dropout测试
    dropout = ManualDropout(p=0.3)
    dropout.train()
    x = torch.ones(10, 20)
    out = dropout(x)
    survived = (out != 0).float().mean()
    print("=== 手写Dropout测试 ===")
    print(f"保留比例: {survived:.2%} (期望70%)")
    print(f"保留值: {out[out != 0][0]:.4f} (期望{1/0.7:.4f})")
    
    # 残差连接测试
    block = ManualResidualBlock(128, 512, dropout=0.1)
    x = torch.randn(2, 10, 128)
    out = block(x)
    print(f"\n=== 残差块测试 ===")
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape} (与输入相同)")
    
    # 梯度验证: 残差连接保证梯度不消失
    x = torch.randn(2, 10, 128, requires_grad=True)
    # 10层残差块堆叠
    out = x
    for _ in range(10):
        out = block(out)
    loss = out.sum()
    loss.backward()
    print(f"10层残差块后的梯度范数: {x.grad.norm():.4f}")
    print("(没有残差连接时, 梯度范数通常接近0)")
```

## 9. 可视化与结果理解

```python
"""Dropout和残差连接可视化"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: Dropout效果——不同p值的网络输出分布
np.random.seed(42)
p_values = [0.0, 0.2, 0.5, 0.8]
for p in p_values:
    outputs = []
    for _ in range(1000):
        # 模拟10个神经元的层
        neurons = np.random.randn(10)
        mask = np.random.binomial(1, 1-p, size=10)
        out = (neurons * mask / (1-p + 1e-10)).sum()
        outputs.append(out)
    axes[0].hist(outputs, bins=50, alpha=0.5, label=f'p={p}')

axes[0].set_title('Dropout对输出的影响', fontsize=13)
axes[0].set_xlabel('输出值')
axes[0].set_ylabel('频次')
axes[0].legend()

# 图2: 残差连接的梯度流
depths = range(1, 51)
# 无残差：梯度指数衰减
grad_no_residual = [0.9**d for d in depths]
# 有残差：梯度保持
grad_with_residual = [1.0 / (1 + 0.01 * d) for d in depths]

axes[1].plot(depths, grad_no_residual, 'r-', linewidth=2, label='无残差连接')
axes[1].plot(depths, grad_with_residual, 'b-', linewidth=2, label='有残差连接')
axes[1].set_yscale('log')
axes[1].set_title('梯度随网络深度的变化', fontsize=13)
axes[1].set_xlabel('网络深度（层数）')
axes[1].set_ylabel('梯度范数（对数尺度）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3: Dropout对过拟合的影响
epochs = range(1, 51)
train_acc_no_drop = [min(0.95 + 0.05 * (1 - np.exp(-e/10)), 1.0) for e in epochs]
val_acc_no_drop = [min(0.85 + 0.05 * (1 - np.exp(-e/10)) - 0.001 * e, 0.88) for e in epochs]
train_acc_drop = [min(0.88 + 0.07 * (1 - np.exp(-e/15)), 0.95) for e in epochs]
val_acc_drop = [min(0.86 + 0.06 * (1 - np.exp(-e/15)), 0.92) for e in epochs]

axes[2].plot(epochs, train_acc_no_drop, 'r-', label='训练(无Dropout)')
axes[2].plot(epochs, val_acc_no_drop, 'r--', label='验证(无Dropout)')
axes[2].plot(epochs, train_acc_drop, 'b-', label='训练(Dropout)')
axes[2].plot(epochs, val_acc_drop, 'b--', label='验证(Dropout)')
axes[2].set_title('Dropout对过拟合的影响', fontsize=13)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('准确率')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dropout_residual_viz.png', dpi=100)
plt.show()

print("图1解读: Dropout越大, 输出分布越分散(方差越大)")
print("图2解读: 无残差连接时梯度指数衰减, 有残差连接时梯度稳定")
print("图3解读: Dropout使训练精度略低, 但验证精度更高(减少过拟合)")
```

## 10. 模型评估

```python
"""评估Dropout和残差连接的效果"""
import torch
import torch.nn as nn

def evaluate_regularization(model_class, train_loader, test_loader,
                             use_dropout=True, n_epochs=20):
    """对比有/无Dropout的训练效果"""
    model = model_class(use_dropout=use_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total * 100
    mode = "有Dropout" if use_dropout else "无Dropout"
    print(f"{mode}: 测试准确率 = {acc:.2f}%")
    return acc
```

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 训练测试差异大 | 训练精度远高于测试 | 过拟合 | 增大Dropout率 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 忘记model.eval() | 测试时结果不稳定 | Dropout在测试时仍生效 | 测试前调用model.eval() |
| 残差维度不匹配 | 报错 | 输入输出维度不同 | 添加线性投影对齐维度 |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| Dropout过大 | 欠拟合 | 信息丢失太多 | 降低Dropout率到0.1-0.2 |

## 12. 学习总结

Dropout和残差连接是深度学习的两大基础技术：

**Dropout**：$\tilde{y} = \frac{m \odot f(x)}{1-p}$，$m \sim \text{Bernoulli}(1-p)$

**残差连接**：$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$

两者在Transformer中配合使用：残差连接保证深层网络梯度流通，Dropout防止过拟合。几乎所有现代深度模型（ViT、LLM、扩散模型等）都使用这两个技术。

## 13. 练习题与思考题

### 基础题1：Inverted Dropout

为什么使用Inverted Dropout（训练时除以 $1-p$）而不是测试时乘以 $1-p$？

**参考答案**：
Inverted Dropout在训练时缩放的好处是测试时不需要任何修改。如果测试时缩放，需要在推理代码中加入额外逻辑。Inverted Dropout将复杂性留在训练阶段，推理代码更简洁高效。

### 基础题2：残差梯度

证明：当残差块 $\mathcal{F}(x)$ 的梯度为0时，整体梯度不为0。

**参考答案**：
$\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial x}{\partial x} = 0 + I = I$
即使残差映射的梯度为0，恒等映射的梯度为 $I$（单位矩阵），保证梯度至少为1。

### 进阶题：Dropout与集成学习

证明Dropout等价于训练 $2^n$ 个子网络的集成（$n$ 为神经元数）。

**参考答案**：
每个神经元有2种状态（保留/丢弃），$n$ 个神经元共有 $2^n$ 种组合。每种组合对应一个子网络。Dropout训练等价于随机采样这些子网络并训练，测试时取所有子网络的平均（通过乘以 $1-p$ 近似）。

### 开放思考题

大语言模型（如GPT-4、DeepSeek）的训练中，Dropout率通常设为0或很小。为什么大模型不需要强正则化？

**参考思路**：
1. **数据量足够大**：训练数据（数万亿token）远超模型参数量，过拟合风险小
2. **模型已有隐式正则化**：大批量SGD、权重衰减、注意力机制本身有正则效果
3. **MoE的稀疏激活**：本身就是一种正则化（每次只激活部分参数）
4. **Dropout可能有害**：对于需要精确记忆知识的任务，Dropout会降低模型容量

## 14. 学习路径建议

### 前置知识
- 前馈神经网络
- 反向传播算法
- 过拟合与正则化概念

### 平行学习
- Batch Normalization
- Layer Normalization
- 权重衰减（L2正则化）

### 进阶方向
- DropConnect、DropPath（随机深度）
- Pre-LN vs Post-LN（Transformer中的归一化位置）
- 大模型中的正则化策略

### 推荐资源
1. **论文**：Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)
2. **论文**：Deep Residual Learning for Image Recognition (He et al., 2016)
3. **课程**：Stanford CS231n - Regularization章节
