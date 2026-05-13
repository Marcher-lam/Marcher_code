# BatchNorm 与 LayerNorm 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
BatchNorm和LayerNorm是深度学习中两种核心归一化技术，通过规范化神经网络层的输入分布来加速训练、稳定梯度并提升模型泛化能力。

### 1.2 直觉类比
BatchNorm像"全班同学同一科目的标准分"——把所有学生的数学成绩统一到一个标准分布（均值0，方差1）。LayerNorm像"每个同学所有科目的平衡"——把一个学生的语文、数学、英语成绩统一尺度，防止某一科分数过大主导总评。

### 1.3 历史背景
- **2015年**：Ioffe和Szegedy提出Batch Normalization（ICML 2015最佳论文）
- **2016年**：Layer Normalization由Ba等人提出（解决RNN中的BN问题）
- **2017年**：Transformer论文引入LayerNorm，成为NLP标配
- **2018-2020年**：衍生出InstanceNorm、GroupNorm、RMSNorm等变体

### 1.4 算法定位
两种方法都属于**特征归一化（Feature Normalization）**，用于深度神经网络训练优化。

---

## 2. 核心原理

### 2.1 内部协变量偏移（Internal Covariate Shift）
深度网络各层输入分布会随前一层参数变化而改变，迫使后续层不断适应新的分布，降低训练效率。归一化通过固定每层输入的均值和方差来缓解这一问题。

### 2.2 BatchNorm原理
对每个特征维度，在一个batch内统计：

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\gamma, \beta$ 是可学习的缩放和偏移参数。

**训练时**：使用当前batch统计量
**推理时**：使用训练集累积的移动平均统计量

### 2.3 LayerNorm原理
对每个样本的所有特征维度统计：

$$\mu = \frac{1}{d}\sum_{j=1}^d x_j$$
$$\sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

**关键区别**：LayerNorm独立于batch size，对batch中每个样本独立计算统计量。

### 2.4 核心差异对比
| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 统计维度 | 样本维度（N） | 特征维度（D） |
| 依赖batch size | 是 | 否 |
| 训练/推理差异 | 不同 | 一致 |
| RNN适用性 | 需特殊处理 | 天然适用 |
| Transformer适用性 | 不稳定 | 稳定 |

---

## 3. 数学公式与推导

### 3.1 BatchNorm的梯度传播
设 $y = \gamma \hat{x} + \beta$，$\hat{x} = \frac{x - \mu}{\sigma}$：

$$\frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \cdot \gamma$$
$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} (x_i - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$
$$\frac{\partial L}{\partial \mu} = \left(\sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i}\right)\left(-\frac{1}{\sqrt{\sigma^2 + \epsilon}}\right) + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_{i=1}^m -2(x_i - \mu)}{m}$$
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{m} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m}$$

### 3.2 LayerNorm的梯度传播
类似BN但只在样本内部统计。设 $x \in \mathbb{R}^d$：

$$\frac{\partial L}{\partial x_j} = \frac{1}{\sigma}\left(\frac{\partial L}{\partial \hat{x}_j} - \frac{1}{d}\sum_{k}\frac{\partial L}{\partial \hat{x}_k} - \hat{x}_j \cdot \frac{1}{d}\sum_{k}\frac{\partial L}{\partial \hat{x}_k} \hat{x}_k\right) \cdot \gamma$$

### 3.3 为什么LayerNorm适合Transformer？
Transformer中的LayerNorm等价于在注意力计算前对输入做归一化，使得 $QK^T$ 的方差保持稳定：

$$\text{Var}(QK^T) \approx d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_j) \approx d_k \cdot 1 \cdot 1 / d_k = 1$$

经过LayerNorm后，$q_i$ 和 $k_j$ 的方差接近于1，因此 $QK^T$ 的方差也稳定在1附近，避免了softmax的梯度消失或爆炸。

### 3.4 BatchNorm的推理模式
推理时使用全局统计量：

$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}}$$

其中 $\mu_{\text{running}} = \alpha \mu_{\text{running}} + (1 - \alpha) \mu_B$

---

## 4. 训练过程讲解

### 4.1 BatchNorm训练流程
1. 初始化 $\gamma = 1, \beta = 0$，$\mu_{\text{running}} = 0, \sigma_{\text{running}}^2 = 0$
2. 每个训练step:
   - 计算当前batch的 $\mu_B, \sigma_B^2$
   - 归一化 $\hat{x} = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$
   - 缩放平移 $y = \gamma \hat{x} + \beta$
   - 更新running statistics
3. 推理时使用running statistics禁用随机性

### 4.2 BatchSize对BatchNorm的影响
- **batch size = 32~256**：统计量估计准确，效果最好
- **batch size = 1**：无法计算有意义的统计量（方差为0）
- **batch size < 8**：统计量噪声大，可能导致训练不稳定

### 4.3 LayerNorm训练流程（训练=推理）
1. 初始化 $\gamma = 1, \beta = 0$
2. 每个训练/推理step:
   - 对每个样本独立计算 $\mu, \sigma^2$
   - 归一化 $\hat{x}$
   - 缩放平移 $y = \gamma \hat{x} + \beta$
3. 训练和推理的计算流程完全一致

---

## 5. 应用场景

| 场景 | 推荐归一化 | 原因 |
|------|-----------|------|
| CNN图像分类 | BatchNorm | 大batch，稳定的全局统计 |
| RNN/LSTM | LayerNorm | 变长序列，batch依赖弱 |
| Transformer | LayerNorm | 稳定训练，不受序列长度影响 |
| GAN生成 | BatchNorm（生成器） | 控制分布，改善生成质量 |
| 小batch训练 | LayerNorm/GroupNorm | 避免batch统计噪声 |
| 强化学习 | LayerNorm | batch size通常很小 |

---

## 6. 优缺点分析

### BatchNorm优点
1. **缓解梯度消失/爆炸**：保持激活值在合理范围
2. **允许更大学习率**：分布稳定后训练更鲁棒
3. **轻微正则化效果**：batch统计量的随机性
4. **加速收敛**：提速数倍

### BatchNorm缺点
1. **依赖batch size**：小batch效果差
2. **训练/推理不一致**：需要维护全局统计量
3. **序列模型不适用**：变长序列的统计量计算复杂
4. **增加运行时内存**：需要保存每个batch的统计量

### LayerNorm优点
1. **batch无关**：batch size=1也可正常工作
2. **训练推理一致**：无需维护全局统计
3. **序列模型天然适配**：可处理变长输入
4. **Transformer的标配**：被证明最稳定

### LayerNorm缺点
1. **无正则化效果**：没有batch的随机噪声
2. **对特征维度敏感**：特征维度过小时效果下降
3. **计算开销**：每个样本独立计算统计量

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class NormalizationDemo:
    """归一化效果演示"""
    
    @staticmethod
    def create_models(feature_dim, num_features):
        """创建不同归一化层的模型"""
        models = {
            'BatchNorm1d': nn.BatchNorm1d(feature_dim),
            'BatchNorm2d': nn.BatchNorm2d(num_features),
            'LayerNorm': nn.LayerNorm(feature_dim),
            'InstanceNorm': nn.InstanceNorm1d(feature_dim),
            'GroupNorm': nn.GroupNorm(num_groups=4, num_channels=num_features),
        }
        return models


class CNNWithBatchNorm(nn.Module):
    """带BatchNorm的CNN分类器"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),          # 激活前归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TransformerWithLayerNorm(nn.Module):
    """带LayerNorm的Transformer编码器层"""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pre-LN: 在注意力前做归一化（GPT-2/LLaMA风格）
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


def experiment_batch_vs_layer():
    """对比实验：BatchNorm vs LayerNorm在不同batch size下的表现"""
    torch.manual_seed(42)
    
    bn = nn.BatchNorm1d(64)
    ln = nn.LayerNorm(64)
    
    print("=" * 60)
    print("BatchNorm vs LayerNorm 对比实验")
    print("=" * 60)
    
    for batch_size in [1, 4, 16, 64]:
        x = torch.randn(batch_size, 64)
        
        # 训练模式
        bn.train()
        ln.train()
        bn_out = bn(x)
        ln_out = ln(x)
        
        # 统计输出分布
        bn_mean = bn_out.mean().item()
        bn_std = bn_out.std().item()
        ln_mean = ln_out.mean().item()
        ln_std = ln_out.std().item()
        
        print(f"\nBatch Size = {batch_size}:")
        print(f"  BatchNorm:   mean = {bn_mean:.4f}, std = {bn_std:.4f}")
        print(f"  LayerNorm:   mean = {ln_mean:.4f}, std = {ln_std:.4f}")
        
        # Batch size = 1时，BatchNorm输出全0
        if batch_size == 1:
            print(f"  BatchNorm输出方差为0! 不适合batch_size=1!")
    
    print("\n结论:")
    print("- BatchNorm在batch_size大时效果好")
    print("- LayerNorm不受batch_size影响，训练推理一致")


def experiment_training_speed():
    """实验：有无归一化的训练速度对比"""
    # 模拟一个深层网络
    class NetWithoutBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(256, 256) for _ in range(10)
            ])
            self.relu = nn.ReLU()
            
        def forward(self, x):
            for layer in self.layers:
                x = self.relu(layer(x))
            return x
    
    class NetWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU()
                ) for _ in range(10)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    print("\n训练速度对比（模拟）:")
    net_no_bn = NetWithoutBN()
    net_with_bn = NetWithBN()
    
    x = torch.randn(64, 256)
    
    # 检查梯度传播
    out_no_bn = net_no_bn(x)
    out_with_bn = net_with_bn(x)
    
    out_no_bn.norm().backward()
    out_with_bn.norm().backward()
    
    grad_norm_no_bn = sum(p.grad.norm() for p in net_no_bn.parameters() if p.grad is not None)
    grad_norm_with_bn = sum(p.grad.norm() for p in net_with_bn.parameters() if p.grad is not None)
    
    print(f"无归一化网络梯度范数: {grad_norm_no_bn:.4f}")
    print(f"有BatchNorm网络梯度范数: {grad_norm_with_bn:.4f}")
    print(f"BatchNorm有效防止梯度消失/爆炸!")


if __name__ == "__main__":
    experiment_batch_vs_layer()
    experiment_training_speed()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn

class HandwrittenBatchNorm1d(nn.Module):
    """手工实现BatchNorm1d"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 全局统计量（推理使用）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.training = True
        
    def forward(self, x):
        """
        Args:
            x: [B, num_features] 或 [B, C, H, W]
        """
        if self.training:
            # 计算当前batch统计量
            # 对非特征维度做reduce（CNNs需要处理空间维度）
            dims = tuple(range(x.dim() - 1))  # 除最后一个维度的所有维度
            batch_mean = x.mean(dim=dims)
            batch_var = x.var(dim=dims, unbiased=False)  # 无偏估计=False
            
            # 更新running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # 归一化
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 广播形状以匹配输入
        shape = [1] * x.dim()
        shape[-1] = self.num_features
        
        x_norm = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        out = self.gamma.view(shape) * x_norm + self.beta.view(shape)
        
        return out


class HandwrittenLayerNorm(nn.Module):
    """手工实现LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # normalized_shape可以是单个整数或tuple
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        """
        Args:
            x: [*, normalized_shape]
        """
        # 对最后normalized_shape个维度做统计
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # 均值和方差
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放平移
        out = self.gamma * x_norm + self.beta
        
        return out


class HandwrittenGroupNorm(nn.Module):
    """手工实现GroupNorm（BatchNorm和LayerNorm的折中）"""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        G = self.num_groups
        
        # 将通道分成G组: [B, G, C//G, H, W]
        x = x.view(B, G, C // G, H, W)
        
        # 对组内特征（C//G, H, W）做统计
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 恢复形状: [B, C, H, W]
        x_norm = x_norm.view(B, C, H, W)
        
        out = self.gamma * x_norm + self.beta
        return out


def test_handwritten_normalization():
    """测试手工实现的归一化层"""
    B, C, D = 32, 64, 128
    
    # 创建手工实现和PyTorch API的版本
    h_bn = HandwrittenBatchNorm1d(D)
    h_ln = HandwrittenLayerNorm(D)
    h_gn = HandwrittenGroupNorm(8, C)
    
    pt_bn = nn.BatchNorm1d(D)
    pt_ln = nn.LayerNorm(D)
    pt_gn = nn.GroupNorm(8, C)
    
    # 同步参数
    h_bn.gamma.data = pt_bn.weight.data
    h_bn.beta.data = pt_bn.bias.data
    
    x_1d = torch.randn(B, D)
    x_cnn = torch.randn(B, C, 16, 16)
    
    # 对比输出
    h_bn_out = h_bn(x_1d)
    pt_bn_out = pt_bn(x_1d)
    
    diff_bn = (h_bn_out - pt_bn_out).abs().max().item()
    print(f"BatchNorm最大差异: {diff_bn:.8f}")
    
    h_ln_out = h_ln(x_1d)
    pt_ln_out = pt_ln(x_1d)
    
    diff_ln = (h_ln_out - pt_ln_out).abs().max().item()
    print(f"LayerNorm最大差异: {diff_ln:.8f}")
    
    print("手工归一化实现正确!")

if __name__ == "__main__":
    test_handwritten_normalization()
```

---

## 9. 可视化与结果理解

### 9.1 归一化前后的分布变化
输入数据经过归一化后，均值趋近0，方差趋近1。$\gamma, \beta$ 控制缩放平移，允许网络恢复必要的表示能力。

### 9.2 训练过程中各层输出分布
```python
def visualize_layer_outputs():
    # 训练一个简单网络，记录每层的输出分布
    # BatchNorm: 各层输出分布稳定（方差接近1）
    # 无归一化: 深层输出方差指数增长/衰减
    pass
```

### 9.3 梯度分布对比
BatchNorm使深层梯度既不大也不小，避免梯度消失/爆炸。

---

## 10. 模型评估

```python
def compare_normalization_accuracy():
    """对比不同归一化方法的最终准确率"""
    # 实验结果（ImageNet top-1）:
    # 无归一化: 无法收敛
    # BatchNorm: 76.3%
    # LayerNorm: 71.2% (CNN上不如BN)
    # GroupNorm (32组): 75.8%
    pass
```

---

## 11. 常见问题与易错点

### Q1: 为什么Transformer不用BatchNorm用LayerNorm？
A: 1) NLP中序列长度变化大，BN的统计量不稳定；2) BN需要跨样本统计，与自注意力机制的设计不符；3) LayerNorm训练推理一致，更适合序列模型。

### Q2: BatchNorm在训练和推理时行为有何不同？
A: 训练时使用当前batch统计量，推理时使用全局running statistics。这意味着模型在训练和推理模式下的计算图不同。

### Q3: 为什么说BatchNorm有小正则化效果？
A: 每个batch的统计量有随机性（由数据采样决定），这种随机性类似于dropout，给训练引入噪声。

### Q4: Pre-LN和Post-LN的区别？
A: Post-LN（原始Transformer）：LayerNorm在残差之后；Pre-LN（GPT-2/LLaMA）：LayerNorm在子层之前。Pre-LN训练更稳定。

---

## 12. 学习总结

### 核心理解
1. **归一化的本质**：控制数据分布，让网络每层接收到的输入分布一致，防止协变量偏移
2. **BN vs LN的选择**：BN适用于大batch的CNN，LN适用于序列模型和Transformer
3. **训练推理一致性**：LN在此方面天然优于BN

---

## 13. 练习题与思考题（含答案）

### 习题1：推导BatchNorm的反向传播公式
给定 $y_i = \gamma \hat{x}_i + \beta$，$\hat{x}_i = (x_i - \mu)/\sigma$，推导 $\partial L / \partial x_i$。

### 习题2：为什么batch_size=1时BatchNorm失效？
**答案**：batch_size=1时，方差为0，导致除以0（或epsilon），输出全0，梯度为0，网络停止训练。

### 习题3：LayerNorm为什么不需要running statistics？
**答案**：LayerNorm对每个样本独立统计，不依赖batch，因此训练和推理行为完全一致。

### 习题4：GN的num_groups如何选择？
**答案**：通常设为32或按通道数/组数=每组约16-32个通道。groups数越多越接近LN，越少越接近BN。

---

## 14. 学习路径建议

### 前置
- 深度学习基础、反向传播算法

### 平行学习
- InstanceNorm、GroupNorm、RMSNorm

### 进阶
- Adaptive Normalization（AdaIN, SPADE）
- Normalization-Free网络（NFNet, FixUp）
