# Pruning 学习文档

## 1. 算法基础认知

模型剪枝（Pruning）是深度学习中用于**压缩神经网络**的核心技术之一，通过移除不重要的参数或结构来减少模型大小和计算量，同时尽量保持模型性能。模型剪枝的理论基础是神经网络的过参数化特性：现代深度学习模型通常包含数百万甚至数十亿参数，但真正对预测有贡献的只是一小部分。剪枝通过识别并移除这些"冗余"参数，可以显著减少存储需求和计算成本，使得大模型能够在资源受限的设备上运行。2019年的彩票假说（Lottery Ticket Hypothesis）进一步揭示了剪枝的深刻意义：子网络（winning tickets）可以通过与原网络相同的初始化达到甚至超越原网络的性能。

理解剪枝需要先理解深度神经网络的过参数化特性。在训练过程中，网络学习到了大量的参数，但只有一部分对最终预测有实质性贡献。剪枝的目标就是识别并移除这些"不重要"的参数，同时尽量保持模型性能。

## 2. 核心原理

模型剪枝的核心原理是**识别并移除不重要的参数或结构**。"重要性"的衡量标准有多种：参数幅度（weight magnitude）是最常用的指标，幅度小的参数对输出的贡献较小，因此可以被移除；梯度幅度反映了参数对损失的影响程度；泰勒展开（Taylor Expansion）通过计算移除参数后损失的变化来评估重要性。剪枝可以在不同粒度上进行：**非结构化剪枝**直接删除单个参数，产生稀疏矩阵，需要特殊的硬件支持；**结构化剪枝**按通道（channel）、滤波器（filter）或注意力头（head）进行，得到的模型dense且易于部署。结构化剪枝是实际应用中的主要选择，因为它不需要额外的稀疏计算支持。

剪枝的基本流程是：1）训练一个性能良好的dense模型；2）评估每个参数的重要性；3）移除不重要的参数；4）可选地微调恢复性能。这个过程可以迭代进行以获得更好的压缩效果。

## 3. 数学公式与推导

### 3.1 非结构化剪枝（Magnitude Pruning）

$$M = |W|$$

$$W_{pruned} = W \cdot (M > \theta)$$

其中θ是阈值，通常设为某个百分位数，如保留前20%的参数，则θ为80%分位数。剪枝mask m_i = 1(|w_i| > θ)，更新后的权重w_i = w_i × m_i。

### 3.2 结构化剪枝（Channel Pruning）

对于卷积层，通道的重要性定义为该通道所有参数的L2范数：

$$s_c = \sqrt{\sum_{i,j} W_{c,i,j}^2}$$

保留重要性最高的k个通道，移除其余。或者使用通道的输出特征的方差作为重要性指标：

$$s_c = \text{Var}(X_c)$$

### 3.3 泰勒重要性（Taylor Pruning）

$$\Delta L = |\frac{\partial L}{\partial w} \cdot w| = |g \cdot w|$$

按照泰勒展开的一阶项评估每个参数的重要性。移除参数后损失的变化：

$$\Delta L \approx |g \cdot w|$$

### 3.4 彩票假说验证

设原网络参数为θ_0，剪枝mask为m，winner ticket满足：

$$f(x, m \odot \theta_0) = f(x, \theta)$$

其中θ是训练后的参数，m⊙θ_0是用相同初始化的子网络。彩票假说表明存在这样的子网络可以使用原始初始化达到原网络的性能。

## 4. 训练过程讲解

典型的剪枝训练过程包括以下步骤：首先**预训练**一个高质量的dense模型；然后**剪枝**根据重要性评估移除参数或结构；可选地进行**微调**恢复性能；重复剪枝和微调的**迭代剪枝**可以获得更好的压缩率。具体流程：使用训练数据对模型进行预训练→计算每个参数的重要性分数→根据剪枝率设置阈值，移除低于���值的参数→在训练数据上微调被剪枝的模型→评估性能，如有需要则回到第二步进行下一轮剪枝。在实际应用中，剪枝率通常从10%开始，逐步增加到50%-90%，每轮微调几个epoch。

训练伪代码：
```
# 预训练
model = train_model(data)

# 剪枝
for round in range(num_rounds):
    compute_importance(model, data)
    apply_pruning(model, ratio)
    fine_tune(model, data)
```

常见的剪枝策略：
1. 一次性剪枝 vs 迭代剪枝
2. 全局剪枝 vs 本地剪枝
3. 训练中剪枝 vs 训练后剪枝

## 5. 应用场景

模型剪枝主要应用场景包括：**移动端部署**，在手机、嵌入式设备上运行大型AI模型；**边缘计算**，减少传输带宽和延迟；**模型压缩**，降低存储和内存需求；**能耗优化**，减少推理时的能耗和计算量；**神经网络搜索**，在NAS中作为搜索空间的一部分。典型应用包括ResNet、EfficientNet、Transformer等模型的压缩。在实际部署中，结构化剪枝后的模型可以直接在标准深度学习框架上运行，无需额外支持。

实际应用案例：
1. MobileNet：通过深度可分离卷积结构化剪枝
2. BERT压缩：剪枝注意力头和FFN层
3. YOLO：剪枝检测头

## 6. 优缺点分析

模型剪枝的优点包括：显著减少模型参数和计算量；降低存储需求和推理延迟；无需特殊硬件支持（结构化剪枝）；可以与其他压缩技术（量化、知识蒸馏）结合使用。缺点包括：剪枝后会损失一定的精度，需要微调来恢复；剪枝过程本身需要计算资源和时间；非结构化剪枝需要特殊的稀疏计算支持；不同任务的最佳剪枝率可能不同，需要大量调参。

| 优点 | 说明 | 适用场景 |
|------|------|----------|
| 压缩显著 | 减少参数和计算 | 移动端 |
| 部署简单 | 无需特殊硬件 | 生产环境 |
| 可组合 | 与量化结合 | 深度压缩 |

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 精度损失 | 需微调恢复 | 迭代剪枝 |
| 调参难 | 剪枝率需调 | 网格搜索 |

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


class MagnitudePruner:
    def __init__(self, model, pruning_ratio=0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                prune.remove(module, 'weight')


class GlobalPruner:
    def __init__(self, model, pruning_ratio=0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune(self):
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,
                              amount=self.pruning_ratio)
        
        for module, name in parameters_to_prune:
            prune.remove(module, name)


class ChannelPruner:
    def __init__(self, model):
        self.model = model
    
    def prune_channels(self, layer_indices, num_channels):
        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) and i in layer_indices:
                num_to_keep = module.out_channels - num_channels
                if num_to_keep > 0:
                    prune.ln_structured(module, name='weight', amount=num_to_keep,
                                   n=2, dim=0)
                    prune.remove(module, 'weight')


class IterativePruner:
    def __init__(self, model, pruning_ratio=0.1, num_iterations=5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.num_iterations = num_iterations
    
    def prune_step(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight',
                                   amount=self.pruning_ratio)
    
    def remove_step(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')


def create_pruned_model(base_model, pruning_ratio=0.5):
    pruner = GlobalPruner(base_model, pruning_ratio)
    pruner.prune()
    return base_model


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = SimpleConvNet()
    
    print(f"Parameters before pruning: {sum(p.numel() for p in model.parameters()):,}")
    
    pruner = GlobalPruner(model, pruning_ratio=0.5)
    pruner.prune()
    
    print(f"Parameters after pruning: {sum(p.numel() for p in model.parameters()):,}")
```

## 8. 手工代码实现

```python
import numpy as np
import torch


def magnitude_prune(weights, pruning_ratio=0.5):
    """基于幅度的剪枝"""
    flat_weights = weights.flatten()
    threshold = np.percentile(np.abs(flat_weights), pruning_ratio * 100)
    mask = np.abs(flat_weights) > threshold
    pruned_weights = flat_weights * mask
    return pruned_weights.reshape(weights.shape)


def channel_importance_prune(conv_weights, num_channels_to_keep):
    """通道剪枝：按通道重要性"""
    num_out_channels, num_in_channels, kh, kw = conv_weights.shape
    channel_importance = np.sum(conv_weights ** 2, axis=(1, 2, 3))
    sorted_importance = np.sort(channel_importance)[::-1]
    threshold = sorted_importance[num_channels_to_keep - 1]
    keep_mask = channel_importance >= threshold
    return keep_mask


def sensitivity_based_prune(model, data_loader, pruning_ratio=0.1):
    """基于敏感度的剪枝"""
    sensitivities = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            sensitivities[name] = np.abs(param.data.cpu().numpy())
    
    all_sensitivities = np.concatenate([v.flatten() for v in sensitivities.values()])
    threshold = np.percentile(all_sensitivities, pruning_ratio * 100)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = np.abs(param.data.cpu().numpy()) > threshold
            param.data = torch.from_numpy(param.data.cpu().numpy() * mask).float()


def lotteryicket_pruning(model, pruning_ratio=0.5):
    """彩票假说风格的剪枝"""
    original_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            original_weights[name] = param.data.clone()
    
    # 计算重要性
    flat_weights = torch.cat([p.data.flatten() for p in model.parameters() if 'weight' in p.name])
    threshold = torch.quantile(torch.abs(flat_weights), pruning_ratio)
    
    # 创建mask
    new_model = type(model)()
    return new_model


if __name__ == '__main__':
    weights = np.random.randn(64, 32, 3, 3)
    keep_mask = channel_importance_prune(weights, 32)
    print(f"Original channels: {weights.shape[0]}, Kept channels: {keep_mask.sum()}")
    
    mag_pruned = magnitude_prune(weights, 0.5)
    print(f"Magnitude pruned: {np.sum(mag_pruned != 0) / mag_pruned.size:.1%} non-zero")
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_sparsity_distribution():
    """可视化权重分布"""
    np.random.seed(42)
    weights = np.random.randn(1000, 1000) * np.random.uniform(0.1, 1.0, (1000, 1000))
    
    flat = np.abs(weights.flatten())
    
    plt.figure(figsize=(10, 6))
    plt.hist(flat, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=np.percentile(flat, 50), color='r', linestyle='--',
               label='50% threshold')
    plt.axvline(x=np.percentile(flat, 90), color='g', linestyle='--',
               label='90% threshold')
    plt.xlabel('Weight Magnitude', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Weight Magnitude Distribution', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('weight_distribution.png', dpi=150)
    plt.show()


def compare_pruning_ratios():
    """比较不同剪枝率"""
    ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    accuracies = [0.95, 0.94, 0.93, 0.91, 0.87, 0.80]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Pruning Ratio', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Pruning Ratio', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pruning_accuracy.png', dpi=150)
    plt.show()


def visualize_channel_importance():
    """可视化通道重要性"""
    np.random.seed(42)
    channels = np.random.randn(128, 64, 3, 3)
    importance = np.sum(channels ** 2, axis=(1, 2, 3))
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Output Channel', fontsize=12)
    plt.ylabel('L2 Norm', fontsize=12)
    plt.title('Channel Importance (L2 Norm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('channel_importance.png', dpi=150)
    plt.show()


def plot_model_size_comparison():
    """可视化模型大小对比"""
    methods = ['Original', 'Unstructured', 'Channel', 'Structured']
    sizes = [100, 15, 25, 40]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, sizes)
    plt.ylabel('Model Size (%)', fontsize=12)
    plt.title('Model Size Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_size.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_sparsity_distribution()
    compare_pruning_ratios()
    visualize_channel_importance()
    plot_model_size_comparison()
```

结果分析：剪枝率在50%以内时，精度下降通常在1%以内；剪枝率达到90%时，精度下降明显。通道级别的L2范数分布显示不同通道的重要性差异很大，某些通道可以安全移除。

## 10. 模型评估

模型剪枝的评估主要关注以下几个方面：**压缩率**，参数减少的比例；**精度损失**，在测试集上评估精度下降；**加速比**，推理时间的减少；**内存占用**，模型文件的减小。在实际应用中，需要在压缩率和精度之间权衡。常用的指标是Top-1 Accuracy和FLOPs（浮点运算次数）减少比例。

评估指标：
1. Compression Ratio: 原始参数/剪枝后参数
2. FLOPs Reduction: FLOPs减少比例
3. Accuracy: Top-1/Top-5 Accuracy
4. Inference Speed: 推理延迟

## 11. 常见问题与易错点

常见问题包括：**剪枝率设置**，过高导致精度大幅下降；**微调不足**，微调epoch数不够；**层间依赖**，结构化剪枝时移除的通道影响后续层。使用时的易错点：**剪枝后直接评估**，忘记微调；**剪枝mask冲突**，多次剪枝叠加；**不同层使用相同剪枝率**，不同层可能有不同的最优剪枝率。

解决方案：
1. 使用迭代剪枝
2. 充分微调
3. 不同层使用不同的剪枝率

## 12. 学习总结

模型剪枝是神经网络压缩的核心技术，通过移除不重要的参数或结构来减少模型大小和计算量。核心理念是利用网络的稀疏性和过参数化特性。非结构化剪枝产生稀疏矩阵，结构化剪枝更易于部署。剪枝与微调结合可以获得更好的压缩效果。学习剪枝时，重点理解不同剪枝方法的原理和适用场景。

## 13. 练习题与思考题（含答案）

**练习题1**：写出L1非结构化剪枝的公式。

答案：mask = (|W| > θ)，W_pruned = W × mask

**练习题2**：通道剪枝中使用什么指标评估通道重要性？

答案：L2范数 s_c = √(ΣW_{c,i,j}²)，或者通道输出特征的方差。

**思考题1**：彩票假说的核心发现是什么？

答案：存在子网络（winning ticket）使用原始初始化可以达到或超过原网络的性能，子网络通过剪枝得到。

**思考题2**：如何平衡剪枝率和精度？

答案：采用迭代剪枝，每轮剪枝10-20%，微调恢复精度。可以使用验证集监控最佳剪枝率。

### 13.3 详细答案与解析

#### 练习：计算

**问题**：权重矩阵[64, 32, 3, 3]，剪枝率50%，计算保留的通道数。

**答案**：
```
importance = sum(w^2 for each output channel)
select top 32 channels based on importance
```

## 14. 学习路径建议

学习模型剪枝：
1. 神经网络基础
2. 剪枝方法原理
3. 实践剪枝代码
4. 彩票假说
5. 深度压缩

### 14.1 扩展资源

**论文**：
1. Frankle & Carlin. "The Lottery Ticket Hypothesis"
2. Li et al. "Understanding the Mechanism"

**框架**：
1. PyTorch prune
2. TensorFlow Model Optimization