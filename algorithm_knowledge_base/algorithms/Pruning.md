# Pruning 学习文档

## 1. 算法基础认知

模型剪枝（Pruning）是深度学习中用于**压缩神经网络**的核心技术之一，通过移除不重要的参数或结构来减少模型大小和计算量，同时尽量保持模型性能。模型剪枝的理论基础是神经网络的过参数化特性：现代深度学习模型通常包含数百万甚至数十亿参数，但真正对预测有贡献的只是一小部分。剪枝通过识别并移除这些"冗余"参数，可以显著减少存储需求和计算成本，使得大模型能够在资源受限的设备上运行。2019年的彩票假说（Lottery Ticket Hypothesis）进一步揭示了剪枝的深刻意义：子网络（winning tickets）可以通过与原网络相同的初始化达到甚至超越原网络的性能。

## 2. 核心原理

模型剪枝的核心原理是**识别并移除不重要的参数或结构**。"重要性"的衡量标准有多种：参数幅度（weight magnitude）是最常用的指标，幅度小的参数对输出的贡献较小，因此可以被移除；梯度幅度反映了参数对损失的影响程度；泰勒展开（Taylor Expansion）通过计算移除参数后损失的变化来评估重要性。剪枝可以在不同粒度上进行：**非结构化剪枝**直接删除单个参数，产生稀疏矩阵，需要特殊的硬件支持；**结构化剪枝**按通道（channel）、滤波器（filter）或注意力头（head）进行，得到的模型dense且易于部署。结构化剪枝是实际应用中的主要选择，因为它不需要额外的稀疏计算支持。

## 3. 数学公式与推导

非结构化剪枝（Magnitude Pruning）：

$$M = |W|$$

$$W_{pruned} = W \cdot (M > \theta)$$

其中θ是阈值，通常设为某个百分位数，如保留前20%的参数，则θ为80%分位数。

结构化剪枝（Channel Pruning）：

对于卷积层，通道的重要性定义为该通道所有参数的L2范数：

$$s_c = \sqrt{\sum_{i,j} W_{c,i,j}^2}$$

保留重要性最高的k个通道，移除其余。

泰勒重要性（Taylor Pruning）：

$$\Delta L = |\frac{\partial L}{\partial w} \cdot w|$$

按照泰勒展开的一阶项评估每个参数的重要性。

彩票假说验证：

设原网络参数为θ_0，剪枝mask为m，winner ticket满足：

$$f(x, m \odot \theta_0) = f(x, \theta)$$

其中θ是训练后的参数，m⊙θ_0是用相同初始化的子网络。

## 4. 训练过程讲解

典型的剪枝训练过程包括以下步骤：首先**预训练**一个高质量的dense模型；然后**剪枝**根据重要性评估移除参数或结构；可选地进行**微调**恢复性能；重复剪枝和微调的**迭代剪枝**可以获得更好的压缩率。具体流程：使用训练数据对模型进行预训练→计算每个参数的重要性分数→根据剪枝率设置阈值，移除低于阈值的参数→在训练数据上微调被剪枝的模型→评估性能，如有需要则回到第二步进行下一轮剪枝。在实际应用中，剪枝率通常从10%开始，逐步增加到50%-90%，每轮微调几个epoch。

## 5. 应用场景

模型剪枝主要应用场景包括：**移动端部署**，在手机、嵌入式设备上运行大型AI模型；**边缘计算**，减少传输带宽和延迟；**模型压缩**，降低存储和内存需求；**能耗优化**，减少推理时的能耗和计算量；**神经网络搜索**，在NAS中作为搜索空间的一部分。典型应用包括ResNet、EfficientNet、Transformer等模型的压缩。在实际部署中，结构化剪枝后的模型可以直接在标准深度学习框架上运行，无需额外支持。

## 6. 优缺点分析

模型剪枝的优点包括：显著减少模型参数和计算量；降低存储需求��推理延迟；无需特殊硬件支持（结构化剪枝）；可以与其他压缩技术（量化、知识蒸馏）结合使用。缺点包括：剪枝后会损失一定的精度，需要微调来恢复；剪枝过程本身需要计算资源和时间；非结构化剪枝需要特殊的稀疏计算支持；不同任务的最佳剪枝率可能不同，需要大量调参。

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
    
    def remove step(self):
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
    flat_weights = weights.flatten()
    threshold = np.percentile(np.abs(flat_weights), pruning_ratio * 100)
    mask = np.abs(flat_weights) > threshold
    pruned_weights = flat_weights * mask
    return pruned_weights.reshape(weights.shape)


def channel_importance_prune(conv_weights, num_channels_to_keep):
    num_out_channels, num_in_channels, kh, kw = conv_weights.shape
    channel_importance = np.sum(conv_weights ** 2, axis=(1, 2, 3))
    threshold = np.sort(channel_importance)[-num_channels_to_keep]
    keep_mask = channel_importance >= threshold
    return conv_weights[keep_mask], keep_mask


def sensitivity_based_prune(model, data_loader, pruning_ratio=0.1):
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


if __name__ == '__main__':
    weights = np.random.randn(64, 32, 3, 3)
    pruned, _ = channel_importance_prune(weights, 32)
    print(f"Original shape: {weights.shape}, Pruned shape: {pruned.shape}")
    
    mag_pruned = magnitude_prune(weights, 0.5)
    print(f"Magnitude pruned: {np.sum(mag_pruned != 0) / mag_pruned.size:.1%} non-zero")
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_sparsity_distribution():
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


if __name__ == '__main__':
    visualize_sparsity_distribution()
    compare_pruning_ratios()
    visualize_channel_importance()
```

结果分析：剪枝率在50%以内时，精度下降通常在1%以内；剪枝率达到90%时，精度下降明显。通道级别的L2范数分布显示不同通道的重要性差异很大，某些通道可以安全移除。

## 10. 模型评估

模型剪枝的评估主要关注以下几个方面：**压缩率**，参数减少的比例；**精度损失**，在测试集上评估精度下降；**加速比**，推理时间的减少；**内存占用**，模型文件的减小。在实际应用中，需要在压缩率和精度之间权衡。常用的指标是Top-1 Accuracy和FLOPs（浮点运算次数）减少比例。

## 11. 常见问题与易错点

常见问题包括：**剪枝率设置**，过高导致精度大幅下降；**微调不足**，微调epoch数不够；**层间依赖**，结构化剪枝时移除的通道影响后续层。使用时的易错点包括：**剪枝后直接评估**，忘记微调；**剪枝mask冲突**，多次剪枝叠加；**不同层使用相同剪枝率**，不同层可能有不同的最优剪枝率。

## 12. 学习总结

模型剪枝是神经网络压缩的核心技术，通过移除不重要的参数或结构来减少模型大小和计算量。核心理念是利用网络的稀疏性和过参数化特性。非结构化剪枝产生稀疏矩阵，结构化剪枝更易于部署。剪枝与微调结合可以获得更好的压缩效果。学习剪枝时，重点理解不同剪枝方法的原理和适用场景。

## 13. 练习题与思考题与思考题（含答案）

**练习题1**：写出L1非结构化剪枝的公式。

答案：mask = (|W| > θ)，W_pruned = W × mask

**练习题2**：通道剪枝中使用什么指标评估通道重要性？

答案：L2范数 s_c = √(ΣW_{c,i,j}²)，或者通道输出特征的方差。

**思考题1**：彩票假说的核心发现是什么？

答案：存在子网络（winning ticket）使用原始初始化可以达到或超过原网络的性能，子网络通过剪枝得到。


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：Pruning的[核心概念]是什么？

**答案**：**答案是[B]**。

**解析**：
Pruning的核心机制是[机制描述]。根据算法的数学定义，有：
$$核心公式$$
代入[具体值]后，验证可得正确答案为[B]。

选项分析：
- A：这是对[另一概念]的描述，与Pruning不符
- B：✓ 正确，这是[核心概念]的准确定义
- C：虽然有一定关联，但不是Pruning的主要特性
- D：这是[另一算法]的特征，在Pruning中不适用

#### 练习2：手动计算

**问题**：给定以下数据，请手动计算Pruning的[参数/结果]：
- 输入：$X = [x_1, x_2, ...]$
- 标签：$y = [y_1, y_2, ...]$

**答案**：**计算结果为[具体值]**

**解析**：
**步骤1**：根据Pruning的定义，计算[第一中间量]
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

**问题**：Pruning在[特定场景]下效果不佳，请分析原因并提出改进方案。

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

学习模型剪枝建议按照以下路径进行：先学习神经网络基础和参数概念；理解不同剪枝方法的原理；实践非结构和结构化剪枝；学习彩票假说和相关工作；结合量化、知识蒸馏进行深度压缩。