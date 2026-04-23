#!/usr/bin/env python3
"""补全Dropout.md缺失的第13-14章"""
import re

path = "/Users/marcher/Desktop/Marcher_code/algorithm_knowledge_base/algorithms/Dropout.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

if '## 13.' not in content:
    add = """
## 13. 练习题与思考题
### 13.1 基础练习题
**练习1：Dropout的作用机制**
问题：在训练阶段使用Dropout(p=0.5)时，请解释每个神经元被"丢弃"的概率是多少，以及这与推理阶段的权重缩放有什么关系？
**答案**：每个神经元在每次训练迭代中被丢弃的概率为p=0.5。训练时保留概率为1-p=0.5。推理阶段由于所有神经元都参与计算，需要将权重乘以(1-p)来保持期望一致，即权重缩放。

**练习2：Dropout与其他正则化的对比**
问题：Dropout、L2正则化、数据增强都是防止过拟合的技术，请分析它们各自的作用机制和适用场景。
**答案**：Dropout通过随机丢弃神经元减少共适应；L2正则化通过惩罚大的权重参数；数据增强通过增加训练样本多样性。Dropout适合大型神经网络，数据增强适合数据有限场景，L2正则化适合参数模型。

### 13.2 进阶思考题
**思考题：变分Dropout与Monte Carlo Dropout**
问题：标准Dropout在推理时使用权重缩放作为近似，而MC Dropout在推理时保持Dropout开启并多次采样。请分析两种方法的差异，以及MC Dropout如何提供预测的不确定性估计。
**答案**：权重缩放是确定性近似，计算快速但精度有限。MC Dropout保持随机性，多次采样得到预测分布的均值和方差，从而估计预测不确定性。这对于贝叶斯推断和异常检测非常有用。

## 14. 学习路径建议
### 14.1 前置知识
- 神经网络基础（前馈神经网络、反向传播）
- PyTorch/TensorFlow基础
- 概率论基础（期望、方差）
- 过拟合与正则化概念

### 14.2 平行算法
- **Weight Decay（L2正则化）**：参数范数惩罚，与Dropout互补
- **Batch Normalization**：通过mini-batch统计量归一化，间接正则化
- **Mixup/CutMix**：数据增强的正则化方法

### 14.3 进阶算法
- **变分Dropout**：将Dropout解释为变分推断，实现贝叶斯神经网络
- **DropConnect**：Dropout的神经元级别推广，随机丢弃权重而非激活
- **Spatial Dropout**：对整个特征图通道Dropout，用于CNN

### 14.4 推荐资源
**书籍**：《深度学习》第7章（正则化），《神经网络与深度学习》第5章
**论文**：
- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"（Srivastava et al., 2014）
- "Variational Dropout and the Local Reparameterization Trick"（Kingma et al., 2015）
**代码**：PyTorch官方Dropout实现 - torch.nn.Dropout
"""
    content = content.rstrip()
    if not content.endswith('```'):
        content += '\n```'
    content += add
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Dropout.md: 补全后 {len(re.findall(r'^## ', content, re.M))} 章")
else:
    print("✓ Dropout.md: 已有13/14章")