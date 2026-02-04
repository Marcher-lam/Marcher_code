# 深度学习基础

本模块涵盖深度学习的核心概念、神经网络架构和实践。

## 📚 目录结构

```
003-deep-learning/
├── 01-neural-networks.ipynb      # 神经网络基础
├── 02-cnn.ipynb                  # 卷积神经网络
├── 03-rnn.ipynb                  # 循环神经网络
├── 04-attention.ipynb            # 注意力机制
├── 05-transformer.ipynb          # Transformer架构
├── 06-training-techniques.ipynb  # 训练技巧
└── README.md
```

## 🎯 学习目标

### 1. 神经网络基础
- 感知机与多层感知机（MLP）
- 前向传播与反向传播
- 激活函数（ReLU、Sigmoid、Tanh、GELU）
- 损失函数（MSE、交叉熵）
- 优化器（SGD、Adam、AdamW）

### 2. 卷积神经网络（CNN）
- 卷积层、池化层
- 经典架构：LeNet、AlexNet、VGG、ResNet
- 卷积的可视化与理解
- 迁移学习

### 3. 循环神经网络（RNN）
- RNN、LSTM、GRU
- 序列建模
- 梯度消失与梯度爆炸
- 双向RNN

### 4. 注意力机制与Transformer
- Self-Attention机制
- Transformer架构
- 位置编码
- BERT、GPT系列

### 5. 训练技巧
- Dropout、Batch Normalization
- 学习率调度
- 数据增强
- 正则化技术
- 混合精度训练

## 📖 核心概念

### 前向传播
```
输入层 → 隐藏层 → 输出层
  X    →   h1   →   y
            ↓
       h = f(Wx + b)
```

### 反向传播
```
损失函数 L
    ↓
∂L/∂y → ∂L/∂h → ∂L/∂W → ∂L/∂b
    ↓           ↓
  梯度      更新参数
```

## 🛠️ 技术栈

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
```

## 📝 学习路径

```
1. 神经网络基础（MLP）
   ↓
2. CNN（图像处理）
   ↓
3. RNN（序列处理）
   ↓
4. Transformer（现代架构）
   ↓
5. 实际项目
```

## 💡 实践项目

### 初级
- [ ] 手写数字识别（MLP）
- [ ] CIFAR-10分类（CNN）

### 中级
- [ ] 图像分类（ResNet）
- [ ] 文本分类（LSTM/Transformer）
- [ ] 情感分析

### 高级
- [ ] 目标检测
- [ ] 图像生成（GAN）
- [ ] 序列生成

## 📚 推荐资源

### 书籍
- 《深度学习》（花书）
- 《动手学深度学习》
- 《Python深度学习》

### 课程
- CS231n: CNN for Visual Recognition
- CS224n: NLP with Deep Learning
- Fast.ai Practical Deep Learning

### 框架文档
- PyTorch Documentation
- TensorFlow Documentation

## 🔗 关键概念对比

| 概念 | CNN | RNN | Transformer |
|------|-----|-----|-------------|
| 适用数据 | 图像、网格 | 序列 | 序列、图像 |
| 参数共享 | 空间维度 | 时间维度 | 全局注意力 |
| 并行化 | 高 | 低 | 高 |
| 长程依赖 | 弱 | 中 | 强 |

## 💻 编程实践

### 标准流程
1. 数据准备（Dataset、DataLoader）
2. 模型定义（继承nn.Module）
3. 损失函数和优化器
4. 训练循环
5. 验证和测试
6. 模型保存和加载

### 最佳实践
- 使用GPU加速
- 梯度裁剪
- 早停法（Early Stopping）
- 模型检查点
- TensorBoard可视化
