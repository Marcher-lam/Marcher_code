# 自编码器 (AE) 学习文档

> 无监督学习数据压缩表示，生成模型的基础起点。

> 来源线索：本节内容根据原书中关于"自编码器"的相关章节（第3章3.1节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** 自编码器通过编码器将输入压缩为低维隐表示，再由解码器重构输入，学习数据的有效压缩表示。

**直觉类比：** 自编码器像一个"信息压缩专家"——提炼最关键的特征用简短描述传递，接收方再尽量还原。编码器是"提炼"，解码器是"还原"，隐表示是"简短描述"。

**历史背景：** 自编码器最早由 Hinton 和 Salakhutdinov 于 2006 年提出，开启了深度学习的"无监督预训练"时代。后来发展为 VAE、DAE 等变体。

**算法定位：** 无监督学习、降维、特征学习、生成模型基础。

**前置知识：** 神经网络、反向传播、MSE 损失、PyTorch。

---

## 2. 核心原理

AE 由编码器 $E$ 和解码器 $D$ 组成：$z = E(x)$，$\hat{x} = D(z)$。隐表示 $z$ 维度远小于 $x$（信息瓶颈），强制网络学习最重要的特征。

### 关键概念

- **信息瓶颈**：隐维度远小于输入维度，防止恒等映射
- **重构损失**：MSE 或 BCE 衡量重构与输入的差异

---

## 3. 数学公式

$$z = \sigma(W_e x + b_e), \quad \hat{x} = \sigma(W_d z + b_d)$$

$$\mathcal{L} = \|x - \hat{x}\|^2$$

---

## 4. 训练过程讲解

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| latent_dim | 2 ~ 256 | 16 |
| lr | 1e-4 ~ 1e-3 | 1e-3 |
| batch_size | 32 ~ 256 | 128 |

---

## 5. 应用场景

1. **数据降维**：比 PCA 更强的非线性降维
2. **特征提取**：隐表示作为下游任务特征
3. **去噪**：去噪自编码器（DAE）
4. **异常检测**：重构误差大的样本可能是异常

---

## 6. 优缺点分析

### 优点
1. **无需标签**：完全无监督
2. **非线性降维**：比 PCA 表达能力更强

### 缺点
1. **不能生成**：隐空间不规则，采样后解码无意义
2. **可能过拟合**：容量大时学到恒等映射

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, 784))
        return self.decoder(z), z

model = AutoEncoder(latent_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(5):
    total = 0
    for x, _ in loader:
        recon, z = model(x)
        loss = nn.functional.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total/len(loader.dataset):.4f}')
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleAE:
    def __init__(self, input_dim=10, latent_dim=3, lr=0.01):
        scale = 0.01
        self.lr = lr
        self.W1 = np.random.randn(input_dim, 32) * scale
        self.W2 = np.random.randn(32, latent_dim) * scale
        self.W3 = np.random.randn(latent_dim, 32) * scale
        self.W4 = np.random.randn(32, input_dim) * scale

    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        z = np.maximum(0, x @ self.W1) @ self.W2
        recon = self.sigmoid(np.maximum(0, z @ self.W3) @ self.W4)
        return recon, z

np.random.seed(42)
X = np.random.randn(100, 10)
ae = SimpleAE(input_dim=10, latent_dim=3)
recon, z = ae.forward(X)
print(f"压缩: {X.shape[1]}D → {z.shape[1]}D")
```

---

## 9-14. 练习与路径

**题1：** AE 为什么不能用来生成新图像？

**参考答案：** 隐空间不规则，编码只出现在特定区域。从空白区域采样解码得到无意义结果。VAE 通过 KL 约束解决此问题。

### 学习路径
- 前置：神经网络、反向传播
- 进阶：VAE、DAE、VQ-VAE
- 推荐：Hinton & Salakhutdinov (2006)
