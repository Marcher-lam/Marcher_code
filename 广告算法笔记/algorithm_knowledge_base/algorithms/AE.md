# AE（自编码器）学习文档

## 1. 算法基础认知

自编码器（Autoencoder, AE）是一种无监督学习的神经网络，目标是学习数据的压缩表示。它由编码器（Encoder）和解码器（Decoder）两部分组成，中间通过一个低维的瓶颈层（Bottleneck）连接。核心思想是：让输出尽可能还原输入，迫使瓶颈层捕获数据中最本质的特征。

AE 不是一种具体算法，而是一类架构范式。根据瓶颈层的设计和训练目标的不同，衍生出了 VAE、DAE、Sparse AE 等多种变体。

## 2. 核心原理

AE 的工作流程分为三步：

1. **编码**：输入 $x$ 经过编码器映射到低维隐空间，得到隐向量 $z = f_\theta(x)$
2. **瓶颈**：$z$ 的维度远小于输入维度，形成信息瓶颈，迫使网络只保留最重要的信息
3. **解码**：隐向量 $z$ 经过解码器重构输出 $\hat{x} = g_\phi(z)$

信息瓶颈是 AE 的关键设计——因为维度远小于输入，网络无法简单复制输入，必须学习有意义的压缩表示。这使得 AE 天然具备降维和特征提取能力。

## 3. 数学公式与推导

AE 的优化目标是最小化重构误差：

$$\mathcal{L}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2 = \frac{1}{N} \sum_{i=1}^{N} \|x_i - g_\phi(f_\theta(x_i))\|^2$$

其中：
- $f_\theta$ 是编码器，参数为 $\theta$
- $g_\phi$ 是解码器，参数为 $\phi$
- $z = f_\theta(x) \in \mathbb{R}^d$，$d \ll D$（输入维度）

若输入为连续值，使用 MSE 损失；若输入为二值，使用交叉熵损失：

$$\mathcal{L} = -\sum_{i=1}^{D} [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

## 4. 训练过程讲解

1. 从数据集中采样一个 batch 的输入 $x$
2. 前向传播：$x \to z \to \hat{x}$
3. 计算重构损失 $\mathcal{L} = \|x - \hat{x}\|^2$
4. 反向传播，更新编码器和解码器参数 $\theta, \phi$
5. 重复直到收敛

训练完成后，编码器 $f_\theta$ 可作为特征提取器，瓶颈层的输出 $z$ 即为数据的低维表示。

## 5. 应用场景

- **降维与可视化**：类似 PCA 但可学习非线性映射
- **特征提取**：用编码器输出作为下游任务的输入特征
- **图像去噪**：DAE 的基础架构
- **异常检测**：异常样本重构误差显著偏高
- **数据压缩**：学习紧凑的数据表示

## 6. 优缺点分析

**优点：**
- 无监督学习，不需要标签数据
- 可学习非线性降维，比 PCA 表达能力更强
- 结构简单，训练方便

**缺点：**
- 隐空间缺乏结构化约束，不可用于生成
- 过拟合风险高，可能学到恒等映射（当隐层维度 ≥ 输入维度时）
- 缺乏概率解释，隐空间不连续

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        x_recon, z = model(batch_x)
        loss = criterion(x_recon, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class AutoencoderNumpy:
    def __init__(self, input_dim, hidden_dim, latent_dim, lr=0.001):
        self.lr = lr
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * scale2
        self.b2 = np.zeros(latent_dim)
        self.W3 = np.random.randn(latent_dim, hidden_dim) * scale2
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.randn(hidden_dim, input_dim) * scale1
        self.b4 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        self.h1 = self.relu(x @ self.W1 + self.b1)
        self.z = self.h1 @ self.W2 + self.b2
        self.h2 = self.relu(self.z @ self.W3 + self.b3)
        self.out = self.sigmoid(self.h2 @ self.W4 + self.b4)
        return self.out, self.z

    def backward(self, x):
        m = x.shape[0]
        d_out = (self.out - x) * self.out * (1 - self.out)
        dW4 = self.h2.T @ d_out / m
        db4 = d_out.mean(axis=0)
        dh2 = d_out @ self.W4.T * self.relu_grad(self.h2)
        dW3 = self.z.T @ dh2 / m
        db3 = dh2.mean(axis=0)
        dz = dh2 @ self.W3.T
        dW2 = self.h1.T @ dz / m
        db2 = dz.mean(axis=0)
        dh1 = dz @ self.W2.T * self.relu_grad(self.h1)
        dW1 = x.T @ dh1 / m
        db1 = dh1.mean(axis=0)
        for param, grad in [
            (self.W4, dW4), (self.b4, db4), (self.W3, dW3), (self.b3, db3),
            (self.W2, dW2), (self.b2, db2), (self.W1, dW1), (self.b1, db1),
        ]:
            param -= self.lr * grad

    def train_step(self, x):
        out, z = self.forward(x)
        loss = np.mean((out - x) ** 2)
        self.backward(x)
        return loss
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    sample = test_dataset.data[:10].float().view(10, -1) / 255.0
    recon, latent = model(sample)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(sample[i].view(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('原始')
axes[1, 0].set_title('重构')
plt.savefig('ae_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **重构误差（MSE）**：越低越好，衡量 AE 还原输入的能力
- **下游任务性能**：用隐向量 $z$ 做分类/聚类，评估表示质量
- **可视化**：将隐向量用 t-SNE 降维到 2D，检查同类样本是否聚集

## 11. 常见问题与易错点

- **恒等映射问题**：当隐层维度 ≥ 输入维度时，网络可能学到直接复制，应加入正则化或降低隐层维度
- **隐空间不连续**：AE 的隐空间可能有空洞，采样生成时效果差——这是 VAE 要解决的问题
- **损失不下降**：检查学习率是否过大/过小，或网络深度是否匹配数据复杂度

## 12. 学习总结

自编码器（AE）的核心贡献在于提出"编码-解码"这一简洁而强大的架构范式：通过在低维瓶颈层上强制信息压缩，迫使网络自动发现数据中最具代表性的隐含特征。这一思想深刻影响了后续几乎所有生成模型和表示学习方法的设计。

AE 的关键优势是完全无监督、结构简单、训练稳定，适合快速获取数据的低维表示。它最适合用于降维可视化、特征预训练和异常检测等场景，尤其是当标签数据稀缺时。但需注意 AE 的隐空间缺乏结构化约束，不能直接用于生成新样本。

在知识体系中，AE 是整个生成模型家族的起点：VAE 在其基础上引入概率建模和 KL 约束使隐空间连续可采样，DAE 通过添加噪声增强鲁棒性并启发了扩散模型的思想，而 GAN 则用对抗训练替代了重构损失。

工业实践中，AE 常被用作大规模特征的预训练或压缩工具，例如在广告系统中对高维稀疏用户特征做降维后供下游 CTR 模型使用。训练时需警惕隐层维度大于输入维度时的恒等映射问题，可通过添加正则化或 dropout 缓解。

## 13. 练习题与思考题（含答案）

**Q1：AE 与 PCA 的区别是什么？**

A1：PCA 是线性降维，只能捕获数据中的线性关系；AE 是非线性降维，通过非线性激活函数可以捕获复杂的非线性结构。当 AE 去掉所有激活函数时，理论上等价于 PCA。

**Q2：为什么 AE 不能直接用作生成模型？**

A2：AE 的隐空间没有概率约束，隐向量之间的区域可能是空白的、无意义的。直接从隐空间采样可能生成不合理的样本。VAE 通过引入 KL 散度约束隐空间分布来解决这个问题。

**Q3：如何用 AE 做异常检测？**

A3：用正常数据训练 AE，测试时计算重构误差。正常样本重构误差低，异常样本由于分布不同，重构误差会显著偏高，可通过设定阈值来判别。

## 14. 学习路径建议

1. 先掌握前馈神经网络和反向传播
2. 理解 AE 的编码-解码架构
3. 学习 VAE（添加概率约束）和 DAE（添加噪声）
4. 进阶学习 GAN 等其他生成模型
5. 了解扩散模型中的 AE 思想
