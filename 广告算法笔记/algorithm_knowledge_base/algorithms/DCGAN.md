# DCGAN（深度卷积生成对抗网络）学习文档

## 1. 算法基础认知

DCGAN（Deep Convolutional GAN）由 Radford 等人于 2015 年提出，是将卷积神经网络与 GAN 结合的里程碑工作。原始 GAN 使用全连接网络，生成图像质量较差。DCGAN 通过引入卷积/转置卷积结构和一系列架构设计约束，大幅提升了 GAN 的训练稳定性和生成质量，成为后续几乎所有图像 GAN 的基础架构。

## 2. 核心原理

DCGAN 的核心贡献不是新算法，而是一套**架构设计指南**：

1. **全卷积网络**：$G$ 和 $D$ 均使用卷积层替代全连接层，$G$ 用转置卷积上采样，$D$ 用步进卷积下采样
2. **Batch Normalization**：$G$ 和 $D$ 中除输出层外都使用 BN，稳定训练
3. **去除池化层**：用步进卷积替代池化，让网络自己学习下采样方式
4. **激活函数**：$G$ 中间层用 ReLU，输出用 Tanh；$D$ 用 LeakyReLU
5. **生成器架构**：从 $z$ 开始，逐步通过转置卷积将空间分辨率放大（如 $1\times1 \to 4\times4 \to 8\times8 \to 16\times16 \to 32\times32$）

转置卷积（反卷积）是 DCGAN 的关键操作——它将低维特征图逐步上采样到高维图像。

## 3. 数学公式与推导

DCGAN 的目标函数与原始 GAN 相同：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**转置卷积的数学含义**：普通卷积 $y = W * x$ 的转置操作 $x' = W^T * y$ 就是转置卷积。它将低分辨率特征映射回高分辨率空间，实现上采样。

**Batch Normalization**：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

BN 使得每层输入的分布稳定，缓解了 GAN 训练中的梯度问题。

## 4. 训练过程讲解

1. 从噪声分布采样 $z$，通过 $G$ 的转置卷积层逐步上采样生成图像
2. 真实图像和生成图像分别通过 $D$ 的步进卷积层，输出真/假判别
3. 交替训练 $D$ 和 $G$，与原始 GAN 训练流程相同
4. 关键超参数：学习率 $2 \times 10^{-4}$，$\beta_1 = 0.5$（Adam）， latent_dim = 100

DCGAN 的 $G$ 利用卷积的局部性，生成的图像比全连接 GAN 更连贯、更清晰。

## 5. 应用场景

- **图像生成**：生成高质量的人脸、物体、场景
- **特征提取**：$D$ 的卷积层可作为预训练特征提取器
- **超分辨率**：类似架构用于图像放大
- **图像编辑**：在隐空间中操纵图像属性

## 6. 优缺点分析

**优点：**
- 架构规范，训练比原始 GAN 稳定得多
- 利用卷积的归纳偏置，生成图像质量显著提升
- 成为后续 GAN 研究的标准基线

**缺点：**
- 仍有模式坍塌问题
- 仅适用于图像数据
- 分辨率受限（原始论文 64×64），更高分辨率需要改进架构

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

latent_dim = 100
G = DCGANGenerator(latent_dim)
D = DCGANDiscriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(50):
    for real_x, _ in loader:
        bs = real_x.size(0)
        real_labels = torch.ones(bs, 1)
        fake_labels = torch.zeros(bs, 1)

        z = torch.randn(bs, latent_dim, 1, 1)
        fake_x = G(z)
        d_loss = criterion(D(real_x), real_labels) + criterion(D(fake_x.detach()), fake_labels)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        z = torch.randn(bs, latent_dim, 1, 1)
        g_loss = criterion(D(G(z)), real_labels)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
    print(f"Epoch {epoch+1}, D: {d_loss.item():.4f}, G: {g_loss.item():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def conv2d(x, W, b, stride=1, padding=0):
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = W.shape
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out = np.zeros((n, c_out, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = x[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[:, :, i, j] = np.tensordot(patch, W, axes=([1,2,3],[1,2,3])) + b
    return out

def conv_transpose2d(x, W, b, stride=2, padding=1):
    n, c_in, h_in, w_in = x.shape
    c_out, _, kh, kw = W.shape
    h_out = (h_in - 1) * stride - 2 * padding + kh
    w_out = (w_in - 1) * stride - 2 * padding + kw
    out = np.zeros((n, c_out, h_out, w_out))
    for i in range(h_in):
        for j in range(w_in):
            h_start = i * stride - padding
            w_start = j * stride - padding
            for ci in range(c_out):
                for ki in range(kh):
                    for kj in range(kw):
                        hi, wi = h_start + ki, w_start + kj
                        if 0 <= hi < h_out and 0 <= wi < w_out:
                            out[:, ci, hi, wi] += np.sum(x[:, :, i, j] * W[ci, :, ki, kj], axis=1)
    out += b
    return out

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha * x, x)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

G.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1)
    samples = G(z).cpu()
    samples = samples * 0.5 + 0.5

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        idx = i * 4 + j
        axes[i, j].imshow(samples[idx, 0], cmap='gray')
        axes[i, j].axis('off')
plt.suptitle('DCGAN 生成结果')
plt.savefig('dcgan_samples.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **FID Score**：生成图像与真实图像在 Inception 特征空间的距离
- **IS Score**：衡量生成图像质量和多样性
- **视觉质量**：直接观察图像清晰度和连贯性
- **训练稳定性**：观察 $D$ 和 $G$ 损失曲线是否保持平衡

## 11. 常见问题与易错点

- **生成器输出全为同一颜色**：检查学习率、BN 是否正确、是否使用了 Tanh 输出
- **棋盘格伪影**：转置卷积的已知问题，解决方案包括使用 resize-convolution 代替转置卷积
- **$G$ 和 $D$ 的训练不平衡**：通常需要 $D$ 略强于 $G$，但不能太强
- **BatchNorm 位置**：$G$ 的输出层不加 BN，$D$ 的输入层不加 BN

## 12. 学习总结

DCGAN 的核心贡献是提出了 CNN-GAN 的标准架构范式。通过卷积/转置卷积、BatchNorm、步进卷积等设计，大幅提升了 GAN 的稳定性和生成质量。DCGAN 是理解 StyleGAN、PGGAN 等后续工作的基础。

## 13. 练习题与思考题（含答案）

**Q1：DCGAN 相比原始 GAN 的关键改进是什么？**

A1：将全连接层替换为卷积/转置卷积，引入 BatchNorm，使用步进卷积替代池化。这些改动利用了卷积的局部性和参数共享，提升了图像生成质量和训练稳定性。

**Q2：为什么 $D$ 的输入层不加 BatchNorm？**

A2：因为 $D$ 的输入是真实数据或生成数据，如果加 BN 会改变真实数据的分布统计量，影响判别能力。实验表明输入层加 BN 会导致训练不稳定。

**Q3：转置卷积和普通上采样+卷积有什么区别？**

A3：转置卷积是可学习的上采样，参数更少但可能产生棋盘格伪影。上采样+卷积先用插值放大再卷积，效果更平滑但计算量更大。

## 14. 学习路径建议

1. 掌握 GAN 基础原理和训练方法
2. 理解卷积操作和转置卷积
3. 实现 DCGAN 并复现论文结果
4. 进阶：WGAN、ProgGAN、StyleGAN
5. 了解 GAN 在扩散模型时代的新定位
