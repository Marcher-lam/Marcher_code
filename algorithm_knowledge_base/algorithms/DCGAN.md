# DCGAN 学习文档

> Deep Convolutional GAN，将卷积神经网络引入生成对抗网络的里程碑式工作。

---

## 1. 算法基础认知

### 1.1 一句话定义

DCGAN（Deep Convolutional Generative Adversarial Network）是2015年提出的将深度卷积神经网络应用于生成对抗网络的基础架构，通过卷积操作替代全连接层，实现了更稳定的GAN训练和更高质量的图像生成。

### 1.2 直觉类比

将DCGAN想象为**艺术家与鉴赏家的对抗训练**：生成器像一位正在学习绘画的艺术家，不断尝试创作新作品；判别器像一位鉴赏家，不断提高鉴赏标准。在这种对抗过程中，艺术家（生成器）的技艺越来越精湛，最终能够创作出足以以假乱真的作品。

### 1.3 历史背景

- **2014年**：Goodfellow提出原始GAN，使用全连接网络
- **2015年**：Radford等发表DCGAN论文，将CNN引入GAN
- **2016-2017年**：DCGAN成为GAN研究的基础架构
- **2018年后**：各种改进版本如WGAN、PGGAN、CycleGAN等相继出现

### 1.4 算法定位

- **类型**：生成模型 -> 生成对抗网络
- **输出**：生成图像（64×64或更高分辨率）
- **模型类型**：生成模型/无监督学习
- **核心创新**：卷积结构替代全连接，BN替换为其他归一化方法

### 1.5 前置知识

- GAN基础：生成器、判别器、对抗训练
- CNN基础：卷积、池化、激活函数
- 深度学习框架：PyTorch张量操作
- 优化基础：梯度下降、损失函数

---

## 2. 核心原理

### 2.1 核心思想

DCGAN的核心思想是将卷积神经网络引入生成对抗网络，通过以下架构设计实现稳定训练：

1. **生成器**：使用转置卷积（Fractional Strided Conv）进行上采样，从噪声向量生成图像
2. **判别器**：使用标准卷积进行下采样，判断图像真伪
3. **架构约束**：移除全连接层，使用Batch Normalization（在生成器最后层除外）
4. **激活函数**：生成器使用ReLU，判别器使用LeakyReLU

### 2.2 工作流程

```
噪声向量 z → 生成器 G → 生成的图像 G(z)
真实图像 x → 判别器 D → 真假判断
对抗训练：min_G max_D V(D, G)
```

### 2.3 关键架构改进

| 原始GAN | DCGAN |
|--------|------|
| 全连接层 | 转置卷积/标准卷积 |
| Sigmoid激活 | LeakyReLU |
| Batch Norm（所有层） | 生成器最后层不使用BN |
| 池化层 | 步长卷积 |
| 激活输出 | Tanh输出 |

### 2.4 生成器架构

```
输入: 噪声 z ~ N(0, I), shape=(batch, 100)
  ↓
全连接层 + BN + ReLU: (100, 512, 4, 4)
  ↓
转置卷积 + BN + ReLU: (256, 3, 8, 8)
  ↓
转置卷积 + BN + ReLU: (128, 3, 16, 16)
  ↓
转置卷积 + BN + ReLU: (64, 3, 32, 32)
  ↓
转置卷积 + Tanh: (3, 3, 64, 64)
输出: 图像 G(z), shape=(batch, 3, 64, 64)
```

### 2.5 判别器架构

```
输入: 图像 x, shape=(batch, 3, 64, 64)
  ↓
卷积 + LeakyReLU: (64, 3, 32, 32)
  ↓
卷积 + BN + LeakyReLU: (128, 3, 16, 16)
  ↓
卷积 + BN + LeakyReLU: (256, 3, 8, 8)
  ↓
卷积 + BN + LeakyReLU: (512, 3, 4, 4)
  ↓
全连接层 + Sigmoid: (1)
输出: D(x), 图像为真的概率
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $z$ | 噪声向量 | $(batch, 100)$ |
| $x$ | 真实图像 | $(batch, 3, H, W)$ |
| $G(z)$ | 生成图像 | $(batch, 3, H, W)$ |
| $D(\cdot)$ | 判别概率 | $(batch, 1)$ |
| $\theta_g$ | 生成器参数 | - |
| $theta_d$ | 判别器参数 | - |

### 3.2 对抗损失函数

DCGAN使用二元交叉熵作为对抗损失：

$$
V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

**判别器目标**：最大化对上式的期望
$$
\max_D V(D, G) = \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]]
$$

**生成器目标**：最小化判别器能够区分的概率
$$
\min_G V(D, G) = \mathbb{E}_z[\log(1 - D(G(z))))]
$$

等价形式（生成器使用-log D目标）：
$$
\max_G \mathbb{E}_z[\log D(G(z))]
$$

### 3.3 卷积操作数学

**标准卷积（前向）**：
```
输入 X ∈ ℝ^(N, C_in, H, W)
卷积核 W ∈ ℝ^(C_out, C_in, k, k)
输出 Y ∈ ℝ^(N, C_out, H', W')

Y[n, c_out, h, w] = Σ_c Σ_i Σ_j X[n, c, h+i, w+j] * W[c_out, c, i, j]
```

**转置卷积（前向）**：
```
输入 X ∈ ℝ^(N, C_in, H, W)
卷积核 W ∈ ℝ^(C_in, C_out, k, k)
输出 Y ∈ ℝ^(N, C_out, H', W')

Y[n, c_out, h', w'] = Σ_c Σ_i Σ_j X[n, c, h'-i, w'-j] * W[c, c_out, i, j]
```

### 3.4 训练稳定性分析

DCGAN成功的关键因素：

1. **特征空间约束**：BN确保每层输入数据分布稳定
2. **梯度流动**：LeakyReLU避免梯度消失
3. **模式稳定**：噪声输入空间连续便于插值
4. **上采样质量**：转置卷积产生更少伪影

### 3.5 损失 landscapes

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_landscape():
    """可视化GAN损失 landscapes"""
    
    # 模拟损失曲面
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 判别器损失（简化版）
    Z = np.log(1 + np.exp(X)) + np.log(1 + np.exp(-Y))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('D parameters')
    ax.set_ylabel('G parameters')
    ax.set_title('GAN Loss Landscape')
    plt.savefig('dcgan_loss.png', dpi=150)
    plt.show()

plot_loss_landscape()
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import torch
import torchvision.transforms as transforms

def get_transforms():
    """DCGAN图像预处理 - 训练CelebA数据集"""
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return transform
```

### 4.2 模型定义

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """DCGAN生成器"""
    
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: z, (100, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (nc, 64, 64)
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """DCGAN判别器"""
    
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
```

### 4.3 训练循环

```python
import torch.optim as optim

def train_dcgan(netG, netD, dataloader, num_epochs=5, lr=0.0002, beta1=0.5):
    """DCGAN训练循环"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = netG.to(device)
    netD = netD.to(device)
    
    criterion = nn.BCELoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            # ==== 训练判别器 ====
            netD.zero_grad()
            
            output = netD(real_images)
            errD_real = criterion(output, real_labels)
            
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            errD_fake = criterion(output, fake_labels)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            
            # ==== 训练生成器 ====
            netG.zero_grad()
            
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images)
            errG = criterion(output, real_labels)
            
            errG.backward()
            optimizerG.step()
            
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
    return G_losses, D_losses
```

### 4.4 收敛条件与监控

```python
def monitor_training(G_losses, D_losses):
    """监控训练收敛"""
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('DCGAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    d_to_g_ratio = np.array(D_losses[-100:]) / (np.array(G_losses[-100:]) + 1e-8)
    print(f"D/G ratio (last 100 iter): {np.mean(d_to_g_ratio):.2f}")
```

### 4.5 超参数推荐

| 超参数 | 作用 | 推荐值 |
|--------|------|--------|
| nz | ��声��度 | 100 |
| batch_size | 批量大小 | 128 |
| learning_rate | 学习率 | 0.0002 |
| beta1 | Adam动量 | 0.5 |
| num_epochs | 训练轮数 | 5-100 |
| ngf/ndf | 特征维度 | 64 |

---

## 5. 应用场景

### 5.1 典型应用

- **人脸生成**：生成 celebrity 人脸图像
- **艺术风格迁移**：生成特定风格图像
- **数据增强**：生成扩充训练数据
- **图像修复**：填补图像缺失部分
- **超分辨率**：图像超分辨率重建

### 5.2 适用数据特征

- 图像数据（RGB格式）
- 分辨率建议 64×64 以上
- 训练样本充足（万级）
- 类别明确的数据集

### 5.3 不适用场景

- 小样本数据集
- 高分辨率图像生成
- 条件生成（需要CGAN）
- 特定属性控制

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 稳定训练 | 架构改进使GAN训练更稳定 |
| 高质量生成 | 比全连接GAN生成质量更高 |
| 可解释特征 | 潜在空间可做算术运算 |
| 端到端 | 可以无监督学习 |
| 广泛应用 | 成为后续GAN的基础 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 模式坍塌 | 生成单一模式 | WGAN、混合损失 |
| 训练不稳定 | 需细致调参 | 谱归一化 |
| 伪影 | 生成图像Artifacts | 渐进增长 |
| 计算成本 | 需GPU资源 | 减少分辨率 |

---

## 7. 调库实现（PyTorch）

### 7.1 使用torchvision DCGAN示例

```python
import torch
import torch.nn as nn
from torchvision import utils

def use_dcgan_library():
    """PyTorch DCGAN使用示例"""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nz = 100
    ngf = 64
    ndf = 64
    
    netG = Generator(nz, ngf, 3).to(DEVICE)
    netD = Discriminator(ndf, 3).to(DEVICE)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    noise = torch.randn(8, nz, 1, 1, device=DEVICE)
    fake_images = netG(noise)
    
    print(f"Generated images shape: {fake_images.shape}")
    print(f"Value range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
    
    return netG, netD

use_dcgan_library()
```

### 7.2 完整训练脚本

```python
def complete_training_script():
    """完整DCGAN训练脚本"""
    
    import os
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MNIST(root='./data', train=True, transform=get_transforms(), download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    print("Training DCGAN on MNIST...")
    print(f"Device: {DEVICE}")
    print(f"Batches: {len(dataloader)}")
    
    return dataloader, netG, netD
```

---

## 8. 手工代码实现

### 8.1 简化的NumPy实现

```python
import numpy as np

class SimpleDCGAN:
    """简化版DCGAN NumPy实现"""
    
    def __init__(self, nz=100, img_size=64):
        self.nz = nz
        self.img_size = img_size
        self.W1 = np.random.randn(nz, 512) * 0.01
        self.W2 = np.random.randn(512, img_size*img_size) * 0.01
    
    def generate(self, z):
        """生成图像"""
        h = np.dot(z, self.W1)
        h = np.maximum(0, h)
        img = np.dot(h, self.W2)
        img = np.tanh(img)
        return img.reshape(-1, self.img_size, self.img_size)
    
    def discriminate(self, img):
        """判别图像"""
        img_flat = img.reshape(img.shape[0], -1)
        h = np.dot(img_flat, self.W2.T)
        h = np.maximum(0, h)
        score = np.dot(h, self.W1)
        prob = 1 / (1 + np.exp(-score))
        return prob


if __name__ == '__main__':
    dcgan = SimpleDCGAN()
    z = np.random.randn(4, 100)
    imgs = dcgan.generate(z)
    print(f"Generated: {imgs.shape}")
```

### 8.2 手动实现卷积层

```python
import numpy as np

def conv2d_manual(X, W, stride=1, padding=0):
    """手动实现二维卷积"""
    c_out, c_in, k, k = W.shape
    n, c_in_h, h_in_w = X.shape
    
    h_out = (h_in_w + 2*padding - k) // stride + 1
    w_out = (h_in_w + 2*padding - k) // stride + 1
    
    X_pad = np.pad(X, ((0,0), (padding, padding), (padding, padding)))
    Y = np.zeros((n, c_out, h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            Y[:, :, i, j] = np.tensordot(
                X_pad[:, :, h_start:h_start+k, w_start:w_start+k],
                W, axes=([1,2],[1,2])
            )
    
    return Y
```

---

## 9. 可视化与结果理解

### 9.1 生成结果可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_generated_images(netG, num_images=16):
    """可视化生成的图像"""
    
    device = next(netG.parameters()).device
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    
    with torch.no_grad():
        fake_images = netG(noise).cpu()
    
    grid = utils.make_grid(fake_images, nrow=4, normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('DCGAN Generated Images')
    plt.savefig('generated_images.png', dpi=150)
    plt.show()


def visualize_interpolation(netG):
    """潜在空间插值可视化"""
    
    device = next(netG.parameters()).device
    
    z1 = torch.randn(1, 100, 1, 1, device=device)
    z2 = torch.randn(1, 100, 1, 1, device=device)
    
    interpolation = []
    for alpha in np.linspace(0, 1, 10):
        z = alpha * z1 + (1-alpha) * z2
        with torch.no_grad():
            img = netG(z).cpu()
        interpolation.append(img[0])
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolation[i].permute(1, 2, 0))
        ax.axis('off')
    plt.title('Latent Space Interpolation')
    plt.savefig('interpolation.png', dpi=150)
    plt.show()
```

### 9.2 损失曲线分析

```python
def analyze_loss_curves(G_losses, D_losses):
    """分析损失曲线"""
    
    import pandas as pd
    
    df = pd.DataFrame({
        'G_loss': G_losses,
        'D_loss': D_losses
    })
    
    rolling = df.rolling(window=100).mean()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rolling['G_loss'], label='Generator')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(rolling['D_loss'], label='Discriminator')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_analysis.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import pairwise_distances

def evaluate_dcgan(netG, real_images, num_samples=1000):
    """DCGAN评估指标"""
    
    device = next(netG.parameters()).device
    
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake_images = netG(noise).cpu().numpy()
    
    real = real_images[:num_samples].numpy()
    
    metrics = {}
    
    real_mean = np.mean(real, axis=(0, 2, 3))
    fake_mean = np.mean(fake_images, axis=(0, 2, 3))
    metrics['mean_diff'] = np.mean(np.abs(real_mean - fake_mean))
    
    real_std = np.std(real, axis=(0, 2, 3))
    fake_std = np.std(fake_images, axis=(0, 2, 3))
    metrics['std_diff'] = np.mean(np.abs(real_std - fake_std))
    
    return metrics
```

### 10.2 常用评估方法

- **Inception Score (IS)**：使用Inception网络计算
- **Frechet Inception Distance (FID)**：计算真实与生成特征的分布距离
- **视觉检查**：人工判断生成质量
- **潜在空间操作**：在潜在空间做向量运算

---

## 11. 常见问题与易错点

### 11.1 训练不收敛

**问题**：判别器损失快速降到0，生成器无法学习

**原因**：
- 生成器太弱，判别器太强
- 学习率过高
- 标签噪声不足

**解决方案**：
```python
# 使用标签平滑
real_labels = torch.rand(batch_size) * 0.1 + 0.9
fake_labels = torch.rand(batch_size) * 0.1
```

### 11.2 模式坍塌

**问题**：生成器总是生成相似的图像

**原因**：
- 损失函数不够强
- 潜在空间不连续

**解决方案**：
- 使用小批量判别
- 特征匹配
- 渐进式增长

### 11.3 伪影问题

**问题**：生成图像有棋盘格伪影

**原因**：转置卷积不均匀重叠

**解决方案**：
- 调整卷积核大小为奇数
- 使用上采样+卷积替代转置卷积

---

## 12. 学习总结

### 12.1 核心要点回顾

1. **架构创新**：使用卷积替代全连接
2. **关键设计**：BN、激活函数选择
3. **训练目标**：对抗损失优化
4. **潜在空间**：可做向量运算
5. **应用广泛**：图像生成各种任务

### 12.2 从DCGAN到其他GAN

```
DCGAN
  ↓
WGAN (稳定训练)
  ↓
CGAN (条件生成)
  ↓
CycleGAN (风格迁移)
  ↓
StyleGAN (高质量)
  ↓
BigGAN (大规模)
```

### 12.3 学习建议

1. 先理解GAN基础原理
2. 实现DCGAN架���
3. 在小数据集实验
4. 观察潜在空间性质
5. 对比不同GAN变体

---

## 13. 练习题与思考题

### 练习题

**练习1**：计算DCGAN生成器的参数量，假设 nz=100, ngf=64, nc=3

<details>
<summary>答案</summary>

```
转置卷积层1: 100*64*8*4*4 = 819200
转置卷积层2: 512*256*4*4 = 2,097,152
转置卷积层3: 256*128*4*4 = 524,288
转置卷积层4: 128*64*4*4 = 131,072
转置卷积层5: 64*3*4*4 = 12,288
Total: ~3.6M 参数
```

</details>

**练习2**：为什么DCGAN使用Tanh而不是Sigmoid作为生成器输出？

<details>
<summary>答案</summary>

Tanh输出范围[-1, 1]，与图像归一化范围匹配，便于判别器学习。如果使用Sigmoid[0, 1]，需要调整数据预处理。

</details>

### 思考题

**思考题1**：DCGAN与后续GAN（如StyleGAN）的主要区别是什么？

<details>
<summary>答案</summary>

| 方面 | DCGAN | StyleGAN |
|------|------|--------|
| 潜在空间 | 随机噪声 | 潜在向量映射 |
| 归一化 | Batch Norm | Adaptive Norm |
| 生成质量 | 64×64 | 1024×1024 |
| 控制能力 | 无 | 风格控制 |

</details>

**思考题2**：如何解决DCGAN的模式坍塌问题？

<details>
<summary>答案</summary>

1. 小批量判别：查看多个生成样本的多样性
2. 特征匹配：匹配真实特征统计
3. 渐进式增长：先小分辨率后增大
4. 混合损失：结合像素和感知损失
5. 谱归一化：限制判别器梯度

</details>

---

## 14. 学习路径建议

### 第一阶段（1-2天）

1. 理解GAN基础概念
2. 学习DCGAN架构设计
3. 实现基础生成器/判别器

### 第二阶段（2-3天）

1. 完整训练DCGAN
2. 可视化生成结果
3. 理解潜在空间

### 第三阶段（3-5天）

1. 对比不同GAN变体
2. 解决训练问题
3. 实现条件生成

### 推荐资源

- **论文**：《Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks》
- **代码**：PyTorch/examples
- **数据集**：CelebA, LSUN
- **项目**：ProGAN, StyleGAN

---

*DCGAN是深度学习生成模型的重要基石，其架构思想影响了后续众多GAN变体。理解DCGAN对于学习生成模型至关重要。*