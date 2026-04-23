# GAN（生成对抗网络）学习文档

## 1. 算法基础认知

生成对抗网络（Generative Adversarial Network, GAN）由 Ian Goodfellow 于 2014 年提出，是深度学习中最具影响力的生成模型之一。GAN 包含两个相互博弈的网络：**生成器（Generator）** 尽可能生成逼真数据来欺骗判别器，**判别器（Discriminator）** 尽可能区分真实数据和生成数据。通过对抗训练，生成器最终学会生成逼真样本。

## 2. 核心原理

GAN 的核心是一个**极小极大博弈（Minimax Game）**：

- **生成器 $G(z)$**：从随机噪声 $z$（通常为标准正态分布）生成假数据
- **判别器 $D(x)$**：输出输入 $x$ 为真实数据的概率

训练过程中，$G$ 和 $D$ 交替更新：
- 固定 $G$，训练 $D$ 更好地区分真假
- 固定 $D$，训练 $G$ 更好地欺骗 $D$

理论最优状态下，生成分布与真实分布完全一致，$D(x) = 0.5$。

## 3. 数学公式与推导

**GAN 的目标函数**：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**最优判别器推导**：

对固定 $G$，最优化 $D$：

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

将 $D^*_G$ 代回目标函数：

$$C(G) = \max_D V(D, G) = -\log 4 + 2 \cdot D_{JS}(p_{data} \| p_g)$$

其中 $D_{JS}$ 为 Jensen-Shannon 散度。当且仅当 $p_g = p_{data}$ 时，$C(G)$ 取到全局最小值 $-\log 4$。

**实际训练的损失函数**（非饱和生成器损失）：

$$L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

$$L_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

## 4. 训练过程讲解

1. 从噪声分布 $p_z$ 采样一批 $z$
2. 生成假数据 $\hat{x} = G(z)$
3. **训练判别器**：用真实数据 $x$（标签 1）和假数据 $\hat{x}$（标签 0）训练 $D$，最大化正确分类概率
4. 从噪声分布重新采样 $z$
5. **训练生成器**：生成假数据 $\hat{x} = G(z)$，以标签 1 欺骗 $D$，最小化 $D$ 正确识别的概率
6. 交替重复以上步骤

关键：判别器训练步数通常多于生成器（如 5:1 比例），保持 $D$ 有足够的判别能力作为 $G$ 的"老师"。

## 5. 应用场景

- **图像生成**：生成逼真的人脸、场景、艺术品
- **图像转换**：风格迁移、超分辨率（SRGAN）
- **数据增强**：生成训练样本
- **广告创意**：自动生成广告图像
- **图像修复**：补全缺失区域
- **视频生成**：预测未来帧

## 6. 优缺点分析

**优点**：
- 生成样本清晰锐利，视觉质量高
- 不需要显式建模数据分布
- 训练不需要 MCMC 采样

**缺点**：
- 训练不稳定，$G$ 和 $D$ 需要精细平衡
- 模式崩塌（Mode Collapse）：生成器只生成少数几种样本
- 梯度消失：判别器太强时生成器梯度消失
- 评估困难：没有显式的似然函数

## 7. 调库实现（Python）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim=64, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(50):
    for real_x, _ in loader:
        batch_size = real_x.size(0)
        real_x = real_x.view(batch_size, -1) * 2 - 1
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        z = torch.randn(batch_size, 64)
        fake_x = G(z)

        d_loss = criterion(D(real_x), real_labels) + criterion(D(fake_x.detach()), fake_labels)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        z = torch.randn(batch_size, 64)
        fake_x = G(z)
        g_loss = criterion(D(fake_x), real_labels)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
```

## 8. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.tanh(self.fc3(x))

class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.sigmoid(self.fc3(x))

class GANManual:
    def __init__(self, latent_dim=64, data_dim=784, hidden_dim=256):
        self.G = SimpleGenerator(latent_dim, hidden_dim, data_dim)
        self.D = SimpleDiscriminator(data_dim, hidden_dim)
        self.latent_dim = latent_dim

    def train_step(self, real_data, opt_G, opt_D):
        batch_size = real_data.size(0)
        ones = torch.ones(batch_size, 1)
        zeros = torch.zeros(batch_size, 1)

        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.G(z).detach()
        d_real = self.D(real_data)
        d_fake = self.D(fake_data)
        d_loss = -(torch.log(d_real + 1e-8).mean() + torch.log(1 - d_fake + 1e-8).mean())
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.G(z)
        g_loss = -torch.log(self.D(fake_data) + 1e-8).mean()
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        return d_loss.item(), g_loss.item()
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

G.eval()
with torch.no_grad():
    z = torch.randn(16, 64)
    samples = G(z).view(-1, 28, 28)
    samples = (samples + 1) / 2

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap='gray')
    ax.axis('off')
plt.suptitle('GAN Generated Samples')
plt.savefig('gan_samples.png')
plt.show()
```

## 10. 模型评估

- **IS（Inception Score）**：评估生成质量和多样性
- **FID（Fréchet Inception Distance）**：比较生成分布与真实分布的特征距离，越低越好
- **可视化检查**：人眼判断生成质量
- **模式覆盖**：检查是否覆盖了数据的所有模式

## 11. 常见问题与易错点

- **模式崩塌**：生成器只产生少数几种输出。可使用 WGAN、SNGAN 等稳定训练的变体
- **训练不稳定**：$D$ 和 $G$ 能力不匹配导致震荡。使用 Adam 优化器，$\beta_1=0.5$ 有助于稳定
- **梯度消失**：原始 GAN 的 $\log(1-D(G(z)))$ 在训练早期梯度很小，因此使用非饱和损失
- **标签平滑**：真实标签用 0.9 而非 1.0 可提高稳定性

## 12. 学习总结

GAN 开创了对抗训练的生成模型范式，通过博弈论框架让生成器和判别器共同进步。虽然训练不稳定，但其生成质量远超同期方法，催生了 DCGAN、WGAN、StyleGAN 等大量改进。

## 13. 练习题与思考题

**Q1**：为什么原始 GAN 的生成器损失 $\log(1-D(G(z)))$ 在训练早期会有梯度消失问题？

**A1**：训练早期 $G$ 生成的数据质量差，$D$ 轻易区分真假，$D(G(z))$ 接近 0，此时 $\log(1-D(G(z)))$ 的梯度非常小，导致 $G$ 几乎无法学习。改用 $-\log D(G(z))$ 可在 $D(G(z))$ 接近 0 时提供更大梯度。

**Q2**：什么是模式崩塌？如何缓解？

**A2**：生成器找到几种能欺骗判别器的样本后反复生成，忽略了数据的其他模式。WGAN、SNGAN、Mini-batch Discrimination、Unrolled GAN 等方法可缓解。

**Q3**：GAN 为什么不用 BCE 损失直接训练生成器？

**A3**：BCE 本身就是 GAN 的损失函数。关键是训练策略：不能同时更新 $G$ 和 $D$，需要交替训练并保持两者能力的动态平衡。

## 14. 学习路径建议

1. 理解对抗训练思想 → 2. 实现基础 GAN → 3. 学习 DCGAN（引入卷积） → 4. 了解 WGAN（Wasserstein 距离） → 5. 学习条件 GAN（CGAN） → 6. 探索 StyleGAN、CycleGAN 等前沿工作
