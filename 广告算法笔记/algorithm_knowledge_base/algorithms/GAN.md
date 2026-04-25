# GAN（生成对抗网络）学习文档

## 1. 算法基础认知

生成对抗网络（Generative Adversarial Network, GAN）由 Ian Goodfellow 于 2014 年提出，是深度学习中最具影响力的生成模型之一。GAN 的核心思想是让两个神经网络进行对抗博弈：生成器（Generator）试图生成逼真的假数据来欺骗判别器，判别器（Discriminator）试图区分真实数据和生成数据。通过对抗训练，生成器最终学会生成以假乱真的数据。

## 2. 核心原理

GAN 包含两个网络：

- **生成器 $G$**：输入随机噪声 $z \sim p_z$，输出伪造样本 $G(z)$，目标是让判别器无法区分真假
- **判别器 $D$**：输入样本 $x$，输出 $D(x) \in [0, 1]$ 表示"真实"的概率，目标是正确区分真假

两者构成极小极大博弈（Minimax Game）：$G$ 想最小化 $D$ 的判别能力，$D$ 想最大化判别能力。

## 3. 数学公式与推导

GAN 的目标函数（Minimax 损失）：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**最优判别器**：固定 $G$ 时，最优判别器为：

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**全局最优**：当 $p_g = p_{data}$ 时，$D^*(x) = \frac{1}{2}$，此时 $V(G^*, D^*) = -\log 4$

**推导**：将 $D^*_G$ 代入 $V$，可得：

$$V(G, D^*_G) = -\log 4 + 2 \cdot D_{JS}(p_{data} \| p_g)$$

其中 $D_{JS}$ 是 Jensen-Shannon 散度。因此最小化 $V$ 等价于最小化真实分布与生成分布之间的 JS 散度。

## 4. 训练过程讲解

GAN 的训练采用交替优化策略：

**步骤 1：训练判别器 $D$**
1. 从真实数据采样一批 $x$
2. 从噪声分布采样 $z$，生成假数据 $\hat{x} = G(z)$
3. 更新 $D$ 最大化：$\log D(x) + \log(1 - D(\hat{x}))$

**步骤 2：训练生成器 $G$**
1. 从噪声分布采样 $z$
2. 更新 $G$ 最小化：$\log(1 - D(G(z)))$
3. 实际中常用最大化 $\log D(G(z))$ 代替（提供更强梯度）

## 5. 应用场景

- **图像生成**：生成逼真的人脸、风景、艺术作品
- **图像编辑**：风格迁移、超分辨率、图像修复
- **数据增强**：生成合成数据扩充训练集
- **文本生成**：SeqGAN 等变体
- **广告创意生成**：自动生成广告素材

## 6. 优缺点分析

**优点：**
- 生成样本清晰、逼真
- 不需要显式建模数据分布
- 训练不需要 MCMC 采样

**缺点：**
- 训练不稳定，需要仔细平衡 $G$ 和 $D$
- 模式坍塌（Mode Collapse）：生成器只产生少数几种样本
- 评估困难，没有统一的损失指标

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    latent_dim = 100
    G = Generator(latent_dim)
    D = Discriminator()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(50):
        d_losses, g_losses = [], []
        for real_x, _ in loader:
            real_x = real_x.view(real_x.size(0), -1)
            batch_size = real_x.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            z = torch.randn(batch_size, latent_dim)
            fake_x = G(z)
            d_loss = criterion(D(real_x), real_labels) + criterion(D(fake_x.detach()), fake_labels)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            z = torch.randn(batch_size, latent_dim)
            fake_x = G(z)
            g_loss = criterion(D(fake_x), real_labels)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
        print(f"Epoch {epoch+1}, D Loss: {sum(d_losses)/len(d_losses):.4f}, G Loss: {sum(g_losses)/len(g_losses):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class GANNumpy:
    def __init__(self, data_dim, latent_dim, hidden_dim, lr=0.0002):
        self.lr = lr
        self.latent_dim = latent_dim
        he = lambda n: np.sqrt(2.0 / n)
        self.G_W1 = np.random.randn(latent_dim, hidden_dim) * he(latent_dim)
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, data_dim) * he(hidden_dim)
        self.G_b2 = np.zeros(data_dim)
        self.D_W1 = np.random.randn(data_dim, hidden_dim) * he(data_dim)
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1) * he(hidden_dim)
        self.D_b2 = np.zeros(1)

    def lrelu(self, x, alpha=0.2):
        return np.maximum(alpha * x, x)

    def lrelu_grad(self, x, alpha=0.2):
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def generate(self, z):
        self.G_h = self.lrelu(z @ self.G_W1 + self.G_b1)
        out = np.tanh(self.G_h @ self.G_W2 + self.G_b2)
        return out

    def discriminate(self, x):
        self.D_h = self.lrelu(x @ self.D_W1 + self.D_b1)
        out = self.sigmoid(self.D_h @ self.D_W2 + self.D_b2)
        return out

    def train_discriminator(self, real_x, fake_x):
        d_real = self.discriminate(real_x)
        d_fake = self.discriminate(fake_x)
        loss = -np.mean(np.log(d_real + 1e-8) + np.log(1 - d_fake + 1e-8))
        m = real_x.shape[0]
        d_out_real = d_real * (1 - d_real)
        d_out_fake = d_fake * (1 - d_fake)
        dW2 = (self.D_h[:m].T @ (d_real - 1) + self.D_h[m:].T @ d_fake) / m if False else \
              (self.D_h.T @ np.vstack([(d_real - 1), d_fake])) / (2 * m)
        db2 = np.mean(np.vstack([d_real - 1, d_fake]), axis=0)
        self.D_W1 -= self.lr * (real_x.T @ (d_out_real * (d_real - 1)) + fake_x.T @ (d_out_fake * d_fake)).dot(self.D_W2.T * self.lrelu_grad(self.D_h)) / (2 * m)
        self.D_W2 -= self.lr * dW2
        self.D_b2 -= self.lr * db2
        return loss

    def train_generator(self, batch_size):
        z = self.sample_latent(batch_size)
        fake_x = self.generate(z)
        d_fake = self.discriminate(fake_x)
        loss = -np.mean(np.log(d_fake + 1e-8))
        m = batch_size
        d_out = d_fake * (1 - d_fake)
        grad_W2 = self.G_h.T @ (d_out * (1.0 / (d_fake + 1e-8))) @ np.ones((1, 1)) * self.D_W2.T
        dW2 = (self.G_h.T @ (d_fake - 1)) / m
        db2 = np.mean(d_fake - 1, axis=0)
        grad_from_D = ((d_fake - 1) / m) @ self.D_W2.T * self.lrelu_grad(self.G_h)
        dW1 = z.T @ grad_from_D
        db1 = np.mean(grad_from_D, axis=0)
        self.G_W1 -= self.lr * dW1
        self.G_b1 -= self.lr * db1
        self.G_W2 -= self.lr * dW2
        self.G_b2 -= self.lr * db2
        return loss

    def train(self, real_data, epochs=100, batch_size=32):
        n = real_data.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            d_losses, g_losses = [], []
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                real_x = real_data[batch_idx]
                z = self.sample_latent(real_x.shape[0])
                fake_x = self.generate(z)
                d_loss = self.train_discriminator(real_x, fake_x)
                g_loss = self.train_generator(real_x.shape[0])
                d_losses.append(d_loss)
                g_losses.append(g_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")

    def sample_latent(self, batch_size):
        return np.random.randn(batch_size, self.latent_dim)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

G.eval()
with torch.no_grad():
    z = torch.randn(25, latent_dim)
    samples = G(z).view(-1, 28, 28)
    samples = samples * 0.5 + 0.5

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(samples[i * 5 + j], cmap='gray')
        axes[i, j].axis('off')
plt.suptitle('GAN 生成的手写数字')
plt.savefig('gan_samples.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **IS（Inception Score）**：衡量生成图像的质量和多样性，越高越好
- **FID（Fréchet Inception Distance）**：衡量生成分布与真实分布的距离，越低越好
- **视觉检查**：观察生成样本的清晰度和多样性
- **训练曲线**：监控 $D$ 和 $G$ 的损失变化，理想情况下应保持动态平衡

## 11. 常见问题与易错点

- **模式坍塌**：生成器只产生有限种类样本。解决方案：WGAN、Mini-batch discrimination、unrolled GAN
- **训练不稳定**：$D$ 太强导致 $G$ 梯度消失。解决方案：降低 $D$ 学习率、标签平滑、使用 WGAN-GP
- **梯度消失**：当 $D$ 太强时，$\log(1-D(G(z)))$ 的梯度趋近于零。用 $\log D(G(z))$ 代替
- **学习率设置**：GAN 对学习率敏感，推荐使用 Adam 且 $\beta_1 = 0.5$

## 12. 学习总结

GAN 开创了对抗训练的范式，深刻影响了整个生成模型领域。核心是 $G$ 和 $D$ 的博弈均衡。关键挑战在于训练稳定性。后续的 WGAN、DCGAN、StyleGAN 等变体都在不同方面改进了原始 GAN。

## 13. 练习题与思考题（含答案）

**Q1：为什么实际训练中用 $\log D(G(z))$ 代替 $\log(1-D(G(z)))$？**

A1：在训练初期 $G$ 较弱，$D$ 轻易识别假样本，$D(G(z)) \approx 0$，$\log(1-D(G(z)))$ 的梯度非常小。用 $\log D(G(z))$ 在 $D(G(z)) \approx 0$ 时梯度更大，提供更强的学习信号。

**Q2：什么是模式坍塌？**

A2：生成器找到几种能骗过判别器的样本后就反复输出这几种，忽略了数据分布中的其他模式。结果看起来像噪声或只覆盖了部分数据多样性。

**Q3：GAN 的训练目标从信息论角度是什么？**

A3：最小化生成分布 $p_g$ 和真实分布 $p_{data}$ 之间的 Jensen-Shannon 散度。当且仅当 $p_g = p_{data}$ 时达到全局最优。

## 14. 学习路径建议

1. 理解 GAN 的 minimax 博弈框架
2. 实现基础 GAN 并理解训练不稳定性
3. 学习 DCGAN（卷积结构）、WGAN（Wasserstein 距离）
4. 进阶：StyleGAN、CycleGAN、Pix2Pix
5. 了解 GAN 与 VAE、扩散模型的对比
