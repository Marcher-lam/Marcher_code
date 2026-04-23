# VAE（变分自编码器）学习文档

## 1. 算法基础认知

变分自编码器（Variational Autoencoder, VAE）由 Kingma 和 Welling 于 2013 年提出，是一种**概率生成模型**。与普通 AE 不同，VAE 对隐空间施加概率分布约束（通常为高斯分布），使其具备生成新样本的能力。核心创新是**重参数化技巧（Reparameterization Trick）**和**ELBO 目标函数**。

## 2. 核心原理

VAE 假设数据由隐变量 $z$ 生成，生成过程为：

1. 从先验分布 $p(z)$（标准正态）采样隐变量 $z$
2. 通过生成网络 $p_\theta(x|z)$ 生成数据 $x$

由于真实后验 $p(z|x)$ 不可计算，VAE 用编码器 $q_\phi(z|x)$ 来近似。训练目标是最大化数据的对数似然下界（ELBO）。

**重参数化技巧**：将随机采样从计算图中分离，$z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

## 3. 数学公式与推导

**ELBO 推导**：

$$\log p(x) = \log \int p_\theta(x|z)p(z)dz \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

即：

$$L_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{正则项}}$$

**KL 散度解析解**（当 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$，$p(z) = \mathcal{N}(0, I)$）：

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{J}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**重参数化**：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

## 4. 训练过程讲解

1. 输入 $x$，编码器输出均值 $\mu$ 和方差 $\log\sigma^2$
2. 重参数化采样：$z = \mu + \sigma \odot \epsilon$
3. 解码器生成：$\hat{x} = p_\theta(x|z)$
4. 计算损失：$L = -\text{重建项} + \text{KL项}$
5. 反向传播更新参数
6. 生成时：从 $\mathcal{N}(0,I)$ 采样 $z$，送入解码器

## 5. 应用场景

- **图像生成**：生成人脸、数字等
- **数据增强**：生成新样本扩充训练集
- **表征学习**：学习结构化的隐空间表示
- **广告创意生成**：生成广告图像素材
- **药物分子生成**：在隐空间中搜索新分子结构
- **语音合成**：VAE-based 语音生成

## 6. 优缺点分析

**优点**：
- 隐空间结构化，支持插值和采样生成
- 理论优雅，有明确的概率论基础
- 训练稳定，不存在 GAN 的模式崩塌问题

**缺点**：
- 生成样本偏模糊（因为使用了高斯分布假设和 MSE 重建损失）
- KL 散度可能导致后验坍缩（Posterior Collapse）
- 重建质量和生成质量之间存在 trade-off

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

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        x_recon, mu, logvar = model(batch_x)
        loss = vae_loss(x_recon, batch_x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader.dataset):.4f}")
```

## 8. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class VAEManual(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def generate(self, num_samples):
        z = torch.randn(num_samples, self.decoder.fc1.in_features)
        return self.decoder(z)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

with torch.no_grad():
    z = torch.randn(16, 20)
    samples = model.decoder(z).view(-1, 28, 28)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap='gray')
    ax.axis('off')
plt.suptitle('VAE Generated Samples')
plt.savefig('vae_samples.png')
plt.show()

with torch.no_grad():
    z1 = torch.randn(1, 20)
    z2 = torch.randn(1, 20)
    alphas = torch.linspace(0, 1, 10).unsqueeze(1)
    z_interp = z1 * (1 - alphas) + z2 * alphas
    interp_imgs = model.decoder(z_interp).view(-1, 28, 28)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(interp_imgs[i], cmap='gray')
    ax.axis('off')
plt.suptitle('Latent Space Interpolation')
plt.savefig('vae_interpolation.png')
plt.show()
```

## 10. 模型评估

- **重建损失**：衡量重建质量
- **KL 散度**：衡量后验与先验的距离
- **FID / IS**：评估生成样本质量
- **隐空间可视化**：观察不同类别的聚类结构
- **插值连续性**：隐空间中线性插值是否产生平滑过渡

## 11. 常见问题与易错点

- **后验坍缩（KL Collapse）**：KL 项过大导致隐变量被忽略，所有 $z$ 退化为先验。可用 KL 退火（线性增加 KL 权重）缓解
- **生成模糊**：MSE/BCE 损失导致生成结果趋向均值。可换用更复杂的解码分布
- **重参数化技巧**：忘记此技巧将导致无法对采样操作反向传播
- **$\log\sigma^2$ vs $\sigma$**：网络输出的是 $\log\sigma^2$ 而非 $\sigma$，保证方差为正

## 12. 学习总结

VAE 将概率图模型与神经网络结合，通过变分推断实现端到端训练。它为生成模型提供了坚实的理论基础，是理解扩散模型等前沿工作的重要前置知识。

## 13. 练习题与思考题

**Q1**：为什么需要重参数化技巧？

**A1**：采样操作 $z \sim q(z|x)$ 的梯度无法直接反向传播。重参数化将随机性转移到外部变量 $\epsilon$，使 $z = \mu + \sigma \cdot \epsilon$ 关于 $\mu, \sigma$ 可导。

**Q2**：VAE 与 AE 的核心区别是什么？

**A2**：AE 学到任意形状的隐空间；VAE 约束隐空间服从高斯分布，使其可采样生成。VAE 的损失多了 KL 散度正则项。

**Q3**：什么是后验坍缩？如何解决？

**A3**：当 KL 权重过大时，模型为最小化 KL 项会忽略输入信息，后验退化为先验。解决方法包括 KL 退火、Free Bits、使用更强的解码器等。

## 14. 学习路径建议

1. 掌握 AE 基础 → 2. 学习变分推断理论 → 3. 理解 VAE 的 ELBO 目标 → 4. 实践 VAE 代码 → 5. 了解 VQ-VAE、β-VAE 等改进 → 6. 扩展到扩散模型
