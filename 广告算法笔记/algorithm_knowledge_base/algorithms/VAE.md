# VAE（变分自编码器）学习文档

## 1. 算法基础认知

变分自编码器（Variational Autoencoder, VAE）由 Kingma 和 Welling 于 2013 年提出，是一种结合深度学习和变分推断的生成模型。与普通 AE 只学习确定性编码不同，VAE 让编码器输出一个概率分布（均值和方差），再从该分布中采样得到隐向量，从而使隐空间具有连续性和结构化特性，支持从隐空间采样生成新样本。

## 2. 核心原理

VAE 的核心思想是：假设数据 $x$ 由隐变量 $z$ 生成，我们无法直接观测 $z$，但可以通过变分推断来近似后验分布 $p(z|x)$。

**编码器**：$q_\phi(z|x)$ 近似后验分布，输出均值 $\mu$ 和方差 $\sigma^2$
**采样**：从 $\mathcal{N}(\mu, \sigma^2)$ 中采样 $z$
**解码器**：$p_\theta(x|z)$ 从 $z$ 重构 $x$

关键技术——**重参数化技巧**：将随机采样操作从计算图中分离，使得梯度可以正常反向传播：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

## 3. 数学公式与推导

VAE 最大化数据的对数似然的变分下界（ELBO）：

$$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

因此损失函数为 ELBO 的负数：

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{重构损失}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{正则项}}$$

KL 散度有解析解（假设 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$，$p(z) = \mathcal{N}(0, I)$）：

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

推导过程：将两个高斯分布的 KL 散度公式代入展开即可得到上式。KL 项迫使编码分布接近标准正态，保证隐空间的连续性和完备性。

## 4. 训练过程讲解

1. 输入 $x$ 通过编码器，输出均值 $\mu$ 和对数方差 $\log\sigma^2$
2. 重参数化采样：$z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$
3. $z$ 通过解码器，输出重构 $\hat{x}$
4. 计算损失 = 重构损失 + KL 散度
5. 反向传播更新参数

训练完成后，直接从 $\mathcal{N}(0, I)$ 采样 $z$，送入解码器即可生成新样本。

## 5. 应用场景

- **图像生成**：生成人脸、数字等
- **数据增强**：生成合成样本扩充训练集
- **表示学习**：学习结构化的隐空间表示
- **药物分子生成**：在化学空间中生成新分子
- **语音合成**：如 VAE-Voice

## 6. 优缺点分析

**优点：**
- 隐空间连续且有结构，支持插值生成
- 有概率论基础，训练稳定
- 可同时用于生成和推断

**缺点：**
- 生成样本通常比 GAN 模糊
- KL 散度可能导致后验坍塌（KL vanishing）
- 需要平衡重构损失和 KL 项

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        x_recon, mu, logvar = model(batch_x)
        loss = vae_loss(x_recon, batch_x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class VAENumpy:
    def __init__(self, input_dim, hidden_dim, latent_dim, lr=0.001):
        self.lr = lr
        self.latent_dim = latent_dim
        sc = lambda n_in, n_out: np.sqrt(2.0 / n_in)
        self.W_enc = np.random.randn(input_dim, hidden_dim) * sc(input_dim, hidden_dim)
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * sc(hidden_dim, latent_dim)
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * sc(hidden_dim, latent_dim)
        self.b_logvar = np.zeros(latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * sc(latent_dim, hidden_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * sc(hidden_dim, input_dim)
        self.b_dec2 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        self.x = x
        self.h_enc = self.relu(x @ self.W_enc + self.b_enc)
        self.mu = self.h_enc @ self.W_mu + self.b_mu
        self.logvar = self.h_enc @ self.W_logvar + self.b_logvar
        self.std = np.exp(0.5 * self.logvar)
        self.eps = np.random.randn(*self.mu.shape)
        self.z = self.mu + self.std * self.eps
        self.h_dec = self.relu(self.z @ self.W_dec1 + self.b_dec1)
        self.out = self.sigmoid(self.h_dec @ self.W_dec2 + self.b_dec2)
        return self.out

    def compute_loss(self):
        recon = -np.sum(self.x * np.log(self.out + 1e-8) + (1 - self.x) * np.log(1 - self.out + 1e-8))
        kl = -0.5 * np.sum(1 + self.logvar - self.mu ** 2 - np.exp(self.logvar))
        return recon + kl

    def generate(self, n_samples=5):
        z = np.random.randn(n_samples, self.latent_dim)
        h = self.relu(z @ self.W_dec1 + self.b_dec1)
        return self.sigmoid(h @ self.W_dec2 + self.b_dec2)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    z_samples = torch.randn(10, 20)
    generated = model.decode(z_samples)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(generated[i].view(28, 28), cmap='gray')
    axes[i].axis('off')
plt.suptitle('VAE 生成的数字')
plt.savefig('vae_generated.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **重构质量**：MSE 或 BCE 越低越好
- **生成质量**：FID（Fréchet Inception Distance）越低越好
- **隐空间结构**：可视化 latent space 的 2D 投影，检查不同类别是否平滑过渡
- **KL 散度值**：监控 KL 项是否趋近合理值（过低说明后验坍塌）

## 11. 常见问题与易错点

- **后验坍塌（KL Vanishing）**：KL 项变为 0，编码器忽略输入，解码器只依赖先验。解决方案：KL 退火（逐步增大 KL 权重）或使用 Free Bits
- **生成模糊**：VAE 使用像素级 MSE/BCE 损失，倾向于输出均值，导致模糊
- **$\beta$-VAE**：引入权重 $\beta$ 控制 KL 项强度，$\beta > 1$ 鼓励更解耦的表示

## 12. 学习总结

VAE 是将概率图模型与深度学习结合的经典之作。核心贡献包括：变分下界（ELBO）作为训练目标、重参数化技巧使梯度可传播、KL 散度约束隐空间结构。理解 VAE 对掌握现代生成模型至关重要。

## 13. 练习题与思考题（含答案）

**Q1：为什么需要重参数化技巧？**

A1：直接从 $q_\phi(z|x)$ 采样的操作不可微，无法反向传播梯度。重参数化将随机性转移到外部变量 $\epsilon$，使 $z = \mu + \sigma \cdot \epsilon$ 对 $\mu, \sigma$ 可微。

**Q2：VAE 和 AE 的本质区别是什么？**

A2：AE 学确定性映射，VAE 学概率分布；AE 隐空间不连续无法生成，VAE 隐空间受正态先验约束，可以采样生成。

**Q3：KL 散度项在 VAE 中起什么作用？**

A3：KL 项迫使编码分布 $q_\phi(z|x)$ 接近标准正态 $p(z)$，保证隐空间连续无空洞，使得从任意位置采样都能生成有意义的样本。

## 14. 学习路径建议

1. 先掌握 AE 和基础概率论（贝叶斯定理、高斯分布）
2. 理解变分推断的核心思想
3. 学习 VAE 的 ELBO 推导和重参数化技巧
4. 进阶：$\beta$-VAE、VQ-VAE、NVAE 等变体
5. 对比学习 GAN 和扩散模型
