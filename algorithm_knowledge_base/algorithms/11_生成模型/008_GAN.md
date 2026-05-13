# 生成对抗网络 (GAN) 学习文档

> 生成器与判别器的对抗博弈，开创深度生成模型新纪元。

> 来源线索：本节内容根据原书中关于"生成对抗网络"的相关章节（第4章4.1-4.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** GAN 通过生成器和判别器的对抗训练，使生成器学会生成逼真的数据。

**直觉类比：** 想象一个伪造名画的画家（生成器）和一个鉴定师（判别器）在持续博弈。画家不断尝试画出更逼真的假画，鉴定师不断提高鉴别能力。经过无数次交锋，画家的仿造技术达到了以假乱真的水平——这就是 GAN 的核心思想。

**历史背景：** GAN 由 Ian Goodfellow 于 2014 年提出，是生成模型领域的里程碑。论文 "Generative Adversarial Nets" 提出了对抗训练的极简框架，却催生了后续无数变体（DCGAN、WGAN、StyleGAN 等），广泛应用于图像生成、风格迁移、数据增强等领域。

**算法定位：** 无监督/生成模型，通过对抗训练学习数据分布，用于生成新样本。

**前置知识：** 深度学习基础、概率分布、损失函数、梯度下降、PyTorch。

---

## 2. 核心原理

### 核心思想

GAN 的核心是两个网络的**极小极大博弈（Minimax Game）**：
- **生成器 G**：接收随机噪声 z，尝试生成与真实数据无异的假数据 $G(z)$
- **判别器 D**：接收数据样本，判断它是真实的还是生成的

生成器的目标是"骗过"判别器，判别器的目标是"不被骗"。当两者达到纳什均衡时，生成器学到了真实数据的分布。

### 工作流程

1. 从先验分布 $p_z$ 中采样随机噪声 $z$
2. 生成器将 $z$ 映射为假样本 $G(z)$
3. 判别器分别接收真实样本 $x$ 和假样本 $G(z)$
4. 判别器输出每个样本为"真实"的概率
5. 分别更新判别器（提高鉴别力）和生成器（提高欺骗力）
6. 重复步骤 1-5 直到达到均衡

### 关键概念

- **极小极大博弈**：生成器最小化目标函数（让判别器犯错），判别器最大化目标函数（正确区分真假）
- **模式崩塌（Mode Collapse）**：生成器只学会生成少数几种样本，丧失多样性
- **训练不稳定性**：G 和 D 的训练需要保持平衡，一方过强会导致另一方无法学习
- **JS 散度**：原始 GAN 隐式地最小化真实分布与生成分布之间的 Jensen-Shannon 散度

### 几何/直观解释

```
GAN 的对抗训练过程：

随机噪声 z → [生成器 G] → 假样本 G(z) →┐
                                          ├→ [判别器 D] → 真/假概率
真实数据 x ─────────────────────────────→┘

训练目标：
  判别器 D: 最大化 log(D(x)) + log(1 - D(G(z)))   ← 学会区分真假
  生成器 G: 最小化 log(1 - D(G(z)))                 ← 学会骗过 D

训练循环：
  ┌─────────────────────────────────────────┐
  │ 1. 冻结 G，训练 D（提高鉴别力）           │
  │ 2. 冻结 D，训练 G（提高欺骗力）           │
  │ 3. 重复 N 轮                             │
  │ 结果: G 生成的样本越来越逼真              │
  └─────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $p_{data}$ | 真实数据分布 |
| $p_z$ | 噪声先验分布（通常为标准正态） |
| $G(z;\theta_g)$ | 生成器网络，参数 $\theta_g$ |
| $D(x;\theta_d)$ | 判别器网络，参数 $\theta_d$ |
| $x$ | 真实数据样本 |
| $z$ | 随机噪声向量 |

### GAN 的目标函数

GAN 的训练是一个极小极大优化问题：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**推导过程：**

**步骤1：最优判别器**

对于固定的生成器 $G$，最优判别器 $D^*_G$ 为：

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

这个结果由对 $V$ 关于 $D$ 求导并令其为 0 得到。直观理解：最优判别器在真实数据密度高的地方输出接近 1，在生成数据密度高的地方输出接近 0。

**步骤2：将最优判别器代入目标函数**

将 $D^*_G$ 代入 $V(D,G)$，经过推导可得：

$$\min_G V(D^*_G, G) = -\log 4 + 2 \cdot JSD(p_{data} \| p_g)$$

其中 $JSD$ 是 Jensen-Shannon 散度。这意味着训练生成器等价于最小化真实分布与生成分布之间的 JS 散度。

**步骤3：全局最优**

当且仅当 $p_g = p_{data}$ 时，目标函数达到全局最小值 $-\log 4$。此时 $D^*(x) = 1/2$，即判别器无法区分真假样本。

### 训练算法

每个训练迭代：
1. **更新判别器**（重复 $k$ 步）：
   - 采样真实数据 mini-batch $\{x^{(1)}, \ldots, x^{(m)}\}$
   - 采样噪声 mini-batch $\{z^{(1)}, \ldots, z^{(m)}\}$
   - 更新 $\theta_d$：$\theta_d \leftarrow \theta_d + \eta \nabla_{\theta_d} \frac{1}{m}\sum_{i=1}^m [\log D(x^{(i)}) + \log(1-D(G(z^{(i)})))]$

2. **更新生成器**（1 步）：
   - 采样噪声 mini-batch $\{z^{(1)}, \ldots, z^{(m)}\}$
   - 更新 $\theta_g$：$\theta_g \leftarrow \theta_g - \eta \nabla_{\theta_g} \frac{1}{m}\sum_{i=1}^m \log(1-D(G(z^{(i)})))$

### 非饱和生成器损失

实践中，生成器使用 $\max_G \log D(G(z))$ 替代 $\min_G \log(1-D(G(z)))$，因为后者在训练初期（D 很容易区分真假时）梯度接近 0，导致 G 无法学习。这被称为非饱和启发式（Non-saturating heuristic）。

---

## 4. 训练过程讲解

### 数据预处理

- 图像像素值归一化到 [-1, 1]（使用 Tanh 作为生成器输出激活函数）
- 不需要标签（无监督训练）

### 参数初始化

- 生成器和判别器均使用正态分布随机初始化（均值 0，标准差 0.02）
- 偏置初始化为 0

### 迭代过程

1. 每轮先训练判别器 k 次（通常 k=1）
2. 再训练生成器 1 次
3. 监控 D 的损失和 G 的损失，两者应交替下降
4. 可视化 G 生成的图像，直观判断质量

### 收敛判断

- 生成样本质量不再改善
- 判别器损失接近 $\log 0.5 = -0.693$（无法区分真假）
- 使用 FID（Frechet Inception Distance）等指标量化评估

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| lr_g | 生成器学习率 | 1e-5 ~ 2e-4 | 2e-4 |
| lr_d | 判别器学习率 | 1e-5 ~ 2e-4 | 2e-4 |
| beta1 | Adam 一阶矩系数 | 0.0 ~ 0.5 | 0.5 |
| latent_dim | 噪声维度 | 32 ~ 256 | 100 |
| batch_size | 批大小 | 32 ~ 128 | 64 |

---

## 5. 应用场景

### 1. 图像生成
从随机噪声生成逼真的图像（人脸、风景、艺术品等）。GAN 能学习到训练数据的分布并生成相似但不重复的新图像。StyleGAN 系列在此领域取得了突破性成果。

### 2. 图像超分辨率
将低分辨率图像转换为高分辨率图像（SRGAN）。GAN 的对抗损失使生成的图像比纯 MSE 损失更清晰、更具真实感。

### 3. 图像修复
修复图像中的缺失或损坏区域。GAN 利用上下文信息生成合理的填充内容。

### 4. 数据增强
为分类或检测任务生成额外的训练数据，尤其在医学影像等数据稀缺领域特别有价值。

### 不适用场景
- 需要精确控制生成内容的场景（原始 GAN 缺乏控制机制）
- 离散数据生成（文本、图结构等），GAN 的连续梯度信号难以直接应用于离散空间

---

## 6. 优缺点分析

### 优点

1. **生成质量高**：GAN 生成的图像通常比 VAE 等其他生成模型更清晰、更逼真，因为对抗训练直接优化"逼真度"。
2. **无需显式密度建模**：不需要假设数据的概率分布形式，避免了复杂的似然计算。
3. **训练框架灵活**：G 和 D 可以用任何网络架构，对抗训练的思想可以嵌入各种模型。

### 缺点

1. **训练不稳定**：G 和 D 的训练需要精心平衡。D 太强则 G 梯度消失，G 太强则 D 失去指导作用。缓解思路：使用 WGAN 的 Wasserstein 损失、谱归一化、梯度惩罚。
2. **模式崩塌（Mode Collapse）**：生成器可能只学会生成少数几种样本。缓解思路：使用 Mini-batch Discrimination、Unrolled GAN、或多样性正则化。
3. **评估困难**：没有直接的似然指标，需要 FID、IS 等间接评估方法。
4. **超参数敏感**：学习率、网络架构、训练轮数等对结果影响巨大。

### 与同类算法对比

| 特性 | GAN | VAE | 扩散模型 |
|------|-----|-----|---------|
| 生成质量 | 高（锐利） | 中（略模糊） | 高（锐利） |
| 训练稳定性 | 不稳定 | 稳定 | 稳定 |
| 多样性 | 中（模式崩塌风险） | 高 | 高 |
| 采样速度 | 快 | 快 | 慢（多步去噪） |
| 理论基础 | 博弈论 | 变分推断 | 扩散过程 |
| 可控性 | 低（原始GAN） | 中 | 高 |

---

## 7. 调库实现

使用 PyTorch 实现一个 DCGAN 生成 MNIST 手写数字：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

# 超参数
latent_dim = 100
lr = 0.0002
beta1 = 0.5
batch_size = 64
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 生成器：将噪声映射为图像
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),  # → 256×7×7
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # → 128×14×14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),  # → 1×28×28
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.main(z.view(-1, latent_dim, 1, 1))

# 判别器：判断图像真假
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # → 64×14×14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # → 128×7×7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )

    def forward(self, x):
        return self.main(x)

# 初始化模型和优化器
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练循环
fixed_noise = torch.randn(64, latent_dim, device=device)  # 固定噪声用于可视化
for epoch in range(epochs):
    d_loss_total = 0
    g_loss_total = 0
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size_curr = real_images.size(0)

        # 真实标签=1，假标签=0
        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # === 训练判别器 ===
        optimizer_D.zero_grad()
        # 真实样本的损失
        d_real = D(real_images)
        loss_real = criterion(d_real, real_labels)
        # 生成假样本并计算损失
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_images = G(z).detach()  # detach 防止梯度传到 G
        d_fake = D(fake_images)
        loss_fake = criterion(d_fake, fake_labels)
        # 判别器总损失
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_D.step()

        # === 训练生成器 ===
        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_images = G(z)
        d_fake = D(fake_images)
        # 生成器希望判别器将假样本判断为真（非饱和损失）
        g_loss = criterion(d_fake, real_labels)
        g_loss.backward()
        optimizer_G.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    print(f'Epoch [{epoch+1}/{epochs}] D_loss: {d_loss_total/len(dataloader):.4f} '
          f'G_loss: {g_loss_total/len(dataloader):.4f}')

# 生成一些图像用于可视化
with torch.no_grad():
    fake = G(fixed_noise).cpu()
grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
print("生成完成！可以用 matplotlib 可视化 grid 张量")
```

---

## 8. 手工代码实现

使用 PyTorch tensor 操作从零实现 GAN 核心训练逻辑：

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleGAN:
    """手工实现的简易 GAN（全连接网络）"""

    def __init__(self, data_dim, latent_dim=32, hidden_dim=128, lr=0.0002):
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 生成器：噪声 → 真实数据空间
        self.G = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()  # 输出 [-1, 1]
        ).to(self.device)

        # 判别器：数据 → 真假概率
        self.D = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        ).to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train_step(self, real_data):
        """单步训练：返回 D_loss 和 G_loss"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # --- 训练判别器 ---
        self.opt_D.zero_grad()
        # 真实样本的判别
        d_real = self.D(real_data)
        loss_real = self.criterion(d_real, real_labels)
        # 生成假样本
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.G(z).detach()
        d_fake = self.D(fake_data)
        loss_fake = self.criterion(d_fake, fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        self.opt_D.step()

        # --- 训练生成器 ---
        self.opt_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.G(z)
        d_fake = self.D(fake_data)
        g_loss = self.criterion(d_fake, real_labels)  # 希望被判为真
        g_loss.backward()
        self.opt_G.step()

        return d_loss.item(), g_loss.item()

    def generate(self, n_samples):
        """生成新样本"""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.G(z).cpu().numpy()
        self.G.train()
        return samples


# 测试：学习生成二维高斯分布
if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # 目标分布：环形分布
    angles = np.random.uniform(0, 2*np.pi, 5000)
    radius = 2.0 + np.random.randn(5000) * 0.3
    real_data = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    real_tensor = torch.FloatTensor(real_data)

    # 训练 GAN
    gan = SimpleGAN(data_dim=2, latent_dim=16, hidden_dim=64)
    for epoch in range(200):
        indices = np.random.choice(len(real_data), 64)
        batch = real_tensor[indices]
        d_loss, g_loss = gan.train_step(batch)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}')

    # 生成样本
    fake_samples = gan.generate(1000)
    print(f"生成样本均值: ({fake_samples[:,0].mean():.2f}, {fake_samples[:,1].mean():.2f})")
    print(f"生成样本标准差: ({fake_samples[:,0].std():.2f}, {fake_samples[:,1].std():.2f})")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_gan_training():
    """可视化 GAN 训练过程中生成分布的演变"""
    # 模拟不同训练阶段的生成分布
    np.random.seed(42)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 真实分布（环形）
    angles = np.random.uniform(0, 2*np.pi, 500)
    radius = 2.0 + np.random.randn(500) * 0.3
    real_x = radius * np.cos(angles)
    real_y = radius * np.sin(angles)

    stages = [
        ('初始阶段', np.random.randn(500, 2) * 0.5),
        ('训练中期', np.column_stack([
            np.random.randn(500) * 1.5,
            np.random.randn(500) * 1.5
        ])),
        ('训练后期', np.column_stack([
            (2 + np.random.randn(500)*0.5) * np.cos(np.random.uniform(0, 2*np.pi, 500)),
            (2 + np.random.randn(500)*0.5) * np.sin(np.random.uniform(0, 2*np.pi, 500))
        ])),
    ]

    for idx, (title, fake) in enumerate(stages):
        ax = axes[idx]
        ax.scatter(real_x, real_y, alpha=0.3, s=5, c='blue', label='真实数据')
        ax.scatter(fake[:, 0], fake[:, 1], alpha=0.3, s=5, c='red', label='生成数据')
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')

    # 训练损失曲线
    axes[3].plot(np.arange(100), np.exp(-np.arange(100)/20) + np.random.randn(100)*0.1 + 0.5,
                 label='D loss', color='blue')
    axes[3].plot(np.arange(100), np.exp(-np.arange(100)/25) + np.random.randn(100)*0.2 + 0.8,
                 label='G loss', color='red')
    axes[3].set_title('训练损失曲线', fontsize=12)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('gan_training.png', dpi=100, bbox_inches='tight')
    plt.show()

visualize_gan_training()
```

**结果解读：**
- **初始阶段**：生成器产生的数据集中在原点附近，与环形分布差距很大
- **训练中期**：生成分布开始扩展，但形态与真实分布仍有明显差异
- **训练后期**：生成分布趋近环形，与真实分布高度重合
- **损失曲线**：D loss 和 G loss 交替下降，最终趋于稳定

---

## 10. 模型评估

### 评估指标

GAN 没有直接的似然评估，常用以下指标：

1. **FID（Frechet Inception Distance）**：计算真实图像和生成图像在 Inception 网络特征空间中的分布距离，越低越好
2. **IS（Inception Score）**：衡量生成图像的清晰度和多样性，越高越好
3. **可视化评估**：直观查看生成图像的质量和多样性

```python
def compute_fid_simple(real_features, fake_features):
    """简化版 FID 计算"""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid
```

---

## 11. 常见问题与易错点

### 数据层面

1. **图像归一化范围错误**
   - 现象：生成器输出全为同一颜色，或训练不收敛
   - 原因：Tanh 输出 [-1,1] 但数据归一化到 [0,1]
   - 解决：确保数据归一化到 [-1,1] 与 Tanh 匹配

2. **忘记 detach 假样本**
   - 现象：训练判别器时生成器参数也被更新
   - 原因：`G(z)` 的计算图连接到 G 的参数
   - 解决：训练 D 时使用 `G(z).detach()`

### 模型层面

1. **模式崩塌**
   - 现象：生成器只生成几乎相同的图像
   - 原因：G 发现了能骗过 D 的少数模式就停止探索
   - 解决：使用 WGAN、增加 Mini-batch Discrimination、降低学习率

2. **梯度消失/爆炸**
   - 现象：G 或 D 的损失变为 0 或 NaN
   - 原因：D 过强导致 G 梯度为 0
   - 解决：降低 D 的训练频率、使用谱归一化、使用标签平滑

### 调参层面

1. **Adam 的 beta1 参数设置**
   - 默认 0.9 对 GAN 训练太大，导致动量震荡
   - 解决：GAN 中通常设 beta1=0.5

---

## 12. 学习总结

### 核心思想回顾

GAN 的核心是生成器与判别器之间的对抗博弈。生成器尝试生成逼真的假样本来欺骗判别器，判别器尝试正确区分真假样本。通过交替训练，双方都不断进步，最终生成器学会了真实数据分布。原始 GAN 存在训练不稳定和模式崩塌的问题，催生了 WGAN、StyleGAN 等改进版本。

### 关键公式

1. GAN 目标函数：$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$
2. 最优判别器：$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$
3. 训练等价于最小化 JS 散度：$2 \cdot JSD(p_{data} \| p_g) - \log 4$

### 与相关算法的联系

- **VAE**：另一种生成模型，使用变分推断，训练稳定但生成质量略低
- **WGAN**：使用 Wasserstein 距离替代 JS 散度，解决训练不稳定问题
- **扩散模型**：通过逐步去噪生成，质量更高但速度更慢

### 后续学习方向

- DCGAN、WGAN-GP、StyleGAN 等改进变体
- 条件 GAN（CGAN）：引入条件信息控制生成
- 扩散模型：GAN 的有力竞争者

---

## 13. 练习题与思考题

### 基础题

**题1：** 为什么训练判别器时需要 detach 生成器的输出？如果不 detach 会发生什么？

**参考答案：**
训练判别器时，目标仅是更新 D 的参数。如果不 detach，`G(z)` 的计算图会连接到 G 的参数，D 的反向传播会同时更新 G 的参数，导致 G 在不该被更新时被错误地更新，破坏训练平衡。

**题2：** 为什么 GAN 使用 Tanh 而不是 Sigmoid 作为生成器的输出激活函数？

**参考答案：**
Tanh 输出范围 [-1, 1]，配合归一化到 [-1, 1] 的训练数据使用。相比 Sigmoid 的 [0, 1]，Tanh 以 0 为中心，梯度更大，训练更高效。Sigmoid 在输出接近 0 或 1 时梯度极小，会减缓训练。

### 进阶题

**题3：** 为什么原始 GAN 存在模式崩塌问题？WGAN 是如何缓解这个问题的？

**参考答案：**
模式崩塌的原因：原始 GAN 使用 JS 散度作为隐式度量，当 $p_{data}$ 和 $p_g$ 支撑集不重叠时，JS 散度为常数 $\log 2$，梯度为 0。这导致 G 找到一个"骗过" D 的模式后停止探索。WGAN 使用 Wasserstein 距离替代 JS 散度，即使两个分布不重叠，Wasserstein 距离也能提供有意义的梯度信号，引导 G 向 $p_{data}$ 的所有模式靠拢。

### 开放思考题

**题4：** GAN 和扩散模型各自在什么场景下更有优势？未来生成模型的发展方向是什么？

**参考答案思路：**
GAN 的优势在于采样速度快（单次前向传播），适合实时生成场景；扩散模型的优势在于训练稳定、生成质量高、多样性好、可控性强。在需要快速采样的场景（如实时游戏、交互设计）GAN 仍有价值。未来趋势可能是两者的融合，如 GAN 作为扩散模型的加速解码器，或一致性模型等新方法。

---

## 14. 学习路径建议

### 前置算法
- 自编码器（理解编码-解码架构）
- 深度学习基础（损失函数、梯度下降）
- 概率分布基础

### 平行算法
- VAE（另一种生成模型，对比理解）
- 归一化方法（BatchNorm 在 GAN 中的关键作用）

### 进阶算法
- DCGAN（卷积 GAN，生成图像的标准架构）
- WGAN/WGAN-GP（解决训练不稳定问题）
- StyleGAN（高分辨率人脸生成）
- 条件 GAN（CGAN、Pix2Pix、CycleGAN）

### 推荐资源
1. **论文**：Goodfellow et al., "Generative Adversarial Nets" (2014) — GAN 原始论文
2. **论文**：Arjovsky et al., "Wasserstein GAN" (2017) — WGAN 理论基础
3. **课程**：Stanford CS231N 生成模型部分 — GAN 的系统讲解
