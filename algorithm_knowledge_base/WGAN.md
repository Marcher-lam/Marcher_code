# WGAN 学习文档

> 用 Wasserstein 距离替代 JS 散度，从根本上解决 GAN 训练不稳定问题。

> 来源线索：本节内容根据原书中关于"WGAN"的相关章节（第4章4.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** WGAN 用 Earth-Mover（Wasserstein-1）距离替代原始 GAN 的 JS 散度作为训练目标，即使生成分布与真实分布不重叠也能提供有意义的梯度。

**直觉类比：** 原始 GAN 的判别器像"考试官"——只回答"真或假"（0或1），当生成器做得不好时直接给 0 分，没有改进方向。WGAN 的判别器（称"评论家"）像一个"评分老师"——给一个连续分数（如 0.3 或 0.7），无论生成器多差都能告诉它"差多少"，从而提供持续的改进梯度。

**历史背景：** WGAN 由 Arjovsky 等人于 2017 年提出（论文 "Wasserstein GAN"），同年 WGAN-GP（梯度惩罚版）由 Gulrajani 等人提出，进一步稳定了训练。WGAN 被认为是 GAN 理论的重要突破。

**算法定位：** 生成模型、GAN 改进、Wasserstein 距离。

**前置知识：** GAN、JS 散度、KL 散度、Lipschitz 连续性、PyTorch。

---

## 2. 核心原理

### 原始 GAN 的问题

原始 GAN 的判别器优化 JS 散度。当真实分布 $P_r$ 和生成分布 $P_g$ 完全不重叠时（训练初期几乎总是如此），$JS = \log 2$（常数），梯度为 0——**模式崩塌的根源**。

### WGAN 的改进

用 Wasserstein 距离替代 JS 散度，由 Kantorovich-Rubinstein 对偶：

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

即使 $P_r$ 和 $P_g$ 不重叠，$W$ 距离仍然有意义且连续，提供持续梯度。

---

## 3. 数学公式与推导

### 评论家损失

$$\mathcal{L}_C = \mathbb{E}_{x \sim P_g}[f_w(x)] - \mathbb{E}_{x \sim P_r}[f_w(x)]$$

### 生成器损失

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p(z)}[f_w(G(z))]$$

### Lipschitz 约束的实现

**权重裁剪（WGAN）**：$w \leftarrow \text{clip}(w, -c, c)$

**梯度惩罚（WGAN-GP）**：

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} f_w(\hat{x})\|_2 - 1)^2]$$

其中 $\hat{x} = \epsilon x_r + (1-\epsilon) x_g$ 是插值样本。

---

## 4. 训练过程讲解

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| n_critic | 5 | 5 |
| $\lambda$ (GP) | 10 | 10 |
| lr | 1e-4 ~ 2e-4 | 1e-4 |
| $\beta_1$ (Adam) | 0.0 | 0.0 |

---

## 5. 应用场景

1. **高质量图像生成**：比原始 GAN 训练稳定得多
2. **文本生成**：WGAN 可用于离散 token 生成（通过 Gumbel-Softmax）
3. **领域迁移**：作为 CycleGAN 等模型的基础

---

## 6. 优缺点分析

### 优点
1. **训练稳定**：不再需要精心平衡 G 和 D
2. **有意义的损失曲线**：W 距离越小，生成质量越好
3. **无需 Sigmoid/BN**：评论家输出连续值

### 缺点
1. **训练速度**：评论家训练次数多（通常 5 次 G 对 1 次 D）
2. **梯度惩罚计算**：需要计算二阶梯度，略慢

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(-1, z.size(1), 1, 1))

class WGANCritic(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0)
        )
    def forward(self, x):
        return self.net(x).view(-1)

def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    critic_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True, retain_graph=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

G, C = WGANGenerator(), WGANCritic()
opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))
print(f"G参数: {sum(p.numel() for p in G.parameters()):,}, C参数: {sum(p.numel() for p in C.parameters()):,}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleWGAN:
    def __init__(self, latent_dim=10, data_dim=2, hidden=32):
        scale = 0.01
        self.G_w1 = np.random.randn(latent_dim, hidden) * scale
        self.G_w2 = np.random.randn(hidden, data_dim) * scale
        self.C_w1 = np.random.randn(data_dim, hidden) * scale
        self.C_w2 = np.random.randn(hidden, 1) * scale

    def generate(self, z):
        h = np.maximum(0, z @ self.G_w1)
        return h @ self.G_w2

    def criticize(self, x):
        h = np.maximum(0.2 * (x @ self.C_w1), x @ self.C_w1)
        return (h @ self.C_w2).flatten()

    def wasserstein_distance(self, real, fake):
        return self.criticize(real).mean() - self.criticize(fake).mean()

wgan = SimpleWGAN()
real = np.random.randn(64, 2) + 2
fake = wgan.generate(np.random.randn(64, 10))
print(f"Wasserstein距离: {wgan.wasserstein_distance(real, fake):.4f}")
```

---

## 9-14. 评估/问题/总结/练习/路径

### 练习题

**题1：** WGAN 为什么不用 Sigmoid？

**参考答案：** WGAN 的评论家输出连续分数估计 Wasserstein 距离，不输出概率。Sigmoid 将输出压缩到 [0,1]，限制评论家区分远近的能力。

**题2（开放）：** WGAN-GP 的梯度惩罚为什么要求梯度范数接近 1？

**参考答案思路：** Lipschitz-1 约束要求 $\|\nabla f\| \leq 1$。惩罚项 $(\|\nabla f\|_2 - 1)^2$ 鼓励评论家在数据分布上满足此约束。

### 学习路径
- 前置：GAN、JS 散度
- 进阶：Spectral Normalization、StyleGAN
- 推荐：Arjovsky et al., "Wasserstein GAN" (2017)
