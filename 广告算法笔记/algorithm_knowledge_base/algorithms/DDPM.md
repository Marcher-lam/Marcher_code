# DDPM（去噪扩散概率模型）学习文档

## 1. 算法基础认知

去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）由 Ho 等人于 2020 年提出，是当前最主流的生成模型之一。DDPM 包含两个过程：**前向过程**逐步向数据添加高斯噪声直到变成纯噪声，**反向过程**学习逐步去噪，从噪声中恢复出干净数据。其生成质量超越了 GAN，同时训练更加稳定。

## 2. 核心原理

**前向过程（加噪）**：给定干净数据 $x_0$，逐步添加高斯噪声，共 $T$ 步：

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

直接从 $x_0$ 得到任意步 $x_t$ 的公式：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，$\alpha_t = 1 - \beta_t$。

**反向过程（去噪）**：训练一个神经网络 $\epsilon_\theta(x_t, t)$ 预测第 $t$ 步添加的噪声 $\epsilon$，然后用预测的噪声逐步去噪。

## 3. 数学公式与推导

**训练目标**：简化后的损失函数非常优雅：

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中：
- $t \sim \text{Uniform}(\{1, ..., T\})$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**反向过程单步去噪**：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

其中 $z \sim \mathcal{N}(0, I)$（$t > 1$ 时），$\sigma_t = \sqrt{\beta_t}$。

**噪声调度（Noise Schedule）**：常用的线性调度 $\beta_t$ 从 $\beta_1 = 10^{-4}$ 线性增长到 $\beta_T = 0.02$。

## 4. 训练过程讲解

1. 采样干净数据 $x_0$
2. 随机采样时间步 $t \sim \text{Uniform}(1, T)$
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I)$
4. 计算 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. 网络预测噪声 $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
6. 计算损失 $\|\epsilon - \hat{\epsilon}\|^2$ 并反向传播

**采样（生成）过程**：

1. 从 $\mathcal{N}(0, I)$ 采样 $x_T$
2. 从 $t = T$ 到 $t = 1$，逐步用 $\epsilon_\theta(x_t, t)$ 去噪得到 $x_{t-1}$
3. 最终得到生成样本 $x_0$

## 5. 应用场景

- **图像生成**：DALL·E、Stable Diffusion、Imagen 的核心
- **音频生成**：DiffWave、Grad-TTS
- **视频生成**：视频扩散模型
- **3D 生成**：点云、NeRF 生成
- **分子生成**：药物设计

## 6. 优缺点分析

**优点：**
- 生成质量极高，超越 GAN
- 训练稳定，不存在模式坍塌
- 有坚实的概率论基础
- 可控生成（classifier guidance）

**缺点：**
- 采样速度慢，需要 $T$ 步迭代（通常 $T = 1000$）
- 计算资源消耗大
- 训练需要大量数据和时间

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.time_linear1 = nn.Linear(time_emb_dim, 128)
        self.time_linear2 = nn.Linear(time_emb_dim, 64)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(self.pool(h1)))
        h2 = h2 + self.time_linear1(t_emb).unsqueeze(-1).unsqueeze(-1)
        h3 = torch.relu(self.conv3(self.up(h2)))
        h3 = h3 + self.time_linear2(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.conv4(h3 + h1)
        return out

class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def train_loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.add_noise(x0, t, noise)
        t_input = (t.float() / self.T).unsqueeze(1)
        pred_noise = self.model(x_t, t_input)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_input = (t.float() / self.T).unsqueeze(1)
            pred_noise = self.model(x, t_input)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
            if t_idx > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)
        return x

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleUNet()
    ddpm = DDPM(model, T=200, beta_start=1e-4, beta_end=0.02)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(20):
        total_loss = 0
        for batch_x, _ in loader:
            loss = ddpm.train_loss(batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class DDPMNumpy:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def forward_step(self, x0, t):
        noise = np.random.randn(*x0.shape)
        sqrt_alpha_bar = np.sqrt(self.alpha_bars[t])
        sqrt_one_minus = np.sqrt(1 - self.alpha_bars[t])
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return x_t, noise

    def reverse_step(self, x_t, predicted_noise, t):
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        coeff1 = 1.0 / np.sqrt(alpha_t)
        coeff2 = (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
        x_prev = coeff1 * (x_t - coeff2 * predicted_noise)
        if t > 0:
            x_prev += np.sqrt(beta_t) * np.random.randn(*x_t.shape)
        return x_prev

    def full_sample(self, predict_fn, shape):
        x = np.random.randn(*shape)
        for t in reversed(range(self.T)):
            pred_noise = predict_fn(x, t)
            x = self.reverse_step(x, pred_noise, t)
        return x

    def compute_loss(self, x0, predict_fn):
        t = np.random.randint(0, self.T)
        x_t, noise = self.forward_step(x0, t)
        pred_noise = predict_fn(x_t, t)
        return np.mean((noise - pred_noise) ** 2)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

samples = ddpm.sample((16, 1, 32, 32))
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(samples[i*4+j, 0], cmap='gray')
        axes[i, j].axis('off')
plt.suptitle('DDPM 生成结果')
plt.savefig('ddpm_samples.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **FID Score**：最常用的扩散模型评估指标
- **IS Score**：衡量生成质量和多样性
- **采样速度**：记录生成单张图像所需时间
- **噪声预测精度**：验证 $\epsilon_\theta$ 在不同 $t$ 下的 MSE

## 11. 常见问题与易错点

- **$\bar{\alpha}_t$ 计算错误**：必须是累积乘积，不是求和
- **时间步编码**：网络需要知道当前是第几步，常用正弦位置编码
- **采样太慢**：$T=1000$ 步很慢，可用 DDIM 加速到 50 步
- **$\beta$ 调度选择**：线性调度不是最优的，cosine 调度通常效果更好

## 12. 学习总结

DDPM 的优雅之处在于：训练极其简单（只需预测噪声），但生成效果极强。它用 $T$ 步的渐进式去噪取代了 GAN 的一步生成，换来了训练稳定性和生成质量的飞跃。DDPM 是 Stable Diffusion、DALL·E 等产品的核心算法。

## 13. 练习题与思考题（含答案）

**Q1：DDPM 的训练为什么比 GAN 更稳定？**

A1：DDPM 的训练目标是一个简单的回归问题（预测噪声），不需要对抗博弈，不存在 $G$ 和 $D$ 的平衡问题。每一步的训练都是独立的均方误差最小化。

**Q2：为什么 DDPM 的采样速度慢？如何加速？**

A2：因为需要逐步执行 $T$ 次去噪（通常 1000 步）。加速方法：DDIM（确定性采样，可减至 50 步）、一致性模型、蒸馏等。

**Q3：前向过程中 $\beta_t$ 的作用是什么？**

A3：$\beta_t$ 控制每一步添加的噪声量。较小的 $\beta_t$ 使得每步变化小，反向过程更容易学习。但 $\beta_t$ 也不能太小，否则需要太多步才能完全变成噪声。

## 14. 学习路径建议

1. 掌握 VAE 和 GAN 的基础
2. 理解前向/反向扩散过程的数学推导
3. 实现 DDPM（简化版 UNet + 噪声预测）
4. 学习 DDIM（加速采样）、Stable Diffusion（潜在空间扩散）
5. 了解 score-based model 与 DDPM 的统一理论
