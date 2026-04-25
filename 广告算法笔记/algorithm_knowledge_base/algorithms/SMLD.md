# SMLD（基于得分匹配的朗之万动力学生成模型）学习文档

## 1. 算法基础认知

SMLD（Score Matching with Langevin Dynamics）由 Yang Song 和 Stefano Ermon 等人于 2019 年提出，是基于得分函数的生成模型。核心思想：训练一个神经网络估计数据分布的得分函数（即对数概率密度的梯度 $\nabla_x \log p(x)$），然后通过 Langevin 动力学采样来生成数据。SMLD 的关键创新在于使用**多个噪声级别的得分匹配**（NCSN），解决了低密度区域得分估计不准确的问题。

## 2. 核心原理

**得分函数**：$\nabla_x \log p(x)$ 指向概率密度增长最快的方向。如果我们能准确估计得分函数，就可以沿着梯度方向从随机噪声出发，逐步移动到高密度区域（即真实数据分布）。

**Langevin 动力学采样**：

$$x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z, \quad z \sim \mathcal{N}(0, I)$$

当 $\epsilon \to 0$ 且步数 $\to \infty$ 时，$x_t$ 收敛到 $p(x)$ 的样本。

**多噪声级别（NCSN）**：单一得分网络在低密度区域估计不准。解决方案：用一系列递增的噪声级别 $\sigma_1 < \sigma_2 < ... < \sigma_L$ 扰动数据，在每个噪声级别分别估计得分，然后按噪声从大到小的顺序依次采样。

## 3. 数学公式与推导

**去噪得分匹配（DSM）损失**：

$$\mathcal{L} = \frac{1}{L} \sum_{i=1}^{L} \lambda(\sigma_i) \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)} \left[\left\|s_\theta(\tilde{x}, \sigma_i) - \frac{\tilde{x} - x}{\sigma_i^2}\right\|^2\right]$$

其中：
- $s_\theta(\tilde{x}, \sigma_i)$ 是噪声条件得分网络（NCSN）
- $\lambda(\sigma_i) = \sigma_i^2$ 是权重（使不同噪声级别的损失尺度一致）
- $\frac{\tilde{x} - x}{\sigma_i^2} = \nabla_{\tilde{x}} \log p(\tilde{x}|x)$ 是真实的条件得分

**为什么 $\frac{\tilde{x}-x}{\sigma_i^2}$ 是得分函数？**

因为 $p(\tilde{x}|x) = \frac{1}{(2\pi\sigma_i^2)^{d/2}} \exp\left(-\frac{\|\tilde{x}-x\|^2}{2\sigma_i^2}\right)$，对其取对数再求梯度：

$$\nabla_{\tilde{x}} \log p(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma_i^2}$$

## 4. 训练过程讲解

1. 选择噪声级别序列 $\{\sigma_1, ..., \sigma_L\}$（几何级数，如 1, 10, 50, 100）
2. 对每个训练样本 $x$，随机选择噪声级别 $\sigma_i$
3. 采样 $\tilde{x} = x + \sigma_i \epsilon, \epsilon \sim \mathcal{N}(0, I)$
4. 网络预测得分 $s_\theta(\tilde{x}, \sigma_i)$
5. 计算与真实得分 $-\frac{\epsilon}{\sigma_i}$ 的 MSE 损失
6. 反向传播更新参数

**采样过程（退火 Langevin 动力学）**：从最高噪声 $\sigma_L$ 开始，逐步降低噪声，每个级别运行若干步 Langevin 动力学。

## 5. 应用场景

- **图像生成**：高质量图像合成
- **图像修复**：利用条件得分函数填充缺失区域
- **图像着色**：条件生成
- **异常检测**：利用得分函数检测分布外样本
- **数据压缩**：与 ODE 结合

## 6. 优缺点分析

**优点：**
- 不需要对抗训练，训练稳定
- 理论优雅，有明确的概率解释
- 生成质量高，无模式坍塌

**缺点：**
- 采样需要多步 Langevin 动力学，速度较慢
- 需要精心设计噪声级别序列
- 噪声级别数量增加时训练成本增大

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class NCSN(nn.Module):
    def __init__(self, in_ch=1, hidden=64, num_sigmas=10):
        super().__init__()
        self.sigmas = nn.Parameter(torch.zeros(num_sigmas), requires_grad=False)
        self.embed = nn.Embedding(num_sigmas, hidden)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        )

    def forward(self, x, sigma_idx):
        sigma_emb = self.embed(sigma_idx)
        sigma_map = sigma_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        h = self.net(torch.cat([x, sigma_map], dim=1))
        h = self.up(h)
        score = self.dec(h)
        return score / self.sigmas[sigma_idx].view(-1, 1, 1, 1)

sigmas = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])

def dsm_loss(model, x0, sigmas):
    idx = torch.randint(0, len(sigmas), (x0.size(0),), device=x0.device)
    sigma = sigmas[idx].view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    x_tilde = x0 + sigma * noise
    score = model(x_tilde, idx)
    target = -noise / sigma
    return (score - target).pow(2).sum(dim=(1, 2, 3)).mean()

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = NCSN(num_sigmas=len(sigmas))
model.sigmas.data = sigmas
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        loss = dsm_loss(model, batch_x, sigmas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SMLDNumpy:
    def __init__(self, sigmas, input_dim, hidden_dim, lr=1e-4):
        self.sigmas = np.array(sigmas)
        self.num_sigmas = len(sigmas)
        self.lr = lr
        self.input_dim = input_dim
        he = lambda n: np.sqrt(2.0 / n)
        self.W1 = np.random.randn(input_dim + self.num_sigmas, hidden_dim) * he(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * he(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, input_dim) * he(hidden_dim)
        self.b3 = np.zeros(input_dim)
        self.sigma_embeddings = np.random.randn(self.num_sigmas, self.num_sigmas) * 0.01

    def forward(self, x, sigma_idx):
        sigma_emb = np.zeros((x.shape[0], self.num_sigmas))
        sigma_emb[np.arange(x.shape[0]), sigma_idx] = 1.0
        h = np.maximum(0, np.hstack([x, sigma_emb]) @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        score = h @ self.W3 + self.b3
        sigma = self.sigmas[sigma_idx].reshape(-1, 1)
        return score / sigma

    def train_step(self, x):
        sigma_idx = np.random.randint(0, self.num_sigmas, size=x.shape[0])
        sigma = self.sigmas[sigma_idx].reshape(-1, 1)
        noise = np.random.randn(*x.shape)
        x_tilde = x + sigma * noise
        score = self.forward(x_tilde, sigma_idx)
        target = -noise / sigma
        loss = np.mean((score - target) ** 2)
        return loss

    def annealed_langevin_sample(self, shape, steps_per_sigma=100, step_size=0.001):
        x = np.random.randn(*shape)
        for sigma in reversed(self.sigmas):
            for _ in range(steps_per_sigma):
                sigma_idx = np.where(self.sigmas == sigma)[0][0]
                idx_arr = np.full(shape[0], sigma_idx, dtype=int)
                score = self.forward(x, idx_arr)
                x = x + step_size * sigma ** 2 * score + np.sqrt(2 * step_size * sigma ** 2) * np.random.randn(*x.shape)
        return x
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def annealed_langevin(model, shape, sigmas, steps=100, eps=2e-5):
    x = torch.randn(shape)
    for sigma in reversed(sigmas):
        step_size = eps * sigma.item() ** 2
        idx = torch.full((shape[0],), len(sigmas) - 1 - torch.searchsorted(sigmas.flip(0), sigma))
        idx = (sigmas == sigma).nonzero().item()
        idx_t = torch.full((shape[0],), idx, dtype=torch.long)
        for _ in range(steps):
            score = model(x, idx_t)
            x = x + step_size * score + torch.sqrt(torch.tensor(2 * step_size)) * torch.randn_like(x)
    return x

samples = annealed_langevin(model, (16, 1, 32, 32), sigmas)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(samples[i*4+j, 0], cmap='gray')
        axes[i, j].axis('off')
plt.suptitle('SMLD/NCSN 生成结果')
plt.savefig('smld_samples.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **FID Score**：标准图像生成评估
- **Inception Score**：质量和多样性
- **得分估计精度**：在已知分布上对比真实得分和预测得分
- **采样收敛性**：观察 Langevin 采样的轨迹是否收敛

## 11. 常见问题与易错点

- **噪声级别选择**：$\sigma_1$ 应小于数据的典型尺度，$\sigma_L$ 应足够大覆盖整个数据空间
- **Langevin 步长**：与 $\sigma^2$ 成正比是关键，否则不同噪声级别下采样不稳定
- **得分函数尺度**：网络输出需要除以 $\sigma$ 来匹配得分的尺度，否则大噪声级别梯度太小
- **训练不收敛**：检查噪声级别是否合理、权重 $\lambda(\sigma_i)$ 是否正确

## 12. 学习总结

SMLD 是扩散模型的重要前驱工作。它提出了多噪声级别得分匹配和退火 Langevin 采样的范式，解决了原始得分匹配在低密度区域失效的问题。SMLD 后来与 DDPM 在 SDE 框架下被统一，构成了现代扩散模型的理论基础。

## 13. 练习题与思考题（含答案）

**Q1：为什么需要多噪声级别的得分匹配？**

A1：数据分布的低密度区域样本稀少，在这些区域的得分函数估计极不准确。添加噪声后，低密度区域被"填满"，得分函数变得更易估计。多级噪声确保从全局结构（高噪声）到精细细节（低噪声）都能准确捕获。

**Q2：退火 Langevin 动力学为什么从高噪声到低噪声采样？**

A2：高噪声时的得分函数更平滑，容易引导样本走向数据分布的大致区域；逐步降低噪声，让采样在越来越精细的尺度上调整。类似于先确定大致位置，再精细定位。

**Q3：SMLD 和 DDPM 本质上是同一种方法吗？**

A3：是的。Yang Song 在 2021 年的论文中证明，SMLD 和 DDPM 分别对应同一个 SDE 的两种不同离散化方式。SMLD 对应方差爆炸（VE）SDE，DDPM 对应方差保留（VP）SDE。

## 14. 学习路径建议

1. 理解得分函数的概念和 Langevin 动力学
2. 学习去噪得分匹配（DSM）的训练方法
3. 实现 NCSN 和退火 Langevin 采样
4. 对比 SMLD 和 DDPM，理解统一 SDE 框架
5. 进阶：Flow Matching、一致性模型等最新方法
