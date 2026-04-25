# DM（扩散模型）学习文档

## 1. 算法基础认知

扩散模型（Diffusion Model, DM）是一类基于随机微分方程（SDE）和得分匹配（Score Matching）的生成模型家族。DDPM 是离散化的扩散模型，而 DM 更广义地包含了基于 SDE 的连续时间扩散、基于得分函数的理论框架，以及 score-based generative model 与 DDPM 的统一视角。本节聚焦于连续扩散过程和得分匹配的核心理论。

## 2. 核心原理

扩散模型的理论框架由三个核心要素构成：

1. **得分函数（Score Function）**：数据分布的梯度 $\nabla_x \log p(x)$，指向概率密度增长最快的方向
2. **得分匹配（Score Matching）**：训练网络 $s_\theta(x)$ 来估计得分函数，避免了直接计算 $\nabla_x \log p(x)$ 需要归一化常数的困难
3. **Langevin 动力学采样**：利用得分函数通过迭代采样生成数据

**SDE 视角**：前向过程可以用连续 SDE 描述：

$$dx = f(x, t)dt + g(t)dw$$

反向过程用反向 SDE：

$$dx = [f(x, t) - g^2(t) \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

其中 $\nabla_x \log p_t(x)$ 就是需要学习的得分函数。

## 3. 数学公式与推导

**去噪得分匹配（DSM）损失**：

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}(0, T)} \mathbb{E}_{x_0 \sim p_{data}} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma_t^2 I)} \left[\lambda(t) \|s_\theta(x_0 + \sigma_t \epsilon, t) + \frac{\epsilon}{\sigma_t}\|^2\right]$$

其中 $\lambda(t)$ 是权重函数，$\sigma_t$ 是时间 $t$ 对应的噪声水平。

**为什么学习得分函数等价于预测噪声？**

对于加噪数据 $x_t = x_0 + \sigma_t \epsilon$，其得分函数为：

$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{\epsilon}{\sigma_t^2}$$

因此 $s_\theta(x_t, t) \approx -\frac{\epsilon}{\sigma_t^2}$，即学习得分函数等价于学习预测噪声 $\epsilon$。

**概率流 ODE（确定性采样）**：

$$dx = \left[f(x, t) - \frac{1}{2}g^2(t) \nabla_x \log p_t(x)\right]dt$$

使用 ODE 求解器可以高效采样，且支持精确的对数似然计算。

## 4. 训练过程讲解

1. 采样训练数据 $x_0$ 和时间 $t$
2. 采样噪声 $\epsilon \sim \mathcal{N}(0, I)$
3. 计算加噪样本 $x_t = x_0 + \sigma_t \epsilon$
4. 网络预测得分 $s_\theta(x_t, t)$
5. 计算 DSM 损失并反向传播
6. 训练完成后，用 Langevin 动力学或 ODE 求解器采样

**噪声条件化**：关键设计是让网络同时以 $(x_t, t)$ 为输入，即噪声条件网络（NCSN），使单个网络可以处理不同噪声级别。

## 5. 应用场景

- **图像/视频/音频生成**：统一框架
- **图像修复与编辑**：利用条件生成
- **分子和蛋白质结构生成**
- **科学计算**：求解逆问题
- **数据压缩**：利用概率流 ODE

## 6. 优缺点分析

**优点：**
- 统一的理论框架（涵盖 DDPM 和 score-based model）
- 连续时间视角更优雅，支持各种 SDE 求解器
- 支持 ODE 采样，可精确计算似然
- 训练稳定，无模式坍塌

**缺点：**
- 理论门槛较高（需要理解 SDE 和随机过程）
- 采样速度仍然较慢
- 计算资源消耗大

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class ScoreNet(nn.Module):
    def __init__(self, in_ch=1, hidden_dim=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv1 = nn.Conv2d(in_ch, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, in_ch, 3, padding=1)
        self.t_proj1 = nn.Linear(time_dim, hidden_dim * 2)
        self.t_proj2 = nn.Linear(time_dim, hidden_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        h2 = h2 + self.t_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        h3 = torch.relu(self.conv3(h2))
        h4 = torch.relu(self.conv4(self.up(h3)))
        h4 = h4 + self.t_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        h5 = torch.relu(self.conv5(h4 + h1))
        return self.conv_out(h5)

sigma_min, sigma_max = 0.01, 50.0
T = 1.0

def sigma_schedule(t):
    return sigma_min * (sigma_max / sigma_min) ** t

def dsm_loss(model, x0):
    t = torch.rand(x0.size(0), 1, device=x0.device)
    sigma = sigma_schedule(t).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    x_t = x0 + sigma * noise
    score = model(x_t, t)
    target = -noise / sigma
    loss = (score - target).pow(2).sum(dim=(1, 2, 3)).mean()
    return loss

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ScoreNet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        loss = dsm_loss(model, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class DiffusionModelNumpy:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, num_steps=500):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_steps))

    def perturb_data(self, x0, sigma):
        noise = np.random.randn(*x0.shape)
        return x0 + sigma * noise, noise

    def langevin_sample(self, score_fn, shape, num_steps=500, lr_factor=0.1):
        x = np.random.randn(*shape) * self.sigma_max
        for i in reversed(range(num_steps)):
            sigma = self.sigmas[i]
            step_size = lr_factor * sigma ** 2
            score = score_fn(x, sigma)
            noise = np.random.randn(*shape)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
        return x

    def euler_ode_sample(self, score_fn, shape, num_steps=500):
        x = np.random.randn(*shape) * self.sigma_max
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = 1.0 - i / num_steps
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            score = score_fn(x, sigma)
            x = x + dt * (-sigma * score * sigma)
        return x

    def compute_dsm_loss(self, x0, score_fn):
        sigma = np.random.choice(self.sigmas)
        x_t, noise = self.perturb_data(x0, sigma)
        score = score_fn(x_t, sigma)
        target = -noise / sigma
        return np.mean((score - target) ** 2)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def langevin_sample(model, shape, num_steps=500, eps=0.1):
    x = torch.randn(shape) * sigma_max
    sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), num_steps))
    for i in reversed(range(num_steps)):
        sigma = sigmas[i]
        t = torch.full((shape[0], 1), (i / num_steps))
        score = model(x, t)
        step_size = eps * sigma.item() ** 2
        x = x + step_size * score + np.sqrt(2 * step_size) * torch.randn_like(x)
    return x

samples = langevin_sample(model, (16, 1, 32, 32))
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(samples[i*4+j, 0], cmap='gray')
        axes[i, j].axis('off')
plt.suptitle('Score-Based Diffusion Model 生成结果')
plt.savefig('dm_samples.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **FID Score**：标准评估指标
- **对数似然**：利用概率流 ODE 可精确计算
- **采样质量**：视觉检查生成样本的清晰度和多样性
- **采样效率**：对比不同求解器的采样速度和质量

## 11. 常见问题与易错点

- **SDE vs ODE 采样**：SDE 采样有随机性，质量通常更好；ODE 采样确定性，支持精确似然计算
- **噪声调度选择**：指数调度通常优于线性调度，需要覆盖足够的噪声范围
- **得分函数的尺度**：不同噪声级别下得分函数的尺度差异很大，需要适当的权重 $\lambda(t)$
- **Langevin 采样的步长**：步长太大导致发散，太小收敛慢

## 12. 学习总结

扩散模型的广义视角统一了 DDPM 和 score-based model。核心是学习得分函数 $\nabla_x \log p_t(x)$，通过 SDE 描述扩散过程，通过反向 SDE 或 ODE 采样。理解这个框架对掌握现代生成模型至关重要。

## 13. 练习题与思考题（含答案）

**Q1：连续时间扩散模型与离散 DDPM 有什么关系？**

A1：DDPM 是连续 SDE 的离散化。当 SDE 的 $f(x,t) = -\frac{1}{2}\beta(t)x$，$g(t) = \sqrt{\beta(t)}$，并在离散时间步上用 Euler-Maruyama 方法离散化，就得到 DDPM。

**Q2：概率流 ODE 的优势是什么？**

A2：ODE 是确定性的，支持精确的对数似然计算、潜在空间插值、以及使用高效的 ODE 求解器（如 Runge-Kutta）加速采样。

**Q3：为什么得分匹配可以避免配分函数的计算？**

A3：得分匹配通过 Fisher 散度 $\mathbb{E}[\|\nabla_x \log p(x) - s_\theta(x)\|^2]$ 来训练，利用分部积分可以将目标转化为只涉及数据分布和模型的表达式，不需要计算归一化常数 $Z$。

## 14. 学习路径建议

1. 掌握 DDPM 的前向/反向过程
2. 学习得分匹配的基本理论
3. 理解 SDE 视角的统一框架（Yang Song 的论文）
4. 学习概率流 ODE 和高效采样方法
5. 进阶：Flow Matching、Rectified Flow 等最新发展
