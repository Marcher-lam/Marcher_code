# SMLD (Score Matching Likelihood Diffusion) 分数匹配扩散模型 学习文档

> SMLD是基于分数匹配的扩散模型生成方法，通过学习数据分布的梯度（分数）来实现采样

---

## 1. 算法基础认知

### 1.1 一句话定义

**SMLD（Score Matching Likelihood Diffusion）** 是一种基于分数匹配（Score Matching）原理的扩散生成模型，其核心思想是通过学习对数概率密度函数的梯度（即"分数"$\nabla_x \log p(x)$）来建模数据分布，然后利用 Langevin Dynamics 进行采样生成。

### 1.2 直觉类比

想象一滴墨水在清水中扩散的过程：墨水分子会从高浓度区域（密集的地方）不断地向低浓度区域（稀疏的地方）运动，最终达到均匀分布。**SMLD的核心**是反向思考这个问题——如果我们知道"墨水分子在任何位置应该往哪个方向移动"（即分数方向），我们就可以从清水的"无结构状态"开始，逐步向墨水聚集的方向移动，最终生成"有结构的墨水"。这个"往哪个方向移动"的信息就是$\nabla_x \log p(x)$。

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 2011 | Score Matching (Hyvarinen) - 原始分数匹配 |
| 2015 | Denoising Score Matching - 去噪分数匹配 |
| 2019 | NSC (Song & Ermon) - 噪声条件分数网络 |
| 2021 | SMLD (Song & Ermon) - 扩散模型框架 |
| 2021 | DDPM - 去噪扩散概率模型 |
| 2022 | EDM - 改进的扩散模型 |
| 2023 | 加速采样 (PNDM, DPM-Solver) |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 生成模型 / 扩散模型 |
| 核心 | 学习分数函数 $\nabla_x \log p(x)$ |
| 对比 | DDPM的两种推导方式之一 |
| 优点 | 训练稳定，可处理任意噪声 |

### 1.5 前置知识

- 概率论（概率密度函数、对数似然）
- 随机微分方程（Langevin Dynamics）
- 神经网络（ResNet、U-Net）
- Python + PyTorch

---

## 2. 核心原理

### 2.1 分数匹配原理

**核心思想**：不直接学习概率密度 $p(x)$，而是学习分数（对数密度的梯度）：
$$\nabla_x \log p(x) = \frac{\partial \log p(x)}{\partial x} = \frac{1}{p(x)} \cdot \frac{\partial p(x)}{\partial x}$$

**为什么学习分数**：
- 分数是有方向的梯度向量，指向概率增加的方向
- 分数归一化友好：不需要知道归一化常数
- 可用于 Langevin 采样

### 2.2 扩散过程

**前向扩散（数据 → 噪声）**：
$$x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

或者用连续时间表示：
$$d x_t = -\frac{1}{2} \beta_t \cdot x_t \cdot dt + \sqrt{\beta_t} \cdot dW_t$$

其中 $\beta_t$ 是方差 schedule，$W_t$ 是 Wiener 过程。

**逆向过程（噪声 → 数据）**：
根据贝叶斯定理和随机微分方程，可以推导出逆向SDE：
$$d x_t = \left[\frac{1}{2} x_t - s_\theta(x_t, t)\right] dt + \sqrt{\beta_t} \cdot dW_t$$

其中 $s_\theta(x_t, t)$ 是学习的分数网络。

### 2.3 噪声条件分数网络

**挑战**：直接学习 $p(x_t)$ 的分数很困难

**解决方案**：引入条件概率 $p(x_t|x_0)$：
$$p(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

其中 $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$

**分数**：
$$\nabla_{x_t} \log p(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t}$$

**条件分数网络**：$s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t)$

### 2.4 训练目标

**���噪分数匹配损失**：
$$L = \mathbb{E}_{t, x_0, \epsilon} \left[ \| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \|^2 \right]$$

其中：
- $t \sim U[1, T]$
- $x_0 \sim p_{data}$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x_0$ | 原始数据 |
| $x_t$ | t时刻的加噪数据 |
| $\beta_t$ | 方差 schedule |
| $\alpha_t = 1-\beta_t$ |  |
| $\bar{\alpha}_t$ | 累积产品 |
| $\epsilon$ | 高斯噪声 |
| $s_\theta(x_t, t)$ | 分数网络 |
| $\sigma(t)$ | 标准差 schedule |

### 3.2 前向扩散（离散）

**每步添加高斯噪声**：
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**边缘分布**：
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

### 3.3 分数定义

**对数密度的梯度**：
$$\nabla_x \log p(x) = \frac{1}{p(x)} \nabla_x p(x)$$

**高斯分布的分数**：
对于 $p(x) = \mathcal{N}(x; \mu, \sigma^2)$：
$$\nabla_x \log p(x) = -\frac{x-\mu}{\sigma^2}$$

**加噪数据的分数**：
$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

### 3.4 损失函数推导

**目标**：让 $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_\theta(x_t)$

**分数匹配损失**（简化形式）：
$$L_{SM} = \mathbb{E}_{x} \left[ \frac{1}{2} \| \nabla_x \log p_\theta(x) - \nabla_x \log p_{data}(x) \|^2 \right]$$

**去噪分数匹配**（更实用）：
由链式法则可得：
$$\nabla_x \log p_{data}(x) = \mathbb{E}_{q(x_t|x)}[\nabla_x \log q(x_t|x)]$$

代入 $x_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1-\bar{\alpha}_t} \epsilon$：
$$\nabla_{x_t} \log q(x_t|x) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此：
$$L = \mathbb{E}_{t, x, \epsilon} \left[ \left\| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \right\|^2 \right]$$

### 3.5 Langevin 采样

**MCMC采样**：
$x_{i+1} = x_i + \alpha \cdot \nabla_x \log p(x_i) + \sqrt{2\alpha} \cdot z_i$

其中 $z_i \sim \mathcal{N}(0, I)$

**迭代步骤**：
1. 计算分数：$s = \nabla_x \log p(x_i)$
2. 添加梯度方向：$x_i = x_i + \alpha \cdot s$
3. 添加噪声：$x_i = x_i + \sqrt{2\alpha} \cdot z_i$
4. 迭代直到收敛

**SMLD采样**（用学习的分数）：
$$x_{i+1} = x_i + \alpha \cdot s_\theta(x_i) + \sqrt{2\alpha} \cdot z_i$$

### 3.6 逆向SDE

**目标**：从 $x_T \sim \mathcal{N}(0, \sigma_{max}^2 I)$ 采样回 $x_0$

**逆向SDE**：
$$d x_t = \left[ f(t) x_t - g(t)^2 s_\theta(x_t, t) \right] dt + g(t) dW_t$$

对于线性 schedule $f(t) = -\frac{1}{2} \beta(t), g(t) = \sqrt{\beta(t)}$：
$$d x_t = \left[ \frac{1}{2} x_t - \beta(t) s_\theta(x_t, t) \right] dt + \sqrt{\beta(t)} dW_t$$

---

## 4. PyTorch实现

### 4.1 分数网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ScoreNet(nn.Module):
    """分数网络：预测 \nabla_x log p(x_t)"""
    
    def __init__(self, input_dim, hidden_dim=128, time_embed_dim=64):
        super(ScoreNet, self).__init__()
        self.input_dim = input_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 网络主体
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, input_dim) - 加噪数据
            t: (batch, 1) - 时间步 [0, 1]
        
        Returns:
            scores: (batch, input_dim) - 预测的分数
        """
        t_embed = self.time_mlp(t)
        h = torch.cat([x, t_embed], dim=-1)
        scores = self.net(h)
        return scores


class UNetScoreNet(nn.Module):
    """U-Net风格的分数网络（用于图像）"""
    
    def __init__(self, channels=3, hidden_channels=64, time_embed_dim=256):
        super(UNetScoreNet, self).__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 下采样
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels*4, 3, stride=2, padding=1)
        
        # 瓶颈
        self.bottleneck = nn.Conv2d(hidden_channels*4, hidden_channels*4, 3, padding=1)
        
        # 上采样 + 残差连接
        self.up1 = nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 4, stride=2, padding=1)
        self.conv_up1 = nn.Conv2d(hidden_channels*4, hidden_channels*2, 3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, stride=2, padding=1)
        self.conv_up2 = nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1)
        
        # 输出
        self.out = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        
        self.norm = nn.GroupNorm(32, hidden_channels)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        
        # 下采样
        h1 = F.silu(self.norm(self.conv1(x)))
        h2 = F.silu(self.norm(self.conv2(h1)))
        h3 = F.silu(self.norm(self.conv3(h2)))
        
        # 瓶颈
        h = F.silu(self.bottleneck(h3))
        
        # 上采样
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = F.silu(self.conv_up1(h))
        
        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = F.silu(self.conv_up2(h))
        
        out = self.out(h)
        return out
```

### 4.2 SMLD训练

```python
class SMLD:
    """SMLD训练器"""
    
    def __init__(self, score_net, device='cuda'):
        self.score_net = score_net.to(device)
        self.device = device
        
        # 超参数
        self.num_steps = 1000
        self.batch_size = 32
        self.lr = 1e-4
        
        self.optimizer = torch.optim.Adam(self.score_net.parameters(), lr=self.lr)
    
    def train_step(self, x0):
        """
        单步训练
        
        Args:
            x0: (batch, dim) - 原始数据
        """
        batch_size = x0.shape[0]
        
        # 随机采样时间步
        t = torch.rand(batch_size, 1, device=self.device) * (1 - 1e-5) + 1e-5
        
        # 计算 bar_alpha_t
        log_alpha_bar_t = -t * 10  # 简化：使用指数schedule
        bar_alpha_t = torch.exp(log_alpha_bar_t)
        
        # 添加噪声
        eps = torch.randn_like(x0)
        xt = bar_alpha_t.sqrt() * x0 + (1 - bar_alpha_t).sqrt() * eps
        
        # 预测分数
        st = self.score_net(xt, t)
        
        # 目标分数 = -eps / sqrt(1 - bar_alpha_t)
        target = -eps / (1 - bar_alpha_t).sqrt()
        
        # 损失
        loss = F.mse_loss(st, target)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_loop(self, dataloader, num_epochs):
        """完整训练循环"""
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                x0 = batch.to(self.device)
                loss = self.train_step(x0)
                total_loss += loss
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
    
    def langevin_sample(self, xt_init, num_steps=100, step_size=0.01):
        """
        Langevin采样
        
        Args:
            xt_init: 初始噪声 (1, dim)
            num_steps: 采样步数
            step_size: 步长
        
        Returns:
            x_sample: 采样结果
        """
        self.score_net.eval()
        
        x = xt_init.clone()
        
        with torch.no_grad():
            for _ in range(num_steps):
                # 计算分数
                t = torch.ones(1, 1, device=self.device)  # 使用t=1表示纯噪声
                score = self.score_net(x, t)
                
                # Langevin更新
                x = x + step_size * score + math.sqrt(2 * step_size) * torch.randn_like(x)
        
        return x
```

### 4.3 SMLD采样

```python
def smld_sampling(score_net, init_noise, num_steps=100, schedule='cosine'):
    """
    SMLD采样（使用逆向SDE的离散化）
    
    Args:
        score_net: 分数网络
        init_noise: 初始噪声
        num_steps: 采样步数
        schedule: beta schedule类型
    
    Returns:
        samples: 采��结果
    """
    
    # 生成schedule
    if schedule == 'linear':
        betas = torch.linspace(1e-4, 0.02, num_steps)
    elif schedule == 'cosine':
        t = torch.arange(num_steps + 1) / num_steps
        alphas = torch.cos(t * math.pi / 2) ** 2
        betas = alphas[:-1] - alphas[1:]
        betas = torch.clip(betas, 0, 0.999)
    
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    x = init_noise
    
    for t in range(num_steps):
        # 计算分数
        t_tensor = torch.ones_like(x[:, :1]) / num_steps * t
        score = score_net(x, t_tensor)
        
        # 采样更新
        if t < num_steps - 1:
            # 有噪声的更新
            x = x + ((1 - alphas[t]) / (1 - alpha_bars[t])) * score * betas[t]
            x = x + betas[t].sqrt() * torch.randn_like(x)
        else:
            # 最后一步无噪声
            x = x + ((1 - alphas[t]) / (1 - alpha_bars[t])) * score * betas[t]
        
        alpha_bars = alpha_bars[1:]
        betas = betas[1:]
    
    return x


def annealed_langevin_dynamics(score_net, init_noise, num_steps=100, 
                        num_langevin_steps=10, sigma_min=0.01, sigma_max=1):
    """
    退火Langevin动力学采样
    """
    
    # 不同的噪声水平
    sigmas = torch.linspace(sigma_max, sigma_min, num_steps)
    
    x = init_noise
    
    for sigma in sigmas:
        # 该噪声水平下的 Langevin 采样
        for _ in range(num_langevin_steps):
            noise = torch.randn_like(x)
            score = score_net(x, sigma)
            
            # 步长
            epsilon = 1e-4
            x = x + epsilon * score + math.sqrt(2 * epsilon) * sigma * noise
        
        # 降低噪声水平时进行跳跃
        x = x + noise * (sigmas[0] - sigma)
    
    return x
```

### 4.4 完整训练

```python
class SMLDModel(nn.Module):
    """完整的SMLD模型"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.score_net = ScoreNet(input_dim, hidden_dim)
    
    def forward(self, x, t):
        return self.score_net(x, t)
    
    def training_loss(self, x0):
        batch_size = x0.shape[0]
        
        # 时间步
        t = torch.rand(batch_size, 1, device=x0.device)
        
        # 噪声水平
        sigma = t * 10  # 简化
        x_noisy = x0 + sigma * torch.randn_like(x0)
        
        # 目标分数
        target = -x_noisy / sigma.pow(2)
        
        # 预测
        prediction = self.score_net(x_noisy, t)
        
        return F.mse_loss(prediction, target)
    
    def sample(self, shape):
        """生成样本"""
        x = torch.randn(shape, device=next(self.parameters()).device)
        
        for t in range(1000, 0, -1):
            t_batch = torch.ones(shape[0], 1, device=x.device) * t / 1000
            score = self.score_net(x, t_batch)
            
            # 简化的采样（有噪声）
            alpha = 0.001
            x = x + alpha * score + math.sqrt(2 * alpha) * torch.randn_like(x)
        
        return x
```

---

## 5. 代码示例

### 5.1 完整示例

```python
import torch
import numpy as np
import matplotlib.pyplot as plt


def demo_smmld():
    """演示SMLD"""
    
    print("=" * 60)
    print("SMLD (Score Matching Likelihood Diffusion) 演示")
    print("=" * 60)
    
    # 创建简单的2D数据（环形分布）
    n_samples = 10000
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(1, 0.1, n_samples)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.stack([x, y], axis=1).astype(np.float32)
    
    print(f"数据形状: {data.shape}")
    print(f"数��范围: [{data.min():.2f}, {data.max():.2f}]")
    
    # 创建模型
    model = ScoreNet(input_dim=2, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练
    data_tensor = torch.from_numpy(data)
    
    print("\n训练中...")
    for epoch in range(1000):
        # 随机采样batch
        idx = np.random.choice(len(data), 32)
        batch = data_tensor[idx]
        
        # 训练步
        loss = model.training_loss(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 采样
    print("\n采样中...")
    model.eval()
    with torch.no_grad():
        samples = model.sample(shape=(1000, 2))
    
    samples = samples.numpy()
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始数据
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.3, s=1)
    axes[0].set_title('原始数据')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    
    # 采样
    axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    axes[1].set_title('SMLD采样')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('smld_demo.png', dpi=150)
    plt.close()
    
    print("\n可视化已保存到 smld_demo.png")
    
    return model


def compute_score_field():
    """计算并可视化分数场"""
    
    model = ScoreNet(input_dim=2, hidden_dim=128)
    model.eval()
    
    # 创建网格
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    points = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis=1)).float()
    
    # 计算分数
    t = torch.ones(len(points), 1) * 0.5  # 中间时间步
    
    with torch.no_grad():
        scores = model(points, t)
    
    scores = scores.numpy()
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.quiver(points[:, 0], points[:, 1], scores[:, 0], scores[:, 1])
    plt.scatter([0], [0], c='red', s=100, marker='*')
    plt.title('分数场 (指向高密度区域)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig('score_field.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    model = demo_smmld()
    compute_score_field()
```

---

## 6. 应用场景

### 6.1 图像生成

| 应用 | 说明 |
|------|------|
| **无条件生成** | 从噪声生成图像 |
| **条件生成** | 类标签条件生成 |
| **超分辨率** | 低分辨率到高分辨率 |
| **修复** | Image Inpainting |

### 6.2 其他应用

| 应用 | 说明 |
|------|------|
| **分子生成** | 药物分子设计 |
| **音频生成** | 语音合成 |
| **点云生成** | 3D形状生成 |

### 6.3 代码

```python
# 图像生成示例
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("models/stable-diffusion")
image = pipeline("a cat sitting on a chair").images[0]
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **训练稳定** | 分数匹配损失不涉及归一化常数 |
| **灵活** | 可处理任意noise schedule |
| **理论基础好** | 与Langevin动力学联系紧密 |
| **采样多样** | Langevin有内在随机性 |

### 7.2 缺点

| 缺点 | 说明 | 缓解 |
|------|------|------|
| **采样慢** | 需要多步迭代 | 加速采样 |
| **计算大** | 每步需要网络前向 | 参数共享 |
| **模式崩溃** | 可能生成单一模式 | 退火采样 |

### 7.3 对比

| ��法 | 训练 | 采样 | 质量 |
|------|------|------|------|
| SMLD | 分数匹配 | 100-2000步 | 高 |
| DDPM | 去噪重建 | 1000步 | 高 |
| GAN | 对抗 | 1步 | 中 |
| VAE | 重建 | 1步 | 中 |

---

## 8. 常见问题与易错点

### 8.1 问题1：分数归一化

**问题**：分数范数不稳定

**解决**：使用相对分数而非绝对
$$\hat{s}_\theta = s_\theta / (1 - \bar{\alpha}_t)$$

### 8.2 问题2：采样质量差

**问题**：采样结果模糊

**解决**：使用退火Langevin
```python
# 从高噪声到低噪声
for sigma in [1.0, 0.5, 0.2, 0.1]:
    langevin_steps(x, sigma=sigma)
```

### 8.3 问题3：模式单一

**问题**：生成多样性差

**解决**：增加Langevin噪声或使用混合策略

---

## 9. 学习总结

### 9.1 核心要点

1. **分数匹配**：学习 $\nabla_x \log p(x)$ 而非 $p(x)$
2. **噪声条件**：通过条件化简化学习
3. **Langevin采样**：基于分数的MCMC

### 9.2 关键公式

$$\nabla_x \log p(x_t) \approx \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

$$L = \mathbb{E}[ \| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \|^2 ]$$

### 9.3 学习路径

概率基础 → 分数匹配 → SMLD → DDPM → 加速采样

---

## 10. 练习题

### 10.1 基础题

1. 解释为什么学习分数而不是概率密度
2. 分析Langevin采样的每步含义

### 10.2 进阶题

3. 实现自己的数据分布的SMLD
4. 比较SMLD和DDPM的差异

### 10.3 答案

<details>
<summary>答案1</summary>

分数的优势：
1. 不需要归一化常数
2. 梯度有明确方向
3. Langevin可直接使用

</details>

<details>
<summary>答案2</summary>

Langevin每步：
1. score方向移动
2. 添加随机噪声探索
3. 逐步收敛到高密度区域

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段

1. 理解分数概念
2. 理解扩散过程
3. 实现基础SMLD

### 11.2 第二阶段

1. 理解Langevin采样
2. 实现退火采样
3. 调参实践

### 11.3 第三阶段

1. 学习加速采样
2. 阅读相关论文
3. 项目实践

---

## 12. 可视化与结果理解

```python
def visualize_sampling_process():
    """可视化采样过程"""
    
    import matplotlib
    matplotlib.use('Agg')
    
    # 初始化噪声
    x = torch.randn(100, 2) * 3
    
    # 记录每步
    history = [x.clone()]
    
    for step in range(50):
        score = score_net(x, t)
        x = x + 0.01 * score + math.sqrt(0.02) * torch.randn_like(x)
        history.append(x.clone())
    
    # 可视化
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    
    for i, step in enumerate(range(0, 50, 5)):
        ax = axes[i // 10, i % 10]
        h = history[step]
        ax.scatter(h[:, 0], h[:, 1], s=1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Step {step}')
    
    plt.tight_layout()
    plt.show()
```

---

## 13. 模型评估

### 13.1 评估指标

| 指标 | 说明 |
|------|------|
| FID | 图像质量 |
| IS | 多样性 |
| Precision/Recall | 模式覆盖 |

### 13.2 代码

```python
from torch_fid import FIDScore

fid = FIDScore()
score = fid(real_samples, generated_samples)
print(f"FID: {score}")
```

---

## 14. 进阶内容

### 14.1 与DDPM关系

SMLD和DDPM本质上是等价的：

**DDPM损失**：
$$L = \mathbb{E}[\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$$

**SMLD损失**：
$$L = \mathbb{E}[\| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \|^2]$$

关系：$\epsilon_\theta = -s_\theta \cdot \sqrt{1-\bar{\alpha}_t}$

### 14.2 加速采样

1. **DDIM**：确定性采样
2. **DPM-Solver**：解ODE
3. **PNDM**： predictor-corrector

### 14.3 推荐资源

- Score Matching (Hyvarinen, 2005)
- Generative Modeling by Score Matching (Song & Ermon, 2019)
- SMLD (Song & Ermon, 2021)

---

**文档结束**

*参考论文：Generative Modeling by Score Matching (Song & Ermon, 2019), Score-Based Generative Modeling (Song & Ermon, 2021)*

## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 7. 调库实现（Python + 完整代码 + 注释）

以下是使用主流框架实现SMLD的代码：

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 数据准备
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
train_set, test_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,2))
    def forward(self, x): return self.net(x)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
for epoch in range(50):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        crit(model(bx), by).backward()
        opt.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SMLDNet(nn.Module):
    def __init__(self, dim_in=20, dim_h=64, dim_out=2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim_in, dim_h), nn.Linear(dim_h, dim_out)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

def train(model, X, y, epochs=100, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"Ep{ep+1} loss={loss.item():.4f}")

m = SMLDNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```
