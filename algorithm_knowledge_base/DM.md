# DM 学习文档

> DM (Diffusion Model) 扩散模型是一种基于_score-based_生成模型，通过逐步加噪和去噪过程学习数据分布，已成为当前最强的图像生成模型之一。

---

## 1. 算法基础认知

### 一句话定义
DM 通过学习数据到噪声的正向扩散过程和噪声到数据的逆向去噪过程，能够生成高质量、多样化的图像和数据。

### 直觉类比
想象一滴墨水滴入水中：
- **正向过程**：墨水逐渐扩散，最终均匀分布在整个水杯（完全变成噪声）
- **逆向过程**：从浑浊的水中逆向"溯源"，逐渐恢复出清晰的墨水位置

DM正是学习这个"逆向恢复"的过程，从而可以从随机噪声生成逼真图像。

### 历史背景
- 2015年，Sohl-Dickstein等人提出扩散模型
- 2020年，Ho等人提出DDPM，简化了训练目标
- 2022年，Stable Diffusion将DM与latent空间结合，实现高效图像生成

### 算法定位
- **类型**：生成模型 / 深度学习
- **输出**：与训练数据同分布的新样本
- **模型类型**：UNet + 时间步嵌入

### 前置知识
- 神经网络基础（UNet架构）
- 高斯分布
- VAE/GAN基础（对比理解）

---

## 2. 核心原理

### 2.1 核心思想
DM的核心思想是**两阶段过程**：

1. **正向扩散（Forward Process）**：逐步向数据添加噪声，直到变成纯高斯噪声
   $$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$

2. **逆向去噪（Reverse Process）**：学习从噪声恢复到数据
   $$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t), \sigma_t^2 \mathbf{I})$$

关键insight：**预测噪声而非直接重建**

### 2.2 工作流程
```
数据x_0 → x_1 → x_2 → ... → x_T (噪声)
                           ↓
                      学习逆向过程
                           ↓
噪声x_T ← x_{T-1} ← ... ← x_0 (生成)
```

### 2.3 关键概念
- **噪声调度（Noise Schedule）**：$\beta_t$ 的递增序列
- **时间步嵌入（Time Embedding）**：将t映射到向量
- **Score函数**：$\nabla_x \log p(x)$，或等价的噪声预测
- **DDPM**：简化版的简化扩散概率模型

### 2.4 几何直观
```
┌──────────────────────────────────────────────────────┐
│              正向vs逆向过程                          │
│                                                      │
│  x_0 (清晰图) → x_1 → x_2 → ... → x_T (纯噪声)     │
│     ↑         │         │              │               │
│     │ 正向    │   β递增  │         β_T≈1            │
│     │ q(x_t|x_{t-1})                             │
│                                                      │
│  x_T (噪声)  ← x_{T-1} ← ... ← x_0 (生成图)       │
│     ↓         │         │              │               │
│     │ 逆向    │  学习   │      学习p_θ             │
│     │ p_θ(x_{t-1}|x_t)                            │
└──────────────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|----------|
| $x_0$ | 原始数据 | $\mathbb{R}^{C\times H\times W}$ |
| $x_t$ | t时刻的加噪数据 | 同上 |
| $\beta_t$ | 第t步的噪声方差 | scalar |
| $\alpha_t$ | $1-\beta_t$ | scalar |
| $\bar{\alpha}_t$ | $\prod_{i=1}^t \alpha_i$ | scalar |
| $\epsilon$ | 噪声，$\epsilon \sim \mathcal{N}(0,\mathbf{I})$ | 同上 |
| $\epsilon_\theta(x_t,t)$ | 网络预测的噪声 | 同上 |
| $T$ | 总扩散步数 | 1000 |

### 3.2 正向扩散过程
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$

可等价写为：
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

这意味着我们可以直接获取任意时刻t的加噪数据，无需递归。

### 3.3 逆向去噪过程
目标是学习 $p_\theta(x_{t-1}|x_t)$：

$$\mathcal{L}_{simple} = \mathbb{E}_{t,x_0,\epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2]$$

简化目标：**神经网络直接预测添加的噪声**！

### 3.4 DDPM训练目标
$$\mathcal{L}(\theta) = \mathbb{E}_{t,x_0,\epsilon}\left[||\epsilon - \epsilon_\theta(x_t, t)||^2\right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

### 3.5 生成过程（采样）
从纯噪声 $x_T \sim \mathcal{N}(0,\mathbf{I})$ 开始：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)) + \sqrt{\beta_t}\mathbf{z}$$

其中 $\mathbf{z} \sim \mathcal{N}(0,\mathbf{I})$（最后一步可省略）。

---

## 4. 训练过程讲解

### 4.1 数据预处理
- 图像归一化到 [-1,1]
- 确保数据可以被UNet处理

### 4.2 参数初始化
- UNet权重：Xavier初始化
- 时间嵌入：随机初始化

### 4.3 迭代过程

```python
"""
DM (Diffusion Model) 完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def get_noise_schedule(num_timesteps=1000, schedule='linear'):
    """噪声调度"""
    if schedule == 'linear':
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
    elif schedule == 'cosine':
        steps = torch.arange(num_timesteps + 1)
        alpha_hat = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_hat = alpha_hat / alpha_hat[0]
        betas = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    
    return betas, alphas, alpha_hat


class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class UNet(nn.Module):
    """UNet for Diffusion Model"""
    
    def __init__(self, in_channels=3, out_channels=3, time_dim=128, base_channels=64):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels, time_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels*2, time_dim)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4, time_dim)
        self.enc4 = ResidualBlock(base_channels*4, base_channels*4, time_dim)
        
        # Middle
        self.middle = ResidualBlock(base_channels*4, base_channels*4, time_dim)
        
        # Decoder
        self.dec1 = ResidualBlock(base_channels*4, base_channels*4, time_dim)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2, time_dim)
        self.dec3 = ResidualBlock(base_channels*2, base_channels, time_dim)
        self.dec4 = ResidualBlock(base_channels, base_channels, time_dim)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)
        
        # Middle
        m = self.middle(e4, t_emb)
        
        # Decoder
        d1 = self.dec1(self.upsample(m), t_emb)
        d2 = self.dec2(self.upsample(d1), t_emb)
        d3 = self.dec3(self.upsample(d2), t_emb)
        d4 = self.dec4(self.upsample(d3), t_emb)
        
        return self.out_conv(d4)


class DiffusionModel(nn.Module):
    """扩散模型"""
    
    def __init__(self, in_channels=3, out_channels=3, time_dim=128, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        self.unet = UNet(in_channels, out_channels, time_dim)
        betas, alphas, alpha_hat = get_noise_schedule(num_timesteps)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_hat', alpha_hat)
    
    def forward_diffusion(self, x0, t):
        """正向加噪"""
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        noise = torch.randn_like(x0)
        return torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise, noise
    
    def predict_noise(self, xt, t):
        """预测噪声"""
        return self.unet(xt, t)
    
    def training_step(self, x0):
        """训练步骤"""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device)
        
        # 正向加噪
        xt, noise = self.forward_diffusion(x0, t)
        
        # 预测噪声
        pred_noise = self.predict_noise(xt, t)
        
        # MSE损失
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, shape):
        """采样/生成"""
        device = next(self.parameters()).device
        xT = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device)
            
            pred_noise = self.unet(xT, t_batch)
            
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alpha_hat[t]
            beta_t = self.betas[t]
            alpha_hat_prev = self.alpha_hat[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # 计算均值
            mean = (xT - pred_noise * torch.sqrt(1 - alpha_hat_t) * beta_t.sqrt()) / alpha_t.sqrt()
            
            if t > 0:
                noise = torch.randn_like(xT)
                xT = mean + torch.sqrt(beta_t) * noise
            else:
                xT = mean
        
        return xT


def train_diffusion(model, dataloader, epochs=100, lr=1e-4, device='cuda'):
    """训练"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            
            loss = model.training_step(x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]:.6f}")
    
    return losses


# 使用示例
# model = DiffusionModel(in_channels=3, out_channels=3, num_timesteps=1000)
# model = train_diffusion(model, dataloader)

# 生成
# generated_images = model.sample(shape=(4, 3, 32, 32))
```

### 4.4 收敛条件
- 损失趋于稳定（<0.01量级）
- 生成样本质量主观评估

### 4.5 超参数

| 超参数 | 作用 | 推荐范围 |
|--------|------|----------|
| T | 扩散步数 | 1000 |
| base_channels | UNet基础通道数 | 64~128 |
| learning rate | 学习率 | 1e-4~1e-3 |
| batch_size | 批量 | 16~32 |

---

## 5. 应用场景

### 5.1 典型应用
- **图像生成**：逼真人脸、艺术图像
- **图像编辑**：inpainting、超分辨率
- **文本生成图像**：Stable Diffusion
- **视频生成**

### 5.2 适用数据
- 高维复杂数据
- 需要高质量生成
- 多样性要求高

### 5.3 不适用
- 实时性要求高（T步骤多）
- 计算资源有限

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 训练稳定 | 不mode collapse |
| 高质量生成 | 超越GAN |
| 多样性强 | 捕获完整分布 |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 慢（多步） | DDIM加速 |
| 内存大 | latent DM |

---

## 7. 调库实现

```python
"""
使用diffusers库
"""
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained("ddpm-cifar10-32")
image = pipeline(num_inference_steps=50).images[0]
```

---

## 8. 手工实现

核心简化版：
```python
"""
DM 核心简化实现
"""

import torch
import torch.nn as nn

class SimpleDiffusion:
    """简化扩散模型"""
    
    def __init__(self, num_timesteps=1000):
        self.T = num_timesteps
        # 简化的噪声调度
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas)
    
    def add_noise(self, x0, t, noise):
        """正向过程"""
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        return torch.sqrt(alpha_hat) * x0 + torch.sqrt(1 - alpha_hat) * noise
    
    def denoise_step(self, xt, pred_noise, t):
        """逆向过程单步"""
        alpha = self.alphas[t]
        alpha_hat = self.alpha_hat[t]
        beta = self.betas[t]
        
        # 均值
        mean = (xt - pred_noise * torch.sqrt(1 - alpha_hat) / alpha.sqrt()) / alpha.sqrt()
        
        if t > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(beta) * noise
        return mean
    
    def sample(self, unet, shape):
        """完整采样"""
        xt = torch.randn(shape)
        for t in reversed(range(self.T)):
            pred = unet(xt, t)
            xt = self.denoise_step(xt, pred, t)
        return xt
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt

def show_diffusion_process(model, x0, save_path='diffusion.png'):
    """展示扩散过程"""
    fig, axes = plt.subplots(1, 11, figsize=(20, 2))
    
    # 原始
    axes[0].imshow(x0[0].permute(1,2,0)/2+0.5)
    axes[0].set_title('x0')
    axes[0].axis('off')
    
    # 加噪
    for i, t in enumerate([0, 100, 200, 400, 600, 800, 900, 950, 980, 999]):
        xt, _ = model.forward_diffusion(x0, t)
        axes[i+1].imshow(xt[0].permute(1,2,0).clamp(-1,1)/2+0.5)
        axes[i+1].set_title(f't={t}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
```

---

## 10. 评估

```python
def evaluate_fid(real_images, generated_images):
    """FID分数评估"""
    from pytorch_fid.fid_score import calculate_fid
    return calculate_fid(real_images, generated_images)
```

---

## 11. 常见问题

### 11.1 生成质量差
- 增加T步数
- 检查UNet架构

### 11.2 训练慢
- 减小图像分辨率
- 使用DDIM采样

---

## 12. 总结

### 核心要点
1. **两阶段**：正向加噪+逆向去噪
2. **预测噪声**：简化训练目标
3. **UNet**：时间条件网络
4. **多步生成**：高质量但慢

### 算法链
```
DM → DDPM → Stable Diffusion → SDXL
    ↓
  DDIM（加速）
```

---

## 13. 练习题

**习题1**：正向加噪公式

<details>
<summary>答案</summary>

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

</details>

**习题2**：为什么预测噪声而非重建？

<details>
<summary>答案</summary>

预测噪声目标更简单（与t相关），训练更稳定。

</details>

---

## 14. 学习路径

- 初级：理解原理，运行demo
- 中级：实现UNet，调参
- 高级：Latent DM，ControlNet

### 推荐资源
- **论文**：Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
- **代码**：https://github.com/hojonathanho/diffusion