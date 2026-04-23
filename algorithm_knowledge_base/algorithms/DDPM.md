# DDPM 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

去噪扩散概率模型（DDPM）通过逐步加噪破坏数据，再学习逆向去噪过程来生成样本，前向过程将数据逐渐转换为纯噪声，逆向过程学习如何从噪声恢复数据。

### 1.2 直觉类比

DDPM像一个经验丰富的修复师：假设你有一幅名画，每一代修复师都会在上代的基础上做微小改动。DDPM学习的是这个过程的"逆过程"——如何从一幅破损的画逐步还原出原貌。

### 1.3 历史背景

DDPM由Sohl-Dickstein等人在2015年提出，Ho等人在2020年简化了训练目标，提出了DDPM。2020-2021年，DDPM在图像生成质量上超越了GAN，成为主流生成模型。

### 1.4 算法定位

- 类型：无监督学习
- 输出：图像生成
- 模型类别：生成模型（扩散模型）

### 1.5 前置知识

- 概率论基础
- 神经网络基础
- U-Net架构
- 条件概率

## 2. 核心原理

### 2.1 核心思想

DDPM由两个过程组成：
- **前向过程（Forward Process）**：逐步添加噪声，直到图像变为纯噪声
- **逆向过程（Reverse Process）**：学习如何从噪声逐步恢复图像

### 2.2 工作流程

1. 前向：为数据$x_0$添加T步噪声，得到$x_1, x_2, ..., x_T$
2. 训练：训练网络预测噪声或$x_t$
3. 采样：从纯噪声$x_T$开始，迭代去噪得到$\hat{x}_0$

### 2.3 关键概念

- **噪声调度（Noise Schedule）**：控制每步添加的噪声量
- **时间嵌入（Time Embedding）**：让网络知道当前在哪个时间步
- **U-Net**：常用的去噪网络结构

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x_0$ | 原始数据 |
| $x_t$ | t时刻的带噪数据 |
| $\beta_t$ | t时刻的噪声方差 |
| $\alpha_t$ | 1 - \beta_t |
| $\epsilon$ | 噪声 |
| $\epsilon_\theta$ | 预测噪声的网络 |

### 3.2 前向过程

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

闭合形式（直接采样任意t）：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

其中$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$

### 3.3 逆向过程

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

### 3.4 训练目标

简化目标（预测噪声）：
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

### 3.5 采样过程

从$x_T \sim \mathcal{N}(0, I)$开始：
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

## 4. 训练过程讲解

### 4.1 数据预处理

- 归一化到[-1, 1]
- 数据增强（随机水平翻转）

### 4.2 网络架构

- U-Net with attention
- Time embedding
- 残差连接

### 4.3 超参数

- T: 1000（扩散步数）
- beta_start: 0.0001
- beta_end: 0.02
- learning_rate: 0.0001
- batch_size: 128

### 4.4 训练流程

```python
for epoch in range(n_epochs):
    for x0 in dataloader:
        t = random.randint(1, T)
        epsilon = random_noise()
        
        x_t = sqrt(bar_alpha[t]) * x0 + sqrt(1 - bar_alpha[t]) * epsilon
        
        loss = ||epsilon - model(x_t, t)||^2
        loss.backward()
        optimizer.step()
```

## 5. 应用场景

### 5.1 应用

- 高质量图像生成
- 文本到图像（DALL-E, Stable Diffusion）
- 图像编辑
- 超分辨率

### 5.2 适用

- 需要高质量生成
- 多样性要求高

### 5.3 不适用

- 实时生成（太慢）
- 资源受限场景

## 6. 优缺点分析

### 6.1 优点

- 生成质量极高
- 训练稳定
- 多样性好
- 可控性强

### 6.2 缺点

- 采样慢（需要多步）
- 计算量大

### 6.3 对比

| 特性 | DDPM | GAN | VAE |
|------|------|-----|-----|
| 生成质量 | 极高 | 高 | 中 |
| 训练稳定 | 极高 | 低 | 高 |
| 采样速度 | 慢 | 快 | 快 |
| 多样性 | 高 | 中 | 高 |

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch numpy matplotlib
```

### 7.2 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + self.time_emb(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_channels * 4),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        channels = [base_channels * m for m in channel_mults]
        
        self.downs = nn.ModuleList()
        for i in range(len(channels)):
            self.downs.append(nn.ModuleList([
                ResidualBlock(channels[i], channels[i], base_channels * 4),
                ResidualBlock(channels[i], channels[i], base_channels * 4)
            ]))
            if i < len(channels) - 1:
                self.downs.append(nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1))
        
        self.mid = ResidualBlock(channels[-1], channels[-1], base_channels * 4)
        
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(channels[i], channels[i-1], 3, stride=2, padding=1, output_padding=1))
            self.ups.append(nn.ModuleList([
                ResidualBlock(channels[i-1] * 2, channels[i-1], base_channels * 4),
                ResidualBlock(channels[i-1] * 2, channels[i-1], base_channels * 4)
            ]))
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = self.conv_in(x)
        
        hs = []
        for i in range(0, len(self.downs), 2):
            for j in range(2):
                x = self.downs[i][j](x, t_emb)
            hs.append(x)
            x = self.downs[i+1](x)
        
        x = self.mid(x, t_emb)
        
        for i in range(0, len(self.ups), 3):
            x = self.ups[i](x)
            x = torch.cat([x, hs.pop()], dim=1)
            x = self.ups[i+1][0](x, t_emb)
            x = self.ups[i+1][1](x, t_emb)
        
        return self.conv_out(x)


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.model = UNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
    def forward_process(self, x0, t):
        """前向加噪"""
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.alpha_bar[t] ** 0.5
        sqrt_one_minus = (1 - self.alpha_bar[t]) ** 0.5
        
        return sqrt_alpha_bar[:, None, None, None] * x0 + sqrt_one_minus[:, None, None, None] * noise, noise
    
    def training_loss(self, x0):
        """训练损失"""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        
        x_t, noise = self.forward_process(x0, t)
        
        noise_pred = self.model(x_t, t)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sampling(self, n_samples, shape):
        """采样"""
        x = torch.randn(n_samples, *shape, device=self.device)
        
        for t in reversed(range(self.T)):
            z = torch.randn_like(x) if t > 0 else 0
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            noise_pred = self.model(x, torch.full((n_samples,), t, device=self.device))
            
            mean = (1 / alpha_t ** 0.5) * (x - beta_t / (1 - alpha_bar_t) ** 0.5 * noise_pred)
            std = (beta_t ** 0.5)
            
            x = mean + std * z
        
        return x


def train_simple_ddpm():
    """简单训练示例"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    ddpm = DDPM(T, beta_start, beta_end, device)
    
    print(f"参数量: {sum(p.numel() for p in ddpm.model.parameters())}")
    
    # 模拟训练
    print("开始训练（演示）...")
    
    # 生成样本公司
    n_samples = 16
    samples = ddpm.sampling(n_samples, (3, 32, 32))
    
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2
    samples = np.clip(samples, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i].transpose(1, 2, 0))
            ax.axis('off')
    plt.suptitle('DDPM Generated Samples')
    plt.tight_layout()
    plt.savefig('ddpm_samples.png', dpi=150)
    plt.show()
    
    return ddpm


if __name__ == "__main__":
    ddpm = train_simple_ddpm()
```

### 7.3 结果示例

DDPM训练后生成的图像质量极高，可与GAN媲美。

## 8. 手工代码实现

### 8.1 简化DDPM

```python
import numpy as np

class SimpleDDPM:
    """简化版DDPM（示意）"""
    
    def __init__(self, T=100):
        self.T = T
        # 简化实现...
```

## 9. 可视化

### 9.1 去噪过程

展示从纯噪声逐步去噪的过程。

## 10. 模���评估

### 10.1 指标

- FID（Frechet Inception Distance）
- IS（Inception Score）

## 11. 常见问题

### 11.1 采样慢

- 解决：DDIM加速采样

### 11.2 T选择

- 通常1000步效果好

## 12. 学习总结

### 12.1 核心

- 前向加噪 + 逆向去噪
- 逐步生成

### 12.2 公式

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

### 12.3 联系

- 前序：VAE, GAN → DDPM → Stable Diffusion

## 13. 练习题与思考题

### 13.1 基础

1. DDPM的两个过程？

答案：前向加噪、逆向去噪


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：本算法的核心机制是什么？请简述其工作原理。

**答案与解析**：

**步骤1**：识别问题类型
根据算法定义，这是一个[类型：监督/无监督/生成/强化学习]任务。

**步骤2**：应用核心公式
$$核心公式 = [具体公式]$$
该公式的意义是[解释公式含义]。

**步骤3**：验证答案
代入具体数据验证：[计算过程]
最终结果符合预期，说明理解正确。

**答案**：算法的核心是通过[机制]实现[目标]，属于[算法类别]。

---

#### 练习2：手动计算

**问题**：给定数据[X=具体值, y=具体值]，手动计算[算法名]的[参数/结果]。

**答案与解析**：

**步骤1**：准备数据
$X = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$  
$y = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}$

**步骤2**：应用算法步骤
根据[算法名]的定义，计算第一步：
$$第一步 = [具体公式代入] = [数值]$$

**步骤3**：继续计算
$$第二步 = [公式] = [结果]$$

**步骤4**：得到最终答案
$$最终结果 = [综合计算] = [具体数值]$$

**验证**：将结果带回原式检验 $[验证过程]$，确认正确。

---

#### 思考题：改进分析

**问题**：本算法在[特定场景]下存在哪些局限性？请提出改进方案。

**答案与解析**：

**局限性分析**：
1. **局限性1**：[具体表现]，原因是[原因解释]
2. **局限性2**：[具体表现]，原因是[原因解释]

**改进方案对比**：

| 改进方法 | 原理 | 优势 | 代价 |
|---------|------|------|------|
| 方法A | [原理] | [好处] | [额外成本] |
| 方法B | [原理] | [好处] | [额外成本] |
| 方法C | [原理] | [好处] | [额外成本] |

**推荐方案**：在实际应用中优先考虑[方法A]，因为[理由]。
## 14. 学习路径建议

前置：神经网络 → GAN → DDPM → Stable Diffusion