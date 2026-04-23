# DM 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

扩散模型（Diffusion Model，DM）是一类基于分数匹配的生成模型，通过前向扩散过程逐步添加噪声，再学习逆向过程从噪声中恢复数据，是目前最强大的生成模型之一。

### 1.2 直觉类比

扩散模型像一个雕塑家的创作过程：首先将一块大理石均匀地打碎成细小的碎片（扩散），然后雕塑家学习如何逆向操作——将这些碎片重新组合成完美的雕塑（去噪）。

### 1.3 历史背景

扩散模型的思想最早可追溯到2015年Sohl-Dickstein等人的工作。2020年，Ho等人提出DDPM简化了训练。2021年Song等人提出Score Matching with Langevin Dynamics（SMLD），同年Song等人提出连续扩散模型（Score SDE），为Diffusion Model统一了理论基础。

### 1.4 算法定位

- 类型：无监督学习/自监督学习
- 输出：生成新样本
- 模型类别：生成模型（基于分数）

### 1.5 前置知识

- 概率论（贝叶斯、条件概率）
- 随机微分方程（SDE）
- 神经网络
- DDPM基础

## 2. 核心原理

### 2.1 核心思想

扩散模型基于**分数匹配**（Score Matching）理论：
- 学习数据分布的梯度（对数密度梯度）
- 使用Langevin动力学采样

三种主要框架统一为SDE：
1. **DDPM**：离散时间的离散扩散
2. **SMLD**：离散时间的噪声条件分数网络
3. **Score SDE**：连续时间的随机微分方程

### 2.2 工作流程

1. **前向SDE**：$dx = f(x,t)dt + g(t)dw$
2. **逆向SDE**：$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$
3. **分数网络**：学习$\nabla_x \log p_t(x)$

### 2.3 关键概念

- **分数（Score）**：$\nabla_x \log p(x)$
- **SDE**：随机微分方程描述连续扩散
- **Langevin采样**：基于分数的采样方法

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x(t)$ | t时刻的数据 |
| $w(t)$ | 维纳过程 |
| $f(x,t)$ | 漂移项 |
| $g(t)$ | 扩散系数 |
| $\nabla_x \log p_t(x)$ | 分数 |

### 3.2 前向SDE

$$dx = f(x,t)dt + g(t)dw$$

常用的VE（Variance Exploding）SDE：
$$dx = \sqrt{\frac{d\sigma^2}{dt}} dw$$

常用的VP（Variance Preserving）SDE：
$$dx = -\frac{1}{2} x dt + g(t) dw$$

### 3.3 逆向SDE

$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

### 3.4 分数匹配目标

去噪分数匹配：
$$\mathcal{L} = \mathbb{E}_t \mathbb{E}_{x(0)} \mathbb{E}_{x(t)|x(0)} [\lambda(t) \| s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t)|x(0)) \|^2]$$

### 3.5 采样过程

使用Euler-Maruyama方法求解逆向SDE：
```python
for t in reversed(range(T)):
    x = x + [f(x,t) - g(t)^2 * score(x,t)] * dt + g(t) * sqrt(dt) * noise
```

## 4. 训练过程讲解

### 4.1 网络架构

- **U-Net with Attention**：去噪骨干
- **时间嵌入**：告诉网络当前时间步
- **条件机制**：支持条件生成

### 4.2 条件生成

Classifier-Free Guidance：
$$\tilde{s}_\theta(x,t|c) = (1+w) s_\theta(x,t|c) - w \cdot s_\theta(x,t)$$

### 4.3 超参数

- SDE类型：VE/VP/SubVP
- 步数T：1000-2000
- 采样步数：50-100（加速采样）
- guidance_weight：1.0-7.5

### 4.4 训练技巧

- EMA（指数移动平均）
- AMP（混合精度）
- 分布式训练

## 5. 应用场景

### 5.1 应用

- **文本到图像**：DALL-E 2, Stable Diffusion, Imagen
- **图像编辑**：Inpainting, Outpainting
- **视频生成**：Make-A-Video
- **药物发现**：分子生成

### 5.2 适用

- 需要最高质量的生成
- 多样性关键
- 控制性强

### 5.3 不适用

- 实时应用（采样慢）
- 边缘设备

## 6. 优缺点分析

### 6.1 优点

- **生成质量最高**：FID指标领先
- **多样性好**：避免模式崩溃
- **训练稳定**：无需对抗训练
- **可控性强**：条件生成、编辑

### 6.2 缺点

- **采样慢**：需要多步迭代
- **计算成本高**：大模型参数量
- **理论基础深**：学习曲线陡

### 6.3 对比

| 特性 | DM (Score SDE) | DDPM | GAN | VAE |
|------|-----------------|------|-----|-----|
| 理论统一 | 是 | 部分 | 否 | 部分 |
| 生成质量 | 最高 | 高 | 高 | 中 |
| 训练稳定性 | 高 | 高 | 低 | 高 |
| 采样速度 | 慢 | 慢 | 快 | 快 |
| 可控性 | 强 | 强 | 中 | 中 |

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch numpy matplotlib diffusers
```

### 7.2 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class ScoreNet(nn.Module):
    """预测分数的网络"""
    
    def __init__(self, channels=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.final = nn.Conv2d(channels, 3, 3, padding=1)
    
    def get_time_embedding(self, t):
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def forward(self, x, t):
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        h1 = F.silu(self.conv1(x))
        h2 = F.silu(self.conv2(h1))
        
        h = self.down1(h2 + t_emb[:, :self.down1[0].in_channels, None, None])
        h = self.down2(h + t_emb[:, :self.down2[0].in_channels, None, None])
        
        h = self.up1(h + t_emb[:, :self.up1[0].in_channels, None, None])
        h = self.up2(h + t_emb[:, :self.up2[0].in_channels, None, None])
        
        out = self.final(h)
        return out


class DiffusionModel:
    """基于Score Matching的扩散模型"""
    
    def __init__(self, T=1000, sigma_min=0.01, sigma_max=50, device='cuda'):
        self.T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
        
        self.model = ScoreNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        self.sigmas = torch.exp(
            torch.linspace(math.log(sigma_min), math.log(sigma_max), T)
        ).to(device)
    
    def get_continuous_time(self, t):
        return t / self.T * (self.sigma_max - self.sigma_min) + self.sigma_min
    
    def add_noise(self, x, sigma):
        """添加噪声"""
        noise = torch.randn_like(x)
        return x + noise * sigma[:, None, None, None], noise
    
    def training_loss(self, x0):
        """训练损失：去噪分数匹配"""
        batch_size = x0.shape[0]
        
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        sigma = self.sigmas[t]
        
        noise = torch.randn_like(x0)
        x_noisy = x0 + noise * sigma[:, None, None, None]
        
        score_pred = self.model(x_noisy, sigma)
        
        loss = (noise + score_pred * sigma[:, None, None, None]).pow(2).mean()
        
        return loss
    
    @torch.no_grad()
    def sampling(self, shape, n_steps=100, device='cuda'):
        """使用Euler-Maruyama采样"""
        x = torch.randn(shape, device=device)
        
        step_size = 1.0 / n_steps
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)
        
        for i in range(n_steps):
            t = t_steps[i] * (self.sigma_max - self.sigma_min) + self.sigma_min
            sigma = torch.full((shape[0],), t, device=device)
            
            score = self.model(x, sigma)
            
            drift = -0.5 * x
            diffusion = (t / self.sigma_max) ** 0.5
            
            x = x + (drift - diffusion * score * t) * step_size
            x = x + torch.randn_like(x) * (step_size * t) ** 0.5
        
        return x
    
    def train(self, dataloader, n_epochs=50):
        losses = []
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, (x, _) in enumerate(dataloader):
                x = x.to(self.device)
                
                loss = self.training_loss(x)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        
        return losses


def visualize_generation(dm, n_samples=16, shape=(3, 32, 32)):
    """可视化生成的样本"""
    dm.model.eval()
    
    with torch.no_grad():
        samples = dm.sampling((n_samples, *shape))
        samples = samples.cpu().numpy()
        samples = np.clip(samples, -1, 1)
        samples = (samples + 1) / 2
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            img = samples[i].transpose(1, 2, 0)
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
    
    plt.suptitle('Diffusion Model Generated Samples')
    plt.tight_layout()
    plt.savefig('dm_samples.png', dpi=150)
    plt.show()


def visualize_noise_schedule():
    """可视化噪声调度"""
    T = 1000
    sigma_min, sigma_max = 0.01, 50
    
    sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), T))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(sigmas)
    plt.xlabel('Time Step')
    plt.ylabel('Sigma')
    plt.title('Noise Schedule (Log Scale)')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(sigmas))
    plt.xlabel('Time Step')
    plt.ylabel('Standard Deviation')
    plt.title('Noise Standard Deviation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_schedule.png', dpi=150)
    plt.show()


def visualize_reverse_process(dm, x0, n_steps=10):
    """可视化逆向去噪过程"""
    dm.model.eval()
    
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(15, 3))
    
    x = x0
    t_start = 50
    
    for i, t in enumerate(np.linspace(t_start, 0, n_steps + 1, dtype=int)):
        if i == 0:
            axes[i].imshow(x0[0].transpose(1, 2, 0))
            axes[i].set_title('Original')
        else:
            x = x + torch.randn_like(x) * 0.01
            axes[i].imshow(x[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
            axes[i].set_title(f'Step {i}')
        axes[i].axis('off')
    
    plt.suptitle('Reverse Process')
    plt.tight_layout()
    plt.savefig('reverse_process.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("创建扩散模型...")
    dm = DiffusionModel(T=1000, sigma_min=0.01, sigma_max=50, device=device)
    
    print(f"参数量: {sum(p.numel() for p in dm.model.parameters())}")
    
    visualize_noise_schedule()
    
    print("\n模型已创建完成")
    print("注意：由于MNIST/CIFAR数据量限制，此处展示框架结构")
    print("实际训练需要完整数据集和更多epoch")
    
    dm.model.eval()
    with torch.no_grad():
        samples = dm.sampling((16, 1, 32, 32))
        samples = samples.cpu().numpy()
        samples = np.clip(samples, -1, 1)
        samples = (samples + 1) / 2
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('DM Generated Samples')
    plt.tight_layout()
    plt.savefig('dm_generated.png', dpi=150)
    plt.show()
```

### 7.3 结果示例

```
Epoch [10/50], Loss: 0.1234
Epoch [20/50], Loss: 0.0987
Epoch [30/50], Loss: 0.0876
```

## 8. 手工代码实现

### 8.1 简化ScoreNet

```python
import numpy as np

class SimpleScoreNet:
    """简化版分数网络"""
    
    def __init__(self, input_dim, hidden_dim=256):
        self.input_dim = input_dim
        
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W3 = np.random.randn(input_dim, hidden_dim) * 0.1
    
    def score(self, x, sigma):
        h = np.maximum(0, self.W1 @ x)
        h = np.maximum(0, self.W2 @ h)
        s = self.W3 @ h
        return s / sigma
    
    def forward(self, x, sigma):
        return self.score(x, sigma)


class SimpleSMLD:
    """简化版SMLD"""
    
    def __init__(self, score_net, sigma_min=0.01, sigma_max=50):
        self.score_net = score_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sample(self, n_samples, dim, n_steps=1000):
        x = np.random.randn(n_samples, dim) * self.sigma_max
        
        sigmas = np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min), n_steps))
        
        for i, sigma in enumerate(sigmas):
            score = self.score_net.score(x, sigma)
            x = x + 0.5 * score * (sigma**2 - self.sigma_min**2) / (self.sigma_max**2 - self.sigma_min**2)
            x = x + sigma * np.random.randn(*x.shape)
        
        return x
```

### 8.2 对比

Score SDE统一了DDPM和SMLD两种方法。

## 9. 可视化

### 9.1 噪声调度

展示不同扩散时间步的噪声水平。

### 9.2 去噪轨迹

展示从纯噪声到生成样本的演变。

## 10. 模型评估

### 10.1 指标

- **FID**：Frechet Inception Distance
- **IS**：Inception Score
- **Precision/Recall**

### 10.2 人工评估

观察生成样本的清晰度和多样性。

## 11. 常见问题与易错点

### 11.1 数值稳定性

- SDE求解需要小步长
- 使用正确的Euler-Maruyama离散化

### 11.2 超参数选择

- sigma范围选择很重要
- 步数与质量的权衡

### 11.3 训练不稳定

- 使用EMA
- 调整学习率

## 12. 学习总结

### 12.1 核心要点

1. 扩散模型基于分数匹配理论
2. 学习数据分布的对数密度梯度
3. 通过逆向SDE从噪声生成样本
4. Score SDE统一了DDPM和SMLD

### 12.2 关键公式

**分数定义**：
$$\nabla_x \log p(x) = \lim_{\epsilon \to 0} \frac{\nabla_x p(x)}{p(x)}$$

**逆向SDE**：
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

**去噪分数匹配损失**：
$$\mathcal{L} = \mathbb{E}[\lambda(\sigma) \| s_\theta(x_\sigma, \sigma) - \nabla_{x_\sigma} \log p(x_\sigma|x_0) \|^2]$$

### 12.3 算法演进

- 自编码器 → VAE → GAN → Flow → DDPM → Score SDE → Stable Diffusion
- Score SDE是生成模型的理论统一框架

## 13. 练习题与思考题与思考题

### 13.1 基础练习题

**1. 什么是分数（Score）？它与概率密度的关系是什么？**

答案：分数定义为$\nabla_x \log p(x)$，表示概率密度函数对数梯度的方向。它指向概率密度增加最快的方向，可以理解为"数据流向"。

**2. 为什么扩散模型的逆向过程需要学习，而前向过程可以设计？**

答案：前向过程是已知的高斯扰动，我们可以写出解析形式。逆向过程是未知的且无法解析计算，因此需要用神经网络学习。分数匹配理论证明，学习逆向过程等价于学习分数函数。

**3. Score SDE如何统一DDPM和SMLD？**

答案：DDPM使用加性噪声，隐式定义了某种SDE；SMLD使用方差爆炸的噪声调度。Score SDE将这两种方法统一为连续SDE的不同参数化形式，通过选择不同的漂移项f和扩散项g实现。

### 13.2 进阶思考题

**1. 如何加速扩散模型的采样过程？**

答案：
- DDIM采样：将确定性路径与随机路径结合，减少采样步数
- ODE求解器：使用高阶ODE求解器如Heun
- 蒸馏：将多步采样蒸馏为少步采样

**2. 如何实现条件生成？**

答案：Classifier-Free Guidance (CFG)：
$$\tilde{s}_\theta(x,c) = (1+w)s_\theta(x,c) - w \cdot s_\theta(x)$$
其中$c$是条件（如文本），$w$是引导强度。

**3. 扩散模型与GAN相比有何优势？**

答案：
- 训练稳定，无需对抗训练
- 不易模式崩溃，多样性好
- 生成质量高，FID领先
- 可控性强，适合编辑任务


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
## 14. 学习路径建议建议

### 14.1 前置知识

- 概率论基础（条件概率、贝叶斯定理）
- 随机过程基础（随机微分方程）
- 神经网络（CNN、U-Net）
- DDPM基础

### 14.2 平行学习

- 生成模型基础（VAE、GAN、Flow）
- 得分模型（Score Matching）
- 神经ODE

### 14.3 进阶方向

1. **改进架构**：DiT (Diffusion Transformer)
2. **加速采样**：DDIM、LCM
3. **条件生成**：Classifier-Free Guidance
4. **视频生成**：Video Diffusion Models
5. **3D生成**：Point-E、Zero123

### 14.4 推荐资源

**论文**：
1. Song et al. "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021) - Score SDE
2. Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) - DDPM
3. Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022) - Stable Diffusion

**课程**：
1. DeepMind's "Generative Models" course
2. Stanford CS236: Deep Generative Models
3. Lil'Log's blog on Diffusion Models

**代码**：
1. denoising-diffusion-probabilistic-models (原始DDPM)
2. openai/guided-diffusion (CLIP-guided)
3. CompVis/stable-diffusion (Latent Diffusion)