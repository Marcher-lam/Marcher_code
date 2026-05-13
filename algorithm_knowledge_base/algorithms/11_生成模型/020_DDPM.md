# 去噪扩散概率模型 (DDPM) 学习文档

> 从纯噪声逐步去噪还原图像，开启生成模型新时代。

> 来源线索：本节内容根据原书中关于"扩散模型"的相关章节（第11章11.1-11.2节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** DDPM 通过学习逆转逐步加噪的过程，从纯高斯噪声中生成高质量图像。

**直觉类比：** 想象一滴墨水滴入清水：墨水逐渐扩散，最终均匀分布（正向扩散）。DDPM 要做的，就是学会这个过程的"倒带"——从均匀的浅色水中逐步还原出墨滴。当然，这违反热力学第二定律，但神经网络可以通过学习来近似这个逆过程。

**历史背景：** DDPM 由 Ho、Jain 和 Abbeel 在 2020 年提出（论文 "Denoising Diffusion Probabilistic Models"），奠定了扩散模型在图像生成领域的基础。后续 DDIM、Stable Diffusion、DALL-E 等模型均基于此框架。

**算法定位：** 生成模型、潜变量模型，属于基于分数的生成模型家族。

**前置知识：** 高斯分布、马尔可夫链、变分推断、U-Net 架构、PyTorch。

---

## 2. 核心原理

### 核心思想

DDPM 包含两个过程：
1. **正向扩散**：逐步向数据添加高斯噪声，经过 T 步后数据变为纯噪声
2. **反向去噪**：训练神经网络（通常为 U-Net）学习每一步如何去除少量噪声

关键发现：不需要学习完整的逆向分布，只需要学习**预测每一步添加的噪声**即可。

### 工作流程

1. **训练**：
   - 取一张图像 $x_0$
   - 随机选择时间步 $t$
   - 用公式直接计算加噪后的 $x_t$（利用重参数化技巧）
   - 训练网络预测添加的噪声 $\epsilon$
   - 最小化预测噪声与真实噪声的均方误差

2. **采样（生成）**：
   - 从标准高斯分布采样 $x_T \sim \mathcal{N}(0, I)$
   - 从 $t=T$ 到 $t=1$，每步用训练好的网络预测噪声
   - 减去预测噪声，加入少量随机性得到 $x_{t-1}$
   - 最终得到 $x_0$（生成的图像）

### 关键概念

- **噪声调度（Noise Schedule）**：控制每步添加噪声量的一系列参数 $\beta_1, \ldots, \beta_T$
- **重参数化技巧**：允许直接从 $x_0$ 采样任意时刻 $t$ 的 $x_t$，无需逐步加噪
- **U-Net**：编码器-解码器架构，带跳跃连接，用于预测噪声
- **时间嵌入**：将时间步 $t$ 编码为向量注入网络，告知网络当前处于哪个去噪阶段

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $x_0$ | 原始图像 |
| $x_t$ | 第 $t$ 步加噪后的图像 |
| $T$ | 总扩散步数（通常 1000） |
| $\beta_t$ | 第 $t$ 步的噪声方差（递增，从 $10^{-4}$ 到 0.02） |
| $\alpha_t = 1 - \beta_t$ | |
| $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ | 累积乘积 |
| $\epsilon_\theta$ | 噪声预测网络（U-Net） |

### 正向扩散

每步加噪定义为马尔可夫转移：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**关键性质：可以直接从 $x_0$ 采样任意 $x_t$**

利用重参数化技巧递推展开：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这是 DDPM 训练效率的关键——不需要逐步加噪，一步到位。

### 反向过程

反向过程用神经网络参数化：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**均值 $\mu_\theta$ 的推导：**

通过变分推断，最优均值可以表示为：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

因此，训练网络 $\epsilon_\theta$ 来预测噪声 $\epsilon$ 即可。

### 损失函数

最终的简化损失函数：

$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

直观理解：网络预测的噪声与实际添加的噪声之间的均方误差。

### 采样公式

从 $x_t$ 得到 $x_{t-1}$：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

其中 $z \sim \mathcal{N}(0, I)$（当 $t > 1$），$\sigma_t = \sqrt{\beta_t}$。

---

## 4. 训练过程讲解

### 数据预处理
- 图像归一化到 [-1, 1]
- 可选：随机水平翻转等数据增强

### 参数初始化
- U-Net 权重：标准正态初始化
- 噪声调度：线性或余弦调度

### 迭代过程
1. 采样 batch 的图像 $x_0$
2. 随机采样时间步 $t \sim \text{Uniform}(\{1, \ldots, T\})$
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I)$
4. 计算加噪样本 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. 预测噪声 $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
6. 计算 MSE 损失 $\|\epsilon - \hat{\epsilon}\|^2$
7. 反向传播，更新 $\epsilon_\theta$ 参数

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| T (扩散步数) | 100 ~ 2000 | 1000 |
| $\beta_1$ | 1e-5 ~ 1e-3 | 1e-4 |
| $\beta_T$ | 0.01 ~ 0.1 | 0.02 |
| 学习率 | 1e-5 ~ 2e-4 | 2e-4 |
| batch_size | 16 ~ 128 | 64 |

---

## 5. 应用场景

### 1. 图像生成
从噪声生成高质量图像（人脸、风景、艺术画等）。Stable Diffusion、DALL-E 等的核心就是扩散模型。

### 2. 图像修复与编辑
利用条件扩散模型进行图像修复（inpainting）、外扩（outpainting）和局部编辑。

### 3. 图像超分辨率
将低分辨率图像逐步提升到高分辨率（SR3、SRDiff）。

### 4. 音频/视频生成
扩散模型已成功应用于语音合成、音乐生成和视频生成。

---

## 6. 优缺点分析

### 优点
1. **训练稳定**：只需优化简单的 MSE 损失，无需 GAN 的对抗平衡
2. **生成质量高**：超越 GAN 的 FID 分数，细节更丰富
3. **多样性好**：不存在 GAN 的模式崩塌问题
4. **理论基础扎实**：有变分推断和随机微分方程的理论支撑

### 缺点
1. **采样速度慢**：需要 1000 步逐步去噪，比 GAN 慢几个数量级。缓解：DDIM（50步）、一致性模型（1步）
2. **计算成本高**：每次前向传播都要运行完整的 U-Net
3. **训练时间长**：需要大量数据和时间训练

### 与同类对比

| 特性 | DDPM | GAN | VAE |
|------|------|-----|-----|
| 生成质量 | 极高 | 高 | 中 |
| 训练稳定性 | 高 | 低 | 高 |
| 采样速度 | 慢 | 快 | 快 |
| 多样性 | 高 | 中 | 高 |
| 可控性 | 高 | 低 | 中 |

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 噪声调度
T = 200
beta_start, beta_end = 1e-4, 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 简化的 U-Net 用于 MNIST
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        # 编码器
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())
        # 解码器
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128+128, 64, 4, stride=2, padding=1), nn.ReLU())
        self.dec1 = nn.Conv2d(64+64, 1, 3, padding=1)  # 输出预测噪声

        self.time_proj = nn.Linear(128, 256)  # 时间嵌入投影

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        # 编码
        e1 = self.enc1(x)                            # (B, 64, 28, 28)
        e2 = self.enc2(e1)                            # (B, 128, 14, 14)
        e3 = self.enc3(e2)                            # (B, 256, 7, 7)
        e3 = e3 + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        # 解码 + 跳跃连接
        d3 = self.dec3(e3)                            # (B, 128, 14, 14)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))   # (B, 64, 28, 28)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))   # (B, 1, 28, 28)
        return d1

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

sqrt_alphas_cp = sqrt_alphas_cumprod.to(device)
sqrt_one_cp = sqrt_one_minus_alphas_cumprod.to(device)

for epoch in range(10):
    total_loss = 0
    for x_0, _ in loader:
        x_0 = x_0.to(device)
        B = x_0.size(0)
        # 随机采样时间步
        t = torch.randint(0, T, (B,), device=device)
        # 采样噪声
        noise = torch.randn_like(x_0)
        # 直接计算 x_t
        x_t = sqrt_alphas_cp[t].view(-1,1,1,1) * x_0 + sqrt_one_cp[t].view(-1,1,1,1) * noise
        # 预测噪声
        pred_noise = model(x_t, t)
        # MSE 损失
        loss = nn.functional.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {total_loss/len(loader):.4f}')
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleDDPM:
    """从零实现 DDPM 的核心采样逻辑"""

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # 后验分布的方差
        self.posterior_variance = self.betas * (1.0 - np.concatenate([[1], self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """正向扩散：从 x_0 直接采样 x_t"""
        if noise is None:
            noise = np.random.randn(*x_0.shape)
        return (self.sqrt_alphas_cumprod[t] * x_0 +
                self.sqrt_one_minus_alphas_cumprod[t] * noise)

    def p_sample(self, model_predict, x_t, t):
        """单步反向去噪：从 x_t 得到 x_{t-1}"""
        # 计算均值
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        # 均值: mu = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
        mu = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_t * model_predict)
        if t > 0:
            # 加入少量噪声（除了最后一步）
            noise = np.random.randn(*x_t.shape)
            sigma = np.sqrt(self.posterior_variance[t])
            return mu + sigma * noise
        return mu

    def sample_loop(self, noise_predict_fn, shape):
        """完整的采样循环：从纯噪声生成图像"""
        # 从纯噪声开始
        x = np.random.randn(*shape)
        for t in reversed(range(self.T)):
            # 预测噪声
            pred_noise = noise_predict_fn(x, t)
            # 单步去噪
            x = self.p_sample(pred_noise, x, t)
        return x

# 测试
if __name__ == '__main__':
    ddpm = SimpleDDPM(T=100)
    x_0 = np.array([1.0, 0.5, -0.3])  # 模拟一个数据点
    # 测试正向扩散
    x_t = ddpm.q_sample(x_0, t=50)
    print(f"x_0: {x_0}")
    print(f"x_50 (加噪后): {x_t}")
    # 测试采样（使用虚拟噪声预测函数）
    def dummy_predict(x, t):
        return np.random.randn(*x.shape) * 0.1
    samples = ddpm.sample_loop(dummy_predict, shape=(3,))
    print(f"采样结果: {samples}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_diffusion_process():
    """可视化扩散过程的逐步加噪和去噪"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 模拟正向扩散（逐步加噪）
    steps = [0, 50, 100, 150, 199]
    np.random.seed(42)
    x_0 = np.random.randn(28, 28)

    for i, t in enumerate(steps):
        alpha_t = np.exp(-t * 0.01)
        noise = np.random.randn(28, 28)
        x_t = alpha_t * x_0 + np.sqrt(1 - alpha_t) * noise
        axes[0, i].imshow(x_t, cmap='gray', vmin=-3, vmax=3)
        axes[0, i].set_title(f't={t}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('正向加噪', fontsize=12)

    # 模拟反向去噪
    for i, t in enumerate(reversed(steps)):
        quality = 1 - t / 200
        x_approx = quality * x_0 + (1 - quality) * np.random.randn(28, 28) * 0.3
        axes[1, 4-i].imshow(x_approx, cmap='gray', vmin=-3, vmax=3)
        axes[1, 4-i].set_title(f't={t}')
        axes[1, 4-i].axis('off')
    axes[1, 0].set_ylabel('反向去噪', fontsize=12)

    plt.suptitle('DDPM 正向加噪与反向去噪过程', fontsize=14)
    plt.tight_layout()
    plt.savefig('ddpm_process.png', dpi=100, bbox_inches='tight')
    plt.show()

visualize_diffusion_process()
```

---

## 10. 模型评估

### 评估指标

1. **FID (Frechet Inception Distance)**：越低越好，衡量真实分布与生成分布的距离
2. **IS (Inception Score)**：越高越好，衡量生成质量和多样性
3. **视觉评估**：直观检查生成图像的质量、多样性和一致性

---

## 11. 常见问题与易错点

### 数据层面
1. **噪声调度选择不当**
   - 现象：生成图像模糊或包含伪影
   - 解决：使用余弦调度（improved DDPM）替代线性调度

### 模型层面
1. **时间嵌入维度不足**
   - 现象：模型无法区分不同去噪阶段
   - 解决：使用正弦位置编码或更宽的时间 MLP

2. **采样步数过少**
   - 现象：生成质量明显下降
   - 解决：使用 DDIM 加速，或一致性模型

### 调参层面
1. **T 设置过大导致采样慢**
   - T=1000 质量最好但采样需要 1000 步前向传播
   - 解决：训练用 T=1000，采样用 DDIM 只需 50 步

---

## 12. 学习总结

DDPM 的核心是两个过程：正向扩散逐步加噪将数据变为纯噪声，反向去噪训练神经网络预测每步的噪声。关键公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 允许直接从 $x_0$ 计算任意时刻的加噪样本。损失函数简化为预测噪声与真实噪声的 MSE，训练简单稳定。

### 关键公式
1. 正向加噪：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
2. 损失函数：$L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$
3. 采样：$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)) + \sigma_t z$

---

## 13. 练习题与思考题

### 基础题

**题1：** 为什么 DDPM 的正向过程使用马尔可夫链？直接一步加噪到纯噪声是否可行？

**参考答案：**
逐步加噪形成马尔可夫链使得每步的变化很小，保证反向过程每步可以用高斯分布近似（当 $\beta_t$ 很小时，后验 $q(x_{t-1}|x_t, x_0)$ 接近高斯）。如果一步到位加噪，反向过程需要一步从纯噪声恢复原图，这远比逐步去噪困难，因为一步映射的信息瓶颈太严重。

**题2：** DDPM 的损失函数为什么是预测噪声而不是直接预测原图？

**参考答案：**
数学上，预测噪声与预测 $x_0$ 是等价的（给定 $x_t$ 和噪声 $\epsilon$，可以推出 $x_0$）。但预测噪声在实践中效果更好，因为：(1) 噪声的分布在所有时间步都是标准高斯，目标分布一致；(2) 网络只需学习"减去什么"而非"恢复什么"，学习目标更简单。

### 进阶题

**题3：** DDIM 相比 DDPM 的关键改进是什么？为什么 DDIM 能用更少的步数采样？

**参考答案：**
DDIM 将反向过程的马尔可夫假设推广为非马尔可夫过程。关键发现：DDPM 的训练目标只依赖于边际分布 $q(x_t|x_0)$，而不依赖于联合分布 $q(x_{1:T}|x_0)$。因此可以定义不同的反向过程（不一定是高斯随机过程），DDIM 选择了确定性的反向过程（去掉采样中的随机噪声项），使得可以用更少的步数确定性采样。

### 开放思考题

**题4：** 扩散模型相比 GAN 的核心优势是什么？在哪些场景下 GAN 仍然更适合？

**参考答案思路：**
核心优势在于训练稳定性和生成多样性。扩散模型不存在模式崩塌，损失函数是简单的 MSE。但在实时生成场景（如游戏、视频会议）中，GAN 的单步前向传播远快于扩散模型的多步迭代。混合方法（如使用 GAN 作为扩散模型的蒸馏加速器，或一致性模型）可能是未来方向。

---

## 14. 学习路径建议

### 前置算法
- VAE（理解变分下界和重参数化技巧）
- U-Net（理解编码器-解码器架构）
- 高斯分布和马尔可夫链

### 平行算法
- GAN（对比理解不同生成模型的优缺点）
- Score-based Models（DDPM 的理论前身）

### 进阶算法
- DDIM（加速采样）
- Stable Diffusion（潜空间扩散）
- CLIP（文本条件引导）
- 一致性模型（一步生成）

### 推荐资源
1. **论文**：Ho et al., "Denoising Diffusion Probabilistic Models" (2020) — DDPM 原始论文
2. **论文**：Song et al., "Denoising Diffusion Implicit Models" (DDIM)
3. **博客**：Lilian Weng 的 "What are Diffusion Models?" — 扩散模型的系统综述
