# WGAN 学习文档

## 1. 算法基础认知

### 1.1 研究背景

WGAN（Wasserstein Generative Adversarial Network）由Martin Arjovsky等人在2017年提出，旨在解决传统GAN训练中的核心问题：模式崩塌（mode collapse）和训练不稳定。传统GAN使用JS散度（Jenson-Shannon Divergence）来衡量生成分布与真实分布的差异，但当两个分布不重叠时，JS散度会变成常数，导致梯度消失。WGAN引入Wasserstein距离（也称Earth-Mover距离）作为新的度量，提供有意义且平滑的梯度信号。

### 1.2 核心思想

WGAN的核心创新在于用Wasserstein-1距离替代JS散度。作为生成对抗网络的重要改进，它解决了梯度消失问题，使得训练更加稳定，并为生成质量提供更好的优化信号。其判别器不再执行二分类任务，而是学习一个满足1-Lipschitz约束的函数来估算Wasserstein距离。

### 1.3 技术定位

WGAN是GAN发展史上的里程碑，后续的WGAN-GP、SNGAN、BigGAN等都在此基础上改进。在图像生成、数据增强和分布建模等任务中广泛应用。

---

## 2. 核心原理

### 2.1 Wasserstein距离定义

Wasserstein-1距离（Earth-Mover距离）衡量将生成分布转换为真实分布所需的"工作量"：

$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[||x - y||]$$

其中$\Pi(P_r, P_g)$表示所有以$P_r$和$P_g$为边缘分布的联合分布集合。当两个分布完全相同时，$W(P_r, P_g) = 0$。

### 2.2 Kantorovich-Rubinstein对偶

使用对偶形式简化计算：

$$W(P_r, P_g) = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{y \sim P_g}[f(y)]$$

其中$f$是任意1-Lipschitz函数。判别器学习这个函数来估算距离。

### 2.3 Lipschitz约束实现

WGAN通过权重裁剪（Weight Clipping）强制实现Lipschitz约束：

$$W \leftarrow \text{clip}(W, -c, c)$$

通常$c = 0.01$。这确保判别器不会变得过于复杂。

### 2.4 目标函数

$$\min_G \max_D L(D, G) = \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim P_z}[D(G(z))]$$

生成器最小化这个距离，判别器最大化这个距离（估算这个距离）。

---

## 3. 数学公式与推导

### 3.1 损失函数

判别器（ critic ）损失：
$$L_D = \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim P_z}[D(G(z))]$$

生成器损失：
$$L_G = -\mathbb{E}_{z \sim P_z}[D(G(z))]$$

### 3.2 Lipschitz约束

判别器必须满足$||f||_L \leq 1$，即对于任意$x, y$：

$$|f(x) - f(y)| \leq ||x - y||$$

权重裁剪实现：
$$W = \text{clip}(W, -c, c)$$

### 3.3 训练稳定性

WGAN相比标准GAN的优势：

1. **连续可微**：即使分布不重叠，Wasserstein距离仍然连续
2. **有意义梯度**：提供有意义的梯度信号指导生成器更新
3. **模式检测**：能检测到模式崩塌

---

## 4. 训练过程讲解

### 4.1 训练步骤

```
算法：WGAN训练
输入：真实数据分布Pr，生成器G，判别器D，裁剪常数c=0.01
输出：训练后的生成器和判别器

1. 初始化：
   θ_G ~ N(0, 0.01²)
   θ_D ~ N(0, 0.01²)

2. For iteration in 1..num_iter：
   a. 采样小批量真实样本 {x_i} ~ Pr
   b. 采样噪声 {z_j} ~ N(0, I)
   c. 生成样本 {x_g} = G(z_j)
   d. 判别器损失：
       L_D = ΣD(x_i)/m - ΣD(x_g)/m
   e. 更新判别器θ_D
   f. 权重裁剪：θ_D ← clip(θ_D, -c, c)
   g. 采样新噪声 {z_k}
   h. 生成器损失：
       L_G = -ΣD(G(z_k))/m
   i. 更新生成器θ_G

3. 返回 G*, D*
```

### 4.2 超参数设置

| 参数 | 推荐值 |
|------|--------|
| 批大小 | 64 |
| 学习率 | 5e-5 |
| 裁剪常数c | 0.01 |
| n_critic | 5 |
| 迭代次数 | 100K+ |

### 4.3 训练技巧

1. **多步判别器**：每步生成器更新前，先更新n_critic次判别器
2. **RMSprop**：推荐使用RMSprop优化器
3. **无_batch_norm**：判别器不使用BatchNorm（权重裁剪已经约束网络）
4. **标签smooth**：可使用标签平滑

---

## 5. 应用场景

### 5.1 图像生成

- 人脸图像生成
- 场景图像生成
- 艺术风格图像

### 5.2 数据增强

- 生成稀缺类别样本
- 扩充训练数据集
- 解决类别不平衡

### 5.3 分布建模

- 复杂分布拟合
- 异常检测
- 对抗样本生成

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 训练稳定 | 解决梯度消失问题 |
| 质量较高 | 生成图像质量好 |
| 收敛性好 | 损失与生成质量相关 |
| 检测模式崩塌 | 能检测训练问题 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 权重裁剪 | 可能导致容量未充分利用 |
| 梯度问题 | 梯度裁剪也有梯度消失 |
| 训练慢 | 需要更多判别器更新 |
| 裁剪敏感 | 对裁剪值敏感 |

### 6.3 WGAN vs 标准GAN

| 方面 | 标准GAN | WGAN |
|------|--------|------|
| 距离度量 | JS散度 | Wasserstein距离 |
| 梯度 | 不连续 | 连续 |
| 判别器 | Sigmoid分类 | 近似Wasserstein |
| 稳定性 | 差 | 好 |

---

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
import glob
from PIL import Image


class Generator(nn.Module):
    """WGAN生成器"""
    
    def __init__(self, latent_dim=128, img_channels=3, features_g=64):
        super().__init__()
        
        self.init_size = 8
        
        self.init = nn.Sequential(
            nn.Linear(latent_dim, features_g * 4 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2),
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(features_g, img_channels, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, z):
        out = self.init(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    """WGAN判别器（Critic）"""
    
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 3, 1, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features_d, features_d * 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features_d * 2, features_d * 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features_d * 4, features_d * 8, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(features_d * 8, 1),
        )
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class WGAN:
    """
    WGAN: Wasserstein GAN
    Reference: https://arxiv.org/abs/1701.07875
    """
    
    def __init__(
        self,
        latent_dim=128,
        img_channels=3,
        img_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        clip_value=0.01,
        n_critic=5,
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        self.n_critic = n_critic
        
        self.generator = Generator(latent_dim, img_channels).to(device)
        self.discriminator = Discriminator(img_channels).to(device)
        
        self.opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5)
        self.opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
        
        self.losses = []
        print(f"WGAN initialized on {device}")
        
    def train_step(self, real_images):
        """单步训练"""
        
        batch_size = real_images.size(0)
        
        # 判别器更新
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        
        real_loss = self.discriminator(real_images).mean()
        fake_loss = self.discriminator(fake_images).mean()
        d_loss = real_loss - fake_loss
        
        self.opt_d.zero_grad()
        d_loss.backward(retain_graph=True)
        self.opt_d.step()
        
        # 权重裁剪
        for param in self.discriminator.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)
        
        # 生成器更新（每n_critic步更新一次）
        if len(self.losses) % self.n_critic == 0:
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(z)
            g_loss = -self.discriminator(fake_images).mean()
            
            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
            
            return d_loss.item(), g_loss.item()
        
        return d_loss.item(), 0.0
    
    def train(
        self,
        dataloader,
        num_iters=100000,
        log_interval=100,
        save_dir="wgan_results",
    ):
        """训练WGAN"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.generator.train()
        self.discriminator.train()
        
        iterator = iter(dataloader)
        
        for step in tqdm(range(num_iters)):
            try:
                real_images = next(iterator)[0].to(self.device)
            except StopIteration:
                iterator = iter(dataloader)
                real_images = next(iterator)[0].to(self.device)
            
            real_images = real_images * 2 - 1
            
            d_loss, g_loss = self.train_step(real_images)
            self.losses.append((d_loss, g_loss))
            
            if step % log_interval == 0:
                print(f"Step {step}, D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
                
            if step % 10000 == 0 and step > 0:
                self.generate_samples(step, save_dir)
                
        print("Training complete!")
        
    def generate_samples(self, step, save_dir):
        """生成样本"""
        
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(16, self.latent_dim, device=self.device)
            samples = self.generator(z)
            
        samples = (samples + 1) / 2
        
        grid = torchvision.utils.make_grid(samples, nrow=4)
        torchvision.utils.save_image(grid, os.path.join(save_dir, f"samples_{step}.png"))
        
        self.generator.train()


def main():
    """WGAN训练示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wgan = WGAN(
        latent_dim=128,
        img_channels=3,
        img_size=64,
        device=device,
        clip_value=0.01,
        n_critic=5,
    )
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    wgan.train(dataloader, num_iters=50000)


if __name__ == "__main__":
    main()
```

---

## 8. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleGenerator(nn.Module):
    """简化版生成器"""
    
    def __init__(self, latent_dim=100, hidden_dim=128, output_dim=784):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.tanh(self.fc3(x))
        return x


class SimpleDiscriminator(nn.Module):
    """简化版判别器（Critic）"""
    
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class WGANLite:
    """简化版WGAN实现"""
    
    def __init__(self, latent_dim=100, img_dim=784, device="cuda"):
        self.device = device
        self.latent_dim = latent_dim
        
        self.G = SimpleGenerator(latent_dim, 128, img_dim).to(device)
        self.D = SimpleDiscriminator(img_dim, 128).to(device)
        
        self.opt_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        
    def train_step(self, real_images):
        """单步训练"""
        
        batch_size = real_images.size(0)
        
        # 判别器
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.G(z)
        
        d_real = self.D(real_images).mean()
        d_fake = self.D(fake_images).mean()
        d_loss = d_real - d_fake
        
        self.opt_D.zero_grad()
        d_loss.backward(retain_graph=True)
        self.opt_D.step()
        
        # 权重裁剪
        for p in self.D.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        # 生成器
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.G(z)
        g_loss = -self.D(fake_images).mean()
        
        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate(self, num_samples):
        """生成样本"""
        
        self.G.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            samples = self.G(z)
            
        self.G.train()
        return samples


def main():
    """WGAN lite示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wgan = WGANLite(latent_dim=100, device=device)
    
    # 演示数据
    real = torch.randn(64, 784).to(device) * 2 - 1
    
    for step in range(100):
        d_loss, g_loss = wgan.train_step(real)
        
        if step % 20 == 0:
            print(f"Step {step}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
            
    generated = wgan.generate(4)
    print(f"Generated shape: {generated.shape}")


if __name__ == "__main__":
    main()
```

---

## 9. 可视化与结果理解

### 9.1 训练曲线

训练过程中，WGAN的损失应该呈现下降趋势并趋于收敛。判别器损失接近零时，表示能够有效区分真实样本和生成样本。

### 9.2 生成质量评估

生成的图像应该：具有清晰的结构、纹理自然、无明显伪影。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| FID | Frechet Inception Distance，越低越好 |
| IS | Inception Score，越高越好 |
| 人工评估 | 人工判断生成质量 |

---

## 11. 常见问题与易错点

### 11.1 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 生成器退化 | 裁剪过严 | 增大clip_value |
| 梯度消失 | 学习率太低 | 调大学习率 |
| 模式崩塌 | 判别器太弱 | 增加判别器容量 |

### 11.2 关键点

1. 必须使用权重裁剪
2. 判别器不使用BatchNorm
3. n_critic通常设为5

---

## 12. 学习总结

### 12.1 核心要点

WGAN的关键创新在于使用Wasserstein距离替代JS散度，配合权重裁剪实现Lipschitz约束，使得训练更加稳定。核心优点包括梯度连续可微、能够检测模式崩塌，以及损失值与生成质量相关。

### 12.2 后续发展

WGAN-GP通过梯度惩罚替代权重裁剪，解决了容量未充分利用的问题。SNGAN引入谱归一化进一步稳定训练。

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题

**1. WGAN使用什么距离来衡量生成分布与真实分布的差异？**
A. KL散度
B. JS散度
C. Wasserstein距离
D. 交叉熵

答案：C

**2. WGAN如何实现Lipschitz约束？**
A. Dropout
B. 权重裁剪
C. BatchNorm
D. 标签平滑

答案：B

**3. WGAN的判别器输出是什么？**
A. 概率值
B. Wasserstein距离估计
C. 二元分类
D. 特征向量

答案：B

### 13.2 简答题

**1. WGAN相比标准GAN的优势是什么？**

答：WGAN使用Wasserstein距离替代JS散度，即使生成分布与真实分布不重叠，仍能提供有意义的梯度信号，解决了训练不稳定和梯度消失的问题。同时，损失值与生成质量相关，可以作为训练进程的监控指标。

**2. WGAN权重裁剪的缺点是什么？**

答：权重裁剪强制约束网络参数，可能导致判别器容量无法充分利用，限制了其学习复杂函数的能力。当裁剪值过小时尤其明显。

---


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

需要掌握GAN基础和PyTorch深度学习框架。

### 14.2 学习路线

建议先学习标准GAN，然后理解Wasserstein距离理论，最后掌握WGAN实现。

### 14.3 进阶方向

可以学习WGAN-GP、SNGAN、BigGAN等后续改进算法。