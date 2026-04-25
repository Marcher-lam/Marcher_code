# DAE（去噪自编码器）学习文档

## 1. 算法基础认知

去噪自编码器（Denoising Autoencoder, DAE）由 Vincent 等人于 2008 年提出，是 AE 的重要变体。核心思想：人为地向输入数据添加噪声（破坏），然后训练模型从损坏的输入中还原出干净的原始数据。这种"先破坏再修复"的训练策略迫使网络学习到更鲁棒、更有意义的特征表示。

## 2. 核心原理

DAE 的训练流程：

1. **加噪**：对原始输入 $x$ 施加随机破坏，得到损坏版本 $\tilde{x} \sim q_D(\tilde{x}|x)$
2. **编码**：将 $\tilde{x}$ 编码为隐表示 $z = f_\theta(\tilde{x})$
3. **解码**：从 $z$ 重构原始干净输入 $\hat{x} = g_\phi(z)$
4. **损失**：计算 $\hat{x}$ 与原始 $x$（而非 $\tilde{x}$）之间的重构误差

为什么有效？因为要从损坏输入恢复原始数据，网络必须理解数据的内在结构和模式，不能简单记忆输入。这类似于人类视觉系统——即使图片部分被遮挡，我们仍能识别内容。

常见噪声类型：
- **高斯噪声**：$\tilde{x} = x + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$
- **掩码噪声**：随机将部分像素置零
- **椒盐噪声**：随机将部分像素设为最大或最小值

## 3. 数学公式与推导

DAE 的优化目标：

$$\mathcal{L}(\theta, \phi) = -\mathbb{E}_{x \sim p_{data}} \mathbb{E}_{\tilde{x} \sim q_D(\tilde{x}|x)} [\log p_\phi(x | f_\theta(\tilde{x}))]$$

对于均方误差：

$$\mathcal{L} = \mathbb{E}_{x} \mathbb{E}_{\tilde{x}|x} \left[\|x - g_\phi(f_\theta(\tilde{x}))\|^2\right]$$

从score matching 角度理解：DAE 实际上在学习数据分布的得分函数（score function）。当使用高斯噪声时，DAE 的重构目标等价于估计 $\sigma^2 \nabla_x \log p(x)$，即指向数据高密度区域的方向。

## 4. 训练过程讲解

1. 取一个 batch 的干净数据 $x$
2. 对 $x$ 施加噪声得到 $\tilde{x}$（如随机遮蔽 30% 的像素）
3. 将 $\tilde{x}$ 输入 DAE，前向传播得到重构 $\hat{x}$
4. 计算 $\hat{x}$ 与原始 $x$ 之间的损失
5. 反向传播更新参数

噪声水平是关键超参数：太低则训练接近普通 AE，太高则信息丢失过多难以恢复。通常掩码比例为 0.2~0.5。

## 5. 应用场景

- **图像去噪**：去除照片中的噪点、伪影
- **特征学习**：学习比 AE 更鲁棒的表示
- **预训练**：作为深度网络的逐层预训练策略
- **数据填充**：恢复缺失的数据值
- **语音增强**：去除语音信号中的噪声

## 6. 优缺点分析

**优点：**
- 学习到的特征表示更鲁棒，泛化能力更强
- 训练简单，只需在 AE 基础上加噪即可
- 可作为无监督预训练方法

**缺点：**
- 噪声类型和强度需要手动选择
- 仍然不是生成模型（隐空间无概率约束）
- 重构质量受噪声水平影响大

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def add_noise(self, x, noise_factor=0.3):
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0.0, 1.0)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = DenoisingAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(30):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        noisy_x = model.add_noise(batch_x, noise_factor=0.3)
        x_recon, z = model(noisy_x)
        loss = criterion(x_recon, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class DenoisingAutoencoderNumpy:
    def __init__(self, input_dim, hidden_dim, latent_dim, lr=0.001):
        self.lr = lr
        he = lambda n_in, n_out: np.sqrt(2.0 / n_in)
        self.W1 = np.random.randn(input_dim, hidden_dim) * he(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * he(hidden_dim, latent_dim)
        self.b2 = np.zeros(latent_dim)
        self.W3 = np.random.randn(latent_dim, hidden_dim) * he(latent_dim, hidden_dim)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.randn(hidden_dim, input_dim) * he(hidden_dim, input_dim)
        self.b4 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def add_gaussian_noise(self, x, scale=0.3):
        return np.clip(x + np.random.randn(*x.shape) * scale, 0, 1)

    def add_mask_noise(self, x, mask_ratio=0.3):
        mask = (np.random.rand(*x.shape) > mask_ratio).astype(float)
        return x * mask

    def forward(self, x):
        self.x_clean = x
        self.x_noisy = self.add_mask_noise(x)
        self.h1 = self.relu(self.x_noisy @ self.W1 + self.b1)
        self.z = self.relu(self.h1 @ self.W2 + self.b2)
        self.h2 = self.relu(self.z @ self.W3 + self.b3)
        self.out = self.sigmoid(self.h2 @ self.W4 + self.b4)
        return self.out

    def backward(self):
        m = self.x_clean.shape[0]
        d = (self.out - self.x_clean) * self.out * (1 - self.out)
        dW4 = self.h2.T @ d / m
        db4 = d.mean(axis=0)
        dh2 = d @ self.W4.T * self.relu_grad(self.h2)
        dW3 = self.z.T @ dh2 / m
        db3 = dh2.mean(axis=0)
        dz = dh2 @ self.W3.T * self.relu_grad(self.z)
        dW2 = self.h1.T @ dz / m
        db2 = dz.mean(axis=0)
        dh1 = dz @ self.W2.T * self.relu_grad(self.h1)
        dW1 = self.x_noisy.T @ dh1 / m
        db1 = dh1.mean(axis=0)
        for p, g in [(self.W4, dW4), (self.b4, db4), (self.W3, dW3), (self.b3, db3),
                      (self.W2, dW2), (self.b2, db2), (self.W1, dW1), (self.b1, db1)]:
            p -= self.lr * g

    def train_step(self, x):
        out = self.forward(x)
        loss = np.mean((out - self.x_clean) ** 2)
        self.backward()
        return loss
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    sample = test_dataset.data[:5].float().view(5, -1) / 255.0
    noisy = model.add_noise(sample, noise_factor=0.5)
    recon, _ = model(noisy)

fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for i in range(5):
    axes[0, i].imshow(sample[i].view(28, 28), cmap='gray')
    axes[0, i].set_title('原始')
    axes[0, i].axis('off')
    axes[1, i].imshow(noisy[i].view(28, 28), cmap='gray')
    axes[1, i].set_title('加噪')
    axes[1, i].axis('off')
    axes[2, i].imshow(recon[i].view(28, 28), cmap='gray')
    axes[2, i].set_title('去噪')
    axes[2, i].axis('off')
plt.savefig('dae_denoising.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **去噪效果**：比较加噪-去噪后的 PSNR（峰值信噪比）和 SSIM（结构相似度）
- **重构误差**：MSE 越低越好
- **下游任务**：用编码器提取特征做分类，与普通 AE 对比准确率

## 11. 常见问题与易错点

- **噪声过强**：噪声水平太高会导致信息不可逆丢失，模型无法恢复
- **噪声过弱**：模型退化为普通 AE，失去鲁棒性优势
- **损失目标混淆**：重构目标应该是干净输入 $x$，而非加噪后的 $\tilde{x}$
- **噪声类型不匹配**：训练时用的噪声类型应与实际应用场景的噪声类型一致

## 12. 学习总结

DAE 通过"先破坏再修复"的策略，迫使模型学习数据的本质结构。其核心价值在于：简单有效、鲁棒性强、无需标签。DAE 的思想深刻影响了后续的扩散模型——DDPM 本质上就是对多个噪声级别的 DAE 的系统性扩展。

## 13. 练习题与思考题（含答案）

**Q1：DAE 与 AE 的关键区别是什么？**

A1：AE 的输入和目标是同一个 $x$；DAE 的输入是加噪后的 $\tilde{x}$，目标是原始干净的 $x$。DAE 通过加噪迫使网络学习更鲁棒的表示。

**Q2：为什么 DAE 比普通 AE 学到的特征更好？**

A2：因为 DAE 需要从不完整/受损的输入中恢复原始数据，网络必须理解数据的内在模式和结构，而不能简单记忆像素级信息。

**Q3：DAE 与扩散模型有什么关系？**

A3：扩散模型可以看作是多级 DAE 的扩展。DDPM 在不同噪声级别训练去噪网络，本质上是在做 T 步的渐进式去噪，而 DAE 只做一步去噪。

## 14. 学习路径建议

1. 掌握普通 AE 的原理和实现
2. 理解 DAE 的噪声注入机制和鲁棒性原理
3. 学习 Masked AE（如 MAE）等进阶变体
4. 扩展到扩散模型（DDPM），理解多级去噪
5. 了解 score matching 与 DAE 的理论联系
