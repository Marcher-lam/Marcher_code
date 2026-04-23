# DAE（去噪自编码器）学习文档

## 1. 算法基础认知

去噪自编码器（Denoising Autoencoder, DAE）由 Vincent 等人于 2008 年提出。核心思想：对输入数据**人为添加噪声**，然后训练网络从噪声版本中**恢复原始干净数据**。通过学习去除噪声的过程，网络被迫捕获数据的鲁棒特征表示。

DAE 是自编码器的重要变体，也是后续扩散模型（DDPM）的理论基础之一。

## 2. 核心原理

DAE 的训练流程：

1. **噪声注入**：将干净输入 $x$ 通过某种 corruption 过程得到噪声版本 $\tilde{x} \sim q(\tilde{x}|x)$
2. **编码-解码**：将 $\tilde{x}$ 输入编码器得到隐表示 $h$，再解码得到重建 $\hat{x}$
3. **目标**：最小化 $\hat{x}$ 与**原始干净 $x$**（而非 $\tilde{x}$）之间的差异

关键洞察：网络必须理解数据的内在结构才能从噪声中恢复信号，因此学到的表示比普通 AE 更鲁棒。

## 3. 数学公式与推导

**噪声分布**（常用随机掩码噪声）：

$$q(\tilde{x}|x) : \tilde{x}_j = \begin{cases} 0 & \text{以概率 } p \\ x_j & \text{以概率 } 1-p \end{cases}$$

也可用高斯噪声：$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$

**损失函数**：

$$L_{DAE} = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{\tilde{x} \sim q(\tilde{x}|x)} \left[ L(x, g_\phi(f_\theta(\tilde{x}))) \right]$$

其中 $L$ 通常为 MSE 或交叉熵。

**与得分匹配的联系**（理论意义）：

DAE 的最优解等价于学习数据的**得分函数（score function）** $\nabla_x \log p(x)$。具体地，当使用高斯噪声时：

$$s_\theta(x) \approx \frac{1}{\sigma^2}(g_\theta(\tilde{x}) - \tilde{x})$$

这为后续的扩散模型提供了理论基础。

## 4. 训练过程讲解

1. 取一个干净样本 $x$
2. 随机添加噪声生成 $\tilde{x}$（如随机置零部分像素、添加高斯噪声）
3. 将 $\tilde{x}$ 输入编码器：$h = f_\theta(\tilde{x})$
4. 解码器重建：$\hat{x} = g_\phi(h)$
5. 计算损失 $L = \|x - \hat{x}\|^2$（注意目标是干净 $x$）
6. 反向传播更新参数
7. 测试时直接输入带噪数据即可去噪

## 5. 应用场景

- **图像去噪**：去除照片中的噪声
- **特征学习**：学习更鲁棒的特征表示
- **预训练**：作为深度网络的逐层预训练方法
- **推荐系统/广告**：对稀疏用户行为特征去噪，学习鲁棒的 embedding
- **数据修复**：补全缺失的数据
- **扩散模型的基础**：DDPM 本质上是多步 DAE

## 6. 优缺点分析

**优点**：
- 学到的表示对输入扰动具有鲁棒性
- 避免学习恒等映射（因为输入被噪声破坏）
- 可处理部分缺失的输入
- 理论上等价于某种形式的得分匹配

**缺点**：
- 需要选择合适的噪声类型和强度
- 单步去噪能力有限，严重噪声下效果差
- 不适合直接用于生成（与 VAE 不同）

## 7. 调库实现（Python）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def add_noise(self, x, noise_factor=0.3):
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0.0, 1.0)

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out

model = DenoisingAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        noisy_x = model.add_noise(batch_x, noise_factor=0.3)
        output = model(noisy_x)
        loss = criterion(output, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

## 8. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_gaussian_noise(x, scale=0.3):
    return torch.clamp(x + torch.randn_like(x) * scale, 0.0, 1.0)

def add_mask_noise(x, mask_prob=0.3):
    mask = torch.bernoulli(torch.full_like(x, 1 - mask_prob))
    return x * mask

class DAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

class DAEDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class DAEManual(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=32):
        super().__init__()
        self.encoder = DAEEncoder(input_dim, hidden_dim)
        self.decoder = DAEDecoder(hidden_dim, input_dim)

    def forward(self, x, noise_type='gaussian', noise_level=0.3):
        if noise_type == 'gaussian':
            noisy_x = add_gaussian_noise(x, noise_level)
        else:
            noisy_x = add_mask_noise(x, noise_level)
        h = self.encoder(noisy_x)
        x_recon = self.decoder(h)
        return x_recon, noisy_x
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    test_x = dataset.data[:5].float().view(5, -1) / 255.0
    noisy_x = model.add_noise(test_x, noise_factor=0.5)
    denoised = model(noisy_x)

fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for i in range(5):
    axes[0, i].imshow(test_x[i].view(28, 28), cmap='gray')
    axes[0, i].set_title('Original')
    axes[1, i].imshow(noisy_x[i].view(28, 28), cmap='gray')
    axes[1, i].set_title('Noisy')
    axes[2, i].imshow(denoised[i].detach().view(28, 28), cmap='gray')
    axes[2, i].set_title('Denoised')
plt.tight_layout()
plt.savefig('dae_results.png')
plt.show()
```

## 10. 模型评估

- **去噪 PSNR/SSIM**：衡量去噪后的图像质量
- **重建误差**：干净输入与重建结果的 MSE
- **特征质量**：用编码器提取的特征在下游任务上的表现
- **鲁棒性测试**：在不同噪声水平下的去噪效果

## 11. 常见问题与易错点

- **噪声过强**：噪声太大导致信息完全丢失，网络无法恢复
- **噪声过弱**：噪声太小退化为普通 AE，失去去噪训练的意义
- **损失目标错误**：应对干净 $x$ 计算损失，而非对 $\tilde{x}$ 计算
- **噪声类型选择**：应根据实际场景选择噪声类型（高斯、掩码、椒盐等）

## 12. 学习总结

DAE 通过"破坏-恢复"的学习范式，迫使网络学习数据的本质结构。它不仅是实用的去噪工具，更是扩散模型的核心思想来源——DDPM 可以理解为多步迭代的 DAE。

## 13. 练习题与思考题

**Q1**：为什么 DAE 比 AE 更不容易学到恒等映射？

**A1**：因为输入被噪声破坏，网络不可能简单复制输入。它必须学会从不完整/损坏的信息中恢复原始信号，这迫使它学习数据的有意义特征。

**Q2**：DAE 与扩散模型的关系是什么？

**A2**：扩散模型的每一步去噪本质上就是一个 DAE。DDPM 训练网络预测噪声（或去噪后的数据），相当于在不同噪声水平上训练一系列 DAE。

**Q3**：掩码噪声和高斯噪声各适合什么场景？

**A3**：掩码噪声适合处理数据缺失问题（如推荐系统中的稀疏特征）；高斯噪声适合连续值的去噪任务（如图像去噪）。

## 14. 学习路径建议

1. 掌握 AE 基础 → 2. 学习 DAE 的噪声注入思想 → 3. 理解 DAE 与得分匹配的理论联系 → 4. 学习多步去噪 → 5. 进入 DDPM 扩散模型
