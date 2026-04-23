# AE（自编码器）学习文档

## 1. 算法基础认知

自编码器（Autoencoder, AE）是一种无监督学习的神经网络，目标是学习输入数据的压缩表示。它由两部分组成：**编码器（Encoder）** 将输入映射到低维隐空间，**解码器（Decoder）** 从隐空间重建输入。瓶颈层（Bottleneck）迫使网络捕捉数据中最关键的特征。

与 PCA 不同，AE 是非线性降维方法，能学习更复杂的数据流形结构。

## 2. 核心原理

AE 的核心思想是：如果网络能从低维表示中完美重建输入，那么这个低维表示就包含了数据的本质信息。

- **编码器**：$h = f_\theta(x)$，将高维输入 $x$ 压缩到低维隐向量 $h$
- **解码器**：$\hat{x} = g_\phi(h)$，从隐向量重建输入
- **瓶颈层**：隐层维度远小于输入维度，形成信息瓶颈，迫使网络学习紧凑表示

训练目标是最小化重建误差，使得 $\hat{x} \approx x$。

## 3. 数学公式与推导

**损失函数（重建损失）**：

$$L_{AE} = \frac{1}{N}\sum_{i=1}^{N}\|x^{(i)} - \hat{x}^{(i)}\|^2 = \frac{1}{N}\sum_{i=1}^{N}\|x^{(i)} - g_\phi(f_\theta(x^{(i)}))\|^2$$

其中 $\|\cdot\|^2$ 为欧氏距离（MSE），对于二值输入也可用交叉熵：

$$L_{CE} = -\sum_{j=1}^{d}[x_j \log \hat{x}_j + (1-x_j)\log(1-\hat{x}_j)]$$

**信息瓶颈**：设输入维度 $d$，隐层维度 $z$，要求 $z \ll d$，迫使网络丢弃冗余信息，只保留最重要的特征。

## 4. 训练过程讲解

1. 输入数据 $x$（通常归一化到 $[0,1]$）
2. 编码器前向传播：$h = f_\theta(x)$
3. 解码器前向传播：$\hat{x} = g_\phi(h)$
4. 计算重建损失 $L = \|x - \hat{x}\|^2$
5. 反向传播更新 $\theta$ 和 $\phi$
6. 重复直到收敛

训练完成后，编码器部分可提取为特征提取器，隐向量 $h$ 即为数据的低维表示。

## 5. 应用场景

- **降维与可视化**：替代 PCA 进行非线性降维
- **特征提取**：提取数据的压缩特征用于下游任务
- **去噪**：作为 DAE 的基础
- **异常检测**：异常样本重建误差大，可据此检测
- **图像压缩**：学习紧凑的图像表示
- **广告系统**：用户/物品 embedding 学习、特征降维

## 6. 优缺点分析

**优点**：
- 无需标签，纯无监督学习
- 非线性降维能力强于 PCA
- 结构灵活，可堆叠为深度自编码器

**缺点**：
- 隐空间缺乏结构，无法直接用于生成新样本
- 容易过拟合，当隐层维度 >= 输入维度时退化为恒等映射
- 重建质量与隐空间连续性之间存在矛盾

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

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out

model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(20):
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.view(batch_x.size(0), -1)
        output = model(batch_x)
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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class AutoencoderManual(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    test_images = dataset.data[:10].float().view(10, -1) / 255.0
    recon, codes = model(test_images)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(test_images[i].view(28, 28), cmap='gray')
    axes[0, i].set_title('Original')
    axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
    axes[1, i].set_title('Reconstructed')
plt.tight_layout()
plt.savefig('ae_reconstruction.png')
plt.show()

all_codes = []
all_labels = []
with torch.no_grad():
    for x, y in loader:
        _, z = model(x.view(x.size(0), -1))
        all_codes.append(z)
        all_labels.append(y)
codes = torch.cat(all_codes).numpy()
labels = torch.cat(all_labels).numpy()
plt.figure(figsize=(8, 6))
scatter = plt.scatter(codes[:, 0], codes[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
plt.colorbar(scatter)
plt.title('AE Latent Space (2D)')
plt.savefig('ae_latent.png')
plt.show()
```

## 10. 模型评估

- **重建误差（MSE）**：越低越好，直接反映重建质量
- **可视化**：观察重建图像与原图的差异
- **隐空间可视化**：隐向量在 2D 空间中的分布是否聚类
- **下游任务**：用隐向量做分类，评估特征质量

## 11. 常见问题与易错点

- **恒等映射问题**：隐层维度 >= 输入维度时，网络可能直接学恒等映射，失去降维意义
- **未归一化输入**：输入数据需要归一化到 [0,1]，否则重建困难
- **隐空间不连续**：AE 隐空间可能有"空洞"，不能保证插值生成合理样本
- **过拟合**：容量过大时记住了训练数据，泛化差。可用正则化（如 Dropout）缓解

## 12. 学习总结

AE 是生成模型的基础架构，核心是通过信息瓶颈学习数据的压缩表示。它简单直观，但隐空间缺乏概率结构，不能直接用于生成。VAE、DAE 等变体在此基础上做了重要改进。

## 13. 练习题与思考题

**Q1**：AE 与 PCA 的本质区别是什么？

**A1**：PCA 是线性降维，AE 通过非线性激活函数实现非线性降维。当 AE 隐层只有一层且无激活函数时，等价于 PCA。

**Q2**：为什么 AE 不能直接用于生成？

**A2**：AE 的隐空间没有概率约束，隐向量分布不规则。随机采样隐空间的点解码后可能生成无意义的样本。VAE 通过引入 KL 散度约束解决了这个问题。

**Q3**：如何用 AE 做异常检测？

**A3**：用正常数据训练 AE，异常样本重建误差会显著高于正常样本，设定阈值即可检测异常。

## 14. 学习路径建议

1. 掌握 AE 基础结构与训练 → 2. 理解隐空间的概念 → 3. 学习 VAE（引入概率框架） → 4. 学习 DAE（去噪） → 5. 了解扩散模型（DDPM）中的编码-解码思想
