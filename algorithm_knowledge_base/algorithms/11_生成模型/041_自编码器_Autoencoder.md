# 自编码器 (Autoencoder) 学习文档

> 来源线索：本节内容根据原书中涉及自编码器和特征压缩的相关章节整理、扩展与教学化改写。

> 学会压缩再重建——自编码器是无监督特征学习和数据压缩的基础架构。

## 1. 算法基础认知

**一句话定义**：自编码器通过将输入压缩到低维瓶颈再重建原始输入，学习数据的有效表示。

**直觉类比**：想象你需要在一张很小的纸上记住一幅复杂的画。你必须提炼出画中最关键的特征（编码），然后在另一张纸上凭记忆重新画出来（解码）。你画得越像，说明你提炼的特征越好。

**历史背景**：自编码器的概念最早由Hinton和Salakhutdinov在2006年提出，用于无监督预训练。随后发展出多种变体：去噪自编码器（DAE）、变分自编码器（VAE）、向量量化自编码器（VQ-VAE）等。

**算法定位**：深度学习 / 无监督学习 / 表示学习。

**前置知识**：
- 前馈神经网络
- 损失函数（MSE、交叉熵）
- 梯度下降和反向传播

## 2. 核心原理

### 核心思想

自编码器由编码器和解码器组成：

- **编码器** $f_{enc}$：将高维输入 $x$ 映射到低维隐表示 $z$（瓶颈层）
- **解码器** $f_{dec}$：从隐表示 $z$ 重建原始输入 $\hat{x}$

训练目标：让 $\hat{x}$ 尽可能接近 $x$。由于瓶颈层的维度远小于输入，模型必须学会压缩信息——只保留最重要的特征。

### 工作流程

1. 输入数据 $x$（如图像、文本向量）
2. 编码器将 $x$ 压缩为隐表示 $z = f_{enc}(x)$，$z$ 维度远小于 $x$
3. 解码器从 $z$ 重建 $\hat{x} = f_{dec}(z)$
4. 计算重建损失 $\mathcal{L} = \|x - \hat{x}\|^2$
5. 反向传播更新编码器和解码器的参数

### 关键概念

- **瓶颈层(Bottleneck)**：编码器的输出层，维度最低，迫使模型压缩信息
- **重建损失**：衡量原始输入和重建输出的差异
- **过完备风险**：如果隐表示维度≥输入维度，模型可能学到恒等映射（不压缩）
- **隐表示(Latent)**：编码器的输出，是数据的有意义表示

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 输入数据 | $(d,)$ |
| $z$ | 隐表示（编码） | $(k,)$, $k \ll d$ |
| $\hat{x}$ | 重建输出 | $(d,)$ |
| $W_{enc}, b_{enc}$ | 编码器参数 | - |
| $W_{dec}, b_{dec}$ | 解码器参数 | - |

### 编码过程

$$z = f_{enc}(x) = \sigma(W_{enc} x + b_{enc})$$

### 解码过程

$$\hat{x} = f_{dec}(z) = \sigma(W_{dec} z + b_{dec})$$

### 损失函数

$$\mathcal{L} = \|x - \hat{x}\|^2 = \sum_{i=1}^{d}(x_i - \hat{x}_i)^2$$

### 带正则化的自编码器

为防止学到恒等映射，添加稀疏正则化：

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \|z\|_1$$

## 4. 训练过程讲解

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| 隐表示维度 $k$ | 信息压缩程度 | 输入维度的10%-50% | 取决于任务 |
| 学习率 | 优化步长 | 1e-4 到 1e-3 | 1e-3 |
| 编码器层数 | 编码网络深度 | 1-4 | 2 |
| 正则化系数 | 稀疏约束强度 | 1e-5 到 1e-3 | 1e-4 |

## 5. 应用场景

1. **数据降噪**：去噪自编码器（DAE）在输入上添加噪声，训练模型学习去除噪声恢复原始数据。

2. **特征提取**：用编码器的隐表示作为下游任务的特征输入（如分类、聚类）。

3. **数据压缩**：学习高效的数据压缩方案（如VQ-VAE用于图像和音频压缩）。

4. **异常检测**：正常样本重建误差低，异常样本重建误差高——用重建误差作为异常分数。

5. **生成模型基础**：VAE、VQ-VAE等生成模型的自编码器变体。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 无监督学习，不需要标签 | 重建质量受瓶颈维度限制 |
| 可学习有意义的特征表示 | 可能学到恒等映射（维度不够低时） |
| 可用于降噪、压缩、异常检测 | 生成能力不如GAN和扩散模型 |
| 架构简单，易于实现 | 隐表示可能不够平滑 |

## 7. 调库实现

```python
"""使用 PyTorch 实现自编码器"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """标准自编码器"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        
        # 编码器: input_dim -> hidden -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 解码器: latent -> hidden -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class ConvAutoencoder(nn.Module):
    """卷积自编码器（用于图像）"""
    
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # 编码器: (B, 1, 28, 28) -> (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        
        # 解码器: (B, latent_dim) -> (B, 1, 28, 28)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class DenoisingAutoencoder(nn.Module):
    """去噪自编码器"""
    
    def __init__(self, input_dim, latent_dim, noise_factor=0.3):
        super().__init__()
        self.noise_factor = noise_factor
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid()
        )
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return torch.clamp(x + noise, 0, 1)
    
    def forward(self, x):
        x_noisy = self.add_noise(x) if self.training else x
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat, z, x_noisy


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 标准自编码器测试
    ae = Autoencoder(input_dim=784, latent_dim=32, hidden_dim=128)
    x = torch.randn(4, 784)
    x_hat, z = ae(x)
    
    print("=== 自编码器测试 ===")
    print(f"输入: {x.shape}")
    print(f"隐表示: {z.shape} (压缩 {784//32}x)")
    print(f"重建: {x_hat.shape}")
    
    # 重建误差
    mse = nn.MSELoss()(x, x_hat)
    print(f"重建MSE: {mse.item():.4f}")
    
    # 参数量对比
    total = sum(p.numel() for p in ae.parameters())
    print(f"参数量: {total:,}")
```

## 8. 手工代码实现

```python
"""从零实现自编码器（不使用nn.Sequential，手动管理参数）"""
import torch
import torch.nn as nn


class ManualAutoencoder:
    """手写自编码器"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=64, lr=1e-3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器权重
        self.W1 = torch.randn(input_dim, hidden_dim) * 0.01
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(hidden_dim, latent_dim) * 0.01
        self.b2 = torch.zeros(latent_dim)
        
        # 解码器权重
        self.W3 = torch.randn(latent_dim, hidden_dim) * 0.01
        self.b3 = torch.zeros(hidden_dim)
        self.W4 = torch.randn(hidden_dim, input_dim) * 0.01
        self.b4 = torch.zeros(input_dim)
        
        self.lr = lr
    
    def _relu(self, x):
        return torch.maximum(x, torch.zeros_like(x))
    
    def encode(self, x):
        h = self._relu(x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return z
    
    def decode(self, z):
        h = self._relu(z @ self.W3 + self.b3)
        out = torch.sigmoid(h @ self.W4 + self.b4)
        return out
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def train_step(self, x):
        """单步训练（简化的梯度下降）"""
        # 前向传播
        x_hat, z = self.forward(x)
        loss = ((x - x_hat) ** 2).mean()
        
        # 简化反向传播: 使用PyTorch autograd
        x_param = x.clone().detach().requires_grad_(True)
        
        # 实际训练中应使用autograd或手动推导梯度
        # 这里使用数值近似进行参数更新
        with torch.no_grad():
            # 计算重建误差的梯度方向
            error = x_hat - x  # (batch, input_dim)
            
            # 简化更新（不是精确梯度，仅演示思路）
            # 精确实现需要完整的反向传播链
            grad_W4 = self._relu(z @ self.W3 + self.b3).T @ error
            self.W4 -= self.lr * grad_W4 / x.shape[0]
        
        return loss.item()


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    ae = ManualAutoencoder(input_dim=20, latent_dim=5, hidden_dim=16)
    x = torch.rand(8, 20)
    
    x_hat, z = ae.forward(x)
    print("=== 手写自编码器测试 ===")
    print(f"输入: {x.shape}")
    print(f"隐表示: {z.shape} (压缩 {20//5}x)")
    print(f"重建: {x_hat.shape}")
    
    loss = ((x - x_hat) ** 2).mean()
    print(f"重建MSE: {loss.item():.4f}")
```

## 9. 可视化与结果理解

```python
"""自编码器可视化"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 压缩-重建效果
latent_dims = [2, 5, 10, 20, 50, 100, 200]
reconstruction_mse = [0.35, 0.18, 0.08, 0.04, 0.015, 0.005, 0.002]  # 模拟数据

axes[0].plot(latent_dims, reconstruction_mse, 'o-', color='#3498db', linewidth=2)
axes[0].set_title('隐表示维度 vs 重建误差', fontsize=13)
axes[0].set_xlabel('隐表示维度 k')
axes[0].set_ylabel('重建MSE')
axes[0].grid(True, alpha=0.3)

# 图2: 去噪效果
np.random.seed(42)
original = np.sin(np.linspace(0, 4*np.pi, 50))
noisy = original + np.random.randn(50) * 0.3
# 模拟去噪后的信号
denoised = original + np.random.randn(50) * 0.05

axes[1].plot(original, 'b-', linewidth=2, label='原始信号')
axes[1].plot(noisy, 'r.', alpha=0.5, label='加噪输入')
axes[1].plot(denoised, 'g--', linewidth=2, label='去噪重建')
axes[1].set_title('去噪自编码器效果', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3: 异常检测
normal_scores = np.random.exponential(0.02, 100)
anomaly_scores = np.random.exponential(0.15, 20) + 0.1
scores = np.concatenate([normal_scores, anomaly_scores])
labels = ['正常'] * 100 + ['异常'] * 20

axes[2].hist(normal_scores, bins=20, alpha=0.7, color='#2ecc71', label='正常样本')
axes[2].hist(anomaly_scores, bins=10, alpha=0.7, color='#e74c3c', label='异常样本')
axes[2].axvline(x=0.08, color='black', linestyle='--', label='阈值')
axes[2].set_title('异常检测: 重建误差分布', fontsize=13)
axes[2].set_xlabel('重建误差')
axes[2].set_ylabel('样本数')
axes[2].legend()

plt.tight_layout()
plt.savefig('autoencoder_viz.png', dpi=100)
plt.show()

print("图1解读: 隐表示维度越大重建误差越小, 但维度太大失去压缩意义")
print("图2解读: 去噪自编码器能有效去除噪声, 恢复原始信号")
print("图3解读: 正常样本重建误差小, 异常样本重建误差大, 可用于异常检测")
```

## 10. 模型评估

```python
"""自编码器评估"""
def evaluate_autoencoder(model, dataloader):
    """评估自编码器的重建质量"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x in dataloader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x_hat, z = model(x)
            loss = nn.MSELoss()(x, x_hat)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    
    avg_mse = total_loss / total_samples
    print(f"平均重建MSE: {avg_mse:.6f}")
    print(f"平均重建PSNR: {-10 * math.log10(avg_mse):.2f} dB")
    return avg_mse
```

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 学到恒等映射 | 隐表示无意义 | 瓶颈维度太高 | 减小隐表示维度 |
| 重建模糊 | 细节丢失 | MSE损失导致平均化 | 使用感知损失或对抗损失 |

## 12. 学习总结

自编码器的核心：$x \xrightarrow{encode} z \xrightarrow{decode} \hat{x}$，目标是最小化 $\|x - \hat{x}\|^2$。

关键变体：去噪自编码器（DAE）、变分自编码器（VAE）、VQ-VAE。

## 13. 练习题与思考题

### 基础题1：压缩比计算

输入维度784（28×28图像），隐表示维度32。压缩比是多少？编码器参数量是多少？

**参考答案**：
- 压缩比 = 784 / 32 = 24.5x
- 编码器参数（单层）= 784 × 32 + 32 = 25,120

### 基础题2：恒等映射

为什么隐表示维度等于输入维度时，自编码器可能学到恒等映射？

**参考答案**：
当隐表示维度≥输入维度时，存在一个完美解：$W_{enc} = I, W_{dec} = I$，即编码器和解码器都是恒等矩阵。此时 $\hat{x} = x$，损失为零，但模型没有学到有意义的特征。解决方案：添加正则化或限制隐表示维度。

### 进阶题：自编码器与PCA的关系

证明：当编码器和解码器都是单层线性网络（无激活函数）时，自编码器学到的隐表示等价于PCA的前k个主成分。

**参考答案**：
线性自编码器的优化目标是 $\min \|X - X W_{enc} W_{dec}\|^2$。这等价于找到矩阵 $W = W_{enc} W_{dec}$ 使得 $\|X - XW\|^2$ 最小，且 $W$ 的秩为 $k$。根据Eckart-Young定理，最优解是 $W = V_k V_k^T$，其中 $V_k$ 是 $X^TX$ 的前k个特征向量。这正是PCA的投影矩阵。

### 开放思考题

VQ-VAE用离散码本替代连续隐表示，这在多模态生成中有什么优势？

**参考思路**：
离散表示的优势：
1. 可以与语言模型统一：离散码可以当作"token"，用自回归模型生成
2. 避免了VAE的"后验坍塌"问题
3. 更适合多模态统一（图像token + 文本token + 音频token）

## 14. 学习路径建议

### 前置知识
- 前馈神经网络
- MSE损失函数
- 无监督学习概念

### 平行学习
- PCA（线性降维）
- 变分自编码器（VAE）

### 进阶方向
- VQ-VAE和FSQ（离散表示）
- 扩散模型（另一种生成方法）
- 自监督对比学习（SimCLR、BYOL）

### 推荐资源
1. **论文**：Reducing the Dimensionality of Data with Neural Networks (Hinton & Salakhutdinov, 2006)
2. **课程**：Stanford CS231n - Autoencoders章节
3. **论文**：Neural Discrete Representation Learning (VQ-VAE) (van den Oord et al., 2017)
