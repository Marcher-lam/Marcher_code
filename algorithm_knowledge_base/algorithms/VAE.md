# VAE学习文档

## 1. 算法基础认知

变分自编码器（Variational Autoencoder, VAE）是一种基于变分推断的深度生成模型，由Kingma和Welling在2014年提出。VAE将自编码器的潜在空间建模为概率分布，而非确定性的数值，从而能够从潜在空间中进行采样生成新样本。VAE的核心创新在于使用重参数化技巧（reparameterization trick）来实现端到端的可微训练。

VAE与经典自编码器的根本区别在于：
- 经典自编码器：潜在空间是确定性的嵌入向量
- VAE：潜在空间是概率分布，可以采样生成新样本

VAE的应用包括：
- 图像生成与重建
- 异常检测
- 半监督学习
- 风格迁移
- 药物分子生成

## 2. 核心原理

### 2.1 生成模型框架

VAE假设数据$\mathbf{x}$由潜在变量$\mathbf{z}$生成，生成过程为：
$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

其中：
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：先验分布
- $p_\theta(\mathbf{x}|\mathbf{z})$：解码器（生成网络）
- $p(\mathbf{z})$：先验分布（标准高斯）

### 2.2 变分推断

由于直接优化$p_\theta(\mathbf{x})$涉及 intractable 的积分，VAE使用变分推断近似：

使用变分下界（Evidence Lower Bound, ELBO）：
$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

其中：
- $q_\phi(\mathbf{z}|\mathbf{x})$：编码器（推理网络）
- 第一项：重建损失
- 第二项：KL散度（正则化项）

### 2.3 重参数化技巧

为了使ELBO可微，需要从分布$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$中采样。

重参数化技巧：
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

这样就能通过$\boldsymbol{\epsilon}$传递梯度，使整个网络端到端可微。

## 3. 数学公式与推导

### 3.1 ELBO的完整推导

从边际似然出发：
$$\log p_\theta(\mathbf{x}^{(i)}) = \log \int p_\theta(\mathbf{x}^{(i)}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

引入变分分布$q_\phi(\mathbf{z}|\mathbf{x})$：
$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z}|\mathbf{x}))$$

由于$D_{KL} \geq 0$，所以：
$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}(\theta, \phi; \mathbf{x})$$

ELBO可以分解为：
$$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

### 3.2 KL散度的解析解

当$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$，$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$时：

$$D_{KL} = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

这提供了KL项的闭式解，无需Monte Carlo近似。

### 3.3 重建损失

对于二元数据（二值图像）：
$$\log p_\theta(\mathbf{x}|\mathbf{z}) = \sum_j x_j \log \hat{x}_j + (1-x_j) \log(1-\hat{x}_j)$$

对于连续数据（实值图像）：
$$\log p_\theta(\mathbf{x}|\mathbf{z}) = -\frac{1}{2} \sum_j (x_j - \hat{x}_j)^2 + \text{const}$$

通常使用MSE作为重建损失。

## 4. 训练过程讲解

### 4.1 网络结构

**编码器（Inference Network）**：
$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))$$

输出$\boldsymbol{\mu}$和$\log \boldsymbol{\sigma}^2$（使用log防止方差为负）。

**解码器（Generative Network）**：
$$p_\theta(\mathbf{x}|\mathbf{z})$$

通常是生成网络，输出重建的$\mathbf{x}$。

### 4.2 训练流程

1. 输入图像$\mathbf{x}$通过编码器得到$\boldsymbol{\mu}, \boldsymbol{\sigma}$
2. 通过重参数化采样$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$
3. $\mathbf{z}$通过解码器得到$\hat{\mathbf{x}}$
4. 计算ELBO损失：
   - 重建损失：$\|\mathbf{x} - \hat{\mathbf{x}}\|^2$
   - KL损失：闭式解
5. 反向传播更新参数
6. 循环直到收敛

### 4.3 生成新样本

训练好后，从先验分布$\mathcal{N}(\mathbf{0}, \mathbf{I})$采样$\mathbf{z}$，通过解码器生成新图像：
$$\mathbf{x}_{new} \sim p_\theta(\mathbf{x}|\mathbf{z}), \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

## 5. 应用场景

### 5.1 图像生成

- 人脸生成
- 艺术风格生成
- 图像补全

### 5.2 异常检测

- 工业缺陷检测
- 医学影像异常检测
- 入侵检测

### 5.3 半监督学习

- 使用VAE的潜在空间进行分类
- 结合监督学习提高性能

### 5.4 分子生成

- 药物分子生成
- 材料设计
- SMILES序列生成

### 5.5 其它生成任务

- 语音合成
- 音乐生成
- 对话生成

## 6. 优缺点分析

### 优点

1. **可生成新样本**：能学习数据分布进行采样生成
2. **潜在空间连续**：插值操作有意义
3. **端到端可训练**：所有模块可微
4. **理论基础扎实**：基于变分推断
5. **半监督友好**：潜在空间可用于下游任务

### 缺点

1. **生成质量一般**：不如GAN
2. **后验坍塌**：KL项趋向于0
3. **推断能力弱**：对复杂分布建模能力有限
4. **假设限制**：假设高斯分布
5. **训练不稳定**：需要平衡重建和KL

## 7. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class VAEEncoder(nn.Module):
    """
    VAE编码器
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAEEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出均值和方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE解码器
    """
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(VAEDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """
    变分自编码器
    
    参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        latent_dim: 潜在空间维度
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        
        # 解码器
        self.decoder = VAEDecoder(latent_dim, hidden_dims[::-1], input_dim)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """前向传播"""
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        """计算VAE损失（ELBO）"""
        # 重建损失
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        loss = recon_loss + kl_loss
        
        return loss, recon_loss, kl_loss
    
    def generate(self, num_samples):
        """从先验分布生成新样本"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            return self.decoder(z)
    
    def encode(self, x):
        """编码到潜在空间"""
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            return mu
    
    def decode(self, z):
        """从潜在空间解码"""
        with torch.no_grad():
            return self.decoder(z)


class ConditionalVAE(nn.Module):
    """
    条件VAE（CVAE）
    
    参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度
        latent_dim: 潜在空间维度
        num_classes: 条件类别数
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 编码器（加入条件信息）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
        # 解码器（加入条件信息）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, labels):
        """
        前向传播
        
        参数:
            x: 输入
            labels: 条件标签（one-hot）
        """
        # 合并输入和标签
        x_labeled = torch.cat([x, labels], dim=1)
        
        # 编码
        h = self.encoder(x_labeled)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        z_labeled = torch.cat([z, labels], dim=1)
        x_recon = self.decoder(z_labeled)
        
        return x_recon, mu, logvar
    
    def generate(self, labels):
        """条件生成"""
        with torch.no_grad():
            z = torch.randn(labels.size(0), self.latent_dim)
            z_labeled = torch.cat([z, labels], dim=1)
            return self.decoder(z_labeled)


class BetaVAE(nn.Module):
    """
    Beta-VAE：调整KL权重的VAE
    
    参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度
        latent_dim: 潜在空间维度
        beta: KL权重（>1使潜在空间更独立）
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        
        self.beta = beta
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


class VQVAE(nn.Module):
    """
    VQ-VAE：向量量化VAE
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, num_embeddings):
        super(VQVAE, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def forward(self, x):
        # 编码
        z = self.encoder(x)
        
        # 量化
        z_flat = z.view(-1, self.latent_dim)
        
        # 最近邻查找
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.T)
        
        indices = d.argmin(dim=1)
        z_q = self.embedding(indices)
        
        # 停止梯度传���
        z_q = z_q + (z - z_q).detach()
        
        # 解码
        x_recon = self.decoder(z_q)
        
        return x_recon, indices


class VAEConv(nn.Module):
    """
    卷积VAE用于图像
    """
    
    def __init__(self, in_channels=1, latent_dim=20):
        super(VAEConv, self).__init__()
        
        # 编码器
        self.enc_conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        
        self.enc_fc = nn.Linear(128 * 4 * 4, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 解码器
        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.dec_conv3 = nn.ConvTranspose2d(32, in_channels, 3, 2, 1, 1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # 解码
        h = F.relu(self.dec_fc(z))
        h = h.view(h.size(0), 128, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = self.dec_conv3(h)
        
        return h, mu, logvar


def test_vae():
    """测试VAE"""
    print("=" * 50)
    print("测试VAE")
    print("=" * 50)
    
    # 配置
    batch_size = 32
    input_dim = 784  # 28x28
    hidden_dims = [256, 128]
    latent_dim = 20
    
    x = torch.randn(batch_size, input_dim)
    
    # 测试VAE
    print("1. 基础VAE:")
    vae = VAE(input_dim, hidden_dims, latent_dim)
    x_recon, mu, logvar = vae(x)
    print(f"   输入: {x.shape}")
    print(f"   重建: {x_recon.shape}")
    print(f"   参数: {sum(p.numel() for p in vae.parameters()):,}")
    print()
    
    # 测试损失
    loss, recon_loss, kl_loss = vae.loss(x, x_recon, mu, logvar)
    print(f"   损失: {loss.item():.4f}")
    print(f"   重建损失: {recon_loss.item():.4f}")
    print(f"   KL损失: {kl_loss.item():.4f}")
    print()
    
    # 测试生成
    print("2. 生成新样本:")
    samples = vae.generate(8)
    print(f"   生成形状: {samples.shape}")
    print()
    
    # 测试卷积VAE
    print("3. 卷积VAE:")
    vae_conv = VAEConv(in_channels=1, latent_dim=20)
    x_img = torch.randn(batch_size, 1, 28, 28)
    x_recon, mu, logvar = vae_conv(x_img)
    print(f"   输入: {x_img.shape}")
    print(f"   重建: {x_recon.shape}")
    print()
    
    # 训练演示
    print("=" * 50)
    print("训练演示")
    print("=" * 50)
    
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.randn(1000, input_dim).astype(np.float32) * 0.5 + 0.25
    X = torch.from_numpy(X)
    
    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)
    
    # 训练
    vae = VAE(input_dim, [256, 128], latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    
    for epoch in range(10):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_x in train_loader:
            batch_x = batch_x[0]
            
            # ��向��播
            x_recon, mu, logvar = vae(batch_x)
            loss, recon_loss, kl_loss = vae.loss(batch_x, x_recon, mu, logvar)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        n_batches = len(train_loader)
        print(f"Epoch {epoch+1}: Loss={total_loss/n_batches:.4f}, "
              f"Recon={total_recon/n_batches:.4f}, KL={total_kl/n_batches:.4f}")


if __name__ == "__main__":
    test_vae()
```

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np


class ManualVAE:
    """手动实现VAE（简化版，仅用于理解原理）"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 编码器权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)
        
        # 解码器权重
        self.W2 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b3 = np.zeros(input_dim)
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std
    
    def encode(self, x):
        """编码"""
        h = np.maximum(0, x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar
    
    def decode(self, z):
        """解码"""
        h = np.maximum(0, z @ self.W2 + self.b2)
        return h @ self.W3 + self.b3
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def kl_divergence(self, mu, logvar):
        """KL散度的闭式解"""
        return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    
    def reconstruction_loss(self, x, x_recon):
        """MSE重建损失"""
        return np.sum((x - x_recon)**2)
    
    def loss(self, x):
        """计算ELBO损失"""
        x_recon, mu, logvar = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = self.kl_divergence(mu, logvar)
        
        elbo = -(recon_loss + kl_loss)
        
        return elbo, recon_loss, kl_loss
    
    def generate(self, n_samples):
        """生成新样本"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)


def test_manual_vae():
    """测试手动实现"""
    print("=" * 50)
    print("测试VAE手动实现")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 配置
    input_dim = 100
    hidden_dim = 50
    latent_dim = 10
    
    # 创建VAE
    vae = ManualVAE(input_dim, hidden_dim, latent_dim)
    
    # 输入
    x = np.random.randn(32, input_dim)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    x_recon, mu, logvar = vae.forward(x)
    
    print(f"重建形状: {x_recon.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"方差形状: {logvar.shape}")
    
    # 损失
    elbo, recon, kl = vae.loss(x)
    print(f"\nELBO: {elbo:.4f}")
    print(f"重建损失: {recon:.4f}")
    print(f"KL损失: {kl:.4f}")
    
    # 生成
    samples = vae.generate(8)
    print(f"\n生成样本形状: {samples.shape}")
    
    # 潜在空间插值
    print("\n潜在空间插值:")
    z1 = np.random.randn(latent_dim)
    z2 = np.random.randn(latent_dim)
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z = alpha * z1 + (1 - alpha) * z2
        sample = vae.decode(z.reshape(1, -1))[0]
        print(f"  alpha={alpha}: mean={sample.mean():.4f}, std={sample.std():.4f}")


if __name__ == "__main__":
    test_manual_vae()
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_vae():
    """可视化VAE"""
    print("=" * 50)
    print("可视化VAE")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. VAE结构
    ax = axes[0, 0]
    ax.axis('off')
    ax.text(0.5, 0.6, 'VAE架构\n\n'
                         '输入x → Encoder → μ,σ\n'
                         '    ↓ 重参数化\n'
                         '    z → Decoder → x̂\n\n'
                         '损失:\n'
                         '  ELBO = 重建 + KL\n'
                         '  KL = -½∑(1+logσ²-μ²-logσ²)',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('VAE架构')
    
    # 2. 潜在空间分布
    ax = axes[0, 1]
    np.random.seed(42)
    
    # 真实分布
    z_real = np.random.randn(500, 2)
    ax.scatter(z_real[:, 0], z_real[:, 1], alpha=0.5, c='blue', label='真实分布')
    
    # 学习到的分布
    z_learned = np.random.randn(500, 2) * 0.8 + 0.1
    ax.scatter(z_learned[:, 0], z_learned[:, 1], alpha=0.5, c='red', label='学习分布')
    
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.set_title('潜在空间分布')
    ax.legend()
    ax.grid(True)
    
    # 3. KL权重影响
    ax = axes[0, 2]
    betas = [0.1, 0.5, 1.0, 2.0, 4.0]
    capacities = [5.1, 6.8, 8.2, 12.5, 18.3]
    ax.bar(betas, capacities, color='steelblue')
    ax.set_xlabel('Beta (KL权重)')
    ax.set_ylabel('潜在空间容量')
    ax.set_title('Beta-VAE: KL权重与容量')
    
    # 4. 重建质量
    ax = axes[1, 0]
    latent_dims = [2, 5, 10, 20, 50]
    recon_errors = [0.45, 0.32, 0.25, 0.18, 0.15]
    ax.plot(latent_dims, recon_errors, 'o-')
    ax.set_xlabel('潜在维度')
    ax.set_ylabel('重建误差')
    ax.set_title('潜在维度 vs 重建误差')
    ax.grid(True)
    
    # 5. 生成样本演变
    ax = axes[1, 1]
    steps = np.arange(100)
    fid_scores = 150 - 0.8 * steps + np.random.randn(100) * 5
    ax.plot(steps, fid_scores, 'b-', alpha=0.7)
    ax.axhline(50, color='r', linestyle='--', label='真实FID')
    ax.set_xlabel('训练步骤')
    ax.set_ylabel('FID分数')
    ax.set_title('生成质量演变')
    ax.legend()
    
    # 6. 不同VAE变体对比
    ax = axes[1, 2]
    variants = ['VAE', 'CVAE', 'Beta-VAE', 'VQ-VAE', 'AAE']
    scores = [8.2, 7.8, 7.2, 6.5, 6.1]
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    ax.barh(variants, scores, color=colors)
    ax.set_xlabel('重建误差 (越低越好)')
    ax.set_title('VAE变体对比')
    
    plt.tight_layout()
    plt.savefig('vae_visualization.png', dpi=150)
    print("可视化已保存为 vae_visualization.png")


def analyze_kl_term():
    """分析KL项"""
    print("\n" + "=" * 50)
    print("KL散度项分析")
    print("=" * 50)
    
    print("当μ=0, σ=1: D_KL = 0")
    print("当μ≠0: 鼓励潜在空间接近原点")
    print("当σ≠1: 鼓励潜在空间接近单位方差")
    
    print("\n平衡重建和KL:")
    print("  - 重建为主：生成质量好，但潜在空间混乱")
    print("  - KL为主：潜在空间整齐，但重建质量差")


if __name__ == "__main__":
    visualize_vae()
    analyze_kl_term()
```

## 10. 模型评估

### 10.1 评估指标

- **重建误差**：MSE/Perceptual Loss
- **KL散度**
- **生成质量**：FID分数
- **潜在空间分析**：后验分布
- **可视化**：插值生成

### 10.2 实验评估代码

```python
import torch
import time


def evaluate_vae():
    """评估VAE"""
    print("=" * 50)
    print("评估VAE性能")
    print("=" * 50)
    
    # 配置
    batch_size = 64
    input_dim = 784
    x = torch.randn(batch_size, input_dim)
    
    models = {
        "基础VAE": lambda: VAE(input_dim, [256, 128], 20),
        "大VAE": lambda: VAE(input_dim, [512, 256], 50),
        "卷积VAE": lambda: VAEConv(1, 20),
    }
    
    for name, model_fn in models.items():
        model = model_fn()
        
        params = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            if "卷积" in name:
                x_img = torch.randn(batch_size, 1, 28, 28)
                out = model(x_img)
            else:
                out = model(x)
        
        print(f"{name:<12}: 参数 {params:>10,}, 重建 {out[0].shape}")


if __name__ == "__main__":
    evaluate_vae()
```

## 11. 常见问题与易错点

### 常见问题

1. **后验坍塌**
   - 原因：KL项权重过大
   - 解决：使用β-VAE，或 warm-up

2. **重建质量差**
   - 原因：潜在维度太小
   - 解决：增大latent_dim

3. **训练不稳定**
   - 原因：学习率太大
   - 解决：减小学习率，使用KL annealing

### 易错点

1. **混淆logvar和var**
   - 使用logvar避免数值不稳定

2. **忽视批次维度**
   - 确认batch维度

3. **损失计算错误**
   - ELBO = 重建 + KL

## 12. 学习总结

### 核心要点

1. **VAE使用变分推断**
   - ELBO作为损失函数
   - KL项作为正则化

2. **重参数化技巧**
   - $z = \mu + \sigma \odot \epsilon$
   - 允许梯度反向传播

3. **潜在空间**
   - 连续可插值
   - 采样生成新样本

4. **平衡重建和KL**
   - β-VAE调参
   - KL annealing

### 关键公式

- ELBO: $\mathcal{L} = \mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$
- KL: $D_{KL} = \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$
- 重参数化: $z = \mu + \sigma \odot \epsilon$

### 最佳实践

1. 使用大batch
2. KL annealing
3. 使用CNNs处理图像
4. 监控KL/重建比例

## 13. 练习题与思考题

### 基础练习

1. **推导KL闭式解**
   - 推导当分布为高斯时KL的解析解

2. **实现重参数化**
   - 推导$z = \mu + \sigma \odot \epsilon$的梯度

3. **计算ELBO**
   - 给定参数计算ELBO

### 进阶练习

4. **实现CVAE**
   - 实现条件VAE

5. **实现β-VAE**
   - 实现可调β的VAE

6. **分析后验坍塌**
   - 分析后验坍塌的原因

### 思考题

7. VAE与GAN的区别？
8. 如何改进VAE的生成质量？
9. VAE在分子生成中的应用？

### 答案

1. **答案**: $D_{KL} = \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$

2. **答案**: $\nabla_z = \nabla_\mu + \nabla_{\log\sigma} \odot \epsilon$

3. **答案**: ELBO = reconstruction + KL

4. **答案**: ConditionalVAE类

5. **���案**: BetaVAE类

6. **答案**: β过大导致后验坍塌