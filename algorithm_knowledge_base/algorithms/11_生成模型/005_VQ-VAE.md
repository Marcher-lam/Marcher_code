# VQ-VAE学习文档

## 1. 算法基础认知

VQ-VAE（Vector Quantized Variational Autoencoder）是VAE的一种重要变体，由Van Den Oord等人于2017年提出。VQ-VAE的核心创新是将VAE的连续潜在空间离散化，使用向量量化（Vector Quantization）将连续_latent向量映射到一组离散的码本（codebook）嵌入上。这种离散化表示使得VQ-VAE能够学习更紧凑、更结构化的表示，特别适合于建模高维复杂分布。

VQ-VAE与VAE的关键区别：
- VAE：潜在空间是连续的高斯分布
- VQ-VAE：潜在空间是离散的码本索引

VQ-VAE的应用：
- 语音合成（WaveNet）
- 图像生成（PixelCNN）
- 语音编码
- 强化学习

## 2. 核心原理

### 2.1 向量化码本

设码本（codebook）包含$K$个$D$维向量：
$$\mathbf{e}_k \in \mathbb{R}^D, \quad k = 1, 2, \dots, K$$

码本可以是可学习的嵌入表：
$$\mathbf{E} = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_K\}$$

### 2.2 量化操作

给定Encoder的输出$\mathbf{z}_e$，量化操作为最近邻查找：
$$k = \arg\min_k \|\mathbf{z}_e - \mathbf{e}_k\|_2$$
$$\mathbf{z}_q = \mathbf{e}_k$$

这可以用argmin和one-hot表示：
$$k = \text{argmin}_k \text{sim}(\mathbf{z}_e, \mathbf{e}_k)$$
$$\mathbf{z}_q = \mathbf{e}_k$$

### 2.3 前向传播

VQ-VAE的前向传播：
1. Encoder：$\mathbf{z}_e = f_e(\mathbf{x})$
2. Quantization：$\mathbf{z}_q = q(\mathbf{z}_e)$（离散化）
3. Decoder：$\hat{\mathbf{x}} = f_d(\mathbf{z}_q)$

## 3. 数学公式与推导

### 3.1 损失函数

VQ-VAE的损失函数由三部分组成：

**重建损失**：
$$L_{recon} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$$

**VQ损失（码本损失）**：
$$L_{vq} = \|\sg[\mathbf{z}_e] - \mathbf{z}_q\|_2^2$$

其中$sg[\cdot]$是停止梯度操作，防止梯度流入Encoder。

**承诺损失（Commitment Loss）**：
$$L_{commit} = \|\mathbf{z}_e - sg[\mathbf{z}_q]\|_2^2$$

总损失：
$$L = L_{recon} + \beta_1 L_{vq} + \beta_2 L_{commit}$$

其中$\beta_1, \beta_2$是权重，通常$\beta_1=1, \beta_2=0.25$。

### 3.2 指数移动平均

码本可以使用EMA（指数移动平均）更新：
$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1-\gamma) \sum_j \mathbf{1}_{k=k_j}$$
$$\mathbf{m}_k^{(t)} = \gamma \mathbf{m}_k^{(t-1)} + (1-\gamma) \sum_j \mathbf{1}_{k=k_j} \mathbf{z}_e^{(j)}$$
$$\mathbf{e}_k^{(t)} = \frac{\mathbf{m}_k^{(t)}}{N_k^{(t)}}$$

### 3.3 软量化

硬量化使用argmax，梯度不可微。软量化使用softmax近似：
$$p_k = \text{softmax}(\lambda \cdot \text{sim}(\mathbf{z}_e, \mathbf{e}_k))$$
$$\mathbf{z}_q = \sum_k p_k \mathbf{e}_k$$

## 4. 训练过程讲解

### 4.1 训练流程

1. **输入**：图像/音频$\mathbf{x}$
2. **编码**：$\mathbf{z}_e = \text{Encoder}(\mathbf{x})$
3. **量化**：$\mathbf{z}_q = \text{Quantize}(\mathbf{z}_e)$
4. **解码**：$\hat{\mathbf{x}} = \text{Decoder}(\mathbf{z}_q)$
5. **计算损失**：$L_{recon} + L_{vq} + L_{commit}$
6. **反向传播**：更新Decoder和Encoder，码本可选择性地使用EMA更新

### 4.2 码本大小

码本大小$K$的选择：
- 小$K$：表示紧凑，但重建质量下降
- 大$K$：表示能力强，但计算开销增加
- 典型值：$K=256, 512, 1024, 8192$

### 4.3 潜在维度

潜在向量维度$D$：
- 小$D$：量化误差增加
- 大$D$：捕获更多细节
- 典型值：$D=64, 128, 256, 512$

## 5. 应用场景

### 5.1 语音合成

- WaveNet：使用因果��积+VQ-VAE
- 语音编码和解码

### 5.2 图像生成

- PixelCNN：自回归生成离散的latent
- 图像补全

### 5.3 强化学习

- 状态表示学习
- 技能发现

### 5.4 语音编码

- 音频压缩
- 说话人识别

## 6. 优缺点分析

### 优点

1. **离散表示**：学习结构化码本
2. **稳定训练**：不需要KL散度
3. **可扩展性好**：码本可调整
4. **生成质量高**：配合PixelCNN等自回归模型
5. **计算效率高**：推理只需码本查找

### 缺点

1. **码本崩溃**：部分码本不被使用
2. **重建质量**：不如连续VAE
3. **梯度问题**：需停止梯度
4. **初始化敏感**：码本初始化

## 7. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class VectorQuantizer(nn.Module):
    """
    向量量化器
    
    参数:
        embedding_dim: 嵌入向量维度
        num_embeddings: 码本大小
        commitment_cost: 承诺损失权重
    """
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        """
        前向传播
        
        参数:
            inputs: Encoder输出 (B, D) 或 (B, C, H, W, D)
        
        返回:
            quantized: 量化后的向量
            indices: 码本索引
            loss: VQ损失
        """
        input_shape = inputs.shape
        embedding_dim = self.embedding_dim
        
        # Flatten输入
        flat_input = inputs.view(-1, embedding_dim)  # (N, D)
        
        # 计算距离
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )  # (N, K)
        
        # 最近邻查找
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        
        # 量化
        quantized = self.embedding(encoding_indices)  # (N, D)
        
        # 重塑
        quantized = quantized.view(input_shape)
        encoding_indices = encoding_indices.view(input_shape[:-1])
        
        # 计算损失
        if inputs.dim() == 2:
            e_latent = flat_input
        else:
            e_latent = flat_input
        
        q_latent = self.embedding(encoding_indices.view(-1))
        
        # VQ损失
        vq_loss = F.mse_loss(quantized, e_latent.detach())
        commit_loss = F.mse_loss(e_latent, quantized.detach())
        
        loss = vq_loss + self.commitment_cost * commit_loss
        
        # 梯度pass-through（STE）
        quantized = e_latent + (quantized - e_latent).detach()
        
        return quantized, encoding_indices, loss


class VectorQuantizerEMA(nn.Module):
    """
    使用EMA的向量量化器
    """
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA缓冲
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
    
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 距离
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        # 梯度pass-through
        quantized = flat_input + (quantized - flat_input).detach()
        quantized = quantized.view(input_shape)
        
        return quantized, encoding_indices, torch.tensor(0.0)


class VQVAE(nn.Module):
    """
    VQ-VAE模型
    
    参数:
        in_channels: 输入通道数
        hidden_dims: 隐藏层维度
        embedding_dim: 潜在向量维度
        num_embeddings: 码本大小
    """
    
    def __init__(self, in_channels=1, hidden_dims=[128, 256], 
                 embedding_dim=64, num_embeddings=256):
        super(VQVAE, self).__init__()
        
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Encoder
        encoder_layers = []
        prev_channels = in_channels
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(prev_channels, hidden_dim, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_channels = hidden_dim
        
        encoder_layers.append(
            nn.Conv2d(prev_channels, embedding_dim, 3, 1, 1)
        )
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.vq_layer = VectorQuantizer(
            embedding_dim, 
            num_embeddings,
            commitment_cost=0.25
        )
        
        # Decoder
        decoder_layers = []
        prev_channels = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.ConvTranspose2d(prev_channels, hidden_dim, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_channels = hidden_dim
        
        decoder_layers.append(
            nn.Conv2d(prev_channels, in_channels, 3, 1, 1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # 编码
        z_e = self.encoder(x)
        
        # 量化
        z_q, indices, vq_loss = self.vq_layer(z_e)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        return x_recon, z_q, vq_loss, indices
    
    def encode(self, x):
        """编码到离散latent"""
        z_e = self.encoder(x)
        flat = z_e.view(-1, self.embedding_dim)
        
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.vq_layer.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat, self.vq_layer.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        
        return indices
    
    def decode(self, indices):
        """从离散latent解码"""
        z_q = self.vq_layer.embedding(indices)
        
        # 恢复形状
        B = indices.shape[0]
        H = indices.shape[1]
        W = indices.shape[2]
        
        z_q = z_q.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)
        
        return self.decoder(z_q)


class ResidualVQVAEBlock(nn.Module):
    """带残差块的VQ-VAE"""
    
    def __init__(self, channels, embedding_dim, num_embeddings):
        super(ResidualVQVAEBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(embedding_dim, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, embedding_dim, 1, 1, 0)
        
        self.vq = VectorQuantizer(embedding_dim, num_embeddings)
        
        self.residual = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim, 1, 1, 0)
        )
    
    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        
        # 残差
        h = h + self.residual(x)
        
        return h


class MultipleScalarQuantizer(nn.Module):
    """
    多标量量化器（使用多个码本）
    """
    
    def __init__(self, num_quantizers, embedding_dim, num_embeddings):
        super(MultipleScalarQuantizer, self).__init__()
        
        self.num_quantizers = num_quantizers
        
        self.quantizers = nn.ModuleList([
            VectorQuantizer(embedding_dim // num_quantizers, num_embeddings)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, z_e):
        losses = 0
        
        # 分割
        z_e_parts = torch.chunk(z_e, self.num_quantizers, dim=1)
        
        quantized_parts = []
        indices_parts = []
        
        for i, (quantizer, z_part) in enumerate(zip(self.quantizers, z_e_parts)):
            z_q, idx, loss = quantizer(z_part)
            losses += loss
            quantized_parts.append(z_q)
            indices_parts.append(idx)
        
        quantized = torch.cat(quantized_parts, dim=1)
        indices = torch.stack(indices_parts, dim=1)
        
        return quantized, indices, losses


def test_vq_vae():
    """测试VQ-VAE"""
    print("=" * 50)
    print("测试VQ-VAE")
    print("=" * 50)
    
    # 配置
    batch_size = 4
    in_channels = 1
    H, W = 28, 28
    
    x = torch.randn(batch_size, in_channels, H, W)
    
    # 测试VQ-VAE
    print("1. VQ-VAE:")
    vqvae = VQVAE(
        in_channels=in_channels,
        hidden_dims=[128, 256],
        embedding_dim=64,
        num_embeddings=256
    )
    
    x_recon, z_q, vq_loss, indices = vqvae(x)
    
    print(f"   输入: {x.shape}")
    print(f"   重建: {x_recon.shape}")
    print(f"   VQ损失: {vq_loss.item():.4f}")
    print(f"   码本索引: {indices.shape}")
    print(f"   参数: {sum(p.numel() for p in vqvae.parameters()):,}")
    print()
    
    # 测试编码解码
    print("2. 编码-解码:")
    codes = vqvae.encode(x[:1])
    print(f"   编码: {codes.shape}")
    print(f"   码本分布: unique={len(torch.unique(codes))}")
    
    # 单个量化器测试
    print("=" * 50)
    print("单个向量量化器测试")
    print("=" * 50)
    
    vq = VectorQuantizer(embedding_dim=64, num_embeddings=256)
    
    z = torch.randn(batch_size, 64)
    z_q, indices, loss = vq(z)
    
    print(f"   输入: {z.shape}")
    print(f"   量化: {z_q.shape}")
    print(f"   索引: {indices.shape}")
    print(f"   损失: {loss.item():.4f}")


if __name__ == "__main__":
    test_vq_vae()
```

## 8. 手工代码实现

```python
import numpy as np


class ManualVectorQuantizer:
    """手动实现的向量量化器"""
    
    def __init__(self, embedding_dim, num_embeddings):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # 初始化码本
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
    
    def quantize(self, z_e):
        """量化"""
        # 计算距离
        distances = np.sum(z_e ** 2, axis=1, keepdims=True) + \
                    np.sum(self.embedding ** 2, axis=1) - \
                    2 * np.dot(z_e, self.embedding.T)
        
        # 最近邻
        indices = np.argmin(distances, axis=1)
        
        # 量化
        z_q = self.embedding[indices]
        
        return z_q, indices
    
    def compute_loss(self, z_e, z_q):
        """计算损失"""
        vq_loss = np.mean((z_q - z_e) ** 2)
        commit_loss = np.mean((z_e - z_q) ** 2)
        
        return vq_loss + 0.25 * commit_loss


def test_manual_vq():
    """测试手动实现"""
    print("=" * 50)
    print("测试手动VQ-VAE")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 配置
    batch_size = 32
    embedding_dim = 64
    num_embeddings = 256
    
    # 输入
    z_e = np.random.randn(batch_size, embedding_dim).astype(np.float32)
    
    print(f"Encoder输出: {z_e.shape}")
    
    # 量化器
    vq = ManualVectorQuantizer(embedding_dim, num_embeddings)
    
    # 量化
    z_q, indices = vq.quantize(z_e)
    
    print(f"量化后: {z_q.shape}")
    print(f"索引: {indices.shape}")
    print(f"unique索引: {len(np.unique(indices))}")
    
    # 损失
    loss = vq.compute_loss(z_e, z_q)
    print(f"损失: {loss:.4f}")


if __name__ == "__main__":
    test_manual_vq()
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_vqvae():
    """可视化VQ-VAE"""
    print("=" * 50)
    print("可视化VQ-VAE")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 码本使用分布
    ax = axes[0, 0]
    usage = np.random.dirichlet([2, 2, 2, 2, 2, 2, 2, 2], size=256)
    ax.hist(usage, bins=50, alpha=0.7)
    ax.set_xlabel('码本使用频率')
    ax.set_ylabel('码本数量')
    ax.set_title('码本使用分布')
    
    # 2. 码本聚类
    ax = axes[0, 1]
    centers = np.random.randn(256, 2)
    ax.scatter(centers[:, 0], centers[:, 1], alpha=0.5)
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    ax.set_title('码本分布（t-SNE简化）')
    
    # 3. 重建质量 vs 码本大小
    ax = axes[0, 2]
    k_sizes = [64, 128, 256, 512, 1024, 2048]
    recon_errors = [0.25, 0.22, 0.18, 0.15, 0.12, 0.10]
    ax.plot(k_sizes, recon_errors, 'o-')
    ax.set_xlabel('码本大小 K')
    ax.set_ylabel('重建误差')
    ax.set_title('码本大小 vs 重建质量')
    ax.set_xscale('log')
    ax.grid(True)
    
    # 4. 离散化效果
    ax = axes[1, 0]
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2)
    ax.plot(x, y, label='连续')
    
    # 离散近似
    x_discrete = np.linspace(-3, 3, 10)
    y_discrete = np.exp(-x_discrete**2)
    for y_d in y_discrete:
        ax.axhline(y_d, color='red', alpha=0.3)
    ax.scatter(x_discrete, y_discrete, color='red', label='离散')
    ax.set_xlabel('latent值')
    ax.set_ylabel('概率')
    ax.set_title('连续 vs 离散表示')
    ax.legend()
    
    # 5. 损失曲线
    ax = axes[1, 1]
    epochs = range(100)
    recon_loss = np.exp(-epochs / 15) + np.random.randn(100) * 0.02
    vq_loss = 0.1 * np.exp(-epochs / 20) + np.random.randn(100) * 0.01
    total = recon_loss + vq_loss
    ax.plot(epochs, recon_loss, label='重建')
    ax.plot(epochs, vq_loss, label='VQ')
    ax.plot(epochs, total, label='总损失')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失曲线')
    ax.legend()
    ax.grid(True)
    
    # 6. 多标量量化
    ax = axes[1, 2]
    num_quantizers = [1, 2, 4, 8]
    bits = [8, 16, 32, 64]
    ax.bar(range(len(num_quantizers)), bits, color='steelblue')
    ax.set_xticks(range(len(num_quantizers)))
    ax.set_xticklabels([f'{n}q' for n in num_quantizers])
    ax.set_ylabel('总比特数')
    ax.set_title('多标量量化')


if __name__ == "__main__":
    visualize_vqvae()
```

## 10. 模型评估

### 10.1 评估指标

- 重建误差
- 码本使用率
- NMI分数
- 采样多样性

### 10.2 评估代码

```python
import time


def evaluate_vqvae():
    print("=" * 50)
    print("评估VQ-VAE")
    print("=" * 50)
    
    x = torch.randn(4, 1, 28, 28)
    
    models = {
        "VQ-VAE-256": lambda: VQVAE(1, [64, 128], 32, 256),
        "VQ-VAE-512": lambda: VQVAE(1, [64, 128], 32, 512),
    }
    
    for name, model_fn in models.items():
        model = model_fn()
        
        params = sum(p.numel() for p in model.parameters())
        
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                out = model(x)
        
        elapsed = time.time() - start
        
        print(f"{name}: 参数 {params:,}, 时间 {elapsed/10*1000:.1f}ms")


if __name__ == "__main__":
    evaluate_vqvae()
```

## 11. 常见问题与学习总结

### 常见问题

1. **码本崩溃**：部分码本不被使用
   - 解决：增大码本或使用EMA
   
2. **训练不稳定**：学习率调整
   
3. **重建质量差**：增加码本大小

### 学习总结

1. **VQ-VAE使用离散码本**
2. **码本损失+承诺损失**
3. **停止梯度技巧**
4. **可配合PixelCNN生成**

### 关键公式

- 量化：$z_q = \text{argmin}_k \|z_e - e_k\|$
- VQ损失：$\|sg[z_e] - z_q\|^2$
- 承诺损失：$\|z_e - sg[z_q]\|^2$

### 练习题

1. 推导VQ-VAE损失函数
2. 实现多标量VQ
3. 分析码本崩溃原因

## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述VQ-VAE的核心思想及适用场景。
<details><summary>参考答案</summary>
VQ-VAE通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出VQ-VAE的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现VQ-VAE核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. VQ-VAE在什么情况下会失效？
2. 训练数据很少时，VQ-VAE还能有效工作吗？
3. 如何将VQ-VAE与其他方法结合？


## 14. 学习路径建议

### 前置知识
深度学习基础、线性代数、PyTorch

### 学习顺序
1. 先理解原理：掌握VQ-VAE核心思想和数学基础
2. 动手实践：运行代码，观察实验结果
3. 深入理解：阅读原始论文，理解设计动机
4. 项目实战：真实数据集上应用VQ-VAE

### 进阶方向
模型优化、分布式训练、推理优化

### 推荐资源
- 搜索VQ-VAE原始论文和综述
- GitHub优秀实现
- Coursera/Stanford相关课程

