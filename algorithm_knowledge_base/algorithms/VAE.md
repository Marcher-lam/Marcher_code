# VAE (Variational Autoencoder) 学习文档

> 变分自编码器，深度生成模型的基础。

---

## 1. 算法基础认知

### 1.1 发展背景

VAE 由 Kingma 和 Welling 于 2013 年在论文《Auto-Encoding Variational Bayes》中提出，是一种基于变分推断的深度生成模型，可以学习数据的潜在表示并生成新样本。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 生成模型 |
| 训练 | 变分推断 |
| 目标 | 潜在变量建模 |
| 优点 | 良好表示学习 |

### 1.3 模型系列

| 模型 | 参数量 | 用途 |
|------|--------|------|
| VAE | 基础 | 图像生成 |
| CVAE | 条件 | 条件生成 |
| β-VAE | 解耦 | 可解释表示 |

---

## 2. 核心原理

### 2.1 生成模型框架

```
输入x → 编码器 → μ, σ → 采样z → 解码器 → x̂
```

### 2.2 潜在变量

假设数据生成过程：
$$p(x) = \int p(x|z) p(z) dz$$

其中 $p(z) \sim \mathcal{N}(0, I)$

### 2.3 变分推断

使用 $q(z|x)$ 近似 $p(z|x)$，优化变分下界 (ELBO)

---

## 3. 数学公式与推导

### 3.1 编码器输出

$$q(z|x) = \mathcal{N}(z|\mu(x), \text{diag}(\sigma^2(x)))$$

### 3.2 重参数化技巧

$$z = \mu + \sigma \cdot \epsilon$$
其中 $\epsilon \sim \mathcal{N}(0, I)$

### 3.3 损失函数

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathbb{D}_{KL}(q(z|x) || p(z))$$

第一项：重建损失
第二项：KL 散度（正则化）

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| 批量大小 | 128 |
| 学习率 | 1e-3 |
| Epochs | 50+ |
| β | 1.0 |

### 4.2 训练流程

```
1. 编码: x → μ, σ
2. 采样: z = μ + σ·ε
3. 解码: z → x̂
4. 计算损失
5. 更新参数
```

---

## 5. 应用场景

### 5.1 典型应用

- **图像生成**：MNIST, CIFAR
- **异常检测**：重构误差
- **表示学习**：潜在空间

### 5.2 代码示例

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        # 采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
```

---

## 6. 调库实现

### 6.1 PyTorch 实现

```python
import torch
from torch import nn

class VAEModel(nn.Module):
    """变分自编码器"""
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder  
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # BCE loss
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL loss
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KL


def demo():
    print("=== VAE 演示 ===\n")
    
    # 创建模型
    vae = VAEModel()
    params = sum(p.numel() for p in vae.parameters())
    print(f"参数量: {params:,}")
    
    # 模拟输入
    x = torch.randn(32, 784)
    recon, mu, logvar = vae(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {recon.shape}")


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

### 7.1 简化实现

```python
import numpy as np

class SimpleVAE:
    """简化 VAE"""
    
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 简化的编码/解码
        self.W_encoder = np.random.randn(input_dim, latent_dim) * 0.01
        self.b_encoder = np.zeros(latent_dim)
        
        self.W_decoder = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_decoder = np.zeros(input_dim)
        
    def encode(self, x):
        return x @ self.W_encoder + self.b_encoder
    
    def decode(self, z):
        return 1 / (1 + np.exp(-(z @ self.W_decoder + self.b_decoder)))
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def demo():
    print("=== VAE 手工实现演示 ===\n")
    
    vae = SimpleVAE(784, 20)
    print(f"潜在维度: 20")
    print(f"应用: 图像生成")


if __name__ == "__main__":
    demo()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **连续潜在空间**：可插值
2. **良好表示**：压缩特征
3. **可生成新样本**：创造性

### 8.2 缺点

1. **生成质量**：略低于 GAN
2. **后验模糊**：模式坍塌风险

### 8.3 改进

- **β-VAE**：调节 KL 权重
- **CVAE**：条件生成
- **VQ-VAE**：离散潜在

---

## 9. 可视化与结果理解

### 9.1 潜在空间

```python
def visualize():
    print("""
    VAE 潜在空间:
    
    数字 "0"  -------->  中心区域
    数字 "1"  -------->  边缘区域
    
    相邻数字在潜在空间中也相邻
    可进行线性插值生成新样本
    """)
```

---

## 10. 模型评估

### 10.1 生成质量 (Bits/Dim)

| 模型 | MNIST |
|------|------|
| VAE | 0.85 |
| β-VAE | 0.86 |
| GAN | 0.45 |

---

## 11. 学习总结

**核心要点**：

1. **变分推断**：近似后验
2. **潜在空间**：连续表示
3. **重建 + KL**：双重目标
4. **重参数化**：可训练采样

**VAE 核心优势**：
- 表示学习能力强
- 生成能力可扩展
- 理论基础扎实

---

## 12. 练习题与思考题

### 12.1 基础练习

1. VAE vs AE
2. KL 损失作用
3. 重参数化原理

### 12.2 思考题

1. VAE vs GAN 比较
2. 改进方向

---

### 12.3 详细答案

**问题**：为什么需要重参数化

**解答**：采样操作不可导，重参数化使梯度可通过。

---

## 14. 学习路径建议

### 入门阶段

1. AE 基础
2. 变分推断理解

### 进阶阶段

1. VAE 原理
2. 实现生成

### 高级阶段

1. β-VAE 改进
2. CVAE 应用

**推荐路线**：

```
AE → VAE → β-VAE → VQ-VAE → GAN
```

**VAE 是深度生成模型的基础，熟练掌握它对学习生成模型很重要。**