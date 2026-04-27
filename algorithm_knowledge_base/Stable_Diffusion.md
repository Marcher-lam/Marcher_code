# Stable Diffusion 学习文档

> 基于潜在扩散模型的高效图像生成系统。

## 1. 算法基础认知

### 一句话定义

Stable Diffusion是 Stability AI 发布的开源图像生成模型，使用LDM（潜在扩散模型）实现高效文生图。

### 历史背景

- **2022年8月**：Stable Diffusion发布
- **核心创新**：潜在空间扩散，大幅降低计算量

### 算法定位

Stable Diffusion是**开源图像生成模型**，属于LDM系列。

---

## 2. 核心原理

### LDM架构

1. **自编码器**：图像 → 潜在空间 → 图像
2. **扩散模型**：在潜在空间进行去噪
3. **条件编码器**：文本 → 潜在空间条件

### 核心优势

- 仅在低维潜在空间操作，计算效率高
- 支持条件生成（文本、图像）

---

## 3. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

class LDMConditional(nn.Module):
    """条件潜在扩散模型"""
    def __init__(self, latent_dim=4, hidden=128, text_dim=768):
        super(LDMConditional, self).__init__()
        self.latent_dim = latent_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(256, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )
        
        # 文本条件编码
        self.text_encoder = nn.Linear(text_dim, hidden)
        
        # U-Net结构
        self.down1 = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden, hidden*2, 3, stride=2, padding=1),
            nn.GroupNorm(64, hidden*2),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden*2, hidden, 4, 2, 1),
            nn.GroupNorm(32, hidden),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU()
        )
        
    def forward(self, x, t, text_cond):
        # 时间嵌入
        t_emb = self.get_timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)
        
        # 文本条件
        text_emb = self.text_encoder(text_cond)
        
        # 下采样
        h1 = self.down1(x + t_emb.unsqueeze(-1).unsqueeze(-1))
        h2 = self.down2(h1 + text_emb.unsqueeze(-1).unsqueeze(-1))
        
        # 上采样
        h3 = self.up1(h2)
        out = self.up2(h3 + text_emb.unsqueeze(-1).unsqueeze(-1))
        
        return out
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class AutoEncoder(nn.Module):
    """VAE编码器和解码器"""
    def __init__(self, in_channels=3, latent_dim=4):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, 2, 1)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1)
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

# 使用HuggingFace diffusers
def generate_with_stable_diffusion(prompt):
    """使用预训练模型生成"""
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    image = pipe(prompt).images[0]
    return image

# 本地简化实现
def simple_generation():
    """简化推理流程"""
    # 假设已有模型
    latent_model = LDMConditional(latent_dim=4)
    vae = AutoEncoder(in_channels=3, latent_dim=4)
    
    # 随机初始噪声
    z = torch.randn(1, 4, 32, 32)
    
    # 扩散去噪（简化：直接使用VAE解码）
    # 实际使用DDPM/DDIM采样
    with torch.no_grad():
        # 简化的文本条件（用随机向量模拟）
        text_cond = torch.randn(1, 768)
        
        # 多次迭代去噪
        for _ in range(50):
            z = z - 0.01 * latent_model(z, torch.tensor([50]), text_cond)
        
        # 解码为图像
        image = vae.decode(z)
        
    return image

if __name__ == "__main__":
    result = simple_generation()
    print(f"生成图像形状: {result.shape}")
```

---

## 4. 性能对比

| 模型 | 参数量 | 生成速度 | 质量 |
|------|--------|----------|------|
| DALL-E 2 | ~10B | 慢 | 高 |
| Stable Diffusion | ~1B | 快 | 高 |
| Midjourney | - | 中 | 高 |

---

## 5. 学习路径

- 前置：扩散模型, VAE, CLIP
- 进阶：SD XL, ControlNet