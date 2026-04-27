# DALL-E 2（又名unCLIP）学习文档

> 结合CLIP和扩散模型的图像生成模型。

## 1. 算法基础认知

### 一句话定义

DALL-E 2是OpenAI于2022年4月发布的文本到图像模型，结合CLIP特征表示和扩散模型生成。

### 历史背景

- **2022年4月**：DALL-E 2发布
- **核心创新**：CLIP引导的扩散模型

### 算法定位

DALL-E 2是**文本到图像生成模型**，基于扩散模型。

---

## 2. 核心原理

### 三阶段生成

1. **CLIP文本编码**：文本 → 文本特征
2. **先验模型**：文本特征 → 图像特征（扩散或自回归）
3. **解码器**：图像特征 → 图像（扩散模型GLIDE）

### 无分类器引导

$$\epsilon_\theta(x_t, c) = (1-w)\epsilon_\theta(x_t) + w\epsilon_\theta(x_t, c)$$

其中c是文本条件，w是引导强度。

---

## 3. 代码实现

```python
import torch
import torch.nn as nn

class DALLE2(nn.Module):
    """DALL-E 2简化实现"""
    def __init__(self, clip_dim=512, image_dim=768):
        super(DALLE2, self).__init__()
        
        # CLIP文本编码器（冻结）
        from transformers import CLIPTextModel, CLIPTokenizer
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.eval()
        for p in self.clip_text.parameters():
            p.requires_grad = False
            
        # 先验模型（文本→图像特征）
        self.prior = DiffusionPrior(clip_dim=512, image_dim=768)
        
        # 解码器（GLIDE风格的扩散模型）
        self.decoder = DiffusionDecoder(image_dim=768)
        
    def forward(self, text):
        # CLIP文本编码
        with torch.no_grad():
            text_features = self.clip_text(text).last_hidden_state
        
        # 先验：文本特征 → 图像特征
        image_features = self.prior(text_features)
        
        # 解码：图像特征 → 图像
        images = self.decoder(image_features)
        
        return images

class DiffusionPrior(nn.Module):
    """扩散先验：将文本特征转为图像特征"""
    def __init__(self, clip_dim=512, image_dim=768):
        super(DiffusionPrior, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(256, image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, image_dim)
        )
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(image_dim, 8, image_dim * 4),
            12
        )
        
    def forward(self, text_features, t=None):
        B = text_features.shape[0]
        
        if t is None:
            t = torch.randint(0, 1000, (B,))
            
        # 时间嵌入
        t_emb = self.get_timestep_embedding(t, 256)
        t_emb = self.time_embed(t_emb).unsqueeze(1)
        
        # 融合文本特征和时间
        x = text_features.unsqueeze(1) + t_emb
        x = self.model(x)
        
        return x.squeeze(1)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class DiffusionDecoder(nn.Module):
    """扩散解码器"""
    def __init__(self, image_dim=768):
        super(DiffusionDecoder, self).__init__()
        # 简化的U-Net结构
        self.time_embed = nn.Sequential(
            nn.Linear(256, image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, image_dim)
        )
        
        # 上采样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(image_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1)
        )
        
    def forward(self, image_features, t=None):
        B = image_features.shape[0]
        
        if t is None:
            t = torch.randint(0, 1000, (B,))
            
        t_emb = self.time_embed(self.get_timestep_embedding(t, 256))
        
        # 简化：直接reshape为2D
        x = image_features.unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb.view(B, -1, 1, 1)
        
        # 上采样生成图像
        x = x.repeat(1, 1, 4, 4)  # 简化：空间扩展
        x = self.decoder(x)
        
        return torch.sigmoid(x)
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

# 推理
def generate_image(prompt):
    dalle2 = DALLE2()
    dalle2.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        images = dalle2(text.input_ids)
        
    return images

if __name__ == "__main__":
    print("DALL-E 2模型已定义")
```

---

## 4. 训练两阶段

**阶段1**：训练CLIP（对比学习）
**阶段2**：
- 训练先验（文本特征→图像特征）
- 训练解码器（图像特征→图像）

---

## 5. 学习路径

- 前置：DALL-E, CLIP, 扩散模型
- 进阶：DALL-E 3, Stable Diffusion