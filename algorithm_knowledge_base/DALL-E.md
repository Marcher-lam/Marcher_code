# DALL-E 学习文档

> OpenAI提出的第一个文本到图像生成模型。

## 1. 算法基础认知

### 一句话定义

DALL-E是OpenAI于2021年2月提出的首个零样本文本到图像生成模型，结合dVAE和Transformer。

### 历史背景

- **2021年2月**：DALL-E论文发布
- **参数量**：120亿
- **核心创新**：两阶段文本到图像生成

### 算法定位

DALL-E是**文本到图像生成模型**，属于多模态生成模型。

---

## 2. 核心原理

### 两阶段生成

1. **阶段一**：dVAE训练视觉码本
2. **阶段二**：Transformer学习文本到图像转换

### 生成流程

```
文本 → 文本编码器 → 文本特征 → Transformer → 图像特征 → dVAE解码器 → 图像
```

### 关键设计

- dVAE: 32×32=1024个视觉token，8192词汇表
- Transformer: 64层，62头
- 重排：使用CLIP选择最佳图像

---

## 3. 代码实现

```python
import torch
import torch.nn as nn

class DALLEModel(nn.Module):
    """DALL-E简化实现"""
    def __init__(self, text_vocab=16384, image_vocab=8192, d_model=1024):
        super(DALLEModel, self).__init__()
        self.text_vocab = text_vocab
        self.image_vocab = image_vocab
        
        # 文本编码器
        self.text_embed = nn.Embedding(text_vocab, d_model)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, 256, d_model))
        
        # 图像token编码器
        self.image_embed = nn.Embedding(image_vocab, d_model)
        self.image_pos_embed = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        # Transformer解码器（自回归）
        decoder_layer = nn.TransformerEncoderLayer(d_model, 16, d_model * 4)
        self.transformer = nn.TransformerEncoder(decoder_layer, 64)
        
        # 输出头
        self.head = nn.Linear(d_model, image_vocab)
        
    def forward(self, text_ids, image_ids=None):
        """
        text_ids: (B, text_len)
        image_ids: (B, image_len) - 训练时使用
        """
        B = text_ids.shape[0]
        
        # 文本编码
        text_emb = self.text_embed(text_ids) + self.text_pos_embed[:, :text_ids.size(1), :]
        
        if image_ids is not None:
            # 训练模式：连接文本和图像token
            image_emb = self.image_embed(image_ids) + self.image_pos_embed[:, :image_ids.size(1), :]
            x = torch.cat([text_emb, image_emb], dim=1)
        else:
            # 推理模式：仅文本
            x = text_emb
            
        # Transformer处理
        x = self.transformer(x)
        
        # 输出图像token预测
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, text_ids, image_token_count=1024):
        """自回归生成图像token"""
        self.eval()
        B = text_ids.shape[0]
        
        # 文本编码
        text_emb = self.text_embed(text_ids) + self.text_pos_embed[:, :text_ids.size(1), :]
        generated = text_emb
        
        # 逐个生成图像token
        for _ in range(image_token_count):
            x = self.transformer(generated)
            next_token_logits = x[:, -1, :]  # 最后一个位置的输出
            
            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加位置编码
            new_pos = self.image_pos_embed[:, generated.shape[1]:generated.shape[1]+1, :]
            new_emb = self.image_embed(next_token) + new_pos
            
            generated = torch.cat([generated, new_emb], dim=1)
            
        return generated[:, text_ids.size(1):, :]  # 返回图像token

# dVAE（离散VAE）
class dVAE(nn.Module):
    """离散VAE用于图像token化"""
    def __init__(self, vocab_size=8192, hidden=256):
        super(dVAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, vocab_size, 1)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(vocab_size, hidden, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        # 编码
        logits = self.encoder(x)
        # 重参数化
        probs = torch.softmax(logits, dim=1)
        token = torch.argmax(probs, dim=1)
        # 解码
        one_hot = F.one_hot(token, logits.shape[1]).float().permute(0, 3, 1, 2)
        recon = self.decoder(one_hot)
        return recon, token

if __name__ == "__main__":
    # 测试
    dalle = DALLEModel()
    text_ids = torch.randint(0, 16384, (2, 20))
    image_ids = torch.randint(0, 8192, (2, 1024))
    
    output = dalle(text_ids, image_ids)
    print(f"输出形状: {output.shape}")
```

---

## 4. 性能

- 在文本到图像生成任务上实现零样本能力
- 可以进行图像编辑（改变物体位置等）

---

## 5. 学习路径

- 前置：CLIP, VAE, Transformer
- 进阶：DALL-E 2, Stable Diffusion