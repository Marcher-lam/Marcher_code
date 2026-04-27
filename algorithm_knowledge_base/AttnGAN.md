# AttnGAN 学习文档

> 注意力生成式对抗网络（Attentional GAN），通过注意力机制关注文本中相关词汇，生成图像不同细粒度区域，实现文本到图像的高质量生成。

## 1. 算法基础认知

### 一句话定义

AttnGAN（Attention GAN）是一种利用注意力机制实现文本到图像（Text-to-Image）生成的生成对抗网络，通过在多个生成阶段关注文本中不同词汇，逐步生成从粗糙到精细的图像。

### 直觉类比

想象一个画家根据文字描述作画：
1. 先看全句"一只白色的猫在红色的沙发上睡觉"→画一个模糊的场景轮廓
2. 再看具体词汇"白色的猫"→细化猫的形状和颜色
3. 关注"红色的沙发"→细化沙发的颜色和纹理
4. 关注"睡觉"→调整猫的姿势

AttnGAN就是在不同生成阶段关注不同词汇，逐层细化图像。

### 历史背景

- **2017年11月**：AttnGAN论文（Tao Xu等人）发表在CVPR 2018
- **核心创新**：首次将注意力机制引入文本到图像生成领域
- **DAMSM模块**：深度注意力多模态相似度模型，实现词级别的图文匹配

### 算法定位

AttnGAN是**文本到图像生成模型**，属于生成对抗网络（GAN）在跨模态生成领域的应用，采用多阶段生成策略。

---

## 2. 核心原理

### 多层级生成架构

AttnGAN包含m个生成器-判别器对，逐层提高分辨率：

```
文本 → 文本编码器 → 词特征 + 句子特征
                                         ↓
Stage-1: 噪声 + 句子特征 → 粗糙图像 (64×64)
                                         ↓
Stage-2: 噪声 + 注意力加权词特征 → 细化 (128×128)
                                         ↓
Stage-3: 噪声 + 注意力加权词特征 → 精细 (256×256)
```

### DAMSM模块

DAMSM（Deep Attentional Multimodal Similarity Model）是AttnGAN的核心组件：
- 计算生成图像的每个区域与文本中每个词的匹配度
- 提供细粒度的视觉-语义损失

### 注意力机制

在生成器的每个阶段，计算图像特征与词特征的注意力权重：

$$c_j = \sum_{i=1}^{T} \alpha_{ji} e_i$$

其中 $c_j$ 是第 $j$ 个图像区域关注的文本上下文向量，$\alpha_{ji}$ 是注意力权重，$e_i$ 是第 $i$ 个词的词特征。

注意力权重计算：

$$\alpha_{ji} = \frac{\exp(s_{ji})}{\sum_{k=1}^{T} \exp(s_{jk})}$$

$$s_{ji} = \frac{h_j^T e_i}{\|h_j\|\|e_i\|}$$

其中 $h_j$ 是第 $j$ 个图像区域的特征。

---

## 3. 数学公式与推导

### 3.1 生成器损失

每个生成器的无条件损失：

$$\mathcal{L}_{G_i} = -\frac{1}{2} \mathbb{E}_{x \sim p_{G_i}} [\log D_i(x)] - \frac{1}{2} \mathbb{E}_{x \sim p_{G_i}} [\log D_i(x, s)]$$

其中第一项是无条件GAN损失，第二项是条件GAN损失（以文本句子特征 $s$ 为条件）。

### 3.2 判别器损失

每个判别器的损失：

$$\mathcal{L}_{D_i} = -\frac{1}{2} \mathbb{E}_{x \sim p_{data}} [\log D_i(x)] - \frac{1}{2} \mathbb{E}_{x \sim p_{G_i}} [\log (1 - D_i(x))]$$

$$- \frac{1}{2} \mathbb{E}_{x \sim p_{data}} [\log D_i(x, s)] - \frac{1}{2} \mathbb{E}_{x \sim p_{G_i}} [\log (1 - D_i(x, s))]$$

### 3.3 DAMSM损失（词级别匹配）

对于图像区域 $j$ 和词 $i$ 的匹配得分：

$$R(c_j, e_i) = \frac{c_j^T e_i}{\|c_j\|\|e_i\|}$$

图像到文本的注意力：

$$\beta_{ji} = \frac{\exp(R(c_j, e_i))}{\sum_{k=1}^{T} \exp(R(c_j, e_k))}$$

图像级文本表示：

$$s_i = \sum_{j=1}^{N} \beta_{ji} c_j$$

图像-文本匹配得分：

$$R(Q, D) = \log\left(\sum_{i=1}^{T} \exp(\gamma \cdot R(e_i, s_i))\right)^{\frac{1}{\gamma}}$$

DAMSM损失为：

$$\mathcal{L}_{DAMSM} = -\frac{1}{2} \mathbb{E}_{(Q,D) \sim p_{data}} [\log P(D|Q)] - \frac{1}{2} \mathbb{E}_{(Q,D) \sim p_{data}} [\log P(Q|D)]$$

### 3.4 总损失

$$\mathcal{L}_G = \sum_{i=0}^{m-1} \mathcal{L}_{G_i} + \lambda \mathcal{L}_{DAMSM}$$

其中 $\lambda$ 是平衡系数，控制DAMSM损失的权重。

---

## 4. 训练过程讲解

### 阶段一：文本编码

- 使用Bi-LSTM对文本描述编码
- 输出两个部分：
  - 句子特征 $s$（全局特征）
  - 词特征 $e_i$（每个位置的细粒度特征）

### 阶段二：多阶段图像生成

- Stage-1（64×64）：从噪声向量和句子特征生成粗糙图像
- Stage-2（128×128）：使用注意力模块关注相关词汇，细化图像
- Stage-3（256×256）：进一步关注细节词汇，生成高质量图像

### 阶段三：判别器判断

- 每个阶段的判别器判断生成图像的真实性
- 同时判断图像是否与文本描述匹配

### 阶段四：DAMSM损失计算

- 计算生成图像与文本描述的细粒度匹配度
- 提供额外的语义监督信号

### 训练技巧

- **多阶段训练**：逐阶段训练，先训练低分辨率再训练高分辨率
- **注意力引导**：DAMSM损失提供词级别的匹配信号
- **渐进式生成**：从粗糙到精细逐步细化

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 文本到图像生成 | 根据文字描述生成图像 | "一只蓝色眼睛的白猫" |
| 图像编辑 | 根据文字修改图像 | "把猫的颜色改成黑色" |
| 艺术创作 | AI辅助艺术创作 | 根据诗歌生成画作 |
| 数据增强 | 为CV任务生成训练数据 | 生成罕见场景的标注数据 |
| 跨模态检索 | 图文双向检索 | 根据描述搜索生成图像 |

---

## 6. 优缺点分析

### 优点

1. **细粒度生成**：注意力机制使模型能关注文本中的具体词汇，生成与描述更匹配的细节
2. **多阶段渐进**：从粗糙到精细的生成策略提高了图像质量和分辨率
3. **词级匹配**：DAMSM损失提供了比全局匹配更精确的语义信号
4. **可解释性**：注意力权重可以可视化，展示哪些词汇影响了哪些图像区域

### 缺点

1. **训练不稳定**：多阶段GAN训练难度大，容易模式坍塌
2. **计算量大**：多个生成器-判别器对和注意力计算消耗大量资源
3. **分辨率有限**：最高256×256，难以生成高清大图
4. **文本理解局限**：对复杂语义和长文本的理解有限
5. **多样性不足**：对同一文本描述生成的图像多样性不够

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextEncoder(nn.Module):
    """文本编码器：Bi-LSTM编码词和句子特征"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, 2, 
                           batch_first=True, bidirectional=True)
        
    def forward(self, text_ids):
        # text_ids: (B, L)
        emb = self.embedding(text_ids)  # (B, L, embed_dim)
        outputs, (hidden, cell) = self.lstm(emb)
        # outputs: (B, L, hidden_dim) - 每个位置的词特征
        # hidden: (4, B, hidden_dim//2) - 双向最后层
        sentence_feat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_dim)
        return outputs, sentence_feat

class AttentionModule(nn.Module):
    """注意力模块：计算图像区域与文本词汇的注意力"""
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, text_dim)
        
    def forward(self, image_feat, word_feat):
        """
        Args:
            image_feat: (B, N, image_dim) N个图像区域
            word_feat: (B, L, text_dim) L个词
        Returns:
            context: (B, N, text_dim) 每个区域的文本上下文
            attn: (B, N, L) 注意力权重
        """
        # 投影图像特征到文本空间
        img_proj = self.image_proj(image_feat)  # (B, N, text_dim)
        
        # 计算相似度矩阵
        img_proj = F.normalize(img_proj, dim=2)
        word_feat = F.normalize(word_feat, dim=2)
        
        # (B, N, L) 相似度矩阵
        sim = torch.bmm(img_proj, word_feat.transpose(1, 2))
        
        # Softmax得到注意力权重
        attn = F.softmax(sim, dim=2)  # (B, N, L)
        
        # 加权词特征作为每个图像区域的上下文
        context = torch.bmm(attn, word_feat)  # (B, N, text_dim)
        
        return context, attn

class GeneratorBlock(nn.Module):
    """单个生成器块"""
    def __init__(self, input_dim, output_channels=3, target_size=64):
        super().__init__()
        self.target_size = target_size
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256 * (target_size // 4) * (target_size // 4)),
            nn.BatchNorm1d(256 * (target_size // 4) * (target_size // 4)),
            nn.ReLU()
        )
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = self.net(x)
        x = x.view(B, 256, self.target_size // 4, self.target_size // 4)
        return self.up(x)

class AttnGANGenerator(nn.Module):
    """AttnGAN生成器（多阶段）"""
    def __init__(self, noise_dim=100, text_dim=256, hidden_dim=256):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder(hidden_dim=text_dim)
        
        # 注意力模块
        self.attn = AttentionModule(128, text_dim)  # 128为中间特征维度
        
        # 多阶段生成器
        self.gen_stage1 = GeneratorBlock(noise_dim + text_dim, 3, 64)
        self.gen_stage2 = GeneratorBlock(noise_dim + text_dim, 3, 128)
        self.gen_stage3 = GeneratorBlock(noise_dim + text_dim, 3, 256)
        
    def forward(self, noise, text_ids):
        """
        Args:
            noise: (B, noise_dim) 随机噪声
            text_ids: (B, L) 文本token IDs
        Returns:
            fake_images: list of 3阶段生成图像
            attn_weights: 注意力权重
        """
        # 文本编码
        word_feat, sent_feat = self.text_encoder(text_ids)
        
        fake_images = []
        attn_weights = []
        
        # Stage-1: 仅使用句子特征，生成粗糙图像
        gen_input = torch.cat([noise, sent_feat], dim=1)
        img1 = self.gen_stage1(gen_input)
        fake_images.append(img1)
        
        # Stage-2: 使用注意力机制
        # 从img1提取中间特征
        img_feat = img1.view(img1.shape[0], -1, img1.shape[2] * img1.shape[3])  # (B, 3, 4096)
        img_feat = img_feat.transpose(1, 2)  # (B, 4096, 3)
        
        # 注意力：图像区域关注文本词汇
        context, attn1 = self.attn(img_feat, word_feat)
        context_pooled = context.mean(dim=1)  # (B, text_dim)
        
        gen_input2 = torch.cat([noise, context_pooled], dim=1)
        img2 = self.gen_stage2(gen_input2)
        fake_images.append(img2)
        attn_weights.append(attn1)
        
        # Stage-3: 进一步细化
        img_feat2 = img2.view(img2.shape[0], -1, img2.shape[2] * img2.shape[3])
        img_feat2 = img_feat2.transpose(1, 2)
        context2, attn2 = self.attn(img_feat2, word_feat)
        context_pooled2 = context2.mean(dim=1)
        
        gen_input3 = torch.cat([noise, context_pooled2], dim=1)
        img3 = self.gen_stage3(gen_input3)
        fake_images.append(img3)
        attn_weights.append(attn2)
        
        return fake_images, attn_weights

class DiscriminatorBlock(nn.Module):
    """判别器块"""
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(512 * 4 * 4, 1)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

class AttnGAN(nn.Module):
    """完整AttnGAN模型"""
    def __init__(self, noise_dim=100, text_dim=256):
        super().__init__()
        self.generator = AttnGANGenerator(noise_dim, text_dim)
        self.discriminator_stage1 = DiscriminatorBlock(3)
        self.discriminator_stage2 = DiscriminatorBlock(3)
        self.discriminator_stage3 = DiscriminatorBlock(3)
        
    def forward(self, noise, text_ids):
        return self.generator(noise, text_ids)

# 训练循环示例
def train_attngan_step(model, optimizer_G, optimizer_D, real_images, 
                       text_ids, noise, lambda_damsm=5.0):
    """
    单步训练
    Args:
        real_images: list of 3个分辨率的真实图像 [64, 128, 256]
        text_ids: (B, L) 文本
        noise: (B, 100) 噪声
    """
    B = text_ids.shape[0]
    
    # 训练判别器
    optimizer_D.zero_grad()
    fake_images, _ = model.generator(noise, text_ids)
    
    d_loss = 0
    for i, (disc, real, fake) in enumerate(zip(
        [model.discriminator_stage1, model.discriminator_stage2, model.discriminator_stage3],
        real_images, fake_images)):
        
        # 调整图像大小以匹配判别器输入
        if real.shape[-1] != fake.shape[-1]:
            real = F.interpolate(real, size=fake.shape[-1], mode='bilinear')
        
        real_logits = disc(real)
        fake_logits = disc(fake.detach())
        
        d_loss += (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()) / 2
    
    d_loss.backward()
    optimizer_D.step()
    
    # 训练生成器
    optimizer_G.zero_grad()
    fake_images, attn_weights = model.generator(noise, text_ids)
    
    g_loss = 0
    for i, (disc, fake) in enumerate(zip(
        [model.discriminator_stage1, model.discriminator_stage2, model.discriminator_stage3],
        fake_images)):
        fake_logits = disc(fake)
        g_loss += -fake_logits.mean()
    
    g_loss.backward()
    optimizer_G.step()
    
    return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}

if __name__ == "__main__":
    # 测试模型
    model = AttnGAN()
    noise = torch.randn(4, 100)
    text_ids = torch.randint(0, 10000, (4, 20))
    
    fake_images, attn_weights = model(noise, text_ids)
    
    print(f"生成图像数量: {len(fake_images)}")
    print(f"Stage-1: {fake_images[0].shape}")
    print(f"Stage-2: {fake_images[1].shape}")
    print(f"Stage-3: {fake_images[2].shape}")
    print("AttnGAN前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLSTM(nn.Module):
    """手工实现带注意力的LSTM文本编码器"""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM参数（手工）
        self.W_i = nn.Linear(embed_dim + hidden_dim, hidden_dim * 4)
        self.W_h = nn.Linear(hidden_dim, hidden_dim * 4)
        
    def lstm_cell(self, x, h, c):
        """单步LSTM"""
        gates = self.W_i(torch.cat([x, h], dim=1)) + self.W_h(h)
        i, f, o, g = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
    
    def forward(self, text_ids):
        B, L = text_ids.shape
        emb = self.embedding(text_ids)
        
        # 正向LSTM
        h_f = torch.zeros(B, self.hidden_dim // 2)
        c_f = torch.zeros(B, self.hidden_dim // 2)
        h_forwards = []
        
        for t in range(L):
            h_f, c_f = self.lstm_cell(emb[:, t], h_f, c_f)
            h_forwards.append(h_f.unsqueeze(1))
        
        # 反向LSTM
        h_b = torch.zeros(B, self.hidden_dim // 2)
        c_b = torch.zeros(B, self.hidden_dim // 2)
        h_backwards = []
        
        for t in range(L - 1, -1, -1):
            h_b, c_b = self.lstm_cell(emb[:, t], h_b, c_b)
            h_backwards.append(h_b.unsqueeze(1))
        
        h_backwards = list(reversed(h_backwards))
        
        # 拼接双向特征
        word_feat = torch.cat(h_forwards + h_backwards, dim=2)  # (B, L, hidden_dim)
        
        # 句子特征：取平均
        sent_feat = word_feat.mean(dim=1)  # (B, hidden_dim)
        
        return word_feat, sent_feat

class HandcraftAttention(nn.Module):
    """手工注意力计算模块"""
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, key_dim)
        
    def forward(self, query, key, value):
        """
        query: (B, N_q, D_q) - 图像区域特征
        key: (B, N_k, D_k) - 词特征
        value: (B, N_k, D_v) - 词特征
        """
        # 投影查询
        q_proj = F.normalize(self.W_q(query), dim=2)
        k_proj = F.normalize(key, dim=2)
        
        # 相似度矩阵
        scores = torch.bmm(q_proj, k_proj.transpose(1, 2))  # (B, N_q, N_k)
        
        # 注意力权重
        attn = F.softmax(scores * math.sqrt(key.shape[-1]), dim=2)
        
        # 加权上下文
        context = torch.bmm(attn, value)  # (B, N_q, D_v)
        
        return context, attn

class HandcraftDAMSMLoss(nn.Module):
    """手工实现DAMSM损失（词级别匹配损失）"""
    def __init__(self, gamma=5.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, image_feat, word_feat):
        """
        计算图像-文本词级别匹配损失
        Args:
            image_feat: (B, N_img, D) 图像区域特征
            word_feat: (B, N_word, D) 词特征
        """
        B = image_feat.shape[0]
        
        # 归一化
        image_feat = F.normalize(image_feat, dim=2)
        word_feat = F.normalize(word_feat, dim=2)
        
        # 相似度矩阵 (B, N_img, N_word)
        sim = torch.bmm(image_feat, word_feat.transpose(1, 2))
        
        # 图像到文本的注意力
        attn_i2t = F.softmax(sim * self.gamma, dim=2)
        
        # 文本到图像的注意力
        attn_t2i = F.softmax(sim.transpose(1, 2) * self.gamma, dim=2)
        
        # 加权图像级文本特征
        text_context = torch.bmm(attn_i2t, word_feat)  # (B, N_img, D)
        
        # 加权文本级图像特征
        image_context = torch.bmm(attn_t2i, image_feat)  # (B, N_word, D)
        
        # 相似度得分
        s_i2t = torch.sum(image_feat * text_context, dim=2)  # (B, N_img)
        s_t2i = torch.sum(word_feat * image_context, dim=2)  # (B, N_word)
        
        # 损失：图像到文本和文本到图像的对比
        loss_i2t = -torch.log(torch.sum(torch.exp(s_i2t), dim=1) + 1e-8).mean()
        loss_t2i = -torch.log(torch.sum(torch.exp(s_t2i), dim=1) + 1e-8).mean()
        
        return loss_i2t + loss_t2i

# 测试手工实现
if __name__ == "__main__":
    # 测试文本编码器
    text_encoder = AttentionLSTM(vocab_size=10000, embed_dim=256, hidden_dim=512)
    text_ids = torch.randint(0, 10000, (2, 15))
    word_feat, sent_feat = text_encoder(text_ids)
    
    print(f"词特征形状: {word_feat.shape}")  # (2, 15, 512)
    print(f"句子特征形状: {sent_feat.shape}")  # (2, 512)
    
    # 测试注意力模块
    attn = HandcraftAttention(128, 512)
    img_feat = torch.randn(2, 64, 128)
    context, attn_weights = attn(img_feat, word_feat, word_feat)
    
    print(f"上下文特征形状: {context.shape}")  # (2, 64, 512)
    print(f"注意力权重形状: {attn_weights.shape}")  # (2, 64, 15)
    
    # 测试DAMSM损失
    damsmloss = HandcraftDAMSMLoss()
    loss = damsmloss(img_feat, word_feat)
    print(f"DAMSM损失: {loss.item():.4f}")
    
    print("\n所有手工模块测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化

AttnGAN最直观的可视化是注意力热力图，展示文本词汇与图像区域的对应关系：
- "猫" → 猫所在的图像区域变亮
- "红色" → 沙发区域变亮
- "睡觉" → 猫的姿势区域变亮

### 9.2 多阶段生成结果

多阶段生成的逐步细化过程：
- Stage-1：模糊的形状和颜色布局
- Stage-2：轮廓更清晰，纹理开始显现
- Stage-3：细节丰富，质量接近真实图像

### 9.3 训练过程中的损失曲线

- 生成器损失：逐渐下降但伴随震荡
- 判别器损失：维持在0附近
- DAMSM损失：持续下降表明图文匹配度提高

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| IS（Inception Score） | 生成图像的质量和多样性 | Inception v3分类的熵 |
| FID（Fréchet Inception Distance） | 生成分布与真实分布的距离 | 特征分布的均值和协方差差异 |
| R-precision | 图文匹配准确率 | 检索匹配文本的召回率 |
| 人工评分 | 人类对生成质量的评价 | 1-5分评分 |

### 10.2 FID计算示例

```python
def calculate_fid(real_features, fake_features):
    """计算FID分数"""
    mu_real = real_features.mean(dim=0)
    mu_fake = fake_features.mean(dim=0)
    
    cov_real = torch.cov(real_features.T)
    cov_fake = torch.cov(fake_features.T)
    
    diff = mu_real - mu_fake
    cov_mean = cov_real + cov_fake
    
    fid = diff.dot(diff) + torch.trace(cov_real + cov_fake - 2 * torch.linalg.sqrtm(cov_mean))
    return fid.item()
```

---

## 11. 常见问题与易错点

### Q1: AttnGAN和普通GAN的区别是什么？

普通GAN从噪声生成图像，没有条件控制。AttnGAN以文本为条件，通过注意力机制确保生成内容与文本描述一致。

### Q2: 为什么需要多阶段生成？

单阶段直接生成高分辨率图像往往质量差。多阶段从低分辨率到高分辨率逐步细化，每个阶段只负责增加细节，降低了生成难度。

### Q3: DAMSM损失为什么有效？

DAMSM在词级别计算图像与文本的匹配度，比全局句子级别更精细。例如"白色的猫在红色的沙发上"，全局匹配无法区分哪个区域对应"白色"哪个对应"红色"，DAMSM可以。

### Q4: 注意力机制如何帮助生成？

在每个生成阶段，注意力模块让生成器"看到"文本中相关的词汇，从而知道要细化哪些区域。这类似于给画家提供参考文本。

### Q5: 训练不稳定的常见原因？

- 多阶段GAN的平衡难以控制
- DAMSM损失权重过大或过小
- batch size太小导致判别器梯度不稳定

---

## 12. 学习总结

### 核心知识点

1. **多阶段生成**：从64×64到256×256逐步细化
2. **注意力机制**：每个生成阶段关注不同词汇
3. **DAMSM损失**：词级别的图文匹配损失
4. **文本编码器**：Bi-LSTM提取词特征和句子特征

### 架构速记

AttnGAN = Bi-LSTM文本编码器 + 多阶段生成器 + 多阶段判别器 + DAMSM
文本 → 词特征 → 注意力 → 上下文 → 生成图像

### 关键洞见

注意力将文本到图像生成从"全局条件控制"提升到"细粒度局部控制"层面，让生成的每个图像区域都能找到对应的文本描述词汇。

---

## 13. 练习题与思考题（含答案）

### 习题1：注意力机制

**问题**：假设有5个词的特征和16个图像区域，注意力权重矩阵的形状是什么？

**答案**：(16, 5)。每个图像区域对每个词有一个注意力权重。

### 习题2：多阶段生成

**问题**：为什么Stage-1不使用注意力机制？

**答案**：Stage-1生成粗糙的轮廓和布局，只需要句子级别的全局信息即可。注意力机制主要帮助细化阶段关注具体的词汇细节。

### 习题3：DAMSM损失

**问题**：DAMSM损失在计算时为什么需要图像到文本和文本到图像两个方向的注意力？

**答案**：对称设计确保图文双向匹配。图像到文本确保每个区域找到对应的描述词，文本到图像确保每个词在图像中有对应的区域。

### 习题4：对比AttnGAN和StackGAN

**问题**：AttnGAN相对于StackGAN的核心改进是什么？

**答案**：StackGAN是简单的多阶段GAN，每个阶段都用相同的句子特征。AttnGAN引入注意力机制，让不同阶段关注不同的词汇，实现更细粒度的控制。

### 习题5：思考题

**问题**：如果移除了DAMSM损失，AttnGAN还能正常工作吗？效果会怎样？

**答案**：能工作，但生成图像与文本的一致性会大幅下降。DAMSM损失提供了词级别的语义监督，没有它模型只能通过判别器的条件判断来学习图文一致性，这种全局信号对细节控制不够。

---

## 14. 学习路径建议

### 前置知识
- GAN（生成对抗网络）基础
- LSTM / 循环神经网络
- 注意力机制基础
- 图像处理基础（CNN）

### 平行模型
- **StackGAN**：多阶段GAN的先驱（无注意力）
- **StackGAN++**：StackGAN的改进版
- **ControlGAN**：更精细的文本控制GAN

### 进阶方向
- **DF-GAN**：简化高效的文本到图像生成
- **DALL-E**：基于Transformer的文本到图像生成
- **Stable Diffusion**：扩散模型在文本到图像中的应用
- **Imagen**：Google的高质量文本到图像模型

### 学习顺序建议

```
① GAN基础 → ② 条件GAN → ③ 注意力机制 → ④ AttnGAN → ⑤ DALL-E / Stable Diffusion
```
