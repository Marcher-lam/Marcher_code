# iGPT（image GPT）学习文档

> 将GPT架构直接应用于图像的先驱模型，首次证明NLP领域的自回归预训练范式可以迁移到计算机视觉领域，为后续ViT、MAE等模型奠定了基础。

## 1. 算法基础认知

### 一句话定义

iGPT（image GPT）是OpenAI提出的将GPT的像素级自回归预训练范式直接应用于图像的模型，将图像视为像素序列，通过预测下一个像素来学习视觉表示。

### 直觉类比

iGPT就像一个"像素级看图说话"——它不看整只猫，而是一个像素一个像素地预测：
- 看了100个像素后，预测第101个像素的颜色
- 看了1000个像素后，预测第1001个像素的颜色
- ...
通过这种方式，它学会了"像素之间的规律"，进而理解了图像的整体结构。

### 历史背景

- **2020年7月**：OpenAI发布iGPT论文
- **参数量**：最大版本68亿参数（iGPT-XL）
- **核心创新**：首次将GPT自回归范式应用于像素级别
- **后续影响**：证明了自回归预训练可以作为CV的统一范式，启发了ViT、MAE、BEiT等工作

### 算法定位

iGPT是**纯Transformer的CV预训练模型**，属于生成式无监督学习，将NLP的"下一个词预测"范式转化为"下一个像素预测"。

---

## 2. 核心原理

### 像素级建模

iGPT的核心思想：将图像视为像素序列，用自回归方式建模。

```
原始图像 (32×32=1024像素):
[R(1), G(1), B(1), R(2), G(2), B(2), ..., R(1024), G(1024), B(1024)]

自回归预测:
P(x_t | x_<t) → 根据前t-1个像素预测第t个像素
```

### 颜色离散化

原始像素值是0-255的连续值，直接预测太困难。iGPT使用k-means聚类将颜色离散化：

1. 在ImageNet上对所有像素颜色做k-means聚类
2. 得到512个颜色中心（词汇量=512）
3. 每个像素被映射到最近的颜色中心
4. 预测变为"512分类"问题

### 两种预训练模式

iGPT支持两种自监督预训练方式：

1. **自回归模式（AR）**：GPT风格，逐像素预测下一个像素
   ```
   输入:   [CLS] p1 p2 p3 p4 [MASK] p6 p7 ...
   预测:         p1 p2 p3 p4  p5    p6 p7 ...
   ```

2. **自编码模式（AE）**：BERT风格，随机Mask部分像素并预测
   ```
   输入:   [CLS] p1 [MASK] p3 p4 [MASK] p6 ...
   预测:         p1  p2    p3 p4  p5    p6 ...
   ```

### 模型结构

| 版本 | 层数 | 隐藏维度 | 头数 | 参数量 |
|------|------|---------|------|--------|
| iGPT-S | 24 | 512 | 8 | 7600万 |
| iGPT-M | 36 | 1024 | 8 | 4.55亿 |
| iGPT-L | 48 | 1536 | 16 | 14亿 |
| iGPT-XL | 60 | 3072 | 24 | 68亿 |

---

## 3. 数学公式与推导

### 3.1 自回归建模

将图像 $I$ 表示为像素序列 $x_1, x_2, ..., x_N$（每个像素是离散化的颜色ID）：

自回归分解：

$$P(I) = \prod_{t=1}^{N} P(x_t | x_{<t})$$

### 3.2 自回归损失

$$\mathcal{L}_{AR} = -\sum_{t=1}^{N} \log P(x_t | x_{<t}; \theta)$$

### 3.3 自编码（Mask）损失

随机选择像素集合 $M$（15%）进行Mask：

$$\mathcal{L}_{AE} = -\sum_{t \in M} \log P(x_t | x_{\backslash M}; \theta)$$

### 3.4 像素预测的Softmax

第 $t$ 个像素的预测概率：

$$P(x_t = k | x_{<t}) = \frac{\exp(h_t^T W_k)}{\sum_{j=1}^{K} \exp(h_t^T W_j)}$$

其中 $h_t$ 是Transformer输出的第 $t$ 个位置的隐藏状态，$W$ 是预测矩阵，$K=512$ 是颜色词汇量。

### 3.5 序列长度

对于32×32的图像：

$$N = H \times W \times 3 = 32 \times 32 \times 3 = 3072$$

对于64×64的图像：

$$N = 64 \times 64 \times 3 = 12288$$

对于224×224的图像：

$$N = 224 \times 224 \times 3 = 150528$$

这就是为什么iGPT只能处理低分辨率图像——序列太长导致计算复杂度过高。

---

## 4. 训练过程讲解

### 阶段一：数据预处理

1. 图像下采样到低分辨率（如32×32或64×64）
2. 将每个像素的RGB值通过k-means量化为512个颜色ID
3. 将图像展开为像素序列

### 阶段二：序列构建

- 每个像素的R、G、B通道独立预测（或作为一个3维预测）
- 使用[CLS] token作为序列起始
- 位置编码基于像素的空间位置

### 阶段三：自回归训练

1. 输入像素序列（从开头到第t-1个）
2. 预测第t个像素的颜色ID
3. 计算交叉熵损失
4. 反向传播更新参数

### 阶段四：特征提取与微调

预训练完成后，iGPT的特征可以用于下游任务：
- 线性探测（Linear Probe）
- 微调（Fine-tuning）

### 训练细节

- 数据集：ImageNet
- 优化器：Adam
- Batch size：2048
- 学习率：3e-4（warmup + cosine decay）
- 训练轮数：100+ epoch

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 图像分类 | 线性探测或微调 | 在ImageNet上分类 |
| 特征提取 | 通用视觉特征 | 提取特征用于下游任务 |
| 图像生成 | 自回归生成新图像 | 生成32×32图像 |
| 图像补全 | 根据已知区域补全 | 补全图像缺失部分 |
| 密度估计 | 估计图像概率 | 计算图像的似然 |

---

## 6. 优缺点分析

### 优点

1. **统一框架**：首次证明NLP的GPT范式可以用于CV
2. **无监督学习**：不需要任何标注数据
3. **生成式理解**：通过生成学习理解，学到的表示具有生成能力
4. **两种模式**：支持自回归和自编码两种预训练
5. **可解释性**：像素预测过程可观察

### 缺点

1. **分辨率限制**：只能处理32×32或64×64的低分辨率图像
2. **计算效率低**：自回归逐像素预测复杂度高
3. **颜色离散化损失**：k-means离散化导致信息损失
4. **性能受限**：在ImageNet上的线性探测结果不如对比学习方法（如SimCLR）
5. **序列长度**：即使是64×64也有12288个像素，限制了扩展性

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class PixelColorQuantizer:
    """
    像素颜色量化器
    使用k-means将RGB颜色离散化为K个类别
    """
    def __init__(self, n_colors=512):
        self.n_colors = n_colors
        self.kmeans = None
        
    def fit(self, images):
        """
        在图像数据集上训练k-means
        images: (N, H, W, 3) numpy array
        """
        pixels = images.reshape(-1, 3)
        self.kmeans = KMeans(n_clusters=self.n_colors, random_state=0)
        self.kmeans.fit(pixels)
        
    def quantize(self, images):
        """
        量化图像
        images: (B, H, W, 3) 像素值0-255
        Returns: (B, H, W) 颜色ID
        """
        B, H, W, C = images.shape
        pixels = images.reshape(-1, 3)
        labels = self.kmeans.predict(pixels)
        return labels.reshape(B, H, W)
    
    def decode(self, labels):
        """
        从颜色ID解码为RGB
        labels: (B, H, W) 颜色ID
        Returns: (B, H, W, 3) RGB值
        """
        return self.kmeans.cluster_centers_[labels]

class iGPTClassifier(nn.Module):
    """
    iGPT分类器
    使用预训练iGPT做图像分类
    """
    def __init__(self, vocab_size=512, d_model=512, n_layers=24, 
                 n_heads=8, max_seq_len=3072, num_classes=1000):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        
        # Transformer解码器（GPT风格）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True),
            num_layers=n_layers
        )
        
        self.ln = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 像素预测头
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, pixel_ids, task='cls'):
        """
        Args:
            pixel_ids: (B, seq_len) 像素颜色ID序列
            task: 'cls' 分类, 'gen' 生成
        """
        B, L = pixel_ids.shape
        
        x = self.token_embedding(pixel_ids) + self.pos_embedding[:, :L, :]
        x = self.transformer(x)
        x = self.ln(x)
        
        if task == 'cls':
            # 使用[CLS] token（序列第一个位置的输出）
            cls_feat = x[:, 0]
            return self.classifier(cls_feat)
        elif task == 'gen':
            # 预测每个位置的下一像素（shifted）
            logits = self.lm_head(x)
            return logits
        
        return x

class iGPTImageGenerator:
    """
    iGPT图像生成器
    自回归生成新图像
    """
    def __init__(self, model, quantizer):
        self.model = model
        self.quantizer = quantizer
        self.vocab_size = quantizer.n_colors
        
    @torch.no_grad()
    def generate(self, seed_pixels=None, h=32, w=32, temperature=1.0):
        """
        自回归生成图像
        """
        if seed_pixels is None:
            # 从[CLS] token开始
            pixels = torch.zeros(1, 1, dtype=torch.long)
        else:
            pixels = seed_pixels.clone()
        
        model.eval()
        with torch.no_grad():
            for _ in range(h * w - pixels.shape[1] + 1):
                logits = self.model(pixels, task='gen')
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_pixel = torch.multinomial(probs, 1)
                pixels = torch.cat([pixels, next_pixel], dim=1)
        
        # 解码为RGB
        pixel_labels = pixels[:, 1:].reshape(1, h, w).cpu().numpy()
        rgb = self.quantizer.decode(pixel_labels)
        
        return rgb

# 使用示例
class iGPTSimulator(nn.Module):
    """
    iGPT模拟器
    用于概念演示
    """
    def __init__(self, vocab_size=512, d_model=256, n_layers=6, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True),
            n_layers
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x)
    
    def generate(self, start_token, max_length, temperature=1.0):
        """自回归生成像素序列"""
        self.eval()
        with torch.no_grad():
            generated = [start_token]
            for _ in range(max_length):
                inputs = torch.tensor([generated])
                logits = self.forward(inputs)
                next_logits = logits[0, -1] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
        return generated

if __name__ == "__main__":
    print("=" * 50)
    print("iGPT (image GPT) 演示")
    print("=" * 50)
    
    # 1. 颜色量化演示
    print("\n1. 颜色量化演示")
    quantizer = PixelColorQuantizer(n_colors=512)
    sample_images = np.random.randint(0, 256, (100, 32, 32, 3)).astype(np.float32)
    quantizer.fit(sample_images)
    print(f"颜色中心数量: {quantizer.kmeans.cluster_centers_.shape[0]}")
    
    # 2. 自回归生成演示
    print("\n2. 自回归生成演示")
    model = iGPTSimulator()
    start_token = 256  # 随机起始token
    generated = model.generate(start_token, max_length=50, temperature=0.8)
    print(f"生成像素序列长度: {len(generated)}")
    print(f"前10个生成的像素ID: {generated[:10]}")
    
    # 3. 前向传播
    print("\n3. 前向传播测试")
    pixel_ids = torch.randint(0, 512, (2, 100))
    logits = model(pixel_ids)
    print(f"输入形状: {pixel_ids.shape}")
    print(f"输出Logits形状: {logits.shape}")  # (2, 100, 512)
    
    print("\niGPT演示完成!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandcraftPixelSequence(nn.Module):
    """
    手工像素序列Transformer
    核心：将图像作为像素序列，自回归建模
    """
    def __init__(self, vocab_size=512, d_model=512, n_heads=8, 
                 n_layers=12, max_seq_len=4096):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 使用TransformerEncoder（但应用因果掩码模拟解码器）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )
        
    def forward(self, x, causal=True):
        """
        前向传播
        x: (B, seq_len) 像素颜色ID序列
        """
        B, L = x.shape
        
        # 嵌入
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:, :L, :]
        
        # 因果掩码（下三角）
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        else:
            mask = None
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        x = self.ln(x)
        logits = self.head(x)
        
        return logits

class HandcraftiGPT:
    """
    手工iGPT训练流程
    包括颜色量化和自回归训练
    """
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.model = None
        
    def pixels_to_sequence(self, image_pixels):
        """
        将图像像素转换为离散序列
        image_pixels: (B, H, W, 3) 值0-255
        简单量化：将RGB映射到512个bin
        """
        # 简单量化：将256×256×256的空间简化
        B, H, W, C = image_pixels.shape
        # 使用3bit R + 3bit G + 3bit B = 9bit = 512
        r = (image_pixels[:, :, :, 0] // 32).long()  # 8 bins
        g = (image_pixels[:, :, :, 1] // 32).long()  # 8 bins  
        b = (image_pixels[:, :, :, 2] // 32).long()  # 8 bins
        # 8*8*8 = 512
        pixel_ids = (r * 64 + g * 8 + b).view(B, -1)  # (B, H*W)
        return pixel_ids
    
    def sequence_to_pixels(self, pixel_ids, h, w):
        """
        将离散序列解码回像素
        """
        B = pixel_ids.shape[0]
        pixel_ids = pixel_ids.view(B, h, w)
        r = (pixel_ids // 64).float() * 32
        g = ((pixel_ids % 64) // 8).float() * 32
        b = (pixel_ids % 8).float() * 32
        return torch.stack([r, g, b], dim=-1)

# 测试手工实现
if __name__ == "__main__":
    # 1. 手工像素序列模型
    model = HandcraftPixelSequence(vocab_size=512, d_model=256, n_heads=4, n_layers=6)
    
    # 模拟像素序列 (4x4=16像素)
    pixel_ids = torch.randint(0, 512, (2, 16))
    logits = model(pixel_ids, causal=True)
    
    print(f"手工iGPT Logits形状: {logits.shape}")  # (2, 16, 512)
    
    # 2. 像素量化演示
    iGPT = HandcraftiGPT()
    images = torch.randint(0, 256, (2, 8, 8, 3))
    seq = iGPT.pixels_to_sequence(images)
    recon = iGPT.sequence_to_pixels(seq, 8, 8)
    
    print(f"原始图像形状: {images.shape}")
    print(f"像素序列形状: {seq.shape}")  # (2, 64)
    print(f"重建图像形状: {recon.shape}")  # (2, 8, 8, 3)
    
    # 3. 计算自回归损失
    input_ids = pixel_ids[:, :-1]
    target_ids = pixel_ids[:, 1:]
    logits = model(input_ids)  # (2, 15, 512)
    
    loss = F.cross_entropy(
        logits.reshape(-1, 512),
        target_ids.reshape(-1)
    )
    
    print(f"自回归损失: {loss.item():.4f}")
    print("\n手工iGPT测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 生成结果

iGPT可以生成32×32的低分辨率图像：
- 训练后可以生成逼真的物体轮廓和颜色
- 但细节有限（受分辨率限制）
- 生成质量随模型规模增大而提升

### 9.2 特征可视化

iGPT在不同层学习到的特征：
- 低层：边缘、纹理、颜色块
- 中层：形状、图案
- 高层：物体部件、语义概念

### 9.3 AR vs AE模式

- 自回归模式：生成的图像更连贯，但生成速度慢（逐像素）
- 自编码模式：表示能力更强，可以双向理解

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 指标 | iGPT-L | iGPT-XL | ResNet-50 (有监督) |
|------|------|--------|---------|-------------------|
| ImageNet Top-1 (Linear) | Accuracy | 60.3% | 65.2% | 76.1% |
| ImageNet Top-1 (Fine-tune) | Accuracy | 69.2% | 72.6% | 76.1% |
| CIFAR-10 (Fine-tune) | Accuracy | 96.3% | 97.1% | 95.3% |
| CIFAR-100 (Fine-tune) | Accuracy | 82.1% | 83.2% | 78.6% |

### 10.2 关键实验结论

1. **AR > AE**（线性探测）：自回归预训练比自编码在特征质量上更好
2. **模型越大越好**：在所有任务上，XL > L > M > S
3. **中间层特征最优**：大约第20层的特征最适合下游任务
4. **像素级理解**：iGPT虽然只能处理低分辨率，但学到的表示可以迁移到高分辨率

---

## 11. 常见问题与易错点

### Q1: iGPT为什么要做颜色离散化？

直接预测256个值（0-255）的RGB像素是回归问题，很难优化。离散化为512类的分类问题后，变成了分类任务，可以用交叉熵损失优化，训练更稳定。

### Q2: iGPT为什么只能处理低分辨率图像？

因为iGPT将图像展开为像素序列，32×32×3=3072个token已经很长了。如果处理224×224的图像，序列长度是150528个token，自注意力的复杂度是 O(L²) = 226亿，无法计算。

### Q3: iGPT如何处理RGB三个通道？

iGPT将每个像素的R、G、B值视为独立的预测token，或者将它们合并为一个三维的预测。两种方式各有优劣。

### Q4: 为什么自回归模式比自编码模式效果好？

自回归模式强制模型学习完整的像素生成过程，从第一个像素到最后一个像素，这种"生成式理解"迫使模型学习到更丰富的视觉概念。自编码模式只预测部分像素，学习信号较弱。

### Q5: iGPT对比ViT哪个更好？

iGPT和ViT各有优劣：
- iGPT：自回归生成式预训练，可生成图像
- ViT：patch级别分类式预训练，可处理高分辨率
- 后续工作（如MAE、BEiT）结合了两者的优点

---

## 12. 学习总结

### 核心知识点

1. **iGPT = GPT架构 + 像素序列建模**
2. **颜色离散化**：将RGB通过k-means量化为512个颜色ID
3. **自回归预训练**：逐像素预测下一个像素
4. **两种模式**：AR（自回归）和AE（自编码）
5. **分辨率限制**：只能处理32×32/64×64低分辨率

### 架构速记

iGPT = GPT解码器 + 像素token嵌入 + 像素自回归预测

### 关键历史地位

iGPT首次将NLP的"下一个词预测"成功移植到CV的"下一个像素预测"，虽然受限于低分辨率，但开创了"用生成式预训练做视觉理解"的新方向。

---

## 13. 练习题与思考题（含答案）

### 习题1：序列长度

**问题**：32×32的彩色图像，展开为像素序列后有多少个token？

**答案**：32×32×3 = 3072个token（如果是每个通道独立预测）或 32×32=1024个token（如果RGB合并预测）。

### 习题2：颜色词汇量

**问题**：为什么选择512个颜色中心？而不是256或1024？

**答案**：512是2的9次方（2⁹），可以用3bit R + 3bit G + 3bit B表示。太多（如1024）会使分类任务过于困难，太少（如256）会丢失颜色细节。

### 习题3：自回归 vs 自编码

**问题**：iGPT的自回归模式和自编码模式在训练效率上有什么区别？

**答案**：自回归模式每步只能预测1个token（需要L步完成整个序列的预测），训练效率低但学习信号完整。自编码模式可以同时预测被Mask的多个token，训练效率高但学习信号不如自回归完整。

### 习题4：与GPT的异同

**问题**：iGPT和NLP中的GPT有什么异同？

**答案**：相同点：都使用Transformer解码器架构、自回归训练目标。不同点：GPT预测文本token（词汇量50000+），iGPT预测像素颜色ID（词汇量512）；GPT处理文本序列，iGPT处理像素序列。

### 习题5：思考题

**问题**：如果iGPT用patch（如4×4像素块）替代单个像素作为预测单元，会有什么影响？

**答案**：优点：(1) 序列长度减少到1/16，可以处理更高分辨率；(2) 每个预测单元包含更多信息。缺点：(1) 预测词汇量变大（512^16个可能的颜色组合）；(2) 无法精确控制像素级别的细节。这实际上是ViT和MAE的思路。

---

## 14. 学习路径建议

### 前置知识
- GPT / Transformer解码器
- 自回归模型
- k-means聚类
- 图像表示

### 平行模型
- **PixelRNN/PixelCNN**：像素级自回归生成的先驱
- **ViT**：Vision Transformer（patch级，非像素级）
- **GPT-2**：NLP的自回归模型对应

### 进阶方向
- **MAE**：Masked Autoencoder（patch级的自编码）
- **BEiT**：视觉token预测
- **DALL-E**：文本到图像的生成（使用离散VAE + 自回归Transformer）
- **VQGAN**：将图像量化为视觉词汇

### 学习顺序建议

```
① GPT → ② iGPT（像素级自回归） → ③ ViT（patch级分类） → ④ MAE/BEiT（patch级生成）
```
