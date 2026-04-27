# StyleGAN 学习文档

> 通过样式控制实现高质量、高分辨率人脸生成。

> 来源线索：本节内容根据原书中关于"StyleGAN"的相关章节（第5章）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** StyleGAN 通过样式映射网络将潜向量映射为样式向量，在网络不同分辨率层注入样式控制生成图像的外观特征，实现前所未有的生成质量和可控性。

**直觉类比：** 普通生成器像"一次成型的雕塑"，StyleGAN 像"分层作画"——先画粗略轮廓，再逐层添加细节。每一层可独立控制，因此可以精确调整"粗细节"（脸型）和"细细节"（发色）。

**历史背景：** StyleGAN 由 NVIDIA 的 Karras 等人于 2019 年提出，能生成 1024×1024 逼真人脸。StyleGAN2（2020）消除伪影，StyleGAN3（2021）解决纹理粘附问题。

**算法定位：** 生成模型、高质量图像生成、人脸生成。

**前置知识：** GAN、WGAN、AdaIN、PyTorch。

---

## 2. 核心原理

### 核心创新

1. **映射网络**：$z \to w$，使 $w$ 更解耦
2. **AdaIN**：将样式注入每层
3. **样式分层**：不同层控制不同级别特征
4. **随机噪声**：提供细节随机性

### 样式控制层次

| 分辨率层 | 控制特征 | 示例 |
|---------|---------|------|
| 4×4 ~ 8×8 | 粗粒度 | 姿态、脸型 |
| 16×16 ~ 32×32 | 中粒度 | 面部特征、发型 |
| 64×64 ~ 1024×1024 | 细粒度 | 颜色、纹理 |

---

## 3. 数学公式

### AdaIN

$$\text{AdaIN}(x_i, w) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

其中 $(y_s, y_b) = A(w)$ 是可学习仿射变换。

### 映射网络

$$w = \text{MLP}_8(z), \quad z, w \in \mathbb{R}^{512}$$

---

## 4-5. 训练与应用

### 应用场景
1. **人脸生成**：thispersondoesnotexist.com
2. **人脸编辑**：操纵样式向量修改属性
3. **风格迁移**：不同人脸间迁移风格

---

## 6. 优缺点分析

### 优点
1. **极高质量**：1024×1024 逼真图像
2. **可控性强**：样式分层控制
3. **解耦表示**：$w$ 空间比 $z$ 更解耦

### 缺点
1. **训练成本高**
2. **限于特定领域**

---

## 7-8. 代码实现

```python
import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class AdaIN(nn.Module):
    def __init__(self, style_dim=512, num_features=256):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style_fc = nn.Linear(style_dim, num_features * 2)
    def forward(self, x, w):
        s, b = self.style_fc(w).chunk(2, dim=1)
        return s.unsqueeze(-1).unsqueeze(-1) * self.norm(x) + b.unsqueeze(-1).unsqueeze(-1)

class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=256, base_ch=128):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, n_layers=4)
        self.const = nn.Parameter(torch.randn(1, base_ch, 4, 4))
        self.adain1 = AdaIN(latent_dim, base_ch)
        self.conv1 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.adain2 = AdaIN(latent_dim, base_ch)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.adain3 = AdaIN(latent_dim, base_ch)
        self.to_rgb = nn.Conv2d(base_ch, 3, 1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.adain1(self.conv1(self.const.repeat(z.size(0),1,1,1)), w)
        x = self.adain2(x, w)
        x = self.up(x)
        x = self.adain3(self.conv2(x), w)
        return self.to_rgb(x)

model = StyleGANGenerator(latent_dim=256, base_ch=64)
out = model(torch.randn(4, 256))
print(f"生成: {out.shape}, 参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 9-14. 练习与路径

**题1：** 映射网络为什么将 $z$ 映射到 $w$？

**参考答案：** 原始 $z$ 中属性纠缠（如年龄和性别相关）。映射到 $w$ 使其更解耦——操纵 $w$ 的单维度只改变一个属性。

### 学习路径
- 前置：GAN、WGAN
- 进阶：StyleGAN2、StyleGAN3
- 推荐：Karras et al., "A Style-Based Generator Architecture for GANs" (2019)
