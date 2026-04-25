# Stable Diffusion 学习文档

> 文本到图像的扩散模型，Latent 空间高效生成。

---

## 1. 算法基础认知

### 1.1 发展背景

Stable Diffusion 由 Stability AI 于 2022 年发布，是一种基于 Latent 扩散的文本到图像生成模型。通过在 Latent 空间进行扩散，大幅降低了计算成本，使得消费级 GPU 也能生成高质量图像。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 文本到图像生成 |
| 架构 | Latent 扩散 + CLIP |
| 参数 | 约 1B |
| 速度 | 512×512, 几秒 |

### 1.3 模型系列

| 模型 | 参数量 | 分辨率 |
|------|--------|--------|
| SD-v1.4 | 860M | 512 |
| SD-v2.1 | 1B | 768 |
| SDXL | 6.6B | 1024 |

---

## 2. 核心原理

### 2.1 扩散模型

前向过程（加噪）：
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

反向过程（去噪）：
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$$

### 2.2 Latent 扩散

与标准扩散不同，SD 在压缩的 Latent 空间操作：

1. **VAE 编码**：图像 → Latent
2. **Latent 扩散**：在 64×64 空间扩散
3. **VAE 解码**：Latent → 图像

### 2.3 条件引导

使用 CLIP Text Encoder 将文本转换为条件：
$$\epsilon_\theta(x_t, t, c)$$

---

## 3. 数学公式与推导

### 3.1 噪声预测

模型预测噪声：
$$\epsilon_\theta(x_t, t, \text{text})$$

损失函数：
$$\mathbb{E}_{x_0, \epsilon, t} ||\epsilon - \epsilon_\theta(x_t, t, c)||^2$$

### 3.2 Classifier-Free Guidance

$$\hat{\epsilon} = \epsilon_\theta(x_t, null) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, null))$$

$w$ 通常取 7.5。

### 3.3 DDPM vs DDIM

DDIM 采样更快：
$$x_{t-1} = \sqrt{\alpha_{t-1}} (\frac{x_t - \sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}}) + \sqrt{1-\alpha_{t-1}}\epsilon$$

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| Batch | 2048 |
| 学习率 | 1e-4 |
| 步数 | 150K-850K |
| 硬件 | A100 × 256 |

### 4.2 数据集

- LAION-5B：50 亿图文对
- 过滤后：约 5B 高质量对

---

## 5. 应用场景

### 5.1 典型应用

- **AI 绘画**：Midjourney 替代
- **设计辅助**：海报、logo
- **图像编辑**：inpainting/outpainting
- **控制生成**：ControlNet

### 5.2 代码示例

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

prompt = "A cute cat sitting on a sofa"
image = pipe(prompt).images[0]
```

---

## 6. 调库实现

### 6.1 Diffusers 实现

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class StableDiffusionModel:
    """Stable Diffusion 文本到图像"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        
        # 优化采样器
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe.enable_model_cpu_offload()
        
    def generate(self, prompt, num_inference_steps=25, guidance_scale=7.5):
        """图像生成
        
        参数:
            prompt: 文本提示
            num_inference_steps: 采样步数
            guidance_scale: 引导强度
        """
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image
    
    def img2img(self, prompt, image, strength=0.8):
        """图像到图像"""
        return self.pipe.img2img(
            prompt, image, strength=strength
        ).images[0]
    
    def inpaint(self, prompt, image, mask):
        """局部重绘"""
        return self.pipe.inpaint(
            prompt, image, mask
        ).images[0]


def demo():
    print("=== Stable Diffusion 演示 ===\n")
    
    model = StableDiffusionModel()
    print(f"模型加载成功")
    print(f"生成分辨率: 512×512")
    print(f"典型采样步数: 25-50")


if __name__ == "__main__":
    demo()
```

### 6.2 本地部署

```python
# 本地模型加载
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "models/stable-diffusion",
    local_files_only=True
)
```

---

## 7. 手工代码实现

### 7.1 简化扩散

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleDiffusion:
    """简化扩散模型"""
    
    def __init__(self, latent_dim=4, hidden_dim=320):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # UNet 主干
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.Residual(nn.GroupNorm(32, hidden_dim)),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1)
        )
        
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 噪声预测
        return self.net(x + t_emb.unsqueeze(-1).unsqueeze(-1))
    
    def sample(self, shape, num_steps=50):
        """采样"""
        x = torch.randn(shape)
        
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t / num_steps], device=x.device)
            noise = self.forward(x, t_tensor)
            x = x - noise * (1 / num_steps)
        
        return x


def demo():
    print("=== Stable Diffusion 手工实现演示 ===\n")
    
    model = SimpleDiffusion()
    
    # 采样
    x = model.sample((1, 4, 64, 64))
    
    print(f"生成分辨率: {x.shape}")


if __name__ == "__main__":
    demo()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **高效**：Latent 空间操作
2. **开源**：模型权重开放
3. **快速**：消费级 GPU 可运行
4. **可控**：支持多种条件

### 8.2 缺点

1. **细节**：略低于 DALL-E 2
2. **手部**：手指生成问题
3. **文字**：文字生成困难

### 8.3 改进方向

- **SDXL**：更大模型
- **ControlNet**：控制生成
- **Lora**：轻量微调

---

## 9. 可视化与结果理解

### 9.1 采样过程

```python
def visualize():
    print("""
    Stable Diffusion 采样过程:
    
    文本 → CLIP → 条件向量
              ↓
    随机噪声 → UNet 去噪 (50步)
              ↓
    Latent → VAE 解码
              ↓
    图像输出
    
    关键: Latent 空间大幅降低计算量
    """)
```

---

## 10. 模型评估

### 10.1 FID 分数

| 模型 | FID-10K |
|------|----------|
| DALL-E 2 | 10.9 |
| Stable Diffusion | 12.4 |
| Midjourney | 9.6 |

---

## 11. 学习总结

**核心要点**：

1. **Latent 扩散**：压缩空间操作
2. **CLIP 条件**：文本引导
3. **DDIM 采样**：快速生成

**Stable Diffusion 核心优势**：
- 开源可商用
- 消费级 GPU 可运行
- 社区活跃

---

## 12. 练习题与思考题

### 12.1 选择题

1. Stable Diffusion在什么空间操作？
   - A) 像素空间
   - B) Latent空间
   - C) 特征空间

2. 条件引导使用什么模型？
   - A) BERT
   - B) CLIP
   - C) GPT

3. VAE的作用是：
   - A) 文本编码
   - B) 图像编解码
   - C) 去噪

### 12.2 简答题

1. 扩散 vs GAN的区别？
2. Latent空间的作用是什么？

### 12.3 编程题

1. 使用diffusers库生成图片
2. 实现 ControlNet 控制
3. 训练 LoRA

---

## 13. 常见问题与易错点

### Q1: 手部生成有问题？

**答案**：使用ControlNet或局部重绘。

### Q2: 文字生成困难？

**答案**：使用OCR后处理或Inpainting。

### Q3: 如何微调？

**答案**：使用LoRA或Dreambooth。

### Q4: 生成很慢？

**答案**：用DDIM采样或Euler采样器。

### Q5: 显存不够？

**答案**：使用float16或tile分块。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
扩散模型基础
    ↓
CLIP原理
    ↓
Stable Diffusion架构
    ↓
DDPM原理
    ↓
ControlNet
    ↓
LoRA微调
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| DDPM | 扩散基础 |
| DALL-E | 竞品 |
| ControlNet | 控制生成 |
| LoRA | 轻量微调 |

### 14.3 扩展阅读

1. Rombach et al. (2022). Stable Diffusion
2. Zhang et al. (2023). ControlNet

---

## 附录

### A. 参数速查

| 参数 | 推荐值 |
|------|--------|
| steps | 50 |
| guidance | 7.5 |
| size | 512/768 |

### B. 参考

1. Rombach et al. (2022). Stable Diffusion. arXiv:2112.10752

---

**文档结束**