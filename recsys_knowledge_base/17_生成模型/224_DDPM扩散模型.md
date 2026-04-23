# 扩散模型 学习文档

> 从噪声中生成——DDPM、DM、SMLD统一视角

---

## 1. 算法基础认知

**扩散模型（Diffusion Model）** 是当前最强大的生成模型之一，通过逐步加噪和去噪的过程学习数据分布。

### 核心思想

```
前向过程(加噪):  x₀ → x₁ → x₂ → ... → xₜ (纯噪声)
反向过程(去噪):  xₜ → x_{t-1} → ... → x₁ → x₀ (生成数据)

训练: 学习反向过程中每一步如何去噪
生成: 从纯噪声开始，逐步去噪得到数据
```

### DDPM / DM / SMLD 的关系

| 方法 | 论文 | 核心贡献 |
|------|------|---------|
| **DDPM** | Ho et al., 2020 | 离散时间扩散，去噪匹配 |
| **DM** | Song et al., 2021 | 连续时间扩散，SDE框架 |
| **SMLD** | Song & Ermon, 2019 | 基于得分的生成模型 |

> 三者统一在SDE（随机微分方程）框架下。

---

## 3. 数学公式

### 3.1 前向过程

$$x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3.2 反向过程

学习噪声预测模型 $\epsilon_\theta(x_t, t)$：

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

### 3.3 采样

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

---

## 7. 简化实现

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleDiffusion:
    """简化DDPM"""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t):
        """前向: 在x0上加噪声得到xt"""
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, noise
    
    def sample_step(self, model, xt, t):
        """反向: 一步去噪"""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        
        # 预测噪声
        pred_noise = model(xt, t)
        
        # 去噪
        x_prev = (1/torch.sqrt(alpha_t)) * (xt - beta_t/torch.sqrt(1-alpha_bar_t) * pred_noise)
        
        if t > 0:
            x_prev += torch.sqrt(beta_t) * torch.randn_like(xt)
        
        return x_prev

if __name__ == "__main__":
    diff = SimpleDiffusion(T=100)
    x0 = torch.randn(4, 32)
    t = torch.tensor([50])
    xt, noise = diff.add_noise(x0, t)
    print(f"x0: {x0.shape} → xt(t=50): {xt.shape}")
    print("扩散模型核心: 学习从xt恢复x0的去噪过程")
```

---

## 12. 学习总结

1. 扩散模型 = 前向加噪 + 反向去噪
2. DDPM训练目标：预测每步添加的噪声
3. 当前最强生成模型（DALL-E, Stable Diffusion）
4. 在推荐中：生成式推荐、数据增强
