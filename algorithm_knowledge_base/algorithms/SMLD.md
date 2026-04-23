# SMLD 学习文档

## 1. 算法基础认知

### 1.1 研究背景

SMLD（Score Matching with Langevin Dynamics）是由Yang Song和Stefano Ermon在2019年提出的生成模型。它结合了分数匹配（Score Matching）和朗之万动力学（Langevin Dynamics），通过学习数据分布的梯度（分数函数）来生成样本。这种方法不需要GAN的对对抗训练，也不像VAE那样有后验近似问题。

### 1.2 核心思想

SMLD的核心思想是直接学习数据分布的对数概率梯度（分数函数）$\nabla_x \log p(x)$，然后使用朗之万动力学从该分数函数中采样。训练目标是最小化分数匹配损失，等价于去噪自编码器的形式。

### 1.3 技术定位

SMLD属于**基于分数的生成模型**范畴，后续的NCSN（Noise Conditional Score Network）和Diffusion Models都建立在此基础上。

---

## 2. 核心原理

### 2.1 分数函数

数据分布的对数概率梯度：

$$s(x) = \nabla_x \log p(x)$$

这个梯度指向概率增加的方向。

### 2.2 分数匹配目标

学习$s_\theta(x) \approx \nabla_x \log p(x)$：

$$L(\theta) = \mathbb{E}_{x \sim p_{data}}[||s_\theta(x) - \nabla_x \log p(x)||^2]$$

等价于去噪目标：

$$L(\theta) = \mathbb{E}_{x, \tilde{x}}[||s_\theta(x) - \frac{x - \tilde{x}}{\sigma^2)||^2]$$

其中$\tilde{x} = x + \sigma \epsilon, \epsilon \sim \mathcal{N}(0, I)$。

### 2.3 朗之万动力学采样

从分数函数采样：

$$x_{t+1} = x_t + \alpha \nabla_x \log p(x_t) + \sqrt{2\alpha} \epsilon_t$$

其中$\epsilon_t \sim \mathcal{N}(0, I)$，$\alpha$是步长。

### 2.4 噪声条件分数网络

添加噪声级别条件：

$$s_\theta(x, \sigma) = \nabla_x \log p(x|sigma)$$

使用多个噪声级别$\sigma_1 > \sigma_2 > ... > \sigma_K$。

---

## 3. 数学公式与推导

### 3.1 分数匹配

给定噪声级别$\sigma$，目标：

$$L = \mathbb{E}_{x, z}[||s_\theta(x + \sigma z) + \frac{x - (x + \sigma z)}{\sigma^2}||^2]$$

简化后：

$$L = \mathbb{E}_{x, z}[||s_\theta(x + \sigma z) + \frac{z}{\sigma}||^2]$$

### 3.2 朗之万采样

迭代采样过程：

$$x_{i+1} = x_i + \alpha s_\theta(x_i, \sigma_{min}) + \sqrt{2\alpha} z_i$$

其中$\sigma_{min}$是最小噪声级别。

### 3.3 采样算法

```
朗之万采样
x ~ N(0, I)
for i = 1..T:
  z ~ N(0, I)
  x = x + α∇_x log p(x) + √(2α)z
return x
```

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 推荐值 |
|------|--------|
| 批大小 | 64 |
| 学习率 | 0.0001 |
| 噪声级别数 | 10 |
| 网络宽度 | 256 |

### 4.2 噪声调度

| 噪声级别 | σ_max 到 σ_min 对数衰减 |
|----------|-------------------|

### 4.3 采样步数

通常需要1000步或更多朗之万迭代。

---

## 5. 应用场景

### 5.1 图像生成

- 高质量图像
- 人脸生成

### 5.2 数据增强

- 样本生成
- 密度估计

### 5.3 异常检测

- 密度低的样本为异常

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 稳定训练 | 无GAN问题 |
| 精确目标 | 明确的训练目标 |
| 灵活采样 | 可调采样参数 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 采样慢 | 需多步迭代 |
| 质量一般 | 不如GAN好 |

---

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreNetwork(nn.Module):
    """分数网络"""
    
    def __init__(self, in_channels=3, hidden=256, num_classes=10):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
        ))
        
        for _ in range(num_classes - 1):
            self.convs.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
            ))
            
        self.convs.append(nn.Conv2d(hidden, in_channels, 3, padding=1))
        
    def forward(self, x, sigma):
        B, C, H, W = x.shape
        
        sigma_embed = torch.log(sigma).view(B, 1, 1, 1)
        
        out = x
        for conv in self.convs:
            out = conv(out)
            
        return out * torch.exp(-sigma_embed)


class SMLD:
    """
    SMLD: Score Matching with Langevin Dynamics
    Reference: https://arxiv.org/abs/1907.05600
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = ScoreNetwork().to(device)
        
        self.sigmas = torch.exp(torch.linspace(-1, -7, 10)).to(device)
        
    def train_step(self, x):
        """单步训练"""
        
        batch_size = x.size(0)
        
        sigma = torch.randint(0, len(self.sigmas), (batch_size,)).to(self.device)
        sigma = self.sigmas[sigma].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise
        
        target = -noise / (sigma ** 2)
        
        score_pred = self.model(x_noisy, sigma)
        
        loss = F.mse_loss(score_pred, target)
        
        return loss
    
    def sample(self, n, shape):
        """朗之万采样"""
        
        self.model.eval()
        
        x = torch.randn(n, *shape, device=self.device)
        
        alpha = 0.0001
        
        for i in range(1000):
            noise = torch.randn_like(x)
            score = self.model(x, self.sigmas[-1].view(1, 1, 1, 1))
            x = x + alpha * score + torch.sqrt(2 * alpha) * noise
            
        self.model.train()
        return x


def main():
    """SMLD示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smld = SMLD(device=device)
    
    x = torch.randn(4, 3, 32, 32).to(device)
    loss = smld.train_step(x)
    print(f"Loss: {loss:.4f}")
    
    generated = smld.sample(4, (3, 32, 32))
    print(f"Generated: {generated.shape}")


if __name__ == "__main__":
    main()
```

---

## 8. 手工代码实现

```python
import torch
import torch.nn as nn


class SimpleScoreNet(nn.Module):
    """简化分数网络"""
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        
    def forward(self, x, sigma=1.0):
        return self.net(x)


class SimpleSMLD:
    """简化SMLD"""
    
    def __init__(self):
        self.model = SimpleScoreNet()
        
    def train_step(self, x):
        noise = torch.randn_like(x)
        x_noisy = x + noise
        
        score_pred = self.model(x_noisy)
        target = -noise
        
        loss = ((score_pred - target) ** 2).mean()
        return loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smld = SimpleSMLD().to(device)
    
    x = torch.randn(2, 3, 8, 8).to(device)
    loss = smld.train_step(x)
    print(f"Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
```

---

## 9. 可视化与结果理解

SMLD生成图像逐渐清晰，采样步数越多质量越高。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| FID | 生成质量 |
| NLL | 对数似然 |

---

## 11. 常见问题与易错点

采样步数不足会导致质量差。

---

## 12. 学习总结

SMLD通过学习分数函数实现了稳定的生成模型训练，为后续的扩散模型奠定了基础。

---

## 13. 练习题与思考题与思考题（含答案）

### 13.1 选择题

**1. SMLD学习什么是？**
A. 生成器
B. 分数函数
C. 判别器

答案：B

**2. 朗之万动力学用于？**
A. 训练
B. 采样
C. 评估

答案：B

---


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：SMLD的[核心概念]是什么？

**答案**：**答案是[B]**。

**解析**：
SMLD的核心机制是[机制描述]。根据算法的数学定义，有：
$$核心公式$$
代入[具体值]后，验证可得正确答案为[B]。

选项分析：
- A：这是对[另一概念]的描述，与SMLD不符
- B：✓ 正确，这是[核心概念]的准确定义
- C：虽然有一定关联，但不是SMLD的主要特性
- D：这是[另一算法]的特征，在SMLD中不适用

#### 练习2：手动计算

**问题**：给定以下数据，请手动计算SMLD的[参数/结果]：
- 输入：$X = [x_1, x_2, ...]$
- 标签：$y = [y_1, y_2, ...]$

**答案**：**计算结果为[具体值]**

**解析**：
**步骤1**：根据SMLD的定义，计算[第一中间量]
$$第一计算 = [公式]$$
代入数据：$第一计算 = [代入数值] = [结果1]$

**步骤2**：继续计算[第二中间量]
$$第二计算 = [公式]$$
代入数据：$第二计算 = [结果2]$

**步骤3**：得到最终结果
$$最终结果 = f(第一计算, 第二计算) = [最终值]$$

**步骤4**：验证
将结果带回原式检验：$[验证过程]$，确认符合约束条件。

#### 思考题：改进分析

**问题**：SMLD在[特定场景]下效果不佳，请分析原因并提出改进方案。

**答案**：

**问题分析**：
1. [局限性1]：具体表现是[现象]，原因是[原因]
2. [局限性2]：具体表现是[现象]，原因是[原因]

**改进方案**：

**方案1：[改进方法名称]**
- **原理**：[解释改进的核心思想]
- **优势**：[改进后带来的好处]
- **实现**：[简要实现说明]

**方案2：[改进方法名称]**
- **原理**：[解释核心思想]
- **��价**：[需要付出的额外计算或复杂度]
- **适用场景**：[何时使用该改进]

## 14. 学习路径建议建议

学习分数匹配理论，理解朗之万采样，实现SMLD。