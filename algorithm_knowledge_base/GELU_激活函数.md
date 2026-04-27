# GELU 激活函数 学习文档

> 基于高斯分布的平滑激活函数，曾是Transformer模型的首选激活。

> 来源线索：本节内容根据原书附录C C.2节关于GELU激活函数的讲解整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
GELU (Gaussian Error Linear Unit) 使用高斯累积分布函数对输入进行平滑加权的激活函数。

### 直觉类比
ReLU的开关是二元的——低于0=完全关，高于0=完全开。GELU给这个开关加上"概率性"——一个值即使略低于0，也有一定概率被"留下"（取决于它的相对大小）。

### 历史背景
GELU于 2016 年由 Hendrycks 和 Gimpel 提出。如 BERT (2018)、GPT-2 (2019)、早期的GPT-3、Google的Gemma系列都使用GELU。2020年后逐渐被SiLU取代，目前只在部分模型中保留。

### 算法定位
- **类型**：激活函数 / 非线性函数
- **性质**：神经网络的基本组件

## 2. 核心原理

### 数学定义

**精确形式**（计算复杂）：
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准高斯分布的累积分布函数(CDF)。

**tanh近似**（实际中常用，速度更快）：
$$\text{GELU}(x) \approx 0.5x \cdot \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right]$$

### 物理直觉
GELU可以理解为"随机正则化的期望"——在输入x上乘以一个服从伯努利分布的mask，mask=1的概率取决于x相对于其他值的大小（从高斯分布看）。这个概率由Φ(x)给出。

### 与SiLU的对比
```
GELU(x) = x·Φ(x)    # Φ是高斯CDF (复杂)
SiLU(x)  = x·σ(x)    # σ是sigmoid (简单)
```

| 特征 | GELU | SiLU |
|------|------|------|
| 平滑性 | 是 | 是 |
| 计算速度 | 较慢（erf或tanh逼近） | 较快(sigmoid) |
| 最小值 | ≈-0.169 | ≈-0.278 |
| x→+∞ | ≈x | ≈x |
| x→-∞ | →0 | →0 |

## 3. 优缺点分析

| 优点 | 说明 |
|------|------|
| 平滑概率性解释 | 随机正则化直觉让训练更稳定 |
| 大量历史成果 | BERT/GPT-2时代积累了大量使用经验 |
| 表现稳定 | 在各种任务上表现一致可预测 |

| 缺点 | 说明 |
|------|------|
| 计算慢 | erf或tanh比sigmoid慢 |
| 逐渐被替代 | SiLU已成为新模型的事实标准 |
| GPU kernel融合难 | 复杂的公式难以优化成高效GPU核 |

## 4-14. 实现、问题、练习

### 调库实现
```python
import torch.nn.functional as F
y = F.gelu(x)  # 默认使用tanh近似
y = F.gelu(x, approximate='tanh')  # 明确指定
```

### 手工实现
```python
import torch
def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(
        (2/torch.pi)**0.5 * (x + 0.044715 * x**3)
    ))
```

### 练习题
**题1**：为什么新模型大多选择SiLU而非GELU？

**参考答案**：(1)计算效率：SiLU的sigmoid比GELU的erf/tanh在GPU上更快；(2)SwiGLU门控机制中SiLU与门控设计配合良好；(3)大规模实验中建模性能差异不显著——既然差不多，选更快的。

### 学习路径
- **前置**：ReLU、sigmoid
- **进阶**：SiLU/Swish、GeGLU、各种激活函数在深度模型中的行为对比
