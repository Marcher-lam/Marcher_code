# AdamW 与学习率调度 学习文档

> 来源线索：本节内容根据原书中涉及优化器和学习率调度的相关章节整理、扩展与教学化改写。

> AdamW修复了Adam的权重衰减缺陷，配合余弦退火实现稳定的大模型训练。

## 1. 算法基础认知

**一句话定义**：AdamW是Adam优化器的改进版，将权重衰减从梯度更新中解耦；学习率调度控制训练过程中学习率的变化策略。

**直觉类比**：
- **AdamW**：像一个聪明的学生，不仅记住正确方向（动量），还自动调整步伐（自适应学习率），并且有规律地"遗忘"旧知识（权重衰减）防止死记硬背。
- **学习率调度**：像开车——刚上路时慢慢加速（warmup），途中保持速度，快到终点时减速（退火），避免冲过头。

**历史背景**：Adam由Kingma和Ba在2015年提出，结合了Momentum和RMSProp的优点。Loshchilov和Hutter在2017年发现Adam的权重衰减实现有缺陷，提出AdamW。Cosine Annealing由Loshchilov和Hutter在2017年提出，被广泛用于Transformer训练。

**算法定位**：深度学习 / 优化算法 / 训练技巧。

**前置知识**：
- 梯度下降基础
- 一阶和二阶矩估计
- 学习率和权重衰减的概念

## 2. 核心原理

### AdamW核心思想

Adam的问题是将L2正则化和梯度更新混在一起（将权重衰减项加入梯度）。AdamW将权重衰减**解耦**出来：

- **Adam + L2**：$g_t = \nabla L + \lambda w_t$（将正则项加入梯度，被自适应学习率缩放）
- **AdamW**：梯度更新 $\to$ 权重衰减单独作用（不受自适应学习率影响）

### 学习率调度核心思想

训练过程中动态调整学习率：

1. **Warmup**：从很小的学习率线性增加到目标值（避免初期梯度不稳定）
2. **Cosine Annealing**：按余弦函数从峰值衰减到最小值
3. **Constant**：保持恒定学习率

### 工作流程（AdamW + Cosine Schedule）

1. 初始化参数 $\theta_0$，一阶矩 $m_0 = 0$，二阶矩 $v_0 = 0$
2. 计算梯度 $g_t$
3. 更新一阶矩：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
4. 更新二阶矩：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
5. 偏差校正：$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$
6. 参数更新：$\theta_t = \theta_{t-1} - \eta_t (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})$

## 3. 数学公式与推导

### AdamW更新规则

$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right) - \eta_t \lambda \theta_t$$

其中：
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Adam vs AdamW的区别

**Adam with L2**：
$$g_t = \nabla f(\theta_t) + \lambda \theta_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot \text{Adam}(g_t)$$

问题：$\lambda \theta_t$ 被自适应学习率缩放，导致正则化效果不均匀。

**AdamW（解耦权重衰减）**：
$$g_t = \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \text{Adam}(g_t) - \eta \lambda \theta_t$$

权重衰减直接作用于参数，不受Adam自适应学习率的影响。

### Cosine Annealing

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

### Warmup + Cosine Decay

$$\eta_t = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t - T_{warmup}}{T - T_{warmup}}\pi)) & t \geq T_{warmup} \end{cases}$$

## 4. 训练过程讲解

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| $\beta_1$ | 一阶矩衰减率 | 0.9 | 0.9 |
| $\beta_2$ | 二阶矩衰减率 | 0.95-0.999 | 0.95 |
| $\epsilon$ | 数值稳定性 | 1e-8 | 1e-8 |
| $\lambda$ | 权重衰减系数 | 0.01-0.1 | 0.1 |
| $\eta_{max}$ | 峰值学习率 | 1e-5 到 1e-3 | 3e-4 |
| $T_{warmup}$ | warmup步数 | 总步数的1-10% | 总步数的2% |

## 5. 应用场景

1. **大语言模型预训练**：GPT、LLaMA、DeepSeek等模型都使用AdamW + Cosine Annealing。

2. **多模态模型训练**：CLIP等模型使用AdamW配合不同的学习率调度。

3. **微调预训练模型**：AdamW是微调的标准优化器，通常使用较小的学习率和较短的warmup。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 训练稳定、收敛快 | 显存占用高（需存储m和v） |
| 权重衰减效果正确 | 超参数较多 |
| 自适应学习率适合稀疏梯度 | 某些情况下泛化不如SGD |

## 7. 调库实现

```python
"""AdamW 和学习率调度的 PyTorch 实现"""
import torch
import torch.nn as nn
import math

class CosineAnnealingWithWarmup:
    """Warmup + Cosine Annealing 学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps, total_steps,
                 eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.eta_max = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # 线性warmup
            return self.eta_max * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * progress))


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建简单模型
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    
    # AdamW优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 学习率调度
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        eta_min=1e-6
    )
    
    # 模拟训练
    lrs = []
    for step in range(200):
        lrs.append(scheduler.get_lr())
        x = torch.randn(4, 128)
        y = torch.randint(0, 10, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print("=== AdamW + Cosine Annealing 测试 ===")
    print(f"初始学习率: {lrs[0]:.6f}")
    print(f"峰值学习率: {max(lrs):.6f}")
    print(f"最终学习率: {lrs[-1]:.6f}")
    print(f"Warmup步数: 100, 总步数: 200")
```

## 8. 手工代码实现

```python
"""从零实现AdamW优化器"""
import torch

class ManualAdamW:
    """手写AdamW优化器"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # 初始化一阶矩和二阶矩
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                
                g = p.grad
                
                # 更新一阶矩
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                # 更新二阶矩
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
                
                # 偏差校正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Adam更新 + 解耦权重衰减
                p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建测试参数
    W = torch.randn(10, 5, requires_grad=True)
    target = torch.randn(5)
    
    # 手写AdamW
    optimizer = ManualAdamW([W], lr=1e-2, weight_decay=0.01)
    
    print("=== 手写AdamW测试 ===")
    for step in range(10):
        optimizer.zero_grad()
        loss = ((W.sum(dim=0) - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        if step % 2 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    
    # 与PyTorch AdamW对比
    W2 = W.detach().clone().requires_grad_(True)
    opt2 = torch.optim.AdamW([W2], lr=1e-2, weight_decay=0.01)
    
    for step in range(10):
        opt2.zero_grad()
        loss = ((W2.sum(dim=0) - target) ** 2).sum()
        loss.backward()
        opt2.step()
    
    diff = (W - W2).abs().max()
    print(f"\n与PyTorch AdamW差异: {diff:.6f}")
```

## 9. 可视化与结果理解

```python
"""学习率调度可视化"""
import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

total_steps = 1000
warmup_steps = 100
eta_max = 3e-4
eta_min = 1e-6

# 图1: 不同学习率调度策略对比
steps = np.arange(total_steps)

# Cosine Annealing with Warmup
cosine_lr = []
for t in steps:
    if t < warmup_steps:
        cosine_lr.append(eta_max * t / warmup_steps)
    else:
        progress = (t - warmup_steps) / (total_steps - warmup_steps)
        cosine_lr.append(eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * progress)))

# Linear Decay with Warmup
linear_lr = []
for t in steps:
    if t < warmup_steps:
        linear_lr.append(eta_max * t / warmup_steps)
    else:
        progress = (t - warmup_steps) / (total_steps - warmup_steps)
        linear_lr.append(eta_max * (1 - progress))

# Constant with Warmup
constant_lr = [eta_max * min(t / warmup_steps, 1.0) for t in steps]

axes[0].plot(steps, cosine_lr, label='Cosine Annealing', linewidth=2)
axes[0].plot(steps, linear_lr, label='Linear Decay', linewidth=2)
axes[0].plot(steps, constant_lr, label='Constant', linewidth=2)
axes[0].axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup结束')
axes[0].set_title('学习率调度策略对比', fontsize=13)
axes[0].set_xlabel('训练步数')
axes[0].set_ylabel('学习率')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 权重衰减对比
weights_no_decay = []
weights_with_decay = []
w1 = 1.0
w2 = 1.0
wd = 0.01
lr = 0.1
for t in range(200):
    w1 -= lr * 0.1  # 无权重衰减
    w2 -= lr * (0.1 + wd * w2)  # AdamW式权重衰减
    weights_no_decay.append(w1)
    weights_with_decay.append(w2)

axes[1].plot(weights_no_decay, label='无权重衰减', linewidth=2)
axes[1].plot(weights_with_decay, label='有权重衰减(AdamW)', linewidth=2)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[1].set_title('权重衰减效果', fontsize=13)
axes[1].set_xlabel('训练步数')
axes[1].set_ylabel('权重值')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3: 不同beta2的影响
beta2_values = [0.99, 0.999, 0.9999]
for b2 in beta2_values:
    m, v = 0.0, 0.0
    b1 = 0.9
    effective_lrs = []
    for t in range(1, 200):
        g = 0.1 * math.sin(t * 0.1) + 0.01  # 模拟梯度
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        effective_lr = m_hat / (math.sqrt(v_hat) + 1e-8)
        effective_lrs.append(effective_lr)
    axes[2].plot(effective_lrs, label=f'β₂={b2}')

axes[2].set_title('不同β₂的有效学习率', fontsize=13)
axes[2].set_xlabel('训练步数')
axes[2].set_ylabel('有效学习率')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adamw_viz.png', dpi=100)
plt.show()

print("图1解读: Cosine Annealing平滑衰减, 线性衰减更激进, 常量不衰减")
print("图2解读: 权重衰减使参数趋向更小的值, 防止过拟合")
print("图3解读: β₂越大, 二阶矩更新越慢, 有效学习率调整越迟缓")
```

## 10. 模型评估

优化器的评估通过训练曲线间接衡量：训练损失下降速度、验证集性能、训练稳定性。

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| Warmup不足 | 训练初期发散 | 大模型初期梯度不稳定 | 增加warmup步数 |
| 权重衰减太大 | 欠拟合 | 正则化过强 | 降低weight_decay到0.01 |
| beta2太大 | 收敛慢 | 二阶矩更新迟缓 | 大模型用0.95 |

## 12. 学习总结

AdamW的关键公式：$\theta_{t+1} = \theta_t - \eta(\hat{m}/(\sqrt{\hat{v}}+\epsilon) + \lambda\theta_t)$

核心改进：将权重衰减从梯度中解耦，使正则化效果不受自适应学习率影响。配合Cosine Annealing + Warmup实现稳定的训练。

## 13. 练习题与思考题

### 基础题1：参数量计算

一个7B参数的模型使用AdamW，需要多少额外显存存储优化器状态？

**参考答案**：
- AdamW需要存储m（一阶矩）和v（二阶矩），各与模型参数相同大小
- 额外显存 = 2 × 7B = 14B参数
- 以float32计：14B × 4 bytes = 56 GB
- 以BF16模型 + float32优化器：7B × 2 + 14B × 4 = 70 GB

### 基础题2：Cosine学习率

总训练步数10000，warmup 500步，峰值学习率1e-4，最小学习率1e-6。计算第5000步的学习率。

**参考答案**：
progress = (5000 - 500) / (10000 - 500) = 4500/9500 ≈ 0.474
lr = 1e-6 + 0.5 × (1e-4 - 1e-6) × (1 + cos(0.474π)) ≈ 1e-6 + 0.5 × 9.9e-5 × (1 + cos(1.488)) ≈ 1e-6 + 0.5 × 9.9e-5 × 0.082 ≈ 5.1e-6

### 进阶题：AdamW vs SGD with Momentum

在什么场景下SGD+Momentum可能优于AdamW？

**参考答案**：
- **图像分类**：CNN模型在ImageNet上SGD通常泛化更好
- **凸优化**：理论保证更好
- **显存受限**：SGD只存储动量（1x），AdamW需要m+v（2x）

### 开放思考题

为什么大语言模型训练几乎都用AdamW而不是SGD？这背后有什么理论和实践原因？

**参考思路**：
1. **稀疏梯度**：语言模型的嵌入层和softmax产生稀疏梯度，Adam的自适应学习率更适合
2. **训练稳定性**：大模型参数多、层深，Adam的二阶矩估计帮助稳定训练
3. **超参数鲁棒性**：Adam对学习率的选择更宽容，SGD需要精心调参
4. **实践验证**：大量实验表明Adam在大模型上训练更稳定、收敛更快

## 14. 学习路径建议

### 前置知识
- 梯度下降
- 动量法（Momentum）
- 学习率的概念

### 平行学习
- 其他优化器（AdaGrad、RMSProp）
- 梯度裁剪（Gradient Clipping）

### 进阶方向
- 分布式训练中的优化器（ZeRO、FSDP）
- 8-bit优化器（bitsandbytes）
- Sophia等新型优化器

### 推荐资源
1. **论文**：Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)
2. **论文**：SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)
3. **博客**：Sebastian Ruder的优化器综述
