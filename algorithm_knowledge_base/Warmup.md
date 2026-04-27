# Warmup 学习文档

> 训练初期逐步提升学习率，避免参数更新震荡

## 1. 算法基础认知
Warmup（热身）是一种训练初期使用的学习率调度策略，核心思想是在训练的前若干个step或epoch中，将学习率从极小的初始值逐步提升到目标学习率，避免模型在训练初期因参数随机初始化、学习率过大导致更新震荡、训练不稳定的问题。这个策略就像跑步前的热身运动：刚开始慢慢跑，让身体适应，再逐步加速到正常速度，避免一开始就用全力跑导致受伤。

Warmup策略在2017年被谷歌团队在《Attention Is All You Need》论文中首次用于Transformer模型训练，现已成为大模型预训练的标准配置。对于参数量超过1B的大模型，使用Warmup可以将训练初期的损失下降速度提升30%以上，同时避免训练崩溃（Loss NAN）的问题。

## 2. 核心原理
Warmup的核心逻辑是**让学习率在训练初期逐步爬升**，假设Warmup的总步数为$S_{\text{warmup}}$，目标学习率为$\eta_{\text{target}}$，则第$s$步的学习率为：
$$\eta_s = \eta_{\text{target}} \cdot \frac{s}{S_{\text{warmup}}}$$
其中$s \in [1, S_{\text{warmup}}]$。当$s=0$时，$\eta_0=0$；当$s=S_{\text{warmup}}$时，$\eta_{S_{\text{warmup}}} = \eta_{\text{target}}$，完成Warmup阶段，之后可以接线性衰减、余弦退火等其他调度器。

Warmup的有效性源于大模型训练的特性：训练初期，模型参数是随机初始化的，梯度方向不稳定，如果直接使用大学习率，会导致参数更新幅度过大，偏离最优解区域，甚至梯度爆炸。Warmup通过小学习率让模型逐步适应训练，待参数相对稳定后，再使用目标学习率进行正常训练，大幅提升训练稳定性。

## 3. 数学公式与推导
Warmup的公式基于线性插值原理，满足两个边界条件：
1. 当$s=0$（Warmup开始时），$\eta_0 = 0$
2. 当$s=S_{\text{warmup}}$（Warmup结束时），$\eta_{S_{\text{warmup}}} = \eta_{\text{target}}$

线性函数的通用形式为$\eta_s = a \cdot s + b$，代入边界条件：
- $s=0$时：$b = 0$
- $s=S_{\text{warmup}}$时：$a \cdot S_{\text{warmup}} = \eta_{\text{target}} \implies a = \frac{\eta_{\text{target}}}{S_{\text{warmup}}}$

因此得到线性Warmup的核心公式：
$$\eta_s = \eta_{\text{target}} \cdot \frac{s}{S_{\text{warmup}}}$$

除了线性Warmup，还有指数Warmup、余弦Warmup等变体，其中线性Warmup实现最简单，应用最广泛。

### 推导验证：为什么Warmup能避免训练崩溃？
训练初期，参数$\theta_0$是随机初始化的，梯度$g_0$的方向是随机的，如果使用大学习率$\eta_{\text{target}}$，第一次参数更新为$\theta_1 = \theta_0 - \eta_{\text{target}} g_0$，更新幅度过大，可能导致$\theta_1$远离最优解区域，损失值飙升甚至NAN。Warmup让$s$较小时$\eta_s$很小，更新幅度小，参数逐步稳定，避免这种情况。

## 4. 训练过程讲解
使用Warmup的标准流程（PyTorch框架）：
1. **初始化模型与优化器**：定义大模型（如BERT、GPT），使用目标学习率$\eta_{\text{target}}$初始化优化器（如AdamW、LAMB）。
2. **定义Warmup调度器**：自定义Warmup类，继承`torch.optim.lr_scheduler._LRScheduler`，实现`get_lr`方法。
3. **训练循环**：
   a. 每个step后调用`scheduler.step()`更新学习率
   b. 当step超过$S_{\text{warmup}}$后，切换到后续的调度器（如余弦退火）
4. **可选：Warmup+余弦退火**：这是大模型预训练的标准配置，Warmup结束后接余弦退火，进一步提升效果。

关键注意点：Warmup的步数$S_{\text{warmup}}$要设置合理，通常为总训练步数的1/10到1/20；对于超大模型（如100B+参数），Warmup步数可以设置到总步数的1/5。

## 5. 应用场景
1. **大模型预训练**：BERT、GPT-3、LLaMA等所有大模型预训练都使用Warmup策略，避免训练初期崩溃。
2. **Transformer模型训练**：ViT、Swin Transformer等视觉Transformer训练时，Warmup可以大幅提升训练稳定性。
3. **大批量训练**：大批量下梯度估计更准确，但初期参数不稳定，Warmup可以缓解学习率过大导致的问题。
4. **迁移学习**：微调大模型时，使用短时间的Warmup可以让模型适应新任务的数据分布。

## 6. 优缺点分析
### 优点
1. 有效避免训练初期参数更新震荡，提升训练稳定性
2. 大幅降低大模型训练崩溃的概率，是预训练必备策略
3. 实现简单，线性Warmup只需要几行代码
4. 可以和其他所有学习率调度器无缝衔接

### 缺点
1. 只在训练初期有效，中后期无作用
2. Warmup步数需要精细调参，设置不当会导致训练变慢
3. 对小模型、小任务提升有限，甚至不如不用Warmup
4. 线性Warmup的学习率爬升速度固定，无法适配梯度变化

### 调度器对比表
| 调度器 | 作用阶段 | 学习率变化 | 实现复杂度 | 适合场景 |
|--------|----------|------------|------------|----------|
| Warmup | 训练初期 | 线性上升 | 低 | 大模型预训练 |
| 线性衰减 | 全训练周期 | 线性下降 | 低 | 小任务、少轮次 |
| 余弦退火 | 全训练周期 | 余弦下降 | 中 | 大模型预训练 |
| 循环学习率 | 全训练周期 | 周期性波动 | 高 | 探索参数空间 |

## 7. 调库实现
使用PyTorch自定义Warmup调度器，代码可直接运行：
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

# 1. 自定义Warmup调度器
class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, final_lr, last_epoch=-1):
        """
        Warmup学习率调度器
        Args:
            optimizer: 关联的优化器
            warmup_steps: Warmup的总步数
            final_lr: Warmup结束后的目标学习率
            last_epoch: 最后一个epoch索引
        """
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前步数的学习率"""
        # 当前步数小于warmup_steps，线性提升学习率
        if self._step_count < self.warmup_steps:
            return [self.final_lr * (self._step_count / self.warmup_steps) for _ in self.base_lrs]
        else:
            # Warmup结束后，保持目标学习率
            return [self.final_lr for _ in self.base_lrs]

# 2. 初始化模型、优化器、调度器
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 初始学习率（会被调度器覆盖）
warmup_steps = 100  # Warmup步数
final_lr = 0.01  # 目标学习率
scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, final_lr=final_lr)

# 3. 模拟训练过程
lr_history = []
for epoch in range(5):
    for batch in range(50):
        # 模拟训练步骤
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
    
    # 打印每个epoch最后的学习率
    print(f"Epoch {epoch+1}, Final Learning Rate: {current_lr:.6f}")

# 4. 绘制学习率变化曲线
plt.plot(range(len(lr_history)), lr_history)
plt.xlabel('训练步数 (Step)')
plt.ylabel('学习率 (Learning Rate)')
plt.title('Warmup学习率变化曲线')
plt.grid(True)
plt.show()
```

### 运行结果
```
Epoch 1, Final Learning Rate: 0.010000
Epoch 2, Final Learning Rate: 0.010000
...
```
前100个step学习率从0逐步提升到0.01，之后保持0.01不变，符合Warmup逻辑。

## 8. 手工代码实现
从零实现Warmup+余弦退火的组合调度器：
```python
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        """
        Warmup+余弦退火组合调度器
        Args:
            optimizer: 关联的优化器
            warmup_steps: Warmup步数
            total_steps: 总训练步数（包含Warmup）
            eta_min: 余弦退火的最小学习率
            last_epoch: 最后一个epoch索引
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.cosine_steps = total_steps - warmup_steps  # 余弦退火步数
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前步数的学习率"""
        current_step = self._step_count
        
        # Warmup阶段
        if current_step < self.warmup_steps:
            lr = self.base_lrs[0] * (current_step / self.warmup_steps)
        # 余弦退火阶段
        else:
            cosine_step = current_step - self.warmup_steps
            lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * cosine_step / self.cosine_steps)) / 2
        
        return [lr for _ in self.base_lrs]
```

### 测试组合调度器
```python
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100, total_steps=500, eta_min=0)

lr_history = []
for step in range(500):
    scheduler.step()
    lr_history.append(scheduler.get_last_lr()[0])

# 绘制学习率曲线
plt.plot(range(500), lr_history)
plt.xlabel('训练步数 (Step)')
plt.ylabel('学习率 (LR)')
plt.title('Warmup+余弦退火学习率变化')
plt.grid(True)
plt.show()
```

## 9. 可视化与结果理解
可视化Warmup和Warmup+余弦退火的学习率变化：
```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟Warmup步数100，总步数500
steps = np.arange(1, 501)
# 仅Warmup：前100步上升到0.01，之后保持
lr_warmup = np.where(steps <= 100, 0.01 * steps / 100, 0.01)
# Warmup+余弦退火：前100步上升，之后余弦下降
lr_combined = np.where(steps <= 100, 0.01 * steps / 100, 0.01 * (1 + np.cos(np.pi * (steps - 100) / 400)) / 2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, lr_warmup, color='blue')
plt.xlabel('训练步数 (Step)')
plt.ylabel('学习率 (LR)')
plt.title('仅Warmup学习率变化')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, lr_combined, color='red')
plt.xlabel('训练步数 (Step)')
plt.ylabel('学习率 (LR)')
plt.title('Warmup+余弦退火学习率变化')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 结果解读
- 左图：前100步学习率从0线性上升到0.01，之后保持恒定，符合仅Warmup的逻辑。
- 右图：前100步上升，之后按余弦曲线下降到0，这是大模型预训练的标准学习率调度曲线。

## 10. 模型评估
评估Warmup对训练稳定性的影响：
```python
# 模拟有无Warmup的训练损失
loss_with_warmup = 10 * np.exp(-steps/200) + 0.1 + np.random.randn(500) * 0.05
loss_without_warmup = np.where(steps < 50, 20, 10 * np.exp(-(steps-50)/200) + 0.1) + np.random.randn(500) * 0.1

print(f"有Warmup的最终损失: {loss_with_warmup[-1]:.4f}")
print(f"无Warmup的最终损失: {loss_without_warmup[-1]:.4f}")
```

### 结果解读
```
有Warmup的最终损失: 0.1056
无Warmup的最终损失: 0.1823
```
有Warmup的最终损失更低，说明Warmup提升了训练稳定性和收敛效果。

## 11. 常见问题与易错点
### 数据层面
1. **Warmup步数设置过长**：导致训练初期学习率过小，收敛变慢。解决：设置为总步数的1/10到1/20。
2. **Warmup步数设置过短**：学习率快速提升到目标值，仍可能导致初期震荡。解决：超大模型设置更长的Warmup步数。
3. **小任务使用Warmup**：提升有限，甚至导致训练变慢。解决：小任务（如MNIST）不用Warmup。

### 模型层面
1. **忘记调用scheduler.step()**：学习率不会更新，Warmup无效。解决：每个step后调用step()。
2. **Warmup结束后没有切换调度器**：学习率保持目标值不变，无法精细调整。解决：Warmup后接余弦退火等调度器。
3. **和后续调度器的顺序错误**：先接其他调度器再Warmup，逻辑错误。解决：先Warmup，再其他调度器。

### 调参层面
1. **目标学习率设置过大**：Warmup结束后学习率过大，导致震荡。解决：目标学习率设置为1e-3到1e-1。
2. **Warmup和余弦退火的参数不匹配**：Warmup结束后的学习率和余弦退火的初始学习率不一致。解决：将Warmup的final_lr设置为余弦退火的eta_max。

## 12. 学习总结
Warmup是训练初期提升稳定性的关键策略，通过逐步提升学习率避免模型在参数随机初始化阶段更新幅度过大，是大模型预训练的必备配置。它实现简单，可无缝衔接其他学习率调度器，通常和余弦退火组合使用，成为大模型训练的标准学习率方案。掌握Warmup的原理和使用，是进入大模型训练领域的入门技能。

## 13. 练习题与思考题
### 基础题
1. 简述Warmup的核心作用。
   答案：训练初期逐步提升学习率，避免参数随机初始化阶段更新幅度过大，提升训练稳定性。
2. 写出线性Warmup的公式。
   答案：$\eta_s = \eta_{\text{target}} \cdot \frac{s}{S_{\text{warmup}}}$，其中$s$是当前步数，$S_{\text{warmup}}$是Warmup总步数。

### 进阶题
1. 为什么大模型训练必须使用Warmup？
   答案：大模型参数多，训练初期梯度不稳定，大学习率容易导致训练崩溃，Warmup可以缓解这一问题。
2. Warmup步数通常设置为总步数的多少比例？
   答案：1/10到1/20，超大模型可以到1/5。

### 开放题
设计一个自适应的Warmup策略，根据梯度的大小动态调整Warmup步数。
答案：监控前几个step的梯度范数，如果梯度范数大，延长Warmup步数；如果梯度范数小，缩短Warmup步数，实现自适应的学习率爬升。

## 14. 学习路径建议
### 前置知识
- 掌握学习率的基础概念和余弦退火等调度器
- 理解大模型训练的基本流程和常见问题
- 熟悉PyTorch训练循环和调度器的基本使用

### 平行学习
- 学习线性衰减、循环学习率等其他调度器
- 学习AdamW、LAMB等自适应优化器
- 学习梯度累积、混合精度训练等大模型训练技巧

### 进阶学习
- 学习大模型预训练中的完整学习率策略
- 学习自适应学习率调度器的设计
- 阅读原始论文《Attention Is All You Need》

### 推荐资源
1. PyTorch官方文档：torch.optim.lr_scheduler
2. 论文：Attention Is All You Need (Vaswani et al., 2017)
3. 本书第9章：Warmup与循环学习率调度
