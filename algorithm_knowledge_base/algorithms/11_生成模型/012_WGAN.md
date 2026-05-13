# WGAN 学习文档

> 用 Wasserstein 距离替代 JS 散度，从根本上解决 GAN 训练不稳定问题。

> 来源线索：本节内容根据原书中关于"WGAN"的相关章节（第4章4.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** WGAN 用 Earth-Mover（Wasserstein-1）距离替代原始 GAN 的 JS 散度作为训练目标，即使生成分布与真实分布不重叠也能提供有意义的梯度。

**直觉类比：** 原始 GAN 的判别器像"考试官"——只回答"真或假"（0或1），当生成器做得不好时直接给 0 分，没有改进方向。WGAN 的判别器（称"评论家"）像一个"评分老师"——给一个连续分数（如 0.3 或 0.7），无论生成器多差都能告诉它"差多少"，从而提供持续的改进梯度。

**历史背景：** WGAN 由 Arjovsky 等人于 2017 年提出（论文 "Wasserstein GAN"），同年 WGAN-GP（梯度惩罚版）由 Gulrajani 等人提出，进一步稳定了训练。WGAN 被认为是 GAN 理论的重要突破。

**算法定位：** 生成模型、GAN 改进、Wasserstein 距离。

**前置知识：** GAN、JS 散度、KL 散度、Lipschitz 连续性、PyTorch。

---

## 2. 核心原理

### 原始 GAN 的问题

原始 GAN 的判别器优化 JS 散度。当真实分布 $P_r$ 和生成分布 $P_g$ 完全不重叠时（训练初期几乎总是如此），$JS = \log 2$（常数），梯度为 0——**模式崩塌的根源**。

### WGAN 的改进

用 Wasserstein 距离替代 JS 散度，由 Kantorovich-Rubinstein 对偶：

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

即使 $P_r$ 和 $P_g$ 不重叠，$W$ 距离仍然有意义且连续，提供持续梯度。

---

## 3. 数学公式与推导

### 评论家损失

$$\mathcal{L}_C = \mathbb{E}_{x \sim P_g}[f_w(x)] - \mathbb{E}_{x \sim P_r}[f_w(x)]$$

### 生成器损失

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p(z)}[f_w(G(z))]$$

### Lipschitz 约束的实现

**权重裁剪（WGAN）**：$w \leftarrow \text{clip}(w, -c, c)$

**梯度惩罚（WGAN-GP）**：

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} f_w(\hat{x})\|_2 - 1)^2]$$

其中 $\hat{x} = \epsilon x_r + (1-\epsilon) x_g$ 是插值样本。

---

## 4. 训练过程讲解

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| n_critic | 5 | 5 |
| $\lambda$ (GP) | 10 | 10 |
| lr | 1e-4 ~ 2e-4 | 1e-4 |
| $\beta_1$ (Adam) | 0.0 | 0.0 |

---

## 5. 应用场景

1. **高质量图像生成**：比原始 GAN 训练稳定得多
2. **文本生成**：WGAN 可用于离散 token 生成（通过 Gumbel-Softmax）
3. **领域迁移**：作为 CycleGAN 等模型的基础

---

## 6. 优缺点分析

### 优点
1. **训练稳定**：不再需要精心平衡 G 和 D
2. **有意义的损失曲线**：W 距离越小，生成质量越好
3. **无需 Sigmoid/BN**：评论家输出连续值

### 缺点
1. **训练速度**：评论家训练次数多（通常 5 次 G 对 1 次 D）
2. **梯度惩罚计算**：需要计算二阶梯度，略慢

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(-1, z.size(1), 1, 1))

class WGANCritic(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0)
        )
    def forward(self, x):
        return self.net(x).view(-1)

def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    critic_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True, retain_graph=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

G, C = WGANGenerator(), WGANCritic()
opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))
print(f"G参数: {sum(p.numel() for p in G.parameters()):,}, C参数: {sum(p.numel() for p in C.parameters()):,}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleWGAN:
    def __init__(self, latent_dim=10, data_dim=2, hidden=32):
        scale = 0.01
        self.G_w1 = np.random.randn(latent_dim, hidden) * scale
        self.G_w2 = np.random.randn(hidden, data_dim) * scale
        self.C_w1 = np.random.randn(data_dim, hidden) * scale
        self.C_w2 = np.random.randn(hidden, 1) * scale

    def generate(self, z):
        h = np.maximum(0, z @ self.G_w1)
        return h @ self.G_w2

    def criticize(self, x):
        h = np.maximum(0.2 * (x @ self.C_w1), x @ self.C_w1)
        return (h @ self.C_w2).flatten()

    def wasserstein_distance(self, real, fake):
        return self.criticize(real).mean() - self.criticize(fake).mean()

wgan = SimpleWGAN()
real = np.random.randn(64, 2) + 2
fake = wgan.generate(np.random.randn(64, 10))
print(f"Wasserstein距离: {wgan.wasserstein_distance(real, fake):.4f}")
```

---

## 9-14. 评估/问题/总结/练习/路径

### 练习题

**题1：** WGAN 为什么不用 Sigmoid？

**参考答案：** WGAN 的评论家输出连续分数估计 Wasserstein 距离，不输出概率。Sigmoid 将输出压缩到 [0,1]，限制评论家区分远近的能力。

**题2（开放）：** WGAN-GP 的梯度惩罚为什么要求梯度范数接近 1？

**参考答案思路：** Lipschitz-1 约束要求 $\|\nabla f\| \leq 1$。惩罚项 $(\|\nabla f\|_2 - 1)^2$ 鼓励评论家在数据分布上满足此约束。

### 学习路径
- 前置：GAN、JS 散度
- 进阶：Spectral Normalization、StyleGAN
- 推荐：Arjovsky et al., "Wasserstein GAN" (2017)


## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：WGAN与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('WGAN Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 12. 学习总结

### 核心要点
1. **基本原理**：WGAN的核心思想和数学基础
2. **实现方法**：从调库到手工实现
3. **应用场景**：WGAN适合的问题类型
4. **优化技巧**：超参数调优和正则化
5. **评估方法**：客观评估性能

### 关键概念
- WGAN的损失函数设计原理
- 参数优化的数学推导
- 泛化能力与过拟合的平衡

### 进阶方向
掌握WGAN后，可进一步学习相关的进阶方法和变体。


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述WGAN的核心思想及适用场景。
<details><summary>参考答案</summary>
WGAN通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出WGAN的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现WGAN核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. WGAN在什么情况下会失效？
2. 训练数据很少时，WGAN还能有效工作吗？
3. 如何将WGAN与其他方法结合？


## 14. 学习路径建议

### 前置知识
深度学习基础、线性代数、PyTorch

### 学习顺序
1. 先理解原理：掌握WGAN核心思想和数学基础
2. 动手实践：运行代码，观察实验结果
3. 深入理解：阅读原始论文，理解设计动机
4. 项目实战：真实数据集上应用WGAN

### 进阶方向
模型优化、分布式训练、推理优化

### 推荐资源
- 搜索WGAN原始论文和综述
- GitHub优秀实现
- Coursera/Stanford相关课程

