# 深度循环注意力模型DRAM 学习文档

> 深度强化学习驱动的循环注意力——用RNN决策"看哪里"。
>
> 来源线索：本节内容根据原书第2章关于"目标搜索与识别"的相关章节整理。

---

## 1. 算法基础认知

**一句话定义：** 深度循环注意力模型（Deep Recurrent Attention Model, DRAM）是RAM的深度扩展，使用更深的RNN架构（多层LSTM）和增强的glimpse网络，通过强化学习学习更加复杂的注意力策略。

**核心思想：** DRAM通过将RAM中的glimpse网络和核心网络替换为更深的结构（多层卷积网络 + 多层LSTM），同时在训练中引入更多技巧（经验回放、目标网络、分布式训练），实现更稳定的训练和更好的性能。

**为什么需要深度？** 传统RAM在简单任务（如MNIST上的聚类数字分类）上有效，但面对更复杂的场景（自然图像中的多目标识别）时，单层RNN和浅层glimpse网络的特征提取能力不足。DRAM通过增加深度来提升模型容量。

**DRAM vs RAM：**

| 特性 | RAM | DRAM |
|------|-----|------|
| Glimpse网络 | 简单全连接 | 多层卷积 |
| RNN核心 | 单层RNN/GRU | 多层LSTM |
| 位置策略 | REINFORCE | REINFORCE + 经验回放 |
| 训练稳定性 | 一般 | 更好 |
| 可处理任务 | 简单分类 | 复杂场景分类/检测 |

---

## 2. 核心原理

### 2.1 模型架构

DRAM的核心结构包括：

1. **Glimpse网络**：从全图中提取以当前注视点为中心的局部图像块（glimpse），并使用深层CNN提取特征。
2. **位置编码**：将注视点位置 $(l_x, l_y)$ 编码为向量。
3. **核心网络**：多层LSTM，融合glimpse特征和位置信息，维持内部状态。
4. **分类网络**：从LSTM隐状态预测类别标签。
5. **位置网络**：从LSTM隐状态预测下一个注视点位置。

### 2.2 Glimpse传感器

在时刻 $t$，以位置 $l_{t-1}$ 为中心提取不同分辨率的glimpse：

$$
g_t = \text{Retina}(I, l_{t-1})
$$

Retina传感器提取多个尺度的图像块：中心高分辨率、周边低分辨率。

### 2.3 注意力策略

使用强化学习（REINFORCE算法）训练位置策略：

$$
l_t \sim \pi(\cdot | h_t)
$$

其中 $\pi$ 是位置策略网络，$h_t$ 是LSTM的隐状态。

### 2.4 损失函数

总损失 = 分类损失 + 强化学习损失 + 基线损失：

$$
\mathcal{L} = \mathcal{L}_{class} + \lambda \cdot \mathcal{L}_{RL} + \beta \cdot \mathcal{L}_{baseline}
$$

---

## 3. 数学公式与推导

### 3.1 Glimpse特征提取

$$
g_t = f_{glimpse}(I, l_{t-1}) = \text{CNN}(\text{Extract}(I, l_{t-1}, K))
$$

其中 $K$ 是 glimpse 大小。

位置编码：

$$
l_{enc} = \text{MLP}(l_{t-1})
$$

融合：

$$
x_t = \text{Linear}([g_t; l_{enc}])
$$

### 3.2 LSTM核心网络

$$
\begin{aligned}
i_t &= \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i) \\
f_t &= \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f) \\
o_t &= \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

### 3.3 REINFORCE梯度

位置策略的REINFORCE梯度：

$$
\nabla_\theta J = \mathbb{E}_{p_\tau} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(l_t | h_t) \cdot R_t \right]
$$

其中 $R_t$ 是折扣累积奖励：

$$
R_t = \sum_{k=t}^T \gamma^{k-t} r_k
$$

$r_k = 1$ 如果分类正确，否则 $r_k = 0$。

引入基线 $b_t$ 降低方差：

$$
\nabla_\theta J = \mathbb{E}_{p_\tau} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(l_t | h_t) \cdot (R_t - b_t) \right]
$$

$b_t$ 从LSTM状态通过额外网络预测。

---

## 4. 训练过程讲解

DRAM的训练通过强化学习进行：

**每个episode：**
1. 初始化 $l_0$ 为图像中心
2. 对 $t = 1$ 到 $T$：
   a. 在 $l_{t-1}$ 处提取glimpse
   b. 通过LSTM更新状态 $h_t$
   c. 采样新位置 $l_t \sim \pi(\cdot|h_t)$
   d. 记录位置和奖励
3. 最终时刻用 $h_T$ 预测类别
4. 计算损失并更新

**技巧：**
- 经验回放：存储过去episode用于训练
- 目标网络：稳定Q值估计
- 熵正则化：鼓励探索

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 图像分类 | 通过选择性注视降低计算量 |
| 多目标识别 | 依次注意不同目标 |
| 视觉问答 | 回答问题需要关注特定区域 |
| 场景理解 | 按序处理场景的不同部分 |
| 动态场景分析 | 跟踪注意力随时间的变化 |
| 计算资源受限场景 | 每次只处理局部，减少计算 |

---

## 6. 优缺点分析

**优点：**
- ✅ **计算高效**：每次只处理局部图像块
- ✅ **可解释**：注意力位置可视为决策过程
- ✅ **强化学习框架**：可处理延迟奖励
- ✅ **深度网络**：特征提取能力强
- ✅ **多尺度感知**：Retina传感器提供多分辨率信息

**缺点：**
- ❌ **训练不稳定**：REINFORCE方差大
- ❌ **采样不可微分**：需要策略梯度
- ❌ **推理慢**：串行处理，无法并行
- ❌ **需要精心调参**：RL超参数敏感
- ❌ **可能陷入局部最优**：注意力策略可能过早收敛

---

## 7. 调库实现

```python
"""DRAM - PyTorch完整实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class RetinaSensor(nn.Module):
    """Retina传感器：多尺度Glimpse提取"""
    
    def __init__(self, in_channels=1, glimpse_size=8, n_scales=3):
        super().__init__()
        self.glimpse_size = glimpse_size
        self.n_scales = n_scales
    
    def forward(self, x, location):
        """
        提取多尺度glimpse
        
        参数:
            x: 输入图像 (batch, C, H, W)
            location: 注视点位置 (batch, 2) 归一化到[-1,1]
        
        返回:
            glimpses: 多尺度glimpse拼接 (batch, C * n_scales * glimpse_size^2)
        """
        batch, C, H, W = x.shape
        device = x.device
        
        # 从位置生成采样网格
        glimpses = []
        
        for scale_idx in range(self.n_scales):
            scale = 1.0 / (2 ** scale_idx)
            gs = int(self.glimpse_size * scale)
            
            # 生成局部网格
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-scale, scale, gs, device=device),
                torch.linspace(-scale, scale, gs, device=device),
                indexing='ij'
            )
            grid = torch.stack([x_grid, y_grid], dim=-1)  # (gs, gs, 2)
            grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
            
            # 加上注视点偏移
            loc_grid = location.view(batch, 1, 1, 2)
            grid = grid + loc_grid
            
            # 采样
            glimpse = F.grid_sample(x, grid, align_corners=True)
            glimpses.append(glimpse.view(batch, -1))
        
        return torch.cat(glimpses, dim=1)


class DeepGlimpseNetwork(nn.Module):
    """深度Glimpse网络"""
    
    def __init__(self, glimpse_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(glimpse_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class DRAM(nn.Module):
    """深度循环注意力模型"""
    
    def __init__(self, in_channels=1, glimpse_size=8, n_scales=3,
                 hidden_dim=256, num_layers=2, n_classes=10, n_glimpses=6):
        super().__init__()
        
        self.n_glimpses = n_glimpses
        self.hidden_dim = hidden_dim
        
        # Retina传感器
        self.retina = RetinaSensor(in_channels, glimpse_size, n_scales)
        
        # 计算glimpse总维度
        glimpse_dim = in_channels * sum(
            int(glimpse_size / (2**s)) ** 2 for s in range(n_scales)
        )
        
        # Glimpse网络
        self.glimpse_net = DeepGlimpseNetwork(glimpse_dim, hidden_dim)
        
        # 位置编码
        self.loc_fc = nn.Linear(2, 64)
        
        # 融合层
        self.fusion = nn.Linear(hidden_dim + 64, hidden_dim)
        
        # 核心网络 (多层LSTM)
        self.core = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # 位置网络（输出高斯分布的参数）
        self.loc_mean = nn.Linear(hidden_dim, 2)
        self.loc_std = nn.Linear(hidden_dim, 2)
        
        # 基线网络
        self.baseline = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, num_glimpses=None):
        """
        前向传播（训练模式）
        
        参数:
            x: 输入图像 (batch, C, H, W)
            num_glimpses: 覆盖n_glimpses
        
        返回:
            log_probs: 分类logits
            locations: 所有注视点位置
            rewards: 每个时刻的奖励
            baselines: 基线值
        """
        batch = x.size(0)
        T = num_glimpses if num_glimpses else self.n_glimpses
        device = x.device
        
        # 初始化LSTM状态
        h = torch.zeros(2, batch, self.hidden_dim, device=device)
        c = torch.zeros(2, batch, self.hidden_dim, device=device)
        
        # 从图像中心开始
        location = torch.zeros(batch, 2, device=device)
        locations = []
        log_probs = []
        baselines = []
        
        for t in range(T):
            # 提取glimpse
            glimpse = self.retina(x, location)
            glimpse_feat = self.glimpse_net(glimpse)
            
            # 位置编码
            loc_feat = F.relu(self.loc_fc(location))
            
            # 融合
            combined = self.fusion(torch.cat([glimpse_feat, loc_feat], dim=1))
            
            # LSTM更新
            _, (h, c) = self.core(combined.unsqueeze(1), (h, c))
            h_t = h[-1]  # 取最后一层
            
            # 预测位置（高斯策略）
            mu = torch.tanh(self.loc_mean(h_t))
            log_sigma = torch.clamp(self.loc_std(h_t), min=-5, max=0)
            sigma = torch.exp(log_sigma)
            
            # 采样位置（使用重参数化）
            eps = torch.randn_like(mu)
            location = mu + sigma * eps
            location = torch.clamp(location, -1, 1)
            
            # 计算log概率（用于REINFORCE）
            log_prob = -0.5 * ((location - mu) / sigma) ** 2 \
                       - log_sigma - np.log(np.sqrt(2 * np.pi))
            log_prob = log_prob.sum(dim=1)
            
            # 基线
            baseline = self.baseline(h_t).squeeze()
            
            locations.append(location)
            log_probs.append(log_prob)
            baselines.append(baseline)
        
        # 分类
        logits = self.classifier(h_t)
        
        return logits, torch.stack(locations), torch.stack(log_probs), torch.stack(baselines)
    
    def inference(self, x, num_glimpses=None):
        """推理模式（贪婪）"""
        batch = x.size(0)
        T = num_glimpses if num_glimpses else self.n_glimpses
        device = x.device
        
        h = torch.zeros(2, batch, self.hidden_dim, device=device)
        c = torch.zeros(2, batch, self.hidden_dim, device=device)
        location = torch.zeros(batch, 2, device=device)
        locations = []
        
        with torch.no_grad():
            for t in range(T):
                glimpse = self.retina(x, location)
                glimpse_feat = self.glimpse_net(glimpse)
                loc_feat = F.relu(self.loc_fc(location))
                combined = self.fusion(torch.cat([glimpse_feat, loc_feat], dim=1))
                _, (h, c) = self.core(combined.unsqueeze(1), (h, c))
                mu = torch.tanh(self.loc_mean(h[-1]))
                location = torch.clamp(mu, -1, 1)
                locations.append(location)
            
            logits = self.classifier(h[-1])
        
        return logits, torch.stack(locations)


def demo():
    model = DRAM(in_channels=1, glimpse_size=8, n_scales=3,
                 hidden_dim=128, num_layers=2, n_classes=10, n_glimpses=6)
    x = torch.randn(4, 1, 28, 28)
    
    logits, locs, log_probs, baselines = model(x)
    
    print(f"分类输出: {logits.shape}")
    print(f"注视点序列: {locs.shape}")
    print(f"位置log概率: {log_probs.shape}")
    print(f"基线: {baselines.shape}")
    
    # 推理
    logits_inf, locs_inf = model.inference(x)
    print(f"\n推理分类输出: {logits_inf.shape}")
    print(f"推理注视点: {locs_inf.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""DRAM - 手工LSTM和Glimpse实现"""
import torch
import torch.nn as nn
import numpy as np


class LSTMCellManual(nn.Module):
    """手工LSTM细胞"""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 所有门合并为一个矩阵
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
    
    def forward(self, x, state):
        h, c = state
        
        gates = x @ self.W_ih.t() + h @ self.W_hh.t() + self.bias
        
        # 分割为四个门
        i, f, g, o = gates.chunk(4, dim=-1)
        
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)     # 细胞状态候选
        o = torch.sigmoid(o)  # 输出门
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class RetinaManual(nn.Module):
    """手工Retina传感器（单尺度）"""
    
    def __init__(self, glimpse_size=8):
        super().__init__()
        self.glimpse_size = glimpse_size
    
    def forward(self, x, location):
        batch, C, H, W = x.shape
        gs = self.glimpse_size
        device = x.device
        
        glimpses = torch.zeros(batch, C, gs, gs, device=device)
        
        # 将归一化位置[-1,1]映射到像素坐标
        cx = (location[:, 0] + 1) * (W - 1) / 2
        cy = (location[:, 1] + 1) * (H - 1) / 2
        
        for b in range(batch):
            x_c = cx[b].item()
            y_c = cy[b].item()
            
            half = gs / 2
            x_start = int(x_c - half)
            y_start = int(y_c - half)
            
            for i in range(gs):
                for j in range(gs):
                    xi = x_start + j
                    yi = y_start + i
                    if 0 <= xi < W and 0 <= yi < H:
                        glimpses[b, :, i, j] = x[b, :, yi, xi]
        
        return glimpses.view(batch, -1)


def test_dram_manual():
    """测试手工组件"""
    lstm = LSTMCellManual(64, 128)
    x = torch.randn(2, 64)
    h = torch.zeros(2, 128)
    c = torch.zeros(2, 128)
    h_new, c_new = lstm(x, (h, c))
    print(f"手工LSTM: 输入 {x.shape}, 输出 {h_new.shape}")
    
    retina = RetinaManual(glimpse_size=8)
    img = torch.randn(2, 1, 28, 28)
    loc = torch.zeros(2, 2)
    g = retina(img, loc)
    print(f"手工Retina: 输入 {img.shape}, glimpse {g.shape}")
    
    print("测试通过")


if __name__ == "__main__":
    test_dram_manual()
```

---

## 9. 可视化与结果理解

### 9.1 注视点序列

- 时刻1: 注视图像中心
- 时刻2: 注视最显著的区域
- 时刻3+: 依次关注任务相关的其他区域
- 最终: 集中在可以判别类别的关键区域

### 9.2 Retina多尺度效果

- 中心高分辨率：精确识别纹理和边缘
- 外围低分辨率：获取上下文信息
- 多尺度融合：兼顾局部细节和全局语境

---

## 10. 模型评估

```python
"""DRAM评估"""
import torch
from sklearn.metrics import accuracy_score


def evaluate_dram():
    model = DRAM(in_channels=1, n_classes=10, n_glimpses=6, hidden_dim=128)
    
    # 模拟MNIST数据
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    
    logits, locs, log_probs, baselines = model(x)
    
    # 分类准确率
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y.numpy(), preds.numpy())
    print(f"分类准确率: {acc:.2f}")
    
    # 注视点分布
    print(f"注视点均值: {locs.mean(dim=[0, 2]).numpy()}")
    print(f"注视点标准差: {locs.std(dim=[0, 2]).numpy()}")


if __name__ == "__main__":
    evaluate_dram()
```

---

## 11. 常见问题与易错点

### Q1: DRAM和RAM的核心区别？
**A:** DRAM使用多层LSTM（RAM是单层RNN/GRU），glimpse网络使用深度CNN（RAM是简单全连接），训练引入更多技巧（经验回放、目标网络）。

### Q2: 为什么使用REINFORCE而不是直接BP？
**A:** 位置采样是离散/随机操作，不可微分。REINFORCE通过策略梯度绕过不可微分采样，直接优化期望奖励。

### Q3: 基线b_t的作用？
**A:** 降低REINFORCE梯度估计的方差。相当于比较实际奖励与预期奖励的相对差异，而非绝对奖励值。

### Q4: Retina传感器为什么重要？
**A:** 人眼视觉中中央凹分辨率高、周边分辨率低。Retina模拟这一特性，在减少计算量的同时保留上下文信息。

### Q5: DRAM如何处理多目标？
**A:** 通过多步注视依次关注不同目标。LSTM的隐状态维持了对已访问区域的记忆，避免重复注意。

---

## 12. 学习总结

**核心要点：**
1. Retina传感器提取多尺度局部图像块
2. 深层LSTM维持注意力状态
3. REINFORCE策略梯度训练位置策略
4. 端到端训练分类 + 注意力

**DRAM vs 其他注意力模型：**

| 模型 | 注意力方式 | 训练方法 | 串行/并行 |
|------|-----------|---------|-----------|
| RAM/DRAM | 硬性（离散位置） | 强化学习 | 串行 |
| 软注意力 | 软性（连续权重） | 反向传播 | 可并行 |
| Transformer | 自注意力 | 反向传播 | 可并行 |

---

## 13. 练习题与思考题

### 基础题

**1.** DRAM为什么需要使用强化学习而不是监督学习？

<details>
<summary>答案</summary>
注视点位置是中间决策，没有ground truth标签。强化学习通过奖励信号（分类是否正确）间接学习位置策略，不需要位置标注。
</details>

**2.** Retina传感器中的多尺度有什么作用？

<details>
<summary>答案</summary>
中心高分辨率提供精细特征用于识别，低分辨率外围提供上下文（目标在场景中的位置、与其他目标的关系）。
</details>

**3.** 为什么基线b_t能降低方差？

<details>
<summary>答案</summary>
$(R_t - b_t)$ 可以理解为"比预期好多少"。如果基线准确预测了预期奖励，梯度的方差主要由相对差异贡献，而非绝对奖励值。数学上，最优基线是 $b_t = \mathbb{E}[R_t]$。
</details>

### 进阶题

**4.** 推导REINFORCE算法在DRAM中的梯度公式。

<details>
<summary>答案</summary>
轨迹 $\tau = (l_1, ..., l_T)$ 的期望奖励梯度: $\nabla J = \mathbb{E}[\sum_t \nabla \log \pi(l_t|h_t) \cdot R_t]$。推导: $J = \int p(\tau)R(\tau)d\tau$, $\nabla J = \int \nabla p(\tau) R(\tau) d\tau = \int p(\tau) \nabla \log p(\tau) R(\tau) d\tau = \mathbb{E}[\nabla \log p(\tau) R(\tau)]$。由于 $p(\tau) = \prod \pi(l_t|h_t)$，$\nabla \log p(\tau) = \sum \nabla \log \pi(l_t|h_t)$。
</details>

**5.** 如何修改DRAM使其支持软注意力？

<details>
<summary>答案</summary>
将位置采样替换为注意力权重分布 $\alpha_t = \text{softmax}(f(h_t))$，glimpse特征变为所有位置的加权和 $g_t = \sum \alpha_{t,i} \cdot x_i$。此时不需要REINFORCE，可直接反向传播。
</details>

---

## 14. 学习路径建议

### 预备知识
- RNN/LSTM基础
- 强化学习基础（策略梯度）
- CNN特征提取
- PyTorch高级API

### 进阶方向
1. **DRAM -> Soft Attention**：可微分的软注意力机制
2. **DRAM -> A2C/PPO**：更先进的强化学习算法
3. **DRAM -> Transformer**：完全基于注意力的架构
4. **DRAM -> VisualRL**：视觉强化学习的通用框架

### 推荐阅读
- Mnih et al. "Recurrent Models of Visual Attention." NIPS 2014.
- Ba et al. "Multiple Object Recognition with Visual Attention." ICLR 2015.
- Williams. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." 1992.

### 项目实践
1. 在MNIST上训练DRAM并可视化注意力轨迹
2. 在CIFAR-10上比较DRAM与全卷积网络的性能
3. 实现DRAM的soft attention变体并比较
