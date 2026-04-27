# 空间变换模块STM 学习文档

> 可微分的空间注意力——在CNN中嵌入空间变换能力。
>
> 来源线索：本节内容根据原书第2章关于"目标搜索与识别"的相关章节整理。

---

## 1. 算法基础认知

**一句话定义：** 空间变换模块（Spatial Transformer Module, STM）是一种可微分的、即插即用的空间注意力组件，通过学习的仿射变换参数对特征图进行空间对齐，使网络获得空间不变性。

**核心思想：** 传统CNN通过池化层和卷积层的堆叠获得一定的空间不变性，但这种方法是被动的、低效的。STM通过一个轻量级的子网络主动学习空间变换参数（旋转、缩放、平移、裁剪），然后用这些参数对特征图进行重新采样，将感兴趣的区域"扶正"到规范位置。

**为什么可微分重要？** STM中的所有操作（定位网络、网格生成、采样器）都是可微分的，因此可以通过反向传播端到端训练。这使得网络可以自己学会如何根据任务需要变换特征图。

**STM与STN的关系：** STM（Spatial Transformer Module）本质上是STN（Spatial Transformer Network，Google DeepMind 2015）的同义概念。STN首次提出可微分空间变换，STM强调其作为"即插即用模块"的特性。两者核心结构完全一致。

---

## 2. 核心原理

STM由三个子组件构成：

### 2.1 定位网络（Localization Network）

输入特征图 $U \in \mathbb{R}^{H \times W \times C}$，输出变换参数 $\theta$：

$$
\theta = f_{loc}(U)
$$

对于仿射变换，$\theta \in \mathbb{R}^{6}$（2×3矩阵）：

$$
\theta = \begin{bmatrix}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{bmatrix}
$$

定位网络可以是任何子网络（卷积+全连接），最后输出6个值。初始化时，$\theta$ 应设置为恒等变换：$\theta = [1, 0, 0, 0, 1, 0]$。

### 2.2 网格生成器（Grid Generator）

对输出特征图 $V \in \mathbb{R}^{H' \times W' \times C}$ 中的每个像素 $(x_i^t, y_i^t)$，计算其在输入 $U$ 中对应的采样点 $(x_i^s, y_i^s)$：

$$
\begin{pmatrix}
x_i^s \\
y_i^s
\end{pmatrix}
= \mathcal{T}_\theta(G_i) = 
\begin{bmatrix}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{bmatrix}
\begin{pmatrix}
x_i^t \\
y_i^t \\
1
\end{pmatrix}
$$

### 2.3 采样器（Sampler）

使用双线性插值从输入 $U$ 中采样得到输出 $V$：

$$
V_i = \sum_n^H \sum_m^W U_{nm} \cdot \max(0, 1 - |x_i^s - m|) \cdot \max(0, 1 - |y_i^s - n|)
$$

双线性插值是可微分的，其梯度可以回传到定位网络。

---

## 3. 数学公式与推导

### 3.1 仿射变换矩阵

仿射变换的6个参数分别控制：

$$
\theta = \begin{bmatrix}
s_x \cos \alpha & -s_y \sin \beta & t_x \\
s_x \sin \alpha & s_y \cos \beta & t_y
\end{bmatrix}
$$

其中 $s_x, s_y$ 是缩放，$\alpha, \beta$ 是旋转和剪切，$t_x, t_y$ 是平移。

恒等变换时：$s_x = s_y = 1$，$\alpha = \beta = 0$，$t_x = t_y = 0$。

### 3.2 双线性插值的梯度

双线性插值对采样坐标的梯度：

$$
\frac{\partial V_i}{\partial x_i^s} = \sum_n^H \sum_m^W U_{nm} \cdot \text{sign}(x_i^s - m) \cdot \max(0, 1 - |y_i^s - n|)
$$

其中 $\text{sign}(x_i^s - m)$ 是符号函数。梯度进一步通过网格生成器回传到 $\theta$。

### 3.3 反向传播

通过链式法则：

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial V} \cdot \frac{\partial V}{\partial (x^s, y^s)} \cdot \frac{\partial (x^s, y^s)}{\partial \theta}
$$

其中 $\frac{\partial (x^s, y^s)}{\partial \theta} = (x^t, y^t, 1)$ 来自网格生成器。

---

## 4. 训练过程讲解

STM通过端到端反向传播与主网络一起训练。

**前向传播：**
1. 输入特征图 $U$ 进入STM
2. 定位网络预测变换参数 $\theta$
3. 网格生成器计算采样坐标
4. 采样器输出变换后的特征图 $V$
5. $V$ 送入后续网络层

**反向传播：**
1. 损失函数 $L$ 的梯度从后续层传到 $V$
2. 通过采样器传到采样坐标
3. 通过网格生成器传到 $\theta$
4. 通过定位网络更新其参数

**初始化技巧：** 定位网络的输出层偏置初始化为恒等变换参数，确保训练初期网络行为不变。

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 图像分类 | 对旋转/平移/缩放的鲁棒性 |
| 手写数字识别 | 将倾斜的数字扶正 |
| 人脸对齐 | 自动检测并对齐人脸关键点 |
| 街景文字识别 | 将透视变形的文字矫正 |
| 目标检测 | 生成旋转不变的区域特征 |
| 视觉定位 | 学习视角不变的匹配 |
| 视频稳定 | 学习帧间的变换参数 |

---

## 6. 优缺点分析

**优点：**
- ✅ **可微分**：端到端训练，无需额外标注
- ✅ **即插即用**：可插入CNN的任何位置
- ✅ **显式空间变换**：比池化更主动、更灵活
- ✅ **参数少**：仅需6个参数 + 轻量级定位网络
- ✅ **多任务适用**：分类、检测、分割均可

**缺点：**
- ❌ **仅限仿射变换**：表达能力有限
- ❌ **增加计算量**：额外的定位网络和采样操作
- ❌ **训练不稳定**：初始阶段变换可能过于剧烈
- ❌ **无法处理遮挡**：变换不改变可见内容
- ❌ **特征图裁剪问题**：变换可能导致部分特征图区域空白

---

## 7. 调库实现

```python
"""空间变换模块STM - PyTorch完整实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SpatialTransformerModule(nn.Module):
    """
    空间变换模块 (即插即用)
    
    可插入CNN的任何层之间，学习仿射变换参数
    对特征图进行空间对齐
    """
    
    def __init__(self, in_channels):
        """
        参数:
            in_channels: 输入特征图的通道数
        """
        super().__init__()
        
        # 定位网络: 学习变换参数
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)  # 输出6个仿射变换参数
        )
        
        # 初始化参数为恒等变换
        self._init_weights()
    
    def _init_weights(self):
        """初始化为恒等变换: [[1,0,0],[0,1,0]]"""
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图 (batch, C, H, W)
        
        返回:
            out: 空间变换后的特征图 (batch, C, H', W')
        """
        batch_size = x.size(0)
        
        # 1. 定位网络预测变换参数
        theta = self.localization(x)  # (batch, 6)
        theta = theta.view(batch_size, 2, 3)  # (batch, 2, 3)
        
        # 2. 生成采样网格
        # F.affine_grid 根据 theta 和输出尺寸生成采样坐标
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        # 3. 双线性采样
        out = F.grid_sample(x, grid, mode='bilinear', 
                           padding_mode='zeros', align_corners=True)
        
        return out
    
    def get_transform_matrix(self, x):
        """获取预测的仿射变换矩阵（用于可视化）"""
        theta = self.localization(x)
        return theta.view(-1, 2, 3)


class CNNWithSTM(nn.Module):
    """带有STM的CNN分类网络示例"""
    
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # STM插入在第一个卷积层之后
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.stm = SpatialTransformerModule(16)
        
        # 后续分类网络
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.stm(x)  # 空间变换
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def demo():
    """演示STM前向传播"""
    stm = SpatialTransformerModule(in_channels=3)
    
    # 模拟输入
    x = torch.randn(2, 3, 28, 28)
    y = stm(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 显示变换矩阵
    theta = stm.get_transform_matrix(x)
    print(f"变换矩阵 (batch 0):\n{theta[0].detach().numpy()}")
    
    # 带STM的分类网络
    model = CNNWithSTM(in_channels=1, num_classes=10)
    x2 = torch.randn(4, 1, 28, 28)
    out = model(x2)
    print(f"分类输出: {out.shape}")


def visualize_stm_effect():
    """可视化STM的效果"""
    stm = SpatialTransformerModule(in_channels=1)
    
    # 创建包含数字的简单图像
    x = torch.zeros(1, 1, 28, 28)
    x[0, 0, 5:23, 8:20] = 1.0  # 矩形"数字"
    
    # 前向传播
    y = stm(x)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x[0, 0].detach().numpy(), cmap='gray')
    axes[0].set_title('输入')
    axes[1].imshow(y[0, 0].detach().numpy(), cmap='gray')
    axes[1].set_title('STM输出')
    
    # 获取变换矩阵
    theta = stm.get_transform_matrix(x)
    print(f"变换矩阵: {theta[0].detach().numpy().round(3)}")
    
    plt.tight_layout()
    plt.savefig('stm_effect.png', dpi=150)


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""STM - 手工实现仿射变换和双线性采样"""
import torch
import torch.nn as nn
import numpy as np


def affine_grid_manual(theta, out_size):
    """
    手工仿射网格生成
    
    参数:
        theta: (batch, 2, 3) 仿射矩阵
        out_size: (batch, C, H, W) 输出尺寸
    
    返回:
        grid: (batch, H, W, 2) 采样坐标 (归一化到 [-1, 1])
    """
    batch, C, H, W = out_size
    
    # 生成输出坐标网格
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    
    # 齐次坐标: (x, y, 1) for each position
    ones = torch.ones_like(x)
    coords = torch.stack([x, y, ones], dim=0)  # (3, H, W)
    coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, 3, H, W)
    
    # 仿射变换: (batch, 2, 3) @ (batch, 3, H, W) -> (batch, 2, H, W)
    grid = torch.bmm(theta, coords.view(batch, 3, -1))
    grid = grid.view(batch, 2, H, W).permute(0, 2, 3, 1)  # (batch, H, W, 2)
    
    return grid


def bilinear_sample_manual(input, grid):
    """
    手工双线性采样
    
    参数:
        input: (batch, C, H, W)
        grid: (batch, H_out, W_out, 2) 采样坐标 [-1, 1]
    
    返回:
        output: (batch, C, H_out, W_out)
    """
    batch, C, H, W = input.shape
    H_out, W_out = grid.shape[1], grid.shape[2]
    
    # 将坐标从 [-1, 1] 映射到 [0, H-1] 和 [0, W-1]
    x = (grid[..., 0] + 1) * (W - 1) / 2
    y = (grid[..., 1] + 1) * (H - 1) / 2
    
    output = torch.zeros(batch, C, H_out, W_out)
    
    for b in range(batch):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    xi = x[b, i, j].item()
                    yi = y[b, i, j].item()
                    
                    # 四个邻域像素
                    x0 = int(torch.floor(xi).item())
                    x1 = x0 + 1
                    y0 = int(torch.floor(yi).item())
                    y1 = y0 + 1
                    
                    # 边界裁剪
                    x0 = max(0, min(W-1, x0))
                    x1 = max(0, min(W-1, x1))
                    y0 = max(0, min(H-1, y0))
                    y1 = max(0, min(H-1, y1))
                    
                    # 双线性插值权重
                    wx = xi - x0
                    wy = yi - y0
                    
                    # 插值
                    val = (1-wy)*(1-wx) * input[b, c, y0, x0].item() + \
                          (1-wy)*wx     * input[b, c, y0, x1].item() + \
                          wy*(1-wx)     * input[b, c, y1, x0].item() + \
                          wy*wx         * input[b, c, y1, x1].item()
                    
                    output[b, c, i, j] = val
    
    return output


class STMManual(nn.Module):
    """手工实现的STM"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 6),
        )
        # 初始化为恒等变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        theta = self.localization(x).view(-1, 2, 3)
        grid = affine_grid_manual(theta, x.shape)
        out = bilinear_sample_manual(x, grid)
        return out


def test_stm_manual():
    model = STMManual(1)
    x = torch.randn(1, 1, 8, 8)
    y = model(x)
    print(f"手工STM: 输入 {x.shape} -> 输出 {y.shape}")
    assert y.shape == x.shape, "输出尺寸应与输入相同"


if __name__ == "__main__":
    test_stm_manual()
```

---

## 9. 可视化与结果理解

### 9.1 变换矩阵解读

以恒等变换为基准：
- $\theta_{13}$ 和 $\theta_{23}$：水平和垂直平移
- $\theta_{11}$ 和 $\theta_{22}$：水平和垂直缩放
- $\theta_{12}$ 和 $\theta_{21}$：旋转和剪切

### 9.2 特征图变化

- 输入：数字"5"倾斜在左上角
- STM后：数字"5"被扶正到中心
- 后续分类器：更容易正确分类

---

## 10. 模型评估

```python
"""STM在MNIST上的评估"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def evaluate_stm():
    """在MNIST上比较有无STM的分类性能"""
    
    # 有STM的模型
    class ModelWithSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
            self.stm = SpatialTransformerModule(16)
            self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
            self.fc = nn.Linear(32 * 7 * 7, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.stm(x)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    print("STM模型参数量:", sum(p.numel() for p in ModelWithSTM().parameters()))


if __name__ == "__main__":
    evaluate_stm()
```

---

## 11. 常见问题与易错点

### Q1: 为什么STM的定位网络输出要初始化为恒等变换？
**A:** 训练初期网络参数随机时，变换可能剧烈扭曲特征图，导致梯度爆炸或信息丢失。恒等初始化确保一开始不做变换，让网络逐步学习。

### Q2: 双线性插值为什么是可微分的？
**A:** 双线性插值是连续函数，其偏导数 $\partial V / \partial x^s$ 和 $\partial V / \partial y^s$ 存在（分段常数），因此可以通过梯度回传更新定位网络。

### Q3: STM可以处理哪些变换？
**A:** 基本的是仿射变换（6参数），还可以扩展为薄板样条（TPS，更多参数）处理更复杂的非线性变形。

### Q4: STM在分类任务中一定能提升性能吗？
**A:** 不一定。如果数据已经对齐（如MNIST），STM不会带来明显提升。对于存在几何变形的数据（如街景文字、航拍图像），STM效果显著。

### Q5: padding_mode参数的作用？
**A:** 采样坐标可能超出输入边界，padding_mode='zeros'在边界外补零，'border'用边界值填充，'reflection'镜像填充。

---

## 12. 学习总结

**核心要点：**
1. 三组件：定位网络 + 网格生成器 + 双线性采样器
2. 可微分：所有操作支持反向传播
3. 即插即用：可插入CNN的任何位置
4. 学习空间不变性：旋转、缩放、平移、裁剪

**关键公式：**
$$
\theta = f_{loc}(U), \quad (x^s, y^s) = \theta \cdot (x^t, y^t, 1)^T, \quad V = \text{采样}(U, x^s, y^s)
$$

---

## 13. 练习题与思考题

### 基础题

**1.** 6个仿射参数分别控制什么？

<details>
<summary>答案</summary>
$\theta_{11}$: x方向缩放和旋转余弦；$\theta_{12}$: x方向剪切和旋转正弦的负值；$\theta_{13}$: x方向平移；$\theta_{21}$: y方向旋转正弦；$\theta_{22}$: y方向缩放和旋转余弦；$\theta_{23}$: y方向平移。
</details>

**2.** 为什么需要双线性插值而不是最近邻？

<details>
<summary>答案</summary>
最近邻不可微分（梯度为零或不存在），无法反向传播。双线性插值是连续的且几乎处处可微，允许梯度流动。
</details>

**3.** 如果定位网络输出全零，会发生什么？

<details>
<summary>答案</summary>
全零矩阵映射所有输出像素到输入坐标(0,0)，输出特征图变成常数（输入左上角像素值）。这通常是不希望的，所以初始化为恒等变换。
</details>

### 进阶题

**4.** 如何修改STM支持薄板样条（TPS）变换？

<details>
<summary>答案</summary>
定位网络改为输出控制点坐标（如5×5=25个控制点），网格生成器改为计算TPS插值（径向基函数），采样器不变。这比仿射变换更灵活，能处理局部变形。
</details>

**5.** 为什么STM对大幅旋转图像有效但小幅平移效果不显著？

<details>
<summary>答案</summary>
CNN的卷积操作本身具有一定平移不变性（通过参数共享和池化），但旋转不变性需要大量数据学习。STM正是补足了CNN在旋转不变性上的不足。
</details>

---

## 14. 学习路径建议

### 预备知识
- CNN基础（卷积、池化、反向传播）
- 图像变换（仿射变换、双线性插值）
- PyTorch的 autograd 机制

### 进阶方向
1. **STM -> STN**：完整的空间变换网络
2. **STM -> Deformable Convolution**：可变形卷积，更灵活的采样
3. **STM -> TPS-STN**：薄板样条变换网络
4. **STM -> Deep Feature Flow**：视频中的变形对齐

### 推荐阅读
- Jaderberg et al. "Spatial Transformer Networks." NIPS 2015.
- Dai et al. "Deformable Convolutional Networks." ICCV 2017.

### 项目实践
1. 在MNIST-rot数据集上比较有无STM的分类准确率
2. 可视化STM学习的变换参数在不同输入下的变化
3. 将STM插入预训练ResNet并微调
