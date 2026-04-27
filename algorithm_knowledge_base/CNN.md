# 卷积神经网络 (CNN) 学习文档

> 用卷积核提取图像局部特征，是计算机视觉的基石。

> 来源线索：本节内容根据原书中关于"卷积神经网络"的相关章节（第2章2.4节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** CNN 通过卷积核在图像上滑动提取局部特征，再用池化压缩信息，最终实现分类或检测。

**直觉类比：** 想象你用一个手电筒（卷积核）在一张照片上逐行扫描——每次只照亮一小块区域，把这一块的特征（边缘、纹理、颜色变化）记录下来。扫描完整张照片后，你就得到了一张"特征地图"。再用多层手电筒扫描这些特征地图，就能从低级特征（边缘）逐步组合出高级特征（眼睛、轮子、猫脸）。

**历史背景：** CNN 的思想最早可追溯到 1998 年 Yann LeCun 提出的 LeNet-5，用于手写数字识别。2012 年 AlexNet 在 ImageNet 竞赛中大幅领先，开启了深度学习在计算机视觉领域的爆发式发展。后续 VGGNet、GoogLeNet、ResNet 等经典网络不断刷新性能纪录。

**算法定位：** 监督学习模型，主要用于图像分类、目标检测、语义分割等计算机视觉任务，也可用于序列数据处理（1D-CNN）。

**前置知识：** 线性代数（矩阵运算）、微积分（梯度下降）、Python 编程基础、PyTorch 基础。

---

## 2. 核心原理

### 核心思想

CNN 的核心思想是**局部连接**和**参数共享**。与全连接层每个神经元连接所有输入不同，卷积层的每个神经元只连接输入的一小块局部区域（感受野），且同一层所有位置共享同一组卷积核参数。这使得 CNN 能用远少于全连接网络的参数量高效提取图像的空间特征。

### 工作流程

1. **输入**：一张图像（如 $3 \times 224 \times 224$ 的 RGB 图像）
2. **卷积层**：用多个卷积核在输入上滑动，计算局部加权和，输出特征图
3. **激活函数**：对特征图施加 ReLU 等非线性变换
4. **池化层**：对特征图进行下采样（如最大池化），减小尺寸、增强平移不变性
5. **重复卷积+激活+池化**：逐层提取更高阶特征
6. **展平+全连接**：将特征图展平为一维向量，通过全连接层输出分类结果
7. **Softmax**：输出各类别的概率分布

### 关键概念

- **卷积核（Filter/Kernel）**：一个小型权重矩阵（如 $3 \times 3$），在输入上滑动提取局部特征。可以检测边缘、纹理等模式。
- **步幅（Stride）**：卷积核每次移动的像素数。stride=1 逐像素移动，stride=2 每次跳两格。
- **填充（Padding）**：在输入边缘补零，控制输出尺寸。Same 填充保持尺寸不变，Valid 填充不补零。
- **感受野（Receptive Field）**：输出特征图上一个像素对应的输入图像区域大小。
- **多通道**：每个卷积核的深度与输入通道数一致，输出时所有通道的卷积结果相加得到一个输出通道。

### 几何/直观解释

```
输入图像          卷积核          特征图
┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐
│ 1 │ 1 │ 1 │ 0 │ 0 │   │ 1 │ 0 │ 1 │   │ 4 │ 3 │ 4 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │   │ 0 │ 1 │ 0 │   │ 2 │ 4 │ 3 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
│ 0 │ 0 │ 1 │ 1 │ 1 │   │ 1 │ 0 │ 1 │   │ 2 │ 3 │ 4 │
├───┼───┼───┼───┼───┤   └───┴───┴───┘   └───┴───┴───┘
│ 0 │ 0 │ 1 │ 1 │ 0 │      3×3 核          3×3 输出
├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 0 │ 0 │
└───┴───┴───┴───┴───┘
     5×5 输入

计算: 4 = 1×1 + 1×0 + 1×1 + 0×0 + 1×1 + 1×0 + 0×1 + 0×0 + 1×1
```

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 输入张量，形状 $(N, C_{in}, H_{in}, W_{in})$ |
| $W$ | 卷积核权重，形状 $(C_{out}, C_{in}, k_h, k_w)$ |
| $b$ | 偏置，形状 $(C_{out})$ |
| $s$ | 步幅 (stride) |
| $p$ | 填充 (padding) |
| $Y$ | 输出特征图 |

### 二维卷积运算

对于输入矩阵 $X$ 和卷积核 $K$（大小 $k_h \times k_w$），卷积运算定义为：

$$Y[i,j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[i+m, j+n] \cdot K[m,n]$$

这里 $Y[i,j]$ 是输出特征图在位置 $(i,j)$ 的值，通过输入局部区域与卷积核的逐元素相乘再求和得到。

### 输出尺寸计算

给定输入尺寸 $H_{in} \times W_{in}$，卷积核大小 $k_h \times k_w$，步幅 $s$，填充 $p$，输出尺寸为：

$$H_{out} = \lfloor \frac{H_{in} + 2p - k_h}{s} \rfloor + 1$$

$$W_{out} = \lfloor \frac{W_{in} + 2p - k_w}{s} \rfloor + 1$$

当使用 Same 填充时，填充量 $p = \frac{k - 1}{2}$（要求 $k$ 为奇数），此时若 $s=1$，则 $H_{out} = H_{in}$。

### 多通道卷积

对于 $C_{in}$ 个输入通道和 $C_{out}$ 个输出通道：

$$Y[c_{out}, i, j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[c_{in}, i\cdot s+m, j\cdot s+n] \cdot W[c_{out}, c_{in}, m, n] + b[c_{out}]$$

每个输出通道是由所有输入通道的卷积结果相加再加上偏置得到的。这就是为什么卷积核的参数量为 $C_{out} \times C_{in} \times k_h \times k_w$。

### 参数量分析

一个卷积层的参数量：

$$\text{Params} = C_{out} \times C_{in} \times k_h \times k_w + C_{out}$$

对比全连接层（$H_{in} \times W_{in} \times C_{in} \times C_{out}$），卷积层通过参数共享大幅减少了参数量。例如 $3 \times 3$ 卷积核只需 $9 \times C_{in}$ 个参数（每个输出通道），而全连接层需要 $H_{in} \times W_{in} \times C_{in}$ 个参数。

---

## 4. 训练过程讲解

### 数据预处理

- **归一化**：将像素值从 [0, 255] 缩放到 [0, 1] 或标准化为均值 0、标准差 1
- **数据增强**：随机裁剪、水平翻转、颜色抖动等，增加训练数据多样性
- **Resize**：将图像调整为统一尺寸（如 224×224）

### 参数初始化

- 常用 He 初始化（Kaiming 初始化）：适用于 ReLU 激活函数
- $W \sim \mathcal{N}(0, \sqrt{2/C_{in} \cdot k_h \cdot k_w})$
- 偏置初始化为 0

### 迭代过程

每轮训练：
1. 从数据加载器获取一个 batch 的图像和标签
2. 前向传播：图像经过卷积层 → 激活 → 池化 → 全连接 → 输出概率
3. 计算损失（交叉熵损失）
4. 反向传播：计算各层参数梯度
5. 参数更新：优化器根据梯度更新权重

### 收敛条件

- 验证集损失连续 N 个 epoch 不下降（早停）
- 达到最大训练轮数
- 训练损失低于阈值

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| learning_rate | 控制参数更新步长 | 1e-4 ~ 1e-2 | 1e-3 |
| batch_size | 每批训练样本数 | 16 ~ 256 | 64 |
| num_epochs | 训练轮数 | 10 ~ 200 | 50 |
| kernel_size | 卷积核大小 | 3, 5, 7 | 3 |
| num_filters | 每层卷积核数量 | 16 ~ 512 | 逐层加倍 |
| stride | 步幅 | 1, 2 | 1 |
| padding | 填充 | 0, 1, "same" | 1 |
| dropout | 随机失活率 | 0.2 ~ 0.5 | 0.3 |

---

## 5. 应用场景

### 1. 图像分类
将输入图像归类到预定义类别。CNN 能从像素级信息逐层提取语义特征，非常适合此任务。典型应用：ImageNet 分类、医学影像诊断。

### 2. 目标检测
在图像中定位并识别多个目标。CNN 的特征提取能力使其能精确识别不同尺度、位置的物体。典型应用：自动驾驶中的行人和车辆检测、安防监控。

### 3. 语义分割
对图像的每个像素进行分类。CNN 通过编码器提取特征，再通过解码器恢复像素级预测。典型应用：自动驾驶道路分割、医学图像器官分割。

### 4. 人脸识别
从人脸图像中提取身份特征向量。CNN 学到的深层特征对人脸身份具有高度区分性。典型应用：人脸解锁、身份验证。

### 不适用场景
- 纯序列建模任务（RNN/Transformer 更合适）
- 数据量极少的任务（容易过拟合）
- 不具有空间结构的表格数据

---

## 6. 优缺点分析

### 优点

1. **参数共享大幅减少参数量**：同一卷积核在所有位置共享参数，相比全连接层参数量减少几个数量级。这使得训练更高效，且不易过拟合。

2. **保留空间结构信息**：卷积操作天然保持了输入的二维空间关系，有利于提取图像的局部模式（边缘、纹理、形状）。

3. **平移等变性**：目标在图像中平移后，卷积特征图也相应平移，使得模型对目标位置具有一定鲁棒性。

4. **层次化特征提取**：浅层提取边缘纹理等低级特征，深层提取语义等高级特征，形成从简单到复杂的特征层次。

### 缺点

1. **感受野有限**：单次卷积只看局部区域，需要堆叠多层才能覆盖全局信息。缓解思路：使用空洞卷积（dilation）扩大感受野，或引入全局池化。

2. **对旋转变换不鲁棒**：标准 CNN 对目标旋转不具备不变性。缓解思路：数据增强中加入旋转，或使用旋转等变卷积。

3. **计算资源需求大**：深层 CNN 需要大量 GPU 内存和计算力。缓解思路：使用深度可分离卷积（MobileNet）、模型剪枝、量化等技术。

### 与同类算法对比

| 特性 | CNN | 全连接网络 | ViT (Vision Transformer) |
|------|-----|-----------|------------------------|
| 参数效率 | 高（参数共享） | 低 | 中 |
| 空间信息 | 保留 | 丢失 | 保留 |
| 归纳偏置 | 强（局部性、平移等变性） | 无 | 弱 |
| 小数据表现 | 好 | 一般 | 较差 |
| 大数据表现 | 好 | 一般 | 极好 |
| 训练速度 | 快 | 慢 | 慢 |
| 可解释性 | 中（可视化特征图） | 低 | 中 |

---

## 7. 调库实现

使用 PyTorch 构建一个 CNN 对 MNIST 手写数字进行分类：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备：下载 MNIST 数据集并做简单预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量，像素值归一化到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积块：1通道输入 → 32通道输出
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 32×28×28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 输出: 32×14×14

        # 第二个卷积块：32通道 → 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出: 64×14×14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 输出: 64×7×7

        # 全连接分类器
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)  # 10 个数字类别

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平为一维
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# 3. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练 5 个 epoch
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()       # 梯度清零
        output = model(batch_x)     # 前向传播
        loss = criterion(output, batch_y)  # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    print(f'Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}, '
          f'Accuracy: {100.*correct/total:.2f}%')

# 4. 测试模型
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        _, predicted = output.max(1)
        test_correct += predicted.eq(batch_y).sum().item()
        test_total += batch_y.size(0)

print(f'测试集准确率: {100.*test_correct/test_total:.2f}%')
```

**运行结果示例：**
```
Epoch 1/5, Loss: 0.1623, Accuracy: 95.12%
Epoch 2/5, Loss: 0.0521, Accuracy: 98.38%
Epoch 3/5, Loss: 0.0387, Accuracy: 98.80%
Epoch 4/5, Loss: 0.0298, Accuracy: 99.06%
Epoch 5/5, Loss: 0.0254, Accuracy: 99.20%
测试集准确率: 99.15%
```

---

## 8. 手工代码实现

使用 NumPy 从零实现卷积运算和最大池化的核心逻辑：

```python
import numpy as np

class Conv2D:
    """手工实现的二维卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # He 初始化：适合 ReLU 激活函数
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        前向传播
        x: 输入张量，形状 (N, C_in, H, W)
        返回: 输出张量，形状 (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = self.weights.shape

        # 填充输入
        if self.padding > 0:
            x_padded = np.zeros((N, C_in, H + 2*self.padding, W + 2*self.padding))
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            x_padded = x

        # 计算输出尺寸
        H_out = (H + 2*self.padding - kh) // self.stride + 1
        W_out = (W + 2*self.padding - kw) // self.stride + 1

        # 执行卷积运算：对每个样本、每个输出通道、每个空间位置计算加权和
        output = np.zeros((N, C_out, H_out, W_out))
        for n in range(N):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        # 提取局部区域
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = x_padded[n, :, h_start:h_start+kh, w_start:w_start+kw]
                        # 加权求和：所有输入通道的卷积结果相加
                        output[n, co, i, j] = np.sum(patch * self.weights[co]) + self.bias[co]
        return output


class MaxPool2D:
    """手工实现的最大池化层"""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        前向传播
        x: 输入张量，形状 (N, C, H, W)
        """
        N, C, H, W = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        output = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                # 取每个窗口中的最大值
                patch = x[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                output[:, :, i, j] = np.max(patch, axis=(2, 3))
        return output


# 测试代码
if __name__ == '__main__':
    np.random.seed(42)

    # 模拟一个 batch 的单通道 8x8 图像
    x = np.random.randn(2, 1, 8, 8)

    # 卷积层：1通道输入 → 4通道输出，3×3 卷积核，padding=1 保持尺寸
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
    conv_out = conv.forward(x)
    print(f'卷积输出形状: {conv_out.shape}')  # 预期: (2, 4, 8, 8)

    # ReLU 激活
    relu_out = np.maximum(conv_out, 0)
    print(f'ReLU 输出形状: {relu_out.shape}')  # 预期: (2, 4, 8, 8)

    # 最大池化：2×2 窗口，尺寸减半
    pool = MaxPool2D(pool_size=2, stride=2)
    pool_out = pool.forward(relu_out)
    print(f'池化输出形状: {pool_out.shape}')  # 预期: (2, 4, 4, 4)
```

**运行结果示例：**
```
卷积输出形状: (2, 4, 8, 8)
ReLU 输出形状: (2, 4, 8, 8)
池化输出形状: (2, 4, 4, 4)
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 可视化卷积核和特征图
def visualize_conv_features():
    """可视化第一层卷积核学到的特征"""
    # 创建一个简单的 CNN 并训练
    model = SimpleCNN()

    # 读取一个 MNIST 样本
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = dataset[0]

    # 提取第一层卷积的特征图
    with torch.no_grad():
        # 获取第一层卷积输出
        feature_maps = model.relu1(model.conv1(image.unsqueeze(0)))

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('第一层卷积特征图可视化', fontsize=14)

    # 显示原始图像
    axes[0, 0].imshow(image.squeeze(), cmap='gray')
    axes[0, 0].set_title(f'原始图像\n标签: {label}')
    axes[0, 0].axis('off')

    # 显示前 7 个特征图
    for i in range(7):
        axes[0, i+1].imshow(feature_maps[0, i].numpy(), cmap='viridis')
        axes[0, i+1].set_title(f'特征图 {i+1}')
        axes[0, i+1].axis('off')

    # 显示更多特征图
    for i in range(8):
        if i < feature_maps.shape[1] - 8:
            axes[1, i].imshow(feature_maps[0, i+8].numpy(), cmap='viridis')
            axes[1, i].set_title(f'特征图 {i+9}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('cnn_features.png', dpi=100, bbox_inches='tight')
    plt.show()

# 可视化不同卷积核的作用
def visualize_kernel_effects():
    """展示不同卷积核对图像的不同效果"""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image = dataset[0][0].squeeze().numpy()  # 获取一张 28×28 图像

    # 定义不同类型的卷积核
    kernels = {
        '水平边缘检测': np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
        '垂直边缘检测': np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
        '锐化': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        '模糊': np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]),
    }

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    for idx, (name, kernel) in enumerate(kernels.items()):
        # 手动实现卷积
        h, w = image.shape
        kh, kw = kernel.shape
        output = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

        axes[idx+1].imshow(output, cmap='gray')
        axes[idx+1].set_title(name)
        axes[idx+1].axis('off')

    plt.suptitle('不同卷积核对图像的作用', fontsize=14)
    plt.tight_layout()
    plt.savefig('cnn_kernels.png', dpi=100, bbox_inches='tight')
    plt.show()

visualize_kernel_effects()
```

**结果解读：**
- 水平边缘检测核会在图像中水平边界处产生高亮响应，突出水平线条
- 垂直边缘检测核类似地突出垂直线条
- 锐化核增强图像中的细节和对比度
- 模糊核（均值滤波）使图像变得平滑，减弱噪声

---

## 10. 模型评估

### 评估指标

对于图像分类任务，使用以下指标：

1. **准确率（Accuracy）**：分类正确的样本比例，适合类别均衡的数据
2. **混淆矩阵**：直观展示各类别的分类情况，发现容易混淆的类别对
3. **分类报告**：包含每个类别的精确率、召回率、F1 值

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device):
    """全面评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 准确率
    accuracy = (all_preds == all_labels).mean()
    print(f'总体准确率: {accuracy*100:.2f}%\n')

    # 分类报告
    print("分类报告:")
    print(classification_report(all_labels, all_preds, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:")
    print(cm)

    return accuracy

# 使用示例（需要先训练模型）
# evaluate_model(model, test_loader, device)
```

---

## 11. 常见问题与易错点

### 数据层面

1. **忘记归一化导致训练不收敛**
   - 现象：损失值一直很高不下降，或出现 NaN
   - 原因：像素值 [0, 255] 范围太大，梯度爆炸
   - 解决：必须将像素值归一化到 [0, 1] 或标准化

2. **数据增强过于激进**
   - 现象：训练准确率低，验证准确率反而更高
   - 原因：过多的增强（如大角度旋转、强色彩变换）破坏了原始信息
   - 解决：根据任务选择合理的增强方法，分类任务可用水平翻转，数字识别则不行

3. **batch_size 设置不当**
   - 现象：训练不稳定或速度慢
   - 原因：batch_size 太小导致梯度估计方差大，太大导致泛化能力差
   - 解决：从 32 或 64 开始，根据 GPU 内存调整

### 模型层面

1. **卷积层输出尺寸计算错误**
   - 现象：RuntimeError: size mismatch
   - 原因：全连接层的输入维度与卷积层输出不匹配
   - 解决：先打印卷积输出的 shape，据此计算全连接输入维度

2. **忘记 model.eval() 和 torch.no_grad()**
   - 现象：测试时准确率偏低
   - 原因：Dropout 和 BatchNorm 在测试时行为不同
   - 解决：测试前调用 model.eval()，推理时用 with torch.no_grad()

### 调参层面

1. **学习率过大或过小**
   - 现象：损失震荡不收敛（太大）或下降极慢（太小）
   - 解决：使用学习率调度器（如 ReduceLROnPlateau），或尝试 1e-3 作为起点

---

## 12. 学习总结

### 核心思想回顾

CNN 通过卷积核的局部连接和参数共享，高效地从图像中提取层次化特征。浅层卷积核检测简单的边缘和纹理，深层卷积核组合低级特征形成高级语义理解。池化操作在保留关键信息的同时降低计算量，增强了模型的平移不变性。

### 关键公式

1. 卷积运算：$Y[i,j] = \sum_{m,n} X[i+m, j+n] \cdot K[m,n]$
2. 输出尺寸：$H_{out} = \lfloor (H_{in} + 2p - k) / s \rfloor + 1$
3. 参数量：$\text{Params} = C_{out} \times C_{in} \times k_h \times k_w + C_{out}$

### 与相关算法的联系

- **全连接网络**：CNN 是全连接网络的特化版本，用局部连接和参数共享替代全连接
- **RNN**：CNN 处理空间维度，RNN 处理时间维度，各有侧重
- **Transformer/ViT**：用自注意力替代卷积，弱归纳偏置换取更强的全局建模能力

### 后续学习方向

- 经典 CNN 架构：VGG、ResNet、Inception、EfficientNet
- 目标检测：YOLO、SSD、Faster R-CNN
- 语义分割：U-Net、DeepLab
- 轻量化模型：MobileNet、ShuffleNet

---

## 13. 练习题与思考题

### 基础题

**题1：** 一个 $32 \times 32$ 的单通道图像，经过一个 $5 \times 5$ 卷积核（stride=1, padding=0）后，输出特征图的大小是多少？

**参考答案：**
使用输出尺寸公式 $H_{out} = \lfloor (H_{in} + 2p - k) / s \rfloor + 1$：
- $H_{in} = 32, p = 0, k = 5, s = 1$
- $H_{out} = \lfloor (32 + 0 - 5) / 1 \rfloor + 1 = 28$
输出特征图大小为 $28 \times 28$。

**题2：** 一个卷积层有 64 个 $3 \times 3$ 卷积核，输入是 3 通道 RGB 图像，该层共有多少个参数（含偏置）？

**参考答案：**
- 每个卷积核参数量：$3 \times 3 \times 3 = 27$（3 输入通道 × 3 × 3 卷积核大小）
- 共 64 个输出通道：$64 \times 27 = 1728$
- 偏置：64 个
- 总参数量：$1728 + 64 = 1792$

### 进阶题

**题3：** 为什么 CNN 中的卷积核通常使用奇数大小（如 $3 \times 3$, $5 \times 5$）？

**参考答案：**
奇数大小的卷积核有两个主要优势：
1. **对称填充**：奇数核可以实现对称的 Same 填充。例如 $3 \times 3$ 核填充 $p=1$ 可以保持尺寸不变，而 $2 \times 2$ 核无法做到（$p=1$ 时输出尺寸增加）。
2. **中心定位**：奇数核有明确的中心点，使得卷积操作有自然的锚点位置，便于定位特征。

### 开放思考题

**题4：** 在 Transformer 逐步取代 CNN 的趋势下，你认为 CNN 还有哪些独特优势？在什么场景下 CNN 仍然是更好的选择？

**参考答案思路：**
CNN 的归纳偏置（局部性、平移等变性）在小数据场景下仍是巨大优势——ViT 需要大量数据才能超越 CNN。此外，CNN 推理速度快、部署友好，在边缘设备、实时应用（如自动驾驶）中仍不可替代。混合架构（如 ConvNeXt、CoAtNet）结合了两者优点，可能是未来方向。

---

## 14. 学习路径建议

### 前置算法
- 线性回归、逻辑回归（理解基本优化过程）
- 多层感知机（理解神经网络基础）
- 梯度下降算法（理解反向传播）

### 平行算法
- RNN/LSTM（处理序列数据的另一种深度学习架构）
- 残差网络 ResNet（CNN 的重要改进）

### 进阶算法
- ResNet（残差连接解决深层网络退化问题）
- 目标检测：YOLO、Faster R-CNN
- 语义分割：U-Net、DeepLab
- Vision Transformer（用注意力机制替代卷积）

### 推荐资源
1. **教材**：《深度学习》（花书）第9章——卷积网络的系统理论
2. **论文**：He et al., "Deep Residual Learning for Image Recognition" (ResNet)
3. **课程**：Stanford CS231n（计算机视觉课程，CNN 部分讲解极为清晰）
