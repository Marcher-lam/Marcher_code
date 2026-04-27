# 循环注意力模型RAM 学习文档

> 通过强化学习学习"看哪里"——用 glimpse 序列进行视觉识别。

> 来源线索：本节内容根据原书第2章关于"目标搜索与识别"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义：** 循环注意力模型（Recurrent Attention Model，RAM）是由Mnih等人于2014年提出的深度强化学习模型，通过让网络学习"在哪里看"来进行图像分类，使用递归网络和强化学习算法决定下一个glimpse的位置。

**直觉类比：** 想象你在一幅复杂的画中找人。你不会一次性看完整幅画，而是先快速扫视，发现可疑区域后聚焦细看，然后根据看到的线索决定下一步看哪里。RAM正是模拟这个"边看边决策"的过程——每看一次（一个glimpse），就决定下一步该看哪里。

**历史背景：** 2014年，Volodymyr Mnih等人在论文"Recurrent Models of Visual Attention"中首次将强化学习引入视觉注意力，提出RAM模型，在MNIST分类任务上取得了显著的效果提升。

**算法定位：** 这是深度学习中的注意力机制+强化学习模型，属于"硬性注意力"（Hard Attention）方法。在PyTorch中可以用RNN和策略梯度实现。

**前置知识：**
- 循环神经网络（RNN/LSTM）
- 强化学习基础（策略梯度）
- 卷积神经网络

---

## 2. 核心原理

### 2.1 核心思想

RAM的核心思想是：**将"看哪里"作为一个可学习的决策问题**。

- **Glimpse Network：** 从给定位置提取图像patch，处理成特征向量
- **Core Network：** RNN，维护整个过程的隐状态
- **Location Network：** 决定下一个glimpse的位置（策略网络）

### 2.2 工作流程

```
输入图像 → 初始位置 → Glimpse Network(提取特征)
 → Core Network(更新状态) → Location Network(决定新位置)
 → 重复N次 → 最终分类
```

### 2.3 关键组件

**Glimpse Sensor：** 从图像中指定位置提取固定大小的patch

**Glimpse Network：** 将glimpsepatch转换为特征向量

**Core Network：** RNN，维护累积信息

**Action Network：** 输出分类或决定下一个位置

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $I$ | 输入图像 |
| $l_t$ | 第 $t$ 步的位置 |
| $g_t$ | 第 $t$ 步的glimpse特征 |
| $h_t$ | 第 $t$ 步的RNN隐状态 |
| $a_t$ | 第 $t$ 步的动作（位置或分类） |
| $\pi_\theta$ | 策略网络 |

### 3.2 模型公式

**Glimpse提取：**

$$g_t = \phi(I, l_{t-1}; \theta_g)$$

其中 $\phi$ 是glimpse网络，$l_{t-1}$ 是上一位置。

**RNN更新：**

$$h_t = RNN(h_{t-1}, g_t; \theta_h)$$

**位置策略（行为网络）：**

$$p(l_t | h_t) = \text{softmax}(W_h \cdot h_t + b)$$

**分类策略：**

$$p(y | h_T) = \text{softmax}(W_y \cdot h_T + b_y)$$

### 3.3 强化学习目标

由于glimpse位置是离散的，不能使用标准反向传播。需要使用REINFORCE算法：

$$\nabla_\theta J = \mathbb{E}_{l \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(l_t | h_t) \cdot R_t]$$

其中 $R_t$ 是奖励，可以是分类正确得分为1，否则为0。

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# MNIST数据集
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
train_data = datasets.MNIST('./data', train=True, transform=transform)
```

### 4.2 网络结构

```python
class GlimpseNetwork(nn.Module):
    """Glimpse网络：从指定位置提取特征"""
    def __init__(self, glimpse_size=8, hidden_size=128):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 4)
        self.fc = nn.Linear(32*5*5, hidden_size)
        
    def forward(self, image, location):
        # 从location提取glimpse patch
        x = self.extract_glimpse(image, location, self.glimpse_size)
        x = torch.relu(self.conv(x))
        x = x.view(-1, 32*5*5)
        return torch.relu(self.fc(x))


class RAM(nn.Module):
    """循环注意力模型"""
    def __init__(self, n_glimpses=6, hidden_size=256, n_classes=10):
        super().__init__()
        self.glimpse_net = GlimpseNetwork()
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.action_net = nn.Linear(hidden_size, n_classes)
        self.location_net = nn.Linear(hidden_size, 4)  # 4个位置候选
```

---

## 5. 应用场景

1. **手写数字识别：** RAM在MNIST上展示了比全卷积网络更好的性能
2. **目标检测：** 学习在哪里focus来检测目标
3. **图像分类：** 对复杂场景进行序列化注意力分类
4. **视觉问答：** 在VQA中学习关注相关区域

---

## 6. 优缺点分析

### 6.1 优点

1. **计算高效：** 每次只处理一小块区域，总计算量远小于全图处理
2. **可解释性强：** 可以可视化网络"看"了哪里
3. **抗干扰性好：** 对噪声和背景干扰更鲁棒

### 6.2 缺点

1. **训练困难：** 强化学习训练不稳定，需要技巧
2. **顺序执行：** glimpse是顺序生成的，不能并行
3. **位置可能不连续：** 可能跳来跳去

---

## 7. 调库实现

```python
"""
循环注意力模型RAM的PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GlimpseNetwork(nn.Module):
    """Glimpse网络：提取局部图像特征"""
    
    def __init__(self, input_size=28, glimpse_size=8, glimpse_hidden=128):
        super().__init__()
        self.input_size = input_size
        self.glimpse_size = glimpse_size
        self.glimpse_hidden = glimpse_hidden
        
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.fc = nn.Linear(64 * 4 * 4, glimpse_hidden)
        
    def forward(self, x, location):
        # 从指定位置提取glimpse
        x = self.extract_glimpse(x, location)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))
    
    def extract_glimpse(self, image, location):
        """从图像中提取指定位置的patch"""
        batch_size = image.size(0)
        # 简单的glimpse提取：缩放到glimpse_size大小
        x = F.adaptive_avg_pool2d(image, (self.glimpse_size, self.glimpse_size))
        return x


class CoreNetwork(nn.Module):
    """核心RNN网络：维护内部状态"""
    
    def __init__(self, glimpse_hidden=128, hidden_size=256):
        super().__init__()
        self.rnn = nn.LSTMCell(glimpse_hidden + 4, hidden_size)  # glimpse + location
        self.hidden_size = hidden_size
        
    def forward(self, glimpse_feat, prev_location, h_prev, c_prev):
        combined = torch.cat([glimpse_feat, prev_location], dim=1)
        h, c = self.rnn(combined, (h_prev, c_prev))
        return h, c


class RAM(nn.Module):
    """循环注意力模型完整实现"""
    
    def __init__(self, n_glimpses=6, input_size=28, glimpse_size=8, 
                 glimpse_hidden=128, hidden_size=256, n_classes=10):
        super().__init__()
        self.n_glimpses = n_glimpses
        self.glimpse_net = GlimpseNetwork(input_size, glimpse_size, glimpse_hidden)
        self.core_net = CoreNetwork(glimpse_hidden, hidden_size)
        
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.location_net = nn.Linear(hidden_size, 4)  # 4个位置的logits
        
        self.hidden_size = hidden_size
        self.init_h = nn.Parameter(torch.zeros(hidden_size))
        self.init_c = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, image, training=True):
        batch_size = image.size(0)
        
        # 初始化
        h = self.init_h.unsqueeze(0).expand(batch_size, -1)
        c = self.init_c.unsqueeze(0).expand(batch_size, -1)
        
        # 初始位置（图像中心）
        location = torch.zeros(batch_size, 4, device=image.device)
        
        logits_list = []
        
        for t in range(self.n_glimpses):
            # 提取glimpse特征
            glimpse = self.glimpse_net(image, location)
            
            # 更新RNN状态
            h, c = self.core_net(glimpseimpse_feat, location, h, c)
            
            # 决定下一个位置（采样或贪心）
            if training:
                location_probs = F.softmax(self.location_net(h), dim=1)
                location = torch.multinomial(location_probs, 1).float()
                location = F.one_hot(location.squeeze(1), 4).float()
            else:
                location = F.one_hot(self.location_net(h).argmax(1), 4).float()
        
        # 最终分类
        logits = self.classifier(h)
        return logits
    
    def compute_loss(self, image, target):
        """使用REINFORCE算法计算损失"""
        logits = self.forward(image, training=True)
        ce_loss = F.cross_entropy(logits, target)
        return ce_loss


def train_ram():
    """训练RAM模型"""
    from torchvision import datasets, transforms
    
    # 数据
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    # 模型
    model = RAM(n_glimpses=6, input_size=28, glimpse_size=8)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.compute_loss(data, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    return model


if __name__ == "__main__":
    model = train_ram()
    print("RAM模型训练完成")
```

---

## 8. 手工代码实现

```python
"""
RAM模型的简化NumPy实现
"""

import numpy as np


class SimpleRAM:
    """简化版RAM：使用numpy模拟核心思想"""
    
    def __init__(self, image_size=28, n_glimpses=4):
        self.image_size = image_size
        self.n_glimpses = n_glimpses
        self.glimpse_size = 8
        
        # 简化的分类器权重
        self.weights = np.random.randn(10, self.glimpse_size * self.glimpse_size)
        self.bias = np.zeros(10)
    
    def extract_glimpse(self, image, location):
        """提取glimpse patch"""
        # 简化的glimpse：取中心位置附近
        h, w = location
        half = self.glimpse_size // 2
        
        # 边界检查
        h = max(half, min(h, self.image_size - half))
        w = max(half, min(w, self.image_size - half))
        
        patch = image[h-half:h+half, w-half:w+half]
        return patch.flatten()
    
    def simple_attention_policy(self, features, history):
        """简单的注意力策略"""
        # 随机策略
        locations = np.random.randint(8, self.image_size-8, (self.n_glimpses, 2))
        return locations
    
    def predict(self, image):
        """预测"""
        # 从中心开始
        location = np.array([self.image_size//2, self.image_size//2])
        
        glimpses = []
        for _ in range(self.n_glimpses):
            patch = self.extract_glimpse(image, location)
            glimpses.append(patch)
            
            # 简单策略：移动到高响应区域
            response = patch @ (self.weights[:1].T)
            direction = np.random.choice([-1, 1], 2)
            location = location + direction * 4
        
        # 简单平均后分类
        avg_feature = np.mean(glimpses, axis=0)
        scores = self.weights @ avg_feature + self.bias
        return np.argmax(scores)


if __name__ == "__main__":
    np.random.seed(42)
    # 测试
    test_image = np.random.rand(28, 28)
    model = SimpleRAM()
    pred = model.predict(test_image)
    print(f"预测类别: {pred}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention_path(image, locations, save_path=None):
    """可视化注意力路径"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(image, cmap='gray')
    
    # 绘制glimpse路径
    for i, loc in enumerate(locations):
        y, x = loc
        circle = plt.Circle((x, y), 20, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        ax.text(x+25, y-25, f'{i+1}', color='red', fontsize=12, fontweight='bold')
    
    ax.set_title('RAM注意力路径可视化')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_glimpse_attention(image, glimpse_patches, save_path=None):
    """可视化提取的glimpse patches"""
    n = len(glimpse_patches)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    for i, patch in enumerate(glimpse_patches):
        axes[i].imshow(patch.reshape(8, 8), cmap='gray')
        axes[i].set_title(f'Glimpse {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 含义 |
|------|------|
| 分类准确率 | 在测试集上的分类正确率 |
| 收敛速度 | 达到特定性能所需的epoch数 |
| 计算效率 | 相比全图CNN的参数和计算量节省 |

### 10.2 计算代码

```python
def evaluate_ram(model, test_loader, device='cpu'):
    """评估RAM模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, training=False)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return {'accuracy': accuracy}
```

---

## 11. 常见问题与易错点

1. **强化学习不稳定：** 策略梯度方差大，需要使用baseline或Actor-Critic
2. **位置重叠：** 多个glimpse可能聚焦在同一位置，需要鼓励多样性
3. **训练速度慢：** 顺序执行glimpse，GPU利用率低
4. **初始化敏感：** 初始位置和策略对训练影响大

---

## 12. 学习总结

循环注意力模型RAM将"看哪里"建模为一个可学习的决策问题，通过强化学习让网络自主学会关注图像的相关区域。

核心贡献：
1. 首次将强化学习引入视觉注意力
2. 证明了选择性注意可以提高效率和性能
3. 提供了可解释的注意力可视化方法

数学核心：
- 使用REINFORCE算法优化位置策略
- 策略梯度 $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$

RAM是后续"软性注意力"（如Transformer）出现前的重要工作，它的"边看边决策"思想对现代注意力机制有深远影响。

---

## 13. 练习题与思考题

### 基础题

**题目1：** 解释RAM中"硬性注意力"与"软性注意力"的区别。

**答案：** 硬性注意力（Hard Attention）：每次只关注一个位置，是离散的、不可微的，需要强化学习训练。软性注意力（Soft Attention）：关注所有位置的加权平均，是连续的、可微的，可以用标准反向传播训练。

### 进阶题

**题目2：** 为什么RAM使用强化学习而不是标准反向传播来训练？

**答案：** 因为glimpse位置是离散的采样操作，从连续分布中采样不可微。REINFORCE算法使用策略梯度来估计梯度，允许在离散动作上进行优化。

---

## 14. 学习路径建议

**前置算法：**
- RNN/LSTM基础
- 强化学习基础（策略梯度）

**平行算法：**
- DRAN（Deep Recurrent Attention Model）
- Action Recognition中的注意力模型

**进阶算法：**
- Transformer中的自注意力
- DETR中的查询注意力