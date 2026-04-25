# U-Net 学习文档

## 1. 算法基础认知

U-Net 由 Ronneberger 等人于 2015 年提出，最初用于医学图像分割。其名称来源于网络结构呈 U 形：左侧是收缩路径（编码器），右侧是扩展路径（解码器），中间通过跳跃连接（Skip Connections）将编码器特征传递给解码器。U-Net 凭借其精巧的对称结构和跳跃连接设计，成为图像分割任务的标杆网络，也是扩散模型中噪声预测网络的标准架构。

## 2. 核心原理

U-Net 的核心设计有三个关键要素：

**收缩路径（Contracting Path）**：通过反复的卷积+下采样操作，逐步提取更高层次的语义特征。每一步包含两次 3×3 卷积（+ReLU）和一个 2×2 最大池化进行下采样。通道数在每个下采样步骤翻倍。

**扩展路径（Expanding Path）**：通过上采样（转置卷积或插值）逐步恢复空间分辨率。每一步包含上采样、与对应层跳跃连接拼接、两次 3×3 卷积。通道数在每个上采样步骤减半。

**跳跃连接（Skip Connections）**：将收缩路径中的特征图直接拼接到扩展路径的对应层。这使得解码器可以同时利用高层语义信息和低层细节信息，解决了上采样过程中空间细节丢失的问题。

## 3. 数学公式与推导

**卷积操作**：

$$y_{i,j} = \sum_{m}\sum_{n} w_{m,n} \cdot x_{i+m, j+n} + b$$

**最大池化**（2×2）：

$$y_{i,j} = \max_{0 \leq m,n < 2} x_{2i+m, 2j+n}$$

**跳跃连接拼接**：设编码器特征图为 $f_{enc} \in \mathbb{R}^{C_{enc} \times H \times W}$，上采样后特征图为 $f_{dec} \in \mathbb{R}^{C_{dec} \times H \times W}$，拼接操作为：

$$f_{cat} = \text{Concat}(f_{enc}, f_{dec}) \in \mathbb{R}^{(C_{enc} + C_{dec}) \times H \times W}$$

**分割损失（交叉熵）**：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$

**Dice Loss**（处理类别不平衡）：

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i \hat{y}_i y_i + \epsilon}{\sum_i \hat{y}_i + \sum_i y_i + \epsilon}$$

## 4. 训练过程讲解

1. 输入图像 $x$ 送入编码器，逐层下采样提取多尺度特征
2. 瓶颈层输出最深层的语义特征
3. 解码器逐层上采样，每层与编码器对应层通过跳跃连接拼接
4. 最终输出分割图 $\hat{y}$（每个像素的类别概率）
5. 计算损失（交叉熵 + Dice），反向传播更新参数

**数据增强**对 U-Net 尤为重要，因为医学图像数据集通常较小。常用增强：随机翻转、旋转、弹性变形等。

## 5. 应用场景

- **医学图像分割**：器官、肿瘤、细胞分割
- **扩散模型骨干网络**：DDPM、Stable Diffusion 的噪声预测器
- **遥感图像分割**：建筑、道路、农田提取
- **自动驾驶**：道路、车辆、行人分割
- **图像修复**：缺失区域补全

## 6. 优缺点分析

**优点：**
- 跳跃连接保留了空间细节，分割精度高
- 对小数据集友好（医学领域验证）
- 结构清晰，易于修改和扩展
- 多尺度特征融合能力强

**缺点：**
- 参数量较大（尤其是深层通道数翻倍设计）
- 对高分辨率图像显存消耗大
- 原始版本没有注意力机制，长距离依赖建模弱

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature))
            in_ch = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

model = UNet(in_ch=1, out_ch=1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

dummy_input = torch.randn(4, 1, 128, 128)
dummy_target = torch.randint(0, 2, (4, 1, 128, 128)).float()
for epoch in range(5):
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class Conv2DNumpy:
    def __init__(self, in_ch, out_ch, kernel_size=3):
        self.W = np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * np.sqrt(2.0 / (in_ch * kernel_size ** 2))
        self.b = np.zeros(out_ch)

    def forward(self, x, padding=1):
        self.x = x
        n, c_in, h, w = x.shape
        c_out, _, kh, kw = self.W.shape
        if padding > 0:
            x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
        h_out = h
        w_out = w
        out = np.zeros((n, c_out, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                patch = x[:, :, i:i+kh, j:j+kw]
                for co in range(c_out):
                    out[:, co, i, j] = np.sum(patch * self.W[co], axis=(1, 2, 3)) + self.b[co]
        return out

class MaxPool2DNumpy:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        n, c, h, w = x.shape
        h_out, w_out = h // self.size, w // self.size
        out = x.reshape(n, c, h_out, self.size, w_out, self.size).max(axis=(3, 5))
        return out

class UNetBlockNumpy:
    def __init__(self, in_ch, out_ch):
        self.conv1 = Conv2DNumpy(in_ch, out_ch)
        self.conv2 = Conv2DNumpy(out_ch, out_ch)

    def forward(self, x):
        h = self.conv1.forward(x)
        h = np.maximum(0, h)
        h = self.conv2.forward(h)
        h = np.maximum(0, h)
        return h

class UNetNumpy:
    def __init__(self, in_ch=1, out_ch=1):
        self.enc1 = UNetBlockNumpy(in_ch, 4)
        self.enc2 = UNetBlockNumpy(4, 8)
        self.pool = MaxPool2DNumpy(2)
        self.bottleneck = UNetBlockNumpy(8, 16)
        self.up_conv2 = Conv2DNumpy(16, 8, kernel_size=2)
        self.dec2 = UNetBlockNumpy(16, 8)
        self.up_conv1 = Conv2DNumpy(8, 4, kernel_size=2)
        self.dec1 = UNetBlockNumpy(8, 4)
        self.final_conv = Conv2DNumpy(4, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1.forward(x)
        e2 = self.enc2.forward(self.pool.forward(e1))
        b = self.bottleneck.forward(self.pool.forward(e2))
        d2 = self._upsample_and_cat(b, e2)
        d2 = self.dec2.forward(d2)
        d1 = self._upsample_and_cat(d2, e1)
        d1 = self.dec1.forward(d1)
        out = self.final_conv.forward(d1, padding=0)
        return out

    def _upsample_and_cat(self, x, skip):
        up = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
        if up.shape[2:] != skip.shape[2:]:
            up = up[:, :, :skip.shape[2], :skip.shape[3]]
        return np.concatenate([skip, up], axis=1)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 1, 128, 128)
    output = torch.sigmoid(model(test_input))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(test_input[0, 0], cmap='gray')
axes[0].set_title('输入')
axes[1].imshow(output[0, 0], cmap='gray')
axes[1].set_title('分割输出')
axes[2].imshow((output[0, 0] > 0.5).float(), cmap='gray')
axes[2].set_title('二值化分割')
for ax in axes:
    ax.axis('off')
plt.savefig('unet_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **IoU（Intersection over Union）**：分割区域与真实区域的重叠度，越高越好
- **Dice Coefficient**：$DSC = \frac{2|P \cap G|}{|P| + |G|}$，医学分割常用
- **像素准确率**：分类正确的像素占比
- **Hausdorff 距离**：衡量分割边界的最大偏差

## 11. 常见问题与易错点

- **尺寸不匹配**：下采样和上采样的尺寸可能因为奇数维度而不一致，需要裁剪或填充
- **通道拼接方向**：跳跃连接是通道维度拼接（dim=1），不是相加
- **输出激活**：二分类用 Sigmoid，多分类用 Softmax
- **类别不平衡**：前景区域远小于背景时，使用 Dice Loss 或加权交叉熵

## 12. 学习总结

U-Net 是编码器-解码器架构的经典之作。跳跃连接是其核心创新，解决了上采样过程中空间信息丢失的问题。U-Net 不仅是医学分割的标准工具，也是扩散模型中噪声预测网络的首选架构。理解 U-Net 对掌握图像分割和扩散模型都至关重要。

## 13. 练习题与思考题（含答案）

**Q1：为什么 U-Net 使用拼接（concatenation）而不是相加（addition）来做跳跃连接？**

A1：拼接保留了编码器的完整特征信息，解码器可以通过后续卷积层学习如何融合这些信息。相加要求特征通道数相同且语义对齐，信息融合方式更受限。拼接虽然增加计算量但信息保留更完整。

**Q2：U-Net 在扩散模型中起什么作用？**

A2：在扩散模型中，U-Net 用于预测每一步添加的噪声 $\epsilon_\theta(x_t, t)$。U-Net 的多尺度特征适合处理不同噪声级别下的图像，跳跃连接帮助保留高频细节。

**Q3：如何改进 U-Net 以处理更大分辨率的图像？**

A3：使用更深的编码器（如 ResNet backbone）、引入注意力机制（Attention U-Net）、使用空洞卷积（dilated convolution）扩大感受野、或采用层级式处理策略。

## 14. 学习路径建议

1. 掌握 CNN 基础（卷积、池化、上采样）
2. 理解编码器-解码器架构
3. 实现 U-Net 并在分割数据集上训练
4. 学习 Attention U-Net、U-Net++ 等改进版本
5. 了解 U-Net 在扩散模型中的应用（Stable Diffusion 的 UNet）
