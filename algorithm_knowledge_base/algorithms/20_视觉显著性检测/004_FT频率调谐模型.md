# FT频率调谐模型 学习文档

> 用带通滤波实现边缘清晰的显著物体检测。
> 来源线索：原书第2.2.2节"FT：频率调谐显著性区域检测"。

---

## 1. 算法基础认知

**一句话定义：** FT（Frequency-tuned Salient Region Detection）由Achanta等人于2009年CVPR提出，通过带通滤波（Band-pass Filtering）在频率域中保留显著物体的低频轮廓信息和边缘高频信息，实现全分辨率、边缘清晰的显著物体检测。

**核心思想：** 显著性 = 图像LAB均值与高斯模糊后的LAB图像之间的欧氏距离。高斯模糊相当于低通滤波器，原图减去低通结果等价于带通滤波，保留中间频率成分。

**关键贡献：**
- 首个强调"全分辨率输出"的显著性方法
- 简洁高效，仅需几行代码即可实现
- 在像素级别精确保留显著物体边界

**频率域视角：** 自然图像中，背景通常包含大量低频（平坦区域）和高频成分（纹理噪声），而显著物体通常占据中频范围。FT通过DoG近似带通滤波器，提取中频信息。

---

## 2. 核心原理

### 2.1 频率滤波思想

在频率域中，图像可以分解为不同频率成分：
- **低频**：图像整体亮度变化、大范围平滑区域（天空、墙壁）
- **中频**：显著物体的轮廓和内部结构（人、车、动物）
- **高频**：纹理细节和噪声（草地、沙粒）

FT的目标是保留中频、抑制低频和高频，即带通滤波。

### 2.2 DoG（高斯差分）近似带通滤波

DoG（Difference of Gaussian）定义为两个不同尺度高斯滤波的差：
$$
\text{DoG}(x, y) = G_{\sigma_1}(x, y) - G_{\sigma_2}(x, y)
$$

其中 $G_{\sigma}(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$。

DoG在频率域中近似于带通滤波器：
- $\sigma_1$ 控制低频截止频率（较小 $\sigma$ 保留更多高频）
- $\sigma_2$ 控制高频截止频率（较大 $\sigma$ 滤除更多高频）

### 2.3 FT的简化公式

FT使用一个宽带带通滤波器，由两个高斯滤波器的差近似。但实际实现中，FT采用更巧妙的等价形式：

$$
S(x, y) = \| I_\mu - I_{\text{Gaussian}}(x, y) \|
$$

其中：
- $I_\mu$ 是图像在LAB空间的平均颜色向量（标量或三维向量）
- $I_{\text{Gaussian}}(x, y)$ 是经过高斯滤波（$\sigma=5$）后的LAB图像

**为什么这样有效？** $I_\mu$ 包含了整幅图像所有频率的信息（特别是低频成分），而 $I_{\text{Gaussian}}$ 只保留低频。两者之差等价于从原图中去除了低频成分，即保留了中高频。

---

## 3. 数学公式与推导

### 3.1 LAB色彩空间转换

FT在LAB空间计算，因为LAB的欧氏距离更符合人眼感知差异。

RGB到LAB的转换（简化版本）：
1. RGB $\to$ XYZ（线性变换）
2. XYZ $\to$ LAB（非线性变换）

$$
\begin{aligned}
X &= 0.412453R + 0.357580G + 0.180423B \\
Y &= 0.212671R + 0.715160G + 0.072169B \\
Z &= 0.019334R + 0.119193G + 0.950227B
\end{aligned}
$$

$$
L^* = 116 f(Y/Y_n) - 16, \quad a^* = 500[f(X/X_n) - f(Y/Y_n)], \quad b^* = 200[f(Y/Y_n) - f(Z/Z_n)]
$$

其中 $f(t) = t^{1/3}$ if $t > 0.008856$，否则 $f(t) = 7.787t + 16/116$。

### 3.2 显著性计算公式

对于像素 $(x, y)$，其显著性值为：

$$
S(x, y) = \| \mathbf{I}_\mu - \mathbf{I}_{\text{Gaussian}}(x, y) \|_2
$$

展开为LAB三个通道：

$$
S(x, y) = \sqrt{(L_\mu - L_{\text{Gaussian}}(x,y))^2 + (a_\mu - a_{\text{Gaussian}}(x,y))^2 + (b_\mu - b_{\text{Gaussian}}(x,y))^2}
$$

其中：
- $L_\mu, a_\mu, b_\mu$ 是整幅图像LAB三个通道的均值
- $L_{\text{Gaussian}}, a_{\text{Gaussian}}, b_{\text{Gaussian}}$ 是高斯模糊后的LAB图像

### 3.3 与DoG的关系

FT的公式可以重新解释为DoG滤波器的输出：

如果定义 $\mathbf{I}_{\text{low}}(x,y) = \mathbf{I}_{\text{Gaussian}}(x,y)$ 为低频成分，
而 $\mathbf{I}_\mu$ 是全局平均（即用一个极大尺度的高斯滤波逼近），则：
$$
S(x,y) = \| \mathbf{I}_{\text{very\_low}} - \mathbf{I}_{\text{low}} \| \approx \| \text{DoG}_{\sigma_1, \sigma_2} * \mathbf{I} \|
$$

其中 $\sigma_1 \gg \sigma_2$。这就是"频率调谐"名称的来源。

### 3.4 后处理

对原始显著性图进行归一化：
$$
S_{\text{norm}}(x,y) = \frac{S(x,y) - S_{\min}}{S_{\max} - S_{\min}}
$$

增加高斯平滑去除残差噪声：
$$
S_{\text{final}} = G_{\sigma_{\text{smooth}}} * S_{\text{norm}}
$$

---

## 4. 训练过程讲解

FT是**无参数方法**，不需要训练！它的所有参数（高斯滤波的 $\sigma$、是否使用LAB空间等）都是固定的。

### 4.1 算法步骤

```
输入：RGB图像 I
输出：显著性图 S

1. 将 I 从RGB转换到LAB色彩空间 -> I_lab
2. 计算 I_lab 在整幅图像上的均值 -> I_mean (3个通道的标量)
3. 对 I_lab 的每个通道应用高斯滤波（sigma=5）-> I_blur
4. 计算每个像素的颜色差异向量：
   diff(x,y) = I_mean - I_blur(x,y)
5. 计算L2范数作为显著性值：
   S(x,y) = sqrt(sum(diff(x,y)^2))
6. 归一化 S 到 [0, 1] 范围
7. 可选：高斯平滑后处理
8. 返回 S
```

### 4.2 时间复杂度分析
- LAB转换：$O(HW)$
- 均值计算：$O(HW)$
- 高斯滤波：$O(HW \log \sigma)$（使用快速傅里叶变换或递归滤波）
- 差值计算：$O(HW)$
- 总复杂度：$O(HW)$，线性于像素数

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 图像自动裁剪 | 根据FT显著图自动裁剪出显著区域 |
| 内容感知图像缩放 | 显著区域保持比例，背景拉伸 |
| 图像分割辅助 | 显著图作为分割的初始化或先验 |
| 图像质量评估 | 显著区域的失真权重更高 |
| 视觉跟踪初始化 | 用FT定位初始跟踪目标 |
| 移动端实时处理 | FT计算极快，适合移动设备 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 计算极快，O(N)复杂度 | 仅利用颜色对比度，缺乏语义信息 |
| 全分辨率输出，边缘清晰 | 对纹理丰富的背景容易产生误检 |
| 无需训练数据，即拿即用 | 无法区分多个显著物体 |
| 参数少，仅需设置高斯滤波sigma | 对尺度变化敏感（sigma固定） |
| 理论简洁，易于理解和实现 | 遇复杂场景（光照变化、遮挡）效果退化严重 |

**与现代方法的对比：** FT的AUC通常在0.7-0.8之间，而深度学习方法（如BASNet）可达0.95+。但FT的速度优势使其在实时应用和资源受限场景中仍有价值。

---

## 7. 调库实现（scikit-image + PyTorch）

```python
"""
FT频率调谐模型的完整实现
支持RGB和灰度图像，含LAB色彩空间转换
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import color, io, img_as_float
import torch
import torch.nn.functional as F


class FT:
    """Frequency-tuned Salient Region Detection"""

    def __init__(self, sigma=5, use_lab=True):
        """
        Args:
            sigma: 高斯滤波标准差，控制带通范围
            use_lab: 是否使用LAB色彩空间
        """
        self.sigma = sigma
        self.use_lab = use_lab

    def compute_saliency(self, image):
        """计算FT显著性图
        Args:
            image: (H, W, 3) RGB图像, float32, [0,1]
        Returns:
            saliency: (H, W) 归一化显著性图
        """
        if image.max() > 1.0:
            image = image / 255.0

        # 转换到LAB色彩空间
        if self.use_lab and image.shape[-1] >= 3:
            lab = color.rgb2lab(image)
        else:
            # 灰度图或RGB直接使用
            lab = image if image.shape[-1] == 3 else np.stack([image]*3, axis=-1)

        # 计算整图均值
        if self.use_lab:
            # LAB: L[0,100], a[-128,127], b[-128,127]
            mean_color = lab.mean(axis=(0, 1))
        else:
            mean_color = lab.mean(axis=(0, 1))

        # 高斯模糊（相当于低通滤波）
        blurred = np.zeros_like(lab)
        for c in range(3):
            blurred[:, :, c] = gaussian_filter(lab[:, :, c], sigma=self.sigma)

        # 显著性 = 均值与模糊图像的LAB欧氏距离
        diff = lab - blurred
        saliency = np.sqrt((diff ** 2).sum(axis=2))

        # 归一化到[0,1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    def compute_saliency_torch(self, image_tensor):
        """PyTorch批处理实现
        Args:
            image_tensor: (B, 3, H, W) 归一化RGB张量
        Returns:
            saliency: (B, 1, H, W) 显著性图
        """
        B, C, H, W = image_tensor.shape
        device = image_tensor.device

        # 简化LAB转换（近似）
        # RGB -> LAB线性近似（用于GPU加速）
        r, g, b = image_tensor[:, 0:1], image_tensor[:, 1:2], image_tensor[:, 2:3]
        l = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 * (r - g) + 0.5
        b_ = 0.5 * (r + g - 2 * b) + 0.5
        lab = torch.cat([l * 100, a * 255, b_ * 255], dim=1)

        # 全图均值
        mean_color = lab.mean(dim=(2, 3), keepdim=True)

        # 高斯模糊（使用平均池化近似）
        k = self.sigma * 2 + 1
        blurred = F.avg_pool2d(lab, kernel_size=k, stride=1,
                               padding=k // 2, count_include_pad=True)

        # 欧氏距离
        diff = mean_color - blurred
        saliency = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-8)

        # 归一化
        s_min = saliency.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        s_max = saliency.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)

        return saliency


def demo_ft():
    """演示FT模型"""
    np.random.seed(42)
    # 创建测试图像：背景灰 + 红色方块
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.3
    img[30:70, 30:70] = [0.9, 0.1, 0.1]  # 红色显著区域

    model = FT(sigma=5)
    saliency = model.compute_saliency(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Input', fontsize=12)
    axes[0].axis('off')
    im1 = axes[1].imshow(saliency, cmap='gray')
    axes[1].set_title('FT Saliency (Grayscale)', fontsize=12)
    axes[1].axis('off')
    im2 = axes[2].imshow(saliency, cmap='jet')
    axes[2].set_title('FT Saliency (Jet)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig('ft_demo.png', dpi=150)
    plt.show()
    print(f"FT显著性范围: [{saliency.min():.3f}, {saliency.max():.3f}]")
    print(f"显著区域均值(中心30-70): {saliency[30:70, 30:70].mean():.3f}")
    print(f"背景区域均值: {saliency[:30, :30].mean():.3f}")


if __name__ == '__main__':
    demo_ft()
```

---

## 8. 手工代码实现（NumPy）

```python
"""
FT纯NumPy手工实现
不依赖任何图像处理库（包括scikit-image）
"""
import numpy as np
from scipy.ndimage import gaussian_filter


class FTNumpy:
    """纯NumPy实现的FT核心算法"""

    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def _rgb_to_lab_simple(self, rgb):
        """简化的RGB到LAB转换（手写实现）"""
        rgb = np.clip(rgb, 0, 1)
        # 线性gamma校正
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # RGB -> XYZ
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y_ = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # XYZ -> LAB（简化版，直接计算以避免复杂非线性）
        # 这里使用近似公式，精确应使用CIE标准
        xn, yn, zn = 0.950456, 1.0, 1.088754
        x, y_, z = x / xn, y_ / yn, z / zn

        def f(t):
            return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

        fx, fy, fz = f(x), f(y_), f(z)
        l = 116 * fy - 16
        a = 500 * (fx - fy)
        b_ = 200 * (fy - fz)

        lab = np.stack([l, a, b_], axis=-1)
        return lab

    def compute_saliency(self, image):
        """手工实现FT显著性计算"""
        h, w = image.shape[:2]

        # 转换为LAB
        if image.shape[-1] >= 3:
            lab = self._rgb_to_lab_simple(image)
        else:
            gray = image if image.ndim == 2 else image[:, :, 0]
            lab = np.stack([gray * 100, np.zeros_like(gray), np.zeros_like(gray)], axis=-1)

        # 计算整图均值
        mean_l = lab[:, :, 0].mean()
        mean_a = lab[:, :, 1].mean()
        mean_b = lab[:, :, 2].mean()

        # 高斯滤波（手工实现每个通道）
        blurred = np.zeros_like(lab)
        for c in range(3):
            blurred[:, :, c] = gaussian_filter(lab[:, :, c], sigma=self.sigma)

        # 逐像素计算显著性
        saliency = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                dl = mean_l - blurred[i, j, 0]
                da = mean_a - blurred[i, j, 1]
                db = mean_b - blurred[i, j, 2]
                saliency[i, j] = np.sqrt(dl * dl + da * da + db * db)

        # 归一化
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)

        return saliency

    def compute_dog_alternative(self, image):
        """使用DoG的等价实现"""
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        # 两个高斯滤波的差 = DoG带通滤波
        g1 = gaussian_filter(gray, sigma=self.sigma * 0.5)
        g2 = gaussian_filter(gray, sigma=self.sigma * 2)
        dog = np.abs(g1 - g2)
        # 归一化
        dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
        return dog


def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(64, 64, 3).astype(np.float32)
    img[20:45, 20:45] = [0.8, 0.2, 0.2]

    model = FTNumpy(sigma=5)
    smap = model.compute_saliency(img)
    print(f"FT手工实现 — 显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")

    # 验证：显著区域应有更高的显著性值
    salient_region = smap[20:45, 20:45].mean()
    bg_region = smap[:20, :20].mean()
    print(f"显著区域均值: {salient_region:.3f}")
    print(f"背景区域均值: {bg_region:.3f}")
    print(f"对比度倍数: {salient_region / (bg_region + 1e-8):.2f}x")


if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
"""
FT频率调谐模型可视化
展示频域分析、滤波效果和显著性结果
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import color


def visualize_ft_frequency():
    """可视化FT的频率调谐原理"""
    np.random.seed(42)
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.2
    img[25:75, 25:75] = [0.7, 0.3, 0.3]

    # LAB转换
    lab = color.rgb2lab(img)
    mean_color = lab.mean(axis=(0, 1))

    # 不同sigma的高斯滤波
    sigmas = [1, 3, 5, 10, 20]
    fig, axes = plt.subplots(2, len(sigmas) + 1, figsize=(18, 7))

    # 原图
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('原图', fontsize=12)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for idx, sigma in enumerate(sigmas):
        blurred = np.zeros_like(lab)
        for c in range(3):
            blurred[:, :, c] = gaussian_filter(lab[:, :, c], sigma=sigma)

        # 差值
        diff = np.sqrt(((mean_color[None, None, :] - blurred) ** 2).sum(axis=2))
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        axes[0, idx + 1].imshow(blurred / 100, cmap='gray')
        axes[0, idx + 1].set_title(f'模糊 sigma={sigma}', fontsize=12)
        axes[0, idx + 1].axis('off')

        axes[1, idx + 1].imshow(diff, cmap='jet')
        axes[1, idx + 1].set_title(f'显著性 sigma={sigma}', fontsize=12)
        axes[1, idx + 1].axis('off')

    plt.suptitle('FT频率调谐：不同sigma对显著性检测的影响', fontsize=14)
    plt.tight_layout()
    plt.savefig('ft_sigma_comparison.png', dpi=150)
    plt.show()

    # 频域分析图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    gray = np.mean(img, axis=2)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    axes2[0].imshow(gray, cmap='gray')
    axes2[0].set_title('灰度图', fontsize=12)
    axes2[0].axis('off')

    axes2[1].imshow(magnitude, cmap='inferno')
    axes2[1].set_title('频谱幅度（对数）', fontsize=12)
    axes2[1].axis('off')

    # FT相当于提取中频
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    bp = np.zeros_like(magnitude)
    r1, r2 = 5, 25
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    mask = (x * x + y * y >= r1 * r1) & (x * x + y * y <= r2 * r2)
    bp[mask] = magnitude[mask]

    axes2[2].imshow(bp, cmap='inferno')
    axes2[2].set_title(f'带通滤波 (r={r1}~{r2})', fontsize=12)
    axes2[2].axis('off')

    plt.tight_layout()
    plt.savefig('ft_frequency_analysis.png', dpi=150)
    plt.show()
    print("FT频域分析图已保存")


if __name__ == '__main__':
    visualize_ft_frequency()
```

---

## 10. 模型评估

### 10.1 FT论文评估结果
FT论文在公开数据集上与ITTI、SR等方法对比，使用Precision-Recall曲线和F-measure评估：

| 方法 | F-measure | MAE | 速度(秒/图) |
|------|-----------|-----|------------|
| ITTI (1998) | 0.452 | 0.241 | ~1.0 |
| SR (2007) | 0.460 | 0.237 | ~0.2 |
| **FT (2009)** | **0.624** | **0.178** | **~0.1** |
| AC (2008) | 0.554 | 0.201 | ~0.3 |
| GBVS (2007) | 0.538 | 0.212 | ~2.0 |

### 10.2 评估指标详解
- **PR曲线**：通过二值化阈值从0到1遍历，计算Precision和Recall
- **自适应阈值**：$T = \frac{2}{HW}\sum_{x,y} S(x,y)$（显著图均值的2倍）
- **F-measure**：$F_{\beta^2} = \frac{(1+\beta^2)PR}{\beta^2 P + R}$，FT论文使用 $\beta^2 = 0.3$

### 10.3 评估代码
```python
def ft_evaluate(saliency, gt_mask):
    """FT模型评估"""
    # 自适应阈值
    T = 2 * saliency.mean()
    binary = (saliency > T).astype(np.int32)

    tp = np.sum((binary == 1) & (gt_mask > 0.5))
    fp = np.sum((binary == 1) & (gt_mask <= 0.5))
    fn = np.sum((binary == 0) & (gt_mask > 0.5))

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f_beta = 1.3 * prec * rec / (0.3 * prec + rec + 1e-8)
    mae = np.mean(np.abs(saliency - gt_mask))

    return {'Precision': prec, 'Recall': rec, 'F': f_beta, 'MAE': mae}
```

---

## 11. 常见问题与易错点

### Q1: FT为什么使用LAB空间而非RGB？
**A:** LAB空间的欧氏距离更符合人眼感知差异。RGB空间中，颜色差异与人类感知不成线性关系。例如，RGB中(255,0,0)和(200,0,0)的欧氏距离与感知差异不匹配。

### Q2: sigma参数如何选择？
**A:** 原论文推荐 $\sigma=5$。如果sigma太小，滤波不足，显著性图包含过多纹理噪声；sigma太大，滤波过度，显著物体边缘模糊。

### Q3: FT和SR（谱残差）的区别？
**A:** SR在频率域操作（计算log频谱减去平滑后的频谱），FT在空间域操作（均值减高斯模糊）。两者都是无监督方法，但FT直接在空间域计算，更直观且保留边缘信息。

### Q4: FT为什么能保持边缘清晰？
**A:** 高斯模糊虽然平滑了区域内部，但边缘处仍有明显颜色跳变。均值减去模糊后，边缘处的差异仍然很大，因此边缘得以保留。

### Q5: FT对光照变化敏感吗？
**A:** 是的。光照变化会导致LAB空间的L通道（亮度）发生变化，影响显著性值。预处理中的光照归一化可以缓解此问题。

---

## 12. 学习总结

### 12.1 核心要点
- **FT = 均值 - 模糊**：最简单的显著性检测公式之一
- **频率域解释**：带通滤波 = 保留中频、抑制低高频
- **全分辨率输出**：输出与原图等大的显著性图
- **无监督无参数**：无需训练，即拿即用

### 12.2 与其他方法的对比
- **ITTI**：生物启发式，多尺度金字塔，输出小尺寸
- **SR**：频域谱残差，速度快但边缘模糊
- **AC**：多尺度局部对比度，计算量大
- **FT**：全分辨率、边缘清晰、速度适中

### 12.3 实用性评估
FT在实际使用中的效果很大程度上取决于场景复杂度：
- 简单背景 + 单个显著物体 → 效果良好
- 复杂背景 + 多个物体 → 效果显著下降

---

## 13. 练习题与思考题（含答案）

### 练习1：手动计算FT
**题目：** 给定一个2x2的灰度图像 $I = [[100, 50], [50, 100]]$，高斯模糊核为 $K = \frac{1}{9}[[1,1,1],[1,1,1],[1,1,1]]$（3x3均值滤波近似），计算FT显著性图。

**答案：**
1. 均值：$(100+50+50+100)/4 = 75$
2. 模糊结果（边界补零）：
   - $I_{blur}(0,0) = (100+50+50+100)/4 = 75$（简化，实际需要3x3窗口）
   - 近似每个像素的模糊结果 ≈ 62.5
3. 显著性：$S = |75 - 62.5| = 12.5$ 对每个像素
4. 实际各像素略有差异，但整体均匀

### 练习2：修改sigma的影响
**题目：** 解释为什么当 $\sigma \to 0$ 时，FT显著性图趋近于0。

**答案：** 当 $\sigma \to 0$ 时，高斯核趋近于脉冲函数（Dirac delta），$I_{\text{Gaussian}} \to I$。因此 $S(x,y) = \|I_\mu - I(x,y)\|$，这不是带通滤波而是高通滤波，保留的是每个像素与均值的差异——背景区域的纹理噪声也会被保留，显著性图会充满噪声。

### 练习3：代码填空
**题目：** 补全以下FT代码中的缺失部分。
```python
def ft_saliency(image):
    lab = color.rgb2lab(image)
    mean = lab.mean(axis=(0, 1), keepdims=True)
    blurred = np.zeros_like(lab)
    for c in range(3):
        # 缺失：对第c通道应用高斯滤波
        blurred[:,:,c] = _______
    diff = mean - blurred
    # 缺失：计算L2范数
    saliency = _______
    return saliency
```

**答案：**
```python
blurred[:,:,c] = gaussian_filter(lab[:,:,c], sigma=5)
saliency = np.sqrt((diff ** 2).sum(axis=2))
```

### 练习4：思考题
**题目：** 如果背景不是平滑的（例如草地纹理），FT的显著性图会怎样？如何改进？

**答案：** 草地纹理包含大量高频成分，FT的带通滤波器无法完全抑制这些高频，导致背景区域也产生较高的显著性值。改进方法：
1. 增大sigma值，增强低通滤波效果
2. 先进行纹理滤波（如LBP、Gabor滤波），去除纹理后再做FT
3. 结合超像素分割，以区域为单位计算显著性，降低纹理影响

---

## 14. 学习路径建议

### 前置知识
1. **数字图像处理基础**：傅里叶变换、滤波（低通/高通/带通）、色彩空间
2. **频率域分析**：频谱、频率成分与图像内容的关系
3. **视觉感知基础**：人眼对颜色和对比度的感知特性

### 后续学习
1. **频域显著性方法扩展**：SR（谱残差）、PFT（相位谱傅里叶变换）、HFT（超复数傅里叶变换）
2. **空间域显著性改进**：RC（基于直方图对比度）、HC（基于直方图对比度）
3. **基于学习的显著性检测**：从手工特征到深度特征，再到端到端网络
4. **多尺度融合方法**：MDF、HSD、SCHED等

### 推荐文献
1. Achanta R, et al. "Frequency-tuned salient region detection." CVPR 2009. (原始论文)
2. Achanta R, et al. "Salient region detection and segmentation." ICVS 2008. (前期工作AC方法)
3. Hou X, Zhang L. "Saliency detection: A spectral residual approach." CVPR 2007. (频域对比方法SR)
4. Borji A, Itti L. "State-of-the-art in visual attention modeling." TPAMI 2013. (综述)
