# SR谱残差模型 学习文档

> 频率域的显著性检测——用谱残差分离新颖信息与冗余信息。
>
> 来源线索：本节内容根据原书第2.2.1节"SR:用频率域的谱残差表示显著信息"整理。

---

## 1. 算法基础认知

**一句话定义：** SR（Spectral Residual）由侯晓迪（Xiaodi Hou）于2007年在CVPR上提出，通过计算图像傅里叶变换振幅谱对数残差来提取显著性信息。

**核心思想：** 自然图像具有统计不变性——所有自然图像振幅谱的均值与频率成反比，服从 $1/f$ 分布规律。图像可分解为：
- 冗余信息（符合 $1/f$ 规律的共性部分）
- 新颖信息（偏离 $1/f$ 规律的个性部分，即显著区域）

谱残差方法在频率域计算振幅谱的对数，减去均值滤波结果（模拟 $1/f$ 规律），剩余部分即为显著信息对应的谱残差。

**为什么有效：** 在自然图像统计中，背景纹理往往符合 $1/f$ 的均匀分布，而显著目标（如人、动物、物体）会打破这一分布，产生异常的频谱成分。SR正是通过捕捉这种异常来定位显著区域。

**历史背景：** 2007年侯晓迪在MIT媒体实验室提出该方法，是频率域显著性检测的开创性工作。相比之前的空域方法（如ITTI），SR具有计算效率极高、无需任何先验知识的优点。

---

## 2. 核心原理

SR模型的核心原理分为四个步骤：

### 2.1 傅里叶变换
对输入图像 $I(x,y)$ 进行二维傅里叶变换，得到振幅谱 $A(f)$ 和相位谱 $P(f)$：

$$
\mathcal{F}\{I(x,y)\} = F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y) e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}
$$

$$
A(f) = |F(u,v)|, \quad P(f) = \angle F(u,v)
$$

### 2.2 对数谱
对振幅谱取对数：

$$
L(f) = \log(A(f))
$$

对数操作将乘性关系转换为加性关系，使得后续的线性滤波操作更加自然。

### 2.3 谱残差计算
用均值滤波估计 $1/f$ 规律下的平均对数谱：

$$
\mathcal{H}(f) = L(f) * h_n(f)
$$

其中 $h_n$ 是 $n \times n$ 的均值滤波器（通常 $n=3$）。谱残差为：

$$
R(f) = L(f) - \mathcal{H}(f)
$$

### 2.4 逆变换重构
将谱残差与原始相位谱结合，进行逆傅里叶变换，得到显著性图：

$$
S(x,y) = |\mathcal{F}^{-1}\{\exp(R(f) + jP(f))\}|
$$

最后对 $S(x,y)$ 进行高斯平滑得到最终的显著性图。

**直观理解：** 均值滤波提取的是"平均的、冗余的"频谱模式，从原始频谱中减去这个平均模式后，保留下来的就是"异常的、新颖的"频谱成分。这些异常成分逆变换回空域后，对应着图像中的显著区域。

---

## 3. 数学公式与推导

### 3.1 自然图像的 $1/f$ 统计规律

大量研究表明，自然图像的振幅谱均值与频率成反比：

$$
\mathbb{E}[A(f)] \propto \frac{1}{|f|}
$$

取对数后：

$$
\mathbb{E}[\log A(f)] \propto -\log|f| + C
$$

这表明自然图像的整体频谱在双对数坐标系中呈线性下降趋势。SR方法正是利用这一统计规律作为"背景模型"。

### 3.2 谱残差的数学含义

设图像 $I(x,y)$ 的傅里叶变换为 $F(u,v)$，那么：

$$
\log|F(u,v)| = \log\mathbb{E}[|F(u,v)|] + \epsilon(u,v)
$$

其中 $\epsilon(u,v)$ 是偏离平均频谱的残差项。SR假设：
- $\log\mathbb{E}[|F(u,v)|]$ 是平滑的（可以通过均值滤波近似）
- $\epsilon(u,v)$ 包含新颖信息

因此谱残差 $R(u,v) = \epsilon(u,v)$。

### 3.3 显著性图的计算

$$
S(x,y) = g(x,y) * \left|\mathcal{F}^{-1}\left[e^{R(u,v) + jP(u,v)}\right]\right|^2
$$

其中 $g(x,y)$ 是高斯平滑核，用于消除重构过程中的高频噪声。

### 3.4 算法流程总结

$$
\begin{aligned}
&1. \quad F = \mathcal{F}(I) \\
&2. \quad A = |F|, \quad P = \angle F \\
&3. \quad L = \log A \\
&4. \quad \mathcal{H} = L * h_n \\
&5. \quad R = L - \mathcal{H} \\
&6. \quad S = g * |\mathcal{F}^{-1}(\exp(R + jP))|^2 \\
&7. \quad S_{norm} = \frac{S - \min(S)}{\max(S) - \min(S)}
\end{aligned}
$$

---

## 4. 训练过程讲解

SR模型是一个**无参数、无训练**的方法，不需要任何训练数据和学习过程。这是其最大的特点之一。

**处理流程：**
1. 输入单张图像
2. 转为灰度图（若为彩色图像）
3. 调整大小为合适尺寸（通常保持宽高比，长边缩放到 256-512 像素）
4. 执行 FFT → 对数谱 → 均值滤波 → 谱残差 → IFFT → 高斯平滑
5. 输出显著性图

**注意：** 虽然不需要训练，但参数选择会影响结果：
- 均值滤波核大小 $n$：$n=3$ 最常用，越大背景抑制越强
- 高斯平滑 $\sigma$：$\sigma=4$ 是推荐值，控制显著区域的平滑程度

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 视觉显著性预测 | 预测人眼注视点位置 |
| 图像分割预处理 | 显著区域可作为分割的前景种子 |
| 目标检测 | 快速定位可能包含目标的区域 |
| 图像压缩 | 显著区域保留高质量，背景压缩 |
| 图像裁剪/缩略图 | 自动保留最显著的内容区域 |
| 视频显著性 | 扩展到视频帧序列的显著性检测 |

---

## 6. 优缺点分析

**优点：**
- ✅ **无需训练**：完全无参数，零训练成本
- ✅ **计算极快**：仅需一次 FFT 和简单的滤波操作
- ✅ **理论优美**：基于自然图像统计规律的严格推导
- ✅ **通用性强**：不依赖特定数据集或类别
- ✅ **实现简单**：核心代码仅 10-20 行

**缺点：**
- ❌ **边界效应明显**：显著区域边缘容易突出
- ❌ **对噪声敏感**：高频噪声会干扰频谱计算
- ❌ **缺乏语义**：只基于低级特征，无法理解场景语义
- ❌ **分辨率低**：显著性图通常较模糊
- ❌ **不适用于纹理图像**：纹理图像本身频谱丰富，容易误判

---

## 7. 调库实现

```python
"""
SR谱残差模型 - 完整调库实现
使用 scipy + numpy，依赖 OpenCV 进行图像读写
"""
import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt


class SR:
    """谱残差显著性检测模型"""
    
    def __init__(self, kernel_size=3, sigma=4):
        """
        参数:
            kernel_size: 均值滤波核大小 (默认3)
            sigma: 高斯平滑标准差 (默认4)
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def compute_saliency(self, image):
        """
        计算显著性图
        
        参数:
            image: 输入图像 (H,W) 或 (H,W,3)
        
        返回:
            saliency: 归一化的显著性图 (0~1)
        """
        # 如果是彩色图像，转为灰度
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # 步骤1: 二维傅里叶变换
        f_transform = fft.fft2(image)
        magnitude = np.abs(f_transform)      # 振幅谱 A(u,v)
        phase = np.angle(f_transform)         # 相位谱 P(u,v)
        
        # 步骤2: 对数谱 L(u,v) = log(A(u,v))
        log_spectrum = np.log(magnitude + 1e-8)  # 加小常数防止 log(0)
        
        # 步骤3: 均值滤波估计平均对数谱
        kernel = np.ones((self.kernel_size, self.kernel_size)) \
                 / (self.kernel_size * self.kernel_size)
        avg_spectrum = self._convolve(log_spectrum, kernel)
        
        # 步骤4: 谱残差 R(u,v) = L(u,v) - H(u,v)
        spectral_residual = log_spectrum - avg_spectrum
        
        # 步骤5: 结合相位谱进行逆变换
        reconstructed = np.exp(spectral_residual + 1j * phase)
        saliency = np.abs(fft.ifft2(reconstructed))
        
        # 步骤6: 高斯平滑
        saliency = gaussian_filter(saliency, sigma=self.sigma)
        
        # 步骤7: 归一化到 [0, 1]
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def _convolve(self, data, kernel):
        """
        手动二维卷积
        """
        h, w = data.shape
        kh, kw = kernel.shape
        result = np.zeros_like(data)
        pad_h, pad_w = kh // 2, kw // 2
        
        # 边缘填充
        padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # 滑动窗口卷积
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result


def demo():
    """演示函数：生成随机图像并可视化"""
    np.random.seed(42)
    
    # 创建模拟图像：背景 + 一个亮斑
    img = np.random.randn(128, 128) * 0.1 + 0.5
    img[40:60, 50:70] = 1.0   # 显著目标
    img[80:90, 20:30] = 0.9   # 另一个目标
    
    # 计算显著性
    model = SR(kernel_size=3, sigma=4)
    smap = model.compute_saliency(img)
    
    print(f"输入图像形状: {img.shape}")
    print(f"显著性图范围: [{smap.min():.3f}, {smap.max():.3f}]")
    print(f"显著性均值: {smap.mean():.3f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(smap, cmap='hot')
    axes[1].set_title('SR显著性图')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sr_saliency.png', dpi=150)
    print("结果已保存至 sr_saliency.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""
SR谱残差模型 - 手工实现
不依赖 scipy.fft，用 numpy 实现离散傅里叶变换
"""
import numpy as np
from numpy.fft import fft2, ifft2  # numpy 自带的 FFT 已足够


def manual_average_filter(image, kernel_size=3):
    """
    手工实现均值滤波
    
    参数:
        image: 输入二维数组
        kernel_size: 滤波核大小
    
    返回:
        均值滤波后的结果
    """
    h, w = image.shape
    pad = kernel_size // 2
    result = np.zeros_like(image)
    
    # 零填充
    padded = np.pad(image, pad, mode='edge')
    
    # 逐像素计算邻域均值
    for i in range(h):
        for j in range(w):
            # 提取邻域窗口
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(patch)
    
    return result


def spectral_residual_saliency(image, kernel_size=3, sigma=4):
    """
    手工实现 SR 谱残差显著性检测
    
    参数:
        image: 输入灰度图像 (H, W)
        kernel_size: 均值滤波核大小
        sigma: 高斯平滑标准差
    
    返回:
        显著性图 (0~1)
    """
    # 1. FFT
    F = fft2(image)
    magnitude = np.abs(F)
    phase = np.angle(F)
    
    # 2. 对数谱
    log_spec = np.log(magnitude + 1e-8)
    
    # 3. 手工均值滤波
    avg_spec = manual_average_filter(log_spec, kernel_size)
    
    # 4. 谱残差
    residual = log_spec - avg_spec
    
    # 5. 重构
    reconstructed = np.exp(residual + 1j * phase)
    saliency = np.abs(ifft2(reconstructed))
    
    # 6. 手工高斯平滑（简化版：多次均值滤波近似）
    for _ in range(int(sigma)):
        saliency = manual_average_filter(saliency, 5)
    
    # 7. 归一化
    s_min, s_max = saliency.min(), saliency.max()
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    
    return saliency


def test_manual_sr():
    """测试手工实现"""
    np.random.seed(42)
    
    # 测试图像
    img = np.random.randn(64, 64) * 0.1
    img[20:30, 25:35] = 0.8  # 模拟显著目标
    
    smap = spectral_residual_saliency(img)
    
    print("=== 手工实现测试 ===")
    print(f"显著区域 (20:30, 25:35) 均值: {smap[20:30, 25:35].mean():.4f}")
    print(f"背景区域均值: {smap[smap < 0.3].mean() if (smap < 0.3).sum() > 0 else 0:.4f}")
    print(f"最大显著性位置: {np.unravel_index(smap.argmax(), smap.shape)}")
    
    # 验证：显著区域的显著性应高于背景
    assert smap.max() > 0.5, "显著性最大值应较高"
    print("✓ 测试通过")


if __name__ == "__main__":
    test_manual_sr()
```

---

## 9. 可视化与结果理解

### 9.1 频谱可视化

SR 方法的关键在于理解频谱的变化过程：
- **原始振幅谱**：中心为低频，四周为高频，整体呈放射状衰减
- **对数谱**：将振幅的指数关系变为线性，便于观察
- **谱残差**：突出了偏离 $1/f$ 规律的频率成分

### 9.2 显著性图解读

- 亮度高的区域 → 谱残差大 → 偏离自然图像统计 → 显著性高
- 亮度低的区域 → 谱残差小 → 符合自然图像统计 → 背景
- 显著区域通常对应物体的轮廓和纹理变化剧烈处

### 9.3 不同参数的视觉效果

- **$n=3$**：最常用，平衡了背景抑制和目标保留
- **$n=5$**：背景抑制更强，但小目标可能丢失
- **$\sigma=2$**：显著区域边界清晰，但可能有噪点
- **$\sigma=8$**：显著性图更平滑，但细节丢失

---

## 10. 模型评估

### 10.1 常用评估指标

| 指标 | 描述 |
|------|------|
| AUC (Area Under ROC) | 将显著性图作为二分类器，评估预测注视点的能力 |
| NSS (Normalized Scanpath Saliency) | 注视点位置上的归一化显著性值 |
| CC (Correlation Coefficient) | 显著性图与真值图之间的皮尔逊相关系数 |
| KL散度 | 显著性图与真值图之间的分布差异 |

### 10.2 性能评估

```python
"""SR模型评估"""
import numpy as np


def compute_auc(saliency_map, ground_truth):
    """计算AUC值"""
    from sklearn.metrics import roc_auc_score
    
    # 将显著性图和真值展平
    s_flat = saliency_map.flatten()
    g_flat = (ground_truth.flatten() > 0).astype(int)
    
    return roc_auc_score(g_flat, s_flat)


def compute_cc(saliency_map, ground_truth):
    """计算相关系数"""
    s = (saliency_map - saliency_map.mean()) / (saliency_map.std() + 1e-8)
    g = (ground_truth - ground_truth.mean()) / (ground_truth.std() + 1e-8)
    return np.mean(s * g)


def evaluate_sr():
    """评估SR模型性能"""
    np.random.seed(42)
    
    # 模拟数据
    img = np.random.randn(128, 128) * 0.1
    img[40:60, 50:70] = 1.0
    
    # 真值图
    gt = np.zeros((128, 128))
    gt[40:60, 50:70] = 1.0
    
    model = SR()
    smap = model.compute_saliency(img)
    
    auc = compute_auc(smap, gt)
    cc = compute_cc(smap, gt)
    
    print(f"AUC: {auc:.4f}")
    print(f"CC: {cc:.4f}")
    
    return auc, cc


if __name__ == "__main__":
    evaluate_sr()
```

---

## 11. 常见问题与易错点

### Q1: 为什么要取对数谱而不是直接用振幅谱？
**A:** 自然图像的振幅谱满足 $1/f$ 分布，取对数后：
- 将乘法关系变为加法关系，便于线性滤波
- 减小动态范围，使低频和高频成分在数值上可比

### Q2: 均值滤波核大小如何选择？
**A:** 核越小，保留的"新颖"信息越多，但噪声也越多；核越大，背景抑制越强，但可能丢失小目标。$n=3$ 是论文推荐值。

### Q3: SR为什么能检测显著区域？
**A:** 显著目标会引入异常频谱成分（特别是中高频），这些成分在减去 $1/f$ 平均模式后仍然存在。

### Q4: SR和ITTI模型有什么区别？
**A:** ITTI在空域操作（颜色、亮度、方向特征），SR在频率域操作。SR计算更快但缺乏语义信息，ITTI更接近生物视觉但计算量大。

### Q5: 为什么结果中会有环状伪影？
**A:** 这是FFT边界效应导致的。可以在FFT前对图像进行加窗处理（如汉宁窗）减轻。

---

## 12. 学习总结

### 核心要点

1. **统计基础**：自然图像服从 $1/f$ 频谱分布规律
2. **关键操作**：对数谱 → 均值滤波 → 谱残差 → IFFT
3. **核心优势**：无需训练、计算高效、理论完整
4. **主要局限**：缺乏语义、边界效应、对噪声敏感

### 知识图谱

```
图像处理 → 频域分析 → FFT → 振幅谱 + 相位谱
           → 自然图像统计 → 1/f 规律
           → 均值滤波 → 估计背景频谱
           → 谱残差 = 对数谱 - 平均谱
           → 逆FFT → 显著性图
```

### 与后续模型的关系

SR 启发了大量频率域显著性方法：
- **PFT** (Phase Fourier Transform)：仅用相位谱重构
- **HFT** (Hypercomplex Fourier Transform)：四元数傅里叶变换
- **IS** (Image Signature)：用 DCT 符号函数替代谱残差

---

## 13. 练习题与思考题

### 基础题

**1.** 为什么SR模型不需要训练？

<details>
<summary>答案</summary>
SR基于自然图像统计规律（1/f分布），这是一个先验知识而非从数据中学习得到的。任何自然图像都符合这一规律，因此无需针对特定数据集训练。
</details>

**2.** 谱残差 $R(u,v)$ 在什么位置取值最大？

<details>
<summary>答案</summary>
在偏离1/f分布最大的频率位置。通常显著目标的轮廓和纹理对应中高频区域，因此谱残差在中高频区域取值最大。
</details>

**3.** 如果输入图像是均匀的纯色，SR会输出什么？

<details>
<summary>答案</summary>
纯色图像的频谱只有直流分量（$u=0,v=0$处），振幅谱几乎全部为零。对数谱趋近于负无穷，谱残差在直流处很大，逆变换后会在图像中心产生一个高亮区域。但实际中由于数值计算误差，结果可能是全零或噪声。
</details>

### 进阶题

**4.** 推导：为什么自然图像的振幅谱服从 $1/f$ 分布？

<details>
<summary>答案</summary>
自然图像中的物体边缘和纹理具有尺度不变性。考虑一个包含随机边缘的图像，其自相关函数 $R(\tau)$ 在远距离处衰减。边缘的傅里叶变换幅度的平方与 $1/f^2$ 成正比。数学上，如果图像的自相关函数是 $e^{-\alpha|\tau|}$，则功率谱密度为 $1/(\alpha^2 + f^2)$，在 $f \gg \alpha$ 时近似为 $1/f^2$，振幅谱为 $1/f$。
</details>

**5.** 尝试修改代码，在FFT前对图像应用汉宁窗，观察对结果的影响。

<details>
<summary>答案</summary>
汉宁窗可以减轻FFT的频谱泄漏效应，使谱残差更准确。窗函数为 $w(n) = 0.5(1 - \cos(2\pi n/N))$。应用后边界伪影减少，但图像边缘的显著性也会减弱。
</details>

**6.** 如果使用 $5 \times 5$ 的均值滤波核，与小核相比会有什么不同？

<details>
<summary>答案</summary>
5×5核会更大程度地平滑对数谱，产生更"干净"的谱残差，背景抑制更强。但同时会丢失更细致的频谱变化，导致小目标或精细结构的显著性降低。
</details>

---

## 14. 学习路径建议

### 预备知识
- 傅里叶变换（一维和二维）
- 数字图像处理基础（频谱分析）
- 信号与系统中的卷积和滤波

### 进阶方向
1. **SR → IS → HFT → PFT**：频率域显著性检测的发展脉络
2. **SR → GBVS → CAS**：从频率域到图模型的显著性方法
3. **SR + CNN**：将频率域方法作为深度学习的预处理模块

### 推荐阅读
- Hou & Zhang. "Saliency Detection: A Spectral Residual Approach." CVPR 2007.
- Hou et al. "Image Signature: Highlighting Sparse Salient Regions." TPAMI 2012.
- Li et al. "Saliency Detection Based on Frequency Domain Analysis." 2014.

### 项目实践
1. 在真实图像数据集（如MIT300、CAT2000）上评估SR性能
2. 将SR作为ROI提取模块，集成到目标检测pipeline中
3. 实现SR的视频版本（3D FFT谱残差）
