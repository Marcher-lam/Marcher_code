# SUN模型 学习文档

> 基于自然统计量的贝叶斯显著性框架——自下而上与自上而下的统一。
>
> 来源线索：本节内容根据原书第2.2.1节"SUN:基于贝叶斯框架的视觉显著性"整理。

---

## 1. 算法基础认知

**一句话定义：** SUN（Saliency Using Natural statistics）由Zhang、Tong、Marks、Shan和Cottrell于2008年在NIPS上提出，是一个完整的贝叶斯显著性框架，包含自下而上（SUN-BU）和自上而下（SUN-TD）两个子模型，首次在统一的概率框架下处理两种注意力。

**核心思想：** 使用贝叶斯公式将显著性表达为给定位置 $l_z$ 和特征 $f_z$ 的条件下该位置被注视的后验概率：

$$
\text{Sal}(z) = p(C=1 | L=l_z, F=f_z)
$$

通过贝叶斯定理，这个后验概率可以分解为三个因子的乘积：
- **特征似然** $p(F=f_z | C=1)$：显著位置的特征分布
- **位置先验** $p(L=l_z | C=1)$：注视点的空间偏好（中心偏置）
- **特征概率** $p(F=f_z)$：图像中所有位置的特征分布

**为什么是贝叶斯？** 贝叶斯框架自然地融合了自下而上（基于特征的稀有性）和自上而下（基于任务的目标特征知识）两种信号。自下而上对应于 $- \log p(F=f_z)$（特征稀有性），自上而下对应于 $\log p(F=f_z|C=1)$（目标特征匹配度）。

**历史背景：** 2008年Zhang等人提出该模型，是第一个统一处理自上而下和自下而上注意力的贝叶斯框架。与之前的模型相比（如ITTI、GBVS仅处理自下而上），SUN能够融入目标任务知识。

---

## 2. 核心原理

### 2.1 贝叶斯显著性框架

定义二值随机变量 $C \in \{0,1\}$ 表示位置 $z$ 是否被注视（$C=1$ 表示注视）。根据贝叶斯定理：

$$
p(C=1 | L=l_z, F=f_z) = \frac{p(C=1) p(L=l_z, F=f_z | C=1)}{p(L=l_z, F=f_z)}
$$

假设位置和特征在给定 $C$ 的条件下独立：

$$
p(L=l_z, F=f_z | C=1) = p(L=l_z | C=1) \cdot p(F=f_z | C=1)
$$

同时，先验 $p(C=1)$ 是常数。因此：

$$
\text{Sal}(z) \propto \frac{p(L=l_z | C=1) \cdot p(F=f_z | C=1)}{p(F=f_z) \cdot p(L=l_z)}
$$

由于位置分布 $p(L=l_z)$ 通常是均匀的（或可以归入位置先验），简化后：

$$
\text{Sal}(z) \propto p(L=l_z | C=1) \cdot \frac{p(F=f_z | C=1)}{p(F=f_z)}
$$

### 2.2 自下而上模型（SUN-BU）

对于自下而上注意力（无目标任务知识），$p(F=f_z | C=1)$ 是均匀分布（所有特征同等可能），因此：

$$
\text{Sal}_{BU}(z) \propto p(L=l_z | C=1) \cdot \frac{1}{p(F=f_z)}
$$

取对数后：

$$
\log \text{Sal}_{BU}(z) \propto \log p(L=l_z | C=1) - \log p(F=f_z)
$$

其中 $-\log p(F=f_z)$ 就是**点式互信息**，即特征 $f_z$ 的自信息。

### 2.3 自上而下模型（SUN-TD）

对于自上而下注意力（有目标任务，如寻找红色物体），需要知道目标特征的分布 $p(F=f_z | C=1)$：

$$
\text{Sal}_{TD}(z) \propto p(L=l_z | C=1) \cdot \frac{p(F=f_z | C=1)}{p(F=f_z)}
$$

这等价于点式互信息 $\text{PMI}(f_z, C=1)$：

$$
\text{PMI}(f_z, C=1) = \log \frac{p(F=f_z | C=1)}{p(F=f_z)}
$$

### 2.4 特征提取

SUN使用多尺度DoG（Difference of Gaussian）滤波器提取特征，模拟V1皮层简单细胞：

$$
\text{DoG}(x, y; \sigma_1, \sigma_2) = \frac{1}{2\pi\sigma_1^2} e^{-\frac{x^2+y^2}{2\sigma_1^2}} - \frac{1}{2\pi\sigma_2^2} e^{-\frac{x^2+y^2}{2\sigma_2^2}}
$$

多尺度 DoG 提供了不同频率和方向的选择性。SUN-BU使用94维特征（多尺度、多方向的DoG响应）。

---

## 3. 数学公式与推导

### 3.1 从贝叶斯公式到显著性

$$
\begin{aligned}
p(C=1|L,F) &= \frac{p(C=1) p(L,F|C=1)}{p(L,F)} \\
&= \frac{p(C=1) p(L|C=1) p(F|C=1)}{p(L) p(F)} \\
&\propto \frac{p(L|C=1)}{p(L)} \cdot \frac{p(F|C=1)}{p(F)}
\end{aligned}
$$

由于通常 $p(L)$ 被视为均匀分布（或合并到位置先验中）：

$$
p(C=1|L,F) \propto p(L|C=1) \cdot \frac{p(F|C=1)}{p(F)}
$$

### 3.2 点式互信息（PMI）

点式互信息度量两个事件之间的关联程度：

$$
\text{PMI}(x, y) = \log \frac{p(x,y)}{p(x)p(y)} = \log \frac{p(x|y)}{p(x)}
$$

在SUN中：

$$
\text{PMI}(f_z, C=1) = \log \frac{p(F=f_z|C=1)}{p(F=f_z)}
$$

当 $\text{PMI} > 0$ 时，特征 $f_z$ 与注视正相关（可能引起注意）；$\text{PMI} < 0$ 时负相关。

### 3.3 DoG特征提取

DoG滤波器是LoG（Laplacian of Gaussian）的近似：

$$
\text{DoG}(x,y;\sigma_1,\sigma_2) = G(x,y;\sigma_1) - G(x,y;\sigma_2)
$$

其中 $G(x,y;\sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/(2\sigma^2)}$。

SUN使用多个尺度的DoG（除了中心-周围差分，还包括不同方向比）：

$$
\sigma_1 \in \{2, 4, 8, 16, 32\}, \quad \sigma_2 = 1.6\sigma_1
$$

### 3.4 概率密度估计

$p(F=f_z)$ 从自然图像数据集中用广义高斯分布（GGD）拟合：

$$
p(f; \alpha, \beta) = \frac{\beta}{2\alpha\Gamma(1/\beta)} e^{-(|f|/\alpha)^\beta}
$$

GGD可以适配不同的分布形状：$\beta=2$ 时退化为高斯分布，$\beta=1$ 时为拉普拉斯分布。

$p(F=f_z|C=1)$ 从注视点数据中学习（有监督）。

---

## 4. 训练过程讲解

### 4.1 SUN-BU（无监督）

1. **收集自然图像**：大量自然场景图像
2. **DoG特征提取**：在多个尺度上提取DoG响应
3. **GGD拟合**：对每个尺度的DoG响应分布用GGD建模
4. **位置先验学习**：统计注视点的空间分布（中心偏置）
5. **推理**：对新图像提取DonG特征→计算 $-\log p(f_z)$ + 位置先验

### 4.2 SUN-TD（有监督）

1. **收集目标图像**：包含目标的图像及注视点数据
2. **注视点特征提取**：在注视点位置提取DoG特征
3. **GGD拟合**：拟合 $p(F=f_z|C=1)$
4. **推理**：计算 $\log p(f_z|C=1) - \log p(f_z)$ + 位置先验

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 注视点预测（自由观看） | SUN-BU预测自然场景中的注视点 |
| 目标搜索 | SUN-TD在场景中寻找特定目标 |
| 视觉显著性基准 | 统一的贝叶斯框架作为理论基线 |
| 计算神经科学 | 模拟注意力中贝叶斯推理过程 |
| 图像压缩 | 使用显著性信息指导非均匀编码 |
| 人机交互 | 预测用户在界面上的注意力分布 |

---

## 6. 优缺点分析

**优点：**
- ✅ **统一贝叶斯框架**：自下而上和自上而下的有机融合
- ✅ **理论完备**：基于概率论的严格推导
- ✅ **PMI可解释**：显著性有明确的统计意义
- ✅ **位置先验自然**：中心偏置自动融入
- ✅ **多尺度特征**：DoG多尺度提取全面信息

**缺点：**
- ❌ **特征有限**：只使用DoG特征，缺乏颜色、方向等
- ❌ **独立性假设**：位置和特征的独立性假设不完全成立
- ❌ **需要大量训练数据**：GGD拟合和注视点数据需要足够样本
- ❌ **计算量大**：多尺度DoG + 概率密度计算
- ❌ **自上而下需要标注**：SUN-TD需要注视点标注数据

---

## 7. 调库实现

```python
"""
SUN模型 - 完整调库实现
包含SUN-BU（自下而上）和SUN-TD（自上而下）
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gennorm  # 广义高斯分布
import matplotlib.pyplot as plt


class SUN_BU:
    """
    SUN自下而上模型
    
    使用DoG特征提取 + GGD拟合 + 自信息计算
    """
    
    def __init__(self, scales=[4, 8, 16, 32], ratio=1.6):
        """
        参数:
            scales: DoG的中心尺度列表
            ratio: 周围尺度 = scales * ratio
        """
        self.scales = scales
        self.ratio = ratio
        self.ggd_params = {}  # {scale: (alpha, beta)}
    
    def _dog_filter(self, image, sigma):
        """DoG滤波: G(sigma) - G(sigma * ratio)"""
        center = gaussian_filter(image, sigma)
        surround = gaussian_filter(image, sigma * self.ratio)
        return center - surround
    
    def _fit_ggd(self, data):
        """拟合广义高斯分布参数 (alpha, beta)"""
        # 使用矩估计法
        mean = np.mean(data)
        
        # 中心化后的绝对值的矩
        m1 = np.mean(np.abs(data - mean))
        m2 = np.mean((data - mean) ** 2)
        
        # beta的近似估计
        beta = np.sqrt(m2) / (m1 + 1e-8)
        beta = np.clip(beta, 0.3, 5.0)  # 限制范围
        
        # alpha的估计
        from scipy.special import gamma
        alpha = np.sqrt(m2 * gamma(1/beta) / gamma(3/beta))
        
        return alpha, beta
    
    def fit(self, images):
        """
        从自然图像中学习GGD参数
        
        参数:
            images: 图像列表 [(H,W)] 或 [(H,W,C)]
        """
        all_responses = {s: [] for s in self.scales}
        
        for img in images:
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            for s in self.scales:
                resp = self._dog_filter(img, s)
                all_responses[s].extend(resp.flatten())
        
        # 对每个尺度拟合GGD
        for s in self.scales:
            data = np.array(all_responses[s])
            alpha, beta = self._fit_ggd(data)
            self.ggd_params[s] = (alpha, beta)
            print(f"尺度 {s}: alpha={alpha:.3f}, beta={beta:.3f}")
    
    def _log_p_ggd(self, x, alpha, beta):
        """计算广义高斯分布的 log pdf"""
        from scipy.special import gamma, gammaln
        # log p(x) = log(beta) - log(2*alpha) - lgamma(1/beta) - (|x|/alpha)^beta
        log_part1 = np.log(beta) - np.log(2 * alpha) - gammaln(1.0 / beta)
        log_part2 = -(np.abs(x) / alpha) ** beta
        return log_part1 + log_part2
    
    def compute_saliency(self, image):
        """
        计算SUN-BU显著性
        
        参数:
            image: 输入图像
        
        返回:
            saliency: 归一化的显著性图
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        h, w = image.shape
        saliency = np.zeros((h, w))
        
        for s in self.scales:
            alpha, beta = self.ggd_params[s]
            resp = self._dog_filter(image, s)
            log_p = self._log_p_ggd(resp, alpha, beta)
            saliency += -log_p  # 自信息
        
        # 高斯平滑
        from scipy.ndimage import gaussian_filter
        saliency = gaussian_filter(saliency, sigma=4)
        
        # 归一化
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


class SUN_TD:
    """
    SUN自上而下模型
    
    基于PMI: log p(f|C=1) - log p(f)
    需要注视点数据训练 p(f|C=1)
    """
    
    def __init__(self, scales=[4, 8, 16, 32], ratio=1.6):
        self.scales = scales
        self.ratio = ratio
        self.ggd_bg = {}   # 背景分布参数 p(f)
        self.ggd_tg = {}   # 目标分布参数 p(f|C=1)
    
    def fit_bg(self, images):
        """拟合背景分布"""
        all_responses = {s: [] for s in self.scales}
        for img in images:
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            for s in self.scales:
                center = gaussian_filter(img, s)
                surround = gaussian_filter(img, s * self.ratio)
                resp = center - surround
                all_responses[s].extend(resp.flatten())
        
        for s in self.scales:
            data = np.array(all_responses[s])
            m1 = np.mean(np.abs(data))
            m2 = np.mean(data ** 2)
            beta = np.sqrt(m2) / (m1 + 1e-8)
            beta = np.clip(beta, 0.3, 5.0)
            from scipy.special import gamma
            alpha = np.sqrt(m2 * gamma(1/beta) / gamma(3/beta))
            self.ggd_bg[s] = (alpha, beta)
    
    def fit_tg(self, images, fixation_masks):
        """拟合目标分布（注视点位置的特征分布）"""
        all_responses = {s: [] for s in self.scales}
        for img, mask in zip(images, fixation_masks):
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            for s in self.scales:
                center = gaussian_filter(img, s)
                surround = gaussian_filter(img, s * self.ratio)
                resp = center - surround
                all_responses[s].extend(resp[mask > 0])
        
        for s in self.scales:
            data = np.array(all_responses[s])
            if len(data) > 0:
                m1 = np.mean(np.abs(data))
                m2 = np.mean(data ** 2)
                beta = np.sqrt(m2) / (m1 + 1e-8)
                beta = np.clip(beta, 0.3, 5.0)
                from scipy.special import gamma
                alpha = np.sqrt(m2 * gamma(1/beta) / gamma(3/beta))
                self.ggd_tg[s] = (alpha, beta)
    
    def _log_p_ggd(self, x, alpha, beta):
        """广义高斯分布 log pdf"""
        from scipy.special import gamma, gammaln
        log_part1 = np.log(beta) - np.log(2 * alpha) - gammaln(1.0 / beta)
        log_part2 = -(np.abs(x) / alpha) ** beta
        return log_part1 + log_part2
    
    def compute_saliency(self, image):
        """计算SUN-TD显著性 (基于PMI)"""
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        h, w = image.shape
        saliency = np.zeros((h, w))
        
        for s in self.scales:
            center = gaussian_filter(image, s)
            surround = gaussian_filter(image, s * self.ratio)
            resp = center - surround
            
            alpha_bg, beta_bg = self.ggd_bg[s]
            log_p_bg = self._log_p_ggd(resp, alpha_bg, beta_bg)
            
            if s in self.ggd_tg:
                alpha_tg, beta_tg = self.ggd_tg[s]
                log_p_tg = self._log_p_ggd(resp, alpha_tg, beta_tg)
                # PMI = log p(f|C=1) - log p(f)
                saliency += (log_p_tg - log_p_bg)
            else:
                # 无目标信息时退化为自信息
                saliency += -log_p_bg
        
        # 位置先验（中心偏置）
        y, x = np.mgrid[0:h, 0:w]
        center_y, center_x = h / 2, w / 2
        pos_prior = np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * (min(h, w) / 4) ** 2))
        saliency = saliency * pos_prior
        
        # 归一化
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


def demo():
    """演示函数"""
    np.random.seed(42)
    
    # 生成模拟自然图像用于训练
    print("生成训练数据...")
    train_images = []
    for _ in range(50):
        img = np.random.randn(64, 64) * 0.2 + 0.5
        train_images.append(img)
    
    # SUN-BU
    print("\n=== SUN-BU ===")
    bu = SUN_BU(scales=[4, 8, 16])
    bu.fit(train_images)
    
    test_img = np.random.randn(64, 64) * 0.15 + 0.5
    test_img[20:30, 25:35] = 1.5
    
    smap_bu = bu.compute_saliency(test_img)
    print(f"SUN-BU显著性范围: [{smap_bu.min():.3f}, {smap_bu.max():.3f}]")
    
    # SUN-TD
    print("\n=== SUN-TD ===")
    td = SUN_TD(scales=[4, 8, 16])
    td.fit_bg(train_images)
    
    # 模拟注视点数据
    fix_masks = []
    for _ in range(10):
        mask = np.zeros((64, 64))
        mask[25:35, 30:40] = 1
        fix_masks.append(mask)
    
    td.fit_tg(train_images[:10], fix_masks)
    smap_td = td.compute_saliency(test_img)
    print(f"SUN-TD显著性范围: [{smap_td.min():.3f}, {smap_td.max():.3f}]")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title('测试图像')
    axes[0].axis('off')
    axes[1].imshow(smap_bu, cmap='hot')
    axes[1].set_title('SUN-BU (自信息)')
    axes[1].axis('off')
    axes[2].imshow(smap_td, cmap='hot')
    axes[2].set_title('SUN-TD (PMI)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('sun_saliency.png', dpi=150)
    print("\n结果已保存至 sun_saliency.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""
SUN模型 - 手工核心实现
不依赖scipy.stats，自实现DoG和GGD
"""
import numpy as np


def gaussian_kernel_2d(size, sigma):
    """手工2D高斯核"""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            d = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = np.exp(-d / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve_2d(data, kernel):
    """手工2D卷积"""
    h, w = data.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(data)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result


def dog_filter_manual(image, sigma, ratio=1.6):
    """手工DoG滤波"""
    ksize1 = int(2 * np.ceil(2 * sigma) + 1)
    ksize2 = int(2 * np.ceil(2 * sigma * ratio) + 1)
    
    g1 = gaussian_kernel_2d(ksize1, sigma)
    g2 = gaussian_kernel_2d(ksize2, sigma * ratio)
    
    center = convolve_2d(image, g1)
    surround = convolve_2d(image, g2)
    
    return center - surround


def ggd_log_pdf_manual(x, alpha, beta):
    """
    手工广义高斯分布 log pdf
    
    p(x) = beta / (2*alpha*Gamma(1/beta)) * exp(-(|x|/alpha)^beta)
    """
    # 近似 Gamma 函数
    def gamma_approx(z):
        """Stirling近似"""
        if z < 0.5:
            return gamma_approx(z + 1) / z
        return np.sqrt(2 * np.pi / z) * (z / np.e) ** z
    
    log_beta = np.log(beta)
    log_2alpha = np.log(2 * alpha)
    log_gamma = np.log(gamma_approx(1.0 / beta))
    
    log_part = log_beta - log_2alpha - log_gamma
    exp_part = -(np.abs(x) / alpha) ** beta
    
    return log_part + exp_part


def sun_bu_manual(image, scales=[4, 8, 16], ratio=1.6):
    """
    手工SUN-BU实现
    
    参数:
        image: 输入灰度图像
        scales: DoG尺度列表
        ratio: 周围/中心尺度比
    
    返回:
        显著性图
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    h, w = image.shape
    saliency = np.zeros((h, w))
    
    for s in scales:
        resp = dog_filter_manual(image, s, ratio)
        
        # 估计GGD参数（矩估计法）
        data = resp.flatten()
        m1 = np.mean(np.abs(data))
        m2 = np.mean(data ** 2)
        beta = np.sqrt(m2) / (m1 + 1e-8)
        beta = np.clip(beta, 0.3, 5.0)
        alpha = np.sqrt(m2) / np.sqrt(3.0 / beta + 1)  # 简化版
        
        # 计算自信息
        log_p = ggd_log_pdf_manual(resp, alpha, beta)
        saliency += -log_p
    
    # 简单平滑
    kernel = gaussian_kernel_2d(int(2 * np.ceil(2 * 4) + 1), 4)
    saliency = convolve_2d(saliency, kernel)
    
    # 归一化
    s_min, s_max = saliency.min(), saliency.max()
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    
    return saliency


def test_sun_manual():
    """测试手工实现"""
    np.random.seed(42)
    img = np.random.randn(32, 32) * 0.2 + 0.5
    img[10:16, 14:20] = 1.5
    
    print("=== 手工SUN-BU测试 ===")
    smap = sun_bu_manual(img, scales=[4, 8])
    print(f"显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")
    assert smap.max() > 0.5, "显著性最大值应较高"
    print("✓ 测试通过")


if __name__ == "__main__":
    test_sun_manual()
```

---

## 9. 可视化与结果理解

### 9.1 DoG响应图

- 小尺度DoG（如 $\sigma=4$）：响应边缘和细节
- 大尺度DoG（如 $\sigma=32$）：响应大范围亮度变化
- 多尺度融合：综合了不同尺度的结构信息

### 9.2 自信息 vs PMI

- **自信息（SUN-BU）**：$-\log p(f)$，高值对应罕见特征（无论是否为目标）
- **PMI（SUN-TD）**：$\log[p(f|C=1)/p(f)]$，高值对应与目标强相关的特征

### 9.3 位置先验的影响

中心偏置使显著性图在靠近图像中心的位置有更高的基线值，这与眼动实验中注视点倾向于落在图像中心附近的现象一致。

---

## 10. 模型评估

```python
"""SUN模型评估"""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_sun():
    """评估SUN-BU模型"""
    np.random.seed(42)
    
    train_imgs = [np.random.randn(32, 32) for _ in range(20)]
    
    bu = SUN_BU(scales=[4, 8])
    bu.fit(train_imgs)
    
    img = np.random.randn(32, 32) * 0.15 + 0.5
    img[10:16, 14:20] = 1.5
    gt = np.zeros((32, 32))
    gt[10:16, 14:20] = 1.0
    
    smap = bu.compute_saliency(img)
    auc = roc_auc_score(gt.flatten() > 0, smap.flatten())
    print(f"SUN-BU AUC: {auc:.4f}")
    
    return auc


if __name__ == "__main__":
    evaluate_sun()
```

---

## 11. 常见问题与易错点

### Q1: SUN-BU与AIM有什么区别？
**A:** 都基于信息论，但区别在于：(1) SUN使用DoG特征+GGD建模，AIM使用ICA特征+KDE建模；(2) SUN有位置先验；(3) SUN有自上而下版本。

### Q2: PMI的取值有什么意义？
**A:** PMI>0表示特征与注视正相关（该特征出现时更可能被关注），PMI<0表示负相关，PMI=0表示独立。

### Q3: 为什么需要 $\log$ 形式？
**A:** $\log$ 将概率的乘性关系变为加性关系，并且与生物视觉系统中韦伯-费希纳定律（感知与刺激的对数成正比）一致。

### Q4: 位置先验 $p(L|C=1)$ 如何获得？
**A:** 从眼动实验数据中统计注视点位置分布，通常呈现以图像中心为中心的高斯分布。

### Q5: SUN如何融合多尺度信息？
**A:** 简单求和（假设各尺度独立），更复杂的做法是加权求和或用PCA降维后再计算自信息。

---

## 12. 学习总结

### 核心要点

1. **贝叶斯框架**：显著性 = 后验概率 $p(C=1|L,F)$
2. **统一模型**：BU = 自信息，TD = PMI
3. **DoG特征**：多尺度差分高斯模拟V1皮层
4. **GGD建模**：广义高斯分布拟合特征分布
5. **位置先验**：自然融合注视点的空间偏好

### SUN家族比较

| 子模型 | 公式 | 需要 | 输出 |
|--------|------|------|------|
| SUN-BU | $-\log p(f)$ | 自然图像 | 显著性/惊奇度 |
| SUN-TD | $\log \frac{p(f\|C=1)}{p(f)}$ | 注视点数据 | 目标相关显著性 |
| SUN全模型 | 两者+位置先验 | 两者都需要 | 完整贝叶斯框架 |

---

## 13. 练习题与思考题

### 基础题

**1.** 贝叶斯公式中哪些项表示自下而上信息，哪些表示自上而下？

<details>
<summary>答案</summary>
- 自下而上：$1/p(F=f_z)$（或特征稀有性）
- 自上而下：$p(F=f_z|C=1)$（目标特征知识）
- 位置先验：$p(L=l_z|C=1)$ 可属于任一类
</details>

**2.** 如果目标任务搜索"红色圆形"，SUN-TD会如何调整？

<details>
<summary>答案</summary>
$p(F=f_z|C=1)$ 在红色和圆形的特征维度上会有高概率，PMI在红色和圆形区域为正，其他区域为负，导致只有红色圆形区域被标记为显著。
</details>

**3.** 为什么DoG滤波能模拟V1皮层简单细胞？

<details>
<summary>答案</summary>
V1简单细胞的感受野是"中心-周围"拮抗结构，DoG有相同的数学形式：中心兴奋+周围抑制。不同尺度和方向的DoG能模拟不同大小和朝向的简单细胞感受野。
</details>

### 进阶题

**4.** 推导SUN框架下的最优注视策略。

<details>
<summary>答案</summary>
最优注视策略是选择PMI最大的位置，即 $\arg\max_z \log[p(f_z|C=1)/p(f_z)]$。这等价于选择最能减少目标不确定性（最大信息增益）的位置，符合信息最大化的最优搜索理论。
</details>

**5.** 如果 $p(F|C=1) = p(F)$，SUN-TD会变成什么？

<details>
<summary>答案</summary>
如果目标和背景的特征分布完全一样，PMI=0，SUN-TD退化为均匀分布（无信息），只有位置先验起作用。这意味着目标无法通过特征从背景中区分出来。
</details>

**6.** 如何将SUN扩展到视频显著性？

<details>
<summary>答案</summary>
扩展方式包括：(1) 增加时间维度的特征（光流）的PMI；(2) 将位置先验变为时间先验（注视点的时间序列相关）；(3) 引入运动特征的概率分布。
</details>

---

## 14. 学习路径建议

### 预备知识
- 贝叶斯统计与概率论
- 信息论（自信息、互信息）
- 信号处理（DoG滤波器）
- 概率密度估计

### 进阶方向
1. **SUN → Bayesian surprise (Itti & Baldi)**：从先验到后验的KL散度
2. **SUN + Deep Learning**：深度特征替代DoG特征
3. **SUN → Goal-driven attention**：用强化学习建模任务导向注意力

### 推荐阅读
- Zhang et al. "SUN: A Bayesian Framework for Saliency Using Natural Statistics." Journal of Vision 2008.
- Itti & Baldi. "Bayesian Surprise Attracts Human Attention." NIPS 2006.
- Torralba et al. "Contextual Guidance of Attention in Natural Scenes." 2006.

### 项目实践
1. 在MIT1003数据集上实现SUN-BU并与ITTI对比
2. 收集特定目标的注视点数据，训练SUN-TD
3. 尝试用高斯混合模型（GMM）替代GGD进行概率建模
