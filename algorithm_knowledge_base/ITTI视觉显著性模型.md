# ITTI视觉显著性模型 学习文档

> 计算视觉显著性——通过多尺度特征竞争确定视觉注意焦点。

> 来源线索：本节内容根据原书第2章关于"注视点预测"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义：** ITTI模型（Itti-Neisser-Baldin模型）是由Itti等人于1998年提出的经典视觉显著性计算模型，通过提取多尺度的颜色、亮度和方向特征，计算"中心-周围"差异来生成显著性图。

**直觉类比：** 当你走进一个房间，你的眼睛会自然地被那些"与众不同"的东西吸引——比如墙上的一幅红画、桌上的一个亮物体。ITTI模型正是模拟这个过程：它提取图像中与周围不同的区域，这些区域越"突出"，显著性越高。

**历史背景：** 1998年，Laurent Itti、Christof Koch和Pietro Baldi在IEEE CVPR上发表论文"A Model of Saliency-Based Visual Attention for Rapid Scene Analysis"，开创了计算显著性检测领域。

**算法定位：** 这是计算机视觉中的显著性检测模型，属于生物启发式方法。在现代深度学习显著性检测出现之前，这是主流方法。

**前置知识：**
- 图像处理基础（卷积、滤波）
- 多尺度图像表示
- 颜色空间转换

---

## 2. 核心原理

### 2.1 核心思想

ITTI模型模拟人类视觉系统的早期加工机制，通过三个步骤生成显著性图：

1. **多尺度特征提取：** 构建高斯金字塔，提取不同尺度的图像
2. **中心-周围差异计算：** 计算不同尺度间的特征差异
3. **跨尺度融合：** 归一化后融合生成最终显著性图

### 2.2 工作流程

```
输入图像 → 高斯金字塔构建 → 多特征提取（颜色、亮度、方向）
→ Center-Surround差分 → 跨尺度融合 → 显著性图 → 注视点预测
```

### 2.3 关键概念

**高斯金字塔：** 用不同 sigma 的高斯模糊 + 下采样得到的图像层级

**Center-Surround：** 精细尺度（center）与粗糙尺度（surround）的差异，类似视觉感受野

**特征通道：**
- 颜色特征：R、G、B 和 黄-蓝、红-绿 opponent color
- 亮度特征：灰度图像
- 方向特征：Gabor滤波器提取的4个方向（0°, 45°, 90°, 135°）

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $I$ | 输入图像 |
| $\mathcal{G}(I, \sigma)$ | 高斯金字塔，尺度为 $\sigma$ |
| $C-S(s, c)$ | 中心-周围差分，尺度 $s$ 和 $c$ |
| $S$ | 最终显著性图 |

### 3.2 高斯金字塔构建

对输入图像 $I$ 应用不同尺度 $\sigma$ 的高斯模糊：

$$\mathcal{G}(I, \sigma) = G_\sigma * I$$

然后进行 2 倍下采样得到金字塔层级。

### 3.3 中心-周围差分

对于特征图 $F$，中心-周围差定义为：

$$C-S(s, c) = |\mathcal{G}(F, s) - \mathcal{G}(F, c)| \uparrow^s$$

其中 $s > c$（中心尺度更精细），$\uparrow^s$ 表示上采样到尺度 $s$。

典型尺度对：$(s, c) \in \{(2,4), (3,5), (4,6)\}$

### 3.4 特征归一化

为使不同特征通道可比，需要归一化：

$$N(F) = \frac{F - \mu(F)}{\sigma(F)}$$

其中 $\mu$ 是均值，$\sigma$ 是标准差。

### 3.5 显著性图生成

$$S = \frac{1}{3}\sum_{feature \in \{color, intensity, orientation\}} N(feature)$$

---

## 4. 训练过程讲解

### 4.1 特征提取代码

```python
import numpy as np
import cv2


def gaussian_pyramid(image, levels=5):
    """构建高斯金字塔"""
    pyramid = [image]
    for i in range(1, levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def compute_color_features(pyramid):
    """计算颜色特征"""
    features = []
    for img in pyramid:
        if len(img.shape) == 3:
            # RGB通道
            R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
            # Opponent colors
            RG = R - G
            BY = (R + G) / 2 - B
            features.extend([R, G, B, RG, BY])
    return features


def compute_intensity_features(pyramid):
    """计算亮度特征"""
    return [np.mean(img, axis=2) for img in pyramid]


def compute_orientation_features(pyramid):
    """计算方向特征（使用Gabor滤波器）"""
    orientations = [0, 45, 90, 135]
    features = []
    for img in pyramid:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        for theta in orientations:
            kernel = cv2.getGaborKernel((21, 21), 5, theta * np.pi / 180, 10, 0.5)
            filtered = cv2.filter2D(gray, -1, kernel)
            features.append(filtered)
    return features
```

---

## 5. 应用场景

1. **注视点预测：** 预测人眼在自然场景中的注视位置
2. **图像压缩：** 对显著区域高画质，对非显著区域低画质
3. **目标检测引导：** 用显著性图引导物体检测的注意力区域
4. **图像分割辅助：** 显著区域可以作为图像分割的先验

---

## 6. 优缺点分析

### 6.1 优点

1. **生物启发性：** 基于人类视觉系统的早期加工机制
2. **计算高效：** 不需要训练，纯手工特征提取
3. **可解释性强：** 特征和融合过程透明可理解

### 6.2 缺点

1. **特征有限：** 只使用低层特征，缺乏语义理解
2. **手工调参：** 需要大量人工设计参数
3. **不适合复杂场景：** 在复杂自然场景中效果一般

---

## 7. 调库实现

```python
"""
ITTI视觉显著性模型完整实现
使用OpenCV和NumPy实现多尺度特征提取和显著性计算
"""

import numpy as np
import cv2
from scipy.ndimage import zoom


class ITIISaliencyModel:
    """ITTI视觉显著性模型"""
    
    def __init__(self, num_levels=4):
        self.num_levels = num_levels
        self.scales = list(range(2, 2 + num_levels))
    
    def gaussian_pyramid(self, image):
        """构建高斯金字塔"""
        pyramid = [image.astype(np.float32)]
        for _ in range(self.num_levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid
    
    def center_surround(self, fine, coarse):
        """计算中心-周围差分"""
        # 上采样到相同尺寸
        scale_factor = fine.shape[0] / coarse.shape[0]
        upsampled = zoom(coarse, scale_factor, order=1)
        # 裁剪到相同大小
        min_h = min(fine.shape[0], upsampled.shape[0])
        min_w = min(fine.shape[1], upsampled.shape[1])
        return np.abs(fine[:min_h, :min_w] - upsampled[:min_h, :min_w])
    
    def compute_color(self, pyramid):
        """计算颜色特征"""
        features = []
        for img in pyramid:
            if len(img.shape) == 3:
                R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
                R = R / (R + G + B + 1e-8)
                G = G / (R + G + B + 1e-8)
                B = B / (R + G + B + 1e-8)
                # Red-Green and Yellow-Blue opponent channels
                RG = R - G
                BY = (R + G) / 2 - B
                features.extend([R, G, B, RG, BY])
        return features
    
    def compute_intensity(self, pyramid):
        """计算亮度特征"""
        return [np.mean(img, axis=2) for img in pyramid]
    
    def compute_orientation(self, pyramid):
        """计算方向特征"""
        features = []
        for img in pyramid:
            gray = np.mean(img, axis=2)
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel(
                    (21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(gray, -1, kernel)
                features.append(np.abs(filtered))
        return features
    
    def normalize(self, feature_map):
        """归一化特征图"""
        fmin = feature_map.min()
        fmax = feature_map.max()
        if fmax - fmin < 1e-8:
            return feature_map
        return (feature_map - fmin) / (fmax - fmin)
    
    def compute_saliency(self, image):
        """计算显著性图"""
        # 构建金字塔
        pyramid = self.gaussian_pyramid(image)
        
        # 计算特征
        color_feats = self.compute_color(pyramid)
        intensity_feats = self.compute_intensity(pyramid)
        orient_feats = self.compute_orientation(pyramid)
        
        all_features = color_feats + intensity_feats + orient_feats
        
        # 计算Center-Surround差分
        cs_maps = []
        for fine_idx, fine in enumerate(pyramid):
            for coarse_idx in range(fine_idx + 1, len(pyramid)):
                coarse = pyramid[coarse_idx]
                cs = self.center_surround(fine, coarse)
                cs_maps.append(self.normalize(cs))
        
        # 跨尺度融合（简单平均）
        saliency = np.mean(cs_maps, axis=0)
        
        return self.normalize(saliency)
    
    def predict_fixation(self, saliency_map, top_k=5):
        """预测注视点"""
        # 找到前k个显著位置
        flat_idx = np.argsort(saliency_map.flatten())[-top_k:]
        h, w = saliency_map.shape
        points = [(idx % w, idx // w) for idx in flat_idx]
        return points


def demo():
    """演示ITTI模型"""
    np.random.seed(42)
    
    # 创建测试图像：包含显著物体
    img = np.random.rand(240, 320, 3)
    
    # 添加显著红色方块
    img[40:80, 120:160, 0] = 0.9
    img[40:80, 120:160, 1] = 0.1
    img[40:80, 120:160, 2] = 0.1
    
    # 添加不显著蓝色区域
    img[150:180, 200:240, 0] = 0.1
    img[150:180, 200:240, 1] = 0.1
    img[150:180, 200:240, 2] = 0.9
    
    # 创建模型并计算显著性
    model = ITIISaliencyModel()
    saliency = model.compute_saliency(img)
    
    # 预测注视点
    fixation_points = model.predict_fixation(saliency)
    
    print("ITTI显著性检测结果:")
    print(f"  显著性图尺寸: {saliency.shape}")
    print(f"  预测注视点: {fixation_points[-1]}")
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('输入图像')
    axes[0].axis('off')
    
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title('显著性图')
    axes[1].axis('off')
    
    # 标记显著点
    x, y = fixation_points[-1]
    axes[1].plot(x, y, 'c*', markersize=15)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""
ITTI模型的纯NumPy实现（简化版）
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class SimpleITTI:
    """简化版ITTI显著性模型"""
    
    def __init__(self):
        self.sigma_center = 1.0
        self.sigma_surround = 3.0
    
    def compute_features(self, image):
        """计算基础特征"""
        # 亮度
        intensity = np.mean(image, axis=2)
        
        # 颜色（RGB通道差异）
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        color_odd = np.abs(R - G) + np.abs((R+G)/2 - B)
        
        # 简单方向（水平/垂直边缘）
        grad_x = gaussian_filter(intensity, self.sigma_center) - gaussian_filter(intensity, self.sigma_surround)
        grad_y = gaussian_filter(intensity, self.sigma_center) - gaussian_filter(intensity, self.sigma_surround)
        orientation = np.abs(grad_x) + np.abs(grad_y)
        
        return intensity, color_odd, orientation
    
    def compute_saliency(self, image):
        """计算显著性"""
        intensity, color_odd, orientation = self.compute_features(image)
        
        # 归一化
        for feat in [intensity, color_odd, orientation]:
            feat -= feat.min()
            if feat.max() > 0:
                feat /= feat.max()
        
        # 融合
        saliency = (intensity + color_odd + orientation) / 3
        
        return saliency


if __name__ == "__main__":
    np.random.seed(42)
    img = np.random.rand(100, 100, 3)
    
    model = SimpleITTI()
    sal = model.compute_saliency(img)
    print(f"显著性范围: [{sal.min():.3f}, {sal.max():.3f}]")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_multiscale_features(features, save_path=None):
    """可视化多尺度特征"""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    for i, feat in enumerate(features):
        axes[i].imshow(feat, cmap='gray')
        axes[i].set_title(f'尺度 {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_saliency_map(image, saliency, save_path=None):
    """可视化显著性检测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title('显著性图')
    axes[1].axis('off')
    
    # 叠加显示
    axes[2].imshow(image * 0.5 + plt.cm.hot(saliency)[:,:,:3] * 0.5)
    axes[2].set_title('显著区域叠加')
    axes[2].axis('off')
    
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
| AUC | ROC曲线下面积，显著性检测的判别能力 |
| CC | 与眼动数据的相关系数 |
| SIM | 与人类显著性图的相似度 |

### 10.2 计算代码

```python
from sklearn.metrics import roc_auc_score

def evaluate_saliency(pred_saliency, gt_saliency, gt_fixations):
    """评估显著性模型"""
    # AUC
    auc = roc_auc_score(gt_saliency.flatten(), pred_saliency.flatten())
    
    # 相关系数
    cc = np.corrcoef(pred_saliency.flatten(), gt_saliency.flatten())[0, 1]
    
    return {'auc': auc, 'correlation': cc}
```

---

## 11. 常见问题与易错点

1. **特征通道选择不当：** 不同场景可能适合不同特征组合
2. **尺度选择问题：** 中心-周围尺度差需要根据图像内容调整
3. **融合权重主观：** 简单平均可能不是最优的融合方式
4. **缺乏语义理解：** 无法识别高语义显著区域（如人脸）

---

## 12. 学习总结

ITTI模型是视觉显著性检测领域的开创性工作。它模拟人类视觉系统的早期加工机制，通过多尺度特征提取和中心-周围差分计算来识别视觉场景中的显著区域。

核心思想：
1. 人类视觉注意由低层特征驱动的"异类检测"
2. 多尺度处理捕获不同大小的显著物体
3. 颜色、亮度、方向三个特征通道互补

局限性：
- 仅使用低层特征，缺乏语义理解
- 手工设计特征和参数，工作量大
- 在复杂自然场景中效果有限

现代显著性检测已转向深度学习方法，但ITTI模型作为该领域的奠基工作仍具有重要学习和参考价值。

---

## 13. 练习题与思考题

### 基础题

**题目1：** 解释"中心-周围"（Center-Surround）机制在ITTI模型中的作用。

**答案：** 中心-周围机制模拟了视觉感受野的特性——视野中心区域的响应与周围区域的响应差异。当中心区域与周围区域差异大时（如同色背景上的异色物体），会产生强激活，指示潜在的显著位置。

### 进阶题

**题目2：** 比较ITTI模型与深度学习显著性模型的优缺点。

**答案：** ITTI：优点（可解释、无需训练、生物启发）；缺点（特征有限、对复杂场景效果差）。深度学习：优点（语义理解强、鲁棒性好）；缺点（需要大量标注数据、可解释性差）。

---

## 14. 学习路径建议

**前置算法：**
- 奈塞尔两阶段理论
- 高斯金字塔原理

**平行算法：**
- 其他显著性模型：GBVS, AWS, SUN

**进阶算法：**
- 深度学习显著性检测：DeepFix, SALICON
- 注意力机制在目标检测中的应用