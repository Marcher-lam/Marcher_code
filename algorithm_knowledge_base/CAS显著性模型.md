# CAS显著性模型 学习文档

> 上下文感知显著性——结合全局上下文和局部对比度计算显著性。

> 来源线索：本节内容根据原书第2章关于"显著物体检测"的相关章节整理。

## 1. 算法基础认知

**一句话定义：** CAS（Context-Aware Saliency）由Goferman等人于2012年提出，结合局部低层特征（颜色、对比度）和全局上下文信息（罕见性、组织性）来检测视觉显著性。

**直觉类比：** 想象你在一个拥挤的派对中寻找朋友。你不仅会注意到那些穿着亮眼的人（局部对比度），还会注意到那些在人群中显得特别的人——比如唯一一个戴帽子的人（全局罕见性）。CAS正是模拟了这种"既看局部对比，又看全局上下文"的人类视觉注意力机制。

**历史背景：**
- **2012年：** Goferman等人在CVPR发表CAS模型
- **意义：** 首次系统地将"上下文信息"引入显著性检测，突破了以往仅依赖局部对比度的局限

**核心思想：** 显著性不仅取决于局部对比度，还与全局上下文相关。一个区域如果在上下文中罕见、具有组织性、靠近视场中心，则更可能被注意。

**算法定位：** CAS属于**传统计算机视觉显著性检测**方法，基于手工特征而非深度学习，具有较好的可解释性。

---

## 2. 核心原理

### 2.1 四个显著性因子

1. **局部对比度：** 与周围区域的颜色、纹理差异。如果一个区域与其邻域明显不同，则该区域更显著。
2. **全局罕见性：** 在整幅图像中出现频率低的特征更显著。比如在一片绿色草地上的一朵红花。
3. **视觉组织性：** 符合格式塔原则（闭合、对称、连续性）的区域更可能被注意。人眼倾向于将组织良好的区域视为一个整体。
4. **中心先验：** 靠近图像中心的区域更显著。这是对摄影师构图习惯的统计观察——重要物体通常被放置在画面中心附近。

### 2.2 工作流程

```
输入图像 → 分块处理 → 局部对比度计算 → 全局罕见性计算
→ 组织性评估 → 中心先验加权 → 多因子融合 → 显著性图
```

### 2.3 为什么需要上下文信息？

单纯的局部对比度方法容易将"高频纹理区域"误判为显著（例如沙地、草地纹理）。加入全局上下文后，这些纹理区域虽然局部对比度高，但在全局范围内出现频繁，因此"罕见性"得分低，不会被误判为显著。

---

## 3. 数学公式与推导

### 3.1 局部对比度

对于图像块 $p_i$，局部对比度定义为：

$$S_{\text{local}}(p_i) = \frac{1}{K} \sum_{j \in \mathcal{N}_K(p_i)} d(p_i, p_j)$$

其中 $\mathcal{N}_K(p_i)$ 是 $p_i$ 的 $K$ 个最相似邻域块，$d(\cdot,\cdot)$ 是特征空间中的欧氏距离。

### 3.2 全局罕见性

全局罕见性衡量一个图像块在整个图像中出现的频率：

$$S_{\text{global}}(p_i) = \sum_{j=1}^N \exp\left(-\frac{d^2(p_i, p_j)}{2\sigma^2}\right)$$

该值越小，说明该块越罕见（与多数块差异大）。

### 3.3 中心先验

$$S_{\text{center}}(p_i) = 1 - \frac{\|c_i - c_0\|}{\max_j \|c_j - c_0\|}$$

其中 $c_i$ 是块 $p_i$ 的中心坐标，$c_0$ 是图像中心坐标。

### 3.4 总体显著性

四个因子加权融合：

$$S(p_i) = \sum_{k=1}^4 w_k \cdot S_k(p_i)$$

其中 $w_k$ 是各因子的权重，在原文中通过经验确定。

### 3.5 多尺度融合

为了鲁棒地处理不同尺度的显著物体，CAS在多个尺度下分别计算显著性，然后取平均：

$$S_{\text{final}}(p_i) = \frac{1}{M} \sum_{m=1}^M S^{(m)}(p_i)$$

---

## 4. 训练过程讲解

CAS不是基于学习的模型，而是基于先验知识的计算模型，因此**不需要训练**。它直接通过以下步骤计算显著性：

1. **图像预处理：** 将输入图像缩放到统一尺寸
2. **分块采样：** 使用滑窗或随机采样提取图像块，步长为块大小的一半以保证重叠覆盖
3. **特征提取：** 每个块提取颜色均值、颜色标准差、方向直方图等特征
4. **距离计算：** 计算所有块对之间的余弦距离或欧氏距离
5. **因子计算：** 分别计算局部对比度、全局罕见性和中心先验
6. **融合输出：** 加权融合各因子，输出显著性图

---

## 5. 应用场景

1. **图像裁剪与缩略图生成：** 自动识别图像中最显著的区域，生成有信息量的缩略图
2. **图像压缩：** 对显著区域分配更高的编码质量，非显著区域降低质量以节省空间
3. **内容感知图像编辑：** 在图像重定向（seam carving）中保护显著区域不变形
4. **图像检索：** 基于显著性区域提取特征，提高检索精度
5. **广告设计：** 分析用户最可能关注的位置，优化广告布局

---

## 6. 优缺点分析

### 优点
1. **无需训练数据：** 完全基于先验知识，零样本可用
2. **可解释性强：** 每个显著性因子都有明确的物理含义
3. **上下文感知：** 能区分高频纹理和真正显著的物体
4. **多尺度鲁棒：** 多尺度处理适应不同大小的物体

### 缺点
1. **计算效率低：** 需要计算所有块对间的距离，复杂度 $O(N^2)$
2. **特征表达有限：** 仅使用低层视觉特征，缺乏语义理解
3. **权重调参困难：** 四个因子的权重需要手动调整
4. **边缘模糊：** 基于块的输出导致显著性图边缘不锐利
5. **已被深度方法超越：** 深度学习方法的性能远超此类传统方法

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
"""
CAS（上下文感知显著性）的完整Python实现
依赖：numpy, cv2, matplotlib
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt


class CASSaliency:
    """上下文感知显著性模型
    
    通过局部对比度、全局罕见性和中心先验的融合计算显著性。
    
    参数:
        block_size: 图像块大小（像素）
        n_neighbors: 局部对比度使用的近邻数量
        n_scales: 多尺度数量
    """
    
    def __init__(self, block_size=8, n_neighbors=3, n_scales=3):
        self.block_size = block_size
        self.n_neighbors = n_neighbors
        self.n_scales = n_scales
        self.weights = [0.4, 0.3, 0.3]  # 局部、全局、中心先验权重
    
    def extract_blocks(self, image, block_size=None):
        """将图像分割为重叠的图像块，并提取特征
        
        参数:
            image: HxWx3 的RGB图像
            block_size: 块大小
            
        返回:
            blocks: NxD 的特征矩阵
            positions: Nx2 的块位置矩阵
        """
        if block_size is None:
            block_size = self.block_size
        
        h, w = image.shape[:2]
        blocks = []
        positions = []
        stride = block_size // 2  # 50%重叠
        
        for i in range(0, h - block_size + 1, stride):
            for j in range(0, w - block_size + 1, stride):
                # 提取块
                block = image[i:i+block_size, j:j+block_size]
                
                # 特征：颜色均值(3) + 颜色标准差(3) + 方向梯度直方图(8)
                mean_color = block.mean(axis=(0, 1))  # RGB均值
                std_color = block.std(axis=(0, 1))   # RGB标准差
                
                # 简化的方向特征：水平梯度和垂直梯度均值
                gray = cv2.cvtColor(block.astype(np.float32), cv2.COLOR_RGB2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                orientation_feat = np.array([gx.mean(), gy.mean(), (gx**2 + gy**2).mean()])
                
                # 拼接所有特征
                feature = np.concatenate([mean_color, std_color, orientation_feat])
                blocks.append(feature)
                positions.append([i + block_size // 2, j + block_size // 2])
        
        return np.array(blocks), np.array(positions)
    
    def compute_center_prior(self, positions, h, w):
        """计算中心先验
        
        距离图像中心越近，值越大（越显著）。
        """
        center = np.array([h / 2, w / 2])
        center_dist = np.array([np.linalg.norm(p - center) for p in positions])
        center_prior = 1 - center_dist / np.max(center_dist + 1e-8)
        return center_prior
    
    def compute_saliency_single_scale(self, image, block_size=None):
        """在单个尺度下计算显著性
        
        参数:
            image: HxWx3 RGB图像
            
        返回:
            saliency_scores: 每个块的显著性得分
        """
        h, w = image.shape[:2]
        blocks, positions = self.extract_blocks(image, block_size)
        
        if len(blocks) == 0:
            return np.zeros((h, w))
        
        # ---- 步骤1: 计算块间距离 ----
        dists = cosine_distances(blocks)  # N x N 距离矩阵
        
        # ---- 步骤2: 局部对比度 ----
        # 取每个块的K个最近邻的平均距离
        local_saliency = np.zeros(len(blocks))
        for idx in range(len(blocks)):
            # 排除自身（距离为0），取最近的n_neighbors个
            sorted_dists = np.sort(dists[idx])
            if len(sorted_dists) > self.n_neighbors + 1:
                local_saliency[idx] = np.mean(sorted_dists[1:self.n_neighbors+1])
            else:
                local_saliency[idx] = np.mean(sorted_dists[1:])
        
        # 归一化到[0, 1]
        local_saliency = self._normalize(local_saliency)
        
        # ---- 步骤3: 全局罕见性 ----
        # 与所有块的平均距离越大 → 越罕见 → 越显著
        global_saliency = np.mean(dists, axis=1)
        global_saliency = self._normalize(global_saliency)
        
        # ---- 步骤4: 中心先验 ----
        center_prior = self.compute_center_prior(positions, h, w)
        center_prior = self._normalize(center_prior)
        
        # ---- 步骤5: 加权融合 ----
        saliency_scores = (
            self.weights[0] * local_saliency +
            self.weights[1] * global_saliency +
            self.weights[2] * center_prior
        )
        saliency_scores = self._normalize(saliency_scores)
        
        return saliency_scores, positions
    
    def compute_saliency(self, image):
        """计算CAS显著性图（多尺度）
        
        参数:
            image: HxWx3 RGB图像 (0-1范围)
            
        返回:
            saliency_map: HxW 显著性图 (0-1范围)
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        h, w = image.shape[:2]
        
        # 多尺度计算
        combined_saliency = np.zeros((h, w))
        count_map = np.zeros((h, w))
        
        for scale in range(self.n_scales):
            # 不同尺度的块大小
            block_size = self.block_size * (scale + 1)
            
            # 如果块太大，跳过
            if block_size > min(h, w) // 2:
                continue
            
            # 对图像进行下采样（模拟多尺度）
            scale_factor = 1.0 / (scale + 1)
            scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
            scaled_img = cv2.resize(image, (scaled_w, scaled_h))
            
            # 计算该尺度下的显著性
            scores, positions = self.compute_saliency_single_scale(scaled_img, block_size)
            
            # 映射回原始图像尺寸
            for score, pos in zip(scores, positions):
                # 缩放位置回原始尺寸
                orig_y = int(pos[0] / scale_factor)
                orig_x = int(pos[1] / scale_factor)
                
                # 在原始位置周围画块
                half_bs = block_size // 2
                y1 = max(0, orig_y - half_bs)
                y2 = min(h, orig_y + half_bs)
                x1 = max(0, orig_x - half_bs)
                x2 = min(w, orig_x + half_bs)
                combined_saliency[y1:y2, x1:x2] += score
                count_map[y1:y2, x1:x2] += 1
        
        # 平均归一化
        saliency_map = combined_saliency / (count_map + 1e-8)
        return self._normalize(saliency_map)
    
    def _normalize(self, arr):
        """将数组归一化到[0, 1]范围"""
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)


def create_test_image():
    """创建一个包含显著物体的测试图像"""
    img = np.ones((128, 128, 3)) * 0.3  # 灰色背景
    
    # 在中心附近添加一个红色方块（显著物体）
    img[40:70, 40:70] = [0.9, 0.1, 0.1]
    
    # 添加一些不显著的纹理（高频但全局不罕见）
    for i in range(10, 50, 5):
        for j in range(80, 120, 5):
            img[i, j] = [0.35, 0.35, 0.35]
    
    return img


def demo():
    """演示CAS显著性检测"""
    np.random.seed(42)
    img = create_test_image()
    
    model = CASSaliency(block_size=8, n_neighbors=3, n_scales=2)
    saliency_map = model.compute_saliency(img)
    
    print("=== CAS显著性模型演示 ===")
    print(f"图像尺寸: {img.shape}")
    print(f"显著性图范围: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    
    # 红色块区域vs背景
    obj_saliency = saliency_map[40:70, 40:70].mean()
    bg_saliency = saliency_map[:20, :20].mean()
    print(f"目标区域显著性: {obj_saliency:.4f}")
    print(f"背景区域显著性: {bg_saliency:.4f}")
    print(f"对比度比率: {obj_saliency / (bg_saliency + 1e-8):.2f}x")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('CAS显著性图')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('cas_saliency_demo.png', dpi=150)
    print("显著性图已保存到 cas_saliency_demo.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
"""
CAS核心算法的手工实现（不依赖scikit-learn等第三方库）
只使用numpy实现核心逻辑
"""

import numpy as np


def handcraft_cas_saliency(image, block_size=8, k_nearest=3):
    """手工实现的CAS显著性检测
    
    参数:
        image: HxWx3 numpy数组 (0-1范围)
        block_size: 图像块大小
        k_nearest: 局部对比度的近邻数
    
    返回:
        saliency_map: HxW numpy数组
    """
    h, w = image.shape[:2]
    stride = block_size // 2
    
    # ---- 步骤1: 分块 ----
    blocks = []
    positions = []
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            patch = image[i:i+block_size, j:j+block_size]
            # 特征：RGB均值 + RGB标准差
            feat = np.concatenate([patch.mean(axis=(0,1)), patch.std(axis=(0,1))])
            blocks.append(feat)
            positions.append((i + block_size//2, j + block_size//2))
    
    n_blocks = len(blocks)
    if n_blocks == 0:
        return np.zeros((h, w))
    
    blocks = np.array(blocks)
    
    # ---- 步骤2: 手工计算距离矩阵 ----
    dist_matrix = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(i+1, n_blocks):
            # 余弦距离: 1 - cos(θ)
            dot = np.dot(blocks[i], blocks[j])
            norm_i = np.linalg.norm(blocks[i])
            norm_j = np.linalg.norm(blocks[j])
            cos_sim = dot / (norm_i * norm_j + 1e-8)
            dist = 1.0 - cos_sim
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # ---- 步骤3: 局部对比度 ----
    local_scores = np.zeros(n_blocks)
    for i in range(n_blocks):
        # 取最近邻距离的平均（排除自身）
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # 排除自身
        nearest_dists = np.sort(dists)[:k_nearest]
        local_scores[i] = np.mean(nearest_dists)
    
    # ---- 步骤4: 全局罕见性 ----
    global_scores = np.mean(dist_matrix, axis=1)
    
    # ---- 步骤5: 中心先验 ----
    center_y, center_x = h / 2, w / 2
    center_scores = np.zeros(n_blocks)
    for idx, (py, px) in enumerate(positions):
        dist_to_center = np.sqrt((py - center_y)**2 + (px - center_x)**2)
        max_dist = np.sqrt((h/2)**2 + (w/2)**2)
        center_scores[idx] = 1 - dist_to_center / max_dist
    
    # ---- 步骤6: 融合 ----
    # 归一化每个因子
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    local_scores = normalize(local_scores)
    global_scores = normalize(global_scores)
    center_scores = normalize(center_scores)
    
    # 加权融合
    total_scores = (0.4 * local_scores + 0.3 * global_scores + 0.3 * center_scores)
    total_scores = normalize(total_scores)
    
    # ---- 步骤7: 重建显著性图 ----
    saliency_map = np.zeros((h, w))
    count_map = np.zeros((h, w))
    
    for idx, (py, px) in enumerate(positions):
        y1 = max(0, py - block_size//2)
        y2 = min(h, py + block_size//2)
        x1 = max(0, px - block_size//2)
        x2 = min(w, px + block_size//2)
        saliency_map[y1:y2, x1:x2] += total_scores[idx]
        count_map[y1:y2, x1:x2] += 1
    
    saliency_map = saliency_map / (count_map + 1e-8)
    return normalize(saliency_map)


def test_handcraft():
    """测试手工实现"""
    np.random.seed(42)
    
    # 创建简单测试图
    img = np.ones((64, 64, 3)) * 0.2
    img[20:40, 20:40] = [0.8, 0.2, 0.2]  # 红色显著区域
    
    smap = handcraft_cas_saliency(img, block_size=8, k_nearest=3)
    
    obj_val = smap[20:40, 20:40].mean()
    bg_val = smap[:15, :15].mean()
    
    print("=== 手工实现测试 ===")
    print(f"显著区域: {obj_val:.4f}, 背景: {bg_val:.4f}")
    print(f"检测成功!" if obj_val > bg_val else "检测失败!")
    return smap


if __name__ == "__main__":
    test_handcraft()
```

---

## 9. 可视化与结果理解

```python
"""
CAS显著性检测的可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_cas_results(image, saliency_map, threshold=0.5):
    """可视化CAS显著性检测的各阶段结果
    
    参数:
        image: HxWx3 原始图像
        saliency_map: HxW 显著性图
        threshold: 二值化阈值
    """
    binary = (saliency_map > threshold).astype(np.float32)
    
    # 生成热力图叠加
    heatmap = plt.cm.jet(saliency_map)[:, :, :3]
    overlay = 0.5 * image + 0.5 * heatmap
    overlay = overlay / overlay.max()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('(a) 原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(saliency_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('(b) CAS显著性图', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title(f'(c) 二值掩膜 (阈值={threshold})', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('(d) 热力图叠加', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('CAS显著性模型可视化分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cas_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印统计数据
    print("=== 显著性统计 ===")
    print(f"显著性值范围: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    print(f"均值: {saliency_map.mean():.4f}, 标准差: {saliency_map.std():.4f}")
    print(f"高于阈值({threshold})的像素占比: {(saliency_map > threshold).mean()*100:.1f}%")
    
    # 计算显著区域的连通性
    from scipy import ndimage
    labeled, n_objects = ndimage.label(binary)
    print(f"检测到的显著目标数: {n_objects}")
    for obj_id in range(1, n_objects + 1):
        obj_size = (labeled == obj_id).sum()
        print(f"  目标 {obj_id}: {obj_size} 像素 ({obj_size / (h*w) * 100:.1f}%)")


def plot_saliency_profile(image, saliency_map, row=None, col=None):
    """绘制显著性沿某一行的剖面图，对比原始图像强度"""
    h, w = image.shape[:2]
    gray = np.mean(image, axis=2)
    
    if row is None:
        row = h // 2
    if col is None:
        col = w // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 水平剖面
    axes[0].plot(gray[row, :], 'b-', label='图像强度（灰度）', alpha=0.7)
    axes[0].plot(saliency_map[row, :], 'r-', label='显著性值', linewidth=2)
    axes[0].set_xlabel('x 坐标 (像素)')
    axes[0].set_ylabel('值')
    axes[0].set_title(f'水平剖面 (row={row})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 垂直剖面
    axes[1].plot(gray[:, col], 'b-', label='图像强度（灰度）', alpha=0.7)
    axes[1].plot(saliency_map[:, col], 'r-', label='显著性值', linewidth=2)
    axes[1].set_xlabel('y 坐标 (像素)')
    axes[1].set_ylabel('值')
    axes[1].set_title(f'垂直剖面 (col={col})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cas_profile.png', dpi=150)
    plt.close()
    print(f"显著性剖面已保存到 cas_profile.png")


if __name__ == "__main__":
    np.random.seed(42)
    img = np.ones((128, 128, 3)) * 0.3
    img[40:70, 40:70] = [0.9, 0.1, 0.1]
    img[80:100, 80:100] = [0.1, 0.1, 0.9]
    
    from CAS显著性模型 import CASSaliency
    model = CASSaliency(block_size=8, n_neighbors=3, n_scales=2)
    smap = model.compute_saliency(img)
    
    h, w = img.shape[:2]
    visualize_cas_results(img, smap, threshold=0.5)
    plot_saliency_profile(img, smap, row=h//2, col=w//2)
```

---

## 10. 模型评估

CAS模型作为传统显著性检测方法，评估指标与深度学习显著性方法一致：

### 10.1 常用评估指标

**AUC（Area Under ROC Curve）：** 将显著性图视为二分类器（显著vs非显著），计算ROC曲线下面积。AUC越高，模型区分显著与非显著区域的能力越强。

**NSS（Normalized Scanpath Saliency）：** 衡量预测显著性图与真实眼动注视点的一致性：
$$NSS = \frac{1}{N} \sum_{i=1}^N \frac{S(p_i) - \mu_S}{\sigma_S}$$

**CC（Correlation Coefficient）：** 预测显著性图与真实显著性图之间的皮尔逊相关系数：
$$CC = \frac{\text{Cov}(S, G)}{\sigma_S \sigma_G}$$

**SIM（Similarity）：** 两个归一化直方图之间的交集：
$$SIM = \sum_i \min(S_i, G_i)$$

### 10.2 评估代码

```python
def evaluate_saliency(pred_map, gt_map):
    """计算显著性评估指标
    
    参数:
        pred_map: 预测的显著性图（HxW）
        gt_map: 真实显著性图（HxW）或注视点图
    
    返回:
        评估指标字典
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    # 确保是1D
    pred = pred_map.flatten()
    gt = gt_map.flatten()
    
    # 二值化真实图
    gt_binary = (gt > gt.mean()).astype(int)
    
    # AUC
    auc = roc_auc_score(gt_binary, pred)
    
    # CC（相关系数）
    cc = np.corrcoef(pred, gt)[0, 1]
    
    # NSS
    mean_p, std_p = pred.mean(), pred.std()
    nss_pred = (pred - mean_p) / (std_p + 1e-8)
    nss = np.mean(nss_pred[gt_binary == 1])
    
    # SIM（直方图交集）
    bins = 256
    hist_p, _ = np.histogram(pred, bins=bins, range=(0, 1), density=True)
    hist_g, _ = np.histogram(gt, bins=bins, range=(0, 1), density=True)
    sim = np.sum(np.minimum(hist_p, hist_g))
    
    return {'AUC': auc, 'CC': cc, 'NSS': nss, 'SIM': sim}


def cross_validation_demo():
    """在合成数据上评估CAS"""
    np.random.seed(42)
    
    results = []
    for trial in range(5):
        # 生成随机测试图像
        img = np.random.rand(64, 64, 3) * 0.3
        cx, cy = np.random.randint(20, 44, 2)
        img[cy:cy+20, cx:cx+20] = [0.9, 0.1, 0.1]
        
        # 真实显著性图（高斯模糊后的ground truth）
        gt = np.zeros((64, 64))
        gt[cy:cy+20, cx:cx+20] = 1
        from scipy.ndimage import gaussian_filter
        gt = gaussian_filter(gt, sigma=3)
        gt = gt / gt.max()
        
        # CAS预测
        model = CASSaliency(block_size=8, n_neighbors=3, n_scales=2)
        pred = model.compute_saliency(img)
        
        # 评估
        metrics = evaluate_saliency(pred, gt)
        results.append(metrics)
        print(f"试次 {trial+1}: AUC={metrics['AUC']:.4f}, CC={metrics['CC']:.4f}")
    
    # 平均结果
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print(f"\n平均: AUC={avg['AUC']:.4f}, CC={avg['CC']:.4f}, NSS={avg['NSS']:.4f}, SIM={avg['SIM']:.4f}")
```

---

## 11. 常见问题与易错点

**Q1: 为什么CAS的显著性图经常有块状伪影？**
因为CAS是基于图像块的算法，块之间的边界容易产生不连续。解决方法是使用更大的重叠（如75%重叠）或使用高斯加权将块投影到像素空间。

**Q2: 为什么纹理丰富但非显著的区域的局部对比度也很高？**
这是CAS引入"全局罕见性"因子的原因。高频纹理区域虽然局部对比度高，但全局出现频率也高，因此全局罕见性得分低，不会误判为显著。

**Q3: 为什么中心先验在有些图像中会失效？**
当显著物体位于图像边缘时，中心先验会压制其显著性。解决方法是自适应调整中心先验的权重，或使用多种先验的集成。

**Q4: CAS和ITTI模型的主要区别是什么？**
ITTI使用高斯金字塔的多尺度特征（颜色、亮度、方向）计算中心-周围差，而CAS使用图像块的全局对比度和上下文信息。CAS更注重全局上下文，ITTI更注重局部对比。

**Q5: 如何选择块大小？**
块大小应小于预期的显著物体尺寸。如果物体占据图像1/4面积，块大小应小于图像边长1/8。通常8×8到16×16是比较安全的选择。

---

## 12. 学习总结

- **核心贡献：** CAS首次将"上下文感知"引入显著性检测，通过局部对比度、全局罕见性、中心先验的多因子融合实现了鲁棒的显著性检测。
- **技术关键：** 基于图像块的多尺度分析 + 多因子加权融合。
- **本质缺陷：** 基于手工特征，缺乏语义理解能力，已被深度学习显著超越。
- **历史地位：** CAS代表了深度学习普及之前，传统显著性检测方法的最高水平之一，其"上下文感知"的思想后来被深度显著性模型继承。
- **延伸阅读：** 建议对比阅读ITTI模型（纯局部对比）、GBVS（基于图的传播）、以及深度学习中的显著性检测方法（如DeepFix、SalGAN）。

---

## 13. 练习题与思考题（含答案）

**基础题：**

1. CAS的四个显著性因子是什么？分别用一句话解释其物理意义。
> **答案：** (1) 局部对比度——与周围区域的差异度；(2) 全局罕见性——在全图中出现的频率；(3) 视觉组织性——符合格式塔原理的程度；(4) 中心先验——到图像中心的距离。

2. 为什么CAS需要使用多尺度处理？
> **答案：** 不同尺度的显著物体尺寸不同。小块检测小物体，大块检测大物体。多尺度融合可以同时检测不同大小的显著目标，提高鲁棒性。

3. 写出CAS的局部对比度计算公式，并解释每个符号的含义。
> **答案：** $S_{\text{local}}(p_i) = \frac{1}{K} \sum_{j \in \mathcal{N}_K(p_i)} d(p_i, p_j)$，其中 $p_i$ 是目标块，$\mathcal{N}_K(p_i)$ 是 $p_i$ 的K个最近邻块，$d(\cdot,\cdot)$ 是距离度量。

**进阶题：**

4. 如果一张图片是"白墙上的一个红色点"，CAS的四个因子分别会如何响应？
> **答案：** 局部对比度高（红点与白墙差异大）；全局罕见性高（红色在全图仅出现一次）；组织性低（单点无组织结构）；中心先验取决于点位置。总体：该红点会被检测为显著。

5. CAS模型如何改进才能处理语义级别的显著性（如"人"比"路牌"更显著）？
> **答案：** 引入深度特征（如预训练CNN特征）替代手工低层特征，使用语义分割网络获取物体类别先验，或使用注意力机制学习语义重要性的权重映射。

**编程题：**

6. 修改CAS实现，在计算局部对比度时使用高斯加权距离（距离越近的块权重越大）。
> **答案：**
```python
def compute_weighted_local_saliency(dist_matrix, positions, k_nearest=3, sigma=20.0):
    """使用高斯加权的局部对比度"""
    n_blocks = len(dist_matrix)
    scores = np.zeros(n_blocks)
    for i in range(n_blocks):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        nearest_idx = np.argsort(dists)[:k_nearest]
        nearest_dists = dists[nearest_idx]
        # 计算空间距离作为高斯权重
        spatial_dists = np.linalg.norm(
            positions[i] - positions[nearest_idx], axis=1)
        weights = np.exp(-spatial_dists**2 / (2 * sigma**2))
        scores[i] = np.average(nearest_dists, weights=weights)
    return scores
```

---

## 14. 学习路径建议

**前置知识：**
- 颜色空间（RGB、Lab）及其转换
- 图像分块与滑窗技术
- 距离度量（欧氏距离、余弦距离）
- 基本图像特征（颜色、纹理、方向）

**平行学习：**
- ITTI视觉显著性模型（了解不同范式）
- GBVS基于图的显著性（了解传播方法）
- SR谱残差模型（频域显著性方法）

**进阶方向：**
- 深度学习显著性检测（DeepFix、SalGAN、EDN）
- 视频显著性检测（增加时序维度）
- 显著性引导的目标检测（结合显著性+检测任务）
- 可解释AI中的显著性方法（Grad-CAM等）
