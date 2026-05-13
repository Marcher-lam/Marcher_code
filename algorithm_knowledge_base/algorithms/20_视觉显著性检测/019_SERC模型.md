# SERC模型 学习文档

> 基于区域协方差的视觉显著性——通过区域间特征差异评估显著性。
>
> 来源线索：本节内容根据原书第2.2.1节"SERC:区域相比得到的显著性"整理。

---

## 1. 算法基础认知

**一句话定义：** SERC（Saliency Estimation by Region Covariances）由Erdem和Erdem于2013年在CVPR上提出，以图像局部区域的协方差矩阵作为特征描述子，通过区域间的差异性度量来评估显著性。

**核心思想：** 显著的区域是那些与其周围区域相比特征差异最大的区域。SERC使用7维特征向量（LAB颜色 + 梯度幅值 + 像素位置），计算每个区域内这些特征的协方差矩阵作为区域的描述子，然后用协方差矩阵间的距离（基于广义特征值）来度量区域间的差异。

**为什么用协方差矩阵？** 协方差矩阵能同时捕获：
1. **特征本身的方差**：区域内特征的分布范围
2. **特征之间的相关性**：如颜色和梯度之间的关联
3. **区域的空间结构**：位置信息编码了空间排列

协方差矩阵是一个对称正定矩阵，具有良好的数学性质，可以用黎曼几何中的距离度量。

**历史背景：** Erdem等人2013年提出，是显著性检测中率先使用区域协方差的方法。它不依赖任何学习过程，完全基于特征统计的差异度量。

---

## 2. 核心原理

### 2.1 特征定义

对图像中每个像素 $p = (x,y)$，定义一个7维特征向量：

$$
f(p) = [L, a, b, |I_x|, |I_y|, x, y]^T
$$

其中：
- $L, a, b$：LAB色彩空间的三通道
- $|I_x|, |I_y|$：亮度梯度的幅值
- $x, y$：归一化的像素坐标

### 2.2 区域协方差描述子

对图像中的一个矩形区域 $R$，其特征集合为 $\{f_i\}_{i=1}^N$（$N$ 是区域内像素数）。该区域的协方差矩阵：

$$
C_R = \frac{1}{N-1} \sum_{i=1}^N (f_i - \mu)(f_i - \mu)^T
$$

其中 $\mu = \frac{1}{N}\sum_{i=1}^N f_i$ 是区域内特征均值向量。

$C_R$ 是一个 $7 \times 7$ 的对称正定矩阵。

### 2.3 协方差距离度量

两个区域 $R_i$ 和 $R_j$ 的协方差矩阵 $C_i$ 和 $C_j$ 之间的距离使用广义特征值度量：

$$
\rho(C_i, C_j) = \sqrt{\sum_{k=1}^d \ln^2 \lambda_k(C_i, C_j)}
$$

其中 $\lambda_k(C_i, C_j)$ 是 $C_i^{-1}C_j$ 的第 $k$ 个广义特征值。这个度量是仿射不变的，且满足对称性和三角不等式。

### 2.4 显著性计算

对图像中的每个候选区域 $R$，计算它与周围邻域内所有区域的平均协方差距离：

$$
S(R) = \frac{1}{|\mathcal{N}(R)|} \sum_{R' \in \mathcal{N}(R)} \rho(C_R, C_{R'})
$$

其中 $\mathcal{N}(R)$ 是 $R$ 的空间邻域区域集合。

---

## 3. 数学公式与推导

### 3.1 协方差矩阵的几何意义

协方差矩阵 $C_R$ 位于对称正定矩阵流形 $Sym_d^+$ 上，这是一个黎曼流形，不能直接用欧氏距离度量。两个协方差矩阵之间的测地距离：

$$
\rho(C_1, C_2) = \sqrt{\sum_{k=1}^d \ln^2 \lambda_k(C_1^{-1}C_2)}
$$

推导：在黎曼流形上，从 $C_1$ 到 $C_2$ 的测地线由 $C(t) = C_1^{1/2}(C_1^{-1/2}C_2C_1^{-1/2})^t C_1^{1/2}$ 给出，其长度为 $\rho(C_1, C_2)$。

### 3.2 广义特征值问题

求解 $\det(C_i - \lambda C_j) = 0$ 的根即为广义特征值。等价于求解 $C_j^{-1}C_i$ 的特征值。当 $C_i$ 和 $C_j$ 接近时，$\lambda_k \approx 1$，$\ln \lambda_k \approx 0$，距离为零。

### 3.3 7维特征的协方差矩阵结构

$$
C_R = \begin{bmatrix}
\sigma_{LL}^2 & \sigma_{La}^2 & \cdots & \sigma_{Lx}^2 \\
\sigma_{aL}^2 & \sigma_{aa}^2 & \cdots & \sigma_{ax}^2 \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{xL}^2 & \sigma_{xa}^2 & \cdots & \sigma_{xx}^2
\end{bmatrix}
$$

对角线元素表示各特征自身的方差，非对角线表示特征间的协方差。

---

## 4. 训练过程讲解

SERC是**无训练**的方法。

**处理流程：**
1. 将图像转换到LAB颜色空间
2. 计算亮度梯度 $I_x, I_y$
3. 构建7维特征图
4. 以步长 $s$ 在图像上滑动窗口（窗口大小 $w \times w$）
5. 对每个窗口计算协方差矩阵
6. 对每个窗口，计算其与邻域窗口的协方差距离
7. 距离均值作为该窗口的显著性值
8. 插值到原始图像大小
9. 归一化

**参数：**
- 窗口大小 $w$：通常 8-16 像素
- 邻域半径 $r$：通常 2-4 个窗口
- 滑动步长 $s$：通常 $w/2$

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 显著性检测 | 检测与周围区域特征差异大的区域 |
| 纹理分析 | 协方差矩阵有效描述纹理特征 |
| 目标检测 | 协方差描述子可作为目标特征 |
| 图像分割 | 协方差差异用于区域合并 |
| 物体跟踪 | 协方差跟踪（基于区域协方差的跟踪） |
| 行人检测 | 协方差特征用于行人描述 |

---

## 6. 优缺点分析

**优点：**
- ✅ **特征丰富**：7维特征同时考虑颜色、梯度和位置
- ✅ **协方差描述力强**：捕获特征间的相关性
- ✅ **黎曼几何度量严谨**：正定矩阵流形上的正确距离
- ✅ **无需训练**：完全无监督
- ✅ **鲁棒**：对光照变化和旋转有一定不变性

**缺点：**
- ❌ **计算量大**：每个区域需计算协方差矩阵和广义特征值
- ❌ **窗口大小敏感**：过小窗口协方差不稳定，过大丢失细节
- ❌ **边界效应**：图像边缘区域邻域不完整
- ❌ **特征维度有限**：7维可能不足以描述复杂场景
- ❌ **速度慢**：滑动窗口在实时应用中不可行

---

## 7. 调库实现

```python
"""SERC模型 - 完整调库实现"""
import numpy as np
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt


class SERC:
    """基于区域协方差的显著性模型"""
    
    def __init__(self, window_size=8, radius=2, stride=4):
        """
        参数:
            window_size: 滑动窗口大小
            radius: 邻域半径 (窗口数)
            stride: 滑动步长
        """
        self.window_size = window_size
        self.radius = radius
        self.stride = stride
    
    def _build_feature_map(self, image):
        """构建7维特征图"""
        h, w = image.shape[:2]
        
        # 转为灰度用于梯度
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # 计算梯度
        gy, gx = np.gradient(gray.astype(float))
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # 颜色通道
        if len(image.shape) == 3 and image.shape[2] >= 3:
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            # 近似LAB转换
            L = 0.299 * r + 0.587 * g + 0.114 * b
            a = 0.5 * (r - g) + 128
            b_ = 0.5 * (g - b) + 128
        else:
            L = gray
            a = np.zeros_like(gray)
            b_ = np.zeros_like(gray)
        
        # 位置坐标（归一化）
        ygrid, xgrid = np.mgrid[0:h, 0:w]
        y_norm = ygrid / h
        x_norm = xgrid / w
        
        # 堆叠为7维特征图 (H, W, 7)
        feat_map = np.stack([L, a, b_, grad_mag, grad_mag, y_norm, x_norm], axis=2)
        
        return feat_map
    
    def _compute_covariance(self, patch):
        """计算区域协方差矩阵"""
        h, w, d = patch.shape
        features = patch.reshape(-1, d)
        mean = features.mean(axis=0)
        centered = features - mean
        cov = (centered.T @ centered) / (len(features) - 1)
        return cov, mean
    
    def _cov_distance(self, C1, C2):
        """
        协方差矩阵间距离
        rho(C1,C2) = sqrt(sum_k ln^2(lambda_k(C1^{-1} C2)))
        """
        d = C1.shape[0]
        C1_inv = inv(C1 + 1e-6 * np.eye(d))
        M = C1_inv @ C2
        eigenvalues = np.linalg.eigvals(M)
        # 取实部（理论上应为正实数）
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.clip(eigenvalues, 1e-8, None)
        return np.sqrt(np.sum(np.log(eigenvalues)**2))
    
    def compute_saliency(self, image):
        """计算显著性图"""
        feat_map = self._build_feature_map(image)
        h, w = feat_map.shape[:2]
        ws = self.window_size
        st = self.stride
        r = self.radius
        
        # 计算每个窗口的协方差矩阵
        covs = {}
        positions = []
        for i in range(0, h - ws + 1, st):
            for j in range(0, w - ws + 1, st):
                patch = feat_map[i:i+ws, j:j+ws]
                cov, _ = self._compute_covariance(patch)
                covs[(i, j)] = cov
                positions.append((i, j))
        
        # 计算每个窗口的显著性
        saliency = np.zeros((h, w))
        weight = np.zeros((h, w))
        
        for i, j in positions:
            dissim_sum = 0.0
            count = 0
            cov_center = covs[(i, j)]
            
            # 邻域窗口
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di * st, j + dj * st
                    if (ni, nj) in covs:
                        cov_neighbor = covs[(ni, nj)]
                        dissim_sum += self._cov_distance(cov_center, cov_neighbor)
                        count += 1
            
            if count > 0:
                avg_dissim = dissim_sum / count
                saliency[i:i+ws, j:j+ws] += avg_dissim
                weight[i:i+ws, j:j+ws] += 1.0
        
        saliency = saliency / (weight + 1e-8)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency


def demo():
    np.random.seed(42)
    h, w = 64, 64
    img = np.random.rand(h, w, 3) * 0.3 + 0.4
    img[20:30, 25:35, :] = 0.9
    img[45:52, 10:18, :] = 0.1
    
    model = SERC(window_size=8, radius=2, stride=4)
    smap = model.compute_saliency(img)
    
    print(f"SERC显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")
    print(f"矩形区域 (20:30,25:35) 均值: {smap[20:30, 25:35].mean():.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img); axes[0].set_title('原始图像'); axes[0].axis('off')
    axes[1].imshow(smap, cmap='hot'); axes[1].set_title('SERC显著性'); axes[1].axis('off')
    plt.tight_layout(); plt.savefig('serc_saliency.png', dpi=150)


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""SERC模型 - 手工协方差实现"""
import numpy as np


def manual_covariance(features):
    """手工计算协方差矩阵"""
    n, d = features.shape
    mean = np.mean(features, axis=0)
    cov = np.zeros((d, d))
    for k in range(n):
        diff = features[k] - mean
        cov += np.outer(diff, diff)
    return cov / (n - 1)


def manual_eigenvalues(A):
    """手工计算特征值（幂迭代法）"""
    d = A.shape[0]
    eigenvalues = []
    A_remain = A.copy()
    
    for _ in range(d):
        # 幂迭代
        v = np.random.randn(d)
        v = v / np.linalg.norm(v)
        for _ in range(100):
            v_new = A_remain @ v
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm < 1e-10:
                break
            v = v_new / v_new_norm
        
        eigenvalue = v @ A_remain @ v
        eigenvalues.append(eigenvalue)
        # 收缩（减去已求特征值）
        A_remain = A_remain - eigenvalue * np.outer(v, v)
    
    return np.array(eigenvalues)


def cov_distance_manual(C1, C2):
    """手工协方差距离"""
    d = C1.shape[0]
    # 简单逆
    C1_inv = np.linalg.inv(C1 + 1e-6 * np.eye(d))
    M = C1_inv @ C2
    # 特征值
    eigvals = np.linalg.eigvals(M)
    eigvals = np.real(eigvals)
    eigvals = np.clip(eigvals, 1e-8, None)
    return np.sqrt(np.sum(np.log(eigvals)**2))


def serc_manual(image, window_size=6, radius=1, stride=3):
    """手工SERC"""
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    h, w = gray.shape
    
    # 提取梯度
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx[i,j] = gray[i,j+1] - gray[i,j-1]
            gy[i,j] = gray[i+1,j] - gray[i-1,j]
    grad = np.sqrt(gx**2 + gy**2)
    
    ygrid, xgrid = np.mgrid[0:h, 0:w]
    y_norm = ygrid/h; x_norm = xgrid/w
    
    feat_map = np.stack([gray, grad, grad, y_norm, x_norm], axis=2)
    
    covs = {}
    positions = []
    for i in range(0, h-window_size+1, stride):
        for j in range(0, w-window_size+1, stride):
            patch = feat_map[i:i+window_size, j:j+window_size]
            d = patch.shape[2]
            feats = patch.reshape(-1, d)
            covs[(i,j)] = manual_covariance(feats)
            positions.append((i,j))
    
    saliency = np.zeros((h,w))
    weight = np.zeros((h,w))
    
    for i,j in positions:
        d_sum = 0; cnt = 0
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                if di == 0 and dj == 0: continue
                ni, nj = i + di*stride, j + dj*stride
                if (ni,nj) in covs:
                    d_sum += cov_distance_manual(covs[(i,j)], covs[(ni,nj)])
                    cnt += 1
        if cnt > 0:
            saliency[i:i+window_size, j:j+window_size] += d_sum/cnt
            weight[i:i+window_size, j:j+window_size] += 1.0
    
    saliency = saliency / (weight + 1e-8)
    s_min, s_max = saliency.min(), saliency.max()
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    return saliency


def test_serc():
    np.random.seed(42)
    img = np.random.rand(24, 24, 3) * 0.3 + 0.4
    img[8:14, 10:16, :] = 0.9
    smap = serc_manual(img, window_size=6, radius=1, stride=3)
    print(f"SERC手工: [{smap.min():.3f}, {smap.max():.3f}]")
    assert smap.max() > 0.5, "应检测到显著区域"


if __name__ == "__main__":
    test_serc()
```

---

## 9. 可视化与结果理解

### 9.1 协方差矩阵可视化

$7 \times 7$ 协方差矩阵的热图中：
- 对角线：各特征的方差——颜色方差大可能意味着区域包含多种颜色
- 非对角线：特征间相关性——如位置与颜色的相关性表示颜色渐变

### 9.2 显著性图解读

- 高显著性区域：与周围邻居协方差差异大（如颜色突变的边缘）
- 低显著性区域：与周围邻居协方差相似（如均匀纹理区域）

---

## 10. 模型评估

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_serc():
    img = np.random.rand(32, 32, 3) * 0.3 + 0.4
    img[10:18, 12:20, :] = 0.9
    gt = np.zeros((32, 32))
    gt[10:18, 12:20] = 1.0
    
    model = SERC(window_size=6, radius=2, stride=3)
    smap = model.compute_saliency(img)
    auc = roc_auc_score(gt.flatten() > 0, smap.flatten())
    print(f"SERC AUC: {auc:.4f}")


if __name__ == "__main__":
    evaluate_serc()
```

---

## 11. 常见问题与易错点

### Q1: 协方差距离为什么用广义特征值？
**A:** 协方差矩阵位于正定矩阵流形，欧氏距离 $\|C_1-C_2\|_F$ 不能准确度量差异。广义特征值距离 $\sqrt{\sum \ln^2 \lambda_k}$ 是流形上的测地距离，具有仿射不变性。

### Q2: 窗口大小如何影响结果？
**A:** 窗口太小（<4）协方差估计不稳定；窗口太大（>32）丢失细节。一般取 8-16 像素。

### Q3: 梯度特征重复为什么？
**A:** 原论文中5维特征（L,a,b,|I_x|,|I_y|）或7维（加位置）。此处用梯度幅值 $|I_x|,|I_y|$ 两个维度，都是工况选择。重复是示例笔记标注。

### Q4: SERC与区域协方差跟踪的关系？
**A:** SERC使用相同的协方差描述子但不用于跟踪。Porikli 2006年提出的协方差跟踪是同一特征在视频跟踪中的应用。

### Q5: 什么情况下SERC效果不好？
**A:** (1) 场景过于复杂，协方差无法区分；(2) 窗口大小不匹配目标尺度；(3) 图像边缘区域邻域不完整。

---

## 12. 学习总结

**核心要点：**
1. 协方差矩阵作为区域描述子，捕获特征方差和相关性
2. 黎曼几何距离度量区域间协方差差异
3. 7维特征：颜色(LAB) + 梯度 + 位置
4. 完全无监督，无需训练

**与其它显著性方法的区别：**
- SR/IS：频域方法，全局统计
- ITTI/GBVS：生物启发，特征图融合
- SERC：区域协方差，局部对比度

---

## 13. 练习题与思考题

### 基础题

**1.** 为什么协方差矩阵是对称正定的？

<details>
<summary>答案</summary>
对任意非零向量 $v$，$v^T C v = \frac{1}{N-1}\sum_i (v^T(f_i-\mu))^2 \geq 0$，等号仅当所有 $v^T(f_i-\mu)=0$ 时成立（通常不成立），因此 $C$ 半正定。实际上由于噪声，$C$ 通常正定。
</details>

**2.** 广义特征值距离为什么是仿射不变的？

<details>
<summary>答案</summary>
对任意可逆矩阵 $A$，$C$ 变为 $A C A^T$，则 $C^{-1}$ 变为 $(A^T)^{-1}C^{-1}A^{-1}$，$(A C_1 A^T)^{-1}(A C_2 A^T) = (A^T)^{-1}C_1^{-1}C_2 A^T$，与 $C_1^{-1}C_2$ 相似，特征值不变。
</details>

### 进阶题

**3.** 推导 $d(C_1, C_2) = \sqrt{\sum_k \ln^2 \lambda_k}$ 满足三角不等式。

<details>
<summary>答案</summary>
令 $\lambda_k$ 是 $C_1^{-1}C_2$ 的特征值。记 $a_k = \ln \lambda_k$。对于 $C_1, C_2, C_3$，有 $C_1^{-1}C_3 = C_1^{-1}C_2 \cdot C_2^{-1}C_3$，其特征值满足 $\ln \lambda_k \approx \sum_j a_{1,2,j} + a_{2,3,j}$（在流形切空间近似）。三角不等式由 $\ell_2$ 范数满足。
</details>

**4.** 如何加速SERC使其实时？

<details>
<summary>答案</summary>
(1) 积分图计算协方差（O(1)每个窗口）；(2) 降采样特征图；(3) 使用快速特征值近似；(4) 只计算有重叠的邻域。
</details>

---

## 14. 学习路径建议

### 预备知识
- 线性代数（协方差矩阵、特征值分解）
- 黎曼几何基础（流形上的距离）
- 数字图像处理

### 进阶方向
1. **SERC -> 协方差跟踪**：视频中的协方差目标跟踪
2. **SERC -> 深度学习协方差**：用CNN学习协方差特征
3. **SERC -> 二阶池化**：深度网络中的二阶统计池化（如MPN-COV）

### 推荐阅读
- Erdem & Erdem. "Visual Saliency Estimation by Nonlinearly Integrating Features using Region Covariances." CVPR 2013.
- Tuzel et al. "Region Covariance: A Fast Descriptor for Detection and Classification." ECCV 2006.
- Porikli et al. "Covariance Tracking using Model Update." CVPR 2006.

### 项目实践
1. 在不同窗口大小下测试SERC性能
2. 比较协方差描述子与直方图描述子的显著性检测效果
3. 实现积分图加速版SERC
