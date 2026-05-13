# SGMR 学习文档

> 基于流形排序的显著性检测——用图传播机制从边界先验中检测显著物体。

## 1. 算法基础认知

**一句话定义：** SGMR（Saliency Detection via Graph-Based Manifold Ranking）由Yang等人于2013年提出，利用流形排序（manifold ranking）和图传播机制，从图像边界区域的先验信息中检测显著物体。

**直觉类比：** 想象在一张照片中，通常照片的四个边都是背景（天空、地面、墙壁）。如果沿着边界区域"涂色"，然后让这个颜色像水流一样从边界向图像内部"扩散"，那些没有扩散到的区域就是"流不进去"的显著物体。SGMR正是利用了这个直觉。

**核心思想：** 图像边界区域通常为背景。以边界超像素作为背景种子（query），通过图上的流形排序传播相似度，得到每个区域属于背景的概率，反之为显著性。

**算法定位：** 传统计算机视觉显著性检测方法，基于图论和流形学习。

## 2. 核心原理

### 2.1 工作流程

```
输入图像 → SLIC超像素分割 → 构建图（超像素为节点）
→ 边界种子选择 → 流形排序传播 → 显著性图 → 优化
```

### 2.2 图的构建

每个超像素作为一个节点，节点之间的边权重通过Lab颜色空间的特征相似度定义：

$$W_{ij} = \exp\left(-\frac{\|c_i - c_j\|^2}{\sigma^2}\right)$$

其中 $c_i$ 和 $c_j$ 是超像素 $i$ 和 $j$ 的Lab颜色均值，$\sigma$ 控制相似度衰减速度。

### 2.3 边界先验

SGMR利用四种边界先验分别计算显著性，然后融合：
- **上边界先验**：上边界超像素作为背景种子
- **下边界先验**：下边界超像素作为背景种子
- **左边界先验**：/左边界超像素作为背景种子
- **右边界先验**：右边界超像素作为背景种子

每个方向独立计算显著性图，然后相乘融合（只有四个方向都认为"不可能是背景"的区域才被判定为显著）。

## 3. 数学公式与推导

### 3.1 流形排序

流形排序的目标函数：

$$f^* = \arg\min_f \frac{1}{2} \sum_{i,j} W_{ij} \left\| \frac{f_i}{\sqrt{d_{ii}}} - \frac{f_j}{\sqrt{d_{jj}}} \right\|^2 + \mu \sum_i \| f_i - y_i \|^2$$

其中：
- $W$ 是相似度矩阵（图邻接矩阵）
- $d_{ii} = \sum_j W_{ij}$ 是度矩阵
- $y$ 是查询种子标签（1=背景种子, 0=未知）
- $\mu$ 是平滑项和拟合项的平衡因子

### 3.2 解析解

目标函数的解析解为：

$$f^* = (I - \alpha S)^{-1} y$$

其中：
- $S = D^{-1/2} W D^{-1/2}$ 是归一化的拉普拉斯矩阵
- $\alpha = 1 / (1 + \mu)$
- $I$ 是单位矩阵

### 3.3 显著性计算

给定边界种子 $y$，得到排序得分 $f^*$。由于 $y$ 标记背景，$f^*$ 度量的是每个节点与背景的相似度。显著性取反：

$$S_i = 1 - f_i^*$$

归一化到[0,1]。

### 3.4 多方向融合

四个方向的显著性图相乘：

$$S_{\text{final}} = S_{\text{top}} \cdot S_{\text{bottom}} \cdot S_{\text{left}} \cdot S_{\text{right}}$$

相乘比相加更严格——只有所有方向都认为显著的区域才被保留。

## 4. 训练过程讲解

SGMR是**无训练**的传统方法。它直接通过以下步骤工作：
1. SLIC超像素分割
2. 计算超像素特征（Lab颜色均值）
3. 构建图（KNN或全连接）
4. 选择边界种子
5. 求解流形排序的解析解
6. 计算显著性
7. 四方向融合
8. 后处理（形态学平滑、高斯模糊）

## 5. 应用场景

1. **显著物体检测**：快速定位图像中的主要物体
2. **图像分割**：作为分割的前处理，提供物体位置先验
3. **图像裁剪**：根据显著性进行内容感知裁剪
4. **视频显著性**：扩展到视频帧序列

## 6. 优缺点分析

### 优点
1. **无需训练**：完全基于先验知识，零样本可用
2. **边界清晰**：超像素级别的处理保留了物体边界
3. **可解释**：流形排序的物理含义明确
4. **速度快**：相比深度方法，计算快

### 缺点
1. **边界假设**：假设物体不接触图像边界（违反时常发生）
2. **特征有限**：仅使用颜色特征，缺乏语义
3. **超像素质量依赖**：SLIC分割质量影响结果
4. **被深度方法超越**：在复杂场景中远不如深度方法

## 7. 调库实现

```python
"""
SGMR（基于流形排序的显著性检测）完整实现
"""

import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
import matplotlib.pyplot as plt


class SGMR:
    """基于流形排序的显著性检测

    参数:
        alpha: 流形排序中的平衡因子 (0~1)
        sigma: 相似度计算中的高斯带宽
        n_segments: 超像素数量
    """

    def __init__(self, alpha=0.99, sigma=10.0, n_segments=200):
        self.alpha = alpha
        self.sigma = sigma
        self.n_segments = n_segments

    def _compute_affinity(self, features):
        """计算超像素间的亲和力矩阵

        使用高斯核度量特征距离。
        """
        n = len(features)
        W = np.zeros((n, n))

        # KNN图（每个节点只连接最近的K个节点）
        K = min(10, n - 1)

        for i in range(n):
            dists = np.sum((features - features[i:i+1]) ** 2, axis=1)
            neighbors = np.argsort(dists)[:K]
            for j in neighbors:
                if i != j:
                    W[i, j] = np.exp(-dists[j] / (2 * self.sigma ** 2))

        # 对称化
        W = (W + W.T) / 2
        return W

    def _manifold_ranking(self, W, query):
        """流形排序：求解 f* = (I - alpha*S)^(-1) * y

        参数:
            W: 亲和力矩阵 (n, n)
            query: 查询种子向量 (n,)

        返回:
            f: 排序得分 (n,)
        """
        n = W.shape[0]
        # 度矩阵
        D = np.diag(W.sum(axis=1))

        # 归一化拉普拉斯: S = D^(-1/2) * W * D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(axis=1)) + 1e-8))
        S = D_inv_sqrt @ W @ D_inv_sqrt

        # 求解 (I - alpha*S) * f = y
        # f = (I - alpha*S)^(-1) * y
        A = np.eye(n) - self.alpha * S
        f = np.linalg.solve(A, query)

        return f

    def compute_saliency(self, image):
        """计算显著性图

        参数:
            image: HxWx3 RGB图像 (0-1范围)

        返回:
            saliency_map: HxW 显著性图 (0-1范围)
        """
        h, w = image.shape[:2]

        # 1. 简单的超像素模拟（实际应使用SLIC）
        # 使用网格分割模拟超像素
        cell_h, cell_w = h // int(np.sqrt(self.n_segments * h / w)), \
                         w // int(np.sqrt(self.n_segments * w / h))
        cell_h, cell_w = max(1, cell_h), max(1, cell_w)

        n_h, n_w = h // cell_h, w // cell_w
        n_segments = n_h * n_w

        # 提取每个"超像素"的特征
        features = []
        positions = []
        for i in range(n_h):
            for j in range(n_w):
                y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                patch = image[y1:y2, x1:x2]
                # 特征: RGB均值
                feat = patch.mean(axis=(0, 1))
                features.append(feat)
                positions.append(((x1 + x2) // 2, (y1 + y2) // 2))

        features = np.array(features)
        positions = np.array(positions)

        # 2. 计算亲和力矩阵
        W = self._compute_affinity(features)

        # 3. 四个方向边界先验
        # 上边界: 第一行超像素
        top_indices = list(range(n_w))
        # 下边界: 最后一行
        bottom_indices = list(range(n_w * (n_h - 1), n_w * n_h))
        # 左边界: 第一列
        left_indices = list(range(0, n_w * n_h, n_w))
        # 右边界: 最后一列
        right_indices = list(range(n_w - 1, n_w * n_h, n_w))

        boundaries = [top_indices, bottom_indices, left_indices, right_indices]
        saliency_scores = []

        for bound_indices in boundaries:
            query = np.zeros(n_segments)
            query[bound_indices] = 1.0
            scores = self._manifold_ranking(W, query)
            saliency = 1.0 - scores  # 反转为显著性
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            saliency_scores.append(saliency)

        # 4. 多方向融合（相乘）
        combined = np.ones(n_segments)
        for scores in saliency_scores:
            combined *= scores

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        # 5. 重建显著性图
        saliency_map = np.zeros((h, w))
        count_map = np.zeros((h, w))
        for k, (sx, sy) in enumerate(positions):
            y1 = max(0, sy - cell_h // 2)
            y2 = min(h, sy + cell_h // 2)
            x1 = max(0, sx - cell_w // 2)
            x2 = min(w, sx + cell_w // 2)
            saliency_map[y1:y2, x1:x2] += combined[k]
            count_map[y1:y2, x1:x2] += 1

        saliency_map = saliency_map / (count_map + 1e-8)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        return saliency_map


def demo():
    np.random.seed(42)

    # 创建测试图像
    img = np.ones((128, 128, 3)) * 0.3
    img[30:80, 30:80] = [0.8, 0.2, 0.2]  # 中心显著物体
    img[90:110, 10:30] = [0.2, 0.8, 0.2]  # 边界物体（应该不显著）

    model = SGMR(alpha=0.99, sigma=10.0, n_segments=256)
    sal = model.compute_saliency(img)

    print("=== SGMR显著性检测 ===")
    print(f"中心物体显著性: {sal[30:80, 30:80].mean():.4f}")
    print(f"边界物体显著性: {sal[90:110, 10:30].mean():.4f}")
    print(f"背景显著性: {sal[:20, :20].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title('输入图像'); axes[0].axis('off')
    axes[1].imshow(sal, cmap='hot')
    axes[1].set_title('SGMR显著性图'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('sgmr_demo.png', dpi=150)


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""SGMR核心手工实现"""
import numpy as np

def manifold_ranking_handcraft(W, y, alpha=0.99):
    """手工流形排序"""
    n = W.shape[0]
    D_sqrt = np.sqrt(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (D_sqrt + 1e-8))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    f = np.linalg.solve(np.eye(n) - alpha * S, y)
    return f

def test():
    np.random.seed(42)
    n = 50
    W = np.random.rand(n, n)
    W = (W + W.T) / 2  # 对称
    y = np.zeros(n)
    y[:5] = 1.0  # 前5个为种子
    f = manifold_ranking_handcraft(W, y)
    print(f"种子得分: {f[:5].mean():.4f}")
    print(f"非种子得分: {f[5:].mean():.4f}")
    print("（种子得分应 > 非种子得分）")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sgmr_boundaries(image, save_path='sgmr_boundaries.png'):
    """可视化SGMR的四个边界先验"""
    h, w = image.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('(a) 原始图像')
    axes[0, 0].axis('off')

    bound_names = ['上边界', '下边界', '左边界', '右边界']
    for i, name in enumerate(bound_names):
        ax = axes[(i+1)//3, (i+1)%3]
        mask = np.zeros((h, w))
        if i == 0: mask[:10, :] = 1
        elif i == 1: mask[-10:, :] = 1
        elif i == 2: mask[:, :10] = 1
        else: mask[:, -10:] = 1
        ax.imshow(image)
        ax.imshow(mask, alpha=0.3, cmap='Reds')
        ax.set_title(f'({chr(98+i)}) {name}种子')
        ax.axis('off')

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存到 {save_path}")

if __name__ == "__main__":
    img = np.random.rand(100, 100, 3)
    visualize_sgmr_boundaries(img)
```

## 10. 模型评估

```python
"""SGMR评估"""
def evaluate_sgmr(pred_map, gt_map):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    p, g = pred_map.flatten(), gt_map.flatten()
    auc = roc_auc_score((g > g.mean()).astype(int), p)
    cc = np.corrcoef(p, g)[0, 1]
    p_norm = (p - p.mean()) / (p.std() + 1e-8)
    nss = p_norm[g > g.mean()].mean()
    return {'AUC': auc, 'CC': cc, 'NSS': nss}

def bench():
    np.random.seed(42)
    from scipy.ndimage import gaussian_filter
    gt = np.zeros((100, 100))
    gt[30:70, 30:70] = 1
    gt = gaussian_filter(gt, sigma=5)
    gt = gt / gt.max()
    pred = np.random.rand(100, 100)
    metrics = evaluate_sgmr(pred, gt)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    bench()
```

## 11. 常见问题与易错点

**Q1: 为什么用四个方向的边界先验？**
单一方向的边界可能不完整（如物体接触上边界），使用四个方向相乘可以在物体接触某一边界时仍被其他方向检测到。

**Q2: 流形排序和PageRank有什么关系？**
流形排序的目标函数和PageRank相似，都基于图上的标签传播。区别在于流形排序使用归一化拉普拉斯矩阵，而PageRank使用转移概率矩阵。

**Q3: 如果物体占据了大部分图像边界（如特写照片），SGMR会失效吗？**
会。SGMR依赖"边界=背景"的假设，特写照片中物体充满画面时，边界种子不再是可靠的背景先验。

## 12. 学习总结

- SGMR是基于流形排序的传统显著性检测方法
- 核心假设：图像边界 = 背景
- 核心技术：图上的标签传播（流形排序）
- 优点：无需训练、速度快、可解释
- 局限：依赖边界假设、特征简单

## 13. 练习题

**基础题：**

1. SGMR为什么使用"相乘"而非"相加"融合四个方向的显著性？
> **答案：** 相乘更严格——只有被所有方向都判定为显著的区域才会保留。如果有一个方向认为该区域是背景（得分接近0），相乘后整个区域被抑制。

2. 流形排序中的alpha参数控制什么？
> **答案：** alpha控制平滑项和拟合项的平衡。alpha越接近1，传播越强（标签更平滑地扩散）；alpha越小，越依赖初始种子（拟合项更强）。

**进阶题：**

3. SGMR的边界先验假设在什么场景下会失效？
> **答案：** 当显著物体接触图像边界时（如特写、全景图），边界种子包含显著物体像素，导致误判。

4. 如何改进SGMR使其更鲁棒？
> **答案：** (1) 引入中心先验（加权边界重要性） (2) 使用深度特征替代颜色特征 (3) 多尺度超像素融合。

## 14. 学习路径

**前置：** 图论基础、流形学习、超像素分割
**平行：** CAS（上下文感知显著性）、ITTI
**进阶：** 深度学习显著性模型（DeepFix、EDN）

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SGMRNet(nn.Module):
    def __init__(self, dim_in=20, dim_h=64, dim_out=2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim_in, dim_h), nn.Linear(dim_h, dim_out)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

def train(model, X, y, epochs=100, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"Ep{ep+1} loss={loss.item():.4f}")

m = SGMRNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```
