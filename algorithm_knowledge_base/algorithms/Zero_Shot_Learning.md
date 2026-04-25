# Zero-Shot Learning 零样本学习 学习文档

> 识别从未见过类别的视觉识别方法

---

## 1. 算法基础认知

### 1.1 一句话定义

Zero-Shot Learning（零样本学习）是一种识别从未在训练集中出现过的类别的视觉识别方法。核心思想是通过语义属性将已知类别和未知类别连接起来，实现"看到未见过的物体"！

### 1.2 直觉类比

Zero-shot Learning就像一个"推理大师"。想象你从未见过"斑马"，但你：
- 知道斑马=马+黑白条纹
- 训练时见过"马"
- 训练时见过"黑色、白色条纹"

所以当你第一次看到斑马时，你能推理出"这是斑马"！

这就是Zero-shot的核心：**用已知知识推理未知类别**！

### 1.3 发展背景

- 2009年，Lampert在论文"Attribute-Based Classification"中提出
- 2014年，Socher将词嵌入引入Zero-shot
- 2016年，Cubuk+Bar提出最新范式
- 广泛应用于开放域识别、细粒度分类

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 视觉识别 → 零样本 |
| 核心 | 语义属性迁移 |
| 特点 | 识别未见过的类别 |

---

## 2. 核心原理

### 2.1 问题定义

**传统识别**：
- 训练类别：$\mathcal{S} = \{猫, 狗, 马\}$
- 测试类别：$\mathcal{S}$ = 必须从训练中选择

**Zero-shot**：
- 训练类别：$\mathcal{S}$
- 测试类别：$\mathcal{T}$，其中 $\mathcal{T} \cap \mathcal{S} = \emptyset$
- 需要识别从未见过的类别！

### 2.2 核心思想

通过**属性空间**连接已知和未知类别：

```
图像 → 视觉特征 → 类别嵌入 → 属性空间 → 预测属性 → 类别
```

### 2.3 架构对比

| 步骤 | 传统 | Zero-shot |
|------|------|----------|
| 特征 | CNN | CNN |
| 类别 | 直接映射 | 属性嵌入 |
| 识别 | 分类器 | 属性推理 |

### 2.4 实现方式

**方式1：属性嵌入**
```
类别 → 语义属性 → 训练模型识别属性 → 推理新类别
```

**方式2：词嵌入**
```
类别 → 词嵌入 → 视觉特征映射 → 推理
```

---

## 3. 数学公式与推导

### 3.1 符号定义

- $\mathcal{S}$：训练类别集合
- $\mathcal{T}$：测试类别集合，满足 $\mathcal{S} \cap \mathcal{T} = \emptyset$
- $x$：视觉特征
- $a_c$：类别 $c$ 的属性向量

### 3.2 属性预测

给定图像特征 $x$，预测属性 $\hat{a}$：
$$\hat{a} = f(x; \theta)$$

其中 $f$ 是神经网络，$\theta$ 是参数。

### 3.3 类别推理

根据预测的属性 $\hat{a}$，计算与各类别属性的相似度：

$$\hat{y} = \arg\max_c \text{sim}(\hat{a}, a_c)$$

使用余弦相似度：
$$\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

### 3.4 损失函数

```python
# 属性分类损失
loss_attr = CrossEntropy(attr_pred, attr_gt)

# 可视性损失（看图能识别属性）
loss_visibility = CrossEntropy(visible_pred, visible_gt)

# 总损失
loss = loss_attr + loss_visibility
```

---

## 4. 训练过程讲解

### 4.1 数据准备

**训练集**：有属性的已知类别
```python
# 数据格式
{
    'image': [img1, img2, ...],
    'attribute': [[0,1,1,0], [1,0,0,1], ...],  # 每个类别的属性
    'class_name': ['cat', 'dog', 'car', ...]
}
```

**属性设计**：
```
斑马的属性：
- 有条纹：是
- 有颜色：是
- 四条腿：是
- 体型：大
- 黑白：是
```

### 4.2 训练流程

```
Step 1: 提取视觉特征（CNN）
Step 2: 训练属性分类器
Step 3: 提取类别属性嵌入
```

### 4.3 属性编码

```python
# 属性编码示例
attributes = {
    'cat': [1, 0, 8, 4],  # 有腿=1, 能飞=0, 大小=8, 颜色=4
    'dog': [1, 0, 6, 3],
    'bird': [0, 1, 3, 2],
    'fish': [0, 1, 2, 3]
}
```

### 4.4 测试流程

```
输入：未知类别图像
    ↓
提取视觉特征
    ↓
预测属性
    ↓
计算与各类别属性的相似度
    ↓
输出：最相似的类别
```

---

## 5. 应用场景

### 5.1 开放域识别

```python
# 识别从未训练的类别
new_classes = ["斑马", "独角兽", "和外星人"]
model.load_attributes(new_classes)

# 识别图像
result = model.recognize(zebra_image)
print(result)  # 斑马
```

### 5.2 细粒度分类

```python
# 区分非常相似的类别
fine_classes = ["波斯猫", "暹罗猫", "孟加拉猫"]
model.load_attributes(fine_classes)
```

### 5.3 增量识别

```python
# 动态添加新类别
model.add_class("Zombie", zombie_attributes)
```

### 5.4 对比传统方法

| 场景 | 传统 | Zero-shot |
|------|------|----------|
| 新类别 | 需重新训练 | 直接识别 |
| 训练数据 | 需要大量 | 少量+属性 |
| 泛化能力 | 差 | **很强** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **识别新类别** | 无需重新训练 |
| **语义关联** | 利用属性关系 |
| **可解释** | 属性可解释 |
| **增量扩展** | 动态添加类别 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 属性设计 | 需要领域知识 |
| 精度有限 | 比闭集差一些 |
| 属性不完整 | 可能遗漏属性 |

### 6.3 注意事项

- 属性定义质量影响很大
- 需要选择有区分度的属性
- 可以使用预训练的词嵌入

---

## 7. 调库实现（Python）

### 7.1 基本用法

```python
import torch
import torchvision.models as models

# 加载预训练模型（ResNet等）
backbone = models.resnet50(pretrained=True)
backbone.fc = torch.nn.Identity()

# 使用预训练词嵌入
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# 类别属性
classes = ["cat", "dog", "bird", "fish"]
class_embeddings = text_model.encode(classes)

# Zero-shot分类
def zero_shot_classify(image, class_embeddings):
    # 图像特征
    image_features = backbone(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 相似度计算
    similarities = image_features @ class_embeddings.T
    
    # 预测
    pred = similarities.argmax(dim=-1)
    return classes[pred]

# 使用
result = zero_shot_classify(test_image, class_embeddings)
print(result)
```

### 7.2 自定义属性

```python
# 定义细粒度属性
attributes = {
    "Persian_cat": {"fur": "long", "size": "medium", "face": "flat", "color": "white"},
    "Siamese_cat": {"fur": "short", "size": "medium", "face": "wedge", "color": "cream"},
}

# 编码属性
import numpy as np

def attribute_to_vector(attribute_dict, attr_vocab):
    vectors = []
    for name, attrs in attribute_dict.items():
        vector = [attr_vocab.get(a, 0) for a in attrs.values()]
        vectors.append(vector)
    return np.array(vectors)

attr_vocab = {"long": 3, "short": 2, "medium": 1, "flat": 3, "wedge": 2, "white": 3, "cream": 2}
attribute_vectors = attribute_to_vector(attributes, attr_vocab)
```

### 7.3 TorchZSL实现

```python
# 使用torchzsl库
from torchzsl import Models

# 创建模型
model = Models.GAZSL(
    in_channels=2048,
    num_attrs=len(attribute_vectors),
    num_classes=len(seen_classes)
)

# 训练
model.fit(train_data, attribute_vectors, seen_classes)

# 推理
predictions = model.predict(test_data, attribute_vectors, unseen_classes)
```

### 7.4 评估

```python
from sklearn.metrics import accuracy_score

# 评估
predictions = []
for image in test_images:
    pred = model.recognize(image)
    predictions.append(pred)

accuracy = accuracy_score(test_labels, predictions)
print(f"Zero-shot Accuracy: {accuracy:.2%}")
```

---

## 8. 手工代码实现

### 8.1 简化Zero-shot

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ZeroShotClassifier:
    """简化版零样本分类器"""
    
    def __init__(self, backbone, class_attributes):
        self.backbone = backbone
        self.class_attributes = torch.FloatTensor(class_attributes)
        self.class_attributes = self.class_attributes / self.class_attributes.norm()
    
    def classify(self, image):
        # 提取图像特征
        with torch.no_grad():
            features = self.backbone(image)
            features = features / features.norm()
        
        # 计算相似度
        similarities = features @ self.class_attributes.T
        
        # 预测
        pred_idx = similarities.argmax().item()
        
        return pred_idx, similarities


# 使用示例
if __name__ == "__main__":
    import torchvision.models as models
    
    # 加载模型
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    # 定义类别属性（简单二元属性）
    class_attributes = np.array([
        [1, 1, 0],  # 猫：有毛、4条腿、不能飞
        [1, 1, 0],  # 狗：有毛、4条腿、不能飞
        [1, 1, 1],  # 鸟：有毛、4条腿、能飞
        [0, 0, 0],  # 鱼：无毛、不能动、不能飞
    ])
    
    # 创建分类器
    classifier = ZeroShotClassifier(resnet, class_attributes)
    
    # 识别
    # image = load_image("test.jpg")
    # pred = classifier.classify(image)
```

### 8.2 使用词嵌入

```python
from sentence_transformers import SentenceTransformer

class WordEmbeddingZSL:
    """使用词嵌入的Zero-shot"""
    
    def __init__(self, backbone, model_name='all-MiniLM-L6-v2'):
        self.backbone = backbone
        self.text_model = SentenceTransformer(model_name)
    
    def set_classes(self, class_names):
        # 编码类别名称
        self.class_embeddings = self.text_model.encode(class_names)
        self.class_embeddings = self.class_embeddings / self.class_embeddings.norm(axis=1, keepdims=True)
        self.class_names = class_names
    
    def classify(self, image):
        # 图像特征
        with torch.no_grad():
            features = self.backbone(image)
            features = features.squeeze()
            features = features / features.norm()
        
        # 转换numpy
        if hasattr(features, 'numpy'):
            features = features.numpy()
        
        # 计算相似度
        similarities = self.class_embeddings @ features
        
        # 预测
        pred_idx = similarities.argmax()
        
        return self.class_names[pred_idx], similarities
```

---

## 9. 可视化与结果理解

### 9.1 属性可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attributes(attribute_dict):
    """可视化类别属性"""
    # 准备数据
    classes = list(attribute_dict.keys())
    attributes = np.array(list(attribute_dict.values()))
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(attributes, cmap='YlOrRd')
    
    ax.set_xticks(range(attributes.shape[1]))
    ax.set_yticks(range(attributes.shape[0]))
    ax.set_xticklabels([f"A{i}" for i in range(attributes.shape[1])])
    ax.set_yticklabels(classes)
    
    plt.colorbar(im)
    plt.title("Class Attributes")
    plt.savefig("zsl_attributes.png", dpi=100)
    plt.show()


# 使用
attributes = {
    'cat': [1, 1, 0, 0],
    'dog': [1, 1, 0, 0],
    'bird': [1, 0, 1, 1],
    'fish': [0, 0, 0, 1]
}
visualize_attributes(attributes)
```

### 9.2 嵌入空间可视化

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embedding_space(image_features, class_embeddings, labels):
    """可视化嵌入空间"""
    # 结合图像特征和类别嵌入
    combined = np.vstack([image_features, class_embeddings])
    
    # PCA降维
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.scatter(combined_2d[:len(image_features), 0], combined_2d[:len(image_features), 1], 
               c='blue', alpha=0.5, label='Images')
    plt.scatter(combined_2d[len(image_features):, 0], combined_2d[len(image_features):, 1], 
               c='red', alpha=1, marker='*', s=200, label='Classes')
    
    for i, name in enumerate(labels[len(image_features):]):
        plt.annotate(name, combined_2d[len(image_features)+i])
    
    plt.legend()
    plt.title("Embedding Space")
    plt.savefig("embedding_space.png", dpi=100)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| Unseen Accuracy | 未见类别识别率 |
| Generalized ZSL | 广义Zero-shot |
| Calibrated Stacking | 校准精度 |

### 10.2 评估代码

```python
from sklearn.metrics import accuracy_score

def evaluate_zsl(model, test_images, test_labels, unseen_classes, seen_classes):
    """评估Zero-shot"""
    correct = 0
    total = 0
    
    for image, label in zip(test_images, test_labels):
        pred = model.classify(image)
        
        if label in unseen_classes:
            if pred in unseen_classes:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0


# 评估
accuracy = evaluate_zsl(classifier, test_images, test_labels, unseen_classes, seen_classes)
print(f"Zero-shot Accuracy: {accuracy:.2%}")
```

---

## 11. 常见问题与易错点

### Q1: Zero-shot和Few-shot的区别？

**答案**：Zero-shot是0个训练样本，Few-shot是少量（K个）样本。

### Q2: 属性如何设计？

**答案**：选择有区分度、可描述的属性。颜色、形状、大小等。

### Q3: 可以使用词嵌入代替属性吗？

**答案**：可以，使用Word2Vec、GloVe等词嵌入作为类别表示。

### Q4: 精度为什么不如传统方法？

**答案**：需要在属性空间表示和视觉空间表示之间建立良好映射。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 属性空间迁移 |
| 实现 | 视觉特征→属性→类别 |
| 优势 | 识别新类别 |
| 应用 | 开放域识别 |

### 12.2 公式汇总

属性预测：
$$\hat{a} = f(x; \theta)$$

相似度计算：
$$\text{sim}(\hat{a}, a_c) = \frac{\hat{a} \cdot a_c}{\|\hat{a}\|\|a_c\|}$$

类别推理：
$$\hat{y} = \arg\max_c \text{sim}(\hat{a}, a_c)$$

---

## 13. 练习题

### 13.1 选择题

1. Zero-shot的核心优势是：
   - A) 精度最高
   - B) 识别新类别
   - C) 最快

2. Zero-shot需要什么：
   - A) 大量训练数据
   - B) 属性或词嵌入
   - C) GPU

### 13.2 简答题

1. 解释Zero-shot Learning的原理。
2. 比较Zero-shot和Few-shot的区别。

### 13.3 编程题

1. ���现基于词嵌入的Zero-shot分类。
2. 设计一个自定义属性集并测试。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
图像识别基础
    ↓
特征提取
    ↓
属性/词嵌入
    ↓
Zero-shot原理
    ↓
实战应用
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| Few-shot | 少量样本 |
| One-shot | 单个样本 |
| Attribute Learning | 属性版 |

### 14.3 扩展阅读

- Lampert et al. (2009). Attribute-Based Classification
- Socher et al. (2014). Zero-Shot Learning via Semantic Embeddings

---

## 附录

### 参考

1. Lampert et al. (2009). Attribute-Based Classification. CVPR.
2. Socher et al. (2014). Zero-Shot Learning via Semantic Embeddings and Attribute Learning Classification. ICML.
3. https://github.com/facebookresearch/low-shot-sinh-of-means

---

**文档结束**