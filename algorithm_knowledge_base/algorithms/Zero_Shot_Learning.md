# Zero-Shot Learning 学习文档

> 识别从未见过类别的视觉识别方法。

---

## 1. 算法基础认知

**Zero-Shot Learning（零样本学习）** 是一种识别从未在训练集中出现过的类别的方法。核心思想是通过语义属性将已知类别和未知类别连接起来。

### 1.1 核心思想

传统识别：
```
训练: 学习猫、狗 → 测试: 识别猫、狗
```

Zero-shot：
```
训练: 学习属性 → 测试: 使用属性识别斑马
```

### 1.2 关键组件

1. **语义属性**
2. **类别嵌入**
3. **视觉-语义映射**

### 1.3 对比

| 方法 | 训练类 | 测试类 | 关系 |
|------|--------|--------|------|
| Standard | 相同 | 相同 | 固定 |
| Zero-shot | 不同 | 新类 | 需桥接 |
| Few-shot | 不同 | 新类 | 需泛化 |

---

## 2. 核心原理

### 2.1 属性学习

```python
# 为每个类定义属性
class_attributes = {
    "cat": [1, 0, 1, 1, 0],  # 有腿、毛、尾巴...
    "dog": [1, 0, 1, 1, 1],
}
```

### 2.2 分类器学习

基于属性建立分类器：
```python
y = argmax P(y|x) * P(y|attributes)
```

### 2.3 映射学习

学习视觉→语义映射：
```python
visual_embedding → semantic_embedding
```

---

## 3. 方法

### 3.1 DAP

Direct Attribute Prediction

### 3.2 CALE

Category-Level Attribute Embedding

### 3.3 SAE

Semantic AutoEncoder

### 3.4 最新方法

- CLIP：视觉语言预训练

---

## 4. 数据集

### 4.1 Animal with Attributes (AWA)

- 50类动物
- 85个二进制属性
- 30K图像

### 4.2 Sun Attributes

- 717类场景
- 102个属性

### 4.3 aPascal/aYahoo

- 80类
- 64个属性

---

## 5. 调库实现

```python
import torch
import torch.nn as nn

class ZeroShotClassifier(nn.Module):
    """零样本分类器"""
    
    def __init__(self, visual_model, class_attributes):
        self.visual = visual_model
        self.attributes = class_attributes  # [num_classes, num_attrs]
    
    def forward(self, images):
        # 视觉特征
        visual_emb = self.visual(images)
        
        # 分类
        scores = visual_emb @ self.attributes.T
        
        return scores


def clip_zero_shot():
    """CLIP零样本"""
    print("=== CLIP Zero-shot ===\n")
    print("""
    1. 图像编码: CLIP编码器
    2. 文本编码: "a photo of [class]"
    3. 对比: 图像vs文本嵌入
    """)


if __name__ == "__main__":
    clip_zero_shot()
```

---

## 6. 手工代码实现

```python
import numpy as np

class SimpleZeroShot:
    """简化版零样本"""
    
    def __init__(self, class_attributes):
        self.attributes = class_attributes
    
    def predict(self, image_feature):
        # 计算与每个类的相似度
        scores = image_feature @ self.attributes.T
        return scores.argmax()


if __name__ == "__main__":
    print("=== Zero-shot核心 ===\n")
    print("1. 属性定义")
    print("2. 视觉-属性映射")
    print("3. 识别新类")
```

---

## 7. 应用场景

### 7.1 图像分类

新类别识别

### 7.2 目标检测

开放词汇检测

### 7.3 视觉问答

新概念理解

---

## 10. 模型评估

### 8.1 指标

- 零样本准确率
- 归纳/转导设置

### 8.2 基线

- 随机：1/N
- DAP、CALE

---

## 9. 常见问题

### 9.1 属性缺失

### 9.2 分布偏移

### 9.3 属性歧义

---

## 10. 学习总结

**Zero-shot要点**：

1. **语义属性**：类级别描述
2. **视觉-语义映射**：桥接两空间
3. **泛化**：识别新类
4. **CLIP**：现代方法

---

## 11. 练习题

1. 零样本和少样本的区别？
2. CLIP为什么有效？

---

## 12. 学习路径

1. 理解属性学习
2. 学习CLIP
3. 实践零样本分类

---

*Zero-shot learning让模型能识别从未见过的物体，是AI的重要突破。*
```
## 13. 练习题与思考题与思考题
### 13.1 基础练习题
**练习1**：本算法的核心机制是什么？请简述其工作原理。
**答案**：本算法的核心是[机制]，通过[步骤]实现[目标]。

**练习2**：给定以下数据，手动计算第一次参数更新。
**答案**：根据[公式]计算，第一次迭代参数更新为[结果]。

### 13.2 进阶思考题
**思考题**：本算法存在哪些局限性？请提出至少2种改进方案。
**答案**：1. [局限性1]→[改进方案1]；2. [局限性2]→[改进方案2]。

## 14. 学习路径建议建议
### 14.1 前置知识
- 线性代数基础（矩阵运算、向量空间）
- 微积分基础（偏导数、梯度）
- Python编程基础（NumPy/PyTorch）
- 机器学习基本概念

### 14.2 平行算法
- [算法A]：[简要对比]
- [算法B]：[简要对比]

### 14.3 进阶算法
- [进阶算法1]：进一步发展方向
- [进阶算法2]：改进方向

### 14.4 推荐资源
**书籍**：《机器学习》周志华，《深度学习》花书
**论文**：[算法名]原论文
**课程**：Andrew Ng机器学习课程
