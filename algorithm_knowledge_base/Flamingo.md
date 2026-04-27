# Flamingo 学习文档

> DeepMind的多模态大模型，少样本学习新范式

---

## 1. 算法基础认知

### 1.1 一句话定义

Flamingo是DeepMind于2022年发布的多模态大模型，可以在少量示例（few-shot）甚至零样本的情况下理解图片和视频，完成视觉问答、图像描述等任务。

### 1.2 直觉类比

Flamingo就像一个"会看会想的AI"。你给它看2-3个"一个小孩骑自行车的图片"+"这是骑自行车"的例子，再给它看一张新图，它就能说出"这个人正在骑三轮车"——它能从很少的例子中学习"举一反三"！

想象你教一个小孩认识新物种：
- 给他看3张"猫"的图片
- 给他看3张"狗"的图片  
- 问他新图片是什么——他说"猫"！

这就是Few-shot学习能力！

### 1.3 发展背景

- 2022年3月，DeepMind团队发布
- 基于DeepMind的Chinchilla语言模型（80B参数）
- 80亿参数的多模态模型
- 实现视觉-语言联合理解

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 多模态 → 视觉+语言 |
| 输出 | 文本描述/问答 |
| 模型 | LLM + Vision Encoder |
| 特点 | Few-shot学习 |

---

## 2. 核心原理

### 2.1 架构设计

```
图像/视频 → 视觉编码器 → Perceiver Resampler → 文本大模型
           │                         │                  │ 
      (冻结)                    │                  │
                            │              (冻结)
                     (少量可训练)
```

### 2.2 关键创新

1. **Perceiver Resampler**：将图像"压缩"成可理解的语言token
2. **冻结预训练**：视觉编码器和LLM都冻结，只训练少量参数
3. **连续prompt**：不是离散文本，而是连续的视觉特征等

### 2.3 对比其他模型

| 模型 | Few-shot | 训练方式 | 特点 |
|------|---------|---------|---------|
| CLIP | 无 | 对比学习 | 通用视觉 |
| BLIP-2 | 部分 | 预训练+Prompt | 灵活 |
| **Flamingo** | **强** | **Few-shot** | **少样本** |

---

## 3. 数学公式与推导

### 3.1 视觉编码

$$v = I_{\phi}(image)$$

其中 $I_{\phi}$ 是预训练的CLIP或NFNet，输出视觉特征。

### 3.2 Perceiver Resampler

使用cross-attention将视觉特征压缩为固定数量：

$$z = \text{Attn}(Q_{latent}, K_v, V_v)$$

其中 $Q_{latent}$ 是可学习的查询向量。

### 3.3 条件生成

$$P(y|x_v, x_t) = \prod_t P(y_t|y_{<t}, x_v, x_t)$$

条件是视觉token $x_v$ 和文本token $x_t$。

### 3.4 损失函数

语言模型损失：
$$\mathcal{L}_{LM} = -\sum_t \log P(y_t|y_{<t})$$

---

## 4. 训练过程讲解

### 4.1 预训练数据

- 大量图像-文本对
- 视频-描述对
- 交错的图像+句子序列

### 4.2 训练策略

1. **第一阶段**：训练Perceiver Resampler
2. **第二阶段**：端到端微调
3. **第三阶段**：In-context few-shot学习

### 4.3 配置

```python
# 训练配置
config = {
    'vision_encoder': 'CLIP ViT-L/14',
    'llm': 'Chinchilla-70B',
    'perceiver_dim': 768,
    'num_latent_tokens': 64,
    'learning_rate': 1e-4,
    'batch_size': 4,
}
```

---

## 5. 应用场景

### 5.1 视觉问答

```
Image: 一只猫
Q: What animal is this?
A: A cat
```

### 5.2 图像描述

```
Image: 一个人在跑步
Description: A person is running
```

### 5.3 视频理解

```
Video: 足球比赛
Answer: Two teams are playing soccer
```

### 5.4 少样本学习

```
Example 1: [猫图] + "cat"
Example 2: [狗图] + "dog"  
Query: [新动物图]
Answer: "rabbit"
```

### 5.5 性能对比

| 模型 | 零样本分类 | Few-shot |
|------|-----------|---------|
| CLIP | 72% | - |
| BLIP-2 | 78% | 80% |
| **Flamingo** | **76%** | **85%** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| Few-shot | 少样本学习 |
| 多模态 | 视觉+语言 |
| 开放域 | 无需微调 |
| 高质量 | SOTA水平 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 规模大 | 80B参数 |
| 商用限制 | 仅研究 |
| 显存高 | 需要A100 |

### 6.3 注意事项

- 目前无法直接使用
- 可用类似模型替代

---

## 7. 类似模型实现

### 7.1 BLIP-2

```python
from transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip2-opt-2.7b")

# 图片问答
inputs = processor("What is in this image?", image, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```

### 7.2 MiniGPT-4

```python
# 类似实现可参考MiniGPT-4
```

### 7.3 训练自己的Few-shot模型

```python
import torch
import torch.nn as nn

class FewShotModel(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()
        
        # Perceiver Resampler
        self.resampler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=vision_dim, nhead=8),
            num_layers=2
        )
        
        self.projection = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, image_features, prompt_embeds):
        # 重采样
        image_tokens = self.resampler(image_features.unsqueeze(0))
        image_tokens = self.projection(image_tokens)
        
        # 拼接prompt
        combined = torch.cat([image_tokens, prompt_embeds], dim=1)
        
        return combined
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimplePerceiver:
    """简化版Perceiver Resampler"""
    
    def __init__(self, vision_dim=768, num_latents=64, latent_dim=768):
        self.num_latents = num_latents
        
        # 可学习的查询向量
        self.latents = np.random.randn(num_latents, latent_dim) * 0.01
        
        # Cross attention简化为线性变换
        self.proj = np.random.randn(vision_dim, latent_dim) * 0.01
    
    def forward(self, vision_features):
        """视觉特征 -> 语言token
        
        vision_features: [vision_dim]
        """
        # 简单实现：线性投影
        image_tokens = vision_features @ self.proj
        
        # 重复到num_latents
        image_tokens = np.tile(image_tokens[:self.num_latents], (1, 1))
        
        return image_tokens
    
    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":
    # 测试
    perceiver = SimplePerceiver(vision_dim=768)
    
    vision_features = np.random.randn(768)
    tokens = perceiver(vision_features)
    
    print(f"输入: {vision_features.shape}")
    print(f"输出: {tokens.shape}")
```

---

## 9. 评估与可视化

### 9.1 评估指标

| 指标 | 说明 |
|------|------|
| Zero-shot | 零样本准确率 |
| Few-shot | K样本准确率 |
| COCO | 图像描述 |
| VQAv2 | 视觉问答 |

### 9.2 评估代码

```python
def evaluate_fewshot(model, image_paths, labels, num_shots=4):
    """Few-shot评估"""
    correct = 0
    
    for label in labels:
        # 选择示例
        examples = select_examples(image_paths, label, n=num_shots)
        
        # 预测
        pred = model.predict_image(query_image, examples)
        
        if pred == label:
            correct += 1
    
    return correct / len(labels)
```

---

## 10. 常见问题

### Q1: 如何获取Flamingo？

**答案**：目前仅研究使用，需申请。

### Q2: 替代方案？

**答案**：BLIP-2、MiniGPT-4效果接近。

### Q3: 为什么Few-shot有效？

**答案**：大模型具有涌现能力。

---

## 11. 学习路径

### 11.1 进阶路径

```
多模态基础
    ↓
CLIP/BLIP
    ↓
Flamingo原理
    ↓
Few-shot学习
    ↓
替代模型
    ↓
实战项目
```

### 11.2 相关算法

| 算法 | 关系 |
|------|------|
| CLIP | 视觉编码 |
| BLIP-2 | 多模态预训练 |
| LLaVA | 开源Flamingo |
| GPT-4V | 多模态LLM |

### 11.3 扩展阅读

1. DeepMind (2022). Flamingo: Visual Language Models
2. Alayrac et al. (2022). Flamingo: A Visual Language Model

---

## 12. 练习题与思考题

### 12.1 选择题

1. Flamingo的核心能力是：
   - A) 图像生成
   - B) Few-shot学习
   - C) 语音识别

2. Perceiver Resampler的作用是：
   - A) 图像分类
   - B) 压缩视觉特征
   - C) 文本生成

3. Flamingo的LLM部分：
   - A) 从头训练
   - B) 冻结微调
   - C) 提示学习

### 12.2 简答题

1. 解释Flamingo的Few-shot工作原理？
2. 为什么冻结预训练模型有效？
3. 比较Flamingo和BLIP-2的区别？

### 12.3 编程题

1. 实现简化版Perceiver Resampler
2. 构建Few-shot提示示例
3. 比较Few-shot vs Zero-shot效果

---

## 13. 常见问题与易错点

### Q1: 如何获取Flamingo模型？

**答案**：需要申请DeepMind模型权限，或使用LLaVA等开源替代。

### Q2: 支持哪些图像格式？

**答案**：支持常见图片格式（PNG, JPG）和视频。

### Q3: 支持中文吗？

**答案**：支持英文为主，需要微调才能支持中文。

### Q4: 上下文长度限制？

**答案**：取决于底层LLM，通常4K-8K tokens。

### Q5: Few-shot示例数量？

**答案**：官方测试1-32个示例均可。

---

## 14. 学习总结

### 14.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心创新 | Perceiver Resampler |
| 训练方式 | 冻结+少量可训练 |
| 关键能力 | Few-shot学习 |
| 模型架构 | LLM + Vision Encoder |

### 14.2 公式汇总

视觉编码：
$$v = I_{\phi}(image)$$

Perceiver Resampler：
$$z = \text{Attn}(Q_{latent}, K_v, V_v)$$

条件生成：
$$P(y|x_v, x_t) = \prod_t P(y_t|y_{<t}, x_v, x_t)$$

---

## 附录

### A. 参数速查

| 参数 | 说明 |
|------|------|
| vision_encoder | CLIP ViT-L/14 |
| llm | Chinchilla-80B |
| perceiver_tokens | 64 |
| few_shot | 1-32 |

### B. 参考

1. DeepMind (2022). Flamingo: Visual Language Models
2. GitHub: deepmind/deepmind-research

---

**文档结束**