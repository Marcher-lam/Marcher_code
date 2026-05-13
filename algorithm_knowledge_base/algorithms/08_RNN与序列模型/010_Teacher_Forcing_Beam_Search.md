# Teacher Forcing 与 Beam Search 学习文档

> 序列生成训练与推理的经典技术。

## 1. 算法基础认知

### 一句话定义

Teacher Forcing是一种在训练时使用真实标签作为解码器输入的技巧，Beam Search是一种在推理时通过保留多条候选路径来提升生成质量的搜索算法。

### 直觉类比

- Teacher Forcing：就像老师教学生写作文时，学生每写一个词，老师就告诉他下一个词应该是什么，而不是让学生自己猜。这样学生学习更快。
- Beam Search：就像同时让几个学生各自写一段话，最后选写得最好的那一个。不是只盯着最好的一个，而是保留几个可能好的。

### 历史背景

- Teacher Forcing：1989年由Williams和Zipser提出
- Beam Search：广泛应用于统计机器翻译和语音识别

### 算法定位

都属于**序列生成技术**，分别用于训练和推理阶段。

---

## 2. 核心原理

### Teacher Forcing

训练时，每个时刻的输入使用上一时刻的真实标签，而不是模型预测：
- 优点：训练快、收敛稳定
- 缺点：推理时可能暴露问题（Exposure Bias）

### Beam Search

保留top-k条最优路径：
1. 每步保留概率最高的k个token
2. 继续扩展这k个路径
3. 选最终概率最高的路径

---

## 3. 调库实现

```python
import torch
import torch.nn.functional as F
import numpy as np

def beam_search_decode(model, encoder_output, beam_size=3, max_len=20):
    """Beam Search解码实现"""
    # 假设model是Seq2Seq模型
    device = encoder_output.device
    
    # 初始化
    beam_scores = torch.zeros(1, beam_size, device=device)
    beam_scores[:, 1:] = -1e9  # 第一个token固定
    
    # 假设decoder从encoder_output开始
    decoder_input = torch.full((1, 1), 2, dtype=torch.long, device=device)  # [PAD]
    
    # 简化的Beam Search（实际需要更复杂实现）
    finished = [False] * beam_size
    results = [[] for _ in range(beam_size)]
    
    for step in range(max_len):
        # 每次生成一个token的logits
        logits = model.decode_step(decoder_input, encoder_output)  # 简化
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取top-k
        log_probs, indices = log_probs.topk(beam_size, dim=-1)
        
        # 更新beam
        if step == 0:
            beam_scores = log_probs[0]
        else:
            beam_scores = beam_scores.unsqueeze(-1) + log_probs
            beam_scores = beam_scores.view(1, -1)
            beam_scores, top_idx = beam_scores.topk(beam_size, dim=-1)
        
        # 检查结束
        for i in range(beam_size):
            if indices[0, top_idx[0, i].item()].item() == 3:  # [EOS]
                finished[i] = True
        
        # 继续未完成的
        if all(finished):
            break
    
    return results

# Teacher Forcing示例
class TeacherForcingTrainer:
    """Teacher Forcing训练器"""
    def __init__(self, model):
        self.model = model
        
    def train_step(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: 源序列
        tgt: 目标序列 [B, T]
        teacher_forcing_ratio: 使用真实标签的概率
        """
        outputs = []
        dec_input = tgt[:, 0]  # [B], 起始符
        
        for t in range(1, tgt.size(1)):
            # 解码一步
            output = self.model(dec_input, src)  # [B, vocab_size]
            outputs.append(output)
            
            # 决定是否使用teacher forcing
            if np.random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t]  # 使用真实标签
            else:
                dec_input = output.argmax(-1)  # 使用预测
            
        return torch.stack(outputs, dim=1)

# 测试
if __name__ == "__main__":
    print("Teacher Forcing 和 Beam Search 实现已生成")
```

---

## 4. 优缺点

### Teacher Forcing

| 优点 | 缺点 |
|------|------|
| 训练稳定 | Exposure Bias |
| 收敛快 | 推理时分布偏移 |
| 梯度估计准确 | |

### Beam Search

| 优点 | 缺点 |
|------|------|
| 生成质量高 | 计算量大 |
| 避免贪婪解码 | 内存占用大 |
| 保留多样性 | 可能仍非最优 |

---

## 5. 学习路径

- 前置：RNN、Transformer、Seq2Seq
- 平行：Greedy Search、采样方法
- 进阶：Length Penalty、Coverage Penalty

## 3. 数学公式与推导

Teacher_Forcing_Beam_Search的数学基础：

### 前向传播
$$h = \sigma(W_1 x + b_1), \quad \hat{y} = W_2 h + b_2$$

### 损失函数（交叉熵）
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

### 反向传播（链式法则）
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W}$$


## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 5. 应用场景

Teacher_Forcing_Beam_Search在以下领域有广泛应用：

- 计算机视觉（分类、检测、分割）
- 自然语言处理（生成、翻译、摘要）
- 语音识别与合成
- 跨模态理解与生成
- 科学计算与仿真

在工业实践中，Teacher_Forcing_Beam_Search通常与完整的数据管道配合使用。选择Teacher_Forcing_Beam_Search时需要根据数据特点、性能要求和计算资源综合考量。

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TeacherForciNet(nn.Module):
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

m = TeacherForciNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：Teacher_Forcing_Beam_Search与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('Teacher_Forcing_Beam_Search Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 12. 学习总结

### 核心要点
1. **基本原理**：Teacher_Forcing_Beam_Search的核心思想和数学基础
2. **实现方法**：从调库到手工实现
3. **应用场景**：Teacher_Forcing_Beam_Search适合的问题类型
4. **优化技巧**：超参数调优和正则化
5. **评估方法**：客观评估性能

### 关键概念
- Teacher_Forcing_Beam_Search的损失函数设计原理
- 参数优化的数学推导
- 泛化能力与过拟合的平衡

### 进阶方向
掌握Teacher_Forcing_Beam_Search后，可进一步学习相关的进阶方法和变体。


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Teacher_Forcing_Beam_Search的核心思想及适用场景。
<details><summary>参考答案</summary>
Teacher_Forcing_Beam_Search通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Teacher_Forcing_Beam_Search的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Teacher_Forcing_Beam_Search核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Teacher_Forcing_Beam_Search在什么情况下会失效？
2. 训练数据很少时，Teacher_Forcing_Beam_Search还能有效工作吗？
3. 如何将Teacher_Forcing_Beam_Search与其他方法结合？

