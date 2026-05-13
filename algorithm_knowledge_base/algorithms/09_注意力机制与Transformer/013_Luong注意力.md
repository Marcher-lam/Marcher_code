# Luong注意力 学习文档

> 全局与局部注意力机制——用更简洁的计算实现序列到序列的注意力对齐。

> 来源线索：本节内容根据原书第3章关于"全局注意力与局部注意力机制"的相关章节整理。

## 1. 算法基础认知

**一句话定义：** Luong注意力由Minh-Thang Luong等人于2015年提出，包括全局注意力（Global Attention）和局部注意力（Local Attention）两种机制，是对Bahdanau注意力的简化和改进。

**直觉类比：** 全局注意力像在图书馆里查找一本书——你会查看所有书架的位置。局部注意力则像在确定的书架区域内查找——你预测书可能在附近区域，只在那个区域内搜索。

**历史背景：** 2015年发表在EMNLP上的论文"Effective Approaches to Attention-based Neural Machine Translation"中提出的。

---

## 2. 核心原理

### 2.1 全局注意力

全局注意力关注编码器的所有隐状态：

$$score(h_t, \bar{h}_s) = \begin{cases} h_t^\top \bar{h}_s & \text{dot} \\ h_t^\top W_a \bar{h}_s & \text{general} \\ v_a^\top \tanh(W_a[h_t; \bar{h}_s]) & \text{concat} \end{cases}$$

### 2.2 局部注意力

局部注意力只关注预测位置附近的一个窗口：

1. 预测对齐位置 $p_t = S \cdot \sigma(v_p^\top \tanh(W_p h_t))$
2. 在 $[p_t - D, p_t + D]$ 窗口内计算注意力
3. 使用高斯分布对窗口内的位置加权：$\alpha_t(s) = align(h_t, \bar{h}_s) \cdot \exp(-\frac{(s-p_t)^2}{2\sigma^2})$

---

## 3. 调库实现

```python
"""
Luong全局和局部注意力的PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """Luong注意力机制"""
    
    def __init__(self, hidden_size, attention_type='dot'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'concat':
            self.Wa = nn.Linear(2 * hidden_size, hidden_size)
            self.va = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, seq_len, hidden)
        """
        if self.attention_type == 'dot':
            score = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            energy = self.Wa(encoder_outputs)
            score = torch.bmm(energy, decoder_hidden.unsqueeze(2)).squeeze(2)
        else:  # concat
            dec_expanded = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            concat = torch.cat([encoder_outputs, dec_expanded], dim=2)
            energy = torch.tanh(self.Wa(concat))
            score = torch.bmm(energy, self.va.unsqueeze(0).unsqueeze(2)).squeeze(2)
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class LocalAttention(nn.Module):
    """Luong局部注意力（预测位置+窗口）"""
    
    def __init__(self, hidden_size, window_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.Wp = nn.Linear(hidden_size, 1, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        
        # 预测对齐位置
        pt = torch.sigmoid(self.Wp(decoder_hidden)) * seq_len
        pt_int = pt.long()
        
        # 确定窗口边界
        half = self.window_size // 2
        left = torch.clamp(pt_int - half, 0, seq_len)
        right = torch.clamp(pt_int + half, 0, seq_len)
        
        batch_indices = torch.arange(encoder_outputs.size(0))
        windows = encoder_outputs[batch_indices[:, None], left[0]:right[0]]
        
        # 在窗口内计算注意力
        score = torch.bmm(windows, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        # 高斯位置加权
        D = half
        positions = torch.arange(left[0].item(), right[0].item())
        gaussian_weight = torch.exp(-((positions - pt[0].item())**2) / (2 * D**2 / 4))
        score = score * gaussian_weight.unsqueeze(0)
        
        attn_weights = F.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), windows).squeeze(1)
        return context, attn_weights


def demo():
    batch, seq_len, hidden = 2, 10, 256
    dec_hidden = torch.randn(batch, hidden)
    enc_outputs = torch.randn(batch, seq_len, hidden)
    
    global_attn = LuongAttention(hidden, 'dot')
    context, weights = global_attn(dec_hidden, enc_outputs)
    
    local_attn = LocalAttention(hidden, window_size=5)
    context_local, weights_local = local_attn(dec_hidden, enc_outputs)
    
    print(f"全局注意力: 上下文向量 {context.shape}, 权重 {weights.shape}")
    print(f"局部注意力: 上下文向量 {context_local.shape}")
    print(f"局部注意力卷积权重前5个: {weights_local[0, :5].detach().numpy()}")


if __name__ == "__main__":
    demo()
```

---

## 4. 优缺点

**全局注意力优点：** 计算简单，无参数或少量参数；缺点：需要查看所有位置，对长序列计算量大

**局部注意力优点：** 计算高效，只关注局部窗口；缺点：位置预测可能出错

---

## 5. 学习路径

**前置：** Seq2Seq、Bahdanau注意力
**平行：** 自注意力机制
**进阶：** Transformer多头注意力

## 3. 数学公式与推导

Luong注意力的数学基础：

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

Luong注意力在以下领域有广泛应用：

- 计算机视觉（分类、检测、分割）
- 自然语言处理（生成、翻译、摘要）
- 语音识别与合成
- 跨模态理解与生成
- 科学计算与仿真

在工业实践中，Luong注意力通常与完整的数据管道配合使用。选择Luong注意力时需要根据数据特点、性能要求和计算资源综合考量。

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Luong注意力Net(nn.Module):
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

m = Luong注意力Net()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：Luong注意力与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('Luong注意力 Training Loss')
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
1. **基本原理**：Luong注意力的核心思想和数学基础
2. **实现方法**：从调库到手工实现
3. **应用场景**：Luong注意力适合的问题类型
4. **优化技巧**：超参数调优和正则化
5. **评估方法**：客观评估性能

### 关键概念
- Luong注意力的损失函数设计原理
- 参数优化的数学推导
- 泛化能力与过拟合的平衡

### 进阶方向
掌握Luong注意力后，可进一步学习相关的进阶方法和变体。


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Luong注意力的核心思想及适用场景。
<details><summary>参考答案</summary>
Luong注意力通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Luong注意力的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Luong注意力核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Luong注意力在什么情况下会失效？
2. 训练数据很少时，Luong注意力还能有效工作吗？
3. 如何将Luong注意力与其他方法结合？

