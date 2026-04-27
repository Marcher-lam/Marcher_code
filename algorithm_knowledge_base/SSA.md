# SSA自注意力句嵌入 学习文档
> 来源线索：本节内容根据原书第3章关于"自注意力机制"的相关章节整理。

## 1. 算法基础认知

### 1.1 为什么需要句嵌入？

在NLP领域，词嵌入（Word Embedding）技术将每个词映射为固定维度向量，并保持语义相近的词在向量空间中也接近。然而，很多应用需要处理的是更高级的语言单位——**句子**。

例如：
- **情感分析**：判断"This movie is amazing!"是正面还是负面
- **文本分类**：判断一篇新闻属于哪个类别
- **语义相似度**：判断两个句子是否表达相同意思

这些任务都要求将不定长的句子映射为固定长度的向量表示，即**句嵌入（Sentence Embedding）**。

### 1.2 传统句嵌入方法的局限

| 方法 | 描述 | 局限性 |
|------|------|--------|
| 取最后一个RNN输出 | 用RNN最后一个隐状态代表全句 | 长句信息丢失，早期内容被稀释 |
| 平均池化 | 对所有词向量取平均 | 忽略词的重要性差异和顺序 |
| 最大池化 | 取每个维度最大值 | 丢失大部分信息 |

### 1.3 SSA的核心思想

SSA（Structured Self-Attention，结构化自注意力）由Zhouhan Lin等人在2017年ICLR提出，其核心创新是：

1. **使用自注意力计算词的重要性权重**：不是简单平均，而是让模型学习每个词对句子语义的贡献
2. **多视角句嵌入**：用一个矩阵而不是一个向量表示句子，矩阵的每行代表一个注意力视角下的句子表示
3. **多样性正则化**：通过惩罚项保证不同视角关注句子的不同部分

### 1.4 直觉理解

想象你是一个评论家在分析一句话。你不会对所有词一视同仁，而是会关注那些承载核心语义的词。比如对于"The food was absolutely terrible!"，你会重点关注"terrible"这个词的情感倾向。

SSA所做的就是：让模型学习"哪些词更重要"，并**从多个角度**（比如情感角度、主题角度、语法角度）分别关注句子的不同部分。

---

## 2. 核心原理

### 2.1 总体架构

SSA模型的架构分为三个部分：

```
输入句子 → 双向LSTM编码 → 自注意力层 → 句嵌入矩阵
```

### 2.2 双向LSTM编码

设输入句子长度为 $n$，每个词的嵌入向量为 $\boldsymbol{w}_t$。SSA首先使用双向LSTM为每个词生成隐状态：

$$
\overrightarrow{\boldsymbol{h}}_t = \overrightarrow{\mathrm{LSTM}}(\boldsymbol{w}_t, \overrightarrow{\boldsymbol{h}}_{t-1})
$$

$$
\overleftarrow{\boldsymbol{h}}_t = \overleftarrow{\mathrm{LSTM}}(\boldsymbol{w}_t, \overleftarrow{\boldsymbol{h}}_{t+1})
$$

$$
\boldsymbol{h}_t = \mathrm{concat}(\overrightarrow{\boldsymbol{h}}_t, \overleftarrow{\boldsymbol{h}}_t)
$$

将所有隐状态拼接为矩阵：

$$
\boldsymbol{H} = (\boldsymbol{h}_1, \boldsymbol{h}_2, \ldots, \boldsymbol{h}_n) \in \mathbb{R}^{n \times 2u}
$$

其中 $u$ 是每个方向LSTM的隐状态维度，$2u$ 是双向拼接后的维度。

### 2.3 单视角自注意力

对于一个注意力视角，SSA通过两层MLP预测注意力权重：

$$
\boldsymbol{a} = \mathrm{softmax}\left(\boldsymbol{w}_2 \tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}})\right)
$$

其中：
- $\boldsymbol{W}_1 \in \mathbb{R}^{d_a \times 2u}$：第一层变换矩阵
- $\boldsymbol{w}_2 \in \mathbb{R}^{d_a}$：第二层权重向量
- $\boldsymbol{a} \in \mathbb{R}^n$：归一化的注意力权重向量

然后用注意力权重对隐状态加权求和得到单视角句嵌入：

$$
\boldsymbol{m} = \boldsymbol{a}^{\mathrm{T}} \boldsymbol{H} \in \mathbb{R}^{2u}
$$

### 2.4 多视角自注意力

为了从 $r$ 个不同视角观察句子，将权重向量 $\boldsymbol{w}_2$ 扩展为权重矩阵 $\boldsymbol{W}_2$：

$$
\boldsymbol{A} = \mathrm{softmax}\left(\boldsymbol{W}_2 \tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}})\right)
$$

其中：
- $\boldsymbol{W}_2 \in \mathbb{R}^{r \times d_a}$：参数矩阵，每行对应一个视角
- $\boldsymbol{A} \in \mathbb{R}^{r \times n}$：注意力权重矩阵，每行是一个归一化的注意力分布

最终句嵌入矩阵：

$$
\boldsymbol{M} = \boldsymbol{A} \boldsymbol{H} \in \mathbb{R}^{r \times 2u}
$$

其中每一行是一个视角下的句子表示。

### 2.5 多样性正则化

为避免多个视角的注意力趋同，引入惩罚项：

$$
P = \left\| (\boldsymbol{A} \boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I}) \right\|_F^2
$$

其中 $\boldsymbol{I}$ 是 $r$ 阶单位矩阵，$\|\cdot\|_F$ 是Frobenius范数。

**理解这个惩罚项**：
- $\boldsymbol{A} \boldsymbol{A}^{\mathrm{T}}$ 的第 $(i, j)$ 元素是第 $i$ 组和第 $j$ 组注意力权重的内积
- 如果两个视角注意力相同，内积为1，惩罚大
- 如果两个视角注意力正交，内积为0，无惩罚
- $-\boldsymbol{I}$ 操作去除对角线（自己和自己的内积恒为1，无需惩罚）

---

## 3. 数学公式与推导

### 3.1 自注意力计算的完整推导

**第一步**：线性变换

$$
\boldsymbol{H}^{\mathrm{T}} \in \mathbb{R}^{2u \times n}, \quad \boldsymbol{W}_1 \in \mathbb{R}^{d_a \times 2u}
$$

$$
\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}} \in \mathbb{R}^{d_a \times n}
$$

这步将每个隐状态从 $2u$ 维映射到 $d_a$ 维的"注意力空间"。

**第二步**：非线性激活

$$
\tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}}) \in \mathbb{R}^{d_a \times n}
$$

使用 $\tanh$ 引入非线性，使模型能学习更复杂的注意力模式。

**第三步**：映射为权重得分

$$
\boldsymbol{W}_2 \tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}}) \in \mathbb{R}^{r \times n}
$$

$\boldsymbol{W}_2$ 的每行定义一个注意力视角，将 $d_a$ 维变换结果映射为每个词的标量得分。

**第四步**：softmax归一化

$$
\boldsymbol{A} = \mathrm{softmax}\left(\boldsymbol{W}_2 \tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}})\right) \in \mathbb{R}^{r \times n}
$$

对每行（每个视角）独立做softmax，确保 $\sum_{j=1}^n \boldsymbol{A}_{ij} = 1$。

### 3.2 为什么softmax是按行计算的

在 $\boldsymbol{A} \in \mathbb{R}^{r \times n}$ 中，第 $i$ 行第 $j$ 列元素 $\boldsymbol{A}_{ij}$ 表示第 $i$ 个视角对第 $j$ 个词的注意力权重。Softmax按行操作确保每个视角的注意力分布是归一化的概率分布。

### 3.3 多样性惩罚项的梯度分析

惩罚项 $P = \| \boldsymbol{A} \boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I} \|_F^2$ 对注意力矩阵 $\boldsymbol{A}$ 的梯度：

$$
\frac{\partial P}{\partial \boldsymbol{A}} = 4\boldsymbol{A}(\boldsymbol{A}^{\mathrm{T}}\boldsymbol{A} - \boldsymbol{I})
$$

这个梯度会在训练过程中推动不同行的注意力向量相互正交，从而保证多样性。

### 3.4 与多头注意力的关系

SSA的多视角句嵌入矩阵本质上正是Transformer中多头注意力的雏形：

| SSA概念 | 对应Transformer概念 |
|---------|-------------------|
| $r$ 个视角 | $h$ 个注意力头 |
| $\boldsymbol{W}_2$ 每行一个视角 | 每个头的Q/K变换 |
| 句嵌入矩阵 $\boldsymbol{M}$ | 多头拼接后的输出 |

---

## 4. 训练过程讲解

### 4.1 整体训练流程

SSA的训练过程分为两个阶段：

**阶段一：任务无关训练（可选）**
- 在大型语料上训练语言模型或进行自监督学习
- LSTM编码器获得通用的语言表示能力

**阶段二：任务相关微调**
- 根据下游任务（分类、相似度等）训练整个模型
- 损失函数 = 任务损失 + $\lambda \cdot P$（多样性惩罚）

### 4.2 前向传播流程

```
输入句子: ["I", "love", "this", "movie"]
  ↓
词嵌入: [w_1, w_2, w_3, w_4]
  ↓
双向LSTM: [h_1, h_2, h_3, h_4], 每个 h ∈ ℝ^{2u}
  ↓
自注意力层:
  A = softmax(W_2 · tanh(W_1 · H^T))  →  A ∈ ℝ^{r×4}
  ↓
句嵌入矩阵: M = A · H  →  M ∈ ℝ^{r×2u}
  ↓
下游任务预测（如分类用MLP将M展平或池化）
```

### 4.3 反向传播

SSA的反向传播涉及：
1. 任务损失对句嵌入矩阵 $\boldsymbol{M}$ 的梯度
2. $\boldsymbol{M}$ 对注意力权重 $\boldsymbol{A}$ 的梯度
3. 注意力权重对 $\boldsymbol{W}_1, \boldsymbol{W}_2$ 的梯度
4. 对LSTM参数的梯度

由于注意力层只是简单的MLP + softmax，梯度计算相对直接。

---

## 5. 应用场景

### 5.1 文本分类

SSA生成的句嵌入矩阵可直接用于文本分类。将 $\boldsymbol{M} \in \mathbb{R}^{r \times 2u}$ 展平或池化后送入分类器。

```
句嵌入矩阵 → 展平(或平均池化) → 全连接层 → softmax → 类别概率
```

### 5.2 语义相似度计算

对于两个句子 $S_1$ 和 $S_2$：
1. 分别得到句嵌入矩阵 $\boldsymbol{M}_1$ 和 $\boldsymbol{M}_2$
2. 计算匹配特征：如逐元素差、逐元素积等
3. 送入分类器判断相似度

### 5.3 情感分析

SSA能自动关注情感词（如"amazing", "terrible"）所在位置，提升情感分类准确率。

### 5.4 自然语言推理（NLI）

判断前提和假设之间的关系（蕴含、矛盾、中立），SSA生成的双视角句嵌入能捕捉不同层面的语义信息。

### 5.5 文本蕴含

与NLI类似，判断一段文本是否能推出另一段文本。

---

## 6. 优缺点分析

### 优点

| 优点 | 说明 |
|------|------|
| **多视角表示** | 矩阵形式嵌入从多个角度理解句子，比单向量更丰富 |
| **可解释性** | 注意力权重直观展示模型关注了哪些词 |
| **简单有效** | 仅需在BiLSTM后添加两层MLP，结构简单 |
| **捕获长距离依赖** | BiLSTM + 注意力可建模句中任意距离的依赖 |
| **多样性保证** | 正则化项确保多视角不趋同 |

### 缺点

| 缺点 | 说明 |
|------|------|
| **依赖BiLSTM** | BiLSTM本身仍存在梯度问题，且无法并行 |
| **超参数敏感** | 视角数 $r$、注意力维度 $d_a$ 需调参 |
| **惩罚权重需要调节** | $\lambda$ 过大导致注意力被迫正交，过小则视角趋同 |
| **无法处理超长文本** | 受限于LSTM的处理能力 |
| **缺乏位置感知** | 注意力本身不考虑词序（但BiLSTM一定程度上弥补） |

---

## 7. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SSAModel(nn.Module):
    """
    SSA: Structured Self-Attention 模型
    
    用于句嵌入的结构化自注意力机制。
    核心思想：使用自注意力从多个视角为句子中的词计算重要性权重，
    得到r个不同的句嵌入向量（矩阵形式）。
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 attn_dim=100, num_views=10, num_classes=2, 
                 penalty_lambda=0.3, bidirectional=True, 
                 num_lstm_layers=1, dropout=0.5):
        """
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐状态维度（单向）
            attn_dim: 注意力中间变换维度 d_a
            num_views: 注意力视角数 r
            num_classes: 分类类别数
            penalty_lambda: 多样性惩罚权重
            bidirectional: 是否使用双向LSTM
            num_lstm_layers: LSTM层数
            dropout: Dropout比例
        """
        super(SSAModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_views = num_views  # r
        self.penalty_lambda = penalty_lambda
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_dim * self.num_directions  # 2u
        
        # 1. 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 3. 自注意力层
        # W1: [attn_dim, 2u]  将隐状态从2u映射到attn_dim
        self.W1 = nn.Linear(self.lstm_output_dim, attn_dim, bias=False)
        # W2: [r, attn_dim]  将attn_dim映射到r个视角的得分
        self.W2 = nn.Linear(attn_dim, num_views, bias=False)
        
        # 4. 分类层（可根据下游任务调整）
        # 将r*2u维的句嵌入映射到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(num_views * self.lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len]，每个元素是词在词汇表中的索引
            return_attention: 是否返回注意力权重矩阵
        
        返回:
            logits: 分类logits [batch_size, num_classes]
            attn_weights: (仅当return_attention=True) 注意力权重矩阵 [batch, r, seq_len]
            penalty: 多样性惩罚项的值
            M: 句嵌入矩阵 [batch, r, 2u]
        """
        batch_size, seq_len = x.shape
        
        # ===== 1. 词嵌入 =====
        # x: [batch, seq_len] -> embedded: [batch, seq_len, embedding_dim]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # ===== 2. 双向LSTM编码 =====
        # lstm_out: [batch, seq_len, 2*hidden_dim] (双向时)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # H = lstm_out, 即论文中的 H ∈ [batch, n, 2u]
        H = lstm_out  # [batch, seq_len, lstm_output_dim]
        
        # ===== 3. 自注意力计算 =====
        # 公式: A = softmax(W2 * tanh(W1 * H^T))
        # 
        # 实现步骤:
        # H: [batch, n, 2u]
        
        # Step 1: W1 * H^T 等价于对每个时间步应用W1
        # W1(H): [batch, n, attn_dim]
        W1_H = self.W1(H)  # [batch, n, attn_dim]
        
        # Step 2: tanh 激活
        # tanh(W1*H^T): [batch, n, attn_dim]
        tanh_W1_H = torch.tanh(W1_H)
        
        # Step 3: W2 * tanh(W1*H^T)
        # W2: [attn_dim, num_views] (nn.Linear的内部转置)
        # raw_attn: [batch, n, num_views]
        raw_attn = self.W2(tanh_W1_H)  # [batch, n, num_views]
        
        # Step 4: softmax 按n维度归一化（对每个视角、每个样本独立归一化）
        # A: [batch, r, n] (转置后r维在前)
        # 在seq_len维度上做softmax
        A = F.softmax(raw_attn, dim=1)  # [batch, n, r]
        A = A.transpose(1, 2)  # [batch, r, n]
        
        # ===== 4. 计算句嵌入矩阵 =====
        # M = A * H: [batch, r, n] @ [batch, n, 2u] -> [batch, r, 2u]
        M = torch.bmm(A, H)  # [batch, r, 2u]
        
        # ===== 5. 计算多样性惩罚 =====
        # P = ||A*A^T - I||_F^2
        # A: [batch, r, n], A^T: [batch, n, r]
        A_AT = torch.bmm(A, A.transpose(1, 2))  # [batch, r, r]
        
        # 创建单位矩阵 I
        I = torch.eye(self.num_views, device=x.device).unsqueeze(0)  # [1, r, r]
        I = I.expand(batch_size, -1, -1)  # [batch, r, r]
        
        # 计算 F 范数的平方
        penalty = torch.norm(A_AT - I, p='fro', dim=(1, 2)) ** 2  # [batch]
        penalty = penalty.mean()  # 标量: 整个batch的平均惩罚
        
        # ===== 6. 分类 =====
        # 将句嵌入矩阵展平: [batch, r*2u]
        M_flat = M.reshape(batch_size, -1)
        logits = self.classifier(M_flat)  # [batch, num_classes]
        
        if return_attention:
            return logits, A, penalty, M
        
        return logits, penalty, M
    
    def get_loss(self, logits, labels, penalty):
        """
        计算总损失 = 分类损失 + lambda * 惩罚项
        
        参数:
            logits: 分类logits [batch, num_classes]
            labels: 真实标签 [batch]
            penalty: 多样性惩罚标量
        
        返回:
            total_loss: 总损失
        """
        # 交叉熵损失
        ce_loss = F.cross_entropy(logits, labels)
        
        # 总损失 = 任务损失 + lambda * 惩罚
        total_loss = ce_loss + self.penalty_lambda * penalty
        
        return total_loss


# ===== 使用示例 =====
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模型
    model = SSAModel(
        vocab_size=5000,
        embedding_dim=128,
        hidden_dim=256,
        attn_dim=100,
        num_views=5,     # 5个注意力视角
        num_classes=2,
        penalty_lambda=0.3,
        bidirectional=True
    )
    
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型结构:\n{model}\n")
    
    # 模拟输入
    batch_size = 4
    seq_len = 20
    x = torch.randint(1, 5000, (batch_size, seq_len))  # 随机词索引
    labels = torch.randint(0, 2, (batch_size,))        # 随机标签
    
    # 前向传播
    logits, A, penalty, M = model(x, return_attention=True)
    
    print(f"输入形状: {x.shape}")
    print(f"Logits形状: {logits.shape}")
    print(f"注意力权重形状 A: {A.shape}")
    print(f"句嵌入矩阵形状 M: {M.shape}")
    print(f"多样性惩罚值: {penalty.item():.4f}")
    
    # 计算损失
    loss = model.get_loss(logits, labels, penalty)
    print(f"总损失: {loss.item():.4f}\n")
    
    # 查看注意力权重（第一个样本，所有视角）
    print("第一个样本的注意力权重（视角0~4 对 前10个词）:")
    attn_np = A[0].detach().numpy()[:, :10]
    for i in range(5):
        print(f"  视角{i}: {np.round(attn_np[i], 3)}")
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np

class SSAManual:
    """
    手工实现SSA自注意力句嵌入
    纯NumPy实现，无框架依赖
    """
    
    def __init__(self, input_dim, attn_dim=100, num_views=10):
        """
        参数:
            input_dim: 输入特征维度（BiLSTM输出维度 2u）
            attn_dim: 注意力中间变换维度 d_a
            num_views: 注意力视角数 r
        """
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.num_views = num_views
        
        # 初始化参数
        # W1: [attn_dim, input_dim]
        scale1 = np.sqrt(2.0 / (attn_dim + input_dim))
        self.W1 = np.random.randn(attn_dim, input_dim) * scale1
        
        # W2: [num_views, attn_dim]
        scale2 = np.sqrt(2.0 / (num_views + attn_dim))
        self.W2 = np.random.randn(num_views, attn_dim) * scale2
        
    def softmax(self, x, axis=-1):
        """
        softmax函数（数值稳定版）
        
        参数:
            x: 输入数组
            axis: 归一化维度
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, H):
        """
        前向传播
        
        参数:
            H: BiLSTM隐状态矩阵 [n, 2u]
               n: 句子长度
               2u: 双向LSTM隐状态维度
        
        返回:
            M: 句嵌入矩阵 [r, 2u]
            A: 注意力权重矩阵 [r, n]
        """
        n = H.shape[0]  # 句子长度
        
        # ===== 1. 计算注意力权重 =====
        # 公式: A = softmax(W2 * tanh(W1 * H^T))
        #
        # H: [n, 2u]
        # W1: [attn_dim, 2u]
        
        # W1 * H^T: [attn_dim, 2u] @ [2u, n] = [attn_dim, n]
        W1_H_T = self.W1 @ H.T  # [attn_dim, n]
        
        # tanh非线性
        tanh_W1_H_T = np.tanh(W1_H_T)  # [attn_dim, n]
        
        # W2 * tanh(W1*H^T): [r, attn_dim] @ [attn_dim, n] = [r, n]
        raw_attn = self.W2 @ tanh_W1_H_T  # [r, n]
        
        # softmax按列归一化（对n个词做归一化）
        # 注意：每行是一个视角，在该视角下对所有词做softmax
        A = self.softmax(raw_attn, axis=1)  # [r, n]
        
        # ===== 2. 计算句嵌入矩阵 =====
        # M = A * H: [r, n] @ [n, 2u] = [r, 2u]
        M = A @ H  # [r, 2u]
        
        # ===== 3. 计算多样性惩罚 =====
        # P = ||A*A^T - I||_F^2
        # A: [r, n], A^T: [n, r], A*A^T: [r, r]
        A_AT = A @ A.T  # [r, r]
        I = np.eye(self.num_views)
        
        diff = A_AT - I
        # F范数的平方 = 所有元素平方和
        penalty = np.sum(diff ** 2)
        
        return M, A, penalty
    
    def compute_gradient(self, H, grad_M):
        """
        计算对W1和W2的梯度（简化版）
        
        参数:
            H: 隐状态矩阵 [n, 2u]
            grad_M: 来自上层的梯度 [r, 2u]
        
        返回:
            grad_W1, grad_W2: 参数梯度
        """
        # 前向传播的中间结果
        W1_H_T = self.W1 @ H.T  # [attn_dim, n]
        tanh_W1_H_T = np.tanh(W1_H_T)  # [attn_dim, n]
        raw_attn = self.W2 @ tanh_W1_H_T  # [r, n]
        A = self.softmax(raw_attn, axis=1)  # [r, n]
        
        # 梯度 W2
        # M = A @ H, A = softmax(W2 @ tanh(W1 @ H^T))
        # grad_W2 = grad_M @ H^T @ (softmax的雅可比) @ tanh(W1@H^T)^T
        # 简化近似：
        grad_W2 = grad_M @ H.T @ A.T @ tanh_W1_H_T.T
        
        # 梯度 W1
        grad_W1 = None  # 完整推导较复杂，这里略去
        
        return grad_W1, grad_W2


def test_ssa_manual():
    """测试手工实现的SSA"""
    print("=" * 60)
    print("测试手工实现的SSA自注意力句嵌入")
    print("=" * 60)
    
    # 参数设置
    input_dim = 64    # BiLSTM输出维度 2u
    attn_dim = 32     # 注意力维度 d_a
    num_views = 4     # 视角数 r
    seq_len = 10      # 句子长度
    
    # 创建SSA实例
    ssa = SSAManual(input_dim, attn_dim, num_views)
    
    # 模拟BiLSTM输出
    np.random.seed(42)
    H = np.random.randn(seq_len, input_dim) * 0.5
    
    print(f"\n输入:")
    print(f"  句子长度 n = {seq_len}")
    print(f"  隐状态维度 2u = {input_dim}")
    print(f"  视角数 r = {num_views}")
    
    # 前向传播
    M, A, penalty = ssa.forward(H)
    
    print(f"\n输出:")
    print(f"  句嵌入矩阵 M 形状: {M.shape}")
    print(f"  注意力权重矩阵 A 形状: {A.shape}")
    print(f"  多样性惩罚 P = {penalty:.4f}")
    
    # 验证每行注意力权重之和为1
    print(f"\n注意力权重行和（应全为1.0）:")
    row_sums = A.sum(axis=1)
    print(f"  {np.round(row_sums, 6)}")
    
    # 打印注意力分布
    print(f"\n注意力分布（4个视角 × 10个词）:")
    for i in range(num_views):
        attn_str = "  ".join([f"{a:.3f}" for a in A[i]])
        print(f"  视角{i}: [{attn_str}]")
    
    # 检查各视角是否不同
    print(f"\n视角间相似度 (A*A^T 非对角线):")
    A_AT = A @ A.T
    for i in range(num_views):
        for j in range(i+1, num_views):
            print(f"  视角{i} × 视角{j} = {A_AT[i, j]:.4f} (越接近0越不同)")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_ssa_manual()
```

---

## 9. 可视化与结果理解

### 9.1 注意力权重可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ssa_attention():
    """可视化SSA多视角注意力权重"""
    
    # 示例句子
    sentence = "This movie was absolutely fantastic and I enjoyed every moment of it"
    words = sentence.lower().split()
    n_words = len(words)
    
    # 模拟4个注意力视角的权重分布
    np.random.seed(42)
    n_views = 4
    
    # 构造有意义的注意力分布
    A = np.zeros((n_views, n_words))
    
    # 视角0: 关注情感词 (fantastic, enjoyed)
    A[0, 4] = 0.4   # fantastic
    A[0, 7] = 0.3   # enjoyed
    A[0, 2:5] += 0.1 
    A[0] /= A[0].sum()
    
    # 视角1: 关注主语和动作 (movie, watching, enjoyed)
    A[1, 1] = 0.3   # movie
    A[1, 2] = 0.1   # was
    A[1, 3] = 0.1   # absolutely
    A[1, 7] = 0.3   # enjoyed
    A[1, 10] = 0.2  # it
    A[1] /= A[1].sum()
    
    # 视角2: 关注程度副词 (absolutely, every)
    A[2, 3] = 0.3   # absolutely
    A[2, 8] = 0.3   # every
    A[2, 4:9] += 0.05
    A[2] /= A[2].sum()
    
    # 视角3: 均匀分布（关注所有词）
    A[3, :] = 1.0 / n_words
    
    # 绘制热力图
    fig, axes = plt.subplots(n_views, 1, figsize=(14, 8))
    
    for i in range(n_views):
        ax = axes[i]
        bars = ax.bar(range(n_words), A[i], color=plt.cm.viridis(A[i]))
        ax.set_xticks(range(n_words))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(f'视角 {i}')
        ax.set_ylim(0, max(A[i]) * 1.3)
        
        # 在柱子上显示数值
        for j, val in enumerate(A[i]):
            if val > 0.05:
                ax.text(j, val + 0.01, f'{val:.2f}', ha='center', 
                       va='bottom', fontsize=8)
    
    axes[0].set_title('SSA多视角注意力权重可视化\n(4个视角对句子中每个词的关注程度)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ssa_attention_viz.png', dpi=150)
    plt.show()
    
    print("SSA注意力可视化已保存为 ssa_attention_viz.png")
    print("\n各视角关注模式:")
    print("  视角0: 关注情感词 (fantastic, enjoyed)")
    print("  视角1: 关注实体和动作 (movie, enjoyed)")
    print("  视角2: 关注程度副词 (absolutely, every)")
    print("  视角3: 均匀关注所有词")

visualize_ssa_attention()
```

### 9.2 多样性惩罚效果可视化

```python
def visualize_penalty_effect():
    """可视化多样性惩罚对不同注意力矩阵的影响"""
    
    # 构造三种情况的注意力矩阵
    n_views = 5
    n_words = 10
    
    # 情况1: 所有视角完全相同（需要惩罚）
    A_same = np.ones((n_views, n_words)) * 0.1
    penalty_same = np.linalg.norm(A_same @ A_same.T - np.eye(n_views), 'fro') ** 2
    
    # 情况2: 各视角完全不同（无需惩罚）
    A_diff = np.zeros((n_views, n_words))
    for i in range(n_views):
        focus_idx = i * 2
        if focus_idx < n_words:
            A_diff[i, focus_idx] = 1.0
        else:
            A_diff[i, -1] = 1.0
    A_diff /= A_diff.sum(axis=1, keepdims=True)
    penalty_diff = np.linalg.norm(A_diff @ A_diff.T - np.eye(n_views), 'fro') ** 2
    
    # 情况3: 部分相似
    A_partial = np.random.rand(n_views, n_words) * 0.5
    A_partial[0] = A_partial[1] * 0.8 + np.random.rand(n_words) * 0.1
    A_partial /= A_partial.sum(axis=1, keepdims=True)
    penalty_partial = np.linalg.norm(A_partial @ A_partial.T - np.eye(n_views), 'fro') ** 2
    
    # 绘制对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    titles = [
        f'视角完全相同\nP = {penalty_same:.2f}',
        f'视角完全不同\nP = {penalty_diff:.2f}',
        f'视角部分相似\nP = {penalty_partial:.2f}'
    ]
    
    for ax, A_mat, title in zip(axes, [A_same, A_diff, A_partial], titles):
        im = ax.imshow(A_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
        ax.set_xlabel('词位置')
        ax.set_ylabel('视角')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('多样性惩罚项效果对比（视角数 r=5, 词数 n=10）', fontsize=14)
    plt.tight_layout()
    plt.savefig('ssa_penalty_comparison.png', dpi=150)
    plt.show()
    
    print("惩罚项效果对比图已保存为 ssa_penalty_comparison.png")
    print(f"\n三种情况的惩罚值:")
    print(f"  视角完全相同: P = {penalty_same:.2f} (需要大幅惩罚)")
    print(f"  视角完全不同: P = {penalty_diff:.2f} (无需惩罚)")
    print(f"  视角部分相似: P = {penalty_partial:.2f} (适度惩罚)")

visualize_penalty_effect()
```

---

## 10. 模型评估

### 10.1 评估指标

句嵌入模型的评估通常分为**内在评估**和**外在评估**：

**内在评估**：

| 指标 | 说明 |
|------|------|
| 语义相似度相关性 | 与人类判断的Spearman/Pearson相关系数 |
| 句子聚类质量 | NMI, ARI等聚类指标 |
| 注意力多样性 | $\| \boldsymbol{A} \boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I} \|_F$ |

**外在评估（下游任务）**：

| 任务 | 指标 |
|------|------|
| 文本分类 | Accuracy, F1 |
| 情感分析 | Accuracy, F1 |
| 自然语言推理 | Accuracy |
| 语义文本相似度 | Pearson/Spearman相关系数 |

### 10.2 评估示例

```python
def evaluate_ssa_model():
    """评估SSA模型在文本分类上的表现"""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # 模拟数据：100个样本，5个类别
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # 模拟真实标签
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # 模拟SSA预测：假设比随机好但不完美
    # 给每个类别80%的正确率，20%随机错误
    y_pred = y_true.copy()
    error_mask = np.random.rand(n_samples) < 0.2
    y_pred[error_mask] = np.random.randint(0, n_classes, error_mask.sum())
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print("=" * 50)
    print("SSA句嵌入模型评估结果")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    
    return accuracy, f1_macro

evaluate_ssa_model()
```

### 10.3 消融实验

SSA模型的关键组件可以通过消融实验验证：

1. **移除多样性惩罚**（$\lambda = 0$）：各视角趋同，句嵌入矩阵退化为单向量重复
2. **单视角 vs 多视角**：$r=1$ 对比 $r>1$，验证多视角有效性
3. **移除BiLSTM**：直接用词嵌入做注意力，验证序列编码的必要性

---

## 11. 常见问题与易错点

### Q1: SSA的句嵌入矩阵 $\boldsymbol{M}$ 如何使用在下游任务中？

**答**：常见的做法有：
- **展平**：将 $\boldsymbol{M} \in \mathbb{R}^{r \times 2u}$ 展平为 $(r \cdot 2u)$ 维向量
- **池化**：对 $r$ 个视角取平均或最大池化，得到 $2u$ 维向量
- **拼接额外特征**：与手工特征拼接后送入分类器

### Q2: 视角数 $r$ 如何选择？

**答**：$r$ 通常取 5~30。过小时不足以覆盖句子的多个语义层面；过大时在多样性惩罚下可能学到无意义的区分。一般根据任务复杂度和句子平均长度调整。

### Q3: 为什么softmax在词维度归一化而不是视角维度？

**答**：softmax在词维度上归一化意味着"在每个视角下，对所有词的重要性做概率化"，使得每个视角得到一个关注分布。如果在视角维度归一化则意味着"每个词在不同视角上的重要性之和为1"，这不符合"多视角描述句子"的语义。

### Q4: 多样性惩罚中 $-\boldsymbol{I}$ 的作用是什么？

**答**：$\boldsymbol{A} \boldsymbol{A}^{\mathrm{T}}$ 的对角线元素 $\boldsymbol{A}_{i:} \boldsymbol{A}_{i:}^{\mathrm{T}} = \|\boldsymbol{A}_{i:}\|^2$，表示第 $i$ 个视角自身的"强度"，这个值总是为1（因为每行softmax归一化后L2范数不为0）。减去单位矩阵 $\boldsymbol{I}$ 的目的是**移除对角线元素**，使惩罚只关注非对角元素（即不同视角之间的相似度）。

### Q5: SSA与Transformer中的多头注意力有什么区别？

**答**：
- **SSA**：注意力权重由两层MLP从BiLSTM隐状态预测得出，权重是通过学习"哪些词重要"得到的
- **多头注意力**：注意力权重由Q、K的点积计算，关注的是"词与词之间的匹配程度"
- 核心区别：SSA是"词的重要性加权"，多头注意力是"词与词的关系建模"

---

## 12. 学习总结

### 12.1 关键知识点

1. **SSA的目标**：将不定长句子编码为固定维度的句嵌入矩阵
2. **多视角注意力**：用 $r$ 个不同的注意力分布分别关注句子的不同部分
3. **两层MLP注意力**：$\mathrm{softmax}(\boldsymbol{W}_2 \tanh(\boldsymbol{W}_1 \boldsymbol{H}^{\mathrm{T}}))$
4. **多样性正则化**：$P = \| \boldsymbol{A} \boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I} \|_F^2$ 防止多视角趋同

### 12.2 与注意力家族的关联

```
注意力机制
 ├── 加性注意力 (Bahdanau et al.)
 │    └── SSA 自注意力句嵌入 (Lin et al., 2017)
 │         └── 多头注意力的雏形
 ├── 乘性注意力 (Luong et al.)
 │    └── Transformer 缩放点积注意力 (Vaswani et al., 2017)
 └── 自注意力 (Self-Attention)
      ├── LSTMN (Cheng et al., 2016)
      └── Transformer (Vaswani et al., 2017)
```

### 12.3 核心思想一句话

> SSA通过"多视角自注意力 + 多样性正则化"，用矩阵而非向量表示句子，让模型从多个角度"看懂"一句话的丰富语义。

---

## 13. 练习题与思考题

### 基础题

**1. 填空题**：
SSA的注意力计算使用了两层神经网络，第一层的激活函数是______，第二层（输出层）的激活函数是______。

**答案**：$\tanh$、softmax。第一层 $\tanh$ 提供非线性变换能力；第二层 softmax 将得分归一化为概率分布。

**2. 判断题**：
SSA的句嵌入矩阵 $\boldsymbol{M}$ 的维度 $r \times 2u$ 中，$2u$ 固定不变，但 $r$ 可以任意设定。

**答案**：正确。$2u$ 由LSTM隐状态维度决定，一旦模型确定即固定；$r$ 是超参数，可以根据任务需求调整。

**3. 选择题**：
多样性惩罚项 $P = \|\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I}\|_F^2$ 中，$\boldsymbol{I}$ 的作用是：
A) 增加模型的记忆能力
B) 去除自身相似度的影响
C) 归一化注意力权重
D) 防止梯度消失

**答案**：B。$\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}}$ 的对角线是视角自身的L2范数（恒为1），减去 $\boldsymbol{I}$ 将其置零，使惩罚只关注不同视角间的相似度（非对角线元素）。

### 进阶题

**4. 推导题**：推导多样性惩罚项 $P = \|\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I}\|_F^2$ 对 $\boldsymbol{A}$ 的梯度。

**答案**：
令 $\boldsymbol{D} = \boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I}$，则 $P = \sum_{i,j} D_{ij}^2 = \mathrm{tr}(\boldsymbol{D}^{\mathrm{T}}\boldsymbol{D})$。

$$
\frac{\partial P}{\partial \boldsymbol{A}} = \frac{\partial}{\partial \boldsymbol{A}} \mathrm{tr}((\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I})^{\mathrm{T}}(\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - \boldsymbol{I}))
$$

利用矩阵微分：
$$
\frac{\partial P}{\partial \boldsymbol{A}} = 4\boldsymbol{A}(\boldsymbol{A}^{\mathrm{T}}\boldsymbol{A} - \boldsymbol{I})
$$

这个梯度表明：当 $\boldsymbol{A}^{\mathrm{T}}\boldsymbol{A}$ 接近 $\boldsymbol{I}$（视角正交）时，梯度接近0；当视角相似度高时，梯度较大，推动它们分离。

**5. 证明题**：证明当 $r=1$ 时，SSA退化为单注意力句嵌入，且多样性惩罚项恒为0。

**答案**：
当 $r=1$ 时，$\boldsymbol{A} \in \mathbb{R}^{1 \times n}$ 是一个行向量。则：
$$
\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} \in \mathbb{R}^{1 \times 1}
$$
为一个标量。由于softmax归一化使得 $\sum_j \boldsymbol{A}_{1j} = 1$，且所有元素非负，所以 $\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} = \sum_j \boldsymbol{A}_{1j}^2 \leq 1$。

$\boldsymbol{I}$ 是 $1 \times 1$ 的单位矩阵，即标量1。则：
$$
P = (\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} - 1)^2
$$

当且仅当 $\boldsymbol{A}$ 是一个one-hot向量（即注意力集中在唯一一个词上）时，$\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}} = 1$，$P=0$。
在其他情况下 $P > 0$。

**但更重要的**：当 $r=1$ 时，$\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}}$ 退化，因为"视角间相似度"的概念需要至少2个视角才成立。实际上，$r=1$ 时该惩罚项不再有意义。

**6. 思考题**：SSA使用BiLSTM编码句子，而Transformer直接使用自注意力编码。试分析这两种方式各自的优劣。

**答案**：

| 特性 | BiLSTM + SSA | Transformer Self-Attention |
|------|-------------|--------------------------|
| 计算方式 | 串行（递归） | 并行 |
| 长距离依赖 | 依赖门控机制缓解梯度问题 | 直接连接，无距离衰减 |
| 位置信息 | 通过递归顺序天然编码 | 需要额外位置编码 |
| 复杂度 | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| 短文本性能 | 较好（LSTM在短文本上成熟） | 较好 |
| 超长文本 | 随长度增加性能下降 | 计算开销随$n^2$增长 |

**7. 思考题**：如果去掉多样性惩罚项，SSA的多视角注意力会怎样？为什么？

**答案**：如果没有多样性惩罚，多个视角的注意力分布会趋于相同。原因在于：
- 所有视角共享相同的 $\boldsymbol{W}_1$ 和 $\boldsymbol{H}$
- $\boldsymbol{W}_2$ 的不同行虽然初始随机，但在训练中会被优化到相似的方向
- 因为对于大多数下游任务，关注最重要的词（如情感词）最有利于降低损失
- 所有视角都会学会关注那些"最有信息量"的词，导致趋同

这正是SSA论文引入多样性惩罚的核心动机。

### 编程题

**8. 实现题**：在SSA模型中增加位置编码，使得注意力可以感知词序。

**答案提示**：
```python
class SSAWithPosition(SSAModel):
    """带位置编码的SSA模型"""
    def __init__(self, max_seq_len=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 可学习的位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, self.lstm_output_dim)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.shape
        
        # 常规前向直到H
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        H = lstm_out  # [batch, n, 2u]
        
        # 添加位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_embed = self.position_embedding(positions)
        
        # 将位置编码加到隐状态上
        H_pos = H + pos_embed
        
        # 后续注意力计算使用 H_pos 而非 H
        W1_H = self.W1(H_pos)
        tanh_W1_H = torch.tanh(W1_H)
        raw_attn = self.W2(tanh_W1_H)
        A = F.softmax(raw_attn, dim=1).transpose(1, 2)
        M = torch.bmm(A, H_pos)
        
        # ... 后续相同
```

---

## 14. 学习路径建议

### 14.1 前置知识

学习SSA前，建议先掌握：

1. **词嵌入基础**：Word2Vec、GloVe等词向量技术
2. **RNN / LSTM**：理解序列建模的基本方法
3. **双向RNN**：理解前向和后向上下文信息的融合
4. **注意力机制基本概念**：Query、Key、Value的理解
5. **Softmax与概率**：理解归一化操作

### 14.2 建议学习路线

```
Step 1: 词嵌入 (Word2Vec / GloVe) → 理解"将词变成向量"
Step 2: RNN / LSTM → 理解"处理序列数据"
Step 3: 双向LSTM → 理解"融合上下文"
Step 4: 注意力机制基础 → 理解"选择重要信息"
Step 5: SSA → 理解"多视角句嵌入"
Step 6: Transformer 多头注意力 → 理解"并行化自注意力"
Step 7: BERT → 理解"预训练句嵌入"
```

### 14.3 延伸阅读

- **原论文**：Lin et al. "A Structured Self-Attentive Sentence Embedding" (ICLR 2017)
- **相关模型**：
  - InferSent (Facebook, 2017)
  - Universal Sentence Encoder (Google, 2018)
  - Sentence-BERT (2019)
  - BERT (Devlin et al., 2019)
- **进阶方向**：
  - 对比学习句嵌入（SimCSE, 2021）
  - 基于Prompt的句嵌入
  - 多语言句嵌入

### 14.4 实践建议

1. 在情感分析数据集（如IMDB、SST-2）上实现SSA
2. 尝试不同的视角数 $r$（如1, 3, 5, 10, 20），观察性能变化
3. 调整惩罚系数 $\lambda$（如0, 0.1, 0.3, 1.0），观察注意力多样性的变化
4. 可视化不同视角的注意力分布，分析每个视角"关注了什么"
5. 将SSA的句嵌入用于语义文本相似度（STS）任务
6. 与平均池化、最大池化等简单方法对比
