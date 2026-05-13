# CRF (Conditional Random Field) 学习文档

> 条件随机场，用于序列标注的判别模型，HMM 的 successor。

---

## 1. 算法基础认知

### 1.1 发展背景

CRF（Conditional Random Field，条件随机场）由 Lafferty 等人于 2001 年在论文《Conditional Random Fields for Probabilistic Online Segmentation》中提出，是用于序列标注和分词的经典概率图模型。作为 HMM（隐马尔可夫模型）的判别式改进，CRF 解决了标注偏差问题，在 NLP 领域广泛应用。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 判别式概率图模型 |
| 结构 | 线性链 CRF |
| 任务 | 序列标注、命名实体识别 |
| 优点 | 全局最优、无标注偏差 |

### 1.3 与生成模型对比

| 特性 | CRF (判别式) | HMM (生成式) |
|------|-------------|--------------|
| 学习目标 | P(Y\|X) | P(X\|Y)P(Y) |
| 特征设计 | 灵活 | 困难 |
| 标注偏差 | 无 | 严重 |
| 全局最优 | 是 | 否（局部） |

---

## 2. 核心原理

### 2.1 概率图结构

CRF 使用线性链结构：

```
X: O O O O O  (观测序列)
      ↓ ↓ ↓
Y: B M E B M  (标签序列)
```

每个 $Y_i$ 只与 $Y_{i-1}$ 和 $X$ 相关。

### 2.2 势函数

CRF 的核心是势函数：

1. **状态势函数**：$s(y_i, x_i)$ - 当前位置特征
2. **转移势函数**：$t(y_{i-1}, y_i)$ - 标签转移

### 2.3 特征模板

CRF 可以使用丰富的特征：

- 当前位置词
- 词性
- 词长度
- 是否数字
- 前/后词

---

## 3. 数学公式与推导

### 3.1 条件概率

给定观测序列 $X = (x_1, ..., x_n)$，标签序列 $Y = (y_1, ..., y_n)$：

$$P(Y|X) = \frac{1}{Z(x)} \exp\left(\sum_{i=1}^n s(y_i, x_i) + \sum_{i=1}^{n-1} t(y_i, y_{i+1})\right)$$

其中 $Z(x)$ 是归一化因子。

### 3.2 归一化因子

$$Z(x) = \sum_Y \exp\left(\sum_{i=1}^n s(y_i, x_i) + \sum_{i=1}^{n-1} t(y_i, y_{i+1})\right)$$

使用前向算法计算：

$$\alpha_0(y) = 1$$
$$\alpha_i(y) = \sum_{y'} \alpha_{i-1}(y') \cdot \exp(s(y', x_i) + t(y', y))$$

### 3.3 参数学习

使用最大似然估计：

$$L(\theta) = \sum_{(x,y)} \log P(y|x,\theta) - \frac{\lambda}{2} \|\theta\|^2$$

采用 L-BFGS 或 SGD 优化。

### 3.4 解码

使用维特比算法：

```python
# 动态规划
for i in range(1, n):
    for y in labels:
        dp[i][y] = max(dp[i-1][y'] * exp(t(y',y) + s(y,x_i)))
```

---

## 4. 训练过程讲解

### 4.1 算法流程

```
Input: 训练集 (X, Y), 特征模板
Output: CRF 模型

1. 特征提取: 构建特征函数
2. 参数初始化
3. 优化: L-BFGS 迭代
4. 解码: 维特比算法预测
```

### 4.2 特征提取

```python
# 简单特征提取
def extract_features(sentence, position):
    features = {}
    # 当前词
    features['word'] = sentence[position]
    # 词性
    features['pos'] = pos_tags[position]
    # 是否数字
    features['isdigit'] = sentence[position].isdigit()
    # 前一个词
    features['prev_word'] = sentence[position-1] if position > 0 else 'BOS'
    return features
```

### 4.3 参数更新

- L-BFGS：二阶近似
- SGD：一阶随机
- L1/L2 正则化

---

## 5. 应用场景

### 5.1 典型应用

- **中文分词**：词边界标注（B/M/E/S）
- **命名实体识别**：人名、机构名、地点
- **词性标注**：标注词性
- **中文依存分析**：依存关系

### 5.2 代码示例

```python
import sklearn_crfsuite

# CRF 模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,  # L1 正��化
    c2=0.1,  # L2 正则化
    max_iterations=100
)

# 训练
crf.fit(X_train, y_train)

# 预测
y_pred = crf.predict(X_test)
```

---

## 6. 优缺点分析

### 6.1 优点

1. **全局最优**：整个序列联合建模
2. **无标注偏差**：避免 HMM 的偏置问题
3. **特征灵活**：可使用任意特征
4. **概率输出**：可计算置信度

### 6.2 缺点

1. **训练慢**：特征空间大
2. **内存高**：存储特征
3. **依赖特征**：特征工程重要

### 6.3 改进方向

- **深度学习**：BiLSTM-CRF
- **预训练**：BERT-CRF
- **多任务**：共享表示

---

## 7. 调库实现

### 7.1 sklearn-crfsuite 实现

```python
try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("sklearn-crfsuite 未安装")

class CRFModel:
    """CRF 序列标注模型
    
    参数:
        algorithm: 优化算法
        c1: L1 正则化
        c2: L2 正则化
    """
    
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1,
                 max_iterations=100):
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.model = None
        
    def fit(self, X, y):
        """训练 CRF
        
        参数:
            X: 句子列表，每个句子是词列表
            y: 标签列表，每个句子是标签列表
        """
        if not CRF_AVAILABLE:
            raise ImportError("请安装: pip install sklearn-crfsuite")
        
        self.model = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations
        )
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测标签"""
        if not CRF_AVAILABLE:
            raise ImportError("请安装: pip install sklearn-crfsuite")
        
        return self.model.predict(X)
    
    def predict_marginals(self, X):
        """预测概率"""
        return self.model.predict_marginals(X)
    
    def score(self, X, y):
        """准确率"""
        y_pred = self.predict(X)
        
        # flat accuracy
        total = sum(len(seq) for seq in y)
        correct = sum(sum(p == t for p, t in zip(pred, true)) 
                   for pred, true in zip(y_pred, y))
        
        return correct / total


def word2features(sentence, i):
    """特征提取"""
    word = sentence[i]
    
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
    }
    
    if i > 0:
        word1 = sentence[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True
    
    if i < len(sentence)-1:
        word1 = sentence[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True
    
    return features


def sent2features(sentence):
    """句子特征提取"""
    return [word2features(sentence, i) for i in range(len(sentence))]


def sent2labels(labels):
    """标签提取"""
    return labels


def demo():
    """CRF 演示"""
    print("=== CRF 序列标注演示 ===\n")
    
    if not CRF_AVAILABLE:
        print("sklearn-crfsuite 未安装")
        print("安装: pip install sklearn-crfsuite")
        return None
    
    # 训练数据：中��分词
    train_sents = [
        ["今天", "天气", "很", "好"],
        ["我", "爱", "自然", "语言", "处理"],
        ["深度", "学习", "是", "人工", "智能"],
    ]
    train_labels = [
        ["B", "B", "B", "S"],
        ["B", "B", "B", "B", "S"],
        ["B", "B", "B", "B", "B", "S"],
    ]
    
    # 特征提取
    X_train = [sent2features(s) for s in train_sents]
    y_train = train_labels
    
    print(f"训练句子数: {len(X_train)}")
    
    # 训练 CRF
    crf = CRFModel(c1=0.1, c2=0.1)
    crf.fit(X_train, y_train)
    
    # 预测
    test_sent = ["明天", "应该", "会", "下雨"]
    X_test = [sent2features(test_sent)]
    y_pred = crf.predict(X_test)[0]
    
    print(f"输入: {test_sent}")
    print(f"输出: {y_pred}")
    
    return crf


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

### 8.1 简化 CRF 实现

```python
import numpy as np

class SimpleCRF:
    """简化版 CRF
    
    参数:
        states: 标签状态
    """
    
    def __init__(self, states):
        self.states = states
        self.num_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        
        # 转移矩阵和发射矩阵
        self.transitions = None
        self.emissions = None
        
    def fit(self, X, y):
        """训练 CRF
        
        参数:
            X: 特征列表
            y: 标签列表
        """
        # 统计计数
        transition_counts = np.ones((self.num_states, self.num_states))
        emission_counts = np.ones((self.num_states, self.num_states))
        
        for seq_x, seq_y in zip(X, y):
            # 序列标签索引
            seq_y_idx = [self.state_to_idx[label] for label in seq_y]
            
            for i, y_idx in enumerate(seq_y_idx):
                # 发射计数
                emission_counts[y_idx] += 1
                
                # 转移计数
                if i > 0:
                    transition_counts[seq_y_idx[i-1], y_idx] += 1
        
        # 归一化为概率
        self.transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        self.emissions = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        
        return self
    
    def predict(self, X):
        """维特比解码"""
        predictions = []
        
        for seq_x in X:
            # 动态规划
            n = len(seq_x)
            dp = np.zeros((n, self.num_states))
            backpointer = np.zeros((n, self.num_states), dtype=int)
            
            # 初始化
            dp[0] = np.log(self.emissions[:, 0]) if hasattr(self.emissions, '__getitem__') else \
                    np.zeros(self.num_states)
            
            # DP
            for i in range(1, n):
                for j in range(self.num_states):
                    probs = dp[i-1] + np.log(self.transitions[:, j])
                    best_prev = np.argmax(probs)
                    dp[i, j] = probs[best_prev]
                    backpointer[i, j] = best_prev
            
            # 回溯
            best_path = np.zeros(n, dtype=int)
            best_path[-1] = np.argmax(dp[-1])
            for i in range(n-2, -1, -1):
                best_path[i] = backpointer[i+1, best_path[i+1]]
            
            # 转换为标签
            pred = [self.states[idx] for idx in best_path]
            predictions.append(pred)
        
        return predictions


def demo_manual():
    """手工实现演示"""
    print("=== CRF 手工实现演示 ===\n")
    
    # 简单数据
    X = [
        [{'word': '今天'}, {'word': '天气'}, {'word': '好'}],
        [{'word': '我'}, {'word': '爱'}, {'word': '学习'}],
    ]
    y = [
        ['B', 'B', 'S'],
        ['B', 'B', 'S'],
    ]
    
    states = ['B', 'S']
    
    # 训练
    crf = SimpleCRF(states)
    crf.fit(X, y)
    
    # 预测
    X_test = [[{'word': '明天'}, {'word': '下雨'}]]
    y_pred = crf.predict(X_test)
    
    print(f"预测: {y_pred}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 模型可视化

```python
def visualize_crf():
    """可视化 CRF 结构"""
    
    print("""
    CRF 线性链结构:
    
    x_1 → y_1 → y_2 → y_3 → ... → y_n
    ↓      ↓     ↓     ↓         ↓
    s(y1)  t(y1,y2)  t(y2,y3)  t(yn-1,yn)
    
    其中:
    x_i: 观测（词语特征）
    s(y_i): 状态势函数
    t(y_i,y_{i+1}): 转移势函数
    """)
```

### 9.2 特征权重

```python
def plot_weights():
    """特征权重可视化"""
    import matplotlib.pyplot as plt
    
    features = ['B', 'M', 'E', 'S']
    weights = np.random.randn(4, 8)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(weights, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xticks(range(8), ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8'])
    plt.yticks(range(4), features)
    plt.title('CRF 特征权重')
    plt.tight_layout()
    plt.savefig('crf_weights.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import classification_report

def evaluate_crf(y_true, y_pred):
    """评估 CRF"""
    
    # 展平
    y_true_flat = [label for seq in y_true for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    
    report = classification_report(y_true_flat, y_pred_flat)
    return report
```

### 10.2 分词评估

| 指标 | 公式 | 典型值 |
|------|------|--------|
| Precision | 正确数/预测数 | 95% |
| Recall | 正确数/标准数 | 95% |
| F1 | 2PR/(P+R) | 95% |

---

## 11. 常见问题与易错点

### 11.1 特征选择

**问题**：特征越多越好？

**解答**：
- 适度特征
- 特征相关
- 避免稀疏

### 11.2 收敛判断

**问题**：训练不收敛

**解决**：
- 调小学习率
- 增加迭代
- 检查特征

### 11.3 内存问题

**问题**：内存溢出

**解决**：
- 减少特征
- 批量处理

---

## 12. 学习总结

**核心要点**：

1. **判别模型**：直接建模 P(Y|X)
2. **状态+转移**：同时考虑当前位置和转移
3. **全局最优**：整个序列联合建模
4. **维特比解码**：动态规划

**CRF 核心优势**：
- 无标注偏差
- 特征灵活
- 全局最优

**学习建议**：

1. 理解概率图模型基础
2. 掌握特征工程
3. 实践序列标注

---

## 13. 练习题与思考题

### 13.1 基础练习

1. CRF vs HMM 的区别
2. 维特比算法推导
3. 特征模板设计

### 13.2 进阶练习

1. 实现完整 CRF
2. BiLSTM-CRF

### 13.3 思考题

1. 深度学习时代的 CRF
2. 如何结合预训练

---

### 13.4 详细答案与解析

#### 练习1：CRF vs HMM

**问题**：CRF 相对 HMM 的优势

**解答**：

| 特性 | CRF | HMM |
|------|-----|-----|
| 模型类型 | 判别式 | 生��式 |
| 标注偏差 | 无 | 有 |
| 特征 | 灵活 | 困难 |
| 最优性 | 全局 | 局部 |

#### 练习2：维特比推导

**问题**：维特比算法原理

**解答**：

动态规划找最佳路径：

$$\delta_i(y) = \max_{y_{i-1}} \delta_{i-1}(y_{i-1}) \cdot \psi(y_{i-1}, y_i, x_i)$$

---

## 14. 学习路径建议

### 入门阶段

1. 学习概率论基础
2. 掌握 HMM
3. 理解 CRF 原理

### 进阶阶段

1. 实现 CRF
2. 特征工程
3. 实践 NER

### 高级阶段

1. BiLSTM-CRF
2. 预训练 + CRF
3. 多任务学习

**推荐路线**：

```
HMM → CRF → BiLSTM-CRF → BERT-CRF
```

**CRF 是序列标注的基础模型，熟练掌握它对 NLP 学习很重要。**