# SnowNLP情感分析 学习文档

> 用轻量级中文NLP库快速量化文本情感——零训练成本的情绪评分利器。
> 来源线索：本节内容根据原书中关于SnowNLP在财务文本分析（第3章3.2节）和舆情监控（第3章3.14节）中的应用整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：SnowNLP是一个轻量级中文自然语言处理库，提供情感分析、分词、关键词提取等功能，特别适合快速对财经文本做情感量化。

**直觉类比**：体温计——把一段中文文字输入SnowNLP，输出0-1的情感分数(越接近1越积极，越接近0越消极)，就像用体温计量体温一样简单直接。

**背景**：SnowNLP是受TextBlob启发的Python中文NLP库，由个人开发者维护。它在金融文本分析中应用广泛，原因是：(1)中文原生支持；(2)无需GPU；(3)输出直观的[0,1]分数；(4)可以快速处理大量文本。原书第3.2节用它分析财报管理层表述，第3.14节用它做舆情情感判断。

**算法定位**：NLP工具 / 情感分析 / 中文文本处理 / 量化因子输入。

**前置知识**：Python基础、基本NLP概念。

## 2. 核心原理

**SnowNLP 功能体系**：
| 功能 | 方法 | 量化应用 |
|------|------|----------|
| 情感分析 | SnowNLP(text).sentiments | 输出0-1分，>0.6偏积极 |
| 中文分词 | SnowNLP(text).words | 提取关键词/实体 |
| 关键词提取 | SnowNLP(text).keywords(n) | 识别文本主题 |
| 文本摘要 | SnowNLP(text).summary(n) | 长文本精简 |
| 拼音转换 | SnowNLP(text).pinyin | 辅助搜索 |

**在量化中的应用位置**：
```
原始文本(新闻/财报/研报/社交媒体)
    │
    ├─ SnowNLP.sentiments → 情感得分(0-1) → 作为ML特征输入
    ├─ SnowNLP.words      → 分词 → 关键词匹配 → 事件分类
    └─ SnowNLP.keywords   → 主题提取 → 题材识别
```

**情感分析的底层原理**：SnowNLP的情感分析基于朴素贝叶斯分类器，用标注好的"积极/消极"语料训练。将文本的词频特征输入贝叶斯模型，输出属于"积极类"的概率。

## 3. 数学公式

### 朴素贝叶斯情感分类
$$
P(\text{积极} | \text{文本}) = \frac{P(\text{文本} | \text{积极}) \cdot P(\text{积极})}{P(\text{文本})}
$$

假设词之间独立：
$$
P(\text{文本} | \text{积极}) = \prod_{w \in \text{文本}} P(w | \text{积极})
$$

最终输出分数就是 $P(\text{积极}|\text{文本})$，即上文"snowNLP(text).sentiments"的返回值。

## 4. 使用过程

**安装**：`pip install snownlp`

**基本使用**：
```python
from snownlp import SnowNLP

# 单条文本
text = "公司业绩大幅增长，管理层对未来发展充满信心"
s = SnowNLP(text)
print(f"情感得分: {s.sentiments:.3f}")  # >0.6 说明模型判断为积极

# 批量处理
texts = ["营收下滑，利润承压", "新产品获得重大突破"]
scores = [SnowNLP(t).sentiments for t in texts]
```

## 5. 应用场景

1. **财报文本情感量化**（原书3.2节）：提取MD&A章节→SnowNLP打分→作为业绩预测模型的文本特征输入
2. **舆情监控**（原书3.14节）：实时抓取新闻→SnowNLP批量打分→正面舆情积累→买入信号
3. **社交媒体情绪**：雪球/股吧帖子情感分析→散户情绪指标
4. **研报观点提取**：券商研报分段情感分析→判断分析师真实态度

## 6. 优缺点分析

**优点**：轻量(纯Python、无GPU)、中文原生、上手快、0训练成本
**缺点**：准确率不如BERT等深度学习模型、对金融专业语言理解有限、训练语料偏电商(非金融)

**替代方案**：
| 工具 | 适用场景 | 优势 |
|------|----------|------|
| SnowNLP | 快速原型、轻量部署 | 快、无GPU |
| BERT微调 | 金融级准确率 | 最高精度 |
| 大模型API | 复杂语义 | 理解能力最强 |

## 7. 调库实现

```python
"""
SnowNLP 情感分析在量化中的应用
批量处理财经文本+作为特征输入ML模型
"""
import numpy as np
import pandas as pd
from snownlp import SnowNLP

# ========== 1. 模拟财经文本数据 ==========
texts = [
    "公司上半年营收增长30%，净利润翻倍，管理层表示将继续加大研发投入",
    "受行业周期影响，公司业绩出现下滑，但管理层对下半年复苏持乐观态度",
    "由于原材料价格上涨和竞争加剧，公司利润空间受到挤压，前景不容乐观",
    "公司新产品获得重大突破，预计将为明年贡献显著增量收入",
    "控股股东宣布减持计划，市场信心受到一定冲击",
    "公司中标重大工程项目，合同金额超过10亿元，对未来业绩形成有力支撑",
    "财报显示应收账款大幅增加，经营性现金流为负，引发市场担忧",
    "行业政策利好频出，公司作为龙头企业将充分受益于政策红利",
    "核心技术人员离职，研发进度可能受到影响",
    "公司回购股份彰显信心，估值处于历史低位具备安全边际",
]

# ========== 2. 批量情感分析 ==========
results = []
for text in texts:
    s = SnowNLP(text)
    score = s.sentiments
    # 分类
    if score > 0.6:
        label = '积极'
    elif score < 0.4:
        label = '消极'
    else:
        label = '中性'
    results.append({
        '文本': text[:40] + '...',
        '情感得分': round(score, 3),
        '情感标签': label,
        '关键词': s.keywords(3),
    })

df = pd.DataFrame(results)
print("SnowNLP 财经文本情感分析结果:")
print(df.to_string(index=False))

# ========== 3. 统计 ==========
print(f"\n积极占比: {(df['情感标签']=='积极').mean():.0%}")
print(f"中性占比: {(df['情感标签']=='中性').mean():.0%}")
print(f"消极占比: {(df['情感标签']=='消极').mean():.0%}")
print(f"平均情感分: {df['情感得分'].mean():.3f}")

# ========== 4. 情感得分作为交易信号 ==========
# 假设：连续N天舆情平均情感分>0.6 → 看多
daily_scores = df['情感得分'].rolling(3).mean()
signal = (daily_scores > 0.6).astype(int)
print(f"\n基于情感的信号: 买入信号数={signal.sum()}")
```

## 8. 手工代码实现

```python
"""
SnowNLP 核心情感分析的简化手工实现
基于朴素贝叶斯的思路
"""
import numpy as np
from collections import Counter


class SimpleSentimentNaiveBayes:
    """极简朴素贝叶斯情感分类器(仅供教学)"""

    def __init__(self):
        # 预设的情感词典
        self.pos_words = {'增长', '利好', '突破', '超预期', '回升', '盈利',
                         '中标', '回购', '增持', '分红', '创新高', '改善'}
        self.neg_words = {'下滑', '亏损', '减持', '下跌', '压力', '风险',
                         '利空', '处罚', '退市', '诉讼', '减值'}

    def tokenize(self, text):
        """简单的2-gram分词"""
        chars = list(text)
        tokens = []
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i+1])
        for i in range(len(chars) - 3):
            tokens.append(''.join(chars[i:i+4]))
        return tokens

    def predict(self, text):
        """预测情感得分(0-1)"""
        tokens = self.tokenize(text)
        pos_count = sum(1 for t in tokens if any(w in t for w in self.pos_words))
        neg_count = sum(1 for t in tokens if any(w in t for w in self.neg_words))
        total = pos_count + neg_count
        return pos_count / total if total > 0 else 0.5


# 测试
if __name__ == '__main__':
    nb = SimpleSentimentNaiveBayes()
    texts = ["业绩大幅增长前景乐观", "利润下滑面临压力"]
    for t in texts:
        score = nb.predict(t)
        print(f"'{t}' -> 情感得分: {score:.2f}")
```

## 9-14. 综合章节

**常见问题**：
- SnowNLP对金融术语不敏感(默认训练语料是电商评论)→用金融语料重新训练或改用BERT金融版
- 长文本情感被稀释(1000字文章中只有一段负面但整体得分中性)→先分段分析再汇总
- "利空出尽"/"否极泰来"等反讽表达被误判→需要上下文理解能力(建议配对大模型)
- 安装依赖问题→`pip install snownlp`一键安装即可

**关键公式**：朴素贝叶斯 P(积极|文本) = P(文本|积极) × P(积极) / P(文本)

**学习路径**：SnowNLP快速上手→文本预处理→情感特征构建→BERT金融微调→大模型API情感分析

**练习题**：
- 基础：一段文本SnowNLP情感得分为0.85。这是什么含义？→模型判断该文本有85%的概率属于"积极"类别，整体偏正面。
- 进阶：为什么不能用单一新闻的情感分直接做交易？→单条新闻噪声太大(SnowNLP准确率有限)，可能有假新闻、误导性标题。应该用多条新闻的滚动均值、并结合信源可信度加权。
- 开放：如何用SnowNLP构建"市场恐慌指数"？→每日采集财经新闻/社交媒体→计算全部文本的情感分均值→取反(1-score)得到恐慌度→滚动标准化后即"SnowNLP恐慌指数"。与大盘涨跌对比验证有效性。

**推荐资源**：SnowNLP GitHub仓库、原书3.2节完整NLP+财务融合代码
