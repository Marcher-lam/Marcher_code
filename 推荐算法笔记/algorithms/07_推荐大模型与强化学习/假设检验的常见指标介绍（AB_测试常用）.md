# 面试题：假设检验的常见指标介绍（A/B 测试常用）

# 面试题：假设检验的常见指标介绍（A/B 测试常用）

# 一、假设检验核心概念解释

# 1. 显著性水平 (α，Significance Level)

定义：在假设检验中，我们愿意接受的犯第一类错误（Type I Error）的最大概率。

第一类错误：原假设（H⋅）是真的，但我们错误地拒绝了它（假阳性）。

常用的显著性水平  

<table><tr><td>显著性水平(a)</td><td>含义</td><td>适用场景</td><td>严格程度</td></tr><tr><td>0.01 (1%)</td><td>允许1%的假阳性概率</td><td>医学研究、高风险决策</td><td>非常严格</td></tr><tr><td>0.05 (5%)</td><td>允许5%的假阳性概率</td><td>行业标准、A/B测试</td><td>标准</td></tr><tr><td>0.10 (10%)</td><td>允许10%的假阳性概率</td><td>探索性研究</td><td>较宽松</td></tr><tr><td>0.20 (20%)</td><td>允许20%的假阳性概率</td><td>初步筛选、快速实验</td><td>宽松</td></tr></table>

选择建议：α越小，检验越严格，越难拒绝原假设；α越大，检验越宽松，越容易检测到差异 (但假阳性风险增加)。

#  原假设 (H⋅) 与备择假设 (H⋅/Hₐ)定义：

 H⋅ (原假设) 通常表示"无效应"、"无差异"，如：新方案和旧方案没有区别。  
 H⋅/Hₐ (备择假设)通常表示"有效应"、"有差异"，如：新方案比旧方案更好。

#  两类错误定义：

 Type I Error (第一类错误) H⋅ 为真时，错误地拒绝了 H⋅（假阳性），概率定位 α。  
 Type II Error (第二类错误) H⋅ 为假时，错误地接受了 H⋅（假阴性），概率定位 β。

# 2. 置信区间 (Confidence Interval, CI)

定义：一个区间估计，表示我们有一定的信心（置信水平）认为真实参数值落在这个区间内。

例如： $9 5 \%$ 置信区间 $[ 1 . 2 \%$ , $3 . 5 \% ]$ 表示我们有 $9 5 \%$ 的把握，真实值在 $1 . 2 \%$ 到 $3 . 5 \%$ 之间。

# 核心公式

$$
\text {置 信 区 间} = \text {点 估 计} \pm \mathrm {Z} _ {\alpha / 2} \times \text {标 准 误 差}
$$

$$
\text {下 界} = \mathrm {X} ^ {-} - \mathrm {Z} _ {\alpha / 2} \times \mathrm {S E}
$$

$$
\text {上 界} = \mathrm {X} ^ {-} + \mathrm {Z} _ {\alpha / 2} \times \mathrm {S E}
$$

# 置信水平与显著性水平的关系

$$
\text {置 信 水 平} = 1 - \alpha
$$

<table><tr><td>显著性水平(a)</td><td>置信水平(1-a)</td><td>解释</td></tr><tr><td>0.01</td><td>99%</td><td>有99%的把握真实值在区间内</td></tr><tr><td>0.05</td><td>95%</td><td>有95%的把握真实值在区间内</td></tr><tr><td>0.10</td><td>90%</td><td>有90%的把握真实值在区间内</td></tr><tr><td>0.20</td><td>80%</td><td>有80%的把握真实值在区间内</td></tr></table>

# 3. Z 分数 (Z-score)

定义：标准正态分布的分位数，表示一个值距离均值有多少个标准差。

用于将原始数据标准化，便于比较不同尺度的数据。

# 计算公式

$$
\mathbf {z} = \left(\mathbf {x} - \mu\right) / \sigma
$$

$$
\begin{array}{l} \mathrm {X} = \text {观 测 值} \\ \mu = \text {总 体 均 值} \\ \sigma = \text {总 体 标 准 差} \\ \end{array}
$$

# 显著性水平与Z分数对应表

<table><tr><td>显著性水平(a)</td><td>置信水平</td><td>Z分数 (Zα/2)</td><td>计算说明</td></tr><tr><td>0.01</td><td>99%</td><td>2.576</td><td>P(Z ≤ 2.576) = 0.995</td></tr><tr><td>0.05</td><td>95%</td><td>1.96</td><td>P(Z ≤ 1.96) = 0.975</td></tr><tr><td>0.10</td><td>90%</td><td>1.645</td><td>P(Z ≤ 1.645) = 0.95</td></tr><tr><td>0.20</td><td>80%</td><td>1.28</td><td>P(Z ≤ 1.28) = 0.90</td></tr></table>

记忆技巧：a越小 $ Z$ 分数越大 置信区间越宽 越难判断显著

# 4. P 值 (P-value)

 P 值定义：在原假设 H⋅ 为真的条件下，观察到当前样本结果（或更极端结果）的概率。  
 通俗理解：P 值越小，说明当前观察到的结果越"不寻常"，越有理由拒绝原假设。

# 判断规则

√P≤α:拒绝H。

结果具有统计显著性

有足够证据支持备择假设 H

XP>α:不能拒绝 H。

结果不具有统计显著性

没有足够证据拒绝原假设

# P值与显著性水平的关系

<table><tr><td>P值范围</td><td>在a=0.05下</td><td>在a=0.01下</td><td>在a=0.10下</td><td>常用表述</td></tr><tr><td>P&lt;0.001</td><td>显著✓</td><td>显著✓</td><td>显著✓</td><td>极其显著***</td></tr><tr><td>0.001≤P&lt;0.01</td><td>显著✓</td><td>显著✓</td><td>显著✓</td><td>非常显著**</td></tr><tr><td>0.01≤P&lt;0.05</td><td>显著✓</td><td>不显著×</td><td>显著✓</td><td>显著*</td></tr><tr><td>0.05≤P&lt;0.10</td><td>不显著×</td><td>不显著×</td><td>显著✓</td><td>边缘显著</td></tr><tr><td>P≥0.10</td><td>不显著×</td><td>不显著×</td><td>不显著×</td><td>不显著</td></tr></table>

A注意：P值不是"原假设为真的概率"！P值是在假设H。为真的前提下，观察到当前或更极端结果的概率。

# 5. 检验统计量 (Test Statistic)

检验统计量定义：根据样本数据计算出的一个数值，用于判断是否拒绝原假设。

# 常见检验方法：

<table><tr><td>检验方法</td><td>公式</td><td>适用条件</td></tr><tr><td>Z 检验</td><td>Z = (X-μθ) / (σ/√n)</td><td>大样本 (n≥30)
已知总体方差 σ²</td></tr><tr><td>t 检验</td><td>t = (X-μθ) / (s/√n)</td><td>小样本
未知总体方差</td></tr><tr><td>卡方检验</td><td>x² = Σ (O-E)² / E</td><td>分类变量
独立性/拟合优度检验</td></tr><tr><td>双样本 t 检验</td><td>t = (X̄ - X̅̄) / √(s1² / n1 + s2² / n2)</td><td>比较两组均值
A/B 测试常用</td></tr></table>

# 二、显著性水平与 Z 分数的对应关系

显著性水平、置信区间、Z分数对照表  

<table><tr><td>显著性水平(a)</td><td>置信水平</td><td>Z分数(双侧)</td><td>Z分数(单侧)</td><td>应用场景</td></tr><tr><td>0.01</td><td>99%</td><td>2.576</td><td>2.326</td><td>医学、高风险决策</td></tr><tr><td>0.05</td><td>95%</td><td>1.96</td><td>1.645</td><td>行业标准</td></tr><tr><td>0.10</td><td>90%</td><td>1.645</td><td>1.28</td><td>探索性研究</td></tr><tr><td>0.20</td><td>80%</td><td>1.28</td><td>0.84</td><td>快速筛选</td></tr></table>

# 三、A/B 测试中的应用示例

# 场景：A/B 测试-评估新推荐算法对GMV的影响

# I实验数据

·对照组 (A)：样本量 $n _ { 1 } = 1 0 0 0 0$ ，人均GMV=￥50.0，标准差 $\mathsf { s } _ { 1 } = \yen 30$   
·实验组 (B)：样本量 $n _ { 2 } = 1 0 0 0 0$ ，人均GMV $=$ ￥52.5，标准差 $S _ { 2 } = \yen 32$   
·提升率 $=$ (52.5- 50) $1 5 0 = 5 \%$

# Step 1:设定假设

H $\mathsf { \Pi } \mu \_ { - } \mathsf { B } = \mu \_ { - } \mathsf { A }$ (新算法无效果)

$\mathsf { H } _ { 1 } \colon \mathsf { H } \_ { \mathsf { B } } > \mathsf { \mu } \_ { \mathsf { A } }$ (新算法有正向效果)

# Step 2:选择显著性水平

$\mathtt { a } = 0 . 0 5$ $9 5 \%$ 置信水平)

# Step 3:计算检验统计量

$$
\mathrm {S E} = \sqrt {\left(\mathrm {s} _ {1} ^ {2} / \mathrm {n} _ {1} + \mathrm {s} _ {2} ^ {2} / \mathrm {n} _ {2}\right)} = \sqrt {(9 0 0 / 1 0 0 0 0 + 1 0 2 4 / 1 0 0 0 0)} = \sqrt {0 . 1 9 2 4} \approx 0. 4 3 9
$$

$$
Z = (X _ {-} B - X _ {-} A) / S E = (5 2. 5 - 5 0) / 0. 4 3 9 \approx 5. 6 9
$$

# Step 4:计算P值

查标准正态分布表： $\mathsf { P } ( Z > 5 . 6 9 ) \approx 0 . 0 0 0 0 0 0 1$

P值 $\mathbf { < 0 . 0 0 1 }$

# Step 5:计算置信区间 $( 9 5 \%$

$$
95 \% \mathrm {CI} = 2.5 \pm 1.96 \times 0.439 = [ 1.64, 3.36 ]
$$

# Step 6:做出决策

结论：由于 $\mathsf { P < } 0 . 0 5$ 且置信区间不包含0，拒绝原假设。

新推荐算法对GMV有显著正向影响，人均GMV提升约￥2.5（ $9 5 \%$ Cl: ?1.64~¥3.36)，提升率 $5 \%$

# 四、Python 代码实现：A/B 测试完整分析

```python
import numpy as np
from scipy import stats

def ab_test_analysis(control, treatment, alpha=0.05):
    n1, n2 = len(control), len(treatment)
    mean1, mean2 = np.mean(control), np.mean(treatment)
    std1, std2 = np.std(control, ddof=1), np.std(treatment, ddof=1)
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    z_score = (mean2 - mean1) / se
    p_value = 1 - stats.norm.cdf(abs(z_score))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = (mean2 - mean1) - z_crit * se
    ci_upper = (mean2 - mean1) + z_crit * se
    lift = (mean2 - mean1) / mean1 * 100

    print(f"=== A/B 测试分析报告 ===")
    print(f"对照组: 均值={mean1:.2f}, 标准差={std1:.2f}, 样本量={n1}")
    print(f"实验组: 均值={mean2:.2f}, 标准差={std2:.2f}, 样本量={n2}")
    print(f"提升率: {lift:.2f}%")
    print(f"Z分数: {z_score:.4f}")
    print(f"P值: {p_value:.6f}")
    print(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"显著性判定(α={alpha}): {'显著 ✓' if p_value < alpha else '不显著 ✗'}")
    return {"z_score": z_score, "p_value": p_value, "ci": (ci_lower, ci_upper), "significant": p_value < alpha}

np.random.seed(42)
control = np.random.normal(50.0, 30.0, 10000)
treatment = np.random.normal(52.5, 32.0, 10000)
result = ab_test_analysis(control, treatment, alpha=0.05)

print(f"\n=== 样本量估算 ===")
def sample_size_calc(delta, sigma, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * 2 * sigma ** 2) / delta ** 2
    return int(np.ceil(n))

delta = 2.5
sigma = 31.0
n_needed = sample_size_calc(delta, sigma)
print(f"检测最小提升{delta}元（σ={sigma}）所需每组样本量: {n_needed}")
```

# 五、统计功效（Power）与样本量

<table><tr><td>概念</td><td>定义</td><td>推荐值</td></tr><tr><td>统计功效 (1-β)</td><td>当H₁为真时，正确拒绝H₀的概率</td><td>≥0.8</td></tr><tr><td>效应量 (Effect Size)</td><td>两组之间差异的大小（如Cohen's d）</td><td>视业务而定</td></tr><tr><td>最小可检测效应 (MDE)</td><td>在给定样本量和功效下能检测到的最小差异</td><td>由业务决定</td></tr></table>

样本量、功效和效应量三者关系：
- 固定功效，效应量越小，所需样本量越大
- 固定效应量，功效越高，所需样本量越大
- 固定样本量，效应量越大，功效越高

# 六、推荐系统A/B测试的特殊考量

1. 网络效应：用户间存在社交关系时，A/B组不是独立的，需使用聚类随机化或社交网络分析方法

2. 指标选择：
- 短期指标：CTR、CVR、GMV（直接可测量）
- 长期指标：用户留存、LTV（需长期观察）
- 注意短期指标提升不等于长期收益

3. 置换检验：当数据不满足正态假设时（如比例类指标），可用Bootstrap或置换检验代替Z检验

4. 多重比较：同时测试多个指标时，需用Bonferroni校正（α/k）或FDR控制，避免假阳性膨胀

# 七、常见误区

1. 误区："P值小于0.05就说明效果很大"
   - 实际：P值只反映统计显著性（差异是否真实），不反映实际显著性（差异是否重要）。大样本下微小的差异也能得到显著的P值。

2. 误区："置信区间包含0就说没有效果"
   - 实际：只能说"没有足够的证据证明有效果"，不等同于"没有效果"。可能是样本量不足导致检验功效不够。

3. 误区："A/B测试只要样本量够大就行"
   - 实际：还需确保(1) 用户随机分配无偏差；(2) 实验期间无外部干扰；(3) 指标定义与业务目标对齐；(4) 新奇效应消退后再评估。

# 第七章：推荐&大模型&强化学习

# 7.1 推荐+大模型面试题：
