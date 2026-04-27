# 竞价强度因子与RankIC 学习文档

> 从开盘前15分钟的竞价数据中挖掘主力资金意图，用统计检验锁定有效因子。
> 来源线索：本节内容根据原书中关于"竞价阶段量价因子的挖掘"（第3章3.9节）的相关内容整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：竞价强度因子是通过集合竞价阶段的成交量和价格偏离度构造的预测因子，用 RankIC（秩相关系数）检验其与未来收益的关系的有效性。

**直觉类比**：开店前的排队情况——如果开门前排队人数（竞价量）远超平时，且排队人愿意出高价（价格偏离），那么开门后大概率生意红火（开盘上涨）。

**背景**：A股开盘前 9:15-9:25 是集合竞价阶段，主力资金常在此布局。市场微观结构学研究表明，竞价量价信息对开盘后短周期收益有显著预测能力。中信、华泰等券商的因子报告均证实了竞价因子的有效性。

**算法定位**：量化因子 / 贝塔策略 / 统计检验 / 日内交易。

**前置知识**：A股集合竞价机制、Spearman 秩相关、假设检验基础。

## 2. 核心原理

**竞价强度因子定义**：

$$
\text{AuctionStrength} = \frac{\text{AuctionVolume}}{\text{PrevAvgVolume}} \times \frac{\text{AuctionPrice} - \text{PrevClose}}{\text{PrevClose}}
$$

- 第一项（竞价成交量/昨日日均量）：衡量竞价阶段资金参与热度
- 第二项（价格偏离度）：衡量竞价价格相对昨收的涨跌方向

因子为正值大 → 带量高开，主力积极做多；因子为负值大 → 带量低开，主力出货。

**RankIC 检验原理**：
- RankIC = Spearman 秩相关系数(因子值, 未来收益)
- 值域 [-1, 1]，正值说明因子与收益正相关
- |RankIC| > 0.03 通常认为因子有效
- ICIR = mean(RankIC) / std(RankIC)，衡量因子稳定性，> 0.5 较理想

**工作流程**：
数据采集(昨收/昨量/竞价数据/开盘30min收益) → 因子计算 → 横截面 RankIC 检验 → 滚动窗口验证 → 信号生成

## 3. 数学公式与推导

### 3.1 竞价强度因子
$$
F_t = \frac{V_{\text{auction}, t}}{\bar{V}_{\text{prev}, 5D}} \times \frac{P_{\text{auction}, t} - P_{\text{prev}, t}}{P_{\text{prev}, t}}
$$

均量建议用 5 日均量更平滑。

### 3.2 Spearman RankIC
$$
\text{RankIC} = \frac{\text{cov}(R_F, R_R)}{\sigma_{R_F} \cdot \sigma_{R_R}}
$$

其中 $R_F$ 是因子值的排名，$R_R$ 是未来收益的排名。

### 3.3 ICIR
$$
\text{ICIR} = \frac{\overline{\text{RankIC}}}{\sigma_{\text{RankIC}}}
$$

### 3.4 统计显著性 t 检验
$$
t = \frac{\overline{\text{RankIC}}}{\sigma_{\text{RankIC}} / \sqrt{n}}
$$

$|t| > 2$ 表示在 95% 置信水平下显著。

## 4. 训练过程讲解

### 4.1 数据准备
- 昨日收盘价、昨日成交量、前N日均量
- 今日竞价成交量、竞价价格（开盘价）
- 开盘后30分钟价格
- 数据清洗：剔除停牌、涨跌停、异常波动的样本

### 4.2 因子计算
- 用 5 日均量替代昨日单日成交量（更稳健）
- 对因子做横截面标准化（去均值除标准差），使不同股票可比

### 4.3 RankIC 检验
- 每交易日计算一组股票的因子值 → 与当日开盘后30分钟收益的 Spearman 秩相关
- 统计 RankIC 的时间序列均值、标准差、ICIR
- 滚动窗口验证（而非单次测试）

### 4.4 关键参数
| 参数 | 推荐 |
|------|------|
| 均量窗口 | 5 日 |
| 收益周期 | 开盘后 30 分钟 |
| 检验窗口 | 滚动 60 交易日 |
| IC 有效阈值 | \|RankIC\| > 0.03 |

## 5. 应用场景

1. **高开套利**：因子强正+高开→开盘做多，持有至30分钟后
2. **低开套利**：因子强负+低开→开盘做空/卖出
3. **主力监控**：持续监控竞价强度因子异动的股票
4. **多因子择时**：竞价因子作为开盘方向判断的辅助

**适用市场**：A股（有集合竞价）、科创板/创业板；**不适用**：美股盘前交易机制不同需调整。

## 6. 优缺点分析

### 优点
- 前瞻性：竞价在开盘前，信号先于盘中交易
- 直观：量×价偏离的逻辑简单清晰
- 统计严谨：RankIC + ICIR 提供客观有效性标准
- 互补性：与盘中因子结合可提升整体胜率

### 缺点
- 极端行情失效：开盘涨停/跌停时空方/多方无法交易
- 流动性要求：小盘股竞价数据噪声大
- 参数敏感性：均量窗口和收益周期的选取影响结果

## 7. 调库实现

```python
"""
竞价强度因子 + RankIC 检验
pandas + numpy + scipy 完整实现
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

np.random.seed(42)

# ========== 1. 模拟数据：30只股票 × 120个交易日 ==========
n_stocks = 30
n_days = 120

stock_codes = [f'{600000+i}' for i in range(n_stocks)]

# 生成面板数据
data = []
for t in range(n_days):
    # 生成有预测能力的竞价因子（与未来收益相关）
    true_ic = 0.08 + np.random.randn() * 0.03  # 每天的真实 IC 有波动
    base_factor = np.random.randn(n_stocks)

    prev_close = np.random.uniform(5, 50, n_stocks)
    auction_price = prev_close + np.random.randn(n_stocks) * 0.3
    auction_vol = np.random.exponential(50000, n_stocks)
    prev_avg_vol = np.random.exponential(40000, n_stocks)

    # 竞价强度因子
    price_dev = (auction_price - prev_close) / prev_close
    auction_strength = (auction_vol / prev_avg_vol) * price_dev

    # 未来收益 = 因子信号 + 噪声（模拟因子有一定预测力）
    future_ret = true_ic * auction_strength + np.random.randn(n_stocks) * 0.02

    for s in range(n_stocks):
        data.append({
            'date': t, 'stock': stock_codes[s],
            'auction_strength': auction_strength[s],
            'future_ret_30min': future_ret[s],
        })

df = pd.DataFrame(data)

# ========== 2. 计算每日 RankIC ==========
dates = sorted(df['date'].unique())
ic_records = []

for dt in dates:
    day_df = df[df['date'] == dt]
    if len(day_df) > 10:
        ic, p_value = spearmanr(day_df['auction_strength'], day_df['future_ret_30min'])
        ic_records.append({'date': dt, 'rank_ic': ic, 'p_value': p_value})

ic_df = pd.DataFrame(ic_records)

# ========== 3. RankIC 统计分析 ==========
ic_mean = ic_df['rank_ic'].mean()
ic_std = ic_df['rank_ic'].std()
icir = ic_mean / ic_std
t_stat = ic_mean / (ic_std / np.sqrt(len(ic_df)))
sig_ratio = (ic_df['rank_ic'] > 0).mean()

print("=" * 50)
print("竞价强度因子 RankIC 检验结果")
print("=" * 50)
print(f"RankIC 均值:     {ic_mean:.4f}")
print(f"RankIC 标准差:   {ic_std:.4f}")
print(f"ICIR:            {icir:.2f}")
print(f"t 统计量:        {t_stat:.2f}")
print(f"正 IC 比例:      {sig_ratio:.1%}")
print(f"有效(IC>0.03)比例: {(ic_df['rank_ic'] > 0.03).mean():.1%}")

# ========== 4. 分层回测验证 ==========
df['factor_quantile'] = df.groupby('date')['auction_strength'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
layer_returns = df.groupby('factor_quantile')['future_ret_30min'].mean()

print("\n5 层分层回测平均收益:")
for q, ret in layer_returns.items():
    print(f"  Q{q+1}: {ret:.4%}")

# ========== 5. 可视化 ==========
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# RankIC 时序
axes[0].bar(ic_df['date'], ic_df['rank_ic'], color=['g' if x > 0 else 'r' for x in ic_df['rank_ic']], alpha=0.7)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axhline(y=ic_mean, color='b', linestyle='--', label=f'均值={ic_mean:.3f}')
axes[0].set_title('每日 RankIC')
axes[0].set_xlabel('交易日')
axes[0].set_ylabel('RankIC')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RankIC 分布
axes[1].hist(ic_df['rank_ic'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0, color='k', linestyle='-')
axes[1].axvline(x=ic_mean, color='r', linestyle='--', label=f'均值={ic_mean:.3f}')
axes[1].set_title('RankIC 分布')
axes[1].set_xlabel('RankIC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 分层收益
layer_names = [f'Q{q+1}' for q in layer_returns.index]
axes[2].bar(layer_names, layer_returns.values, color='steelblue', alpha=0.8)
axes[2].set_title('5层分层回测收益（单调性检验）')
axes[2].set_ylabel('平均收益%')
for i, v in enumerate(layer_returns.values):
    axes[2].text(i, v + 0.0001, f'{v:.3%}', ha='center', fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 8. 手工代码实现

```python
"""
竞价强度因子 + RankIC 手工实现
"""
import numpy as np


class AuctionStrengthFactor:
    """竞价强度因子"""

    def __init__(self, ma_window=5):
        self.ma_window = ma_window

    def calc(self, auction_vol, prev_vols, auction_price, prev_close):
        """计算竞价强度因子"""
        if len(prev_vols) >= self.ma_window:
            avg_vol = np.mean(prev_vols[-self.ma_window:])
        else:
            avg_vol = prev_vols[-1] if len(prev_vols) > 0 else auction_vol
        price_dev = (auction_price - prev_close) / prev_close
        return (auction_vol / avg_vol) * price_dev if avg_vol > 0 else 0


def spearman_rank_ic(x, y):
    """手写 Spearman 秩相关系数"""
    n = len(x)
    # 计算秩
    rank_x = np.zeros(n)
    rank_y = np.zeros(n)
    for i in range(n):
        rank_x[i] = np.sum(x < x[i]) + 0.5 * np.sum(x == x[i])
        rank_y[i] = np.sum(y < y[i]) + 0.5 * np.sum(y == y[i])
    # 皮尔逊相关就是秩相关
    mx, my = np.mean(rank_x), np.mean(rank_y)
    cov = np.sum((rank_x - mx) * (rank_y - my))
    sx = np.sqrt(np.sum((rank_x - mx)**2))
    sy = np.sqrt(np.sum((rank_y - my)**2))
    return cov / (sx * sy + 1e-10) if sx > 0 and sy > 0 else 0


# ========== 测试 ==========
if __name__ == '__main__':
    np.random.seed(42)
    factor = AuctionStrengthFactor(ma_window=5)

    # 模拟测试
    scores = []
    for i in range(20):
        auction_vol = np.random.exponential(50000)
        prev_vols = [np.random.exponential(40000) for _ in range(10)]
        auction_price = 20 + np.random.randn()
        prev_close = 20
        scores.append(factor.calc(auction_vol, prev_vols, auction_price, prev_close))

    # 手写 RankIC 验证
    x = np.array(scores)
    y = x * 0.5 + np.random.randn(20) * 0.1
    ic = spearman_rank_ic(x, y)
    print(f"手写 Spearman RankIC: {ic:.4f}")
```

## 9-10. 可视化与评估

第 7 节代码已包含完整的可视化和评估。解读关键：
- RankIC 大部分为正且均值为正 → 因子有效
- 分层回测呈单调递增（Q1-Q5 收益递增）→ 因子区分度高
- ICIR > 0.5 → 因子稳定性可接受

## 11. 常见问题与易错点

- **数据对齐**：昨日均量与今日竞价必须正确对应日期
- **极端波动处理**：竞价量为 0 或价格异常的样本要去极值
- **样本外验证缺失**：不能只在样本内看 IC，必须滚动窗口验证
- **流动性忽视**：对换手率极低的股票，竞价数据不稳定

## 12. 学习总结

**核心回顾**：竞价强度因子用开盘前的量价信息预测开盘后方向，RankIC 提供统计严谨性。因子逻辑简单但实际检验需要严格的统计流程。

**关键公式**：$F = \frac{V_{\text{auc}}}{\bar{V}_{\text{prev}}} \times \frac{P_{\text{auc}}-P_{\text{prev}}}{P_{\text{prev}}}$

**后续**：扩展到更多竞价指标（撤单率、竞价匹配量）、用机器学习融合多个竞价因子。

## 13. 练习题（含答案）

**题1**（基础）：因子某日竞价量=昨日均量的 3 倍，竞价价格比昨收高 2%。因子值=？
**参考答案**：3 × 0.02 = 0.06。因子正值说明带量高开，看多。

**题2**（进阶）：RankIC 均值=0.04、标准差=0.06、样本数=100天。ICIR和t统计量各是多少？因子是否有效？
**参考答案**：ICIR=0.04/0.06=0.67(>0.5，稳定性可接受)；t=0.04/(0.06/√100)=6.67(>2，高度显著)。因子有效。

## 14. 学习路径建议

**前置**：A股集合竞价机制、Spearman 秩相关

**平行**：开盘缺口因子、隔夜收益率因子、盘前情绪指标

**进阶**：多竞价因子 ML 融合、高频盘口数据结合竞价信号

**推荐资源**：华泰/中信竞价因子研报、scipy.stats.spearmanr 文档
