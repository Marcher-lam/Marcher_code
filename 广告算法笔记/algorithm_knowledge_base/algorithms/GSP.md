# GSP（广义第二价格拍卖）学习文档

## 1. 算法基础认知

GSP（Generalized Second Price，广义第二价格）是目前互联网广告行业最主流的定价机制。核心原理：竞价胜出者按照下一名的 eCPM 来计算自己的扣费价格。Google AdWords 首创并成功应用，平衡了平台收益与广告主体验。

## 2. 核心原理

### GSP 定价公式

$$
CPC_k = \frac{eCPM_{k+1}}{pCTR_k \times 1000} + \delta
$$

参数说明：
- $eCPM_{k+1}$：排名第 $k+1$ 名的广告的 eCPM 值
- $pCTR_k$：排名第 $k$ 名的广告的预估点击率
- $\delta$：最小加价单位（通常为 0.01 元）

### 四种定价机制对比

| 机制类型 | 获胜规则 | 扣费规则 |
|---------|---------|---------|
| 第一价格（FPA） | 出价最高者 | 按最高出价扣费 |
| 第二价格（SPA） | 出价最高者 | 按次高出价+0.01 |
| GSP（广义第二价格） | 多广告位排序 | 各位置按下一位出价 |
| VCG | 多广告位排序 | 基于边际贡献扣费 |

## 3. 数学公式与推导

### Step 1：计算 eCPM

$$
eCPM = pCTR \times Bid \times 1000
$$

### Step 2：排序确定位置

$$
Rank = \text{sort}(eCPM, \text{desc})
$$

### Step 3：计算扣费价格

$$
CPC_k = \frac{eCPM_{k+1}}{pCTR_k \times 1000} + \delta
$$

### Step 4：边界处理

$$
CPC_{last} = \max(Reserve, \delta)
$$

### 关键概念

- **保留价（Reserve Price）**：最低竞价门槛，$Cost = \max(GSP扣费, Reserve)$
- **挤压系数（Squash Factor）**：$eCPM_{adj} = pCTR^\alpha \times Bid \times 1000$，$\alpha \in (0,1]$

## 4. 训练/运行过程讲解

1. 各广告主提交出价 $Bid_i$
2. 系统预估每个广告的点击率 $pCTR_i$
3. 计算每个广告的 $eCPM_i = pCTR_i \times Bid_i \times 1000$
4. 按 $eCPM$ 降序排列，分配广告位
5. 第 $k$ 名广告主的实际扣费 $CPC_k = eCPM_{k+1} / (pCTR_k \times 1000) + \delta$
6. 最后一名扣费取保留价

## 5. 应用场景

- 搜索广告排名（Google、百度）
- 信息流广告排序（头条、腾讯）
- 多广告位展示场景
- 当前主流广告平台的定价机制

### GSP 定价实例

| 广告主 | 出价(Bid) | pCTR | eCPM | 排名 | 实际CPC |
|--------|----------|------|------|------|---------|
| 广告主A | ¥2.0 | 5% | 100 | 第1名 | ¥1.61 |
| 广告主B | ¥1.6 | 5% | 80 | 第2名 | ¥0.81 |
| 广告主C | ¥1.0 | 4% | 40 | 第3名 | ¥0.01 |

**A 的扣费**：$CPC_A = eCPM_B / (pCTR_A \times 1000) + 0.01 = 80 / 50 + 0.01 = 1.61$ 元

**B 的扣费**：$CPC_B = eCPM_C / (pCTR_B \times 1000) + 0.01 = 40 / 50 + 0.01 = 0.81$ 元

## 6. 优缺点分析

**优点**：
- 激励效果好：广告主有动力出真实价格
- 计算简单：比 VCG 容易实现
- 收益平衡：平台收益和广告主体验的良好平衡
- 历史原因：Google AdWords 首创并成功应用

**缺点**：
- 不具备严格激励相容性（多广告位场景下不等于 VCG）
- 广告主可能通过策略性出价获利（与 VCG 相比的理论劣势）
- 依赖 pCTR 估计的准确性

## 7. 调库实现（Python + 完整代码 + 注释）

```python
def gsp_auction(ads, reserve_price=0.01, delta=0.01):
    """
    ads: list of dict with 'bid', 'pctr', 'ad_id'
    Returns: ranked ads with cpc
    """
    for ad in ads:
        ad['ecpm'] = ad['bid'] * ad['pctr'] * 1000
    ads_sorted = sorted(ads, key=lambda x: x['ecpm'], reverse=True)
    for i, ad in enumerate(ads_sorted):
        ad['rank'] = i + 1
        if i < len(ads_sorted) - 1:
            next_ecpm = ads_sorted[i + 1]['ecpm']
            ad['cpc'] = next_ecpm / (ad['pctr'] * 1000) + delta
        else:
            ad['cpc'] = max(reserve_price, delta)
        ad['cpc'] = min(ad['cpc'], ad['bid'])
    return ads_sorted

ads = [
    {'ad_id': 'A', 'bid': 2.0, 'pctr': 0.05},
    {'ad_id': 'B', 'bid': 1.6, 'pctr': 0.05},
    {'ad_id': 'C', 'bid': 1.0, 'pctr': 0.04},
]
result = gsp_auction(ads)
for ad in result:
    print(f"Rank {ad['rank']}: {ad['ad_id']}, CPC={ad['cpc']:.2f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
def gsp_pricing(bids, pctrs, reserve=0.01, delta=0.01):
    ecpm_list = [(b * p * 1000, b, p, i) for i, (b, p) in enumerate(zip(bids, pctrs))]
    ecpm_list.sort(key=lambda x: x[0], reverse=True)
    results = []
    for rank, (ecpm, bid, pctr, idx) in enumerate(ecpm_list):
        if rank < len(ecpm_list) - 1:
            next_ecpm = ecpm_list[rank + 1][0]
            cpc = next_ecpm / (pctr * 1000) + delta
        else:
            cpc = max(reserve, delta)
        cpc = min(cpc, bid)
        results.append({'idx': idx, 'rank': rank + 1, 'cpc': cpc, 'ecpm': ecpm})
    return results
```

## 9. 可视化与结果理解

- 绘制 eCPM 排名 vs 实际 CPC 的柱状对比图
- 展示不同 pCTR 下同一出价广告的实际扣费差异
- 对比 GSP 与第一价格拍卖的平台收入差异

```python
import numpy as np
import matplotlib.pyplot as plt

ads = [
    {'name': '广告主A', 'bid': 2.0, 'pctr': 0.05},
    {'name': '广告主B', 'bid': 1.6, 'pctr': 0.05},
    {'name': '广告主C', 'bid': 1.0, 'pctr': 0.04},
]

for ad in ads:
    ad['ecpm'] = ad['bid'] * ad['pctr'] * 1000

ads_sorted = sorted(ads, key=lambda x: x['ecpm'], reverse=True)
delta = 0.01
for i, ad in enumerate(ads_sorted):
    if i < len(ads_sorted) - 1:
        next_ecpm = ads_sorted[i + 1]['ecpm']
        ad['cpc'] = next_ecpm / (ad['pctr'] * 1000) + delta
    else:
        ad['cpc'] = max(0.01, delta)
    ad['cpc'] = min(ad['cpc'], ad['bid'])

names = [a['name'] for a in ads_sorted]
bids = [a['bid'] for a in ads_sorted]
pctrs = [a['pctr'] for a in ads_sorted]
ecpms = [a['ecpm'] for a in ads_sorted]
cpcs = [a['cpc'] for a in ads_sorted]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x = np.arange(len(names))
bar_width = 0.5

axes[0].bar(x, ecpms, bar_width, color=['#2196F3', '#4CAF50', '#FF9800'], edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, fontsize=11)
axes[0].set_ylabel('eCPM', fontsize=12)
axes[0].set_title('eCPM Ranking (决定排名)', fontsize=13)
for i, v in enumerate(ecpms):
    axes[0].text(i, v + 1, f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - bar_width / 3, bids, bar_width / 1.5, label='Bid (出价)', color='#2196F3', alpha=0.8)
axes[1].bar(x + bar_width / 3, cpcs, bar_width / 1.5, label='CPC (实际扣费)', color='#F44336', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, fontsize=11)
axes[1].set_ylabel('Price (¥)', fontsize=12)
axes[1].set_title('Bid vs GSP CPC (GSP扣费更低)', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)
for i in range(len(names)):
    axes[1].text(i - bar_width / 3, bids[i] + 0.03, f'¥{bids[i]:.1f}', ha='center', fontsize=9)
    axes[1].text(i + bar_width / 3, cpcs[i] + 0.03, f'¥{cpcs[i]:.2f}', ha='center', fontsize=9)

pctr_pct = [p * 100 for p in pctrs]
axes[2].bar(x, pctr_pct, bar_width, color=['#2196F3', '#4CAF50', '#FF9800'], edgecolor='black')
axes[2].set_xticks(x)
axes[2].set_xticklabels(names, fontsize=11)
axes[2].set_ylabel('pCTR (%)', fontsize=12)
axes[2].set_title('预估点击率 pCTR', fontsize=13)
for i, v in enumerate(pctr_pct):
    axes[2].text(i, v + 0.1, f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('GSP 广义第二价格拍卖 — 排序与定价可视化', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gsp_auction_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **平台收入**：$\text{Revenue} = \sum_k CPC_k \times pCTR_k \times impressions$
- **广告主效用**：$U_i = (Bid_i - CPC_i) \times clicks$
- **社会福利**：$\sum_i value_i \times clicks_i$
- 通常 GSP 的平台收入略低于第一价格，但广告主体验更好

## 11. 常见问题与易错点

- GSP 不等于 VCG，在多广告位场景下不保证激励相容
- pCTR 估计误差会直接影响定价公平性
- 保留价设置过低会导致低质量广告消耗预算
- 挤压系数 $\alpha$ 的调节会影响高/低 CTR 广告的排序倾向

## 12. 学习总结

GSP 是当前广告行业主流定价机制，核心是"按下一名 eCPM 定价"。与第一价格相比更激励真实出价，与 VCG 相比计算更简单。理解 GSP 是理解整个广告定价体系的基础。

## 13. 练习题与思考题（含答案）

**Q1**: 为什么广告主 A 出价 ¥2.0 但只支付 ¥1.61？
> A1: GSP 机制下，胜出者按下一名的 eCPM 折算自己的 CPC，而非自己的出价，这激励真实出价。

**Q2**: 如果广告主 A 的 pCTR 从 5% 降到 2%，会发生什么？
> A2: A 的 eCPM = 2% × 2.0 × 1000 = 40，排名将降到第三，因为 B 的 eCPM = 80 > 40。

**Q3**: GSP 与 VCG 的核心区别是什么？
> A3: GSP 按下一名出价定价，VCG 按对他人造成的边际损失定价。VCG 严格激励相容，GSP 在多广告位场景下不保证。

## 14. 学习路径建议

1. 先学习拍卖理论基础（第二价格拍卖）
2. 理解 eCPM = pCTR × Bid × 1000 的含义
3. 学习 GSP 定价机制和实例计算
4. 进阶：学习 VCG 机制、机制设计理论、挤压系数调优
