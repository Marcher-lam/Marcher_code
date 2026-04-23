# GSP（广义第二价格拍卖）学习文档

## 1. 算法基础认知

GSP（Generalized Second Price，广义第二价格）是目前互联网广告行业最主流的定价机制。核心原理：竞价胜出者按照下一名的 eCPM 来计算自己的扣费价格。

## 2. 核心原理

### GSP 定价公式

$$
CPC = \frac{eCPM_{下一名}}{pCTR \times 1000} + \delta
$$

参数说明：
- eCPM(下一名)：排名第二的广告的 eCPM 值
- pCTR：预估点击率
- δ：最小加价单位（通常为 0.01 元）

### 四种定价机制对比

| 机制类型 | 获胜规则 | 扣费规则 |
|---------|---------|---------|
| 第一价格（FPA） | 出价最高者 | 按最高出价扣费 |
| 第二价格（SPA） | 出价最高者 | 按次高出价+0.01 |
| GSP（广义第二价格） | 多广告位排序 | 各位置按下一位出价 |
| VCG | 多广告位排序 | 基于边际贡献扣费 |

## 3. 数学公式与推导

### GSP 定价详细流程

**Step 1：计算 eCPM**
$$
eCPM = pCTR \times Bid \times 1000
$$

**Step 2：排序确定位置**
$$
Rank = \text{sort}(eCPM, \text{desc})
$$

**Step 3：计算扣费价格**
$$
CPC_k = \frac{eCPM_{k+1}}{pCTR_k \times 1000} + \delta
$$

**Step 4：边界处理**
$$
CPC_{last} = \max(Reserve, \delta)
$$

### 关键概念

- **保留价（Reserve Price）**：最低竞价门槛，Cost = max(GSP扣费, Reserve Price)
- **溢价因子（Premium Factor）**：Final Cost = Base Cost × Premium Factor
- **挤压系数（Squash Factor）**：eCPM_adj = pCTR^α × Bid × 1000（α ∈ (0,1]）

## 4. 应用场景

- 搜索广告排名（Google、百度）
- 信息流广告排序
- 多广告位展示场景

## 5. GSP 定价实例

假设 3 个广告主竞争同一个广告位：

| 广告主 | 出价(Bid) | 预估CTR(pCTR) | eCPM | 排名 | 实际CPC扣费 |
|--------|----------|--------------|------|------|------------|
| 广告主A | ¥2.0 | 5% | 100 | 第1名 | ¥1.61 |
| 广告主B | ¥1.6 | 5% | 80 | 第2名 | - |
| 广告主C | ¥1.0 | 4% | 40 | 第3名 | - |

**计算说明**：
广告主A胜出，CPC = eCPM(B) / (pCTR(A) × 1000) + δ = 80 / (0.05 × 1000) + 0.01 = ¥1.61

虽然A出价¥2.0，但实际只需支付¥1.61（基于第二名的eCPM计算）

## 6. 优缺点分析

### 为什么 GSP 是主流？
- 激励效果好：广告主有动力出真实价格
- 计算简单：比 VCG 容易实现
- 收益平衡：平台收益和广告主体验的良好平衡
- 历史原因：Google AdWords 首创并成功应用

### 代码实现

```python
def gsp_pricing(ads):
    """
    ads: list of dict with keys 'bid', 'pctr', 'ad_id'
    Returns: list of dict with ranking and cpc
    """
    for ad in ads:
        ad['ecpm'] = ad['bid'] * ad['pctr'] * 1000
    ads_sorted = sorted(ads, key=lambda x: x['ecpm'], reverse=True)
    reserve_price = 0.01
    for i, ad in enumerate(ads_sorted):
        ad['rank'] = i + 1
        if i < len(ads_sorted) - 1:
            next_ecpm = ads_sorted[i + 1]['ecpm']
            ad['cpc'] = next_ecpm / (ad['pctr'] * 1000) + 0.01
        else:
            ad['cpc'] = max(reserve_price, 0.01)
    return ads_sorted
```

## 7. 学习总结

GSP 是当前广告行业主流定价机制，核心是"按下一名 eCPM 定价"。与第一价格相比更激励真实出价，与 VCG 相比计算更简单。理解 GSP 是理解整个广告定价体系的基础。
