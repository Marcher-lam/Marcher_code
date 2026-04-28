# VaR与CVaR风险度量(VaR/CVaR Risk Measures) 学习文档

> VaR度量最坏情况的分位点，CVaR度量超出VaR的平均损失。

> 来源线索：本节内容根据原书中关于"VaR/CVaR"的相关章节(Ch 19.3)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：VaR(Value at Risk)是损失分布的$\alpha$-分位数；CVaR(Conditional VaR)是超出VaR的平均损失，更保守的风险度量。

**定义**：

$$\text{VaR}_\alpha(X) = \inf\{x : \mathbb{P}(X \leq x) \geq \alpha\}$$

$$\text{CVaR}_\alpha(X) = \mathbb{E}[X | X \geq \text{VaR}_\alpha(X)]$$

**性质**：
- VaR不是一致性风险度量（不满足次可加性）
- CVaR是一致性风险度量（满足单调性、次可加性、正齐次性、平移不变性）
- CVaR $\geq$ VaR（总是更保守）

**在优化中的应用**：

CVaR约束：$\text{CVaR}_\alpha[f(x,\omega)] \leq c$

可线性化为：

$$\min_{x,\eta} \eta + \frac{1}{1-\alpha}\mathbb{E}[\max(0, f(x,\omega)-\eta)]$$

## 4-8. 核心实现

```python
"""VaR和CVaR计算"""
import numpy as np

def compute_var_cvar(returns, alpha=0.95):
    """计算VaR和CVaR"""
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    idx_var = int(np.ceil(alpha * n)) - 1
    var = sorted_returns[idx_var]
    cvar = sorted_returns[idx_var:].mean()
    return var, cvar

def cvar_optimization(scenarios, probs, alpha=0.95):
    """CVaR优化（简单网格搜索）"""
    best_x, best_cvar = 0, np.inf
    for x in np.arange(0, 100, 5):
        losses = np.array([-s*x + 10*min(x, s) for s in scenarios])
        sorted_losses = np.sort(losses)[::-1]
        idx = int(np.ceil((1-alpha)*len(scenarios)))
        cvar = sorted_losses[:idx].mean()
        if cvar < best_cvar:
            best_cvar = cvar
            best_x = x
    return best_x

if __name__ == "__main__":
    np.random.seed(42)
    losses = np.random.normal(0, 1, 10000)
    var, cvar = compute_var_cvar(losses, alpha=0.95)
    print(f"VaR(95%): {var:.3f}")
    print(f"CVaR(95%): {cvar:.3f}")
    print(f"CVaR ≥ VaR: {cvar >= var}")
```

## 9-14. 简要

### 12. 学习总结
VaR：$\alpha$-分位数。CVaR：超出VaR的期望。CVaR是一致性风险度量，可直接嵌入线性规划。

### 13. 练习题
**Q1**：为什么金融监管偏好CVaR而非VaR？
**A1**：VaR不满足次可加性——组合VaR可能大于各资产VaR之和，与分散投资降低风险的直觉矛盾。CVaR满足次可加性，是更合理的风险度量。

### 14. 学习路径
**前置**：概率论、风险概念 | **进阶**：鲁棒优化、分布鲁棒优化
**资源**：原书Ch 19.3、Rockafellar & Uryasev (2000)
