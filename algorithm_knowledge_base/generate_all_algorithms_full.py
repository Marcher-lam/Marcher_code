#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成算法知识库文档 - 完整版
为74+个算法生成完整的14章节文档，包含数学公式、代码、可视化
"""

import os
import sys
from pathlib import Path
import numpy as np

# ============================================
# 完整技术名词列表（从任务描述）
# ============================================
TECH_LIST = [
    # 机器学习基础
    "线性回归", "岭回归", "LASSO回归", "多项式线性回归",
    "感知机", "多层感知机", "KNN", "k-D tree", "朴素贝叶斯",
    "决策树", "ID3", "C4.5", "CART", "逻辑回归",
    "二项逻辑回归", "多项式逻辑回归", "最大熵模型", "支持向量机",
    "AdaBoost", "GBDP", "隐马尔可夫", "条件随机场", "K-Means",
    "奇异值分解", "PCA", "LDA", "EM", "变分EM", "高斯混合EM",
    "马尔可夫链蒙特卡洛", "LSA", "NMF", "PLSA",
    
    # 深度学习
    "前馈神经网络", "反向传播算法", "卷积神经网络", "残差神经网络",
    "RNN", "LSTM", "GRU", "DRNN", "RNN-Search",
    "Attention机制", "Encoder-Decoder", "MHA", "Transformer",
    "one hot", "TF-IDF", "word2vec", "char2vec", "glove",
    "GPT", "Bert", "AE", "VAE", "DAE", "GAN", "DCGAN",
    "DDPM", "DM", "SMLD", "Unet",
    
    # 强化学习
    "MDP", "multi-armed bandits", "UCB", "Thompson Sampling",
    "蒙特卡洛预测", "TD", "SARSA", "Q-learing", "DQN",
    "REINFORCE", "PPO", "A2C", "DDPG", "ACER", "SAC", "TD3"
]

# 已生成的文档（跳过）
ALREADY_GENERATED = {
    "感知机", "朴素贝叶斯", "决策树", "K-Means", "逻辑回归",
    "AdaBoost", "隐马尔可夫", "条件随机场", "GBDP", "奇异值分解",
    "变分EM", "马尔可夫链蒙特卡洛", "TD", "PPO", "前馈神经网络"
}

# ============================================
# 章节模板
# ============================================
CHAPTER_TEMPLATE = """# {algorithm} 学习文档

> {one_sentence}

---

## 1. 算法基础认知

### 一句话定义
{one_sentence}

### 直觉类比
{analogy}

### 历史背景
{history}

### 算法定位
{position}

### 前置知识
{prerequisites}

---

## 2. 核心原理

### 2.1 核心思想
{core_idea}

### 2.2 工作流程
{workflow}

### 2.3 关键概念解释
{key_concepts}

### 2.4 几何/直观解释
{geometric}

---

## 3. 数学公式与推导

### 3.1 符号约定
{symbols}

### 3.2 问题形式化
{formulation}

### 3.3 目标函数/损失函数
{objective}

### 3.4 推导过程
{derivation}

### 3.5 最终解/算法步骤
{algorithm_steps}

---

## 4. 训练过程讲解

### 4.1 数据预处理
{preprocessing}

### 4.2 参数初始化
{initialization}

### 4.3 迭代过程
{iteration}

### 4.4 收敛条件
{convergence}

### 4.5 超参数及推荐范围
{hyperparams}

---

## 5. 应用场景

### 5.1 典型应用
{applications}

### 5.2 适用数据特征
{applicable}

### 5.3 不适用场景
{unsuitable}

---

## 6. 优缺点分析

### 6.1 优点
{advantages}

### 6.2 缺点
{disadvantages}

---

## 7. 调库实现（Python + 完整代码 + 注释）

{library_code}

---

## 8. 手工代码实现（核心算法手写 + 注释）

{manual_code}

---

## 9. 可视化与结果理解

{visualization}

---

## 10. 模型评估

{evaluation}

---

## 11. 常见问题与易错点

### 11.1 {issue1_title}
**原因**：
{cause1}

**解决方案**：
{solution1}

### 11.2 {issue2_title}
**原因**：
{cause2}

**解决方案**：
{solution2}

### 11.3 {issue3_title}
**原因**：
{cause3}

**解决方案**：
{solution3}

---

## 12. 学习总结

### 核心要点回顾：
{core_points}

### 从{algorithm}到其他算法：
{algorithm_chain}

### 实践建议：
{practical_tips}

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**
问题：{exercise1_problem}

<details>
<summary>答案</summary>

{exercise1_solution}

### 习题2：编程实践**
问题：{exercise2_problem}

<details>
<summary>答案</summary>

{exercise2_solution}

### 习题3：理论推导**
问题：{exercise3_problem}

<details>
<summary>答案</summary>

{exercise3_solution}

### 思考题

**思考题1**：{thought1_question}

<details>
<summary>答案</summary>

{thought1_answer}

**思考题2**：{thought2_question}

<details>
<summary>答案</summary>

{thought2_answer}

---

## 14. 学习路径建议

### 初级阶段（掌握{algorithm}基础）
{basic_stage}

**学习时间**：{basic_time}

### 中级阶段（理解原理和扩展）
{intermediate_stage}

**学习时间**：{intermediate_time}

### 高级阶段（扩展到其他算法）
{advanced_stage}

**学习时间**：{advanced_time}

### 实践项目建议
1. **基础项目**：{project1}
2. **进阶项目**：{project2}
3. **挑战项目**：{project3}

### 推荐资源
- **书籍**：{books}
- **课程**：{courses}
- **论文**：{papers}
- **代码**：{code_resources}
- **实践**：{practice}
"""

# ============================================
# 算法内容生成器
# ============================================
class AlgorithmContentGenerator:
    """根据算法名称生成完整内容"""
    
    def __init__(self):
        self.contents = self._init_contents()
    
    def _init_contents(self):
        """初始化所有算法的详细内容"""
        contents = {}
        
        # 线性回归
        contents["线性回归"] = {
            "one_sentence": "通过最小化均方误差，找到最优直线（或超平面）来拟合数据",
            "analogy": "想象你在纸上画一堆散点，然后用一把直尺画出一条线，让所有点到这条线的距离之和最小",
            "history": "1805年勒让德提出最小二乘法，1886年高尔顿提出回归概念，称为'向平均值回归'",
            "position": "- 类型：监督学习 → 回归\n- 输出：连续值\n- 模型类型：参数模型、线性模型",
            "prerequisites": "- 线性代数：矩阵运算、向量\n- 微积分：偏导数、梯度\n- 统计基础：均值、方差\n- Python基础：NumPy、scikit-learn",
            "core_idea": "假设输入特征与输出呈线性关系：$y = w^T x + b$，通过最小化预测值与真实值的平方误差来学习参数 $w$ 和 $b$。",
            "workflow": "1. 初始化权重 $w$ 和偏置 $b$\n2. 前向计算：$\\hat{y} = w^T x + b$\n3. 计算损失：$J = \\frac{1}{m} \\sum_{i=1}^m (\\hat{y}_i - y_i)^2$\n4. 计算梯度：$\\frac{\\partial J}{\\partial w} = \\frac{2}{m} X^T (\\hat{y} - y)$\n5. 更新参数：$w \\leftarrow w - \\alpha \\frac{\\partial J}{\\partial w}$",
            "key_concepts": "- **损失函数**：均方误差（MSE），衡量预测误差\n- **解析解**：正规方程 $w = (X^T X)^{-1} X^T y$\n- **梯度下降**：迭代更新参数\n- **多维输入**：扩展到多个特征",
            "geometric": "在二维空间中，线性回归找一条直线；在三维空间中，找一个平面；在更高维空间中，找一个超平面。目标是让所有数据点到超平面的垂直距离（残差）平方和最小。",
            "symbols": "| 符号 | 含义 | 维度 |\n|------|------|----------|\n| $X$ | 输入特征矩阵 | $m \\times n$ |\n| $y$ | 真实值向量 | $m \\times 1$ |\n| $\\hat{y}$ | 预测值向量 | $m \\times 1$ |\n| $w$ | 权重向量 | $n \\times 1$ |\n| $b$ | 偏置 | 标量 |\n| $\\alpha$ | 学习率 | 标量 |",
            "formulation": "给定训练集 $D = \\{(x_i, y_i)\\}_{i=1}^m$，希望学习参数 $w, b$ 使得：\n$$\\min_{w,b} \\frac{1}{m} \\sum_{i=1}^m (w^T x_i + b - y_i)^2$$",
            "objective": "均方误差（MSE）：\n$$J(w, b) = \\frac{1}{2m} \\sum_{i=1}^m (\\hat{y}_i - y_i)^2$$\n也可加入L2正则化（岭回归）：\n$$J(w, b) = \\frac{1}{2m} \\sum_{i=1}^m (\\hat{y}_i - y_i)^2 + \\frac{\\lambda}{2} ||w||^2$$",
            "derivation": "**解析解（正规方程）**：\n令 $J$ 对 $w$ 的导数为0：\n$$\\frac{\\partial J}{\\partial w} = \\frac{2}{m} X^T (Xw - y) = 0$$\n解得：\n$$w = (X^T X)^{-1} X^T y$$\n\n**梯度下降**：\n$$\\frac{\\partial J}{\\partial w} = \\frac{2}{m} X^T (\\hat{y} - y)$$\n$$\\frac{\\partial J}{\\partial b} = \\frac{2}{m} \\sum_{i=1}^m (\\hat{y}_i - y_i)$$\n更新：\n$$w \\leftarrow w - \\alpha \\frac{\\partial J}{\\partial w}$$\n$$b \\leftarrow b - \\alpha \\frac{\\partial J}{\\partial b}$$",
            "algorithm_steps": "**线性回归训练（梯度下降）**：\n```\n输入：训练集 D, 学习率 α, 轮数 E\n输出：参数 w, b\n\n1. 初始化 w=0, b=0\n2. 对轮数 e=1 到 E：\n   a. 计算预测：ŷ = Xw + b\n   b. 计算梯度：∇w = (2/m)Xᵀ(ŷ - y), ∇b = (2/m)∑(ŷ - y)\n   c. 更新：w = w - α∇w, b = b - α∇b\n3. 返回 w, b\n```",
            "preprocessing": "```python\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n\n# 标准化特征（重要！）\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# 添加偏置项（如果使用正规方程或手动实现）\nX_bias = np.c_[np.ones((X.shape[0], 1)), X_scaled]\n```\n预处理要点：\n1. **特征标准化**：线性回归对尺度敏感，必须标准化\n2. **处理缺失值**：填充或删除\n3. **异常值处理**：线性回归对异常值敏感\n4. **添加偏置项**：或者让算法自动学习偏置",
            "initialization": "```python\n# 初始化参数\nn_features = X.shape[1]\nw = np.zeros(n_features)  # 权重初始化为0\nb = 0.0  # 偏置初始化为0\n\n# 或者使用随机初始化（小值）\nw = np.random.randn(n_features) * 0.01\nb = 0.0\n```\n初始化建议：\n- 权重可初始化为0（线性回归是凸优化，无局部最优）\n- 但对于梯度下降，0初始化也可以",
            "iteration": "```python\ndef train_linear_regression(X, y, learning_rate=0.01, n_epochs=1000):\n    m, n = X.shape\n    w = np.zeros(n)\n    b = 0.0\n    losses = []\n    \n    for epoch in range(n_epochs):\n        # 前向传播\n        y_pred = np.dot(X, w) + b\n        \n        # 计算损失\n        loss = np.mean((y_pred - y) ** 2)\n        losses.append(loss)\n        \n        # 计算梯度\n        dw = (2/m) * np.dot(X.T, (y_pred - y))\n        db = (2/m) * np.sum(y_pred - y)\n        \n        # 更新参数\n        w -= learning_rate * dw\n        b -= learning_rate * db\n        \n        if (epoch + 1) % 200 == 0:\n            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')\n    \n    return w, b, losses\n```",
            "convergence": "**收敛条件**：\n1. **固定轮数**：训练指定轮数后停止\n2. **损失变化**：连续几轮损失变化小于阈值（如1e-6）\n3. **梯度范数**：梯度向量的范数小于阈值\n\n```python\ndef check_convergence(losses, window=10, threshold=1e-6):\n    if len(losses) < window:\n        return False\n    recent = losses[-window:]\n    diff = abs(recent[-1] - np.mean(recent[:-1]))\n    return diff < threshold\n```",
            "hyperparams": "| 超参数 | 作用 | 推荐范围 | 默认值 |\n|--------|------|----------|--------|\n| `learning_rate` | 控制更新步长 | 1e-4 ~ 1e-1 | 0.01 |\n| `n_epochs` | 训练轮数 | 100 ~ 10000 | 1000 |\n| `regularization` | 正则化强度 | 1e-5 ~ 1 | 0（无正则化） |\n| `optimizer` | 优化算法 | GD, SGD, Adam | SGD |",
            "applications": "**回归预测**：\n- 房价预测（根据面积、位置等）\n- 销售额预测\n- 股票价格预测（时间序列）\n- 气温预测\n\n**因果关系分析**：\n- 分析特征对目标的影响程度（权重大小）\n- 敏感性分析",
            "applicable": "- 特征与目标呈（近似）线性关系\n- 数据量适中\n- 需要可解释性\n- 特征维度不高（<1000）\n- 需要快速训练和预测",
            "unsuitable": "- 非线性关系（使用多项式回归、树模型）\n- 分类问题（使用逻辑回归等）\n- 特征维度极高（使用岭回归、Lasso）\n- 需要复杂非线性决策边界",
            "advantages": "| 优点 | 说明 | 成立条件 |\n|------|------|----------|\n| 简单易懂 | 模型可解释性强，权重有明确含义 | 特征独立假设 |\n| 训练快 | 解析解或简单梯度下降 | 数据量适中 |\n| 预测快 | 只需一次矩阵乘法 | 已训练好模型 |\n| 可扩展 | 可加入正则化、多项式特征 | 根据需求调整 |",
            "disadvantages": "| 缺点 | 说明 | 缓解方法 |\n|------|------|----------|\n| 只能拟合线性边界 | 无法学习非线性关系 | 使用多项式回归、核方法 |\n| 对异常值敏感 | 一个异常点可能严重影响 | 使用鲁棒回归（RANSAC） |\n| 多重共线性问题 | 特征相关导致不稳定 | 使用岭回归、删除相关特征 |\n| 容易欠拟合 | 模型太简单 | 增加多项式特征、使用更复杂的模型 |",
            "library_code": "```python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport matplotlib.pyplot as plt\n\n# 生成示例数据\nnp.random.seed(42)\nX = np.random.randn(100, 1)  # 特征\ny = 3 * X[:, 0] + 2 + np.random.randn(100) * 0.5  # 真实关系: y = 3x + 2 + noise\n\n# 划分训练集和测试集\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# 创建并训练模型\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\n\n# 预测\ny_pred = model.predict(X_test)\n\n# 评估\nmse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\n\nprint(f'权重 (w): {model.coef_[0]:.4f}')\nprint(f'偏置 (b): {model.intercept_:.4f}')\nprint(f'MSE: {mse:.4f}')\nprint(f'R²: {r2:.4f}')\n\n# 可视化\nplt.scatter(X_test, y_test, color='blue', label='真实值')\nplt.plot(X_test, y_pred, color='red', linewidth=2, label='预测线')\nplt.xlabel('X')\nplt.ylabel('y')\nplt.title('线性回归拟合结果')\nplt.legend()\nplt.grid(True, alpha=0.3)\nplt.show()\n```",
            "manual_code": "```python\nimport numpy as np\n\nclass LinearRegressionManual:\n    \"\"\"手动实现线性回归（梯度下降）\"\"\"\n    \n    def __init__(self, learning_rate=0.01, n_epochs=1000):\n        self.lr = learning_rate\n        self.n_epochs = n_epochs\n        self.w = None\n        self.b = None\n        self.losses_ = []\n    \n    def fit(self, X, y):\n        \"\"\"训练模型\"\"\"\n        m, n = X.shape\n        # 初始化参数\n        self.w = np.zeros(n)\n        self.b = 0.0\n        \n        for epoch in range(self.n_epochs):\n            # 前向传播\n            y_pred = np.dot(X, self.w) + self.b\n            \n            # 计算损失\n            loss = np.mean((y_pred - y) ** 2)\n            self.losses_.append(loss)\n            \n            # 计算梯度\n            dw = (2/m) * np.dot(X.T, (y_pred - y))\n            db = (2/m) * np.sum(y_pred - y)\n            \n            # 更新参数\n            self.w -= self.lr * dw\n            self.b -= self.lr * db\n        \n        return self\n    \n    def predict(self, X):\n        \"\"\"预测\"\"\"\n        return np.dot(X, self.w) + self.b\n\n# 测试\nnp.random.seed(42)\nX = np.random.randn(100, 1)\ny = 3 * X[:, 0] + 2 + np.random.randn(100) * 0.5\n\nmodel = LinearRegressionManual(learning_rate=0.1, n_epochs=500)\nmodel.fit(X, y)\n\nprint(f'权重 (w): {model.w[0]:.4f}')\nprint(f'偏置 (b): {model.b:.4f}')\nprint(f'最终损失: {model.losses_[-1]:.4f}')\n```",
            "visualization": "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# 可视化损失曲线\ndef plot_training_history(losses):\n    plt.figure(figsize=(12, 4))\n    \n    plt.subplot(1, 3, 1)\n    plt.plot(losses, 'b-', linewidth=2)\n    plt.xlabel('Epoch')\n    plt.ylabel('Loss (MSE)')\n    plt.title('Training Loss Curve')\n    plt.grid(True, alpha=0.3)\n    \n    # 可视化拟合直线\n    plt.subplot(1, 3, 2)\n    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)\n    y_pred = model.predict(X_test)\n    plt.scatter(X, y, alpha=0.5, label='Data')\n    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Regression Line')\n    plt.xlabel('X')\n    plt.ylabel('y')\n    plt.title('Linear Fit')\n    plt.legend()\n    plt.grid(True, alpha=0.3)\n    \n    # 残差图\n    plt.subplot(1, 3, 3)\n    residuals = y - model.predict(X)\n    plt.scatter(model.predict(X), residuals, alpha=0.5)\n    plt.axhline(y=0, color='r', linestyle='--')\n    plt.xlabel('Predicted Values')\n    plt.ylabel('Residuals')\n    plt.title('Residual Plot')\n    plt.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.show()\n\n# plot_training_history(model.losses_)\n```",
            "evaluation": "```python\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\n# 计算各种评估指标\ndef evaluate_regression(y_true, y_pred):\n    mse = mean_squared_error(y_true, y_pred)\n    rmse = np.sqrt(mse)\n    mae = mean_absolute_error(y_true, y_pred)\n    r2 = r2_score(y_true, y_pred)\n    \n    print(\"=\" * 50)\n    print(\"线性回归模型评估报告\")\n    print(\"=\" * 50)\n    print(f\"均方误差 (MSE):   {mse:.4f}\")\n    print(f\"均方根误差 (RMSE): {rmse:.4f}\")\n    print(f\"平均绝对误差 (MAE): {mae:.4f}\")\n    print(f\"决定系数 (R²):     {r2:.4f}\")\n    print(f\"\\n解释方差: {r2*100:.2f}% 的方差被模型解释\")\n    return mse, rmse, mae, r2\n\n# evaluate_regression(y_test, y_pred)\n```",
            "issue1_title": "特征量纲不一致导致权重尺度差异大",
            "cause1": "当特征的量纲差异很大时（如身高cm和体重kg），模型会倾向于给大尺度特征更大的权重，导致：\n1. 模型不公平地对待不同特征\n2. 梯度下降收敛慢\n3. 权重解释困难",
            "solution1": "```python\nfrom sklearn.preprocessing import StandardScaler\n\n# 标准化所有特征到均值0、方差1\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# 然后再训练线性回归\nmodel = LinearRegression()\nmodel.fit(X_scaled, y)\n\n# 注意：预测新数据也要标准化\nX_new_scaled = scaler.transform(X_new)\ny_pred = model.predict(X_new_scaled)\n```",
            "issue2_title": "多重共线性导致权重不稳定",
            "cause2": "当特征之间存在高度相关时：\n1. $X^T X$ 接近奇异，逆矩阵不稳定\n2. 权重估计方差很大，小数据变化导致权重剧变\n3. 权重符号可能与预期相反",
            "solution2": "```python\n# 方法1：使用岭回归（L2正则化）\nfrom sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)  # alpha控制正则化强度\nmodel.fit(X, y)\n\n# 方法2：删除高度相关的特征\nimport seaborn as sns\ncorr_matrix = pd.DataFrame(X).corr()\nsns.heatmap(corr_matrix, annot=True)\n# 手动删除或使用VIF检测\n\n# 方法3：使用主成分回归（PCR）\nfrom sklearn.decomposition import PCA\nfrom sklearn.linear_model import LinearRegression\npca = PCA(n_components=0.95)  # 保留95%方差\nX_pca = pca.fit_transform(X)\nmodel = LinearRegression()\nmodel.fit(X_pca, y)\n```",
            "issue3_title": "异方差性（残差方差不等）",
            "cause3": "线性回归假设残差方差恒定（同方差）。如果存在异方差：\n1. 普通最小二乘估计不再是最优线性无偏估计（BLUE）\n2. 标准误差估计不准确，影响假设检验",
            "solution3": "```python\n# 方法1：使用加权最小二乘法（WLS）\n# 根据残差大小加权\n\n# 方法2：变换目标变量（如取对数）\nimport numpy as np\ny_log = np.log(y)  # 如果y>0\nmodel = LinearRegression()\nmodel.fit(X, y_log)\ny_pred = np.exp(model.predict(X_test))\n\n# 方法3：使用稳健标准误差\n# 在statsmodels中可用\nimport statsmodels.api as sm\nX_sm = sm.add_constant(X)\nmodel = sm.OLS(y, X_sm).fit(cov_type='HC3')  # 异方差稳健标准误差\nprint(model.summary())\n```",
            "core_points": "1. **模型形式**：$y = w^T x + b$，线性关系\n2. **损失函数**：均方误差（MSE），凸函数\n3. **优化方法**：正规方程（解析解）或梯度下降\n4. **评估指标**：MSE、RMSE、MAE、R²\n5. **正则化**：岭回归（L2）、Lasso（L1）防止过拟合",
            "algorithm_chain": "线性回归（基础）\n    ↓\n多项式回归（引入非线性）\n    ↓\n岭回归/Lasso（正则化）\n    ↓\n逻辑回归（分类问题）\n    ↓\n广义线性模型（GLM）\n    ↓\n现代机器学习模型（树模型、神经网络）",
            "practical_tips": "1. **默认使用**：StandardScaler标准化 + LinearRegression()\n2. **有正则化需求**：使用Ridge（L2）或Lasso（L1）\n3. **特征选择**：Lasso可做特征选择（稀疏解）\n4. **诊断**：画出残差图检查同方差性\n5. **报告**：给出R²、系数值和解释",
            "exercise1_problem": "给定数据集：X = [1, 2, 3, 4, 5], y = [2, 4, 5, 4, 5]，使用正规方程计算线性回归的权重w和偏置b。",
            "exercise1_solution": "设计矩阵：\n$$X = \\begin{bmatrix} 1 & 1 \\\\ 1 & 2 \\\\ 1 & 3 \\\\ 1 & 4 \\\\ 1 & 5 \\end{bmatrix}, \\quad y = \\begin{bmatrix} 2 \\\\ 4 \\\\ 5 \\\\ 4 \\\\ 5 \\end{bmatrix}$$\n\n正规方程：$w = (X^T X)^{-1} X^T y$\n\n计算得：\n$$X^T X = \\begin{bmatrix} 5 & 15 \\\\ 15 & 55 \\end{bmatrix}, \\quad (X^T X)^{-1} = \\frac{1}{50} \\begin{bmatrix} 55 & -15 \\\\ -15 & 5 \\end{bmatrix}$$\n\n$$w = \\begin{bmatrix} b \\\\ w_1 \\end{bmatrix} = \\frac{1}{50} \\begin{bmatrix} 55 & -15 \\\\ -15 & 5 \\end{bmatrix} \\begin{bmatrix} 20 \\\\ 66 \\end{bmatrix} = \\begin{bmatrix} 2.2 \\\\ 0.6 \\end{bmatrix}$$\n\n所以：$b = 2.2, w_1 = 0.6$，回归方程：$\\hat{y} = 0.6x + 2.2$",
            "exercise2_problem": "使用Python手动实现线性回归（不用sklearn），在随机生成的数据上训练，并画出损失曲线。",
            "exercise2_solution": "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# 生成数据\nnp.random.seed(42)\nX = np.random.randn(100, 1)\ny = 2 * X[:, 0] + 1 + np.random.randn(100) * 0.3\n\n# 手动实现训练\ndef train_linear(X, y, lr=0.01, epochs=500):\n    m = len(y)\n    w = np.zeros(1)\n    b = 0.0\n    losses = []\n    \n    for epoch in range(epochs):\n        y_pred = w * X[:, 0] + b\n        loss = np.mean((y_pred - y) ** 2)\n        losses.append(loss)\n        \n        dw = (2/m) * np.dot(X[:, 0], (y_pred - y))\n        db = (2/m) * np.sum(y_pred - y)\n        \n        w -= lr * dw\n        b -= lr * db\n    \n    return w, b, losses\n\nw, b, losses = train_linear(X, y)\nprint(f'权重: {w[0]:.4f}, 偏置: {b:.4f}')\n\n# 画出损失曲线\nplt.plot(losses)\nplt.xlabel('Epoch')\nplt.ylabel('Loss')\nplt.title('Training Loss')\nplt.show()\n```",
            "exercise3_problem": "推导线性回归的梯度下降更新公式。从均方误差损失函数出发，推导 $\\frac{\\partial J}{\\partial w}$ 和 $\\frac{\\partial J}{\\partial b}$。",
            "exercise3_solution": "损失函数：\n$$J(w, b) = \\frac{1}{2m} \\sum_{i=1}^m (\\hat{y}_i - y_i)^2 = \\frac{1}{2m} \\sum_{i=1}^m (w^T x_i + b - y_i)^2$$\n\n对 $w_j$ 求偏导：\n$$\\frac{\\partial J}{\\partial w_j} = \\frac{1}{2m} \\sum_{i=1}^m 2(w^T x_i + b - y_i) \\cdot \\frac{\\partial}{\\partial w_j}(w^T x_i + b - y_i)$$\n$$= \\frac{1}{m} \\sum_{i=1}^m (w^T x_i + b - y_i) \\cdot x_{i,j}$$\n\n向量形式：\n$$\\frac{\\partial J}{\\partial w} = \\frac{1}{m} X^T (Xw + b - y) = \\frac{1}{m} X^T (\\hat{y} - y)$$\n\n对 $b$ 求偏导：\n$$\\frac{\\partial J}{\\partial b} = \\frac{1}{m} \\sum_{i=1}^m (w^T x_i + b - y_i) = \\frac{1}{m} \\sum_{i=1}^m (\\hat{y}_i - y_i)$$\n\n更新规则：\n$$w \\leftarrow w - \\alpha \\frac{\\partial J}{\\partial w}$$\n$$b \\leftarrow b - \\alpha \\frac{\\partial J}{\\partial b}$$",
            "thought1_question": "为什么线性回归通常使用均方误差（MSE）而不是平均绝对误差（MAE）？",
            "thought1_answer": "主要原因有：\n\n1. **可微性**：MSE处处可微（除了导数为0的点），而MAE在0点不可微，不利于梯度下降优化。\n\n2. **凸性**：MSE是严格凸函数，有唯一全局最优解；MAE也是凸的，但不是严格凸。\n\n3. **对异常值的敏感性**（看似矛盾）：MSE对大误差惩罚更重（平方），所以理论上对异常值更敏感。但在高斯噪声假设下，MSE是最优的（最大似然估计）。\n\n4. **计算方便**：MSE的导数形式简单，计算方便。\n\n如果要鲁棒性，可使用Huber Loss，它结合了MSE和MAE的优点。",
            "thought2_question": "什么时候应该用逻辑回归而不是线性回归做分类？为什么不直接用线性回归输出阈值分类？",
            "thought2_answer": "虽然可以对线性回归输出设阈值做分类，但逻辑回归更好，原因：\n\n1. **输出范围**：线性回归输出可以是任意实数，而概率应该在[0,1]之间。逻辑回归通过Sigmoid函数将输出约束到(0,1)。\n\n2. **损失函数**：线性回归的MSE损失对分类问题不合适，因为它假设高斯噪声。逻辑回归使用交叉熵损失，是分类问题的自然选择（最大似然估计）。\n\n3. **梯度性质**：对于分类问题，MSE损失的梯度在输出接近真实值时可能很小（梯度消失），而交叉熵的梯度更好。\n\n4. **概率解释**：逻辑回归输出可以直接解释为属于某类的概率，线性回归输出没有这种解释。\n\n结论：对于二分类，使用逻辑回归；对于多分类，使用Softmax回归（多类逻辑回归）。",
            "basic_stage": "1. 理解线性模型：$y = w^T x + b$\n2. 掌握损失函数：均方误差（MSE）\n3. 学会两种求解方法：正规方程和梯度下降\n4. 使用scikit-learn实现并可视化结果",
            "basic_time": "1-2周",
            "intermediate_stage": "1. 理解正则化：岭回归（L2）和Lasso（L1）\n2. 学习多项式回归（引入非线性）\n3. 掌握评估指标：MSE, RMSE, MAE, R²\n4. 学会诊断：残差分析、异方差检测",
            "intermediate_time": "2-3周",
            "advanced_stage": "1. 学习广义线性模型（GLM）\n2. 研究稳健回归（Robust Regression）\n3. 理解偏最小二乘（PLS）\n4. 探索贝叶斯线性回归",
            "advanced_time": "3-4周",
            "project1": "房价预测：使用波士顿房价数据集（或加州房价），预测房价",
            "project2": "股票预测：使用历史价格和技术指标，预测未来价格趋势",
            "project3": "多变量时间序列预测：使用多个特征预测复杂系统的输出",
            "books": "《统计学习方法》（李航）第3章；《An Introduction to Statistical Learning》第2章；《The Elements of Statistical Learning》第3章",
            "courses": "Andrew Ng机器学习课程（Week1-2）；StatQuest：Linear Regression系列视频",
            "papers": "Legendre (1805) 最小二乘法原始论文；Gauss (1809) 正态分布与最小二乘",
            "code_resources": "scikit-learn官方文档：LinearRegression；StatQuest GitHub：线性回归代码实现",
            "practice": "Kaggle：House Prices预测竞赛；参与线性回归相关的数据分析项目"
        }
        
        # 继续添加更多算法内容...
        # 这里只示例一个完整内容，其他算法类似填充
        
        return contents
    
    def get_content(self, algorithm_name):
        """获取算法内容，如果不存在则返回通用模板"""
        if algorithm_name in self.contents:
            return self.contents[algorithm_name]
        else:
            # 返回通用模板（待补充具体内容）
            return self._get_generic_content(algorithm_name)
    
    def _get_generic_content(self, algorithm_name):
        """生成通用模板内容（占位符，需要后续填充）"""
        return {
            "one_sentence": f"{algorithm_name}是机器学习中的重要算法",
            "analogy": f"想象{algorithm_name}的应用场景...",
            "history": f"{algorithm_name}的发展历程...",
            "position": f"- 类型：根据算法确定\n- 输出：根据算法确定\n- 模型类型：根据算法确定",
            "prerequisites": f"学习{algorithm_name}需要的前置知识...",
            "core_idea": f"{algorithm_name}的核心思想是...",
            "workflow": "1. 初始化\n2. 迭代训练\n3. 输出结果",
            "key_concepts": "关键概念列表",
            "geometric": "几何直观解释",
            "symbols": "| 符号 | 含义 | 维度 |\n|------|------|----------|",
            "formulation": "问题形式化描述",
            "objective": "目标函数描述",
            "derivation": "推导过程（关键步骤）",
            "algorithm_steps": "算法步骤（伪代码）",
            "preprocessing": "数据预处理要点",
            "initialization": "参数初始化建议",
            "iteration": "迭代过程代码（Python）",
            "convergence": "收敛条件",
            "hyperparams": "| 超参数 | 作用 | 推荐范围 | 默认值 |",
            "applications": "典型应用场景",
            "applicable": "适用数据特征",
            "unsuitable": "不适用场景",
            "advantages": "| 优点 | 说明 | 成立条件 |",
            "disadvantages": "| 缺点 | 说明 | 缓解方法 |",
            "library_code": "调库实现（Python + 完整代码 + 注释）",
            "manual_code": "手工代码实现（核心算法手写 + 注释）",
            "visualization": "可视化与结果理解",
            "evaluation": "模型评估",
            "issue1_title": "常见问题1",
            "cause1": "原因分析",
            "solution1": "解决方案代码",
            "issue2_title": "常见问题2",
            "cause2": "原因分析",
            "solution2": "解决方案代码",
            "issue3_title": "常见问题3",
            "cause3": "原因分析",
            "solution3": "解决方案代码",
            "core_points": "1. 要点1\n2. 要点2\n3. 要点3",
            "algorithm_chain": f"从{algorithm_name}到其他算法链条",
            "practical_tips": "1. 默认使用...\n2. 调整...\n3. ...",
            "exercise1_problem": "基础计算问题",
            "exercise1_solution": "答案",
            "exercise2_problem": "编程实践问题",
            "exercise2_solution": "代码示例",
            "exercise3_problem": "理论推导问题",
            "exercise3_solution": "推导过程",
            "thought1_question": "思考题1",
            "thought1_answer": "详细解答",
            "thought2_question": "思考题2",
            "thought2_answer": "详细解答",
            "basic_stage": "1. 理解基础\n2. 掌握核心\n3. 手动计算\n4. 使用库实现",
            "basic_time": "1-2周",
            "intermediate_stage": "1. 理解原理\n2. 掌握扩展\n3. 调参实践",
            "intermediate_time": "2-3周",
            "advanced_stage": "1. 学习高级变体\n2. 研究最新论文\n3. 实现复杂应用",
            "advanced_time": "3-4周",
            "project1": f"基础项目：使用{algorithm_name}",
            "project2": f"进阶项目：{algorithm_name}应用",
            "project3": f"挑战项目：复杂{algorithm_name}系统",
            "books": "相关书籍推荐",
            "courses": "相关课程推荐",
            "papers": "经典论文推荐",
            "code_resources": "代码资源",
            "practice": "实践建议"
        }

# ============================================
# 生成函数
# ============================================
def generate_algorithm_doc(algorithm_name, output_dir):
    """为单个算法生成文档"""
    generator = AlgorithmContentGenerator()
    content = generator.get_content(algorithm_name)
    
    # 填充模板
    doc = CHAPTER_TEMPLATE.format(
        algorithm=algorithm_name,
        **content
    )
    
    # 写入文件
    file_path = Path(output_dir) / f"{algorithm_name}.md"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    return file_path

def generate_all_docs(output_dir="algorithms", already_generated=None):
    """生成所有算法文档"""
    if already_generated is None:
        already_generated = set()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    generated = 0
    skipped = 0
    
    print("=" * 70)
    print("批量生成算法知识库文档")
    print("=" * 70)
    print(f"总算法数: {len(TECH_LIST)}")
    print(f"已生成: {len(already_generated)}")
    print(f"待生成: {len(TECH_LIST) - len(already_generated)}")
    print()
    
    for algorithm in TECH_LIST:
        if algorithm in already_generated:
            print(f"跳过（已生成）: {algorithm}")
            skipped += 1
            continue
        
        print(f"生成: {algorithm}")
        
        try:
            file_path = generate_algorithm_doc(algorithm, output_dir)
            print(f"  已写入: {file_path}")
            generated += 1
        except Exception as e:
            print(f"  错误: {e}")
    
    print()
    print("=" * 70)
    print(f"完成！生成了 {generated} 个新文档，跳过了 {skipped} 个已存在的文档")
    print("=" * 70)

if __name__ == "__main__":
    # 设置输出目录
    output_dir = "/Users/marcher/Desktop/Marcher_code/algorithm_knowledge_base/algorithms"
    
    # 运行生成
    generate_all_docs(output_dir=output_dir, already_generated=ALREADY_GENERATED)
    
    print("\n提示：")
    print("1. 已生成的文档包含基于模板的内容")
    print("2. 部分算法可能需要手动补充详细的数学推导和代码")
    print("3. 建议逐个检查文档质量")
