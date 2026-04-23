#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_full_knowledge_base.py
================================
一键生成 **完整的 14 章节算法文档**（含示例公式、可运行代码、可视化框架、练习题占位）。
脚本使用 Jinja2 模板渲染，每篇文档会自动填充：

* 章节标题与统一说明
* 基础数学符号约定（通用符号）
* 适配的库导入（scikit‑learn / PyTorch / Gensim / gym 等）
* 代码骨架（调库实现、手工实现）
* 可视化函数框架
* 练习题与答案占位（TODO 标记）

> **注意**：脚本只能生成 **通用骨架**，每个算法的核心细节（如具体推导、超参数取值、实际应用场景）仍需手动补充。
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

# ----------------------------------------------------------------------
# 1️⃣ 依赖检查
# ----------------------------------------------------------------------
try:
    from jinja2 import Environment, BaseLoader
except ImportError:
    sys.stderr.write(
        "Jinja2 未安装。请先运行: pip install jinja2\n"
    )
    sys.exit(1)

# ----------------------------------------------------------------------
# 2️⃣ 配置区
# ----------------------------------------------------------------------
# (1) 所有算法的原始列表（顺序即「机器学习 → 深度学习 → 强化学习」）
ALGORITHM_LIST = [
    # 机器学习
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

# (2) 简单的「库映射」：根据算法名称判断应使用的主流库与类名。
#    对于不在映射表内的算法，默认使用 scikit‑learn（分类/回归）或 PyTorch（深度学习）占位。
LIB_MAPPING: Dict[str, Dict[str, str]] = {
    # 传统机器学习（sklearn）
    "线性回归": {"module": "sklearn.linear_model", "class": "LinearRegression"},
    "岭回归": {"module": "sklearn.linear_model", "class": "Ridge"},
    "LASSO回归": {"module": "sklearn.linear_model", "class": "Lasso"},
    "多项式线性回归": {"module": "sklearn.preprocessing", "class": "PolynomialFeatures"},
    "感知机": {"module": "sklearn.linear_model", "class": "Perceptron"},
    "多层感知机": {"module": "sklearn.neural_network", "class": "MLPClassifier"},
    "KNN": {"module": "sklearn.neighbors", "class": "KNeighborsClassifier"},
    "k-D tree": {"module": "sklearn.neighbors", "class": "KDTree"},
    "朴素贝叶斯": {"module": "sklearn.naive_bayes", "class": "GaussianNB"},
    "决策树": {"module": "sklearn.tree", "class": "DecisionTreeClassifier"},
    "ID3": {"module": "sklearn.tree", "class": "DecisionTreeClassifier"},   # 同 DecisionTree
    "C4.5": {"module": "sklearn.tree", "class": "DecisionTreeClassifier"},
    "CART": {"module": "sklearn.tree", "class": "DecisionTreeRegressor"},
    "逻辑回归": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
    "二项逻辑回归": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
    "多项式逻辑回归": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
    "最大熵模型": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
    "支持向量机": {"module": "sklearn.svm", "class": "SVC"},
    "AdaBoost": {"module": "sklearn.ensemble", "class": "AdaBoostClassifier"},
    "GBDP": {"module": "sklearn.ensemble", "class": "GradientBoostingClassifier"},
    "K-Means": {"module": "sklearn.cluster", "class": "KMeans"},
    "奇异值分解": {"module": "sklearn.decomposition", "class": "TruncatedSVD"},
    "PCA": {"module": "sklearn.decomposition", "class": "PCA"},
    "LDA": {"module": "sklearn.discriminant_analysis", "class": "LinearDiscriminantAnalysis"},
    # 深度学习（PyTorch）
    "前馈神经网络": {"module": "torch.nn", "class": "Sequential"},
    "卷积神经网络": {"module": "torch.nn", "class": "Conv2d"},
    "残差神经网络": {"module": "torch.nn", "class": "ResNet"},
    "RNN": {"module": "torch.nn", "class": "RNN"},
    "LSTM": {"module": "torch.nn", "class": "LSTM"},
    "GRU": {"module": "torch.nn", "class": "GRU"},
    "Transformer": {"module": "torch.nn", "class": "Transformer"},
    "GPT": {"module": "torch.nn", "class": "TransformerDecoder"},
    "Bert": {"module": "torch.nn", "class": "TransformerEncoder"},
    "AE": {"module": "torch.nn", "class": "AutoEncoder"},
    "VAE": {"module": "torch.nn", "class": "VariationalAutoEncoder"},
    "GAN": {"module": "torch.nn", "class": "GAN"},
    "DCGAN": {"module": "torch.nn", "class": "DCGAN"},
    "Unet": {"module": "torch.nn", "class": "UNet"},
    # 强化学习（gym / 自研函数占位）
    "MDP": {"module": "custom_rl", "class": "MDP"},
    "DQN": {"module": "custom_rl", "class": "DQN"},
    "PPO": {"module": "custom_rl", "class": "PPO"},
    "A2C": {"module": "custom_rl", "class": "A2C"},
    "DDPG": {"module": "custom_rl", "class": "DDPG"},
    "SAC": {"module": "custom_rl", "class": "SAC"},
    "TD3": {"module": "custom_rl", "class": "TD3"},
    # 文本特征工程（Gensim、sklearn）
    "one hot": {"module": "sklearn.preprocessing", "class": "OneHotEncoder"},
    "TF-IDF": {"module": "sklearn.feature_extraction.text", "class": "TfidfVectorizer"},
    "word2vec": {"module": "gensim.models", "class": "Word2Vec"},
    "char2vec": {"module": "gensim.models", "class": "FastText"},
    "glove": {"module": "gensim.models", "class": "KeyedVectors"},
}

# (3) Jinja2 模板（完整 14 章节）——所有章节都已填入“通用示例”或 “TODO”
TEMPLATE_STR = r"""
# {{ alg_name }} 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
{{ basics.definition }}

### 1.2 直觉类比
{{ basics.analogy }}

### 1.3 历史背景
{{ basics.history }}

### 1.4 算法定位
- 类型：{{ basics.type }}
- 输出：{{ basics.output }}
- 模型类别：{{ basics.model_category }}

### 1.5 前置知识
{{ basics.prereq }}

## 2. 核心原理
### 2.1 核心思想
{{ core.thought }}

### 2.2 工作流程
{{ core.workflow }}

### 2.3 关键概念解释
{{ core.key_concepts }}

### 2.4 几何/直观解释
{{ core.geometry }}

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
{% for sym, desc in math.symbols.items() %}
| {{ sym }} | {{ desc }} |
{% endfor %}

### 3.2 问题形式化
{{ math.problem_formulation }}

### 3.3 目标函数/损失函数
{{ math.loss_function }}

### 3.4 推导过程
{{ math.derivation }}

### 3.5 最终解/算法步骤
{{ math.solution }}

## 4. 训练过程讲解
### 4.1 数据预处理
{{ training.preprocessing }}

### 4.2 参数初始化
{{ training.initialization }}

### 4.3 迭代过程
{{ training.iteration }}

### 4.4 收敛条件
{{ training.convergence }}

### 4.5 超参数及推荐范围
{{ training.hyperparams }}

## 5. 应用场景
### 5.1 典型应用
{{ application.examples }}

### 5.2 适用数据特征
{{ application.data_characteristics }}

### 5.3 不适用场景
{{ application.unsuitable }}

## 6. 优缺点分析
### 6.1 优点
{% for item in pros %}
- {{ item }}
{% endfor %}

### 6.2 缺点
{% for item in cons %}
- {{ item }}
{% endfor %}

### 6.3 与同类算法对比
{{ comparison }}

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib {{ lib_install }}
```

### 7.2 完整代码示例
```python
{{ lib_imports }}

{{ lib_code_stub }}
```

### 7.3 运行结果示例
> TODO: 将运行结果截图或输出粘贴至此。

## 8. 手工代码实现
### 8.1 核心算法手写
```python
{{ manual_code_stub }}
```

### 8.2 与调库结果对比
> TODO: 在这里展示手工实现与调库实现的性能对比表格或图示。

## 9. 可视化与结果理解
### 9.1 关键参数可视化
```python
{{ viz_param_stub }}
```

### 9.2 模型性能可视化
```python
{{ viz_performance_stub }}
```

### 9.3 结果解读
> TODO: 对可视化结果进行文字解读。

## 10. 模型评估
### 10.1 评估指标选择
{{ evaluation.metrics }}

### 10.2 交叉验证
```python
{{ evaluation.crossval_stub }}
```

### 10.3 超参数调优
```python
{{ evaluation.hyperopt_stub }}
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
{{ pitfalls.data }}

### 11.2 模型层面常见错误
{{ pitfalls.model }}

### 11.3 调参层面常见误区
{{ pitfalls.hyper }}

## 12. 学习总结
### 12.1 核心要点回顾
{{ summary.key_points }}

### 12.2 关键公式汇总
{{ summary.formulas }}

### 12.3 与前序/后续算法联系
{{ summary.relations }}

## 13. 练习题与思考题
### 13.1 基础练习题
{{ exercises.basic }}

### 13.2 进阶思考题
{{ exercises.advanced }}

### 13.3 详细答案与解析
{{ exercises.answers }}

## 14. 学习路径建议
### 14.1 前置知识
{{ roadmap.prereq }}

### 14.2 平行算法
{{ roadmap.parallel }}

### 14.3 进阶算法
{{ roadmap.advanced }}

### 14.4 推荐资源
{{ roadmap.resources }}
"""

# ----------------------------------------------------------------------
# 4️⃣ 辅助函数：为每个算法生成统一的上下文字典
# ----------------------------------------------------------------------
def build_context(alg_name: str) -> Dict:
    """
    根据算法名称返回渲染模板所需的全部占位数据。
    大多数字段使用通用占位文字（TODO），仅在少数已经能自动推断的地方填入具体值。
    """
    # 基础分类判断（机器学习 / 深度学习 / 强化学习 / 文本特征）
    if alg_name in {
        "线性回归", "岭回归", "LASSO回归", "多项式线性回归",
        "感知机", "多层感知机", "KNN", "k-D tree", "朴素贝叶斯",
        "决策树", "ID3", "C4.5", "CART", "逻辑回归",
        "二项逻辑回归", "多项式逻辑回归", "最大熵模型", "支持向量机",
        "AdaBoost", "GBDP", "隐马尔可夫", "条件随机场",
        "K-Means", "奇异值分解", "PCA", "LDA", "EM", "变分EM",
        "高斯混合EM", "马尔可夫链蒙特卡洛", "LSA", "NMF", "PLSA"
    }:
        domain = "机器学习"
    elif alg_name in {
        "前馈神经网络", "反向传播算法", "卷积神经网络", "残差神经网络",
        "RNN", "LSTM", "GRU", "DRNN", "RNN-Search",
        "Attention机制", "Encoder-Decoder", "MHA", "Transformer",
        "AE", "VAE", "DAE", "GAN", "DCGAN", "DDPM", "DM", "SMLD", "Unet"
    }:
        domain = "深度学习"
    elif alg_name in {
        "MDP", "multi-armed bandits", "UCB", "Thompson Sampling",
        "蒙特卡洛预测", "TD", "SARSA", "Q-learing", "DQN",
        "REINFORCE", "PPO", "A2C", "DDPG", "ACER", "SAC", "TD3"
    }:
        domain = "强化学习"
    elif alg_name in {"one hot", "TF-IDF", "word2vec", "char2vec", "glove"}:
        domain = "文本特征工程"
    else:
        domain = "其它"

    lib_info = LIB_MAPPING.get(alg_name, {})
    lib_module = lib_info.get("module", "TODO_MODULE")
    lib_class = lib_info.get("class", "TODO_CLASS")
    lib_install = ""
    if lib_module.startswith("sklearn"):
        lib_install = "scikit-learn"
    elif lib_module.startswith("torch"):
        lib_install = "torch torchvision"
    elif lib_module.startswith("gensim"):
        lib_install = "gensim"
    elif lib_module.startswith("custom_rl"):
        lib_install = "gym numpy"

    context = {
        "alg_name": alg_name,
        "basics": {
            "definition": f"TODO: 用一句话定义【{alg_name}】的核心任务。",
            "analogy": "TODO: 给出生活中的类比。",
            "history": "TODO: 简述该算法的历史渊源。",
            "type": "监督学习 / 无监督学习 / 强化学习（依据实际）",
            "output": "连续值 / 离散类别 / 概率分布等（依据实际）",
            "model_category": "参数模型 / 非参数模型 / 生成模型",
            "prereq": "- 线性代数\n- 微积分\n- Python 编程（NumPy、pandas）"
        },
        "core": {
            "thought": "TODO: 用简短文字阐述核心思想。",
            "workflow": "1. TODO\n2. TODO\n3. TODO",
            "key_concepts": "- TODO\n- TODO",
            "geometry": "TODO：若有几何意义请给出简要描述。"
        },
        "math": {
            "symbols": {"X": "特征矩阵", "y": "目标向量", "θ": "模型参数"},
            "problem_formulation": "TODO：将该算法的学习任务形式化为数学问题。",
            "loss_function": "TODO：给出常用的损失函数（若有），如 $L(θ)=\\frac{1}{n}\\sum_i\\ell(y_i,\\hat y_i)$。",
            "derivation": "TODO：逐步推导（含每步说明），使用 LaTeX 公式。",
            "solution": "TODO：给出最终的闭式解或迭代更新公式。"
        },
        "training": {
            "preprocessing": "- TODO：特征标准化、缺失值处理等。",
            "initialization": "- TODO：参数初始化方式（零、随机、预训练）。",
            "iteration": "- TODO：若为迭代算法，给出伪代码或循环结构。",
            "convergence": "- TODO：收敛判据（梯度阈值、损失变化、最大迭代次数）。",
            "hyperparams": "- learning_rate: 0.001‑0.1\n- n_iterations: 100‑1000\n- ...（根据实际自行补充）"
        },
        "application": {
            "examples": "- TODO：列举 2‑3 真实业务场景，并说明为何适用。",
            "data_characteristics": "- TODO：数据规模、特征类型、噪声水平等。",
            "unsuitable": "- TODO：哪些情形下不推荐使用本算法。"
        },
        "pros": ["TODO: 优点 1", "TODO: 优点 2", "TODO: 优点 3"],
        "cons": ["TODO: 缺点 1", "TODO: 缺点 2", "TODO: 缺点 3"],
        "comparison": "TODO：与 2‑3 个同类算法的对比表格（可使用 Markdown 表格）。",
        "lib_install": lib_install,
        "lib_imports": f"from {lib_module} import {lib_class}  # TODO: 如有需要，请补全其它 import",
        "lib_code_stub": f"""# 调库实现示例（{alg_name})
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据准备（使用示例数据或自行加载）
X, y = ...  # TODO: 加载或生成数据

# 2. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型并训练
model = {lib_class}()  # 如有超参数请自行设置
model.fit(X_train, y_train)

# 4. 预测与评估
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('R² :', r2_score(y_test, y_pred))
""",
        "manual_code_stub": f"""# 手工实现示例（仅做结构展示）
import numpy as np

class {alg_name.replace(' ', '')}Manual:
    \"\"\"手工实现 {alg_name}（仅示例框架）\"\"\"

    def __init__(self, **kwargs):
        # TODO: 参数初始化
        pass

    def fit(self, X, y):
        # TODO: 训练过程
        pass

    def predict(self, X):
        # TODO: 预测过程
        return np.zeros(len(X))
""",
        "viz_param_stub": """# 参数可视化示例（伪代码）
import matplotlib.pyplot as plt

def plot_hyperparameter_effect(param_values, scores):
    plt.plot(param_values, scores, marker='o')
    plt.xlabel('超参数取值')
    plt.ylabel('评价指标')
    plt.title('超参数对模型性能的影响')
    plt.grid(True)
    plt.show()
""",
        "viz_performance_stub": """# 性能可视化示例（伪代码）
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')
    plt.show()
""",
        "evaluation": {
            "metrics": "- TODO：列出本算法常用的评估指标（如 MSE、Accuracy、BLEU 等）。",
            "crossval_stub": """# 交叉验证示例（sklearn）
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print('CV R²: %.4f ± %.4f' % (scores.mean(), scores.std()))
""",
            "hyperopt_stub": f"""# 超参数调优示例（GridSearchCV）
from sklearn.model_selection import GridSearchCV

param_grid = {{
    'param1': [0.01, 0.1, 1],
    'param2': [10, 100, 1000]
}}

grid = GridSearchCV({lib_class}(), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
print('最佳参数:', grid.best_params_)
"""
        },
        "pitfalls": {
            "data": "- TODO：数据缺失、特征尺度不一致、类别不平衡等常见问题。",
            "model": "- TODO：模型未收敛、梯度爆炸/消失、过拟合等。",
            "hyper": "- TODO：学习率设定不当、迭代次数不足/过多等。"
        },
        "summary": {
            "key_points": "- TODO：本算法的核心概念与适用范围（3‑5 条）。",
            "formulas": "- TODO：列出 1‑3 条关键公式（使用 LaTeX）。",
            "relations": "- TODO：说明与前后置算法的联系（如是线性模型的扩展）"
        },
        "exercises": {
            "basic": "- TODO：提供 2‑3 道基础练习题（含答案占位）。",
            "advanced": "- TODO：提供 1‑2 道进阶思考题。",
            "answers": "- TODO：为上述练习给出完整答案与解析。"
        },
        "roadmap": {
            "prereq": "- TODO：阅读本算法前需要掌握的前置知识。",
            "parallel": "- TODO：列出可并行学习的同层次算法。",
            "advanced": "- TODO：学习本算法后推荐的进阶方向。",
            "resources": "- TODO：提供 2‑3 条高质量外部资源（书籍/论文/课程）。"
        }
    }
    return context

# ----------------------------------------------------------------------
# 5️⃣ 主函数
# ----------------------------------------------------------------------
def main(output_dir: Path):
    """
    生成所有算法的 markdown 文档并写入 output_dir/algorithms/
    """
    env = Environment(loader=BaseLoader(), autoescape=False, keep_trailing_newline=True)
    template = env.from_string(TEMPLATE_STR)

    alg_dir = output_dir / "algorithms"
    alg_dir.mkdir(parents=True, exist_ok=True)

    for alg in ALGORITHM_LIST:
        ctx = build_context(alg)
        rendered = template.render(**ctx)

        filename = alg.replace(" ", "_") + ".md"
        target_path = alg_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"✅ {target_path}")

    print("\n🎉 完成！所有文档已生成到:", alg_dir)

# ----------------------------------------------------------------------
# 6️⃣ 入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    repo_root = Path(__file__).parent
    main(repo_root)
