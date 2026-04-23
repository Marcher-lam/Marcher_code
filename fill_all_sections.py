import os, re, textwrap

# Category detection (same as before)
REGRESSION_KW = ['回归', 'ridge', 'lasso', 'elasticnet', 'svr', 'polynomial']
CLASSIFICATION_KW = ['分类', 'logistic', 'svm', 'decisiontree', 'randomforest', 'adaboost', 'gbdt', 'xgboost', 'lightgbm', 'catboost', 'knn', '朴素贝叶斯', 'naivebayes']
CLUSTERING_KW = ['k-means', 'dbscan', '层次聚类', 'meanshift', 'spectral', 'optics']
DIMRED_KW = ['pca', 'lda', 'svd', 'nmf', 'lsa', 't-sne', 'umap']
DEEP_KW = ['神经网络', 'cnn', 'rnn', 'lstm', 'gru', 'transformer', 'gan', 'vae', 'ae', 'bert', 'gpt', 't5', 'bart', 'distilbert', 'albert', 'vit', 'unet', 'resnet', 'efficientnet']
RL_KW = ['dqn', 'ddpg', 'ppo', 'a2c', 'sac', 'td3', 'reinforce', 'sarsa', 'q-learning']

def detect_category(name):
    lower = name.lower()
    for kw in REGRESSION_KW:
        if kw in lower:
            return 'regression'
    for kw in CLASSIFICATION_KW:
        if kw in lower:
            return 'classification'
    for kw in CLUSTERING_KW:
        if kw in lower:
            return 'clustering'
    for kw in DIMRED_KW:
        if kw in lower:
            return 'dimred'
    for kw in DEEP_KW:
        if kw in lower:
            return 'deep'
    for kw in RL_KW:
        if kw in lower:
            return 'rl'
    return 'generic'

# ---------- Content generators ----------

def sec1(name):
    return f"""\n该章节介绍 **{name}** 的基本概念、历史背景以及核心定位。\n"""

def sec2(name):
    return f"""\n核心原理概述：解释 **{name}** 的工作机制、关键公式或模型结构。\n"""

def sec3(name):
    return f"""\n数学推导：提供 **{name}** 的主要公式推导步骤和关键定理。\n"""

def sec4(name, cat):
    if cat == 'regression':
        tex = """\n训练过程通常采用最小二乘或梯度下降优化目标函数 J(θ)。\n步骤：\n1. 初始化参数 θ；\n2. 计算预测 ŷ = Xθ 并求损失；\n3. 计算梯度 ∇J 并更新 θ；\n4. 重复直至收敛。\n"""
    elif cat == 'classification':
        tex = """\n分类模型的训练类似回归，使用交叉熵或对数损失。\n训练步骤：\n1. 初始化模型参数；\n2. 前向计算得到概率分布；\n3. 计算交叉熵损失；\n4. 反向传播更新参数；\n5. 循环直至收敛。\n"""
    elif cat == 'clustering':
        tex = """\n聚类算法迭代更新簇中心或标签。\n常见流程：\n1. 随机初始化簇中心；\n2. 计算样本到各中心的距离并分配标签；\n3. 更新中心为所属样本均值；\n4. 重复直至簇不再变化。\n"""
    elif cat == 'dimred':
        tex = """\n降维方法通过矩阵分解或随机映射保留数据主要结构。\n示例流程：\n1. 构造数据矩阵 X；\n2. 计算协方差或使用随机投影；\n3. 取前 k 个主成分或特征；\n4. 投影得到低维表示。\n"""
    elif cat == 'deep':
        tex = """\n深度模型的训练采用 minibatch 随机梯度下降（或其变体）。\n典型步骤：\n1. 构建网络结构并初始化权重；\n2. 在每个 mini‑batch 上前向传播得到输出；\n3. 计算损失函数；\n4. 通过自动微分得到梯度并使用优化器更新参数；\n5. 循环若干 epoch；\n6. 通过验证集监控并可能进行 early‑stop。\n"""
    elif cat == 'rl':
        tex = """\n强化学习通过交互环境采样轨迹并依据奖励信号更新策略。\n通用流程：\n1. 初始化策略/价值网络；\n2. 在环境中采样 (state, action, reward, next_state)；\n3. 计算 TD‑error 或策略梯度；\n4. 使用梯度更新网络参数；\n5. 重复直至收敛或达到预设回合数。\n"""
    else:
        tex = "\n训练过程概述：依据具体实现选择合适的优化方式并迭代更新模型参数。\n"
    return f"\n{tex}\n"

def sec5(name, cat):
    if cat in ('regression', 'classification'):
        return f"\n典型应用场景包括：\\n- 金融风险评估\\n- 医疗诊断预测\\n- 销售额预测或用户点击率预测\\n- 文本情感分类\\n"
    if cat == 'clustering':
        return f"\n常见用途：\\n- 客户细分\\n- 图像分割\\n- 异常检测\\n- 文档主题聚类\\n"
    if cat == 'dimred':
        return f"\n主要用于：\\n- 可视化高维数据\\n- 降噪与特征压缩\\n- 加速后续机器学习模型\\n- 信息检索中的相似度搜索\\n"
    if cat == 'deep':
        return f"\n广泛应用于：\\n- 计算机视觉（图像分类、目标检测）\\n- 自然语言处理（机器翻译、文本生成）\\n- 语音识别与合成\\n- 推荐系统与生成模型\\n"
    if cat == 'rl':
        return f"\n适用于：\\n- 游戏智能体\\n- 自动驾驶决策\\n- 机器人控制\\n- 金融交易策略\\n"
    return "\n通用应用场景：数据预测、模式识别、决策支持等。\n"

def sec6(name, cat):
    if cat in ('regression', 'classification'):
        return "\n优点：解释性强、实现简单、对小数据有效。\\n缺点：线性模型受限、对非线性关系表现差，需要特征工程。\n"
    if cat == 'clustering':
        return "\n优势：无需标签、发现潜在结构。\\n局限：对噪声敏感、需要事先指定簇数或距离度量。\n"
    if cat == 'dimred':
        return "\n优点：降维加速、可视化。\\n缺点：信息损失、对噪声敏感。\n"
    if cat == 'deep':
        return "\n优势：强大表达能力、端到端学习。\\n缺点：需要大量数据和计算资源、难以解释。\n"
    if cat == 'rl':
        return "\n优势：可学习复杂策略、无需明确模型。\\n挑战：采样效率低、奖励稀疏、收敛不稳。\n"
    return "\n请根据具体算法自行补充优缺点分析。\n"

# Existing specific generators for sections 7-14 (same as previous script)
def lib_impl(name, cat):
    if cat == 'regression':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 回归实现示例（LinearRegression）
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        print('R^2:', model.score(X_test, y_test))
        ```
        """)
    if cat == 'classification':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 分类实现示例（LogisticRegression）
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        print('Accuracy:', model.score(X_test, y_test))
        ```
        """)
    if cat == 'clustering':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 聚类实现示例（KMeans）
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        print('Cluster centers:', model.cluster_centers_)
        ```
        """)
    if cat == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 降维实现示例（PCA）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        print('Explained variance:', pca.explained_variance_ratio_)
        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # PyTorch 基础模型示例（全连接网络）
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        model = Net(input_dim=100, hidden_dim=50, output_dim=10)
        print(model)
        ```
        """)
    if cat == 'rl':
        return textwrap.dedent(f"""
        ```python
        # OpenAI Gym 环境示例（CartPole）\nimport gym\nenv = gym.make('CartPole-v1')\nobs = env.reset()\nfor _ in range(200):\n    action = env.action_space.sample()\n    obs, reward, done, info = env.step(action)\n    env.render()\n    if done:\n        break\nenv.close()\n        ```
        """)
    return "\n```python\n# TODO: 添加库实现代码\n```"

def hand_impl(name, cat):
    class_name = re.sub(r'[^A-Za-z0-9]', '', name)
    if cat in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 手工实现模板\nimport numpy as np\n\nclass {class_name}:\n    def __init__(self, *args, **kwargs):\n        pass\n    def fit(self, X, y):\n        # TODO: 实现训练过程\n        pass\n    def predict(self, X):\n        # TODO: 实现预测过程\n        return np.zeros(len(X))\n        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 手工实现深度网络简化示例（numpy）\nimport numpy as np\n\nclass SimpleNN:\n    def __init__(self, in_dim, hidden_dim, out_dim):\n        self.W1 = np.random.randn(in_dim, hidden_dim)\n        self.b1 = np.zeros(hidden_dim)\n        self.W2 = np.random.randn(hidden_dim, out_dim)\n        self.b2 = np.zeros(out_dim)\n    def forward(self, x):\n        h = np.maximum(0, x @ self.W1 + self.b1)\n        return h @ self.W2 + self.b2\n        ```
        """)
    return "\n```python\n# TODO: 添加手工实现代码\n```"

def visualization(name, cat):
    if cat in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 可视化示例（散点图）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nX = np.random.randn(200, 2)\ny = np.random.randint(0, 2, 200)\nplt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')\nplt.title('{name} 可视化示例')\nplt.show()\n        ```
        """)
    if cat == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # 降维后可视化（2D）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nX_reduced = np.random.randn(200, 2)\nplt.scatter(X_reduced[:,0], X_reduced[:,1], cmap='plasma')\nplt.title('{name} 降维可视化')\nplt.show()\n        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 特征图可视化示例\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfeature = np.random.rand(8, 8)\nplt.imshow(feature, cmap='gray')\nplt.title('{name} 特征图')\nplt.show()\n        ```
        """)
    return "\n- TODO: 添加可视化示例\n"

def evaluation(name, cat):
    metric_map = {
        'regression': 'mean_squared_error',
        'classification': 'accuracy_score',
        'clustering': 'silhouette_score',
        'dimred': 'explained_variance_score',
        'rl': 'episode_reward',
    }
    metric = metric_map.get(cat, 'accuracy_score')
    return textwrap.dedent(f"""
        ```python
        # 评估示例\nfrom sklearn.metrics import {metric}\n# y_true, y_pred / X, labels 需自行准备\n# print('{metric}:', {metric}(y_true, y_pred))\n        ```
        """)

def common_issues():
    return textwrap.dedent("""
    - 未对特征进行标准化或归一化导致模型不收敛。\n- 超参数（学习率、正则化、层数）需要调参。\n- 过拟合：模型在训练集表现好但在测试集表现差。\n- 计算资源：深度模型常需 GPU 加速。\n""")

def learning_summary(name):
    return f"**学习要点**：{name} 的核心思想是 …（请根据实际算法补充）。掌握其数学推导、实现细节以及适用场景是后续深入学习的基础。"

def exercises(name):
    return textwrap.dedent(f"""
    1. 手动实现 {name} 的核心迭代步骤，并在合成数据上验证。\n2. 使用不同库（如 scikit‑learn 与 PyTorch）实现，并比较训练时间与精度。\n3. 设计可视化函数，展示 {name} 在不同超参数下的表现。\n""")

def learning_path():
    return textwrap.dedent("""
    - 先掌握线性模型（线性回归、逻辑回归）→\n- 再学习树模型（决策树、随机森林、XGBoost）→\n- 深入深度学习模型（CNN、Transformer、GAN）→\n- 进阶章节：自监督学习、强化学习、生成模型等前沿方向。\n""")

# Mapping for each section number
SECTION_FUNCS = {
    1: lambda n, c: sec1(n),
    2: lambda n, c: sec2(n),
    3: lambda n, c: sec3(n),
    4: lambda n, c: sec4(n, c),
    5: lambda n, c: sec5(n, c),
    6: lambda n, c: sec6(n, c),
    7: lib_impl,
    8: hand_impl,
    9: visualization,
    10: evaluation,
    11: lambda n, c: common_issues(),
    12: lambda n, c: learning_summary(n),
    13: lambda n, c: exercises(n),
    14: lambda n, c: learning_path(),
}

# Replace a section's content (including empty) with generated content
def replace_section(content, num, name, cat):
    pattern = rf'(##\s+{num}\.\s+[^\n]*\n)(.*?)(?=\n##\s+\d+\.|\Z)'
    new_body = SECTION_FUNCS[num](name, cat)
    replacement = rf"\1\n{new_body}\n"
    new_content, cnt = re.subn(pattern, replacement, content, flags=re.DOTALL)
    return new_content, cnt

def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    name = os.path.splitext(os.path.basename(path))[0]
    cat = detect_category(name)
    changed = False
    for num in range(1, 15):
        content, cnt = replace_section(content, num, name, cat)
        if cnt:
            changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    alg_dir = 'algorithm_knowledge_base/algorithms'
    for fname in os.listdir(alg_dir):
        if fname.endswith('.md'):
            process_file(os.path.join(alg_dir, fname))

if __name__ == '__main__':
    main()
