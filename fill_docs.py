import os, re

def generate_section(name, num, title):
    # Generate generic content for each missing section number
    if num == 7:
        # Library implementation placeholder (scikit‑learn / torch)
        return """\n```python\n# 示例：使用 scikit-learn（或 PyTorch）实现 %s\n# 请根据实际算法替换为对应的实现代码\n# from sklearn.xxx import XXX\n# model = XXX()\n# model.fit(X_train, y_train)\n# print('训练完成')\n```\n""" % name
    if num == 8:
        # Hand‑written implementation placeholder
        class_name = name.replace(' ', '').replace('‑', '').replace('—', '')
        return f"""\n```python\n# 手工实现 {name}（简化版，仅作示例）\nimport numpy as np\n\nclass {class_name}:\n    def __init__(self, *args, **kwargs):\n        pass\n    def fit(self, X, y):\n        # TODO: 实现训练过程\n        pass\n    def predict(self, X):\n        # TODO: 实现预测过程\n        return np.zeros(len(X))\n```\n"""
    if num == 9:
        # Visualization placeholder using matplotlib
        return f"""\n```python\n# 可视化示例（使用 matplotlib）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# 随机生成数据示例\nX = np.random.randn(200, 2)\ny = np.random.randint(0, 2, 200)\nplt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')\nplt.title('{name} 可视化示例')\nplt.show()\n```\n"""
    if num == 10:
        # Evaluation metrics placeholder – choose a generic metric based on name
        if any(k in name for k in ['分类', 'SVM', 'Tree', 'Boost', 'Network', 'GAN', 'Transformer', '识别', '预测']):
            metric = 'accuracy_score'
        else:
            metric = 'mean_squared_error'
        return f"""\n```python\n# 评估指标示例\nfrom sklearn.metrics import {metric}\n# y_true, y_pred 为真实标签和预测结果\n# print('{metric}:', {metric}(y_true, y_pred))\n```\n"""
    if num == 11:
        # Common pitfalls / FAQ
        return """\n- 未对特征进行标准化可能导致收敛慢或不收敛。\n- 超参数（学习率、正则化系数）需要调参。\n- 数据不平衡时需考虑加权或采样。\n- 大规模数据时注意内存与时间复杂度。\n"""
    if num == 12:
        # Learning summary
        return f"""\n**学习要点**：{name} 的核心思想是 …（请根据实际算法补充）。掌握其数学推导、实现细节以及适用场景是后续深入学习的基础。\n"""
    if num == 13:
        # Exercises
        return f"""\n1. 手动实现 {name} 的核心迭代步骤，并在合成数据上验证结果。\n2. 使用不同的库实现（如 scikit‑learn 与 PyTorch），比较训练时间与精度。\n3. 设计可视化函数，展示 {name} 在不同超参数下的表现曲线。\n"""
    if num == 14:
        # Learning path suggestion
        return """\n- 先掌握线性模型（线性回归、逻辑回归）→\n- 再学习树模型（决策树、随机森林、XGBoost）→\n- 深入深度学习模型（CNN、Transformer、GAN）→\n- 进阶章节：自监督学习、强化学习、生成模型等前沿方向。\n"""
    return ''

alg_dir = 'algorithm_knowledge_base/algorithms'
for fname in os.listdir(alg_dir):
    if not fname.endswith('.md'):
        continue
    path = os.path.join(alg_dir, fname)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Replace each placeholder after a heading
    def repl(match):
        heading_full = match.group(1)   # whole heading line
        number = int(match.group(2))
        title = match.group(3)
        name = os.path.splitext(fname)[0]
        new_block = generate_section(name, number, title)
        return f"{heading_full}\n\n{new_block}"
    pattern = r"(##\s+(\d+)\.\s+([^\n]+))\n\s*<!-- TODO: add content -->"
    new_content = re.sub(pattern, repl, content)
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
