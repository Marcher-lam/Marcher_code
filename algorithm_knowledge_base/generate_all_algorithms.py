#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成算法知识库文档
为算法列表中的每个算法生成完整的14章节Markdown文档
每个文档包含：算法基础认知、核心原理、数学公式、训练过程、应用场景等
"""

import os
import json
from pathlib import Path
import numpy as np

# ============================================
# 1. 算法列表（从原始任务）
# ============================================
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

# 已生成的文档（不用再生成）
ALREADY_GENERATED = [
    "感知机.md", "朴素贝叶斯.md", "决策树.md", "K-Means.md",
    "逻辑回归.md", "AdaBoost.md", "隐马尔可夫.md", "条件随机场.md",
    "GBDP.md", "奇异值分解.md", "变分EM.md", "马尔可夫链蒙特卡洛.md",
    "TD.md", "PPO.md"
]

# ============================================
# 2. 文档模板（14章节结构）
# ============================================
TEMPLATE = """# {algorithm} 学习文档

> {one_sentence_def}

---

## 1. 算法基础认知

### 一句话定义
{one_sentence_def}

### 直觉类比
{analogy}

### 历史背景
{historical_background}

### 算法定位
{algorithm_position}

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
{geometric_intuition}

---

## 3. 数学公式与推导

### 3.1 符号约定
{symbol_table}

### 3.2 问题形式化
{problem_formulation}

### 3.3 目标函数/损失函数
{objective_function}

### 3.4 推导过程
{derivation}

### 3.5 最终解/算法步骤
{algorithm_steps}

---

## 4. 训练过程讲解

### 4.1 数据预处理
{data_preprocessing}

### 4.2 参数初始化
{parameter_initialization}

### 4.3 迭代过程
{iteration_process}

### 4.4 收敛条件
{convergence_criteria}

### 4.5 超参数及推荐范围
{hyperparameters}

---

## 5. 应用场景

### 5.1 典型应用
{typical_applications}

### 5.2 适用数据特征
{applicable_data}

### 5.3 不适用场景
{unsuitable_scenarios}

---

## 6. 优缺点分析

### 6.1 优点
{advantages}

### 6.2 缺点
{disadvantages}

---

## 7. 调库实现（Python + 完整代码 + 注释）

{library_implementation}

---

## 8. 手工代码实现（核心算法手写 + 注释）

{manual_implementation}

---

## 9. 可视化与结果理解

{visualization}

---

## 10. 模型评估

{model_evaluation}

---

## 11. 常见问题与易错点

### 11.1 {common_issue_1_title}
**原因**：
{cause_1}

**解决方案**：
{solution_1}

### 11.2 {common_issue_2_title}
**原因**：
{cause_2}

**解决方案**：
{solution_2}

### 11.3 {common_issue_3_title}
**原因**：
{cause_3}

**解决方案**：
{solution_3}

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
问题：{exercise_1_problem}

<details>
<summary>答案</summary>

{exercise_1_solution}

### 习题2：编程实践**
问题：{exercise_2_problem}

<details>
<summary>答案</summary>

{exercise_2_solution}

### 习题3：理论推导**
问题：{exercise_3_problem}

<details>
<summary>答案</summary>

{exercise_3_solution}

### 思考题

**思考题1**：{thought_1_question}

<details>
<summary>答案</summary>

{thought_1_answer}

**思考题2**：{thought_2_question}

<details>
<summary>答案</summary>

{thought_2_answer}

---

## 14. 学习路径建议

### 初级阶段（掌握{algorithm}基础）
{basic_stage}

**学习时间**：{asic_time}

### 中级阶段（理解原理和扩展）
{intermediate_stage}

**学习时间**：{intermediate_time}

### 高级阶段（扩展到其他算法）
{advanced_stage}

**学习时间**：{advanced_time}

### 实践项目建议
1. **基础项目**：{project_1}
2. **进阶项目**：{project_2}
3. **挑战项目**：{project_3}

### 推荐资源
- **书籍**：{books}
- **课程**：{courses}
- **论文**：{papers}
- **代码**：{code_resources}
- **实践**：{practice}
"""

# ============================================
# 3. 算法内容数据库（示例，实际中需要填充）
# ============================================
# 注意：这是一个示例，实际中每个算法都需要详细内容
# 为演示，我们只提供几个算法的完整内容

ALGORITHM_DATA = {
    "RNN": {
        "one_sentence_def": "循环神经网络，通过隐藏状态建模序列数据，能够处理变长输入输出",
        "analogy": "想象你在读一句话，每读一个词，你会在脑中更新对整句话的理解（隐藏状态），影响对下一个词的预测",
        "historical_background": "RNN由John Hopfield在1982年提出，1980年代成为序列建模的主流，后来发展为LSTM、GRU等变体",
        "algorithm_position": "- 类型：监督学习 → 序列建模\n- 输出：序列标签或值\n- 模型类型：循环神经网络、判别式模型",
        "prerequisites": "- 基础神经网络知识：链式法则、反向传播\n- 序列数据理解：时序数据特性\n- 梯度消失/爆炸：理解RNN训练难点\n- Python基础：PyTorch/TensorFlow、循环结构",
        # ... 其他字段类似，需要填充
    },
    # 更多算法需要添加...
}

# ============================================
# 4. 生成函数
# ============================================
def generate_algorithm_content(algorithm_name):
    """
    为给定算法生成内容
    如果算法在ALGORITHM_DATA中有数据，使用它
    否则，使用通用模板和算法名称生成基本框架
    """
    if algorithm_name in ALGORITHM_DATA:
        return ALGORITHM_DATA[algorithm_name]
    else:
        # 通用内容生成（简化版）
        return {
            "one_sentence_def": f"{algorithm_name}是机器学习/深度学习中的重要算法",
            "analogy": f"想象{algorithm_name}的应用场景...",
            "historical_background": f"{algorithm_name}的历史发展...",
            "algorithm_position": f"- 类型：根据算法确定\n- 输出：根据算法确定\n- 模型类型：根据算法确定",
            "prerequisites": f"学习{algorithm_name}需要的前置知识...",
            # 其他字段类似，提供基本框架
            "core_idea": f"{algorithm_name}的核心思想是...",
            "workflow": "1. 初始化\n2. 迭代训练\n3. 输出结果",
            "key_concepts": "关键概念列表",
            "geometric_intuition": "几何直观解释",
            "symbol_table": "| 符号 | 含义 | 维度 |\n|------|------|----------|",
            "problem_formulation": "问题形式化描述",
            "objective_function": "目标函数描述",
            "derivation": "推导过程（关键步骤）",
            "algorithm_steps": "算法步骤（伪代码）",
            "data_preprocessing": "数据预处理要点",
            "parameter_initialization": "参数初始化建议",
            "iteration_process": "迭代过程代码（Python）",
            "convergence_criteria": "收敛条件",
            "hyperparameters": "| 超参数 | 作用 | 推荐范围 | 默认值 |",
            "typical_applications": "典型应用场景",
            "applicable_data": "适用数据特征",
            "unsuitable_scenarios": "不适用场景",
            "advantages": "| 优点 | 说明 | 成立条件 |",
            "disadvantages": "| 缺点 | 说明 | 缓解方法 |",
            "library_implementation": "调库实现（Python + 完整代码 + 注释）",
            "manual_implementation": "手工代码实现（核心算法手写 + 注释）",
            "visualization": "可视化与结果理解",
            "model_evaluation": "模型评估",
            "common_issue_1_title": "常见问题1",
            "cause_1": "原因分析",
            "solution_1": "解决方案代码",
            "common_issue_2_title": "常见问题2",
            "cause_2": "原因分析",
            "solution_2": "解决方案代码",
            "common_issue_3_title": "常见问题3",
            "cause_3": "原因分析",
            "solution_3": "解决方案代码",
            "core_points": "1. 要点1\n2. 要点2\n3. 要点3",
            "algorithm_chain": f"从{algorithm_name}到其他算法链条",
            "practical_tips": "1. 默认使用...\n2. 调整...\n3. ...",
            "exercise_1_problem": "基础计算问题",
            "exercise_1_solution": "答案",
            "exercise_2_problem": "编程实践问题",
            "exercise_2_solution": "代码示例",
            "exercise_3_problem": "理论推导问题",
            "exercise_3_solution": "推导过程",
            "thought_1_question": "思考题1",
            "thought_1_answer": "详细解答",
            "thought_2_question": "思考题2",
            "thought_2_answer": "详细解答",
            "basic_stage": "1. 理解基础\n2. 掌握核心\n3. 手动计算\n4. 使用库实现",
            "basic_time": "1-2周",
            "intermediate_stage": "1. 理解原理\n2. 掌握扩展\n3. 调参实践",
            "intermediate_time": "2-3周",
            "advanced_stage": "1. 学习高级变体\n2. 研究最新论文\n3. 实现复杂应用",
            "advanced_time": "3-4周",
            "project_1": f"基础项目：使用{algorithm_name}",
            "project_2": f"进阶项目：{algorithm_name}应用",
            "project_3": f"挑战项目：复杂{algorithm_name}系统",
            "books": "相关书籍推荐",
            "courses": "相关课程推荐",
            "papers": "经典论文推荐",
            "code_resources": "代码资源",
            "practice": "实践建议"
        }

def fill_template(algorithm_name, template, content):
    """
    将模板中的占位符替换为实际内容
    """
    filled = template.replace("{algorithm}", algorithm_name)
    
    for key, value in content.items():
        placeholder = "{" + key + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, str(value))
    
    # 替换剩余未填充的占位符
    import re
    filled = re.sub(r'\{[a-z_]+\}', '待补充', filled)
    
    return filled

def generate_all_documents(output_dir="algorithms", already_generated=None):
    """
    为所有算法生成文档
    """
    if already_generated is None:
        already_generated = []
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    generated = 0
    skipped = 0
    
    for algorithm in ALGORITHM_LIST:
        # 检查是否已生成
        filename = f"{algorithm}.md"
        if filename in already_generated:
            print(f"跳过（已生成）: {algorithm}")
            skipped += 1
            continue
        
        print(f"生成: {algorithm}")
        
        # 生成内容
        content = generate_algorithm_content(algorithm)
        filled_template = fill_template(algorithm, TEMPLATE, content)
        
        # 写入文件
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(filled_template)
        
        generated += 1
        print(f"  已写入: {file_path}")
    
    print(f"\n完成！生成了 {generated} 个新文档，跳过了 {skipped} 个已存在的文档")

if __name__ == "__main__":
    # 设置已生成的文档列表
    already = [f.replace('.md', '') for f in ALREADY_GENERATED]
    
    print("="*60)
    print("批量生成算法知识库文档")
    print("="*60)
    print(f"总算法数: {len(ALGORITHM_LIST)}")
    print(f"已生成: {len(ALREADY_GENERATED)}")
    print(f"待生成: {len(ALGORITHM_LIST) - len(ALREADY_GENERATED)}")
    print()
    
    # 生成文档
    generate_all_documents(
        output_dir="../algorithms",
        already_generated=already
    )
    
    print("\n提示：")
    print("1. 生成的文档包含基本框架和占位符")
    print("2. 需要手动填充每个算法的具体内容")
    print("3. 删除或重写已生成的文档请谨慎操作")
    print("4. 建议使用AI辅助填充详细内容")
