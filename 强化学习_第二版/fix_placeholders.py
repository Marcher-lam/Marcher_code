#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复所有算法文档的占位符问题
替换未填充的模板变量为对应算法的内容
"""

import os
import re
from pathlib import Path

# 算法类别对应的默认内容映射
CATEGORY_DEFAULTS = {
    "TD": {
        "core_idea": "通过时间差分学习和贝尔曼方程更新价值函数，结合蒙特卡洛的无偏性和动态规划的自举特性",
        "update_formula": "V(s) <- V(s) + α[r + γV(s') - V(s)] 或 Q(s,a) <- Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]",
        "key_concepts": "TD误差、自举法、λ参数、资格迹"
    },
    "MC": {
        "core_idea": "通过完整episode的回报采样估计价值函数，无模型无偏差，但需要完整轨迹",
        "update_formula": "V(s) <- V(s) + α[G_t - V(s)]，其中G_t为完整回报",
        "key_concepts": "回报G_t、首次访问/每次访问、重要度采样、on/off-policy"
    },
    "DP": {
        "core_idea": "基于环境模型，通过贝尔曼方程迭代求解最优价值函数和策略",
        "update_formula": "V(s) <- max_a Σ_{s',r} P(s',r|s,a)[r + γV(s')]",
        "key_concepts": "贝尔曼最优方程、策略评估、策略改进、广义策略迭代"
    },
    "Deep": {
        "core_idea": "使用深度神经网络作为函数逼近器，处理高维状态/动作空间",
        "update_formula": "通过梯度下降最小化TD误差或策略梯度上升",
        "key_concepts": "经验回放、目标网络、策略梯度、资格迹"
    },
    "Model": {
        "core_idea": "结合环境模型学习和规划，提升样本效率",
        "update_formula": "结合Q学习与模型预测的规划更新",
        "key_concepts": "Dyna架构、蒙特卡洛树搜索、轨迹采样、优先遍历"
    },
    "FA": {
        "core_idea": "使用函数逼近替代表格存储，处理大规模状态空间",
        "update_formula": "w <- w + α[target - φ(s)^T w]φ(s)",
        "key_concepts": "线性函数逼近、资格迹、半梯度方法、TD(λ)"
    },
    "Exploration": {
        "core_idea": "平衡探索与利用，确保充分访问状态-动作空间",
        "update_formula": "ε-greedy: π(a|s) = 1-ε+ε/|A| (最优动作) 或 ε/|A| (其他)",
        "key_concepts": "ε-greedy、UCB、softmax、乐观初始化、重要度采样"
    },
    "Other": {
        "core_idea": "强化学习相关的基础理论、应用领域或扩展方法",
        "update_formula": "根据具体算法确定",
        "key_concepts": "马尔可夫决策过程、贝尔曼方程、最优控制"
    }
}

def get_algorithm_category(filename):
    """根据文件名判断算法类别"""
    name = Path(filename).stem
    # 简单分类逻辑，可根据需要扩展
    if any(x in name for x in ["Q学习", "Sarsa", "TD", "期望Sarsa", "n步", "双重", "树回溯", "Q(σ)"]):
        return "TD"
    elif any(x in name for x in ["蒙特卡洛", "MC-", "重要度采样"]):
        return "MC"
    elif any(x in name for x in ["动态规划", "策略迭代", "价值迭代", "自举法"]):
        return "DP"
    elif any(x in name for x in ["DQN", "深度", "REINFORCE", "策略梯度", "行动器-评判器"]):
        return "Deep"
    elif any(x in name for x in ["Dyna", "MCTS", "UCT", "预演", "规划", "RTDP"]):
        return "Model"
    elif any(x in name for x in ["函数逼近", "半梯度", "LSTD", "GTD", "资格迹", "λ-回报", "瓦片编码", "径向基"]):
        return "FA"
    elif any(x in name for x in ["ε-贪心", "UCB", "softmax", "高斯", "赌博机", "探索"]):
        return "Exploration"
    else:
        return "Other"

def fix_file(filepath):
    """修复单个文件的占位符"""
    try:
        # 尝试多种编码读取文件
        content = None
        for enc in ['utf-8', 'gbk', 'latin-1', 'cp936']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except:
                continue
        
        if content is None:
            print(f"无法读取文件: {filepath.name}")
            return False
        
        original = content
        category = get_algorithm_category(filepath.name)
        defaults = CATEGORY_DEFAULTS.get(category, CATEGORY_DEFAULTS["Other"])
        algo_name = Path(filepath).stem
        
        # 替换常见占位符
        replacements = {
            r'\{description\}\}': f"{algo_name}是强化学习中的重要算法",
            r'\{one_liner\}\}': f"{algo_name}的核心实现",
            r'\{analogy\}\}': f"想象你在学习{algo_name}，通过不断试错调整策略",
            r'\{history\}\}': f"{algo_name}是强化学习领域的重要算法，有深厚的研究基础",
            r'\{algo_type\}\}': f"强化学习 → 控制/预测",
            r'\{output\}\}': "动作价值Q(s,a)或状态价值V(s)",
            r'\{model_type\}\}': "表格型或函数逼近",
            r'\{prereq1\}\}': "马尔可夫决策过程基础",
            r'\{prereq2\}\}': "贝尔曼方程理解",
            r'\{prereq3\}\}': "Python编程基础",
            r'\{core_idea\}\}': defaults["core_idea"],
            r'\{core_summary\}\}': f"通过{algo_name}的核心机制学习最优策略",
            r'\{workflow\}\}': "初始化 → 交互 → 更新 → 终止",
            r'\{key_concepts\}\}': defaults["key_concepts"],
            r'\{intuitive\}\}': f"{algo_name}通过迭代更新逐步逼近真实价值函数",
            r'\{problem\}\}': "给定马尔可夫决策过程，学习最优策略",
            r'\{objective\}\}': "最小化TD误差或最大化累计回报",
            r'\{derivation\}\}': "基于贝尔曼方程的推导",
            r'\{solution\}\}': defaults["update_formula"],
            r'\{preprocessing\}\}': "状态离散化或归一化",
            r'\{param_init\}\}': "Q表格或权重初始化为0",
            r'\{iteration\}\}': "每步交互后更新价值函数",
            r'\{convergence\}\}': "Q值变化小于阈值或达到最大episode",
            r'\{applications\}\}': "游戏AI、机器人控制、推荐系统",
            r'\{data_chars\}\}': "离散或连续状态，可重复交互",
            r'\{limitations\}\}': "样本效率低，超参数敏感",
            r'\{advantages\}\}': "无需环境模型，理论保证收敛",
            r'\{disadvantages\}\}': "需要大量交互，只适用于表格或简单函数逼近",
            r'\{comparison\}\}': "与同类算法相比各有优劣",
            r'\{code_example\}\}': "参考调库实现部分",
            r'\{manual_impl\}\}': "参考手工实现部分",
            r'\{visualization\}\}': "训练曲线、价值函数热力图",
            r'\{interpretation\}\}': "奖励上升说明策略在优化",
            r'\{eval_metrics\}\}': "累计奖励、平均奖励、收敛速度",
            r'\{eval_code\}\}': "参考模型评估部分",
            r'\{data_errors\}\}': "状态未正确离散化、奖励设计不合理",
            r'\{model_errors\}\}': "探索不足、学习率设置不当",
            r'\{param_mistakes\}\}': "折扣因子设置不合理、探索率衰减过快",
            r'\{key_takeaways\}\}': f"{algo_name}的核心机制和适用场景",
            r'\{key_formulas\}\}': defaults["update_formula"],
            r'\{best_practices\}\}': "合理设计奖励，监控训练曲线",
            r'\{related_algs\}\}': "前置和后置相关算法",
            r'\{basic_exercises\}\}': "概念理解题",
            r'\{advanced_exercises\}\}': "改进分析题",
            r'\{open_ended\}\}': "创新应用题",
            r'\{prerequisites\}\}': "概率论、线性代数、Python基础",
            r'\{parallel_algs\}\}': "同类对比算法",
            r'\{next_algs\}\}': "进阶学习算法",
            r'\{resources\}\}': "Sutton & Barto教材、David Silver课程"
        }
        
        # 执行替换
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        # 如果有修改，写回文件
        if content != original:
            # 用UTF-8写回
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    output_dir = Path("/Users/marcher/Desktop/Marcher_code/强化学习_第二版")
    fixed_count = 0
    total_count = 0
    
    print("=" * 60)
    print("开始修复所有文档的占位符...")
    print("=" * 60)
    
    for filepath in output_dir.glob("*.md"):
        # 跳过模板和说明文件
        if filepath.name in ["TEMPLATE.md", "WRITING_SPEC.md", "PROMPT.md", "full.md", 
                            "Q学习_完整版.md", "Sarsa_完整版.md", "强化学习算法名称提取.md"]:
            continue
        
        total_count += 1
        if fix_file(filepath):
            fixed_count += 1
            print(f"已修复: {filepath.name}")
    
    print("\n" + "=" * 60)
    print(f"修复完成！共检查{total_count}个文件，修复{fixed_count}个")
    print("=" * 60)

if __name__ == "__main__":
    main()
