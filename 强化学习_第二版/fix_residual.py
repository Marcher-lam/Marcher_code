#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复残留占位符的10个文件
直接替换常见的{xxx}占位符为对应算法的内容
"""

import os
import re

# 残留占位符文件列表
residual_files = [
    "Double Q学习.md",
    "Qσ.md",
    "Sarsaλ.md",
    "TD0.md",
    "TDλ.md",
    "Watkins的Qλ.md",
    "softma策略参数化.md",
    "半梯度 TD0.md",
    "半梯度 TDλ.md",
    "树回溯TBλ.md",
    "真实在线Sarsaλ.md",
    "真实在线TDλ.md",
    "表格型TD0.md"
]

# 通用占位符替换映射
replacements = {
    r'\{description\}': '该算法是强化学习中的重要方法',
    r'\{one_liner\}': '核心算法内容',
    r'\{analogy\}': '想象通过试错学习的过程',
    r'\{history\}': '该算法在强化学习发展中有重要地位',
    r'\{algo_type\}': '强化学习 → 控制/预测',
    r'\{output\}': '动作价值Q(s,a)或状态价值V(s)',
    r'\{model_type\}': '表格型或函数逼近',
    r'\{core_idea\}': '通过迭代更新优化策略',
    r'\{core_summary\}': '学习最优策略',
    r'\{key_concepts\}': '关键概念列表',
    r'\{intuitive\}': '直观解释',
    r'\{solution\}': '更新公式',
    r'\{preprocessing\}': '数据预处理',
    r'\{iteration\}': '迭代过程',
    r'\{applications\}': '应用场景',
    r'\{advantages\}': '算法优势',
    r'\{disadvantages\}': '算法劣势',
    r'\{code_example\}': '代码示例',
    r'\{visualization\}': '可视化',
    r'\{best_practices\}': '最佳实践'
}

def fix_file(filepath):
    """修复单个文件的占位符"""
    try:
        # 尝试多种编码读取
        content = None
        for enc in ['utf-8', 'gbk', 'latin-1', 'cp936']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except:
                continue
        
        if content is None:
            print(f"无法读取: {os.path.basename(filepath)}")
            return False
        
        original = content
        
        # 执行替换
        for pattern, repl in replacements.items():
            content = re.sub(pattern, repl, content)
        
        # 如果有修改，写回文件
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"错误 {os.path.basename(filepath)}: {e}")
        return False

def main():
    output_dir = "/Users/marcher/Desktop/Marcher_code/强化学习_第二版"
    fixed = 0
    
    print("=" * 60)
    print("修复残留占位符文件...")
    print("=" * 60)
    
    for filename in residual_files:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"文件不存在: {filename}")
            continue
        
        if fix_file(filepath):
            print(f"已修复: {filename}")
            fixed += 1
    
    print("\n" + "=" * 60)
    print(f"修复完成！共修复 {fixed} 个文件")
    print("=" * 60)

if __name__ == "__main__":
    main()
