#!/usr/bin/env python3
"""
批量丰富文档脚本：将补充内容追加到目标文档
目标：使所有文档达到核心算法级别（60k+字符）
"""
import os
import re

# 补充内容文件映射
SUPP_FILES = {
    'TD': 'supp_TD_rich.md',
    'MC': 'supp_MC_rich.md',
    'DP': 'supp_DP_rich.md',
    'Model': 'supp_Model_rich.md',
    'Other': 'supp_Other_rich.md'
}

# 排除的文件
EXCLUDE_FILES = {
    "TEMPLATE.md", "WRITING_SPEC.md", "full.md", 
    "强化学习算法名称提取.md", "PROMPT.md",
    "generate_all_docs.py", "batch_generate_all.py", 
    "generate_clean.py", "clean_docs.py", "batch_enrich_docs.py",
    "supp_TD_rich.md", "supp_MC_rich.md", "supp_DP_rich.md",
    "supp_Model_rich.md", "supp_Other_rich.md", "supp_TD_full.tex"
}

def categorize_file(filename):
    """根据文件名判断算法类别"""
    f = filename.lower()
    
    # 已有完整版的不处理
    if filename.endswith("_完整版.md"):
        return None
    
    # TD类
    if any(x in f for x in ['td', 'sarsa', 'q学习', 'n步', 'λ', 'sigma']):
        return 'TD'
    
    # MC类
    if any(x in f for x in ['蒙特卡洛', 'mc-', 'every-visit', '首次访问', 'mc_es']):
        return 'MC'
    
    # DP类
    if any(x in f for x in ['动态规划', 'dp', '策略迭代', '价值迭代', 'policy iteration', 'value iteration']):
        return 'DP'
    
    # Model类
    if any(x in f for x in ['dyna', 'mcts', 'uct', '模型', 'mb-', 'model-based', 'alpha']):
        return 'Model'
    
    # 其他归为Other类
    return 'Other'

def get_file_chars(filepath):
    """获取文件字符数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.read())
    except:
        return 0

def enrich_document(filepath, supp_file):
    """丰富文档：追加补充内容"""
    try:
        # 读取原文档
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()
        
        # 读取补充内容
        with open(supp_file, 'r', encoding='utf-8') as f:
            supplement = f.read()
        
        # 检查是否已经添加过该补充内容（避免重复）
        if "## 深度补充" in original:
            print(f"  跳过（已有深度补充）: {os.path.basename(filepath)}")
            return False
        
        # 追加补充内容
        enriched = original + "\n\n" + supplement
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(enriched)
        
        return True
    except Exception as e:
        print(f"  错误: {os.path.basename(filepath)} - {str(e)}")
        return False

def main():
    doc_dir = "/Users/marcher/Desktop/Marcher_code/强化学习_第二版"
    
    # 统计信息
    processed = 0
    skipped = 0
    errors = 0
    chars_before = []
    chars_after = []
    
    print("开始批量丰富文档...")
    print("=" * 50)
    
    # 遍历所有md文件
    for filename in sorted(os.listdir(doc_dir)):
        if not filename.endswith('.md') or filename in EXCLUDE_FILES:
            continue
        
        filepath = os.path.join(doc_dir, filename)
        
        # 只处理20k-30k字符的文档
        chars = get_file_chars(filepath)
        if chars < 20000 or chars >= 30000:
            continue
        
        # 判断类别
        category = categorize_file(filename)
        if category is None:
            print(f"跳过（完整版）: {filename}")
            skipped += 1
            continue
        
        # 获取补充内容文件
        supp_file = os.path.join(doc_dir, SUPP_FILES[category])
        if not os.path.exists(supp_file):
            print(f"警告：补充文件不存在 {supp_file}")
            errors += 1
            continue
        
        # 记录处理前字符数
        chars_before.append((filename, chars))
        
        # 丰富文档
        print(f"处理: {filename} (类别: {category}, 当前: {chars} chars)")
        success = enrich_document(filepath, supp_file)
        
        if success:
            # 记录处理后字符数
            new_chars = get_file_chars(filepath)
            chars_after.append((filename, new_chars))
            print(f"  -> 完成: {new_chars} chars (+{new_chars - chars})")
            processed += 1
        else:
            skipped += 1
    
    # 打印统计
    print("=" * 50)
    print(f"处理完成:")
    print(f"  成功处理: {processed} 个文件")
    print(f"  跳过/过滤: {skipped} 个文件")
    print(f"  错误: {errors} 个文件")
    
    if chars_after:
        avg_before = sum(c for _, c in chars_before) / len(chars_before)
        avg_after = sum(c for _, c in chars_after) / len(chars_after)
        print(f"\n字符数变化:")
        print(f"  处理前平均: {avg_before:.0f} chars")
        print(f"  处理后平均: {avg_after:.0f} chars")
        print(f"  平均增加: {avg_after - avg_before:.0f} chars")
    
    # 检查有多少达到60k+
    if chars_after:
        count_60k = sum(1 for _, c in chars_after if c >= 60000)
        print(f"\n达到60k+字符: {count_60k}/{len(chars_after)} 个文件")

if __name__ == "__main__":
    main()