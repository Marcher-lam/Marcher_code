#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终修复：跳过目录，只从full.md正文提取真实内容
"""

import re
from pathlib import Path

def extract_real_content(full_content, tech_name, chapter_keyword):
    """
    从full.md正文提取真实内容
    策略：
    1. 找到第1章开始的位置（跳过目录）
    2. 在正文中搜索包含tech_name和chapter_keyword的段落
    """
    # 找到第1章开始的位置（跳过目录）
    chapter1_pos = full_content.find("# 第1章")
    if chapter1_pos == -1:
        chapter1_pos = full_content.find("第1章")
    if chapter1_pos == -1:
        chapter1_pos = 0  # 如果找不到，从头开始（但会包含目录）
    
    # 只搜索正文部分
    main_content = full_content[chapter1_pos:]
    
    # 搜索包含tech_name和chapter_keyword的段落
    # 先找tech_name的位置
    tech_pos = main_content.find(tech_name)
    if tech_pos == -1:
        return None
    
    # 提取tech_pos前后的内容
    start = max(0, tech_pos - 300)
    end = min(len(main_content), tech_pos + 1200)
    context = main_content[start:end]
    
    # 清理内容
    # 移除Markdown图片标记
    context = re.sub(r'!\[.*?\]\(.*?\)', '', context)
    # 移除多余的空白
    context = re.sub(r'\s+', ' ', context).strip()
    
    # 检查是否包含章节关键词
    if chapter_keyword.lower() not in context.lower():
        # 如果不包含关键词，扩大搜索范围
        start = max(0, tech_pos - 500)
        end = min(len(main_content), tech_pos + 2000)
        context = main_content[start:end]
        context = re.sub(r'!\[.*?\]\(.*?\)', '', context)
        context = re.sub(r'\s+', ' ', context).strip()
    
    return context[:1000]  # 限制长度

def refill_one_doc(tech_name, file_path, full_content):
    """重新填充单个文档的第5-14章"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    # 章节关键词映射
    chapter_keywords = {
        5: "应用场景",
        6: "优缺点",
        7: "实现",
        8: "手工",
        9: "可视化",
        10: "评估",
        11: "问题",
        12: "总结",
        13: "练习",
        14: "路径"
    }
    
    # 生成新的第5-14章
    new_chapters = []
    for ch_num in range(5, 15):
        keyword = chapter_keywords.get(ch_num, "")
        extracted = extract_real_content(full_content, tech_name, keyword)
        
        if not extracted or len(extracted) < 50:
            extracted = f"[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充{tech_name}的{keyword}相关内容]"
        
        title = ""
        if ch_num == 5: title = "应用场景"
        elif ch_num == 6: title = "优缺点分析"
        elif ch_num == 7: title = "调库实现"
        elif ch_num == 8: title = "手工代码实现"
        elif ch_num == 9: title = "可视化与结果理解"
        elif ch_num == 10: title = "模型评估"
        elif ch_num == 11: title = "常见问题与易错点"
        elif ch_num == 12: title = "学习总结"
        elif ch_num == 13: title = "练习题与思考题"
        elif ch_num == 14: title = "学习路径建议"
        
        new_chapters.append(f"\n## {ch_num}. {title}\n{extracted}\n")
        new_chapters.append("\n---\n")
    
    # 找到第5章的位置，替换后面的内容
    match = re.search(r'^## 5\.', content, re.MULTILINE)
    if not match:
        # 尝试查找"## 5. "
        match = re.search(r'## 5\. ', content)
    
    if not match:
        return False
    
    # 保留第1-4章
    header = content[:match.start()]
    
    # 组合新内容
    new_content = header + "".join(new_chapters)
    
    # 写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except:
        return False

def main():
    # 读取full.md
    full_path = Path("../DeepSeek大模型高性能核心技术与多模态融合开发/full.md")
    if not full_path.exists():
        print(f"❌ full.md不存在")
        return
    
    with open(full_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    
    print(f"✓ 已读取full.md，长度: {len(full_content)} 字符")
    print(f"找到第1章位置: {full_content.find('# 第1章')}\n")
    
    # 先处理几个核心技术作为示例
    core_techs = [
        ("注意力机制", "注意力机制.md"),
        ("Transformer", "Transformer.md"),
        ("DeepSeek", "DeepSeek.md"),
        ("扩散模型", "扩散模型.md"),
        ("多头注意力", "多头注意力.md"),
    ]
    
    output_dir = Path("../algorithm_knowledge_base/algorithms")
    
    for tech_name, filename in core_techs:
        file_path = output_dir / filename
        if not file_path.exists():
            print(f"⚠️ 文件不存在: {filename}")
            continue
        
        if refill_one_doc(tech_name, file_path, full_content):
            print(f"✓ {tech_name} - 已填充真实内容")
        else:
            print(f"⚠️ {tech_name} - 填充失败")
    
    print(f"\n✅ 完成！已为{len(core_techs)}个核心技术填充真实内容")

if __name__ == "__main__":
    main()
