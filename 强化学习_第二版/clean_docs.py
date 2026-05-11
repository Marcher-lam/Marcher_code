#!/usr/bin/env python3
import os
import re

# 标准14章节列表
STANDARD_CHAPTERS = [
    "## 1. 算法基础认知",
    "## 2. 核心原理",
    "## 3. 数学公式与推导",
    "## 4. 训练过程讲解",
    "## 5. 应用场景",
    "## 6. 优缺点分析",
    "## 7. 调库实现",
    "## 8. 手工代码实现",
    "## 9. 可视化与结果理解",
    "## 10. 模型评估",
    "## 11. 常见问题与易错点",
    "## 12. 学习总结",
    "## 13. 练习题与思考题",
    "## 14. 学习路径建议"
]

# 排除的非标准文件
EXCLUDE_FILES = {
    "TEMPLATE.md", "WRITING_SPEC.md", "full.md", 
    "强化学习算法名称提取.md", "PROMPT.md",
    "generate_all_docs.py", "batch_generate_all.py", 
    "generate_clean.py", "clean_docs.py"
}

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按章节分割内容（## 开头的行）
    chapters = re.split(r'(^## .+$)', content, flags=re.MULTILINE)
    
    # 重建内容：只保留标准章节
    cleaned_lines = []
    current_chapter = None
    
    for part in chapters:
        part = part.strip()
        if not part:
            continue
        
        # 如果是章节标题
        if part.startswith('## '):
            # 检查是否是标准章节
            if part in STANDARD_CHAPTERS:
                current_chapter = part
                cleaned_lines.append(part)
            else:
                # 非标准章节（如补充内容），跳过
                current_chapter = None
        else:
            # 是章节内容，只保留当前章节是标准章节的内容
            if current_chapter is not None:
                cleaned_lines.append(part)
    
    # 重新组合内容
    cleaned_content = '\n\n'.join(cleaned_lines)
    
    # 检查是否有14个标准章节
    chapter_count = sum(1 for line in cleaned_content.split('\n') if line.startswith('## '))
    
    # 如果章节数不对，补充缺失的章节（空内容）
    if chapter_count < 14:
        existing_chapters = [line for line in cleaned_content.split('\n') if line.startswith('## ')]
        for std_ch in STANDARD_CHAPTERS:
            if std_ch not in existing_chapters:
                # 在对应位置插入缺失的章节
                pass  # 简化处理，先不补充，只清理
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    return chapter_count, len(cleaned_content)

def main():
    doc_dir = "/Users/marcher/Desktop/Marcher_code/强化学习_第二版"
    processed = 0
    errors = []
    
    for filename in os.listdir(doc_dir):
        if not filename.endswith('.md'):
            continue
        if filename in EXCLUDE_FILES:
            continue
        
        filepath = os.path.join(doc_dir, filename)
        try:
            chapter_count, char_count = clean_file(filepath)
            processed += 1
            if chapter_count != 14:
                print(f"警告: {filename} 有 {chapter_count} 个章节 (应为14)")
            if char_count < 10000:
                print(f"警告: {filename} 只有 {char_count} 字符 (应>=10k)")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    print(f"处理完成，共处理 {processed} 个文件")
    if errors:
        print("错误:")
        for err in errors[:10]:
            print(err)

if __name__ == "__main__":
    main()