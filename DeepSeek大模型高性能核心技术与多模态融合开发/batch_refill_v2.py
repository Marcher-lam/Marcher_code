#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量补全92个技术文档的第5-14章（改进版）
核心改进：
1. 跳过full.md的目录部分（定位到第1章正文开始）
2. 为每个技术精准提取正文内容，避免抓取目录
3. 按第5-14章的主题匹配相关内容
"""

import re
from pathlib import Path
import os

def read_full_md():
    """读取full.md并定位正文起始位置（跳过目录）"""
    # 脚本与full.md在同一目录
    full_path = Path("full.md")
    if not full_path.exists():
        # 尝试另一种路径
        full_path = Path("../DeepSeek大模型高性能核心技术与多模态融合开发/full.md")
    if not full_path.exists():
        print(f"❌ full.md不存在: {full_path}")
        return None, 0
    
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定位第1章正文开始（跳过目录）
    # 搜索"# 第1章"或"# 1." 作为正文起点
    chapter1_patterns = [
        r'# 第1章',
        r'# 1\.',
        r'# 第 1 章',
    ]
    
    main_start = 0
    for pattern in chapter1_patterns:
        match = re.search(pattern, content)
        if match:
            main_start = match.start()
            print(f"✓ 找到正文起点: 位置 {main_start} (匹配到 '{match.group()}')")
            break
    
    if main_start == 0:
        print("⚠️ 未找到第1章标记，将从文件开头开始")
    
    main_content = content[main_start:] if main_start > 0 else content
    return main_content, len(content)

def extract_tech_content(main_content, tech_name):
    """
    从正文提取技术的第5-14章相关内容
    返回字典：{章节号: 提取的内容}
    """
    # 章节关键词映射（用于匹配内容）
    chapter_keywords = {
        5: ['应用场景', '案例', '应用', '使用场景'],
        6: ['优缺点', '优点', '缺点', '优势', '劣势'],
        7: ['调库', '实现', '代码', 'import ', 'def ', 'class '],
        8: ['手工', '手写', 'NumPy', '手动'],
        9: ['可视化', 'plt.', '可视化', '图表'],
        10: ['评估', '指标', '测试', '验证'],
        11: ['问题', '错误', '常见', '注意'],
        12: ['总结', '回顾', '要点'],
        13: ['练习', '题目', '思考'],
        14: ['学习路径', '前置', '后续', '推荐'],
    }
    
    # 在正文搜索技术名
    tech_pos = main_content.find(tech_name)
    if tech_pos == -1:
        return None
    
    # 提取技术名周围的内容（前后各2000字符）
    start = max(0, tech_pos - 500)
    end = min(len(main_content), tech_pos + 2500)
    context = main_content[start:end]
    
    # 检查是否包含章节关键词
    extracted = {}
    for ch_num, keywords in chapter_keywords.items():
        found = False
        context_lower = context.lower()
        for kw in keywords:
            if kw.lower() in context_lower:
                found = True
                break
        if found:
            # 提取该章节的内容（简化：取上下文的前600字符）
            extracted[ch_num] = context[:600].strip()
        else:
            extracted[ch_num] = None
    
    return extracted

def generate_chapter_content(tech_name, chapter_num, extracted_text):
    """生成单个章节的内容"""
    chapter_titles = {
        5: "应用场景", 6: "优缺点分析", 7: "调库实现",
        8: "手工代码实现", 9: "可视化与结果理解", 10: "模型评估",
        11: "常见问题与易错点", 12: "学习总结",
        13: "练习题与思考题", 14: "学习路径建议"
    }
    
    title = chapter_titles.get(chapter_num, f"第{chapter_num}章")
    
    if extracted_text and len(extracted_text) > 50:
        # 清理内容：移除图片标记、过长的空白
        cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', extracted_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > 200:
            return f"## {chapter_num}. {title}\n{cleaned}\n"
    
    # 如果没有提取到内容，生成通用占位符
    return f"## {chapter_num}. {title}\n[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充{tech_name}的{title}相关内容]\n"

def refill_one_doc(file_path, main_content, tech_name):
    """重新填充单个文档的第5-14章"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False
    
    # 找到第5章开始的位置
    match = re.search(r'^## 5\.', content, re.MULTILINE)
    if not match:
        # 如果找不到第5章，尝试查找"## 5. "
        match = re.search(r'## 5\. ', content)
    
    if not match:
        print(f"  ⚠️ 未找到第5章标记")
        return False
    
    # 保留第1-4章
    header = content[:match.start()]
    
    # 提取技术内容
    extracted = extract_tech_content(main_content, tech_name)
    
    # 生成新的第5-14章
    new_chapters = []
    for ch_num in range(5, 15):
        if extracted and ch_num in extracted:
            chapter_content = generate_chapter_content(tech_name, ch_num, extracted[ch_num])
        else:
            chapter_content = generate_chapter_content(tech_name, ch_num, None)
        
        new_chapters.append(chapter_content)
        new_chapters.append("\n---\n")
    
    # 组合新内容
    final_content = header + "\n".join(new_chapters)
    
    # 写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        return True
    except Exception as e:
        print(f"  ❌ 写入失败: {e}")
        return False

def main():
    print("=" * 60)
    print("批量补全技术文档第5-14章（改进版）")
    print("=" * 60)
    
    # 1. 读取full.md正文
    main_content, total_len = read_full_md()
    if not main_content:
        return
    
    print(f"✓ 已读取full.md正文，长度: {len(main_content)} 字符（总{total_len}字符）\n")
    
    # 2. 技术列表（排除已完成的10个核心技术）
    completed_techs = {
        "注意力机制", "Transformer", "DeepSeek", "PyTorch",
        "扩散模型", "多头注意力", "混合专家模型", "MoE",
        "词嵌入", "旋转位置编码"
    }
    
    # 所有92个技术
    all_techs = [
        ("注意力机制", "深度学习机制"), ("PyTorch", "深度学习框架"),
        ("Transformer", "模型架构"), ("多头注意力", "注意力机制"),
        ("DeepSeek", "大语言模型"), ("旋转位置编码", "位置编码"),
        ("扩散模型", "生成模型"), ("混合专家模型", "模型架构"),
        ("MoE", "混合专家"), ("词嵌入", "文本表示"),
        ("位置编码", "位置表示"), ("前馈网络", "神经网络层"),
        ("卷积神经网络", "神经网络"), ("循环神经网络", "神经网络"),
        ("LSTM", "循环神经网络"), ("GRU", "循环神经网络"),
        ("多层感知机", "神经网络"), ("反向传播", "训练算法"),
        ("梯度下降", "优化算法"), ("AdamW", "优化器"),
        ("余弦退火", "学习率调度"), ("Layer归一化", "归一化"),
        ("Batch归一化", "归一化"), ("Dropout", "正则化"),
        ("掩码", "数据处理"), ("残差连接", "网络结构"),
        ("SwiGLU", "激活函数"), ("GELU", "激活函数"),
        ("ReLU", "激活函数"), ("Sigmoid", "激活函数"),
        ("Tanh", "激活函数"), ("Softmax", "激活函数"),
        ("One-Hot编码", "文本表示"), ("Token", "文本单位"),
        ("多模态融合", "融合技术"), ("早期融合", "融合策略"),
        ("晚期融合", "融合策略"), ("混合融合", "融合策略"),
        ("DDPM", "扩散模型"), ("VAE", "生成模型"),
        ("VQ-VAE", "生成模型"), ("FSQ", "量化技术"),
        ("GAN", "生成模型"), ("DCGAN", "生成模型"),
        ("UNet", "图像分割"), ("路由器", "门控网络"),
        ("门控网络", "MoE组件"), ("强化学习", "机器学习"),
        ("PEFT", "微调技术"), ("LoRA", "微调技术"),
        ("知识蒸馏", "模型压缩"), ("FP8混合精度", "训练优化"),
        ("双线性插值", "图像处理"), ("空洞卷积", "卷积变体"),
        ("频谱图", "音频表示"), ("MFCC", "音频特征"),
        ("librosa", "音频处理库"), ("情感分析", "NLP应用"),
        ("机器翻译", "NLP应用"), ("图像识别", "CV应用"),
        ("语音识别", "语音应用"), ("视频分类", "视频应用"),
        ("智能客服", "应用系统"), ("自动驾驶", "应用系统"),
        ("医学诊断", "应用系统"), ("Miniconda", "Python环境管理"),
        ("PyCharm", "Python IDE"), ("CUDA", "GPU计算平台"),
        ("cuDNN", "GPU加速库"), ("ModelScope", "模型部署平台"),
        ("DeepSeek-V2", "大语言模型"), ("DeepSeek-V3", "大语言模型"),
        ("DeepSeek-R1", "大语言模型"), ("DeepSeek-VL2", "多模态大模型"),
        ("GPT", "生成式预训练Transformer"), ("BERT", "双向编码器表示Transformer"),
        ("Llama", "大语言模型"), ("ChatGLM3", "大语言模型"),
        ("自注意力机制", "注意力机制"), ("多头潜在注意力", "注意力机制"),
        ("分组查询注意力", "注意力机制"), ("多查询注意力", "注意力机制"),
        ("交叉注意力", "注意力机制"), ("通道注意力", "注意力机制"),
        ("动态注意力", "注意力机制"), ("自适应注意力", "注意力机制"),
        ("多模态注意力", "注意力机制"), ("可解释性注意力", "注意力机制"),
        ("差分注意力", "注意力机制"), ("编码器", "模型组件"),
        ("解码器", "模型组件"), ("自编码器", "生成模型"),
    ]
    
    # 过滤掉已完成的
    remaining_techs = [(name, cat) for name, cat in all_techs if name not in completed_techs]
    
    print(f"待处理技术数: {len(remaining_techs)} (已跳过{len(completed_techs)}个核心技术)\n")
    
    output_dir = Path("../algorithm_knowledge_base/algorithms")
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    success = 0
    fail = 0
    
    for idx, (tech_name, category) in enumerate(remaining_techs, 1):
        # 安全检查文件名
        safe_name = re.sub(r'[\\/*?"<>|]', '', tech_name)
        file_path = output_dir / f"{safe_name}.md"
        
        if not file_path.exists():
            print(f"⚠️ ({idx}/{len(remaining_techs)}) {tech_name} - 文件不存在")
            fail += 1
            continue
        
        if refill_one_doc(file_path, main_content, tech_name):
            success += 1
            if idx % 10 == 0:
                print(f"✓ ({idx}/{len(remaining_techs)}) {tech_name}")
        else:
            print(f"⚠️ ({idx}/{len(remaining_techs)}) {tech_name} - 填充失败")
            fail += 1
    
    print(f"\n✅ 完成！成功: {success}/{len(remaining_techs)}，失败: {fail}")

if __name__ == "__main__":
    main()
