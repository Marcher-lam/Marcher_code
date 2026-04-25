#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新批量补全92个技术文档的真实内容
策略：从full.md提取包含技术名+章节关键词的段落，控制长度，填充到对应章节
"""

import re
import os
from pathlib import Path

def extract_tech_content_smart(full_content, tech_name, chapter_num):
    """
    智能提取技术内容
    策略：
    1. 找到tech_name的所有出现位置
    2. 对每个位置，检查后续内容是否包含章节关键词
    3. 提取匹配最好的段落
    """
    
    # 章节配置
    chapter_map = {
        5: {'keywords': ['应用', '场景', '案例', '用于', '实际'], 'max_len': 600},
        6: {'keywords': ['优点', '缺点', '优势', '劣势', '对比'], 'max_len': 500},
        7: {'keywords': ['代码', '实现', '示例', 'import ', 'def ', 'class '], 'max_len': 800},
        8: {'keywords': ['手工', '手写', 'NumPy', '从零'], 'max_len': 600},
        9: {'keywords': ['可视化', 'plt.', 'imshow', '绘制'], 'max_len': 400},
        10: {'keywords': ['评估', '指标', '测试', '验证'], 'max_len': 400},
        11: {'keywords': ['问题', '错误', '常见', '注意'], 'max_len': 400},
        12: {'keywords': ['总结', '回顾', '要点'], 'max_len': 300},
        13: {'keywords': ['练习', '题目', '思考', '答案'], 'max_len': 600},
        14: {'keywords': ['学习路径', '前置', '进阶', '推荐'], 'max_len': 400},
    }
    
    if chapter_num not in chapter_map:
        return None
    
    config = chapter_map[chapter_num]
    keywords = config['keywords']
    max_len = config['max_len']
    
    # 找到所有包含tech_name的位置
    pattern = re.compile(re.escape(tech_name), re.IGNORECASE)
    matches = list(pattern.finditer(full_content))
    
    best_paragraph = None
    best_score = 0
    
    for match in matches:
        start = max(0, match.start() - 200)  # 往前200字符
        end = min(len(full_content), match.end() + 800)  # 往后800字符
        context = full_content[start:end]
        
        # 计算相关性得分
        score = 1
        context_lower = context.lower()
        
        # 包含章节关键词加分
        for kw in keywords:
            if kw in context_lower:
                score += 3
        
        # 包含常见章节标记加分
        if '###' in context or '##' in context:
            score += 2
        
        # 如果技术水平线附近有章节关键词，得分更高
        line_start = max(0, full_content[:match.start()].rfind('\n'))
        line_end = full_content.find('\n', match.end())
        if line_end == -1:
            line_end = len(full_content)
        current_line = full_content[line_start:line_end]
        
        for kw in keywords:
            if kw in current_line.lower():
                score += 5
        
        # 提取段落
        # 清理：移除图片标记、简化空白
        paragraph = re.sub(r'!\[.*?\]\(.*?\)', '', context)
        paragraph = re.sub(r'#{1,6}\s+', '', paragraph)
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        
        if len(paragraph) > 50 and score > best_score:
            best_score = score
            best_paragraph = paragraph[:max_len]
    
    return best_paragraph

def batch_refill():
    """批量重新填充所有92个技术文档"""
    
    # 读取full.md
    full_path = Path("../DeepSeek大模型高性能核心技术与多模态融合开发/full.md")
    if not full_path.exists():
        print(f"❌ full.md不存在")
        return
    
    with open(full_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    
    print(f"✓ 已读取full.md，长度: {len(full_content)} 字符\n")
    
    # 技术列表
    tech_list = [
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
    
    output_dir = Path("../algorithm_knowledge_base/algorithms")
    if not output_dir.exists():
        print(f"❌ 输出目录不存在")
        return
    
    print(f"开始重新填充 {len(tech_list)} 个技术文档的第5-14章...\n")
    
    success = 0
    fail = 0
    
    for idx, (tech_name, category) in enumerate(tech_list, 1):
        safe_name = re.sub(r'[\\/*?"<>|]', '', tech_name)
        file_path = output_dir / f"{safe_name}.md"
        
        if not file_path.exists():
            print(f"⚠️ ({idx}/{len(tech_list)}) {tech_name} - 文件不存在")
            fail += 1
            continue
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 重构第5-14章
        new_chapters = []
        for ch_num in range(5, 15):
            # 提取内容
            extracted = extract_tech_content_smart(full_content, tech_name, ch_num)
            
            if not extracted:
                extracted = f"[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充{tech_name}相关内容]"
            
            # 章节标题
            chapter_titles = {
                5: "应用场景", 6: "优缺点分析", 7: "调库实现",
                8: "手工代码实现", 9: "可视化与结果理解", 10: "模型评估",
                11: "常见问题与易错点", 12: "学习总结",
                13: "练习题与思考题", 14: "学习路径建议"
            }
            title = chapter_titles[ch_num]
            
            new_chapters.append(f"\n## {ch_num}. {title}\n{extracted}\n")
            new_chapters.append("\n---\n")
        
        # 移除旧的章节（从第5章开始）
        # 找到第4章末尾的---
        parts = content.split("---\n")
        if len(parts) < 5:
            print(f"⚠️ ({idx}/{len(tech_list)}) {tech_name} - 格式错误")
            fail += 1
            continue
        
        # 保留前4章（第1-4章）
        header = "---\n".join(parts[:4]) + "---\n"
        
        # 组合新内容
        new_content = header + "".join(new_chapters)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        success += 1
        if idx % 10 == 0:
            print(f"✓ ({idx}/{len(tech_list)}) {tech_name}")
    
    print(f"\n✅ 完成！成功: {success}/{len(tech_list)}，失败: {fail}")

if __name__ == "__main__":
    batch_refill()
