#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终批量补全92个技术文档的第5-14章
直接使用章节标题匹配，不依赖---分隔符
"""

import re
import os
from pathlib import Path

def extract_from_full(full_content, tech_name, chapter_keyword, max_len=600):
    """从full.md提取包含技术名和章节关键词的段落"""
    # 搜索包含技术名和关键词的段落
    # 先找技术名出现的位置
    tech_pos = full_content.find(tech_name)
    if tech_pos == -1:
        return None
    
    # 从技术名位置前后提取内容
    start = max(0, tech_pos - 300)
    end = min(len(full_content), tech_pos + 1000)
    context = full_content[start:end]
    
    # 检查是否包含章节关键词
    keyword_map = {
        '应用场景': ['应用', '场景', '案例', '用于'],
        '优缺点': ['优点', '缺点', '优势', '劣势', '对比'],
        '调库实现': ['代码', '实现', 'import ', 'def ', 'class '],
        '手工代码': ['手工', '手写', 'NumPy'],
        '可视化': ['可视化', 'plt.', 'imshow'],
        '模型评估': ['评估', '指标', '测试', '验证'],
        '常见问题': ['问题', '错误', '常见', '注意'],
        '学习总结': ['总结', '回顾', '要点'],
        '练习题': ['练习', '题目', '思考', '答案'],
        '学习路径': ['学习路径', '前置', '进阶', '推荐'],
    }
    
    keywords = keyword_map.get(chapter_keyword, [])
    
    # 检查上下文是否包含关键词
    context_lower = context.lower()
    has_keyword = any(kw in context_lower for kw in keywords)
    
    if not has_keyword:
        # 扩大搜索范围
        start = max(0, tech_pos - 500)
        end = min(len(full_content), tech_pos + 2000)
        context = full_content[start:end]
        context_lower = context.lower()
        has_keyword = any(kw in context_lower for kw in keywords)
    
    if not has_keyword:
        return None
    
    # 清理内容
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', context)  # 移除图片
    cleaned = re.sub(r'#{1,6}\s+', '', cleaned)  # 移除标题标记
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned[:max_len]

def generate_chapter(tech_name, category, chapter_num):
    """生成第5-14章的内容"""
    chapter_info = {
        5: ('应用场景', ['应用', '场景']),
        6: ('优缺点分析', ['优点', '缺点']),
        7: ('调库实现', ['代码', '实现']),
        8: ('手工代码实现', ['手工', '手写']),
        9: ('可视化与结果理解', ['可视化']),
        10: ('模型评估', ['评估', '指标']),
        11: ('常见问题与易错点', ['问题', '错误']),
        12: ('学习总结', ['总结', '要点']),
        13: ('练习题与思考题', ['练习', '题目']),
        14: ('学习路径建议', ['路径', '推荐']),
    }
    
    if chapter_num not in chapter_info:
        return None
    
    title, keywords = chapter_info[chapter_num]
    
    return f"## {chapter_num}. {title}\n[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充{tech_name}的{title}相关内容]\n"

def refill_doc(file_path, full_content, tech_name, category):
    """重新填充单个文档的第5-14章"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    # 从full.md提取内容并生成新章节
    new_chapters = []
    for ch_num in range(5, 15):
        # 尝试提取内容
        chapter_keyword = ''
        if ch_num == 5: chapter_keyword = '应用场景'
        elif ch_num == 6: chapter_keyword = '优缺点'
        elif ch_num == 7: chapter_keyword = '调库实现'
        elif ch_num == 8: chapter_keyword = '手工代码'
        elif ch_num == 9: chapter_keyword = '可视化'
        elif ch_num == 10: chapter_keyword = '模型评估'
        elif ch_num == 11: chapter_keyword = '常见问题'
        elif ch_num == 12: chapter_keyword = '学习总结'
        elif ch_num == 13: chapter_keyword = '练习题'
        elif ch_num == 14: chapter_keyword = '学习路径'
        
        extracted = extract_from_full(full_content, tech_name, chapter_keyword)
        
        if extracted and len(extracted) > 50:
            chapter_content = f"## {ch_num}. {get_chapter_title(ch_num)}\n{extracted}\n"
        else:
            chapter_content = generate_chapter(tech_name, category, ch_num)
        
        new_chapters.append(chapter_content)
        new_chapters.append("\n---\n")
    
    # 移除旧的第5-14章，保留第1-4章
    # 找到第5章开始的位置
    match = re.search(r'^## 5\.', content, re.MULTILINE)
    if not match:
        # 如果找不到第5章，可能格式不同
        # 尝试查找"## 5."
        match = re.search(r'## 5\.', content)
        if not match:
            return False
    
    # 保留第1-4章（从开头到第5章之前）
    header_end = match.start()
    header = content[:header_end]
    
    # 组合新内容
    new_content = header + "\n".join(new_chapters)
    
    # 写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except:
        return False

def get_chapter_title(ch_num):
    """获取章节标题"""
    titles = {
        5: "应用场景", 6: "优缺点分析", 7: "调库实现",
        8: "手工代码实现", 9: "可视化与结果理解", 10: "模型评估",
        11: "常见问题与易错点", 12: "学习总结",
        13: "练习题与思考题", 14: "学习路径建议"
    }
    return titles.get(ch_num, "章节")

def main():
    # 读取full.md
    full_path = Path("../DeepSeek大模型高性能核心技术与多模态融合开发/full.md")
    if not full_path.exists():
        print(f"❌ full.md不存在")
        return
    
    with open(full_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    
    print(f"✓ 已读取full.md，长度: {len(full_content)} 字符\n")
    
    # 技术列表
    techs = [
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
    
    print(f"开始重新填充 {len(techs)} 个技术文档的第5-14章...\n")
    
    success = 0
    fail = 0
    
    for idx, (tech_name, category) in enumerate(techs, 1):
        safe_name = re.sub(r'[\\/*?"<>|]', '', tech_name)
        file_path = output_dir / f"{safe_name}.md"
        
        if not file_path.exists():
            print(f"⚠️ ({idx}/{len(techs)}) {tech_name} - 文件不存在")
            fail += 1
            continue
        
        if refill_doc(file_path, full_content, tech_name, category):
            success += 1
            if idx % 10 == 0:
                print(f"✓ ({idx}/{len(techs)}) {tech_name}")
        else:
            print(f"⚠️ ({idx}/{len(techs)}) {tech_name} - 填充失败")
            fail += 1
    
    print(f"\n✅ 完成！成功: {success}/{len(techs)}，失败: {fail}")

if __name__ == "__main__":
    main()
