#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量填充92个技术文档的第5-14章
从full.md提取相关内容替换占位符
"""

import re
import os
from pathlib import Path

def extract_related_paragraphs(full_content, tech_name, max_paragraphs=3):
    """从full.md提取与tech_name相关的段落"""
    # 构建搜索模式
    search_patterns = [
        tech_name,
        tech_name.replace(' ', ''),
        tech_name.lower(),
    ]
    
    # 添加别名
    aliases = {
        '注意力机制': ['Attention', '注意力'],
        'PyTorch': ['torch', 'PyTorch 2.0'],
        'Transformer': ['Transformer', 'transformer'],
        '多头注意力': ['Multi-Head', 'MHA'],
        'DeepSeek': ['DeepSeek-V2', 'DeepSeek-V3', 'DeepSeek-R1'],
    }
    
    if tech_name in aliases:
        search_patterns.extend(aliases[tech_name])
    
    found_paragraphs = []
    seen = set()
    
    for pattern in set(search_patterns):
        # 查找包含pattern的段落（以空行分隔）
        regex = r'([^\\n]+' + re.escape(pattern) + r'[^\\n]+(?=\\n\\s*\\n|\s*$))'
        matches = re.findall(regex, full_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches[:2]:  # 每个pattern最多取2段
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if len(cleaned) > 50 and cleaned not in seen:
                seen.add(cleaned)
                found_paragraphs.append(cleaned)
                if len(found_paragraphs) >= max_paragraphs:
                    break
        if len(found_paragraphs) >= max_paragraphs:
            break
    
    return found_paragraphs

def fill_tech_doc(tech_name, full_content, output_dir):
    """填充单个技术文档的第5-14章"""
    safe_name = re.sub(r'[\\/*?"<>|]', "", tech_name)
    file_path = output_dir / f"{safe_name}.md"
    
    if not file_path.exists():
        return False
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取相关内容
    paragraphs = extract_related_paragraphs(full_content, tech_name)
    extracted_text = '\n\n'.join(paragraphs) if paragraphs else '相关内容待补充'
    
    # 替换占位符
    placeholder_pattern = r'\[从full\.md提取的.*?\]'
    
    def replace_placeholder(match):
        return extracted_text if extracted_text != '相关内容待补充' else match.group(0)
    
    new_content = re.sub(placeholder_pattern, replace_placeholder, content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    # 读取full.md
    full_md_path = Path("../DeepSeek大模型高性能核心技术与多模态融合开发/full.md")
    if not full_md_path.exists():
        print(f"❌ full.md不存在: {full_md_path}")
        return
    
    with open(full_md_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    
    print(f"✓ 已读取full.md，长度: {len(full_content)} 字符\n")
    
    # 完整技术列表
    all_techs = [
        "注意力机制", "PyTorch", "Transformer", "多头注意力",
        "DeepSeek", "旋转位置编码", "扩散模型", "混合专家模型",
        "MoE", "词嵌入", "Miniconda", "PyCharm", "CUDA",
        "cuDNN", "ModelScope", "DeepSeek-V2", "DeepSeek-V3",
        "DeepSeek-R1", "DeepSeek-VL2", "GPT", "BERT", "Llama",
        "ChatGLM3", "自注意力机制", "多头潜在注意力", "分组查询注意力",
        "多查询注意力", "交叉注意力", "通道注意力", "动态注意力",
        "自适应注意力", "多模态注意力", "可解释性注意力", "差分注意力",
        "编码器", "解码器", "自编码器", "前馈网络", "多层感知机",
        "卷积神经网络", "循环神经网络", "LSTM", "GRU",
        "反向传播", "梯度下降", "AdamW", "余弦退火",
        "Layer归一化", "Batch归一化", "Dropout", "掩码", "残差连接",
        "SwiGLU", "GELU", "ReLU", "Sigmoid", "Tanh", "Softmax",
        "位置编码", "One-Hot编码", "Token", "多模态融合",
        "早期融合", "晚期融合", "混合融合", "DDPM", "VAE",
        "VQ-VAE", "FSQ", "GAN", "DCGAN", "UNet",
        "路由器", "门控网络", "强化学习", "PEFT", "LoRA",
        "知识蒸馏", "FP8混合精度", "双线性插值", "空洞卷积",
        "频谱图", "MFCC", "librosa", "情感分析", "机器翻译",
        "图像识别", "语音识别", "视频分类", "智能客服",
        "自动驾驶", "医学诊断",
    ]
    
    output_dir = Path("../algorithm_knowledge_base/algorithms")
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    print(f"开始填充 {len(all_techs)} 个技术文档的第5-14章...\n")
    
    success_count = 0
    for i, tech_name in enumerate(all_techs, 1):
        if fill_tech_doc(tech_name, full_content, output_dir):
            print(f"✓ ({i}/{len(all_techs)}) {tech_name}")
            success_count += 1
        else:
            print(f"⚠️ ({i}/{len(all_techs)}) {tech_name} - 文件不存在")
    
    print(f"\n✅ 完成！成功填充 {success_count}/{len(all_techs)} 个技术文档")

if __name__ == "__main__":
    main()
