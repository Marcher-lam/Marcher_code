#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从full.md提取技术术语并生成对应的markdown文档
使用TEMPLATE.md作为模板
"""

import re
import os
from pathlib import Path

# 读取TEMPLATE.md
template_path = Path("../algorithm_knowledge_base/TEMPLATE.md")
with open(template_path, 'r', encoding='utf-8') as f:
    template_content = f.read()

# 读取full.md
full_md_path = Path("full.md")
with open(full_md_path, 'r', encoding='utf-8') as f:
    full_content = f.read()

# 定义主要技术术语及其分类
technologies = {
    # 深度学习框架和工具
    "PyTorch": "深度学习框架",
    "Miniconda": "Python环境管理",
    "PyCharm": "Python IDE",
    "CUDA": "GPU计算平台",
    "cuDNN": "GPU加速库",
    "ModelScope": "模型部署平台",
    
    # 大模型
    "DeepSeek": "大语言模型",
    "DeepSeek-V2": "大语言模型",
    "DeepSeek-V3": "大语言模型", 
    "DeepSeek-R1": "大语言模型",
    "DeepSeek-VL2": "多模态大模型",
    "GPT": "生成式预训练Transformer",
    "BERT": "双向编码器表示Transformer",
    "Llama": "大语言模型",
    "ChatGLM3": "大语言模型",
    
    # 注意力机制
    "注意力机制": "深度学习机制",
    "自注意力机制": "注意力机制",
    "多头注意力": "注意力机制",
    "多头潜在注意力": "注意力机制",
    "分组查询注意力": "注意力机制",
    "多查询注意力": "注意力机制",
    "旋转位置编码": "位置编码",
    "交叉注意力": "注意力机制",
    "通道注意力": "注意力机制",
    "动态注意力": "注意力机制",
    "自适应注意力": "注意力机制",
    "多模态注意力": "注意力机制",
    "可解释性注意力": "注意力机制",
    "差分注意力": "注意力机制",
    
    # 模型架构
    "Transformer": "模型架构",
    "编码器": "模型组件",
    "解码器": "模型组件",
    "自编码器": "生成模型",
    "前馈网络": "神经网络层",
    "多层感知机": "神经网络",
    "卷积神经网络": "神经网络",
    "循环神经网络": "神经网络",
    "LSTM": "循环神经网络",
    "GRU": "循环神经网络",
    
    # 训练和优化
    "反向传播": "训练算法",
    "梯度下降": "优化算法",
    "AdamW": "优化器",
    "余弦退火": "学习率调度",
    "Layer归一化": "归一化",
    "Batch归一化": "归一化",
    "Dropout": "正则化",
    "掩码": "数据处理",
    "残差连接": "网络结构",
    
    # 激活函数
    "SwiGLU": "激活函数",
    "GELU": "激活函数",
    "ReLU": "激活函数",
    "Sigmoid": "激活函数",
    "Tanh": "激活函数",
    "Softmax": "激活函数",
    
    # 嵌入和编码
    "词嵌入": "文本表示",
    "位置编码": "位置表示",
    "One-Hot编码": "文本表示",
    "Token": "文本单位",
    
    # 多模态融合
    "多模态融合": "融合技术",
    "早期融合": "融合策略",
    "晚期融合": "融合策略",
    "混合融合": "融合策略",
    
    # 生成模型
    "扩散模型": "生成模型",
    "DDPM": "扩散模型",
    "VAE": "生成模型",
    "VQ-VAE": "生成模型",
    "FSQ": "量化技术",
    "GAN": "生成模型",
    "DCGAN": "生成模型",
    "UNet": "图像分割",
    
    # 混合专家
    "混合专家模型": "模型架构",
    "MoE": "混合专家",
    "路由器": "门控网络",
    "门控网络": "MoE组件",
    
    # 其他技术
    "强化学习": "机器学习",
    "PEFT": "微调技术",
    "LoRA": "微调技术",
    "知识蒸馏": "模型压缩",
    "FP8混合精度": "训练优化",
    "双线性插值": "图像处理",
    "空洞卷积": "卷积变体",
    "频谱图": "音频表示",
    "MFCC": "音频特征",
    "librosa": "音频处理库",
    
    # 应用场景
    "情感分析": "NLP应用",
    "机器翻译": "NLP应用",
    "图像识别": "CV应用",
    "语音识别": "语音应用",
    "视频分类": "视频应用",
    "智能客服": "应用系统",
    "自动驾驶": "应用系统",
    "医学诊断": "应用系统",
}

print(f"总共找到 {len(technologies)} 个技术术语")

# 创建输出目录
output_dir = Path("../algorithm_knowledge_base/algorithms")
output_dir.mkdir(parents=True, exist_ok=True)

# 为每个技术生成文档
for tech_name, tech_category in technologies.items():
    # 替换模板中的 <算法名称>
    doc_content = template_content.replace("<算法名称>", tech_name)
    
    # 更新第一行的标题
    doc_content = doc_content.replace(
        "# <算法名称> 学习文档",
        f"# {tech_name} 学习文档"
    )
    
    # 在开头添加分类信息
    header = f"""# {tech_name} 学习文档

> **分类**：{tech_category}  
> **来源**：《DeepSeek大模型高性能核心技术与多模态融合开发》  
> **最后更新**：2026-04-24

---

"""
    
    # 将模板内容附加在header后（保留模板结构）
    doc_content = header + doc_content.split("---", 1)[1] if "---" in doc_content else header + doc_content
    
    # 写入文件
    filename = f"{tech_name}.md"
    # 替换非法文件名字符
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print(f"已生成: {filename}")

print(f"\n完成！共生成 {len(technologies)} 个技术文档")
print(f"输出目录: {output_dir}")
