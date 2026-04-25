# 第3章 AI量化交易策略及实战案例

![](images/fdd026b624c44ba3c3aac1e29fb3e503d97f61e00ff32dbe5d0a1d65ed556642.jpg)  
图3.2 自动化研报分析与回测流程

```python
def call_llm_api(prompt, text, model="moonshot-v1-128k"): response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是专业的量化因子解析专家。"},
        {"role": "user", "content": prompt + "\n" + text}
    ],
    temperature=0.1,
    max_tokens=2048
) return response['choices'] [0] ['message'] ['content'] 
```
