# （4）大模型解析并输出JSON

Python   
import json   
def parse_factor_json(llm_output): try: factors $=$ json.loadss(llm_output) return factors except Exception as e: print("解析失败："，e) return None
