import asyncio
import aiohttp
import time
from statistics import mean

# ==================== 配置 ====================
URL = "http://192.168.0.224:8366/v1"  # vllm 接口地址
MODEL = "xiaoyan"                     # 模型名称
CONCURRENT_REQUESTS = 20              # 并发数
TOTAL_REQUESTS = 100                  # 总请求数
MAX_TOKENS = 50                       # 每次生成的最大 token 数量
# ============================================

# 用于保存每个请求的耗时
request_times = []
success_count = 0
failure_count = 0

# 构造请求 payload
def make_payload(prompt):
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS
    }

# 单个请求任务
async def send_request(session, idx):
    global success_count, failure_count
    prompt = f"压力测试请求 {idx}"
    payload = make_payload(prompt)
    start = time.time()
    try:
        async with session.post(URL, json=payload, timeout=60) as resp:
            data = await resp.json()
            elapsed = time.time() - start
            request_times.append(elapsed)
            success_count += 1
            print(f"[{idx}] 成功, 耗时: {elapsed:.2f}s, 输出: {data}")
    except Exception as e:
        elapsed = time.time() - start
        request_times.append(elapsed)
        failure_count += 1
        print(f"[{idx}] 失败: {e}, 耗时: {elapsed:.2f}s")

# 并发执行请求
async def run_test():
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, i) for i in range(1, TOTAL_REQUESTS + 1)]
        await asyncio.gather(*tasks)

# 输出统计结果
def print_stats():
    if request_times:
        print("\n========= 压测统计 =========")
        print(f"总请求数: {TOTAL_REQUESTS}")
        print(f"成功请求数: {success_count}")
        print(f"失败请求数: {failure_count}")
        print(f"成功率: {success_count / TOTAL_REQUESTS * 100:.2f}%")
        print(f"平均响应时间: {mean(request_times):.2f}s")
        print(f"最大响应时间: {max(request_times):.2f}s")
        print(f"最小响应时间: {min(request_times):.2f}s")
        print("============================\n")

# 主函数
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(run_test())
    total_time = time.time() - start_time
    print_stats()
    print(f"总耗时: {total_time:.2f}s")