import os
# 魔法指令：将 Hugging Face 的下载请求全部重定向到国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from agent_engine import run_agent_async

if __name__ == "__main__":
    import asyncio

    async def run_test():
        print("========== 测试 1：单次工具调用 ==========")
        reply1 = await run_agent_async("北京今天天气怎么样？")
        print(f"Agent 回复: {reply1}\n")

        print("========== 测试 2：并发工具调用 ==========")
        reply2  = await run_agent_async("帮我同时查一下上海明天和广州后天的天气，我看看去哪出差好。")
        print(f"Agent 回复: {reply2}\n")

        print("========== 测试 3：单句多意图 + 依赖大模型常识 ==========")
        reply3 = await run_agent_async("我后天要去深圳出差，查一下那边的天气，然后根据天气告诉我该带什么衣服？")
        print(f"Agent 回复: {reply3}\n")


    asyncio.run(run_test())