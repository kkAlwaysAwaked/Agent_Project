import json
import asyncio
import inspect
from openai import AsyncOpenAI
from Tools.tool_registry import TOOL_REGISTRY
from Tools import my_tools # 注册工具
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 初始化客户端
# 接入 Deepseek API
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
MODEL_NAME = "deepseek-chat"

# 编写单个函数的异步执行包装器
async def safe_execute_tool(tool_call, current_messages : list) -> dict:
    """安全地执行单个工具，并返回标准化格式"""
    function_name = tool_call.function.name

    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        function_args = {}

    print(f"[Async] 开始执行工具: {function_name}, 参数: {function_args}")

    # 执行工具逻辑
    if function_name in TOOL_REGISTRY:
        try:
            function_args["agent_messages"] = current_messages

            # 1. 先执行装饰器的包装层（此时只是进行 Pydantic 校验，瞬间完成）
            exec_result = TOOL_REGISTRY[function_name]["execute"](**function_args)

            # ✅ 修复 2：智能判断底层函数是同步还是异步
            if inspect.iscoroutine(exec_result):
                # 如果是 async def 定义的工具，直接 await 等待结果
                tool_result = await exec_result
            else:
                # 如果是普通 def 定义的工具，第一步其实已经执行完了
                tool_result = exec_result

        except Exception as e:
            tool_result = f"工具执行时发生未捕获异常: {str(e)}"
    else:
        tool_result = f"Error: 找不到名为 {function_name} 的工具。"

    print(f"[Async] 工具 {function_name} 执行完毕")

    return {
        "role" : "tool",
        "tool_call_id" : tool_call.id,
        "name" : function_name,
        "content" : str(tool_result)
    }

async def run_agent_async(user_query : str, max_steps : int = 5) -> str:
    # 载入工具 + 上下文
    available_tools = [tool_info["schema"] for tool_info in TOOL_REGISTRY.values()]
    print(f"[Debug] 当前已注册的工具列表: {list(TOOL_REGISTRY.keys())}")
    messages = [
        {"role" : "system", "content" : "你是一个强大的AI助手，请尽可能并发地使用工具以提高效率。"},
        {"role" : "user", "content" : user_query}
    ]

    step = 0
    while step < max_steps:
        step += 1
        print(f"\n--- [Agent 思考中... 第 {step} 轮] ---")

        # 使用 await 调用异步 API
        response = await client.chat.completions.create(
            model = MODEL_NAME,
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message.model_dump(exclude_none=True))

        if response_message.tool_calls:
            print(f"-> 模型发起并发调用，共 {len(response_message.tool_calls)} 个工具")

            tasks = [safe_execute_tool(tc, current_messages=messages) for tc in response_message.tool_calls]

            results = await asyncio.gather(*tasks, return_exceptions = True)

            for i, res in enumerate(results) :
                if isinstance(res, Exception):
                    # 发生系统级并发异常时的兜底反馈
                    messages.append({
                        "role": "tool",
                        "tool_call_id": response_message.tool_calls[i].id,
                        "name": response_message.tool_calls[i].function.name,
                        "content": f"系统级并发异常: {str(res)}"
                    })
                else:
                    messages.append(res)

            continue
        else :
            print("\n=== Agent 最终回答 ===")
            return response_message.content
    return f"触发系统保护熔断：Agent 已达到最大执行步数限制 ({max_steps} steps)。"




# ----------旧版----------
# def run_agent(user_query : str, max_steps: int = 5) -> str:
#     """Agent 核心状态机"""
#     # 1. 提取所有注册好的工具 Schema 传给大模型
#     # 将所有的工具的schema提取成一个列表
#     available_tools = [tool_info["schema"] for tool_info in TOOL_REGISTRY.values()]
#
#     # 2. 初始化上下文消息列表
#     messages = [
#         {"role" : "system", "content": "你是一个强大的AI助手，请合理使用工具解决用户问题。"},
#         {"role": "user", "content": user_query}
#     ]
#
#     # 3. 启动状态机循环
#     step = 0
#     while step < max_steps:
#         step += 1
#         print(f"\n--- [Agent 思考中... 第 {step} 轮] ---")
#
#         # 状态A：向大模型发起请求
#         response = client.chat.completions.create(
#             model = MODEL_NAME,
#             messages = messages,
#             tools = available_tools,
#             tool_choice = "auto"
#         )
#
#         response_message = response.choices[0].message
#
#         messages.append(response_message.model_dump(exclude_none=True))
#
#         # 判断大模型的意图：是否调用了工具？
#         if response_message.tool_calls:
#             # 状态B：大模型决定调用工具，进入本地执行阶段
#             for tool_call in response_message.tool_calls:
#                 function_name = tool_call.function.name
#                 # 解析模型传过来的 JSON 字符串参数
#                 try:
#                     function_args = json.loads(tool_call.function.arguments)
#                 except json.JSONDecodeError:
#                     function_args = {}
#                 print(f"-> 模型请求调用工具: {function_name}, 参数: {function_args}")
#
#                 # 调用 Day 1 封装好的安全执行函数 (自带 Pydantic 校验和异常捕获)
#                 # Ps：TOOL_REGISTRY字段：
#                 #       TOOL_REGISTRY[name] = {
#                 #         "schema" : tool_schema,
#                 #         "execute" : execute_wrapper（真正的执行函数）
#                 #       }
#                 if function_name in TOOL_REGISTRY:
#                     tool_result = TOOL_REGISTRY[function_name]["execute"](**function_args)
#                 else :
#                     tool_result = f"Error: 找不到名为 {function_name} 的工具。"
#
#                 print(f"<- 工具执行结果: {tool_result}")
#
#                 # 关键规范：必须将工具结果封装为 role="tool" 的消息返回给模型
#                 messages.append({
#                     "role" : "tool",
#                     "tool_call_id" : tool_call.id,
#                     "name" : function_name,
#                     "content": str(tool_result)
#                 })
#
#                 continue
#
#         else:
#             # 模型没有调用工具，而是直接输出了文本，说明任务完成
#             print("\n=== Agent 最终回答 ===")
#             return response_message.content





