# 注册并存放工具
from .tool_registry import register_tool
from RAG_for_FunctionCalling.Search_Internal_Docs import search_internal_docs
from context_manager import simple_context_trimmer

@register_tool
async def RAG(query: str, agent_messages: list = None) -> str :
    """
    【核心知识库检索工具】
    当你需要回答关于内部规定、专业文档、特定业务细节等你不确定的事实性问题时，必须调用此工具。
    该工具接入了强大的混合检索引擎，能够为你提供最准确的背景参考资料。

    Args:
        query (str): 根据用户的最新提问和前文语境，提取出的独立、完整的搜索关键词。
                     请务必消除指代不明（例如将“它怎么用”改写为“XX系统功能使用说明”）。，


    """
    print(f"\n[Tool Calling] 正在触发 RAG，核心检索词: '{query}'")

    # 如果刚开始对话，agent_messages 为空，就用 query 构造一个伪装的 message 让底层 RAG 不报错
    raw_messages = agent_messages or [{"role": "user", "content": query}]
    concise_messages = simple_context_trimmer(messages=raw_messages, max_chars=6000)

    # 调用RAG
    try :
        results = await search_internal_docs(messages = concise_messages)
        # 兜底逻辑
        if not results:
            return "知识库中未检索到相关内容，请尝试更换搜索词或如实告诉用户。"
        formatted_result = "以下是为你检索到的参考资料：\n"
        for i, doc in enumerate(results):
            content = doc.get("content", "").strip()
            source = doc.get("metadata", {}).get("source", "未知来源")
            formatted_result += f"【参考资料 {i + 1}】(来源: {source})\n{content}\n---\n"

        return formatted_result

    except Exception as e:
        return f"检索工具内部发生异常: {str(e)}"


import random
@register_tool
async def weather_consulter(location : str, date : str = "今天") -> str :
    """
        【天气查询工具】
        当你需要查询某个城市的天气情况、气温，或者需要根据天气给出出行/穿搭建议时，必须调用此工具。

        Args:
            location (str): 必须是标准的城市名称，例如 "北京", "上海", "广州"。
            date (str): 查询的日期，例如 "今天", "明天", "本周五" 等。默认为 "今天"。
    """
    print(f"\n[Tool Calling ☁️] 触发天气查询 -> 城市: '{location}', 日期: '{date}'")
    weather_conditions = ["晴朗 ☀️", "多云 ⛅", "小雨 🌧️", "雷阵雨 ⛈️", "微风 🍃"]
    condition = random.choice(weather_conditions)
    low_temp = random.randint(10, 20)
    high_temp = random.randint(22, 35)
    result = f"{location}{date}的天气预计为 {condition}，气温在 {low_temp}°C 到 {high_temp}°C 之间。"
    print(f"[Tool Calling ☁️] 查询完成，返回给大模型: {result}")
    return result
