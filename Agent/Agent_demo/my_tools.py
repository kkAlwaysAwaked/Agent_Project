from tool_registry import register_tool
from RAG_for_FunctionCalling.Search_Internal_Docs import search_internal_docs
from typing import List, Dict, Any
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
