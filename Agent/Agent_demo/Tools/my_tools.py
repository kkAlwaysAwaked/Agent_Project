# 注册并存放工具
from .tool_registry import register_tool
from Agent.Agent_demo.RAG_for_FunctionCalling.Search_Internal_Docs import search_internal_docs
from Agent.Agent_demo.context_manager import simple_context_trimmer
import traceback

# 要返回标准的json
@register_tool
async def RAG(query: str, agent_messages: list = None) -> str :
    """
    【核心知识库检索工具】
    当你需要回答关于内部规定、专业文档、特定业务细节等你不确定的事实性问题时，必须调用此工具。
    该工具接入了强大的混合检索引擎，能够为你提供最准确的背景参考资料。
    【强制规范】：在进行系统故障排查时，你必须**先**调用此工具获取排查手册和数据库表结构/SQL模板，**然后再**去查询真实的数据库。
    Args:
        query (str): 根据用户的最新提问和前文语境，提取出的独立、完整的搜索关键词。
                     请务必消除指代不明（例如将“它怎么用”改写为“XX系统功能使用说明”）。
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
            empty_result = {
                "status": "empty",
                "knowledge_content": "知识库中未检索到相关内容，请尝试更换搜索词或如实告诉用户。"
            }
            return json.dumps(empty_result, ensure_ascii=False)

        formatted_result = "以下是为你检索到的参考资料：\n"
        for i, doc in enumerate(results):
            # 不管上游传来的是什么牛鬼蛇神，通通包容！
            content = ""
            source = "未知来源"

            if isinstance(doc, str):
                # 如果传过来的直接是字符串
                content = doc.strip()
            elif isinstance(doc, dict):
                # 如果是字典，兼容 content 或 text 两种叫法
                content = doc.get("content", doc.get("text", "")).strip()

                # 提取来源：针对你的 docstore 结构精准提取
                if "source" in doc:
                    source = doc["source"]
                elif "metadata" in doc:
                    if isinstance(doc["metadata"], dict):
                        source = doc["metadata"].get("source", "未知来源")
                    else:
                        source = str(doc["metadata"])  # 哪怕 metadata 里是个字符串也接得住

            if not content:
                continue  # 如果实在没内容，跳过这一条

            formatted_result += f"【参考资料 {i + 1}】(来源: {source})\n{content}\n---\n"

        # for i, doc in enumerate(results):
        #     content = doc.get("content", "").strip()
        #     source = doc.get("metadata", {}).get("source", "未知来源")
        #     formatted_result += f"【参考资料 {i + 1}】(来源: {source})\n{content}\n---\n"

        # 包装成json字典返回
        result_dict = {
            "status" : "success",
            "doc_count" : len(results),
            "knowledge_content" : formatted_result
        }
        return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        # Debug:
        print("\n" + "="*50)
        traceback.print_exc()
        print("="*50 + "\n")

        error_result = {
            "status": "error",
            "knowledge_content": f"检索工具内部发生异常: {str(e)}"
        }
        return json.dumps(error_result, ensure_ascii=False)

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

import sqlite3
import json
from pathlib import Path
@register_tool
def query_system_logs(sql_query : str) -> str:
    """
        执行 SQL 语句查询系统日志数据库。
        数据库表结构：logs (id INTEGER, level TEXT, message TEXT, trace_id TEXT)
        注意：只能执行 SELECT 查询语句。不要尝试执行修改或删除操作。
    """
    print(f"\n[DB Tool] Agent 尝试执行 SQL: {sql_query}")

    try :
        # 使用绝对路径
        db_file = (Path(__file__).parent.parent / "SQL" / "system_logs.db").resolve()

        # 组装 URI
        db_uri = f"{db_file.as_uri()}?mode=ro"

        # 【核心防御机制】：使用 uri=True 和 mode=ro 强制以“只读模式”连接数据库
        conn = sqlite3.connect(db_uri, uri=True)
        cursor = conn.cursor()

        cursor.execute(sql_query)
        rows = cursor.fetchall()

        # 提取列名，组装成json返回
        column_names = [description[0] for description in cursor.description]

        conn.close()

        result_dict = {"columns": column_names, "data": rows}
        return json.dumps(result_dict, ensure_ascii=False)

    except sqlite3.OperationalError as e:
        error_msg = f"SQL执行失败！请检查语法、表名或是否越权操作 (OperationalError): {str(e)}"
        print(f"[DB Tool] ⚠️ 拦截到错误并返回给大模型: {error_msg}")
        return error_msg

    except Exception as e:
        return f"数据库未知错误: {str(e)}"