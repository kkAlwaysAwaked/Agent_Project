# 实现了 Rewrite 和 HyDE 逻辑
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configs
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek 的官方接口地址

# Query Rewrite
async def rewrite_query(chat_history : list, latest_query : str):
    system_prompt = """
    你是一个无情感的【搜索查询重写机器】。
    任务：根据历史对话，将用户的最新提问重写为独立的、客观的搜索关键词短语。
    
    【最高级别警告】：
    1. 严禁使用完整句子输出！
    2. 严禁包含任何第一人称（如：我、你、让我）。
    3. 严禁包含任何过渡性废话（如：好的、尝试查询、用户想知道...）。
    只允许输出名词实体！
    
    示例输入：帮我查查是不是OOM
    示例输出：OOM 报错 排查 诊断
    """

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    user_content = f"历史对话：\n{history_str}\n\n最新提问：{latest_query}"

    # http请求头
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # http请求体
    payload = {
        "model" : "deepseek-chat",
        "messages" : [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : user_content}
        ],
        "temperature" : 0.0
    }

    async with httpx.AsyncClient() as client:
        try :
            response = await client.post(
                DEEPSEEK_API_URL,
                headers = headers,
                json = payload,
                timeout = 10.0
            )
            response.raise_for_status()
            # 解析返回的json数据
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except httpx.HTTPError as e:
            print("网络请求失败！")

# HyDE
async def generate_hyde_document(query : str) -> str:
    system_prompt = """
    你是一个底层搜索引擎的【核心关键词提取器】。
    你的唯一任务是：根据用户的历史对话上下文，提取或改写出最适合进行向量数据库检索的 1 到 2 个核心实体名词。
    
    【绝对禁令】：
    1. 绝对禁止输出任何第一人称代词（如：我、你、让我）。
    2. 绝对禁止输出任何解释性、过渡性、祈使语气的废话（如：好的、我来尝试查询、用户需要排查...）。
    3. 只能输出关键词本身，多个关键词用空格隔开。
    
    正确示例："OOM 报错 日志排查"
    错误示例："让我来帮您查询 OOM 报错的日志信息"
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.1
    }

    async with httpx.AsyncClient() as client :
        try :
            response = await client.post(
                DEEPSEEK_API_URL,
                headers = headers,
                json = payload,
                timeout = 15.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except httpx.HTTPError as e:
            print(f"HyDE 生成失败: {e}")
            return ""   # 如果失败，返回空字符串，后续检索时直接退化为只搜原始 Query


async def generate_hyde_vector(messages: list):
    """
    从标准 messages 列表中提取历史对话和最新提问，并调用 Query Rewrite 和 HyDE。
    返回用于后续向量检索的关键文本。
    """
    if not messages:
        return None, None

    # 1. 提取最新提问和历史对话
    latest_query = messages[-1].get("content", "")
    chat_history = messages[:-1]

    # 2. 核心步骤一：执行 Query 改写
    rewritten_query = await rewrite_query(chat_history=chat_history, latest_query=latest_query)

    # 如果改写失败（比如网络波动），降级使用原始的 latest_query
    if not rewritten_query:
        rewritten_query = latest_query

    # 3. 核心步骤二：基于【改写后的独立问题】生成假设性文档 (HyDE)
    hyde_doc = await generate_hyde_document(query=rewritten_query)

    # 4. 返回结果供下游 Embedding 和检索使用
    # ***返回格式建议为一个字典，方便后续扩展
    return {
        "rewritten_query": rewritten_query, # 改写后的query
        "hyde_document": hyde_doc   # 生成的假设性文档
    }







