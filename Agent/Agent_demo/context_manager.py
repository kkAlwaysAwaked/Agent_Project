# 包含主程序Agent的全局异步context manager + 用于RAG的精简context manager

import copy
import logging
import tiktoken
from openai import AsyncOpenAI
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from fastapi_gateway.config import DEEPSEEK_BASE_URL
import httpx

# 配置日志，生产环境推荐使用 logging 而不是 print
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_tokens(messages: list) -> int:
    """计算列表中的 Token 数量（使用 GPT-4 词表近似）"""
    encoder = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for msg in messages:
        num_tokens += 3
        for key, value in msg.items():
            if not value:
                continue
            content_str = str(value)
            num_tokens += len(encoder.encode(content_str))
    num_tokens += 3
    return num_tokens

# I/O 密集型任务 做异步
# 调用deepseek生成摘要
async def summarize_old_messages(old_messages : list, http_client: httpx.AsyncClient):
    """异步调用 DeepSeek 小模型，将旧对话压缩为摘要"""
    if not old_messages:
        return ""

    history_text = []
    for msg in old_messages :
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        if role == "tool" :
            content = content[:200] + "...[工具数据]"
        history_text += f"{role}: {content}\n"

    client = AsyncOpenAI(api_key = DEEPSEEK_API_KEY,
                         base_url=DEEPSEEK_BASE_URL,
                         http_client=http_client
                         )
    prompt = (
        "你是一个记忆压缩助手。请你概括以下对话历史的核心内容。\n"
        "重点提取：用户的主要需求、已做出的决定、重要的已知信息。\n"
        "请用第三人称、简明扼要的语言描述，不要啰嗦。"
    )

    try :
        response = await client.chat.completions.create(
            model = "deepseek-chat",
            messages = [
                {"role" : "system", "content" : prompt},
                {"role" : "user", "content" : f"历史对话如下：\n{history_text}"}
            ],
            max_tokens = 500,
            temperature = 0.3
        )
        summary = response.choices[0].message.content.strip()
        return f"[系统提示：以下是早期对话的记忆总结]\n{summary}"
    except Exception as e:
        return "[系统提示：早期记忆摘要因网络或接口错误失败，部分旧对话已丢弃]"

# 异步主管理器
async def async_context_trimmer(messages : list, http_client: httpx.AsyncClient, max_tokens : int = 6000, keep_latest_n_turns: int = 3) -> list:
    """
    异步智能截断器：处理滑动窗口并生成摘要
    """
    work_messages = copy.deepcopy(messages)

    current_tokens = count_tokens(work_messages)
    if current_tokens <= max_tokens:
        return work_messages

    logger.warning(f"触发智能截断！当前 Token: {current_tokens} > 限制: {max_tokens}")
    system_prompt = None
    if work_messages and work_messages[0].get("role") == "system":
        system_prompt = work_messages[0]
        work_messages = work_messages[1:]

    # 保证要保留的数量一定不会超过当前的消息总数
    keep_count = min(keep_latest_n_turns * 2, len(work_messages))

    # 从头开始取，取到倒数第x个
    old_messages = work_messages[:-keep_count]
    # 从倒数第x个开始取，取到末尾
    latest_massages = work_messages[-keep_count:]

    summary_text = await summarize_old_messages(old_messages)

    optimized = []
    if system_prompt:
        optimized.append(system_prompt)
    if summary_text:
        optimized.append({"role" : "system", "content" : summary_text})

    optimized.extend(latest_massages)

    while len(optimized) > 2 and count_tokens(optimized) > max_tokens:
        optimized.pop(2)

    return optimized



def simple_context_trimmer(messages : list, max_chars : int = 6000) -> list:
    """按字符数粗略截断，只保护 System Prompt 和最新的对话"""

    # 1. 算出当前总字符数量
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)

    if total_chars <= max_chars:
        return messages
    print(f"[Warn] 触发粗粒度截断！当前字符数: {total_chars} > 限制: {max_chars}")
    # 创建安全列表：最佳的
    optimized = []

    # 2. 永远保留第一条 System Prompt
    if messages and messages[0].get("role") == "system":
        optimized.append(messages[0])
        messages = messages[1:]

    # 3. 倒序处理剩余信息（优先保最新的）
    current_length = 0
    safe_history = []

    for msg in reversed(messages):
        msg_copy = msg.copy()
        msg_len = len(str(msg.get("content", "")))

        # 如果是工具返回的超长结果，直接对其“粗暴斩尾”
        if msg.get("role", "") == "tool" and msg_len > 1000:
            original = msg["content"]
            msg_copy["content"] = original[:500] + "\n...[内容过长，已按字符数截断]..."
            msg_len = len(msg["content"])

        # 如果加上这条消息还没超标，就放进来
        if current_length + msg_len < (max_chars - 1000):
            safe_history.append(msg_copy)
            current_length += msg_len
        # 超标了直接丢弃最早的记录
        else:
            break

    optimized.extend(reversed(safe_history))
    return optimized