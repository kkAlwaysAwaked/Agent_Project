import time
import json
import logging
import httpx
from typing import List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from Agent.Agent_demo.context_manager import async_context_trimmer
from Agent.Agent_demo.RAG_for_FunctionCalling.Query_and_HyDE import DEEPSEEK_API_KEY
from Agent.Agent_demo.agent_engine import run_agent_async

# 记录日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_Monitor")

# ==========================================
# 1. 生命周期与连接池
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：建专线
    app.state.http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=100))
    logger.info("全局 HTTPX 客户端已挂载")
    yield
    # 关闭：拆专线
    await app.state.http_client.aclose()
    logger.info("全局 HTTPX 客户端已销毁")

# ==========================================
# 2. 实例化 App 与中间件
# ==========================================
app = FastAPI(title="AI Agent Gateway", lifespan=lifespan)

# 给FastAPI 加一个插件（中间件），实现与前端的通信
# 如果你的前端网页在 网站A，后端代码在 网站B，
# 浏览器默认会阻止它们交流。这个 CORSMiddleware 就是发通行证的。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 允许所有网址前端访问
    allow_credentials=True,
    allow_methods=["*"],    # 允许各种类型的请求方式(GET, POST)和各种请求头
    allow_headers=["*"],
)

# 所有 HTTP 请求都要经过 log_request_time
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)  # 这里放行请求，去走后面的逻辑
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 3,4 实现对进入的包裹进行查验

# ==========================================
# 3. 数据校验模型
# # ==========================================
class ChatRequest(BaseModel):
    # 规定前端传来的 JSON 必须有 prompt，且长度在 1-2000 之间
    # prompt: str = Field(..., min_length=1, max_length=2000)

    # 修改：接收完整的OpenAI格式的 messages数组
    messages : List[Dict[str, str]] = Field(..., description="完整的对话历史列表")
    model: str = "deepseek-chat"

# ==========================================
# 4. 安全鉴权逻辑
# ==========================================
async def verify_token(x_token: str = Header(..., description="只放行带了合法 Token 的请求")):
    # 检查前端请求头里有没有暗号
    if x_token != "WhyNotMe":
        raise HTTPException(status_code=403, detail="Token错误，拒绝访问")
    return x_token


# # ==========================================
# # 5. 核心业务逻辑
# # ==========================================
# async def deepseek_stream_generator(prompt: str, model_name: str, client: httpx.AsyncClient):
#     headers = {
#         "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {"model": model_name,
#                "messages": [{"role": "user", "content": prompt}],
#                "stream": True}
#
#     try:
#         async with client.stream("POST",
#                                  DEEPSEEK_BASE_URL,
#                                  headers=headers,
#                                  json=payload,
#                                  timeout=30.0) as response:
#
#             # SSE逻辑: 接收，迭代DeepSeek传来的流式数据
#             # data: {"id":"123", "choices":[{"delta":{"content":"你"}}], ...}
#             # 实现数据清洗
#             async for line in response.aiter_lines():
#                 if line.startswith("data: "):
#                     data_str = line[6:].strip()
#                     if data_str == "[DONE]":
#                         yield "data:[DONE]\n\n"
#                         break
#                     try:
#                         # 把传回来的json变成python字典，并实现提取逻辑
#                         data_json = json.loads(data_str)
#                         if "content" in data_json["choices"][0].get("delta", {}):
#                             yield f"data:{data_json['choices'][0]['delta']['content']}\n\n"
#                     except json.JSONDecodeError:
#                         continue
#     except Exception as e:
#         yield f"data:[系统异常 {str(e)}]\n\n"


# ==========================================
# 6. 路由入口
# ==========================================
@app.post("/v1/chat")
async def chat_endpoint(
        request_data: ChatRequest,  # 3. 拦截不符合格式的提问
        req: Request,  # 拿到底层请求对象，为了提取连接池
        token: str = Depends(verify_token)  # 触发步骤 4. 查验前端有没有传正确的密码头
):

    # 拿到全局连接池
    http_client = req.app.state.http_client

    try :
        optimized_messages = await async_context_trimmer(
            messages = request_data.messages,
            http_client=http_client,
            max_tokens=6000,
            keep_latest_n_turns=3
        )
    # 把“提问”、“模型名”和“专线”一起塞进agent_engine，并告诉 FastAPI 用流式推给前端
        return StreamingResponse(
            run_agent_async(optimized_messages, http_client),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"网关处理异常: {e}")
        raise HTTPException(status_code=500, detail="服务器内部处理异常")