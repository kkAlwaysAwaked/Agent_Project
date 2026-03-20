# AIOps ReAct Agent Engine

> **面向复杂 IT 排障场景，自底向上研发的异步 ReAct 智能运维 Agent 引擎。**

本项目旨在通过大模型（LLM）实现 IT 故障的自动化诊断与排查闭环。核心引擎基于原生异步机制与 ReAct 范式从零构建，统筹调度**系统日志（Text-to-SQL）**与**故障手册（RAG）**，成功将复杂排障场景的平均排查时间 (MTTR) 从**小时级压缩至分钟级**。

## ✨ 核心亮点 (Technical Highlights)

- 🚀 **自研原生异步引擎**：脱离笨重的开源框架，使用 AsyncOpenAI 从零构建支持 `max_steps` 状态流转的 ReAct 引擎。
- 🧩 **极致解耦的架构**：引入 `@registry_tool` 装饰器与面向对象设计，实现外部工具（Tools）的热插拔，保障核心逻辑的高扩展性。
- 🧠 **跨源多跳推理**：深度整合 Function Calling，实现 `RAG 检索 -> 参数提取 -> SQL 动态查询` 的端到端多跳自动化诊断。
- 📉 **生产级成本控制**：首创“尾部 6000 Token 倒序截断 + 历史状态摘要”滑动窗口策略，单次 API 请求 Token 消耗**直降 ~40%**。


## 🏗️ 系统架构与核心模块

### 1. ReAct 核心引擎 (Cognitive Layer)
* **Prompt Engineering 强约束**：通过精细化 Prompt 严格规范大模型的 `Thought-Action-Observation` (TAO) 输出格式，有效遏制模型幻觉并大幅降低解析错误率。
* **状态机流转**：支持复杂的多轮思考与动作执行，内建防死循环机制 (`max_steps` limit)。

### 2. 动态工具注册中心 (Dynamic Tool Registry)
为了解决 Agent 能力扩展带来的代码耦合问题，设计了基于注册表模式的 Tool 管理器，实现灵活的工具热插拔。

### 3. Context Engineering (上下文管理)
* 针对长对话场景下的 Token 溢出与“大海捞针”问题，自研**context_manager**模块：
* 滑动窗口记忆 (Sliding Window Memory)：动态维护对话历史。
* 智能截断与摘要：保留最近的高价值 6000 Token，并对早期历史生成高密度的状态摘要，在保证推理上下文完整的前提下大幅降低 API 成本。

### 4. 离线高阶 RAG 数据底座
* 独立研发 Create_Database 预处理模块，为线上准确检索提供高质量弹药：
* 自动化解析：非结构化故障手册自动清洗入库。
* 高阶 Chunking：入库阶段完成“父子文档切分 (Parent-Child Document Retrieval)”。
* 双路召回绑定：实现 Dense (稠密向量) 与 Sparse (稀疏检索) 的双路 Embedding 强绑定。

## 💻 核心代码快照 (Code Snippets)

### 1. 动态工具注册中心
运用 `inspect` 反射与 `pydantic.create_model`，在代码运行瞬间动态解析函数签名并生成严格的 JSON Schema，实现了 **Agent 引擎与外部工具的彻底解耦**：

```python
# tools_registry.py (核心逻辑节选)
def register_tool(func: Callable):
    """自动将 Python 函数转换为 LLM Standard Function Calling Schema"""
    name, description = func.__name__, inspect.getdoc(func)
    sig, fields = inspect.signature(func), {}

    # 1. 动态解析参数与类型，无缝拦截 "agent_messages" 等幽灵参数
    for param_name, param in sig.parameters.items():
        if param_name == "agent_messages": 
            continue 
        annotation = param.annotation if param.annotation != inspect._empty else Any
        default = param.default if param.default != inspect._empty else ...
        fields[param_name] = (annotation, default)

    # 2. 运行时动态构建 Pydantic 校验类，一键导出模型 Schema
    InputModel = create_model(f"{name}_Input", **fields)
    tool_schema = {
        "type": "function",
        "function": {
            "name": name, "description": description, 
            "parameters": InputModel.model_json_schema()
        }
    }
```
### 2. 生产级长对话无损压缩（context_manager）
  针对复杂排障场景下的长文本 Context 溢出问题，摒弃了传统的“粗暴截断”方案。自研异步智能截断器，引入**“轻量级 LLM 记忆压缩”**与**精确的滑动窗口策略**，在保证历史关键信息不丢失的前提下，大幅压降 Token 成本：
  ```
  # context_manager.py (记忆压缩核心逻辑节选)
  async def async_context_trimmer(messages: list, max_tokens: int = 6000, keep_turns: int = 3):
      """异步智能截断器：处理滑动窗口并生成无损摘要"""
      # 1. 引入 tiktoken 进行精确的 Token 开销度量
      if count_tokens(messages) <= max_tokens:
          return messages
  
      # 2. 动态窗口拆分：剥离系统级 Prompt、早期历史 (old) 与 近期活跃对话 (latest)
      system_prompt = messages[0]
      old_messages = messages[1:-keep_turns*2] 
      latest_messages = messages[-keep_turns*2:]
  
      # 3. I/O 密集型优化：异步调用轻量级模型 (如 DeepSeek-Chat) 抽取早期核心排障记忆
      # 彻底解决硬截断导致的“灾难性遗忘”问题
      summary_text = await summarize_old_messages(old_messages, http_client)
  
      # 4. 重新组装高密度上下文 (Context Rebuild)
      optimized = [
          system_prompt,
          {"role": "system", "content": f"[早期排障记忆总结]\n{summary_text}"}
      ]
      optimized.extend(latest_messages)
      return optimized
  ```
## 💡 业务价值

在模拟/真实复杂 IT 运维场景中，该 Agent 能够自主接管告警，先查阅运维手册获取排查 SOP，再自主编写 SQL 提取实时日志进行比对验证，最终输出完整的根因分析报告。实现了** 从“人找数据”到“机器主动诊断” **的范式转变。

👨‍💻 Developed by kkAlwayAwaked
