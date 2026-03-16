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
            msg["content"] = original[:500] + "\n...[内容过长，已按字符数截断]..."
            msg_len = len(msg["content"])

        # 如果加上这条消息还没超标，就放进来
        if current_length + msg_len < (max_chars - 1000):
            safe_history.append(msg)
            current_length += msg_len
        # 超标了直接丢弃最早的记录
        else:
            break

    optimized.extend(reversed(safe_history))
    return optimized


