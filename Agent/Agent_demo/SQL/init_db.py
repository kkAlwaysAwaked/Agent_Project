import sqlite3

def init_mock_db():
    conn = sqlite3.connect("system_logs.db")
    cursor = conn.cursor()

    # 建表：包含 id, 报错级别, 报错信息, 追踪ID
    cursor.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, level TEXT, message TEXT, trace_id TEXT)")
    # 插入两条测试数据
    cursor.execute(
        "INSERT OR IGNORE INTO logs (id, level, message, trace_id) VALUES (1, 'ERROR', 'OrderService OOM Crash, Gateway 502', 'TRC-9981')")
    cursor.execute(
        "INSERT OR IGNORE INTO logs (id, level, message, trace_id) VALUES (2, 'INFO', 'User login success', 'TRC-1002')")

    conn.commit()
    conn.close()
    print("✅ 测试数据库 system_logs.db 初始化成功！")

if __name__ == "__main__":
    init_mock_db()