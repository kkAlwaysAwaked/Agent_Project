# 单例模式创建Client客户端，防止并发访问导致崩溃
from qdrant_client import QdrantClient

DB_PATH = r"D:\Qdrant_Database"

# 创建全局唯一实例
client = QdrantClient(path=DB_PATH)