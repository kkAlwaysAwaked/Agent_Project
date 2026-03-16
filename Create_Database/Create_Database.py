# 对文件进行切分 + 入库

import os
import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 一、全局初始化逻辑
print("初始化模型和数据库...")

DB_PATH = r"D:\Qdrant_Database"
DOCSTORE_PATH = os.path.join(DB_PATH, "docstore.json")
COLLECTION_NAME = "hybrid_collection"
# 确保数据库主目录存在
os.makedirs(DB_PATH, exist_ok=True)

# 1. 创建Qdrant客户端
client = QdrantClient(path = DB_PATH)

# 2. 初始化 FastEmbed 模型
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")

# 3. 检查集合是否存在，不存在才创建 (极其重要)
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense_vector": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse_vector": models.SparseVectorParams()}
    )
    print(f"✅ 创建了新的 Qdrant 集合: {COLLECTION_NAME}")
else:
    print(f"ℹ️ 集合 {COLLECTION_NAME} 已存在，准备追加数据。")

# 4. 父文档存储：python字典
# 格式：{"uuid-xxxx": "完整的 Markdown 文本"}
if os.path.exists(DOCSTORE_PATH):
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    print(f"ℹ️ 已加载本地 Docstore，当前包含 {len(docstore)} 个父文档。")
else:
    docstore = {}

# 二、单文本切分和处理逻辑
def ingest_data(markdown_text, source_name="unknown"):
    print(f"\n📦 正在处理文件: {source_name}")

    # 1. 父文档切分
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    parent_docs = markdown_splitter.split_text(markdown_text)

    # 2. 子文档切分：按句号切
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["。", "！", "？", ".", "!", "?"],
        chunk_size=10,
        chunk_overlap=0
    )
    # 暂存子文档 Qdrant 数据点
    points = []

    for p_doc in parent_docs:
        parent_id = str(uuid.uuid4())
        # 存入父文档字典：{"parent_id"："原文"}
        docstore[parent_id] = {
            "source": source_name,
            "content": p_doc.page_content
        }
        # 准备子文档
        child_chunks = child_splitter.split_text(p_doc.page_content)

        for chunk in child_chunks:
            if not chunk.strip(): continue

            # 生成稠密向量和稀疏向量
            dense_vec = list(dense_model.embed([chunk]))[0].tolist()
            sparse_vec = list(sparse_model.embed([chunk]))[0]

            # 组装 Qdrant Point
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),  # 子文档自己的 ID
                vector={
                    "dense_vector": dense_vec,
                    "sparse_vector": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={"parent_id": parent_id, "text": chunk}  # text存入可选，为了调试方便可以留着
            ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  -> ✅ 入库成功！提取了 {len(parent_docs)} 个父文档，{len(points)} 个子句子。")
    else:
        print(f"  -> ⚠️ 文件无有效文本内容。")


# 三、批量读取文件并落盘
def build_database(folder_path):
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        print(f"❌ 找不到原始数据文件夹: {folder_path}")
        return

    file_count = 0
    # 遍历文件夹下的所有 md 和 txt 文件
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 调用处理逻辑
            ingest_data(content, source_name=file_path.name)
            file_count += 1

    # 【核心！】所有文件处理完毕后，把字典保存到 D 盘的 json 文件里
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 数据库构建完毕！共处理了 {file_count} 个文件。")
    print(f"💾 Qdrant 向量数据位于: {DB_PATH}")
    print(f"💾 父文档原文数据位于: {DOCSTORE_PATH}")


# 四、执行入口
if __name__ == "__main__":
    # 在这里填入你存放原始 Markdown/TXT 文本的文件夹路径
    SOURCE_DATA_FOLDER = r"D:\My_Raw_Data"

    # 确保文件夹存在，如果没有则自动创建一个空的
    os.makedirs(SOURCE_DATA_FOLDER, exist_ok=True)

    print(f"请确保你的文本文件已经放在了 {SOURCE_DATA_FOLDER} 下面。")
    build_database(SOURCE_DATA_FOLDER)

