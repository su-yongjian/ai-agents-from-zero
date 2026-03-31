"""
【案例】在 Redis 向量库中做相似性检索（similarity_search_with_score）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 6.3 案例：连接已有索引，做相似性检索

知识点速览：
- 相似性检索的核心流程是：查询文本先向量化，再到向量库中找到与查询向量最接近的若干条记录。
- similarity_search_with_score(query, k) 返回 (Document, score) 列表；很多实现里 score 更接近“距离”，通常越小越相似。
- 代码里把 score 换算成 1 - score，主要是为了更符合初学者直觉；真实项目里应以具体向量库和距离度量定义为准。
- 运行前需确保 Redis 中已有数据，例如先执行同目录下的 RedisVectorStore.py；index_name、redis_url 也必须保持一致。
"""

from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 嵌入模型（与写入时一致，保证向量空间一致）；需在 .env 中配置 aliQwen-api
embeddingsModel = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)

# 2. 连接已有索引（与 RedisVectorStore.py 中 index_name、redis_url 一致）
vector_store = RedisVectorStore(
    embeddingsModel,
    config=RedisConfig(index_name="newsgroups", redis_url="redis://localhost:26379"),
)

# 3. 查询文本 → 向量化 → 在库中做相似度检索；这里取前 3 条结果
query = "我喜欢用什么手机"
results = vector_store.similarity_search_with_score(query, k=3)

print("=== 查询结果 ===")
for i, (doc, score) in enumerate(results, 1):
    # 这里把“距离”近似换算成“相似度”只是为了展示更直观；工程里请以具体返回定义为准
    similarity = 1 - score
    print(f"结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print(f"相似度: {similarity:.4f}")
