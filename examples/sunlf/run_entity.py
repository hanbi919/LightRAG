from lightrag.utils import EmbeddingFunc
from lightrag.lightrag import LightRAG
from lightrag.operate import extract_entities
from lightrag.utils import TiktokenTokenizer
import asyncio
import json
import numpy as np
import os
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="http://139.210.101.45:12455/v1/",
        **kwargs
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="http://139.210.101.45:12456",

    )
# 初始化 LightRAG 实例


async def initialize_rag():
    rag = LightRAG(
        working_dir="./example_rag_storage",
        llm_model_func=llm_model_func,  # 需要提供实际的 LLM 函数
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func
        )  # 需要提供实际的嵌入函数
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# 示例文本内容
sample_text = """
苹果公司（Apple Inc.）是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。
史蒂夫·乔布斯是苹果公司的联合创始人之一，他在1976年与史蒂夫·沃兹尼亚克和罗纳德·韦恩共同创立了该公司。
苹果公司开发了多款知名产品，包括iPhone智能手机、iPad平板电脑和Mac个人电脑。
蒂姆·库克目前担任苹果公司的首席执行官。
"""


# 抽取实体和关系


async def extract_example():
    try:
        rag = await initialize_rag()
        await rag.ainsert(sample_text)
        # 创建分词器
        # tokenizer = TiktokenTokenizer("gpt-4o-mini")

        # # 将文本分块
        # chunks = rag.chunking_func(
        #     tokenizer=tokenizer,
        #     content=sample_text,
        #     split_by_character=None,
        #     split_by_character_only=False,
        #     overlap_token_size=100,
        #     max_token_size=500
        # )

        # # 为每个块添加必要的元数据
        # processed_chunks = {}
        # for i, chunk in enumerate(chunks):
        #     chunk_id = f"chunk-{i}"
        #     processed_chunks[chunk_id] = {
        #         "content": chunk["content"],
        #         "tokens": chunk["tokens"],
        #         "chunk_order_index": i,
        #         "full_doc_id": "doc-example",
        #         "file_path": "example.txt"
        #     }

        # # 使用 extract_entities 函数抽取实体和关系
        # results = await extract_entities(
        #     chunks=processed_chunks,
        #     global_config=rag.__dict__,  # 传递 RAG 实例的配置
        #     llm_response_cache=rag.llm_response_cache,
        #     text_chunks_storage=rag.text_chunks
        # )

        # # 打印结果
        # print("抽取到的实体和关系:")
        # for result in results:
        #     # print(result)
        #     print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"实体抽取过程中出错: {e}")

# 运行异步函数
asyncio.run(extract_example())
