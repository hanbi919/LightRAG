### LightRAG 支持多种检索模式：

1. naive：传统的向量相似度检索
2. local：利用局部关键字进行检索，关注实体细节信息
3. global：利用全局关键字进行检索，关注实体关系和全局知识
4. hybrid：结合了 local 和 global 模式，检索实体和关系信息
5. mix：结合了 hybrid和 naive模式
6. bypass：不进行任何检索，直接由LLM生成答案