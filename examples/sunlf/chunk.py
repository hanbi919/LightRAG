from lightrag.utils import TiktokenTokenizer
from lightrag.operate import chunking_by_token_size

# 创建分词器实例
tokenizer = TiktokenTokenizer("gpt-4o-mini")

# 示例文本
text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出决策或预测，
而无需明确编程来执行特定任务。深度学习是机器学习的一个子集，它模仿人脑的工作方式，
使用神经网络来处理和学习数据中的复杂模式。
"""

# 调用chunking_by_token_size函数进行文本分块
chunks = chunking_by_token_size(
    tokenizer=tokenizer,
    content=text,
    split_by_character="\n",  # 首先按换行符分割
    split_by_character_only=False,  # 不仅按字符分割，还要考虑token大小
    overlap_token_size=100,  # 块间重叠100个token
    max_token_size=300  # 每个块最大300个token
)

# 输出分块结果
print(f"总共生成了 {len(chunks)} 个文本块：")
for i, chunk in enumerate(chunks):
    print(f"\n块 {i+1}:")
    print(f"Token数量: {chunk['tokens']}")
    print(f"内容: {chunk['content'][:100]}...")  # 只显示前100个字符
print(f"总Token数量: {sum(chunk['tokens'] for chunk in chunks)}")