# 1. 准备文档数据
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

docs = [
    "黑神话悟空的战斗如同武侠小说活过来一般，当金箍棒与妖魔碰撞时，火星四溅，招式行云流水。悟空可随心切换狂猛或灵动的战斗风格，一棒横扫千军，或是腾挪如蝴蝶戏花。",    
    "72变神通不只是变化形态，更是开启新世界的钥匙。化身飞鼠可以潜入妖魔巢穴打探军情，变作金鱼能够探索深海遗迹的秘密，每一种变化都是一段独特的冒险。",    
    "每场BOSS战都是一场惊心动魄的较量。或是与身躯庞大的九头蟒激战于瀑布之巅，或是在雷电交织的云海中与雷公电母比拼法术，招招险象环生。",    
    "驾着筋斗云翱翔在这片神话世界，瑰丽的场景令人屏息。云雾缭绕的仙山若隐若现，古老的妖兽巢穴中藏着千年宝物，月光下的古寺钟声回荡在山谷。",    
    "这不是你熟悉的西游记。当悟空踏上寻找身世之谜的旅程，他将遇见各路神仙妖魔。有的是旧识，如同样桀骜不驯的哪吒；有的是劲敌，如手持三尖两刃刀的二郎神。",    
    "作为齐天大圣，悟空的神通不止于金箍棒。火眼金睛可洞察妖魔真身，一个筋斗便是十万八千里。而这些能力还可以通过收集天外陨铁、悟道石等材料来强化升级。",    
    "世界的每个角落都藏着故事。你可能在山洞中发现上古大能的遗迹，云端天宫里寻得昔日天兵的宝库，或是在凡间集市偶遇卖人参果的狐妖。",    
    "故事发生在大唐之前的蛮荒世界，那时天庭还未定鼎三界，各路妖王割据称雄。这是一个神魔混战、群雄逐鹿的动荡年代，也是悟空寻找真相的起点。",    
    "游戏的音乐如同一首跨越千年的史诗。古琴与管弦交织出战斗的激昂，笛萧与木鱼谱写禅意空灵。而当悟空踏入重要场景时，古风配乐更是让人仿佛穿越回那个神话的年代。"
    ] 

# 2. 设置嵌入模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)
print(f"文档向量维度: {doc_embeddings.shape}")

# 3. 创建向量存储
import faiss # pip install faiss-cpu
import numpy as np
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"向量数据库中的文档数量: {index.ntotal}")

# 4. 执行相似度检索
question = "黑神话悟空的战斗系统有什么特点?"
query_embedding = model.encode([question])[0]
distances, indices = index.search(
    np.array([query_embedding]).astype('float32'), 
    k=3
)
context = [docs[idx] for idx in indices[0]]
print("\n检索到的相关文档:")
for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc}")

# 5. 构建提示词
prompt = f"""根据以下参考信息回答问题，并给出信息源编号。
如果无法从参考信息中找到答案，请说明无法回答。
参考信息:
{chr(10).join(f"[{i+1}] {doc}" for i, doc in enumerate(context))}
问题: {question}
答案:"""

# 6. 使用DeepSeek生成答案
from openai import OpenAI
import time
import tiktoken

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 计算生成答案的 tokens 数量
try:
    # 尝试使用自动映射
    encoding = tiktoken.encoding_for_model(os.getenv("OLLAMA_MODEL"))
except KeyError:
    # 如果自动映射失败，手动指定编码
    print("无法自动映射模型到分词器，尝试使用通用编码...")
    # 你可以根据实际情况选择合适的编码，这里以 cl100k_base 为例
    encoding = tiktoken.get_encoding("cl100k_base")

# 计算输入提示词的 tokens 数量
input_tokens = len(encoding.encode(prompt))

# 记录开始时间
start_time = time.time()

response = client.chat.completions.create(
    model="deepseek-chat",  
    messages=[{
        "role": "user",
        "content": prompt
    }],
    max_tokens=1024
)

# 记录结束时间
end_time = time.time()

# 计算输出答案的 tokens 数量
output_tokens = len(encoding.encode(response.choices[0].message.content))

# 计算并打印生成耗时
response_time = end_time - start_time

print(f"\n生成的答案: {response.choices[0].message.content}")
print(f"响应时间: {response_time:.2f} 秒")
print(f"输入提示词的 tokens 数量: {input_tokens}")
print(f"输出答案的 tokens 数量: {output_tokens}")
print(f"总 tokens 计算量: {input_tokens + output_tokens}")
