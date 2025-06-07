# Hands - on - RAG：检索增强生成（RAG）教学项目

## 项目概述
本项目旨在通过实际代码示例，为学习者提供一个全面的检索增强生成（RAG）技术教学。我们将使用不同的大语言模型（Ollama和DeepSeek），围绕自建的相关文档，详细展示RAG从数据准备到答案生成的完整流程。通过这个项目，学习者可以深入理解RAG的工作原理，并掌握如何在实际应用中运用这一技术。

## 项目结构
```plaintext
Hands - on - RAG/
├── Hands - on - RAG - Ollama.py
├── Hands - on - RAG - DeepSeek.py
└── README.md
```

### 代码文件说明
- **Hands - on - RAG - Ollama.py**：利用Ollama模型实现RAG流程，解决关于《黑神话：悟空》的问题。
- **Hands - on - RAG - DeepSeek.py**：采用DeepSeek模型实现RAG流程，同样用于回答《黑神话：悟空》相关问题。

## 环境准备

### 依赖安装
在运行项目前，需要安装以下Python库：
```bash
pip install dotenv sentence - transformers faiss - cpu ollama openai tiktoken
```

### 环境变量配置
在项目根目录下创建一个 `.env` 文件，并添加以下内容：
```plaintext
OLLAMA_MODEL=your_ollama_model_name
DEEPSEEK_API_KEY=your_deepseek_api_key
```
请将 `your_ollama_model_name` 替换为你要使用的Ollama模型名称，将 `your_deepseek_api_key` 替换为你的DeepSeek API密钥。

## RAG流程详解

### 1. 数据准备
在代码中，我们首先定义了一系列关于《黑神话：悟空》的文档数据，这些文档包含了游戏的战斗、技能、剧情等多方面信息。这些文档将作为后续检索的基础数据。
```python
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
```

### 2. 文档嵌入
使用 `SentenceTransformer` 模型将文档转换为向量表示。向量表示可以方便后续的相似度检索。
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)
print(f"文档向量维度: {doc_embeddings.shape}")
```

### 3. 向量存储
利用 `faiss` 库创建向量存储，将文档向量添加到向量数据库中。
```python
import faiss
import numpy as np
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"向量数据库中的文档数量: {index.ntotal}")
```

### 4. 相似度检索
对于给定的问题，将其转换为向量表示，然后在向量数据库中进行相似度检索，找出与问题最相关的文档。
```python
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
```

### 5. 提示词构建
将检索到的相关文档和问题组合成一个提示词，用于输入到语言模型中。
```python
prompt = f"""根据以下参考信息回答问题，并给出信息源编号。
如果无法从参考信息中找到答案，请说明无法回答。
参考信息:
{chr(10).join(f"[{i+1}] {doc}" for i, doc in enumerate(context))}
问题: {question}
答案:"""
```

### 6. 答案生成
根据选择的模型（Ollama或DeepSeek），使用提示词生成答案，并计算响应时间和tokens数量。
```python
# 以Ollama为例
from ollama import chat
import time
import tiktoken

try:
    encoding = tiktoken.encoding_for_model(os.getenv("OLLAMA_MODEL"))
except KeyError:
    print("无法自动映射模型到分词器，尝试使用通用编码...")
    encoding = tiktoken.get_encoding("cl100k_base")

input_tokens = len(encoding.encode(prompt))
start_time = time.time()

response = chat(
    model=os.getenv("OLLAMA_MODEL"),  
    messages=[{
        "role": "user",
        "content": prompt
    }],
)

end_time = time.time()
output_tokens = len(encoding.encode(response.message.content))
response_time = end_time - start_time

print(f"\n生成的答案: {response.message.content}")
print(f"响应时间: {response_time:.2f} 秒")
print(f"输入提示词的 tokens 数量: {input_tokens}")
print(f"输出答案的 tokens 数量: {output_tokens}")
print(f"总 tokens 计算量: {input_tokens + output_tokens}")
```

## 注意事项
- 确保网络连接稳定，以便下载所需的模型和访问API。
- 使用DeepSeek模型时，需提供有效的API密钥。
- 若自动映射模型到分词器失败，脚本会尝试使用通用编码 `cl100k_base`。

## 学习建议
- 仔细阅读代码中的注释，理解每一步的作用和原理。
- 尝试修改问题和文档数据，观察检索结果和生成答案的变化。
- 对比使用Ollama和DeepSeek模型生成的答案，分析不同模型的特点。

## 贡献
如果你有任何改进建议或发现问题，欢迎提交Pull Request或创建Issue。

## 许可证
本项目采用[MIT许可证](https://opensource.org/licenses/MIT)。 
