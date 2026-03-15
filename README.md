# 电商智能客服系统

## 项目简介
一款基于智谱AI（GLM-4）+ LangChain RAG技术构建的电商智能客服系统，支持用户意图识别、知识库问答、多轮对话等核心功能，适配Windows + Python 3.11环境，无GPU依赖。

## 核心功能
- 🧠 意图识别：自动识别用户问题类型（查订单/退款/物流/售后/其他）
- 📚 RAG知识库问答：基于FAISS向量库实现精准的知识库检索回答
- 💬 多轮对话：支持上下文记忆，提升交互体验
- ⚡ 兜底回复：知识库无结果时，调用大模型生成友好回复
- 🛠 实用指令：支持退出/清空/历史等交互指令

## 环境要求
- Python 3.11（推荐）
- Windows/Linux/MacOS（Windows已深度适配）
- 智谱AI API密钥（格式：ClientID.ClientSecret）

## 快速开始

### 1. 克隆/下载项目
将所有文件放在同一目录，目录结构如下：
```
ecommerce-cs/
├── main.py              # 客服系统主入口
├── rag_chain.py         # RAG知识库核心逻辑
├── knowledge_base.py    # 知识库封装
├── llm_client.py        # 智谱AI客户端（HTTP直连）
├── .env                 # 配置文件（需自行创建）
├── requirements.txt     # 依赖清单
└── README.md            # 说明文档
```

### 2. 安装依赖
```bash
# 使用清华镜像源快速安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置环境变量
创建`.env`文件，填入以下内容（替换为你的智谱AI密钥）：
```ini
# 智谱AI配置
ZHIPU_API_KEY=你的智谱AI密钥（ClientID.ClientSecret格式）
ZHIPU_API_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
ZHIPU_MODEL=glm-4
ZHIPU_EMBEDDING_MODEL=embedding-2

# 大模型参数
TEMPERATURE_INTENT=0.1    # 意图识别（低温度保证准确性）
TEMPERATURE_REPLY=0.7     # 回复生成（适度随机性）

# 对话配置
MAX_CONTEXT_LENGTH=20     # 最大上下文长度

# RAG配置
CHUNK_SIZE=200            # 文本分块大小
CHUNK_OVERLAP=20          # 分块重叠长度
TOP_K=3                   # 检索最相似的3条结果
```

### 4. 启动系统
```bash
python main.py
```

### 5. 交互说明
- 输入问题：直接输入想要咨询的内容（如“如何申请退货？”）
- 实用指令：
  - `退出`：结束对话
  - `清空`：清空对话历史
  - `历史`：查看对话记录

## 核心文件说明
| 文件 | 功能 |
|------|------|
| `main.py` | 客服系统主入口，包含交互式对话逻辑、意图识别、回复生成 |
| `rag_chain.py` | 重构版RAG流水线，基于LangChain核心组件实现，避开入参兼容问题 |
| `knowledge_base.py` | 知识库封装，提供问答/清空/添加文档等方法 |
| `llm_client.py` | 智谱AI客户端，HTTP直连模式，兼容所有zhipuai SDK版本 |
| `.env` | 配置文件，存储API密钥、模型参数等敏感信息 |

## 扩展功能
### 1. 添加知识库文档
调用`KnowledgeBase`的`add_knowledge_doc`方法，支持加载txt/md格式的文档：
```python
# 示例：添加售后规则文档
kb = KnowledgeBase()
kb.add_knowledge_doc(["售后规则.txt", "退货流程.md"])
```

### 2. 自定义提示词
修改`rag_chain.py`中的`PromptTemplate`，可自定义客服回复风格、规则：
```python
prompt = PromptTemplate.from_template("""
你是专业的电商客服，回答要求：
1. 语气亲切，使用口语化表达
2. 仅基于知识库回答，不编造信息
3. 无相关信息时回复"非常抱歉，我暂时无法解答这个问题，建议联系人工客服哦~"

知识库内容：
{context}

用户问题：{question}

回答：
""")
```

## 常见问题
### Q1: 报错`module 'zhipuai' has no attribute 'chat'`
A1: 已通过HTTP直连模式绕过SDK兼容问题，无需修改代码，确保`.env`密钥格式正确即可。

### Q2: 报错`Missing some input keys: {'query'}`
A2: 已重构RAG链为极简流水线，彻底避开`RetrievalQA`入参绑定问题，直接使用`rag_chain.py`即可。

### Q3: 客服回复“暂无相关信息”
A3: 未添加知识库文档，调用`add_knowledge_doc`方法导入相关文档后即可返回精准回答。

## 许可证
本项目仅供学习使用，请勿用于商业用途。

## 免责声明
- 本项目仅适配智谱AI GLM-4模型，其他模型需自行调整接口
- 使用前请确保已获取智谱AI的合法API密钥，遵守平台使用规范
