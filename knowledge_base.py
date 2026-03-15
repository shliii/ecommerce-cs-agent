import logging
import os
from dotenv import load_dotenv
from rag_chain import EcommerceRAGChain
from langchain_zhipu import ChatZhipuAI

# 加载.env配置
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_base")

class KnowledgeBase:
    """电商知识库（最终版，适配.env配置）"""
    def __init__(self):
        # 初始化智谱AI LLM（从.env读取密钥和参数）
        self.llm = ChatZhipuAI(
            model=os.getenv("ZHIPU_MODEL", "glm-4"),
            api_key=os.getenv("ZHIPU_API_KEY"),  # 带小数点的密钥
            temperature=float(os.getenv("TEMPERATURE_REPLY", 0.7))
        )
        # 初始化RAG链
        self.rag_chain = EcommerceRAGChain(llm=self.llm)

    def get_answer(self, user_input: str) -> str:
        """获取问答结果"""
        logger.info(f"🔍 处理用户问题：{user_input[:20]}...")
        result = self.rag_chain.run(user_input)
        return result["answer"]

    def clear_memory(self):
        """清空对话记忆"""
        self.rag_chain.clear_chat_history()

    def add_knowledge_doc(self, file_paths: list) -> bool:
        """添加知识库文档"""
        return self.rag_chain.add_documents(file_paths)

# 测试
if __name__ == "__main__":
    kb = KnowledgeBase()
    answer = kb.get_answer("如何申请退货？")
    print(f"回答：{answer}")