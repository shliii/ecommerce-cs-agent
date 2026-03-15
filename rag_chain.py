import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

# 基础导入
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 智谱AI嵌入
from langchain_zhipu import ZhipuAIEmbeddings

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_chain")


class EcommerceRAGChain:
    """电商RAG链（完全重构，避开RetrievalQA入参问题）"""

    def __init__(
            self,
            llm: BaseChatModel,
            embedding: Optional[Embeddings] = None,
            vector_db_path: str = "./ecommerce_vector_db",
            top_k: int = 3
    ):
        self.llm = llm
        self.embedding = embedding or self._get_default_embedding()
        self.vector_db_path = Path(vector_db_path)
        self.top_k = top_k

        # 初始化向量库
        self.vector_db = self._load_or_init_vector_db()
        # 构建极简RAG流水线（替代RetrievalQA，无入参问题）
        self.rag_chain = self._build_rag_pipeline()
        logger.info("✅ RAG链初始化完成（重构版，无入参错误）")

    def _get_default_embedding(self) -> Embeddings:
        """初始化嵌入模型"""
        try:
            return ZhipuAIEmbeddings(
                model=os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2"),
                api_key=os.getenv("ZHIPU_API_KEY")
            )
        except Exception as e:
            logger.warning(f"⚠️ 嵌入模型失败，使用测试嵌入：{e}")
            from langchain_core.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=1024)

    def _load_or_init_vector_db(self) -> FAISS:
        """加载/初始化向量库"""
        if self.vector_db_path.exists() and any(self.vector_db_path.iterdir()):
            logger.info(f"📂 加载向量库：{self.vector_db_path}")
            return FAISS.load_local(
                self.vector_db_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info(f"📁 创建向量库：{self.vector_db_path}")
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            return FAISS.from_texts(["电商知识库初始化"], self.embedding)

    def _build_rag_pipeline(self):
        """
        构建极简RAG流水线：
        1. 检索：根据问题找相似文档
        2. 格式化：拼接检索结果为上下文
        3. 提示词：传入上下文+问题
        4. LLM：生成回答
        5. 解析：输出文本
        """
        # 1. 检索器
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.top_k})

        # 2. 格式化检索结果
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        # 3. 提示词模板（无复杂入参）
        prompt = PromptTemplate.from_template("""
仅基于以下知识库内容回答用户问题，无相关信息时仅回复"暂无相关信息"，不要编造内容：

知识库内容：
{context}

用户问题：{question}

回答：
        """)

        # 4. 构建流水线（核心：无入参绑定，直接传参）
        rag_chain = (
                {
                    "context": retriever | format_docs,  # 检索并格式化上下文
                    "question": RunnablePassthrough()  # 透传用户问题
                }
                | prompt  # 传入提示词
                | self.llm  # LLM生成回答
                | StrOutputParser()  # 解析为字符串
        )
        return rag_chain

    def run(self, question: str) -> Dict[str, Any]:
        """核心问答方法（彻底避开入参问题）"""
        if not question.strip():
            return {"answer": "请输入有效的问题", "source_documents": [], "success": False}

        try:
            # 直接调用流水线，传入纯文本问题（无字典入参）
            answer = self.rag_chain.invoke(question.strip())
            # 检索相关文档（用于返回source_documents）
            source_docs = self.vector_db.similarity_search(question.strip(), k=self.top_k)

            return {
                "answer": answer.strip(),
                "source_documents": source_docs,
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ 问答失败：{e}")
            return {"answer": f"处理问题时出错：{str(e)}", "source_documents": [], "success": False}

    def add_documents(self, file_paths: List[str]) -> bool:
        """添加文档到向量库"""
        try:
            documents = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            for fp in file_paths:
                fp_path = Path(fp)
                if not fp_path.exists():
                    logger.warning(f"⚠️ 文件不存在：{fp}")
                    continue
                # 加载并分块文档
                raw_docs = TextLoader(fp, encoding="utf-8").load()
                split_docs = splitter.split_documents(raw_docs)
                documents.extend(split_docs)

            if documents:
                # 添加到向量库并保存
                self.vector_db.add_documents(documents)
                self.vector_db.save_local(self.vector_db_path)
                logger.info(f"✅ 添加{len(documents)}个文档片段到向量库")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ 添加文档失败：{e}")
            return False


# 测试
if __name__ == "__main__":
    from langchain_zhipu import ChatZhipuAI

    # 初始化LLM
    llm = ChatZhipuAI(
        model=os.getenv("ZHIPU_MODEL", "glm-4"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        temperature=0.1
    )
    # 初始化RAG链
    rag_chain = EcommerceRAGChain(llm=llm)
    # 测试问答
    result = rag_chain.run("如何申请退货？")
    print(f"📝 问题：如何申请退货？")
    print(f"💡 回答：{result['answer']}")