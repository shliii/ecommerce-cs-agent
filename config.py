import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """系统核心配置类"""
    # 智谱AI API配置
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_API_URL = os.getenv("ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4")
    ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2")

    # 大模型生成参数
    TEMPERATURE_INTENT = float(os.getenv("TEMPERATURE_INTENT", 0.1))
    TEMPERATURE_REPLY = float(os.getenv("TEMPERATURE_REPLY", 0.7))

    # 对话管理配置
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 20))

    # RAG配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))
    TOP_K = int(os.getenv("TOP_K", 3))

    # 意图列表
    ALLOWED_INTENTS = ["查订单", "退款", "物流", "售后", "其他"]


# 单例配置实例
config = Config()


# 配置校验
def validate_config():
    if not config.ZHIPU_API_KEY:
        raise ValueError("请配置ZHIPU_API_KEY环境变量")
    # 创建向量库目录
    if not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        os.makedirs(config.CHROMA_PERSIST_DIRECTORY)
    return True