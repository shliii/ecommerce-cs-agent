import logging
import os
import requests  # 直接用HTTP请求，避开SDK版本兼容问题
from dotenv import load_dotenv

# 加载.env配置
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_client")

class ZhipuLLMClient:
    """智谱AI LLM客户端（绕过SDK，直接HTTP调用，兼容所有版本）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 基础配置
            cls._instance.api_key = os.getenv("ZHIPU_API_KEY")
            cls._instance.model = os.getenv("ZHIPU_MODEL", "glm-4")
            cls._instance.api_url = os.getenv("ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
            logger.info("✅ 智谱AI客户端初始化完成（HTTP直连模式）")
        return cls._instance

    def chat_completion(self, messages: list, temperature: float = None) -> str:
        """
        直接调用智谱AI HTTP接口，避开SDK兼容问题
        :param messages: 格式 [{"role": "user", "content": "xxx"}]
        :param temperature: 温度参数
        :return: 回复文本
        """
        try:
            # 构建请求头和参数
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or float(os.getenv("TEMPERATURE_REPLY", 0.7)),
                "stream": False
            }

            # 发送HTTP请求
            response = requests.post(
                url=self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()  # 抛出HTTP错误
            result = response.json()

            # 解析结果
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                logger.info(f"📝 LLM回答生成成功：{answer[:50]}...")
                return answer
            else:
                raise ValueError(f"智谱AI返回空结果：{result}")

        except Exception as e:
            logger.error(f"❌ 智谱AI调用失败：{str(e)}")
            raise e

    def get_llm(self):
        """返回LangChain的ChatZhipuAI实例（供RAG使用）"""
        from langchain_zhipu import ChatZhipuAI
        return ChatZhipuAI(
            model=self.model,
            api_key=self.api_key,
            temperature=float(os.getenv("TEMPERATURE_REPLY", 0.7))
        )

# 全局单例
llm_client = ZhipuLLMClient()

# 测试
if __name__ == "__main__":
    test_msg = [{"role": "user", "content": "你好"}]
    try:
        print(llm_client.chat_completion(test_msg))
    except Exception as e:
        print(f"测试失败：{e}")