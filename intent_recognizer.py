from llm_client import llm_client
from config import config
import logging

logger = logging.getLogger("intent_recognizer")

class IntentRecognizer:
    """用户意图识别模块"""
    
    @classmethod
    def detect_intent(cls, user_input: str) -> str:
        """
        识别用户输入的核心意图
        :param user_input: 用户输入文本
        :return: 标准化意图（查订单/退款/物流/售后/其他）
        """
        # 构造意图识别提示词（精准限定范围）
        system_prompt = f"""
        你是电商客服意图识别助手，仅需识别用户输入的核心意图，且只能从以下列表中选择：{config.ALLOWED_INTENTS}
        规则：
        1. 严格匹配，优先选择最贴合的意图；
        2. 无法匹配时，返回"其他"；
        3. 只返回意图名称，不添加任何额外解释。
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 调用大模型识别意图（低温度保证精准）
        intent = llm_client.chat_completion(messages, temperature=config.TEMPERATURE_INTENT)
        
        # 校验意图合法性
        if intent in config.ALLOWED_INTENTS:
            logger.info(f"识别到意图：{intent}")
            return intent
        else:
            logger.warning(f"意图识别结果不合法：{intent}，默认返回'其他'")
            return "其他"