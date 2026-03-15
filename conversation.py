from config import config
import logging

logger = logging.getLogger("conversation")

class ConversationManager:
    """多轮对话上下文管理模块"""
    
    def __init__(self, max_length: int = None):
        self.max_length = max_length or config.MAX_CONTEXT_LENGTH
        self.context = []  # 上下文列表，元素为{"role": "...", "content": "..."}

    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到上下文
        :param role: 角色（user/assistant/system）
        :param content: 消息内容
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError("角色只能是user/assistant/system")
        
        self.context.append({"role": role, "content": content})
        
        # 控制上下文长度，超过则删除最早的消息
        if len(self.context) > self.max_length:
            removed_msg = self.context.pop(0)
            logger.info(f"上下文超出最大长度，删除最早消息：{removed_msg['content'][:20]}...")

    def get_context(self) -> list:
        """获取完整上下文（适配LLM API格式）"""
        return self.context.copy()

    def clear_context(self) -> None:
        """清空上下文"""
        self.context.clear()
        logger.info("对话上下文已清空")

    def get_context_length(self) -> int:
        """获取当前上下文消息数"""
        return len(self.context)