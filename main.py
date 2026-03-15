import logging
import os
from dotenv import load_dotenv
from datetime import datetime

# 加载.env配置文件（读取智谱AI密钥等参数）
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cs_agent")

# 导入核心模块（修复后的版本）
from knowledge_base import KnowledgeBase
from llm_client import llm_client


class CustomerServiceAgent:
    """电商智能客服核心类（整合意图识别+RAG问答）"""

    def __init__(self):
        # 初始化RAG知识库
        self.knowledge_base = KnowledgeBase()
        # 初始化智谱AI LLM客户端
        self.llm_client = llm_client
        # 对话上下文记忆
        self.chat_history = []
        # 意图识别提示词模板
        self.intent_prompt_template = """
请识别用户输入的意图，仅返回以下标签中的一个：
- 查订单：用户询问订单状态、物流、下单相关问题
- 退款：用户询问退货、退款、售后相关问题
- 物流：用户询问快递、发货、收货相关问题
- 售后：用户询问商品质量、保修、维修相关问题
- 其他：不属于以上类别的问题

用户输入：{user_input}
意图标签：
        """
        logger.info("✅ 电商智能客服初始化完成")

    def _recognize_intent(self, user_input: str) -> str:
        """识别用户意图（基于智谱AI）"""
        try:
            # 构造意图识别的消息格式
            intent_messages = [
                {"role": "user", "content": self.intent_prompt_template.format(user_input=user_input)}
            ]
            # 调用LLM的chat_completion方法（修复后的llm_client）
            intent = self.llm_client.chat_completion(
                messages=intent_messages,
                temperature=float(os.getenv("TEMPERATURE_INTENT", 0.1))  # 低温度保证意图识别准确
            )
            # 清洗意图结果（去除多余空格/换行）
            intent = intent.strip()
            # 兜底处理：若识别结果不在预设标签中，归为"其他"
            valid_intents = ["查订单", "退款", "物流", "售后", "其他"]
            if intent not in valid_intents:
                intent = "其他"
            logger.info(f"🔍 用户意图识别结果：{intent}（输入：{user_input[:20]}...）")
            return intent
        except Exception as e:
            logger.error(f"❌ 意图识别失败：{str(e)}")
            return "其他"

    def _generate_reply(self, user_input: str, intent: str) -> str:
        """生成客服回复（优先RAG知识库，兜底LLM）"""
        try:
            # 第一步：调用RAG知识库获取回答
            rag_answer = self.knowledge_base.get_answer(user_input)

            # 第二步：若RAG无结果，根据意图生成兜底回复
            if rag_answer == "暂无相关信息":
                # 构造兜底回复的提示词
                fallback_prompt = f"""
你是专业的电商客服，针对【{intent}】类问题，友好回复用户：
用户问题：{user_input}
要求：
1. 语气亲切、简洁明了
2. 若无法解答，引导用户联系人工客服
3. 符合电商客服沟通习惯

客服回复：
                """
                # 调用LLM生成兜底回复
                fallback_messages = [{"role": "user", "content": fallback_prompt}]
                rag_answer = self.llm_client.chat_completion(
                    messages=fallback_messages,
                    temperature=float(os.getenv("TEMPERATURE_REPLY", 0.7))
                )

            # 更新对话历史
            self.chat_history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_input,
                "intent": intent,
                "reply": rag_answer
            })
            return rag_answer
        except Exception as e:
            logger.error(f"❌ 生成回复失败：{str(e)}")
            return "非常抱歉，处理您的请求时出现错误，请稍后再试。"

    def handle_user_input(self, user_input: str) -> str:
        """处理用户输入的主方法（整合意图识别+回复生成）"""
        if not user_input.strip():
            return "🙋 请问您有什么问题想要咨询呢？"

        try:
            # 1. 识别用户意图
            intent = self._recognize_intent(user_input)
            # 2. 生成回复
            reply = self._generate_reply(user_input, intent)
            return reply
        except Exception as e:
            logger.error(f"❌ 处理用户请求异常：{str(e)}")
            return "非常抱歉，处理您的请求时出现错误，请稍后再试。"

    def clear_chat_history(self):
        """清空对话历史"""
        self.chat_history.clear()
        self.knowledge_base.clear_memory()  # 清空RAG的对话记忆
        logger.info("🧹 对话历史已清空")

    def show_chat_history(self):
        """展示对话历史（调试用）"""
        print("\n===== 对话历史 =====")
        for idx, msg in enumerate(self.chat_history, 1):
            print(f"{idx}. [{msg['time']}] 用户：{msg['user']}")
            print(f"   意图：{msg['intent']} | 客服：{msg['reply']}")
        print("====================\n")


def main():
    """交互式电商客服主入口"""
    # 初始化客服Agent
    cs_agent = CustomerServiceAgent()

    # 欢迎语
    print("=" * 40)
    print("      电商智能客服系统 v1.0")
    print("=" * 40)
    print("💡 支持问题：查订单、退款、物流、售后等")
    print("💡 输入指令：")
    print("   - 退出：结束对话")
    print("   - 清空：清空对话历史")
    print("   - 历史：查看对话记录")
    print("=" * 40 + "\n")

    # 交互式对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("您：").strip()

            # 指令处理
            if user_input == "退出":
                print("客服：感谢您的咨询，祝您购物愉快！😊")
                break
            elif user_input == "清空":
                cs_agent.clear_chat_history()
                print("客服：对话历史已清空！🧹")
                continue
            elif user_input == "历史":
                cs_agent.show_chat_history()
                continue
            elif not user_input:
                print("客服：请输入有效的问题哦~ 📝")
                continue

            # 处理用户输入并生成回复
            reply = cs_agent.handle_user_input(user_input)

            # 输出客服回复
            print(f"客服：{reply}\n")

        except KeyboardInterrupt:
            # 处理Ctrl+C退出
            print("\n客服：对话已终止，感谢您的咨询！")
            break
        except Exception as e:
            logger.error(f"❌ 对话循环异常：{str(e)}")
            print("客服：系统临时故障，请稍后再试！😥\n")


if __name__ == "__main__":
    # 启动客服系统
    main()