# 电商智能客服AI Agent

## 项目简介
一套基于大语言模型（GLM-4）的电商智能客服系统，具备**意图识别、知识库匹配、多轮对话管理、通用回复生成**四大核心能力，采用工业级模块化设计，可直接部署为电商平台的智能客服解决方案。

## 核心功能
✅ **精准意图识别**：基于LLM识别用户核心诉求（查订单/退款/物流/售后）  
✅ **标准化知识库**：高频问题预设标准答案，保证回复精准性  
✅ **多轮对话管理**：维护上下文，支持连贯的多轮交互  
✅ **通用回复生成**：长尾问题调用大模型生成自然回复  
✅ **鲁棒性保障**：完善的异常处理、超时重试、配置校验机制  

## 技术栈
- **核心语言**：Python 3.9+
- **大模型对接**：智谱AI API（GLM-4）
- **工程化**：环境变量管理、日志记录、模块化设计
- **扩展能力**：支持对接OpenAI API、Flask/FastAPI部署、Docker容器化

## 快速开始
### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/shliii/ecommerce-cs-agent.git
cd ecommerce-cs-agent

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
