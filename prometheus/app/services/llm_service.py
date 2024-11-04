from langchain_community.chat_models import ChatLiteLLM


class LLMService:
  def __init__(self, model_name: str, anthropic_api_key: str):
    self.model = ChatLiteLLM(model=model_name, anthropic_api_key=anthropic_api_key)
