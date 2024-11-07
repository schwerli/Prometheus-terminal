from langchain_community.chat_models import ChatLiteLLM


class LLMService:
  def __init__(self, model_name: str):
    self.model = ChatLiteLLM(model=model_name)
