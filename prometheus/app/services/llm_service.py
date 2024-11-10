import logging

from langchain_community.chat_models import ChatLiteLLM


class LLMService:
  def __init__(self, model_name: str):
    self._logger = logging.getLogger("prometheus.app.services.llm_service")
    self.model = ChatLiteLLM(model=model_name)
    self._logger.info(f"LLM model: {self.model.model}")
