import logging

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMService:
  def __init__(self, model_name: str):
    self._logger = logging.getLogger("prometheus.app.services.llm_service")

    if "claude" in model_name:
      self.model = ChatAnthropic(model=model_name, max_tokens=8192)
    elif "gpt" in model_name:
      self.model = ChatOpenAI(model=model_name, max_tokens=None)
    elif "gemini" in model_name:
      self.model = ChatGoogleGenerativeAI(model=model_name, max_tokens=None)
    self._logger.info(f"LLM model: {self.model.model}")
