from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMService:
  def __init__(
    self, model_name: str, openai_api_key: str, anthropic_api_key: str, gemini_api_key: str
  ):
    if "claude" in model_name:
      self.model = ChatAnthropic(
        model=model_name, api_key=anthropic_api_key, max_tokens=8192, max_retries=3
      )
    elif "gpt" in model_name:
      self.model = ChatOpenAI(
        model=model_name, api_key=openai_api_key, max_tokens=None, max_retries=3
      )
    elif "gemini" in model_name:
      self.model = ChatGoogleGenerativeAI(
        model=model_name, api_key=gemini_api_key, max_tokens=None, max_retries=3
      )
