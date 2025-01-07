from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class CustomChatOpenAI(ChatOpenAI):
  def bind_tools(self, tools, tool_choice=None, **kwargs):
    kwargs["parallel_tool_calls"] = False
    return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


class LLMService:
  def __init__(
    self,
    model_name: str,
    openai_api_key: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    open_router_api_key: str,
  ):
    if "/" in model_name:
      self.model = ChatOpenAI(
        model=model_name,
        api_key=open_router_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=None,
        max_retries=3,
      )
    elif "claude" in model_name:
      self.model = ChatAnthropic(
        model=model_name, api_key=anthropic_api_key, temperature=0.0, max_tokens=8192, max_retries=3
      )
    elif "gpt" in model_name:
      self.model = CustomChatOpenAI(
        model=model_name, api_key=openai_api_key, temperature=0.0, max_tokens=None, max_retries=3
      )
    elif "gemini" in model_name:
      self.model = ChatGoogleGenerativeAI(
        model=model_name, api_key=gemini_api_key, temperature=0.0, max_tokens=None, max_retries=3
      )
