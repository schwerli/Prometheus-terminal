from langchain_anthropic import ChatAnthropic
from langchain_core.messages import trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class CustomChatOpenAI(ChatOpenAI):
  def bind_tools(self, tools, tool_choice=None, **kwargs):
    kwargs["parallel_tool_calls"] = False
    return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


class LLMService:
  def __init__(
    self, model_name: str, openai_api_key: str, anthropic_api_key: str, gemini_api_key: str
  ):
    if "claude" in model_name:
      model = ChatAnthropic(
        model=model_name, api_key=anthropic_api_key, max_tokens=8192, max_retries=3
      )
    elif "gpt" in model_name:
      model = CustomChatOpenAI(
        model=model_name, api_key=openai_api_key, max_tokens=None, max_retries=3
      )
    elif "gemini" in model_name:
      model = ChatGoogleGenerativeAI(
        model=model_name, api_key=gemini_api_key, max_tokens=None, max_retries=3
      )

    trimmer = trim_messages(
      token_counter=model,
      strategy="last",
      max_tokens=100000,
      start_on="human",
      end_on=("human", "tool"),
      include_system=True,
    )
    self.model = trimmer | model
