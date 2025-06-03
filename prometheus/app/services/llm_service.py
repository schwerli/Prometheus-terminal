from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class CustomChatOpenAI(ChatOpenAI):
    def bind_tools(self, tools, tool_choice=None, **kwargs):
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


class LLMService:
    def __init__(
        self,
        advanced_model_name: str,
        base_model_name: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        open_router_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
    ):
        self.advanced_model = get_model(
            advanced_model_name,
            openai_api_key,
            anthropic_api_key,
            gemini_api_key,
            open_router_api_key,
            deepseek_api_key,
        )
        self.base_model = get_model(
            base_model_name, openai_api_key, anthropic_api_key, gemini_api_key, open_router_api_key, deepseek_api_key
        )


def get_model(
    model_name: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    open_router_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
) -> BaseChatModel:
    if "/" in model_name:
        return CustomChatOpenAI(
            model=model_name,
            api_key=open_router_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=None,
            max_retries=3,
        )
    elif "claude" in model_name:
        return ChatAnthropic(
            model=model_name,
            api_key=anthropic_api_key,
            temperature=0.0,
            max_tokens=8192,
            max_retries=3,
        )
    elif "gpt" in model_name:
        return CustomChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.0,
            max_tokens=None,
            max_retries=3,
        )
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=gemini_api_key,
            temperature=0.0,
            max_tokens=None,
            max_retries=3,
        )
    elif "deepseek" in model_name:
        return CustomChatOpenAI(
            model=model_name,
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
            temperature=0.0,
            max_tokens=None,
            max_retries=3,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
