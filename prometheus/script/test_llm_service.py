from langchain_core.messages import HumanMessage

from prometheus.app.services.llm_service import LLMService
from prometheus.configuration.config import settings


def test_model_response():
    llm_service = LLMService(
        advanced_model_name=settings.ADVANCED_MODEL,
        base_model_name=settings.BASE_MODEL,
        openai_format_api_key=settings.OPENAI_FORMAT_API_KEY,
        openai_format_base_url=settings.OPENAI_FORMAT_BASE_URL,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        gemini_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=55000,
    )

    # Test base model
    chat_model = llm_service.base_model
    print(f"\nTesting model Base Model: {chat_model.model_name}")

    # Run a simple chat generation
    response = chat_model.invoke([
        HumanMessage(content="Hello! Tell me a fun fact about space.")
    ])
    print("Response:", response.content)

    # Test advanced model
    chat_model = llm_service.advanced_model
    print(f"\nTesting model Advanced Model: {chat_model.model_name}")
    # Run a simple chat generation
    response = chat_model.invoke([
        HumanMessage(content="Hello! Tell me a fun fact about space.")
    ])
    print("Response:", response.content)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_model_response()
