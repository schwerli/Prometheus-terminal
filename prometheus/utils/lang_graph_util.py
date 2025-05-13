from typing import Callable, Dict, Sequence

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser


def check_remaining_steps(
        state: Dict,
        router: Callable[..., str],
        min_remaining_steps: int,
        remaining_steps_key: str = "remaining_steps",
) -> str:
    original_route = router(state)
    if state[remaining_steps_key] > min_remaining_steps:
        return original_route
    else:
        return "low_remaining_steps"


def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: Sequence[BaseMessage]) -> int:
    """Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For simplicity only supports str Message.contents.
    """
    output_parser = StrOutputParser()
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        msg_content = output_parser.invoke(msg)
        num_tokens += tokens_per_message + str_token_counter(role) + str_token_counter(msg_content)
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


def truncate_messages(
        messages: Sequence[BaseMessage], max_tokens: int = 100000
) -> Sequence[BaseMessage]:
    return trim_messages(
        messages,
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )


def extract_ai_responses(messages: Sequence[BaseMessage]) -> Sequence[str]:
    ai_responses = []
    output_parser = StrOutputParser()
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and (
                index == len(messages) - 1 or isinstance(messages[index + 1], HumanMessage)
        ):
            ai_responses.append(output_parser.invoke(message))
    return ai_responses


def extract_human_queries(messages: Sequence[BaseMessage]) -> Sequence[str]:
    human_queries = []
    output_parser = StrOutputParser()
    for message in messages:
        if isinstance(message, HumanMessage):
            human_queries.append(output_parser.invoke(message))
    return human_queries


def extract_last_tool_messages(messages: Sequence[BaseMessage]) -> Sequence[ToolMessage]:
    tool_messages = []
    last_human_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_index = i
            break

    if last_human_index == -1:
        return []

    for message in messages[last_human_index + 1:]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message)
    return tool_messages


def get_last_message_content(messages: Sequence[BaseMessage]) -> str:
    output_parser = StrOutputParser()
    return output_parser.invoke(messages[-1])


def format_agent_tool_message_history(messages: Sequence[BaseMessage]) -> str:
    formatted_messages = []
    for message in messages:
        if isinstance(message, AIMessage):
            if message.content:
                formatted_messages.append(f"Assistant internal thought: {message.content}")
            if (
                    message.additional_kwargs
                    and "tool_calls" in message.additional_kwargs
                    and message.additional_kwargs["tool_calls"]
            ):
                for tool_call in message.additional_kwargs["tool_calls"]:
                    formatted_messages.append(f"Assistant executed tool: {tool_call['function']}")
        elif isinstance(message, ToolMessage):
            formatted_messages.append(f"Tool output: {message.content}")
    return "\n\n".join(formatted_messages)
