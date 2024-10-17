from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from prometheus.agents import message_types


class ChatHistory:
  def __init__(self, max_size: int) -> None:
    assert max_size > 0
    self.max_size = max_size

    self.chat_history = []

  def add_message(self, message: message_types.Message):
    if len(self.chat_history) >= self.max_size:
      self.chat_history.pop(0)

    self.chat_history.append(message)

  def get_last_message(self) -> Optional[message_types.Message]:
    if self.chat_history:
      return self.chat_history[-1]
    return None

  def to_langchain_chat_history(self):
    langchain_chat_history = []
    for message in self.chat_history:
      if message.role == message_types.Role.assistant:
        langchain_chat_history.append(AIMessage(content=message.message))
      elif message.role == message_types.Role.user:
        langchain_chat_history.append(HumanMessage(content=message.message))
    return langchain_chat_history
