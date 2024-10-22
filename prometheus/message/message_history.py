import uuid
from datetime import datetime
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage

from prometheus.message import message_types
from prometheus.neo4j.message_history_handler import MessageHistoryHandler


class MessageHistory:
  def __init__(self, message_history_handler: MessageHistoryHandler, max_size: int = 10) -> None:
    self.message_history_handler = message_history_handler
    assert max_size > 0
    self.max_size = max_size

    self.message_index = 0
    self.in_memory_message_history = []
    self.conversation_id = None

  def add_message(self, role: message_types.Role, text: str):
    if self.conversation_id is None:
      conversation_id = str(uuid.uuid4())
      title = " ".join(text.split()[:3])
      conversation = message_types.Conversation(conversation_id, title)
      self.message_history_handler.add_conversation(conversation)
      self.conversation_id = conversation_id

    if len(self.in_memory_message_history) >= self.max_size:
      self.in_memory_message_history.pop(0)

    message = message_types.Message(
      str(uuid.uuid4()), self.message_index, role, text, datetime.now()
    )
    self.message_history_handler.add_message(self.conversation_id, message)
    self.in_memory_message_history.append(message)
    self.message_index += 1
    return self.conversation_id

  def load_conversation(self, conversation_id: str):
    if self.conversation_id == conversation_id:
      return

    messages = self.message_history_handler.get_conversation_messages(conversation_id)
    if not messages:
      raise ValueError(f"Conversation with id {conversation_id} does not exists.")
    messages.sort(key=lambda x: x.index)
    self.message_index = messages[-1].index + 1
    self.in_memory_message_history = messages[-self.max_size :]
    self.conversation_id = conversation_id

  def get_all_conversation_messages(self, conversation_id: str) -> Sequence[message_types.Message]:
    return self.message_history_handler.get_conversation_messages(conversation_id)

  def get_all_conversations(self) -> Sequence[message_types.Conversation]:
    return self.message_history_handler.get_all_conversations()

  def to_langchain_chat_history(self):
    langchain_chat_history = []
    for message in self.in_memory_message_history:
      if message.role == message_types.Role.assistant:
        langchain_chat_history.append(AIMessage(content=message.text))
      elif message.role == message_types.Role.user:
        langchain_chat_history.append(HumanMessage(content=message.text))
    return langchain_chat_history
