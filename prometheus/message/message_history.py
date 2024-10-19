import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

from prometheus.message import message_types
from prometheus.neo4j.message_history_handler import MessageHistoryHandler


class MessageHistory:
  def __init__(self, message_history_handeler: MessageHistoryHandler, max_size: int=10) -> None:
    self.message_history_handeler = message_history_handeler
    assert max_size > 0
    self.max_size = max_size

    self.in_memory_message_history = []
    self.conversation_id = None

  def add_message(self, role: message_types.Role, text: str):
    if self.conversation_id is None:
      conversation_id = uuid.uuid4()
      title = ' '.join(text.split()[:3])
      conversation = message_types.Conversation(conversation_id, title)
      self.message_history_handeler.add_conversation(conversation)
      self.conversation_id = conversation_id

    if len(self.in_memory_message_history) >= self.max_size:
      self.in_memory_message_history.pop(0)

    message = message_types.Message(uuid.uuid4(), len(self.in_memory_message_history), role, text, datetime.now())
    self.in_memory_message_history.append(message)

  def to_langchain_chat_history(self):
    langchain_chat_history = []
    for message in self.in_memory_message_history:
      if message.role == message_types.Role.assistant:
        langchain_chat_history.append(AIMessage(content=message.message))
      elif message.role == message_types.Role.user:
        langchain_chat_history.append(HumanMessage(content=message.message))
    return langchain_chat_history
