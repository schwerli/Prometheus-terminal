import dataclasses
from datetime import datetime
import enum
from typing import TypedDict


class Role(enum.StrEnum):
  """Enum of chat roles"""

  user = "USER"
  assistant = "ASSISTANT"


@dataclasses.dataclass(frozen=True)
class Message:
  message_id: str
  index: int
  role: Role
  text: str
  created_at: datetime

  def to_neo4j_message_node(self) -> "Neo4jMessageNode":
    return Neo4jMessageNode(message_id=self.message_id, index=self.index, role=self.role.value, text=self.text, created_at=self.created_at)
  
@dataclasses.dataclass(frozen=True)
class Conversation:
  conversation_id: str
  title: str

  def to_neo4j_conversation_node(self) -> "Neo4jConversationNode":
    return Neo4jConversationNode(conversation_id=self.conversation_id, title=self.title)


class Neo4jMessageNode(TypedDict):
  message_id: str
  index: int
  role: str
  text: str
  created_at: datetime

class Neo4jConversationNode(TypedDict):
  conversation_id: str
  title: str