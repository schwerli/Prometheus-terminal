import enum

import dataclasses


class Role(enum.StrEnum):
  """Enum of chat roles"""

  user = "USER"
  assistant = "ASSISTANT"


@dataclasses.dataclass(frozen=True)
class Message:
  role: Role
  message: str
