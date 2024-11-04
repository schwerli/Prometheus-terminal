from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row


class PostgresService:
  def __init__(self, postgres_uri: str):
    self.postgres_conn = Connection.connect(
      postgres_uri,
      autocommit=True,
      prepare_threshold=0,
      row_factory=dict_row,
    )
    self.checkpointer = PostgresSaver(self.postgres_conn)
    self.checkpointer.setup()

  def close(self):
    self.postgres_conn.close()

  def get_all_thread_ids(self) -> list[str]:
    all_checkpoints = list(self.checkpointer.list(None))
    all_checkpoints = sorted(
      all_checkpoints, key=lambda x: datetime.fromisoformat(x.checkpoint["ts"]), reverse=True
    )
    unique_thread_ids = []
    seen = set()
    for checkpoint in all_checkpoints:
      thread_id = checkpoint.config["configurable"]["thread_id"]
      if thread_id not in seen:
        unique_thread_ids.append(thread_id)
        seen.add(thread_id)

    return unique_thread_ids

  def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
    read_config = {"configurable": {"thread_id": conversation_id}}
    checkpoint = self.checkpointer.get(read_config)
    messages = []
    messages.append({"role": "user", "text": checkpoint["channel_values"]["query"]})
    for message in checkpoint["channel_values"]["messages"]:
      if isinstance(message, HumanMessage):
        messages.append({"role": "user", "text": message.content})
      elif isinstance(message, AIMessage) and not message.additional_kwargs:
        messages.append({"role": "assistant", "text": message.content})
    return messages
