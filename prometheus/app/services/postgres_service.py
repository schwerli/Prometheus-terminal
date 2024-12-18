"""Service for managing PostgreSQL connections and conversation history."""

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage


class PostgresService:
  """Manages PostgreSQL database operations for conversation storage.

  This service handles database connections and provides methods to interact
  with stored conversation data. The main functionality is to provide checkpointer
  for LangGraph.
  """

  def __init__(self, postgres_uri: str):
    pass
    # self.postgres_conn = Connection.connect(
    #  postgres_uri,
    #  autocommit=True,
    #  prepare_threshold=0,
    #  row_factory=dict_row,
    # )
    # self.checkpointer = PostgresSaver(self.postgres_conn)
    # self.checkpointer.setup()

  def close(self):
    self.postgres_conn.close()

  def get_all_thread_ids(self) -> list[str]:
    """Retrieves all unique conversation thread IDs from the database.

    Returns a list of thread IDs sorted by timestamp in descending order
    (most recent first), with duplicates removed.

    Returns:
        List of unique thread identifiers as strings.
    """
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
    """Retrieves all messages for a specific conversation thread.

    Fetches and formats the complete message history for a given conversation,
    including both user and assistant messages, but excludes tool messages or
    call to using tools.

    Args:
      conversation_id: Unique identifier for the conversation thread.

    Returns:
      List of message dictionaries, where each dictionary contains:
        - 'role': Either 'user' or 'assistant'
        - 'text': The message content
    """
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
