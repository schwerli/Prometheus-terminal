import datetime
from typing import Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver


def get_all_thread_ids(checkpointer: BaseCheckpointSaver) -> Sequence[str]:
  all_checkpoints = list(checkpointer.list(None))
  all_checkpoints = sorted(
    all_checkpoints, key=lambda x: datetime.datetime.fromisoformat(x.checkpoint["ts"]), reverse=True
  )
  unique_thread_ids = []
  seen = set()
  for checkpoint in all_checkpoints:
    thread_id = checkpoint.config["configurable"]["thread_id"]
    if thread_id not in seen:
      unique_thread_ids.append(thread_id)
      seen.add(thread_id)

  return unique_thread_ids


def get_messages(checkpointer: BaseCheckpointSaver, thread_id: str) -> Sequence[Mapping[str, str]]:
  read_config = {"configurable": {"thread_id": thread_id}}
  checkpoint = checkpointer.get(read_config)
  messages = []
  for message in checkpoint["channel_values"]["messages"]:
    if isinstance(message, HumanMessage):
      messages.append({"role": "user", "text": message.content})
    elif isinstance(message, AIMessage) and not message.additional_kwargs:
      messages.append({"role": "assistant", "text": message.content})
  return messages
