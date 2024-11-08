from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class BuildClassification(BaseModel):
  exist_build: bool = Field(
    description="Indicates if there is any build system present in the project"
  )
  command_summary: str = Field(
    description="Summary of the build system and list of commands required to build the project"
  )
  fail_log: str = Field(
    description="Contains the error logs if build failed, empty string if successful"
  )


class GeneralBuildSummarizeNode:
  SYS_PROMPT = """\
You are a build system expert analyzing build attempts for software projects. You'll review a history
of commands executed by an agent that attempted to build the project. Examine this build history to:

1. Determine if a build system exists (looking for Makefiles, CMakeLists.txt, package.json, etc.)
2. Analyze the build process and required commands
3. Identify any build failures and their causes

Provide three outputs:
1. exist_build: Boolean indicating if a build system is present
2. command_summary: Concise description of the build system and chronological list of commands needed for building, including:
   - Type of build system detected
   - Required dependencies or setup steps
   - Sequence of build commands to execute
3. fail_log: If build failed, provide the relevant error logs. Empty string if build succeeded

When analyzing commands:
- Focus on essential build steps
- Include dependency installation commands
- List commands in execution order
- Note any required environment setup

When capturing fail logs:
- Include complete error messages
- Focus on build-breaking errors
- Exclude warnings or non-critical messages
- Return empty string if build was successful

The input will contain messages showing the agent's attempts and their results.
"""

  def __init__(self, model: BaseChatModel):
    self.model_with_structured_output = model.with_structured_output(BuildClassification)
    self.sys_prompt = SystemMessage(self.SYS_PROMPT)

  def format_build_history(self, build_messages: Sequence[BaseMessage]):
    formatted_messages = []
    for message in build_messages:
      if isinstance(message, AIMessage):
        formatted_messages.append(f"Assistant message: {message.content}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Tool message: {message.content}")
    return formatted_messages

  def __call__(self, state: IssueAnswerAndFixState):
    message_history = [self.system_prompt] + HumanMessage(
      self.format_build_history(state["build_messages"])
    )
    response = self.model_with_structured_output.invoke(message_history)
    self._logger.debug(f"GeneralBuildSummarizeNode response:\n{response}")

    return {
      "exist_build": response["exist_build"],
      "build_command_summary": response["command_summary"],
      "build_fail_log": response["fail_log"],
    }
