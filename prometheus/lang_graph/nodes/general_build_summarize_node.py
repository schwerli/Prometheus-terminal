from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class BuildClassification(BaseModel):
  exist_build: bool = Field(description="Indicates if there is any build system present in the project")
  summary: str = Field(description="Summary of the sucussful commands that builds the project")
  sucussful: bool = Field(description="Whenever the build was successful or not")


class GeneralBuildSummarizeNode:
  SYS_PROMPT = """\
You are a build system expert analyzing build attempts for software projects. You'll review a history
of commands executed by an agent that attempted to build the project. Examine this build history to:

1. Determine if a build system exists (looking for Makefiles, CMakeLists.txt, package.json, etc.)
2. Assess if the build was successful (all dependencies installed, commands completed, artifacts generated)
3. Extract the working build steps and commands

Provide three outputs:
1. exist_build: Boolean indicating if a build system is present
2. successful: Boolean indicating if the build succeeded
3. summary: Structured build summary containing:
   - Build System: [type or "None"]
   - Status: [Successful/Failed/N/A] with reason
   - Dependencies: [required packages/tools]
   - Environment: [required setup]
   - Steps: [commands with status]
   - Notes: [warnings or issues]

Focus on command accuracy, build order, and critical dependencies. The input will contain messages showing the agent's attempts and their results.
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

    return {"exist_build": response["exist_build"], "build_summary": response["summary"], "build_success": response["sucussful"]}
