from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class BuildClassification(BaseModel):
  summary: str = Field(description="Summary of the sucussful commands that builds the project")
  sucussful: bool = Field(description="Whenever the build was successful or not")


class GeneralBuildSummarizeNode:
  SYS_PROMPT = """\
You are a build system expert responsible for analyzing and summarizing the history of build attempts for
a software project. Your goal is to determine if the build was successful and extract the build steps that worked.

Your responsibilities:
1. Analyze the build message history to determine build success and identify working commands
2. Extract and organize key information including:
   - Build system type identified (Make, CMake, npm, etc.)
   - Required system dependencies that were installed
   - Build commands that were executed
   - Important environment variables or flags
   - Build order and dependencies

Success Criteria:
A build is considered successful if:
- All required dependencies were successfully installed
- The final build commands completed without errors
- No critical build failures remained unresolved
- Required artifacts (binaries, packages, etc.) were generated

Output format:
You must output two pieces of information:
1. successful: A boolean indicating if the build succeeded overall
2. summary: A structured summary of the build process as follows:

Build Summary Structure:
1. Build System: [identified build system]
2. Build Status: [Successful/Failed - with brief reason if failed]
3. Required Dependencies: [list of packages/tools that needed to be installed]
4. Environment Setup: [any environment variables or configuration needed]
5. Build Steps:
   - Step 1: [command with status]
   - Step 2: [command with status]
   ...
6. Additional Notes: [important observations, warnings, or failure points]

Guidelines:
- Carefully examine final build output to determine success
- Include exact command syntax for reproducibility
- Note any specific order dependencies between steps
- Document both successful and failed attempts
- Include version numbers of tools where specified
- Note any platform-specific requirements or issues

Remember:
- Be thorough in analyzing build success/failure
- Include error messages or failure points in the summary
- Maintain exact command syntax
- Order steps logically
- Note any critical dependencies between steps
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
    return {"build_summary": response["summary"], "build_success": response["sucussful"]}
