import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.docker.general_container import GeneralContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import container_command


class GeneralBuildNode:
  SYS_PROMPT = """\
You are a build system expert responsible for building software projects in an Ubuntu container.
You are positioned at the root of the codebase where you can run commands. Your goal is to determine
the correct build approach and execute the build.

Your capabilities:
1. Run commands in the container using the run_command tool from the project root
2. Install system dependencies and build tools
3. Execute build commands

Build process:
1. Examine project structure from root (ls, find, etc.) to identify build system and project type
2. Look for build configuration files and project structure to understand the build system
3. Install required dependencies and build tools for the project
4. Execute the appropriate build commands based on the project's setup

Key restrictions:
- Do not analyze build failures
- Do not attempt to fix any issues found
- Never modify any source or build files
- Stop after executing the build command

Remember:
- Your job is to find and run the build command, nothing more
- All commands are run from the project root directory
- Install any necessary dependencies before building
- Simply execute the build and report that it was run
"""

  def __init__(self, model: BaseChatModel, container: GeneralContainer, before_edit: bool):
    self.tools = self._init_tools(container)
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.before_edit = before_edit
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_build_node")

  def _init_tools(self, container: GeneralContainer):
    tools = []

    run_command_fn = functools.partial(container_command.run_command, container=container)
    run_command_tool = StructuredTool.from_function(
      func=run_command_fn,
      name=container_command.run_command.__name__,
      description=container_command.RUN_COMMAND_DESCRIPTION,
      args_schema=container_command.RunCommandInput,
    )
    tools.append(run_command_tool)

    return tools

  def format_human_message(self, state: IssueAnswerAndFixState) -> HumanMessage:
    message = f"The (incomplete) project structure is:\n{state['project_structure']}"
    if not self.before_edit:
      message += f"\n\nThe previous build summary is:\n{state['build_summary']}"
    return HumanMessage(message)

  def __call__(self, state: IssueAnswerAndFixState):
    if not self.before_edit and "exist_build" in state and not state["exist_build"]:
      self._logger.debug("exist_build is false, skipping build.")
      return {
        "build_messages": [AIMessage(content="Previous agent determined there is no build system.")]
      }

    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "build_messages"
    ]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"GeneralBuildNode response:\n{response}")
    return {"build_messages": [response]}
