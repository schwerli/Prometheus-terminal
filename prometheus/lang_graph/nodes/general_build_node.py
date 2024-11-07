import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.docker.general_container import GeneralContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import container_command


class GeneralBuildNode:
  SYS_PROMPT = """\
You are a build system expert responsible for figuring out how to build software projects in an Ubuntu container.
You are positioned at the root of the codebase where you can run commands. You must first determine the correct
build approach by analyzing the project structure. You can run commands and install dependencies, but you are
STRICTLY FORBIDDEN from modifying any project files.

Your capabilities:
1. Run commands in the container using the run_command tool from the project root
2. Install system dependencies and build tools
3. Analyze build output and diagnose issues
4. Fix build issues through installing packages or trying alternative build commands

Build process:
1. Examine project structure from root (ls, find, etc.) to identify build system and project type
2. Look for key files like Makefile, package.json, CMakeLists.txt, setup.py, etc.
3. Once build system is identified, install required dependencies and tools
4. Execute appropriate build commands for the detected build system
5. If errors occur, try to fix them through allowed methods (never by editing files)

Error handling and fixing:
- For missing dependencies: Install required packages
- For build tool errors: Try installing different versions or alternative tools
- For path issues: Use environment variables or build flags to correct paths
- For configuration issues: Try different build options or flags
- For platform-specific issues: Install necessary platform packages or tools
- If an error can only be fixed by modifying files: Report this clearly but do not make changes

Fixing strategies (without editing files):
- Try installing additional dependencies
- Use different build flags or options
- Set environment variables
- Try alternative build tools or versions
- Use different build commands or arguments
- Configure build paths through flags

Key restrictions:
- NEVER modify any source files or build configuration files
- Only fix through installing packages, environment variables, or build flags
- Report clearly if an error requires file modifications to fix

Remember:
- First focus on understanding how the project should be built
- All commands are run from the project root directory
- Try all possible fixes that don't involve file modifications
- Document your analysis, attempted fixes, and reasoning
- If a build cannot succeed without file modifications, explain why
"""

  def __init__(self, model: BaseChatModel, container: GeneralContainer):
    self.tools = self._init_tools(container)
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_build_node")

  def _init_tools(self, container: GeneralContainer):
    tools = []

    run_command_fn = functools.partial(container_command.run_command, container=container)
    run_command_tool = StructuredTool.from_function(
      func=run_command_fn,
      name=container_command.run_command.__name__,
      description=container_command.FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION,
      args_schema=container_command.RunCommandInput,
    )
    tools.append(run_command_tool)

    return tools

  def __call__(self, state: IssueAnswerAndFixState):
    message_history = [self.system_prompt] + state["build_messages"]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"GeneralBuildNode response:\n{response}")
    return {"build_messages": [response]}
