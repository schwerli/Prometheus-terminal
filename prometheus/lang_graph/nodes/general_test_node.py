import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.docker.general_container import GeneralContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import container_command


class GeneralTestNode:
  SYS_PROMPT = """\
You are a testing expert responsible for running tests for software projects in an Ubuntu container.
You are positioned at the root of the codebase where you can run commands. Your goal is to determine the correct
testing framework and execute the tests.

Your capabilities:
1. Run commands in the container using the run_command tool from the project root
2. Install test-related dependencies and tools
3. Execute test commands

Test process:
1. Examine project structure from root to identify testing framework and patterns
   - Look for test directories (test/, tests/, spec/, __tests__)
   - Identify test files (*_test.*, *.test.*, test_*, *_spec.*)
   - Check for test configuration files (pytest.ini, jest.config.js, etc.)
2. Determine the testing framework used (pytest, jest, gtest, junit, etc.)
3. Install necessary testing tools and dependencies
4. Execute the appropriate test command

Test execution approaches:
- Run all tests in the project
- Run tests by directory or module if specified
- Run specific test cases if requested
- Generate test coverage reports if supported

Key restrictions:
- Do not analyze test results or failures
- Do not attempt to fix any issues found
- Never modify any source or test files
- Stop after running the tests

Remember:
- Your job is to find and run the tests, nothing more
- All commands are run from the project root directory
- Install any necessary dependencies before running tests
- Simply execute the tests and report that they were run
"""

  def __init__(self, model: BaseChatModel, container: GeneralContainer, before_edit: bool):
    self.tools = self._init_tools(container)
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.before_edit = before_edit
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_test_node")

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
      message += f"\n\nThe previous test summary is:\n{state['test_summary']}"
    return HumanMessage(message)

  def __call__(self, state: IssueAnswerAndFixState):
    if not self.before_edit and "exist_test" in state and not state["exist_test"]:
      self._logger.debug("exist_test is false, skipping test.")
      return {
        "build_messages": [
          AIMessage(content="Previous agent determined there is no test framework.")
        ]
      }

    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "test_messages"
    ]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"GeneralTestNode response:\n{response}")
    return {"test_messages": [response]}
