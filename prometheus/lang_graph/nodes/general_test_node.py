"""Test execution handler for software projects in containerized environments.

This module provides functionality to automatically detect and execute tests for
software projects in Ubuntu containers. It analyzes project structures to identify
testing frameworks, installs necessary dependencies, and executes appropriate test
commands while maintaining a strict boundary around test execution (no modifications
or analysis).
"""

import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.docker.general_container import GeneralContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import container_command


class GeneralTestNode:
  """Executes tests for software projects in containerized environments.

  This class provides automated test execution capabilities by analyzing project
  structures, detecting testing frameworks, and running appropriate test commands
  in an Ubuntu container. It focuses solely on test execution without attempting
  to analyze or fix any test failures.
  """

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
    """Initializes the GeneralTestNode with model, container, and test phase.

    Sets up the test executor with necessary tools, prompts, and logging
    configuration for executing tests in a container environment.

    Args:
      model: Language model instance that will be used for test framework
        detection and command generation. Must be a BaseChatModel implementation.
      container: GeneralContainer instance where the test commands will be
        executed. Should be properly configured with an Ubuntu environment.
      before_edit: Boolean flag indicating whether this test attempt is
        happening before or after code edits. Affects behavior and logging.
    """
    self.tools = self._init_tools(container)
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.before_edit = before_edit
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_test_node")

  def _init_tools(self, container: GeneralContainer):
    """Initializes container operation tools.

    Creates and configures the necessary tools for executing commands
    in the container environment.

    Args:
      container: GeneralContainer instance where commands will be executed.

    Returns:
      List of StructuredTool instances configured for container operations.
    """
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
    """Creates a formatted message containing project structure information.

    Formats the project structure and, if applicable, previous test summary
    into a message for the language model.

    Args:
      state: Current state containing project structure and optional previous test information.

    Returns:
      HumanMessage instance containing formatted project information.
    """
    message = f"The (incomplete) project structure is:\n{state['project_structure']}"
    if not self.before_edit:
      message += f"\n\nThe previous test summary is:\n{state['test_command_summary']}"
    return HumanMessage(message)

  def __call__(self, state: IssueAnswerAndFixState):
    """Executes the test process based on the current state.

    Analyzes the project structure to identify testing frameworks and executes
    appropriate test commands in the container environment. Skips test execution
    if the state indicates no test framework exists.

    Args:
      state: Current state containing project information and test status.

    Returns:
      Dictionary containing test execution messages and results to update the state.
      Key 'test_messages' contains a list of execution-related messages.
    """
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
