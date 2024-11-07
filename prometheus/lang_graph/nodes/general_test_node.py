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
You are a testing expert responsible for figuring out how to run tests for software projects in an Ubuntu container.
You are positioned at the root of the codebase where you can run commands. You must first determine the correct testing
framework and approach by analyzing the project structure. You can run commands and install dependencies, but you are
STRICTLY FORBIDDEN from modifying any project files.

Your capabilities:
1. Run commands in the container using the run_command tool from the project root
2. Install test-related dependencies and tools
3. Analyze test output and diagnose failures
4. Fix test environment issues through allowed methods (never by editing files)

Test analysis process:
1. Examine project structure from root to identify testing framework and patterns
   - Look for test directories (test/, tests/, spec/, __tests__)
   - Identify test files (*_test.*, *.test.*, test_*, *_spec.*)
   - Check for test configuration files (pytest.ini, jest.config.js, etc.)
2. Determine the testing framework used (pytest, jest, gtest, junit, etc.)
3. Install necessary testing tools and dependencies
4. Execute appropriate test commands
5. If errors occur, try to fix them through allowed methods

Test execution strategies:
- Run all tests in the project
- Run tests by directory or module if specified
- Run specific test cases if requested
- Generate test coverage reports if supported
- Run tests with different configurations or environments

Error handling and fixing:
- For missing test dependencies: Install required packages
- For test environment issues: Set up necessary environment variables
- For test discovery problems: Try different test pattern flags
- For framework-specific issues: Install correct framework versions
- For test runner errors: Try alternative runner configurations
- If a test can only be fixed by modifying files: Report this clearly

Fixing strategies (without editing files):
- Install additional test dependencies
- Configure test environment variables
- Use different test runner options
- Try alternative test discovery patterns 
- Adjust test timeout settings
- Set framework-specific configuration via command line

Key restrictions:
- NEVER modify any source or test files
- Only fix through installing packages, environment variables, or test flags
- Report clearly if a test issue requires file modifications to fix

Remember:
- First focus on understanding how tests are structured and run
- All commands are run from the project root directory
- Try all possible fixes that don't involve file modifications
- Document your analysis of test setup and any issues found
- Report test results and coverage clearly"""

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
      description=container_command.FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION,
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
      return {"build_messages": [AIMessage(content="Previous agent determined there is no test framework.")]}

    message_history = [self.system_prompt, self.format_human_message(state)] + state["test_messages"]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"GeneralTestNode response:\n{response}")
    return {"test_messages": [response]}
