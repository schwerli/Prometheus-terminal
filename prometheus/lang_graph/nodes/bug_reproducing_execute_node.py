import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import container_command


class BugReproducingExecuteNode:
  SYS_PROMPT = """\
You are a testing expert focused solely on executing a single bug reproduction test file.
Your only goal is to run the test file created by the previous agent and report its output.

Your capabilities:
1. Run test commands to execute the single test file
2. Identify the test framework from the file content (pytest, unittest, etc.)
3. Report the complete test output

Assumptions:
- All dependencies are already installed
- Development environment is fully set up
- Required test frameworks are available
- You only need to focus on executing the specific test file

Process:
1. Review the test file content from previous agent's message
2. Identify the appropriate test command for the single file
3. Execute the test command
4. Report results:
   If successful execution:
   - Include the exact command used
   - Include the complete test output (errors, failures, stack traces)
   
   If execution fails:
   - Include the exact command(s) attempted
   - Include any error messages from the test runner
   - Detail what prevented the test from running

Remember:
- Execute ONLY the single test file written by previous agent
- Use the appropriate test runner command 
- Always include the complete command output
- Do not analyze or interpret the results
- Do not attempt to fix or modify anything
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Project structure:
{project_structure}

Bug context summary:
{bug_context}

Bug reproducing write agent message:
{bug_reproducing_write_message}
"""

  def __init__(self, model: BaseChatModel, container: BaseContainer, kg: KnowledgeGraph):
    self.kg = kg
    self.tools = self._init_tools(container)
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_execute_node")

  def _init_tools(self, container: BaseContainer):
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

  def format_human_message(self, state: BugReproductionState) -> HumanMessage:
    last_bug_producing_write_message = state["bug_reproducing_write_message"][-1]
    last_bug_producing_write_message_content = last_bug_producing_write_message.content
    return HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=state["issue_comments"],
        project_structure=self.kg.get_file_tree(),
        bug_context=state["bug_context"],
        bug_reproducing_write_message=last_bug_producing_write_message_content,
      )
    )

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt, self.format_human_message(state)] + state.get(
      "bug_reproducing_execute_messages", []
    )

    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"BugReproducingExecuteNode response:\n{response}")
    return {
      "bug_reproducing_execute_messages": [response],
      "last_bug_reproducing_execute_message": response,
    }
