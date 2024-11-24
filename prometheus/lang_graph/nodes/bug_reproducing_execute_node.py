import functools
import logging
from typing import Optional, Sequence

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import container_command
from prometheus.utils.issue_util import format_test_commands


class BugReproducingExecuteNode:
  SYS_PROMPT = """\
You are a testing expert focused solely on executing a single bug reproduction test file.
Your only goal is to run the test file created by the previous agent and report its output.

Adapt the user provided test command to execute the single bug reproduction test file, otherwise
figure out what test framework it uses.

DO NOT EDIT ANY FILES. STOP TRYING IF THE TEST EXECUTES.
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Bug reproducing file message:
{bug_reproducing_file_message}

User provided test commands:
{test_commands}

Bug context summary:
{bug_context}
"""

  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    test_commands: Optional[Sequence[str]] = None,
  ):
    self.kg = kg
    self.test_commands = test_commands
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
    test_commands_str = ""
    if self.test_commands:
      test_commands_str = format_test_commands(self.test_commands)
    return HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=state["issue_comments"],
        bug_reproducing_file_message=state["bug_reproducing_file_messages"][-1].content,
        test_commands=test_commands_str,
        bug_context=state["bug_context"],
      )
    )

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "bug_reproducing_execute_messages"
    ]

    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"BugReproducingExecuteNode response:\n{response}")
    return {"bug_reproducing_execute_messages": [response]}
