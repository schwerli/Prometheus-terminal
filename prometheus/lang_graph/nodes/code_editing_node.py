import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import file_operation


class CodeEditingNode:
  SYS_PROMPT = """\
You are a specialized code editing agent responsible for implementing precise code changes to fix issues. You will receive:
1. Issue title and description
2. Context summary from previously retrieved codebase information
3. Access to tools for reading and modifying source code

CORE RESPONSIBILITIES:
1. Analyze Issues
   - Understand the problem from issue description
   - Review provided context to identify affected code
   - Determine root cause
   - Plan minimal necessary changes

2. Implement Changes
   - Make precise code modifications
   - Maintain existing patterns and style
   - Preserve important comments and documentation
   - Consider edge cases and error handling
"""

  HUMAN_PROMPT = """\
The user query is: {query}

The retrieved context summary from another agent:
{summary}
"""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.code_editing_node")

  def _init_tools(self, root_path: str):
    tools = []

    read_file_fn = functools.partial(file_operation.read_file, root_path=root_path)
    read_file_tool = StructuredTool.from_function(
      func=read_file_fn,
      name=file_operation.read_file.__name__,
      description=file_operation.READ_FILE_DESCRIPTION,
      args_schema=file_operation.ReadFileInput,
    )
    tools.append(read_file_tool)

    read_file_with_line_numbers_fn = functools.partial(
      file_operation.read_file_with_line_numbers, root_path=root_path
    )
    read_file_with_line_numbers_tool = StructuredTool.from_function(
      func=read_file_with_line_numbers_fn,
      name=file_operation.read_file_with_line_numbers.__name__,
      description=file_operation.READ_FILE_WITH_LINE_NUMBERS_DESCRIPTION,
      args_schema=file_operation.ReadFileWithLineNumbersInput,
    )
    tools.append(read_file_with_line_numbers_tool)

    create_file_fn = functools.partial(file_operation.create_file, root_path=root_path)
    create_file_tool = StructuredTool.from_function(
      func=create_file_fn,
      name=file_operation.create_file.__name__,
      description=file_operation.CREATE_FILE_DESCRIPTION,
      args_schema=file_operation.CreateFileInput,
    )
    tools.append(create_file_tool)

    delete_fn = functools.partial(file_operation.delete, root_path=root_path)
    delete_tool = StructuredTool.from_function(
      func=delete_fn,
      name=file_operation.delete.__name__,
      description=file_operation.DELETE_DESCRIPTION,
      args_schema=file_operation.DeleteInput,
    )
    tools.append(delete_tool)

    edit_file_fn = functools.partial(file_operation.edit_file, root_path=root_path)
    edit_file_tool = StructuredTool.from_function(
      func=edit_file_fn,
      name=file_operation.edit_file.__name__,
      description=file_operation.EDIT_FILE_DESCRIPTION,
      args_schema=file_operation.EditFileInput,
    )
    tools.append(edit_file_tool)

    return tools

  def format_human_message(self, state: IssueAnswerAndFixState) -> HumanMessage:
    preivous_test_log = ""
    if "after_test_output" in state and state["after_test_output"]:
      preivous_test_log = f"\nYour previous edit resulting in the following test output:\n{state['after_test_output']}\n"
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(query=state["query"], summary=state["summary"]) + preivous_test_log
    )
    return human_message

  def __call__(self, state: IssueAnswerAndFixState):
    tools = self._init_tools(state["project_path"])
    model_with_tool = self.model.bind_tools(tools)
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "code_edit_messages"
    ]
    response = model_with_tool.invoke(message_history)
    return {"code_edit_messages": [response]}
