import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import file_operation


class BugReproducingFileNode:
  SYS_PROMPT = """\
You are a specialized file management agent responsible for handling bug reproduction code.
Your task is to manage the placement and creation of bug reproducing code files in the project.

Your responsibilities:

1. If a bug reproducing file path is provided:
   - Delete the existing file at that path
   - Create a new file at the same path
   - Write the provided bug reproducing code into the new file

2. If no bug reproducing file path is provided:
   - Create a new file in an appropriate location (usually in a 'tests' directory)
   - Write the provided bug reproducing code into the new file
   - Choose a descriptive name for the file that reflects its purpose

You must use the available tools:
- read_file: To verify if Bug reproducing file exist, and if the content is correct after create_file.
- create_file: To write the new bug reproducing code into a new file.
- delete: To remove existing Bug reproducing file before create_file

After completing the file operation, respond with a message indicating the exact path where you wrote the file:
"Bug reproducing code has been written to: {file_path}"

Rules:
1. Always ensure the file extension matches the code's language
2. Place new files in a logical location within the project structure
3. Use clear, descriptive file names that indicate the bug being reproduced
4. Verify file operations are successful before responding
5. Do not modify other files
6. Do not add any explanatory text beyond the required response message
"""

  HUMAN_PROMPT = """\
Bug reproducing code:
{bug_reproducing_code}

Bug reproducing file:
{reproduced_bug_file}
"""

  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
  ):
    self.kg = kg
    self.tools = self._init_tools(kg.get_local_path())
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_file_node")

  def _init_tools(self, root_path: str):
    """Initializes file operation tools with the given root path.

    Args:
      root_path: Base directory path for all file operations.

    Returns:
      List of StructuredTool instances configured for file operations.
    """
    tools = []

    read_file_fn = functools.partial(file_operation.read_file, root_path=root_path)
    read_file_tool = StructuredTool.from_function(
      func=read_file_fn,
      name=file_operation.read_file.__name__,
      description=file_operation.READ_FILE_DESCRIPTION,
      args_schema=file_operation.ReadFileInput,
    )
    tools.append(read_file_tool)

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

    return tools

  def format_human_message(self, state: BugReproductionState) -> HumanMessage:
    reproduced_bug_file = ""
    if "reproduced_bug_file" in state and state["reproduced_bug_file"]:
      reproduced_bug_file = state["reproduced_bug_file"]
    return HumanMessage(
      self.HUMAN_PROMPT.format(
        bug_reproducing_code=state["bug_reproducing_code"],
        reproduced_bug_file=reproduced_bug_file,
        project_structure=self.kg.get_file_tree(),
      )
    )

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "bug_reproducing_file_messages"
    ]

    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"BugReproducingFileNode response:\n{response}")
    return {"bug_reproducing_file_messages": [response]}