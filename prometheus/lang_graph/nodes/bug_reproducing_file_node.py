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
You are a test file manager responsible for extracting and managing bug reproduction test code. Your primary responsibilities are:

1. Extract complete test code from the provided bug reproducing code
2. Handle file operations to save the test code appropriately

Follow these steps for each task:

1. Code Extraction:
- Analyze the bug_reproducing_code to identify the complete test code
- Ensure all necessary imports and dependencies are included
- Maintain the original code structure and formatting
- Extract only the relevant test code, excluding any explanatory text or comments not part of the code

2. File Management:
If a reproduced_bug_file path exists:
- Use the delete tool to remove the existing file
- Use the create_file tool to recreate the file with the extracted test code
- Maintain the original file extension and location

If no reproduced_bug_file path exists:
- Determine appropriate file name based on the test content (e.g., test_[feature].py for Python tests)
- Choose a suitable location in the project structure for the test file
- Use the create_file tool to create the new file with the extracted test code
- Return the created file path in your response

Requirements:
- Always confirm file operations were successful
- Maintain proper file extensions based on the programming language
- Ensure the file path is within the project structure
- Handle any file operation errors gracefully

Tools available:
- read_file: Read contents of a file
- create_file: Create a new file with specified content
- delete: Delete an existing file

Only respond with the path of the file where the test code was saved
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
    self.tools = self._init_tools(str(kg.get_local_path()))
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
        bug_reproducing_code=state["bug_reproducing_write_messages"][-1].content,
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
