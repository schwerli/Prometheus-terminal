"""Knowledge graph-based context provider for codebase queries.

This module implements a specialized context provider that uses a Neo4j knowledge graph
to find relevant code context based on user queries. It leverages a language model
with structured tools to systematically search and analyze the codebase KnowledgeGraph.
"""

import functools
import logging
from typing import Dict

import neo4j
from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.tools import graph_traversal
from prometheus.utils.lang_graph_util import truncate_messages


class ContextProviderNode:
  """Provides contextual information from a codebase using knowledge graph search.

  This class implements a systematic approach to finding relevant code context
  by searching through a Neo4j knowledge graph representation of a codebase.
  It uses a combination of file structure navigation, AST analysis, and text
  search to gather comprehensive context for queries.

  The knowledge graph contains three main types of nodes:
  - FileNode: Represents files and directories
  - ASTNode: Represents syntactic elements from the code
  - TextNode: Represents documentation and text content
  """

  SYS_PROMPT = """\
You are a specialized context gatherer for a codebase stored in a Neo4j knowledge graph. Your purpose 
is to find and return ALL relevant code context - DO NOT solve problems, write code, or add your own analysis. 
Present the context exactly as found without additional commentary or explanation.

## Knowledge Graph Structure
The knowledge graph represents a codebase with three main node types and their relationships:

Nodes:
- **FileNode**: Represents files and directories in the codebase
- **ASTNode**: Represents source code components (tree-sitter parsed syntax nodes)
- **TextNode**: Represents documentation and text content chunks

Edges:
- **HAS_FILE**: Directory → File relationship
- **HAS_AST**: File → AST root node relationship
- **HAS_TEXT**: File → Text chunk relationship
- **PARENT_OF**: AST parent → child relationship
- **NEXT_CHUNK**: Text chunk → next chunk relationship

## Response Format Requirements

1. ALWAYS return maximum relevant context:
   - Include complete class/function implementations
   - Include parent classes/interfaces
   - Include related configuration files
   - Include associated documentation
   - DO NOT add any explanations or analysis of the code
   - DO NOT summarize or explain what the code does
   - ONLY show the exact context as found in the knowledge graph

2. For source code (ASTNodes):
   - MUST include file path
   - MUST include line numbers
   - MUST maintain original formatting
   - MUST include complete implementations
   - DO NOT add your own comments or explanations

Example ASTNode response:
```
File: src/auth/password_manager.py
Lines 15-45:
```python
class PasswordManager:
    def __init__(self, config):
        self.config = config
        
    def hash_password(self, password: str) -> str:
        # Complete implementation included
        salt = generate_salt()
        return hash_with_salt(password, salt)
```

Parent class:
File: src/auth/base_manager.py
Lines 10-25:
```python
class BaseManager:
    def __init__(self, config):
        self.config = config
```
```

3. For documentation (TextNodes):
   - MUST include file path
   - MUST include complete relevant sections
   - MUST preserve formatting
   - MUST include related configuration
   - DO NOT add your own summaries or explanations

Example TextNode response:
```
File: docs/auth/password_handling.md
```markdown
# Password Management
- Secure password hashing using Argon2
- Configurable salt generation
- Automatic upgrade of legacy hashes

## Configuration Options
Default settings in config/auth.yaml:
```yaml
auth:
  hash_algorithm: argon2
  salt_length: 16
  memory_cost: 65536
```
```

The file tree of the codebase:
{file_tree}

All ASTNode types: {ast_node_types}
"""

  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_result: int,
  ):
    """Initializes the ContextProviderNode with model, knowledge graph, and database connection.

    Sets up the context provider with necessary prompts, graph traversal tools,
    and logging configuration. Initializes the system prompt with the current
    file tree structure from the knowledge graph.

    Args:
      model: Language model instance that will be used for query analysis and
        context finding. Must be a BaseChatModel implementation that supports
        tool binding.
      kg: Knowledge graph instance containing the processed codebase structure.
        Used to obtain the file tree for system prompts.
      neo4j_driver: Neo4j driver instance for executing graph queries. This
        driver should be properly configured with authentication and
        connection details.
      max_token_per_result: Maximum number of tokens per retrieved Neo4j result.
    """
    self.neo4j_driver = neo4j_driver
    self.max_token_per_result = max_token_per_result

    ast_node_types_str = ", ".join(kg.get_all_ast_node_types())
    self.system_prompt = SystemMessage(
      self.SYS_PROMPT.format(file_tree=kg.get_file_tree(), ast_node_types=ast_node_types_str)
    )
    self.tools = self._init_tools()
    self.model_with_tools = model.bind_tools(self.tools)

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_provider_node")

  def _init_tools(self):
    """Initializes KnowledgeGraph traversal tools.


    Returns:
      List of StructuredTool instances configured for KnowledgeGraph traversal.
    """
    tools = []

    find_file_node_with_basename_fn = functools.partial(
      graph_traversal.find_file_node_with_basename,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_file_node_with_basename_tool = StructuredTool.from_function(
      func=find_file_node_with_basename_fn,
      name=graph_traversal.find_file_node_with_basename.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithBasenameInput,
    )
    tools.append(find_file_node_with_basename_tool)

    find_file_node_with_relative_path_fn = functools.partial(
      graph_traversal.find_file_node_with_relative_path,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_file_node_with_relative_path_tool = StructuredTool.from_function(
      func=find_file_node_with_relative_path_fn,
      name=graph_traversal.find_file_node_with_relative_path.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_RELATIVE_PATH_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithRelativePathInput,
    )
    tools.append(find_file_node_with_relative_path_tool)

    find_ast_node_with_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_text,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_ast_node_with_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_fn,
      name=graph_traversal.find_ast_node_with_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInput,
    )
    tools.append(find_ast_node_with_text_tool)

    find_ast_node_with_type_fn = functools.partial(
      graph_traversal.find_ast_node_with_type,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_ast_node_with_type_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_fn,
      name=graph_traversal.find_ast_node_with_type.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInput,
    )
    tools.append(find_ast_node_with_type_tool)

    find_ast_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_text_in_file,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_ast_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_in_file_fn,
      name=graph_traversal.find_ast_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInFileInput,
    )
    tools.append(find_ast_node_with_text_in_file_tool)

    find_ast_node_with_type_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_in_file,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_ast_node_with_type_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_in_file_fn,
      name=graph_traversal.find_ast_node_with_type_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInFileInput,
    )
    tools.append(find_ast_node_with_type_in_file_tool)

    find_ast_node_with_type_and_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_and_text,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_ast_node_with_type_and_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_and_text_fn,
      name=graph_traversal.find_ast_node_with_type_and_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_AND_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeAndTextInput,
    )
    tools.append(find_ast_node_with_type_and_text_tool)

    find_text_node_with_text_fn = functools.partial(
      graph_traversal.find_text_node_with_text,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_text_node_with_text_tool = StructuredTool.from_function(
      func=find_text_node_with_text_fn,
      name=graph_traversal.find_text_node_with_text.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInput,
    )
    tools.append(find_text_node_with_text_tool)

    find_text_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_text_node_with_text_in_file,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    find_text_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_text_node_with_text_in_file_fn,
      name=graph_traversal.find_text_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInFileInput,
    )
    tools.append(find_text_node_with_text_in_file_tool)

    get_next_text_node_with_node_id_fn = functools.partial(
      graph_traversal.get_next_text_node_with_node_id,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    get_next_text_node_with_node_id_tool = StructuredTool.from_function(
      func=get_next_text_node_with_node_id_fn,
      name=graph_traversal.get_next_text_node_with_node_id.__name__,
      description=graph_traversal.GET_NEXT_TEXT_NODE_WITH_NODE_ID_DESCRIPTION,
      args_schema=graph_traversal.GetNextTextNodeWithNodeIdInput,
    )
    tools.append(get_next_text_node_with_node_id_tool)

    preview_file_content_with_basename_fn = functools.partial(
      graph_traversal.preview_file_content_with_basename,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    preview_file_content_with_basename_tool = StructuredTool.from_function(
      func=preview_file_content_with_basename_fn,
      name=graph_traversal.preview_file_content_with_basename.__name__,
      description=graph_traversal.PREVIEW_FILE_CONTENT_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.PreviewFileContentWithBasenameInput,
    )
    tools.append(preview_file_content_with_basename_tool)

    get_parent_node_fn = functools.partial(
      graph_traversal.get_parent_node,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    get_parent_node_tool = StructuredTool.from_function(
      func=get_parent_node_fn,
      name=graph_traversal.get_parent_node.__name__,
      description=graph_traversal.GET_PARENT_NODE_DESCRIPTION,
      args_schema=graph_traversal.GetParentNodeInput,
    )
    tools.append(get_parent_node_tool)

    get_children_node_fn = functools.partial(
      graph_traversal.get_children_node,
      driver=self.neo4j_driver,
      max_token_per_result=self.max_token_per_result,
    )
    get_children_node_tool = StructuredTool.from_function(
      func=get_children_node_fn,
      name=graph_traversal.get_children_node.__name__,
      description=graph_traversal.GET_CHILDREN_NODE_DESCRIPTION,
      args_schema=graph_traversal.GetChildrenNodeInput,
    )
    tools.append(get_children_node_tool)

    return tools

  def __call__(self, state: Dict):
    """Processes the current state and traverse the knowledge graph to retrieve context.

    Args:
      state: Current state containing the human query and preivous context_messages.

    Returns:
      Dictionary that will update the state with the model's response messages.
    """
    message_history = [self.system_prompt] + state["context_provider_messages"]
    truncated_message_history = truncate_messages(message_history)
    response = self.model_with_tools.invoke(truncated_message_history)
    self._logger.debug(f"ContextProviderNode response:\n{response}")
    return {"context_provider_messages": [response]}
