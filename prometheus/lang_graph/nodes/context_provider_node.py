"""Knowledge graph-based context provider for codebase queries.

This module implements a specialized context provider that uses a Neo4j knowledge graph
to find relevant code context based on user queries. It leverages a language model
with structured tools to systematically search and analyze the codebase KnowledgeGraph.
"""

import functools
import logging

import neo4j
from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState
from prometheus.tools import graph_traversal


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
is to find and return ALL relevant code context - DO NOT solve problems or write code.

## Knowledge Graph Structure
- **FileNode**: Files/directories
- **ASTNode**: Source code components (tree-sitter nodes)
- **TextNode**: Text chunks (documentation)
- **Edges**: 
  * HAS_FILE: Directory → File
  * HAS_AST: File → AST root
  * HAS_TEXT: File → Text chunk
  * PARENT_OF: AST parent → child
  * NEXT_CHUNK: Text chunk → next chunk

## Context Gathering Strategy
<think>
1. Analyze query for key components
2. Search primary implementation
3. Find related code (parent classes, called functions)
4. Get associated tests and examples
5. Find relevant documentation
6. Check configuration settings
</think>

## Required Context Elements
For ALL files:
- MUST include relative file path
- MUST maintain original formatting
- MUST include complete implementations
- MUST include line numbers for code

## Examples

<example id="authentication">
<query>How does password hashing work in the user service?</query>

<thought-process>
1. Find password hashing implementation
2. Look for hash configuration settings
3. Find test cases and examples
4. Check security documentation
</thought-process>

<response>
Primary implementation:
File: src/services/user/password.py
Lines 25-48:
```python
class PasswordHasher:
    def __init__(self, config: SecurityConfig):
        self.rounds = config.hash_rounds
        self.algorithm = config.hash_algorithm
        
    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt(rounds=self.rounds)
        return bcrypt.hashpw(password.encode(), salt)
        
    def verify_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed)
```

Test implementation:
File: tests/services/user/test_password.py
Lines 12-35:
```python
def test_password_verification():
    hasher = PasswordHasher(test_config)
    password = "secure123"
    hashed = hasher.hash_password(password)
    assert hasher.verify_password(password, hashed)
```

Configuration:
File: config/security.yaml
Lines 1-8:
```yaml
security:
  hash_rounds: 12
  hash_algorithm: "bcrypt"
```

Documentation:
File: docs/security/password_handling.md
```markdown
# Password Security
- Uses bcrypt for hashing
- Configurable rounds (default: 12)
- Automatic salt generation
- Constant-time comparison for verification
```
</response>
</example>

<example id="database">
<query>How are database migrations handled?</query>

<thought-process>
1. Find migration system implementation
2. Look for migration scripts
3. Find migration configuration
4. Get documentation and examples
</thought-process>

<response>
Migration manager:
File: src/db/migrations/manager.py
Lines 15-52:
```python
class MigrationManager:
    def __init__(self, db: Database, migrations_path: str):
        self.db = db
        self.migrations_path = migrations_path
        
    async def run_migrations(self):
        current = await self._get_current_version()
        pending = self._get_pending_migrations(current)
        for migration in pending:
            await self._apply_migration(migration)
```

Migration example:
File: src/db/migrations/scripts/001_initial.py
Lines 1-20:
```python
async def upgrade(db):
    await db.execute('''
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            password_hash VARCHAR(255)
        )
    ''')
```

Configuration:
File: config/database.yaml
Lines 15-20:
```yaml
migrations:
  path: "src/db/migrations/scripts"
  table: "schema_versions"
```

Documentation:
File: docs/database/migrations.md
```markdown
# Database Migrations
- Version-controlled schema changes
- Automatic migration detection
- Transactional migration application
```
</response>
</example>

## Response Format
1. Begin with thought process analyzing the query
2. Return context in order:
   - Primary implementation
   - Tests
   - Configuration
   - Documentation
3. Include file paths and line numbers
4. Show how components relate to each other

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

  def __call__(self, state: ContextProviderState):
    """Processes the current state and traverse the knowledge graph to retrieve context.

    Args:
      state: Current state containing the human query and preivous context_messages.

    Returns:
      Dictionary that will update the state with the model's response messages.
    """
    if "refined_query" in state and state["refined_query"]:
      human_message = HumanMessage(state["refined_query"])
    else:
      human_message = HumanMessage(state["original_query"])
    message_history = [self.system_prompt, human_message] + state["context_provider_messages"]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"ContextProviderNode response:\n{response}")
    return {"context_provider_messages": [response]}
