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
You are an assistant for finding the relevant context from a codebase to a query.
The codebase is pre-processed and stored as a graph in a Neo4j graph database.
You should follow a systematic, multi-step approach to search the knowledge graph and find relevant information.

KNOWLEDGE GRAPH STRUCTURE:
Node Types:
* FileNode: Represent a file/dir
* ASTNode: Represent a tree-sitter node (source code components)
* TextNode: Represent a string (chunk of text, can be code documentation)

Edge Types:
* HAS_FILE: Relationship between two FileNode, if one FileNode is the parent dir of another FileNode.
* HAS_AST: Relationship between FileNode and ASTNode, if the ASTNode is the root AST node for FileNode.
* HAS_TEXT: Relationship between FileNode and TextNode, if the TextNode is a chunk of text from FileNode.
* PARENT_OF: Relationship between two ASTNode, if one ASTNode is the parent of another ASTNode.
* NEXT_CHUNK: Relationship between two TextNode, if one TextNode is the next chunk of text of another TextNode.

SEARCH STRATEGY:
1. Query Analysis
   - Break down the query to identify:
     * Key terms and phrases
     * File types or extensions of interest
     * Code structures (functions, classes, etc.)
     * Documentation requirements
   - Determine search priority:
     * Code structure (ASTNode-focused)
     * Documentation (TextNode-focused)
     * File organization (FileNode-focused)

2. Initial File Location
   - Start with file-level search to identify potential relevant files
   - Use basename and relative path searches strategically
   - Consider file relationships (parent/child directories)
   - Preview file contents to confirm relevance

3. Detailed Content Search
   Based on query type:
   
   For Code Structure Queries:
   - Search for specific AST node types (method_declaration, class_declaration)
   - Look for text patterns within AST nodes
   - Traverse parent/child relationships to understand context
   - Examine surrounding code structures

   For Documentation Queries:
   - Search TextNodes for relevant content
   - Follow NEXT_CHUNK relationships to gather complete context
   - Look for associated code structures
   
   For File Organization Queries:
   - Traverse directory structure using HAS_FILE relationships
   - Examine file metadata and relationships
   - Map relevant file hierarchies

4. Context Expansion
   - For each relevant finding:
     * Get parent nodes to understand broader context
     * Get children nodes to understand implementation details
     * Follow NEXT_CHUNK relationships for complete text segments
     * Look for related files in the same directory

5. Result Refinement
   - Prioritize results based on:
     * Direct match quality
     * Context relevance
     * File proximity to other matches
     * Coverage of query aspects

BEST PRACTICES:
1. Always explain your search strategy before executing it
2. Think step-by-step and document your reasoning
3. Start broad and then narrow down based on initial findings
4. Use multiple search approaches for thorough coverage
5. Maintain search context to avoid redundant exploration
6. Consider file relationships and proximity
7. Verify findings by examining surrounding context

IMPORTANT NOTES:
- Text searches are case-sensitive and exact
- Consider both direct and indirect relationships
- Preview files before deep diving into their content
- Use node_id tracking to maintain context during traversal

The file tree of the codebase:
{file_tree}
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph, neo4j_driver: neo4j.Driver):
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
    """
    self.neo4j_driver = neo4j_driver

    self.system_prompt = SystemMessage(self.SYS_PROMPT.format(file_tree=kg.get_file_tree()))
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
      graph_traversal.find_file_node_with_basename, driver=self.neo4j_driver
    )
    find_file_node_with_basename_tool = StructuredTool.from_function(
      func=find_file_node_with_basename_fn,
      name=graph_traversal.find_file_node_with_basename.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithBasenameInput,
    )
    tools.append(find_file_node_with_basename_tool)

    find_file_node_with_relative_path_fn = functools.partial(
      graph_traversal.find_file_node_with_relative_path, driver=self.neo4j_driver
    )
    find_file_node_with_relative_path_tool = StructuredTool.from_function(
      func=find_file_node_with_relative_path_fn,
      name=graph_traversal.find_file_node_with_relative_path.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_RELATIVE_PATH_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithRelativePathInput,
    )
    tools.append(find_file_node_with_relative_path_tool)

    find_ast_node_with_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_text, driver=self.neo4j_driver
    )
    find_ast_node_with_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_fn,
      name=graph_traversal.find_ast_node_with_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInput,
    )
    tools.append(find_ast_node_with_text_tool)

    find_ast_node_with_type_fn = functools.partial(
      graph_traversal.find_ast_node_with_type, driver=self.neo4j_driver
    )
    find_ast_node_with_type_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_fn,
      name=graph_traversal.find_ast_node_with_type.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInput,
    )
    tools.append(find_ast_node_with_type_tool)

    find_ast_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_text_in_file, driver=self.neo4j_driver
    )
    find_ast_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_in_file_fn,
      name=graph_traversal.find_ast_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInFileInput,
    )
    tools.append(find_ast_node_with_text_in_file_tool)

    find_ast_node_with_type_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_in_file, driver=self.neo4j_driver
    )
    find_ast_node_with_type_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_in_file_fn,
      name=graph_traversal.find_ast_node_with_type_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInFileInput,
    )
    tools.append(find_ast_node_with_type_in_file_tool)

    find_ast_node_with_type_and_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_and_text, driver=self.neo4j_driver
    )
    find_ast_node_with_type_and_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_and_text_fn,
      name=graph_traversal.find_ast_node_with_type_and_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_AND_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeAndTextInput,
    )
    tools.append(find_ast_node_with_type_and_text_tool)

    find_text_node_with_text_fn = functools.partial(
      graph_traversal.find_text_node_with_text, driver=self.neo4j_driver
    )
    find_text_node_with_text_tool = StructuredTool.from_function(
      func=find_text_node_with_text_fn,
      name=graph_traversal.find_text_node_with_text.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInput,
    )
    tools.append(find_text_node_with_text_tool)

    find_text_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_text_node_with_text_in_file, driver=self.neo4j_driver
    )
    find_text_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_text_node_with_text_in_file_fn,
      name=graph_traversal.find_text_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInFileInput,
    )
    tools.append(find_text_node_with_text_in_file_tool)

    get_next_text_node_with_node_id_fn = functools.partial(
      graph_traversal.get_next_text_node_with_node_id, driver=self.neo4j_driver
    )
    get_next_text_node_with_node_id_tool = StructuredTool.from_function(
      func=get_next_text_node_with_node_id_fn,
      name=graph_traversal.get_next_text_node_with_node_id.__name__,
      description=graph_traversal.GET_NEXT_TEXT_NODE_WITH_NODE_ID_DESCRIPTION,
      args_schema=graph_traversal.GetNextTextNodeWithNodeIdInput,
    )
    tools.append(get_next_text_node_with_node_id_tool)

    preview_file_content_with_basename_fn = functools.partial(
      graph_traversal.preview_file_content_with_basename, driver=self.neo4j_driver
    )
    preview_file_content_with_basename_tool = StructuredTool.from_function(
      func=preview_file_content_with_basename_fn,
      name=graph_traversal.preview_file_content_with_basename.__name__,
      description=graph_traversal.PREVIEW_FILE_CONTENT_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.PreviewFileContentWithBasenameInput,
    )
    tools.append(preview_file_content_with_basename_tool)

    get_parent_node_fn = functools.partial(
      graph_traversal.get_parent_node, driver=self.neo4j_driver
    )
    get_parent_node_tool = StructuredTool.from_function(
      func=get_parent_node_fn,
      name=graph_traversal.get_parent_node.__name__,
      description=graph_traversal.GET_PARENT_NODE_DESCRIPTION,
      args_schema=graph_traversal.GetParentNodeInput,
    )
    tools.append(get_parent_node_tool)

    get_children_node_fn = functools.partial(
      graph_traversal.get_children_node, driver=self.neo4j_driver
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
    message_history = [self.system_prompt, HumanMessage(state["query"])] + state["context_messages"]
    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"ContextProviderNode response:\n{response}")
    return {"context_messages": [response]}
