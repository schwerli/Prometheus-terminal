import functools
import logging

import neo4j
from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState
from prometheus.tools import graph_traversal

SYS_PROMPT = """\
You are an assistant for finding the relevant context from a codebase to a query.
The codebase is pre-processed and stored as a graph in a Neo4j graph database.
You have access to different tools that you can use to traversa and search the graph.
Make sure that you have looked at all the details provided in the query, searched all relevant
files, and traversed/searched the graph in a step by step manner.

In the knowledge graph stored in neo4j, we have the following node types:
* FileNode: Represent a file/dir
* ASTNode: Represent a tree-sitter node (source code components)
* TextNode: Represent a string (chunk of text, can be code documentation)

and the following edge types:
* HAS_FILE: Relationship between two FileNode, if one FileNode is the parent dir of another FileNode.
* HAS_AST: Relationship between FileNode and ASTNode, if the ASTNode is the root AST node for FileNode.
* HAS_TEXT: Relationship between FileNode and TextNode, if the TextNode is a chunk of text from FileNode.
* PARENT_OF: Relationship between two ASTNode, if one ASTNode is the parent of another ASTNode.
* NEXT_CHUNK: Relationship between two TextNode, if one TextNode is the next chunk of text of another TextNode.

The graph looks similar to a tree, where the root node is the root directory of the codebase, and files
that are source code or text have corresponding ASTNode and TextNode connected to them.

Here is the unix tree command output for the root directory of the codebase:
{file_tree}
"""


class ContextProviderNode:
  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph, neo4j_driver: neo4j.Driver):
    self.neo4j_driver = neo4j_driver

    self.system_prompt = SystemMessage(SYS_PROMPT.format(file_tree=kg.get_file_tree()))
    self._init_tools()
    self.model_with_tools = model.bind_tools(self.tools)

    self._logger = logging.getLogger("prometheus.agents.context_provider_node")

  def _init_tools(self):
    self.tools = []

    find_file_node_with_basename_fn = functools.partial(
      graph_traversal.find_file_node_with_basename, driver=self.neo4j_driver
    )
    find_file_node_with_basename_tool = StructuredTool.from_function(
      func=find_file_node_with_basename_fn,
      name=graph_traversal.find_file_node_with_basename.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithBasenameInput,
    )
    self.tools.append(find_file_node_with_basename_tool)

    find_file_node_with_relative_path_fn = functools.partial(
      graph_traversal.find_file_node_with_relative_path, driver=self.neo4j_driver
    )
    find_file_node_with_relative_path_tool = StructuredTool.from_function(
      func=find_file_node_with_relative_path_fn,
      name=graph_traversal.find_file_node_with_relative_path.__name__,
      description=graph_traversal.FIND_FILE_NODE_WITH_RELATIVE_PATH_DESCRIPTION,
      args_schema=graph_traversal.FindFileNodeWithRelativePathInput,
    )
    self.tools.append(find_file_node_with_relative_path_tool)

    find_ast_node_with_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_text, driver=self.neo4j_driver
    )
    find_ast_node_with_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_fn,
      name=graph_traversal.find_ast_node_with_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInput,
    )
    self.tools.append(find_ast_node_with_text_tool)

    find_ast_node_with_type_fn = functools.partial(
      graph_traversal.find_ast_node_with_type, driver=self.neo4j_driver
    )
    find_ast_node_with_type_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_fn,
      name=graph_traversal.find_ast_node_with_type.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInput,
    )
    self.tools.append(find_ast_node_with_type_tool)

    find_ast_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_text_in_file, driver=self.neo4j_driver
    )
    find_ast_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_text_in_file_fn,
      name=graph_traversal.find_ast_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTextInFileInput,
    )
    self.tools.append(find_ast_node_with_text_in_file_tool)

    find_ast_node_with_type_in_file_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_in_file, driver=self.neo4j_driver
    )
    find_ast_node_with_type_in_file_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_in_file_fn,
      name=graph_traversal.find_ast_node_with_type_in_file.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeInFileInput,
    )
    self.tools.append(find_ast_node_with_type_in_file_tool)

    find_ast_node_with_type_and_text_fn = functools.partial(
      graph_traversal.find_ast_node_with_type_and_text, driver=self.neo4j_driver
    )
    find_ast_node_with_type_and_text_tool = StructuredTool.from_function(
      func=find_ast_node_with_type_and_text_fn,
      name=graph_traversal.find_ast_node_with_type_and_text.__name__,
      description=graph_traversal.FIND_AST_NODE_WITH_TYPE_AND_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindASTNodeWithTypeAndTextInput,
    )
    self.tools.append(find_ast_node_with_type_and_text_tool)

    find_text_node_with_text_fn = functools.partial(
      graph_traversal.find_text_node_with_text, driver=self.neo4j_driver
    )
    find_text_node_with_text_tool = StructuredTool.from_function(
      func=find_text_node_with_text_fn,
      name=graph_traversal.find_text_node_with_text.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInput,
    )
    self.tools.append(find_text_node_with_text_tool)

    find_text_node_with_text_in_file_fn = functools.partial(
      graph_traversal.find_text_node_with_text_in_file, driver=self.neo4j_driver
    )
    find_text_node_with_text_in_file_tool = StructuredTool.from_function(
      func=find_text_node_with_text_in_file_fn,
      name=graph_traversal.find_text_node_with_text_in_file.__name__,
      description=graph_traversal.FIND_TEXT_NODE_WITH_TEXT_IN_FILE_DESCRIPTION,
      args_schema=graph_traversal.FindTextNodeWithTextInFileInput,
    )
    self.tools.append(find_text_node_with_text_in_file_tool)

    get_next_text_node_with_node_id_fn = functools.partial(
      graph_traversal.get_next_text_node_with_node_id, driver=self.neo4j_driver
    )
    get_next_text_node_with_node_id_tool = StructuredTool.from_function(
      func=get_next_text_node_with_node_id_fn,
      name=graph_traversal.get_next_text_node_with_node_id.__name__,
      description=graph_traversal.GET_NEXT_TEXT_NODE_WITH_NODE_ID_DESCRIPTION,
      args_schema=graph_traversal.GetNextTextNodeWithNodeIdInput,
    )
    self.tools.append(get_next_text_node_with_node_id_tool)

    preview_file_content_with_basename_fn = functools.partial(
      graph_traversal.preview_file_content_with_basename, driver=self.neo4j_driver
    )
    preview_file_content_with_basename_tool = StructuredTool.from_function(
      func=preview_file_content_with_basename_fn,
      name=graph_traversal.preview_file_content_with_basename.__name__,
      description=graph_traversal.PREVIEW_FILE_CONTENT_WITH_BASENAME_DESCRIPTION,
      args_schema=graph_traversal.PreviewFileContentWithBasenameInput,
    )
    self.tools.append(preview_file_content_with_basename_tool)

    get_parent_node_fn = functools.partial(
      graph_traversal.get_parent_node, driver=self.neo4j_driver
    )
    get_parent_node_tool = StructuredTool.from_function(
      func=get_parent_node_fn,
      name=graph_traversal.get_parent_node.__name__,
      description=graph_traversal.GET_PARENT_NODE_DESCRIPTION,
      args_schema=graph_traversal.GetParentNodeInput,
    )
    self.tools.append(get_parent_node_tool)

    get_children_node_fn = functools.partial(
      graph_traversal.get_children_node, driver=self.neo4j_driver
    )
    get_children_node_tool = StructuredTool.from_function(
      func=get_children_node_fn,
      name=graph_traversal.get_children_node.__name__,
      description=graph_traversal.GET_CHILDREN_NODE_DESCRIPTION,
      args_schema=graph_traversal.GetChildrenNodeInput,
    )
    self.tools.append(get_children_node_tool)

  def __call__(self, state: ContextProviderState):
    message_history = [self.system_prompt, HumanMessage(state["query"])] + state["context_messages"]
    response = self.model_with_tools.invoke(message_history)
    return {"context_messages": [response]}
