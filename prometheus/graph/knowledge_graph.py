"""The in-memory knowledge graph representation of a codebase.

In the knowledgegraph, we have the following node types:
* FileNode: Represent a file/dir
* ASTNode: Represent a tree-sitter node
* TextNode: Represent a string

and the following edge types:
* HAS_FILE: Relationship between two FileNode, if one FileNode is the parent dir of another FileNode.
* HAS_AST: Relationship between FileNode and ASTNode, if the ASTNode is the root AST node for FileNode.
* HAS_TEXT: Relationship between FileNode and TextNode, if the TextNode is chunk of text from FileNode.
* PARENT_OF: Relationship between two ASTNode, if one ASTNode is the parent of another ASTNode.
* NEXT_CHUNK: Relationship between two TextNode, if one TextNode is the next chunk of text of another TextNode.

In this way, we have all the directory structure, source code, and text information in a single knowledge graph.
This knowledge graph will be persisted in a graph database (neo4j), where an AI can use it to traverse the
codebase to find the most relevant context for the user query.
"""

from collections import deque
import logging
from pathlib import Path
from typing import Sequence

from prometheus.graph.file_graph_builder import FileGraphBuilder
from prometheus.graph.graph_types import (
  ASTNode,
  FileNode,
  KnowledgeGraphEdge,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
  Neo4jASTNode,
  Neo4jFileNode,
  Neo4jHasASTEdge,
  Neo4jHasFileEdge,
  Neo4jHasTextEdge,
  Neo4jNextChunkEdge,
  Neo4jParentOfEdge,
  Neo4jTextNode,
  TextNode,
)


class KnowledgeGraph:
  def __init__(self, root_dir: Path, max_ast_depth: int):
    """
    Args:
      root_dir: The root directory of the codebase.
      max_ast_depth: The maximum depth AST that we traverse. This can be used to limit
        the number of nodes in the knowledge graph.
    """
    self.max_ast_depth = max_ast_depth
    self._next_node_id = 0
    self._root_node = None
    self._knowledge_graph_nodes = []
    self._knowledge_graph_edges = []
    self._file_graph_builder = FileGraphBuilder(max_ast_depth)
    self._logger = logging.getLogger("prometheus.graph.knowledge_graph")

    self._build_graph(root_dir)

  def _build_graph(self, root_dir: Path):
    """Builds knowledege graph for a codebase at a location.

    Args:
      root_dir: The codebase root directory.
    """
    # The root node for the whole graph
    root_dir_node = FileNode(basename=root_dir.name, relative_path=".")
    kg_root_dir_node = KnowledgeGraphNode(self._next_node_id, root_dir_node)
    self._next_node_id += 1
    self._knowledge_graph_nodes.append(kg_root_dir_node)
    self._root_node = kg_root_dir_node

    file_stack = deque()
    file_stack.append((root_dir, kg_root_dir_node))

    # Now we traverse the file system to parse all the files and create all relationships
    while file_stack:
      file, kg_file_path_node = file_stack.pop()

      # The file is a directory, we create FileNode for all children files.
      if file.is_dir():
        for child_file in sorted(file.iterdir()):
          child_file_node = FileNode(
            basename=child_file.name,
            relative_path=str(child_file.relative_to(root_dir).as_posix()),
          )
          kg_child_file_node = KnowledgeGraphNode(self._next_node_id, child_file_node)
          self._next_node_id += 1
          self._knowledge_graph_nodes.append(kg_child_file_node)
          self._knowledge_graph_edges.append(
            KnowledgeGraphEdge(
              kg_file_path_node,
              kg_child_file_node,
              KnowledgeGraphEdgeType.has_file,
            )
          )

          file_stack.append((child_file, kg_child_file_node))
        continue

      # The file is a file that file_graph_builder supports, it means that we can
      # build a knowledge graph over it.
      if self._file_graph_builder.supports_file(file):
        next_node_id, kg_nodes, kg_edges = self._file_graph_builder.build_file_graph(
          kg_file_path_node, file, self._next_node_id
        )
        self._next_node_id = next_node_id
        self._knowledge_graph_nodes.extend(kg_nodes)
        self._knowledge_graph_edges.extend(kg_edges)
        continue

      # This can only happend for files that are not supported by file_graph_builder.
      self._logger.info(f"Skip parsing {file} because it is not supported.")

  def get_file_nodes(self) -> Sequence[KnowledgeGraphNode]:
    return [
      kg_node
      for kg_node in self._knowledge_graph_nodes
      if isinstance(kg_node.node, FileNode)
    ]

  def get_ast_nodes(self) -> Sequence[KnowledgeGraphNode]:
    return [
      kg_node
      for kg_node in self._knowledge_graph_nodes
      if isinstance(kg_node.node, ASTNode)
    ]

  def get_text_nodes(self) -> Sequence[KnowledgeGraphNode]:
    return [
      kg_node
      for kg_node in self._knowledge_graph_nodes
      if isinstance(kg_node.node, TextNode)
    ]

  def get_has_ast_edges(self) -> Sequence[KnowledgeGraphEdge]:
    return [
      kg_edge
      for kg_edge in self._knowledge_graph_edges
      if kg_edge.type == KnowledgeGraphEdgeType.has_ast
    ]

  def get_has_file_edges(self) -> Sequence[KnowledgeGraphEdge]:
    return [
      kg_edge
      for kg_edge in self._knowledge_graph_edges
      if kg_edge.type == KnowledgeGraphEdgeType.has_file
    ]

  def get_has_text_edges(self) -> Sequence[KnowledgeGraphEdge]:
    return [
      kg_edge
      for kg_edge in self._knowledge_graph_edges
      if kg_edge.type == KnowledgeGraphEdgeType.has_text
    ]

  def get_next_chunk_edges(self) -> Sequence[KnowledgeGraphEdge]:
    return [
      kg_edge
      for kg_edge in self._knowledge_graph_edges
      if kg_edge.type == KnowledgeGraphEdgeType.next_chunk
    ]

  def get_parent_of_edges(self) -> Sequence[KnowledgeGraphEdge]:
    return [
      kg_edge
      for kg_edge in self._knowledge_graph_edges
      if kg_edge.type == KnowledgeGraphEdgeType.parent_of
    ]

  def get_neo4j_file_nodes(self) -> Sequence[Neo4jFileNode]:
    return [kg_node.to_neo4j_node() for kg_node in self.get_file_nodes()]

  def get_neo4j_ast_nodes(self) -> Sequence[Neo4jASTNode]:
    return [kg_node.to_neo4j_node() for kg_node in self.get_ast_nodes()]

  def get_neo4j_text_nodes(self) -> Sequence[Neo4jTextNode]:
    return [kg_node.to_neo4j_node() for kg_node in self.get_text_nodes()]

  def get_neo4j_has_ast_edges(self) -> Sequence[Neo4jHasASTEdge]:
    return [kg_edge.to_neo4j_edge() for kg_edge in self.get_has_ast_edges()]

  def get_neo4j_has_file_edges(self) -> Sequence[Neo4jHasFileEdge]:
    return [kg_edge.to_neo4j_edge() for kg_edge in self.get_has_file_edges()]

  def get_neo4j_has_text_edges(self) -> Sequence[Neo4jHasTextEdge]:
    return [kg_edge.to_neo4j_edge() for kg_edge in self.get_has_text_edges()]

  def get_neo4j_next_chunk_edges(self) -> Sequence[Neo4jNextChunkEdge]:
    return [kg_edge.to_neo4j_edge() for kg_edge in self.get_next_chunk_edges()]

  def get_neo4j_parent_of_edges(self) -> Sequence[Neo4jParentOfEdge]:
    return [kg_edge.to_neo4j_edge() for kg_edge in self.get_parent_of_edges()]
