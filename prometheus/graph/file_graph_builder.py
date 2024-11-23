"""Building knowledge graph for a single file."""

from collections import deque
from pathlib import Path
from typing import Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prometheus.graph.graph_types import (
  ASTNode,
  KnowledgeGraphEdge,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
  TextNode,
)
from prometheus.parser import tree_sitter_parser


class FileGraphBuilder:
  """A class for building knowledge graphs from individual files.

  This class processes files and creates knowledge graph representations using different
  strategies based on the file type. For source code files, it uses tree-sitter to
  create an Abstract Syntax Tree (AST) representation. For markdown files, it creates
  a chain of text nodes based on the document's structure.

  The resulting knowledge graph consists of nodes (KnowledgeGraphNode) connected by
  edges (KnowledgeGraphEdge) with different relationship types (KnowledgeGraphEdgeType).
  """

  def __init__(self, max_ast_depth: int, chunk_size: int, chunk_overlap: int):
    """Initialize the FileGraphBuilder.

    Args:
      max_ast_depth: Maximum depth to traverse in the AST when processing source code files.
        Higher values create more detailed but larger graphs.
      chunk_size: The chunk size for text files.
      chunk_overlap: The overlap size for text files.
    """
    self.max_ast_depth = max_ast_depth
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def supports_file(self, file: Path) -> bool:
    """Checks if we support building knowledge graph for this file."""
    if tree_sitter_parser.supports_file(file):
      return True

    if file.suffix in [".md", ".txt", ".rst"]:
      return True

    return False

  def build_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    """Build knowledge graph for a single file.

    Args:
      parent_node: The parent knowledge graph node that represent the file.
        The node attribute should have type FileNode.
      file: The file to build knowledge graph.
      next_node_id: The next available node id.

    Returns:
      A tuple of (next_node_id, kg_nodes, kg_edges), where next_node_id is the
      new next_node_id, kg_nodes is a list of all nodes created for the file,
      and kg_edges is a list of all edges created for this file.
    """
    # In this case, it is a file that tree sitter can parse (source code)
    if tree_sitter_parser.supports_file(file):
      return self._tree_sitter_file_graph(parent_node, file, next_node_id)

    if file.suffix in [".md", ".txt", ".rst"]:
      return self._text_file_graph(parent_node, file, next_node_id)

  def _tree_sitter_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    """Parse a file to a tree-sitter graph.

    Tree-sitter is an abstract syntax tree (AST) parser to parse source code. We simply
    use the tree representation as the knowledge graph to represent this file.
    In the AST, nodes have PARENT_OF relationship, if one node is a parent of another
    node. For example, 'class_definition' can be a parent of 'function_definition'.
    The root ASTNode is connected to the parent_node using the HAS_AST relationship.

    Args:
      parent_node: The parent knowledge graph node that represent the file.
        The node attribute should have type FileNode.
      file: The file to build knowledge graph.
      next_node_id: The next available node id.

    Returns:
      A tuple of (next_node_id, kg_nodes, kg_edges), where next_node_id is the
      new next_node_id, kg_nodes is a list of all nodes created for the file,
      and kg_edges is a list of all edges created for this file.
    """
    tree_sitter_nodes = []
    tree_sitter_edges = []

    tree = tree_sitter_parser.parse(file)
    if tree.root_node.has_error or tree.root_node.child_count == 0:
      return next_node_id, tree_sitter_nodes, tree_sitter_edges

    ast_root_node = ASTNode(
      type=tree.root_node.type,
      start_line=tree.root_node.start_point[0] + 1,
      end_line=tree.root_node.end_point[0] + 1,
      text=tree.root_node.text.decode("utf-8"),
    )
    kg_ast_root_node = KnowledgeGraphNode(next_node_id, ast_root_node)
    next_node_id += 1
    tree_sitter_nodes.append(kg_ast_root_node)
    tree_sitter_edges.append(
      KnowledgeGraphEdge(parent_node, kg_ast_root_node, KnowledgeGraphEdgeType.has_ast)
    )

    node_stack = deque()
    node_stack.append((tree.root_node, kg_ast_root_node, 1))
    while node_stack:
      tree_sitter_node, kg_node, depth = node_stack.pop()

      if depth > self.max_ast_depth:
        continue

      for tree_sitter_child_node in tree_sitter_node.children:
        child_ast_node = ASTNode(
          type=tree_sitter_child_node.type,
          start_line=tree_sitter_child_node.start_point[0] + 1,
          end_line=tree_sitter_child_node.end_point[0] + 1,
          text=tree_sitter_child_node.text.decode("utf-8"),
        )
        kg_child_ast_node = KnowledgeGraphNode(next_node_id, child_ast_node)
        next_node_id += 1

        tree_sitter_nodes.append(kg_child_ast_node)
        tree_sitter_edges.append(
          KnowledgeGraphEdge(kg_node, kg_child_ast_node, KnowledgeGraphEdgeType.parent_of)
        )

        node_stack.append((tree_sitter_child_node, kg_child_ast_node, depth + 1))
    return next_node_id, tree_sitter_nodes, tree_sitter_edges

  def _text_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
    )
    text = file.open(encoding="utf-8").read()
    documents = text_splitter.create_documents([text])
    return self._documents_to_file_graph(documents, parent_node, next_node_id)

  def _documents_to_file_graph(
    self,
    documents: Sequence[Document],
    parent_node: KnowledgeGraphNode,
    next_node_id: int,
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    """Convert the parsed langchain documents to a knowledge graph.

    The parsed document will form a chain of nodes, where all nodes are connected
    to the parent_node using the HAS_TEXT relationship. The nodes are connected using
    the NEXT_CHUNK relationship in chronological order.

    Args:
      documents: The langchain documents used to create the TextNode.
      parent_node: The parent knowledge graph node that represent the file.
        The node attribute should have type FileNode.
      next_node_id: The next available node id.

    Returns:
      A tuple of (next_node_id, kg_nodes, kg_edges), where next_node_id is the
      new next_node_id, kg_nodes is a list of all nodes created for the file,
      and kg_edges is a list of all edges created for this file.
    """
    document_nodes = []
    document_edges = []

    previous_node = None
    for document in documents:
      text_node = TextNode(
        text=document.page_content,
        metadata=str(document.metadata) if document.metadata else "",
      )
      kg_text_node = KnowledgeGraphNode(next_node_id, text_node)
      next_node_id += 1
      document_nodes.append(kg_text_node)
      document_edges.append(
        KnowledgeGraphEdge(parent_node, kg_text_node, KnowledgeGraphEdgeType.has_text)
      )

      if previous_node:
        document_edges.append(
          KnowledgeGraphEdge(previous_node, kg_text_node, KnowledgeGraphEdgeType.next_chunk)
        )

      previous_node = kg_text_node
    return next_node_id, document_nodes, document_edges
