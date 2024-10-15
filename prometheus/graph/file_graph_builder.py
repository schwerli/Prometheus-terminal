from collections import deque
from pathlib import Path
from typing import Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from prometheus.graph.graph_types import (
  ASTNode,
  KnowledgeGraphEdge,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
  TextNode,
)
from prometheus.parser import tree_sitter_parser


class FileGraphBuilder:
  def __init__(self, max_ast_depth: int):
    self.max_ast_depth = max_ast_depth

  def supports_file(self, file: Path) -> bool:
    if tree_sitter_parser.supports_file(file):
      return True

    if file.suffix == ".md":
      return True

    return False

  def build_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    if tree_sitter_parser.supports_file(file):
      return self._tree_sitter_file_graph(parent_node, file, next_node_id)

    if file.suffix == ".md":
      return self._markdown_file_graph(parent_node, file, next_node_id)

  def _tree_sitter_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    tree_sitter_nodes = []
    tree_sitter_edges = []

    tree = tree_sitter_parser.parse(file)
    if tree.root_node.has_error or tree.root_node.child_count == 0:
      return next_node_id, tree_sitter_nodes, tree_sitter_edges

    ast_root_node = ASTNode(
      type=tree.root_node.type,
      start_line=tree.root_node.start_point[0],
      end_line=tree.root_node.end_point[0],
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
          start_line=tree_sitter_child_node.start_point[0],
          end_line=tree_sitter_child_node.end_point[0],
          text=tree_sitter_child_node.text.decode("utf-8"),
        )
        kg_child_ast_node = KnowledgeGraphNode(next_node_id, child_ast_node)
        next_node_id += 1

        tree_sitter_nodes.append(kg_child_ast_node)
        tree_sitter_edges.append(
          KnowledgeGraphEdge(
            kg_node, kg_child_ast_node, KnowledgeGraphEdgeType.parent_of
          )
        )

        node_stack.append((tree_sitter_child_node, kg_child_ast_node, depth + 1))
    return next_node_id, tree_sitter_nodes, tree_sitter_edges

  def _markdown_file_graph(
    self, parent_node: KnowledgeGraphNode, file: Path, next_node_id: int
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
    headers_to_split_on = [
      ("#", "Header 1"),
      ("##", "Header 2"),
      ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
      headers_to_split_on=headers_to_split_on
    )
    text = file.open(encoding="utf-8").read()
    documents = markdown_splitter.split_text(text)
    return self._documents_to_file_graph(documents, parent_node, next_node_id)

  def _documents_to_file_graph(
    self,
    documents: Sequence[Document],
    parent_node: KnowledgeGraphNode,
    next_node_id: int,
  ) -> Tuple[int, Sequence[KnowledgeGraphNode], Sequence[KnowledgeGraphEdge]]:
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
          KnowledgeGraphEdge(
            previous_node, kg_text_node, KnowledgeGraphEdgeType.next_chunk
          )
        )

      previous_node = kg_text_node
    return next_node_id, document_nodes, document_edges
