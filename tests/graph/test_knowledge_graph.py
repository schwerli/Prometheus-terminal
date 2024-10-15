from prometheus.graph.graph_types import (
  ASTNode,
  FileNode,
  KnowledgeGraphEdgeType,
  TextNode,
)
from prometheus.graph.knowledge_graph import KnowledgeGraph
from tests.test_utils import test_project_paths


def test_build_graph():
  knowledge_graph = KnowledgeGraph(test_project_paths.TEST_PROJECT_PATH, 1000)

  assert knowledge_graph._next_node_id == 97
  # 8 FileNode
  # 85 ASTnode
  # 4 TextNode
  assert len(knowledge_graph._knowledge_graph_nodes) == 97
  assert len(knowledge_graph._knowledge_graph_edges) == 99

  file_nodes = [
    kg_node
    for kg_node in knowledge_graph._knowledge_graph_nodes
    if isinstance(kg_node.node, FileNode)
  ]
  assert len(file_nodes) == 8

  ast_nodes = [
    kg_node
    for kg_node in knowledge_graph._knowledge_graph_nodes
    if isinstance(kg_node.node, ASTNode)
  ]
  assert len(ast_nodes) == 85

  text_nodes = [
    kg_node
    for kg_node in knowledge_graph._knowledge_graph_nodes
    if isinstance(kg_node.node, TextNode)
  ]
  assert len(text_nodes) == 4

  parent_of_edges = [
    kg_edge
    for kg_edge in knowledge_graph._knowledge_graph_edges
    if kg_edge.type == KnowledgeGraphEdgeType.parent_of
  ]
  assert len(parent_of_edges) == 82

  has_file_edges = [
    kg_edge
    for kg_edge in knowledge_graph._knowledge_graph_edges
    if kg_edge.type == KnowledgeGraphEdgeType.has_file
  ]
  assert len(has_file_edges) == 7

  has_ast_edges = [
    kg_edge
    for kg_edge in knowledge_graph._knowledge_graph_edges
    if kg_edge.type == KnowledgeGraphEdgeType.has_ast
  ]
  assert len(has_ast_edges) == 3

  has_text_edges = [
    kg_edge
    for kg_edge in knowledge_graph._knowledge_graph_edges
    if kg_edge.type == KnowledgeGraphEdgeType.has_text
  ]
  assert len(has_text_edges) == 4

  next_chunk_edges = [
    kg_edge
    for kg_edge in knowledge_graph._knowledge_graph_edges
    if kg_edge.type == KnowledgeGraphEdgeType.next_chunk
  ]
  assert len(next_chunk_edges) == 3
