from prometheus.graph.file_graph_builder import FileGraphBuilder
from prometheus.graph.graph_types import (
  ASTNode,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
  TextNode,
)
from tests.test_utils import test_project_paths


def test_supports_file():
  file_graph_builder = FileGraphBuilder(0)

  assert file_graph_builder.supports_file(test_project_paths.C_FILE)
  assert file_graph_builder.supports_file(test_project_paths.JAVA_FILE)
  assert file_graph_builder.supports_file(test_project_paths.MD_FILE)
  assert file_graph_builder.supports_file(test_project_paths.PYTHON_FILE)

  assert file_graph_builder.supports_file(test_project_paths.DUMMY_FILE) is False


def test_build_python_file_graph():
  file_graph_builder = FileGraphBuilder(1000)

  parent_kg_node = KnowledgeGraphNode(0, None)
  next_node_id, kg_nodes, kg_edges = file_graph_builder.build_file_graph(
    parent_kg_node, test_project_paths.PYTHON_FILE, 0
  )

  assert next_node_id == 11
  assert len(kg_nodes) == 11
  assert len(kg_edges) == 11

  # Test if some of the nodes exists
  argument_list_ast_node = ASTNode(
    type="argument_list", start_line=0, end_line=0, text='("Hello world!")'
  )
  string_ast_node = ASTNode(type="string", start_line=0, end_line=0, text='"Hello world!"')

  found_argument_list_ast_node = False
  for kg_node in kg_nodes:
    if kg_node.node == argument_list_ast_node:
      found_argument_list_ast_node = True
  assert found_argument_list_ast_node

  found_string_ast_node = False
  for kg_node in kg_nodes:
    if kg_node.node == string_ast_node:
      found_string_ast_node = True
  assert found_string_ast_node

  # Test if some of the edges exists
  found_edge = False
  for kg_edge in kg_edges:
    if (
      kg_edge.source.node == argument_list_ast_node
      and kg_edge.target.node == string_ast_node
      and kg_edge.type == KnowledgeGraphEdgeType.parent_of
    ):
      found_edge = True
  assert found_edge


def test_build_markdown_file_graph():
  file_graph_builder = FileGraphBuilder(1000)

  parent_kg_node = KnowledgeGraphNode(0, None)
  next_node_id, kg_nodes, kg_edges = file_graph_builder.build_file_graph(
    parent_kg_node, test_project_paths.MD_FILE, 0
  )

  assert next_node_id == 4
  assert len(kg_nodes) == 4
  assert len(kg_edges) == 7

  # Test if some of the nodes exists
  header_b_text_node = TextNode(
    text="Text under header B.", metadata="{'Header 1': 'A', 'Header 2': 'B'}"
  )
  header_c_text_node = TextNode(
    text="Text under header C.", metadata="{'Header 1': 'A', 'Header 2': 'C'}"
  )

  found_header_b_text_node = False
  for kg_node in kg_nodes:
    if kg_node.node == header_b_text_node:
      found_header_b_text_node = True
  assert found_header_b_text_node

  found_header_c_text_node = False
  for kg_node in kg_nodes:
    if kg_node.node == header_c_text_node:
      found_header_c_text_node = True
  assert found_header_c_text_node

  # Test if some of the edges exists
  found_edge = False
  for kg_edge in kg_edges:
    if (
      kg_edge.source.node == header_b_text_node
      and kg_edge.target.node == header_c_text_node
      and kg_edge.type == KnowledgeGraphEdgeType.next_chunk
    ):
      found_edge = True
  assert found_edge
