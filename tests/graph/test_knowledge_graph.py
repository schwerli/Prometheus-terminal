from prometheus.graph.knowledge_graph import KnowledgeGraph
from tests.test_utils import test_project_paths


def test_build_graph():
  knowledge_graph = KnowledgeGraph(test_project_paths.TEST_PROJECT_PATH, 1000)

  assert knowledge_graph._next_node_id == 96
  # 8 FileNode
  # 84 ASTnode
  # 4 TextNode
  assert len(knowledge_graph._knowledge_graph_nodes) == 96
  assert len(knowledge_graph._knowledge_graph_edges) == 98

  assert len(knowledge_graph.get_file_nodes()) == 8
  assert len(knowledge_graph.get_ast_nodes()) == 84
  assert len(knowledge_graph.get_text_nodes()) == 4
  assert len(knowledge_graph.get_parent_of_edges()) == 81
  assert len(knowledge_graph.get_has_file_edges()) == 7
  assert len(knowledge_graph.get_has_ast_edges()) == 3
  assert len(knowledge_graph.get_has_text_edges()) == 4
  assert len(knowledge_graph.get_next_chunk_edges()) == 3
