import pytest
from prometheus.graph.knowledge_graph import KnowledgeGraph
from tests.test_utils import test_project_paths

from testcontainers.neo4j import Neo4jContainer
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


def test_build_graph():
  knowledge_graph = KnowledgeGraph(1000)
  knowledge_graph.build_graph(test_project_paths.TEST_PROJECT_PATH)

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


def test_get_file_tree():
  knowledge_graph = KnowledgeGraph(1000)
  knowledge_graph.build_graph(test_project_paths.TEST_PROJECT_PATH)
  file_tree = knowledge_graph.get_file_tree()
  expected_file_tree = """\
test_project
├── bar
|   ├── test.java
|   └── test.py
├── foo
|   ├── test.dummy
|   └── test.md
└── test.c"""
  assert file_tree == expected_file_tree

def test_from_neo4j():
  kg = KnowledgeGraph(1000)
  kg.build_graph(test_project_paths.TEST_PROJECT_PATH)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = KnowledgeGraphHandler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, "neo4j", 100)
    handler.write_knowledge_graph(kg)
    
    read_kg = handler.read_knowledge_graph()

    assert read_kg == kg
    