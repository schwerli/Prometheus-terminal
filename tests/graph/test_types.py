from prometheus.graph.types import (
  ASTNode,
  FileNode,
  KnowledgeGraphEdge,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
  TextNode,
)


def test_to_neo4j_file_node():
  basename = "foo"
  relative_path = "foo/bar/baz.py"
  node_id = 1

  file_node = FileNode(basename, relative_path)
  knowldege_graph_node = KnowledgeGraphNode(node_id, file_node)
  neo4j_file_node = knowldege_graph_node.to_neo4j_node()

  assert isinstance(neo4j_file_node, dict)

  assert "node_id" in neo4j_file_node
  assert "basename" in neo4j_file_node
  assert "relative_path" in neo4j_file_node

  assert neo4j_file_node["node_id"] == node_id
  assert neo4j_file_node["basename"] == basename
  assert neo4j_file_node["relative_path"] == relative_path


def test_to_neo4j_ast_node():
  type = "method_declaration"
  start_line = 1
  end_line = 5
  text = "print('Hello world')"
  node_id = 1

  ast_node = ASTNode(type, start_line, end_line, text)
  knowldege_graph_node = KnowledgeGraphNode(node_id, ast_node)
  neo4j_ast_node = knowldege_graph_node.to_neo4j_node()

  assert isinstance(neo4j_ast_node, dict)

  assert "node_id" in neo4j_ast_node
  assert "type" in neo4j_ast_node
  assert "start_line" in neo4j_ast_node
  assert "end_line" in neo4j_ast_node
  assert "text" in neo4j_ast_node

  assert neo4j_ast_node["node_id"] == node_id
  assert neo4j_ast_node["type"] == type
  assert neo4j_ast_node["start_line"] == start_line
  assert neo4j_ast_node["end_line"] == end_line
  assert neo4j_ast_node["text"] == text


def test_to_neo4j_text_node():
  text = "Hello world"
  node_id = 1

  text_node = TextNode(text)
  knowldege_graph_node = KnowledgeGraphNode(node_id, text_node)
  neo4j_text_node = knowldege_graph_node.to_neo4j_node()

  assert isinstance(neo4j_text_node, dict)

  assert "node_id" in neo4j_text_node
  assert "text" in neo4j_text_node

  assert neo4j_text_node["node_id"] == node_id
  assert neo4j_text_node["text"] == text


def test_to_neo4j_has_file_edge():
  source_basename = "source"
  source_relative_path = "foo/bar/source.py"
  source_node_id = 1
  target_basename = "target"
  target_relative_path = "foo/bar/target.py"
  target_node_id = 10

  source_file_node = FileNode(source_basename, source_relative_path)
  source_knowledge_graph_node = KnowledgeGraphNode(source_node_id, source_file_node)
  target_file_node = FileNode(target_basename, target_relative_path)
  target_knowledge_graph_node = KnowledgeGraphNode(target_node_id, target_file_node)
  knowledge_graph_edge = KnowledgeGraphEdge(
    source_knowledge_graph_node,
    target_knowledge_graph_node,
    KnowledgeGraphEdgeType.has_file,
  )
  neo4j_has_file_edge = knowledge_graph_edge.to_neo4j_edge()

  assert isinstance(neo4j_has_file_edge, dict)

  assert "source" in neo4j_has_file_edge
  assert "target" in neo4j_has_file_edge

  assert neo4j_has_file_edge["source"] == source_knowledge_graph_node.to_neo4j_node()
  assert neo4j_has_file_edge["target"] == target_knowledge_graph_node.to_neo4j_node()


def test_to_neo4j_has_ast_edge():
  source_basename = "source"
  source_relative_path = "foo/bar/source.py"
  source_node_id = 1
  target_type = "return_statement"
  target_start_line = 7
  target_end_line = 9
  target_text = "return True"
  target_node_id = 10

  source_file_node = FileNode(source_basename, source_relative_path)
  source_knowledge_graph_node = KnowledgeGraphNode(source_node_id, source_file_node)
  target_ast_node = ASTNode(
    target_type, target_start_line, target_end_line, target_text
  )
  target_knowledge_graph_node = KnowledgeGraphNode(target_node_id, target_ast_node)
  knowledge_graph_edge = KnowledgeGraphEdge(
    source_knowledge_graph_node,
    target_knowledge_graph_node,
    KnowledgeGraphEdgeType.has_ast,
  )
  neo4j_has_ast_edge = knowledge_graph_edge.to_neo4j_edge()

  assert isinstance(neo4j_has_ast_edge, dict)

  assert "source" in neo4j_has_ast_edge
  assert "target" in neo4j_has_ast_edge

  assert neo4j_has_ast_edge["source"] == source_knowledge_graph_node.to_neo4j_node()
  assert neo4j_has_ast_edge["target"] == target_knowledge_graph_node.to_neo4j_node()


def test_to_neo4j_parent_of_edge():
  source_type = "method_declaration"
  source_start_line = 1
  source_end_line = 5
  source_text = "print('Hello world')"
  source_node_id = 1
  target_type = "return_statement"
  target_start_line = 7
  target_end_line = 9
  target_text = "return True"
  target_node_id = 10

  source_ast_node = ASTNode(
    source_type, source_start_line, source_end_line, source_text
  )
  source_knowledge_graph_node = KnowledgeGraphNode(source_node_id, source_ast_node)
  target_ast_node = ASTNode(
    target_type, target_start_line, target_end_line, target_text
  )
  target_knowledge_graph_node = KnowledgeGraphNode(target_node_id, target_ast_node)
  knowledge_graph_edge = KnowledgeGraphEdge(
    source_knowledge_graph_node,
    target_knowledge_graph_node,
    KnowledgeGraphEdgeType.parent_of,
  )
  neo4j_parent_of_edge = knowledge_graph_edge.to_neo4j_edge()

  assert isinstance(neo4j_parent_of_edge, dict)

  assert "source" in neo4j_parent_of_edge
  assert "target" in neo4j_parent_of_edge

  assert neo4j_parent_of_edge["source"] == source_knowledge_graph_node.to_neo4j_node()
  assert neo4j_parent_of_edge["target"] == target_knowledge_graph_node.to_neo4j_node()


def test_to_neo4j_has_text_edge():
  source_basename = "source"
  source_relative_path = "foo/bar/source.py"
  source_node_id = 1
  target_text = "Hello world"
  target_node_id = 10

  source_file_node = FileNode(source_basename, source_relative_path)
  source_knowledge_graph_node = KnowledgeGraphNode(source_node_id, source_file_node)
  target_text_node = TextNode(target_text)
  target_knowledge_graph_node = KnowledgeGraphNode(target_node_id, target_text_node)
  knowledge_graph_edge = KnowledgeGraphEdge(
    source_knowledge_graph_node,
    target_knowledge_graph_node,
    KnowledgeGraphEdgeType.has_text,
  )
  neo4j_has_text_edge = knowledge_graph_edge.to_neo4j_edge()

  assert isinstance(neo4j_has_text_edge, dict)

  assert "source" in neo4j_has_text_edge
  assert "target" in neo4j_has_text_edge

  assert neo4j_has_text_edge["source"] == source_knowledge_graph_node.to_neo4j_node()
  assert neo4j_has_text_edge["target"] == target_knowledge_graph_node.to_neo4j_node()


def test_to_neo4j_next_chunk_edge():
  source_text = "Hello"
  source_node_id = 1
  target_text = "world"
  target_node_id = 10

  source_text_node = TextNode(source_text)
  source_knowledge_graph_node = KnowledgeGraphNode(source_node_id, source_text_node)
  target_text_node = TextNode(target_text)
  target_knowledge_graph_node = KnowledgeGraphNode(target_node_id, target_text_node)
  knowledge_graph_edge = KnowledgeGraphEdge(
    source_knowledge_graph_node,
    target_knowledge_graph_node,
    KnowledgeGraphEdgeType.has_text,
  )
  neo4j_next_chunk_edge = knowledge_graph_edge.to_neo4j_edge()

  assert isinstance(neo4j_next_chunk_edge, dict)

  assert "source" in neo4j_next_chunk_edge
  assert "target" in neo4j_next_chunk_edge

  assert neo4j_next_chunk_edge["source"] == source_knowledge_graph_node.to_neo4j_node()
  assert neo4j_next_chunk_edge["target"] == target_knowledge_graph_node.to_neo4j_node()
