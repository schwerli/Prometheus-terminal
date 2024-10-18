"""The neo4j handler for writing the knowledge graph to neo4j."""

from typing import Sequence

from neo4j import GraphDatabase, ManagedTransaction

from prometheus.graph.graph_types import (
  Neo4jASTNode,
  Neo4jFileNode,
  Neo4jHasASTEdge,
  Neo4jHasFileEdge,
  Neo4jHasTextEdge,
  Neo4jNextChunkEdge,
  Neo4jParentOfEdge,
  Neo4jTextNode,
)
from prometheus.graph.knowledge_graph import KnowledgeGraph


class KnowledgeGraphHandler:
  """The handler to writing the Knowledge graph to neo4j."""

  def __init__(
    self, uri: str, user: str, password: str, database: str, batch_size: int
  ):
    """
    Args:
      uri: The neo4j uri.
      user: The neo4j username.
      password: The neo4j password.
      database: The neo4j database name.
      batch_size: The maximum number of nodes/edges written to neo4j each time.
    """
    self.driver = GraphDatabase.driver(uri, auth=(user, password))
    self.database = database
    self.batch_size = batch_size

  def _init_database(self, tx: ManagedTransaction):
    """Initialization of the neo4j database."""

    # Create constraints for node_id attributes.
    # It also means that node_id will be indexed.
    queries = [
      "CREATE CONSTRAINT unique_file_node_id IF NOT EXISTS "
      "FOR (n:FileNode) REQUIRE n.node_id IS UNIQUE",
      "CREATE CONSTRAINT unique_ast_node_id IF NOT EXISTS "
      "FOR (n:ASTNode) REQUIRE n.node_id IS UNIQUE",
      "CREATE CONSTRAINT unique_text_node_id IF NOT EXISTS "
      "FOR (n:TextNode) REQUIRE n.node_id IS UNIQUE",
    ]
    for query in queries:
      tx.run(query)

  def _write_file_nodes(
    self, tx: ManagedTransaction, file_nodes: Sequence[Neo4jFileNode]
  ):
    """Write Neo4jFileNode to neo4j."""
    query = """
      UNWIND $file_nodes AS file_node
      CREATE (a:FileNode {node_id: file_node.node_id, basename: file_node.basename, relative_path: file_node.relative_path})
    """
    for i in range(0, len(file_nodes), self.batch_size):
      file_nodes_batch = file_nodes[i : i + self.batch_size]
      tx.run(query, file_nodes=file_nodes_batch)

  def _write_ast_nodes(self, tx: ManagedTransaction, ast_nodes: Sequence[Neo4jASTNode]):
    """Write Neo4jASTNode to neo4j."""
    query = """
      UNWIND $ast_nodes AS ast_node
      CREATE (a:ASTNode {node_id: ast_node.node_id, start_line: ast_node.start_line, end_line: ast_node.end_line, type: ast_node.type, text: ast_node.text})
    """
    for i in range(0, len(ast_nodes), self.batch_size):
      ast_nodes_batch = ast_nodes[i : i + self.batch_size]
      tx.run(query, ast_nodes=ast_nodes_batch)

  def _write_text_nodes(
    self, tx: ManagedTransaction, text_nodes: Sequence[Neo4jTextNode]
  ):
    """Write Neo4jTextNode to neo4j."""
    query = """
      UNWIND $text_nodes AS text_node
      CREATE (a:TextNode {node_id: text_node.node_id, text: text_node.text, metadata: text_node.metadata})
    """
    for i in range(0, len(text_nodes), self.batch_size):
      text_nodes_batch = text_nodes[i : i + self.batch_size]
      tx.run(query, text_nodes=text_nodes_batch)

  def _write_has_file_edges(
    self, tx: ManagedTransaction, has_file_edges: Sequence[Neo4jHasFileEdge]
  ):
    """Write Neo4jHasFileEdge to neo4j."""
    query = """
      UNWIND $edges AS edge
      MATCH (source:FileNode), (target:FileNode)
      WHERE source.node_id = edge.source.node_id AND target.node_id = edge.target.node_id
      CREATE (source) -[:HAS_FILE]-> (target)
    """
    for i in range(0, len(has_file_edges), self.batch_size):
      has_file_edges_batch = has_file_edges[i : i + self.batch_size]
      tx.run(query, edges=has_file_edges_batch)

  def _write_has_ast_edges(
    self, tx: ManagedTransaction, has_ast_edges: Sequence[Neo4jHasASTEdge]
  ):
    """Write Neo4jHasASTEdge to neo4j."""
    query = """
      UNWIND $edges AS edge
      MATCH (source:FileNode), (target:ASTNode)
      WHERE source.node_id = edge.source.node_id AND target.node_id = edge.target.node_id
      CREATE (source) -[:HAS_AST]-> (target)
    """
    for i in range(0, len(has_ast_edges), self.batch_size):
      has_ast_edges_batch = has_ast_edges[i : i + self.batch_size]
      tx.run(query, edges=has_ast_edges_batch)

  def _write_has_text_edges(
    self, tx: ManagedTransaction, has_text_edges: Sequence[Neo4jHasTextEdge]
  ):
    """Write Neo4jHasTextEdge to neo4j."""
    query = """
      UNWIND $edges AS edge
      MATCH (source:FileNode), (target:TextNode)
      WHERE source.node_id = edge.source.node_id AND target.node_id = edge.target.node_id
      CREATE (source) -[:HAS_TEXT]-> (target)
    """
    for i in range(0, len(has_text_edges), self.batch_size):
      has_text_edges_batch = has_text_edges[i : i + self.batch_size]
      tx.run(query, edges=has_text_edges_batch)

  def _write_parent_of_edges(
    self, tx: ManagedTransaction, parent_of_edges: Sequence[Neo4jParentOfEdge]
  ):
    """Write Neo4jParentOfEdge to neo4j."""
    query = """
      UNWIND $edges AS edge
      MATCH (source:ASTNode), (target:ASTNode)
      WHERE source.node_id = edge.source.node_id AND target.node_id = edge.target.node_id
      CREATE (source) -[:PARENT_OF]-> (target)
    """
    for i in range(0, len(parent_of_edges), self.batch_size):
      parent_of_edges_batch = parent_of_edges[i : i + self.batch_size]
      tx.run(query, edges=parent_of_edges_batch)

  def _write_next_chunk_edges(
    self, tx: ManagedTransaction, next_chunk_edges: Sequence[Neo4jNextChunkEdge]
  ):
    """Write Neo4jNextChunkEdge to neo4j."""
    query = """
      UNWIND $edges AS edge
      MATCH (source:TextNode), (target:TextNode)
      WHERE source.node_id = edge.source.node_id AND target.node_id = edge.target.node_id
      CREATE (source) -[:NEXT_CHUNK]-> (target)
    """
    for i in range(0, len(next_chunk_edges), self.batch_size):
      next_chunk_edges_batch = next_chunk_edges[i : i + self.batch_size]
      tx.run(query, edges=next_chunk_edges_batch)

  def write_knowledge_graph(self, kg: KnowledgeGraph):
    """Write the knowledge graph to neo4j.

    Args:
      kg: The knowledge graph to write to neo4j.
    """
    with self.driver.session() as session:
      session.execute_write(self._init_database)

      session.execute_write(self._write_file_nodes, kg.get_neo4j_file_nodes())
      session.execute_write(self._write_ast_nodes, kg.get_neo4j_ast_nodes())
      session.execute_write(self._write_text_nodes, kg.get_neo4j_text_nodes())

      session.execute_write(self._write_has_ast_edges, kg.get_neo4j_has_ast_edges())
      session.execute_write(self._write_has_file_edges, kg.get_neo4j_has_file_edges())
      session.execute_write(self._write_has_text_edges, kg.get_neo4j_has_text_edges())
      session.execute_write(
        self._write_next_chunk_edges, kg.get_neo4j_next_chunk_edges()
      )
      session.execute_write(self._write_parent_of_edges, kg.get_neo4j_parent_of_edges())

  def close(self):
    """Close the driver."""
    self.driver.close()
