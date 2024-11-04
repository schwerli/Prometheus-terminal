from pathlib import Path
from typing import Optional

from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j import knowledge_graph_handler


class KnowledgeGraphService:
  def __init__(self, neo4j_service: Neo4jService, neo4j_batch_size: int, max_ast_depth: int):
    self.kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
      neo4j_service.neo4j_driver, neo4j_batch_size
    )
    self.max_ast_depth = max_ast_depth
    self.kg = self._load_existing_knowledge_graph()

  def _load_existing_knowledge_graph(self) -> Optional[KnowledgeGraph]:
    if self.kg_handler.knowledge_graph_exists():
      return self.kg_handler.read_knowledge_graph()
    return None

  def build_and_save_knowledge_graph(self, path: Path):
    if self.exists():
      self.clear()

    kg = KnowledgeGraph(self.max_ast_depth)
    kg.build_graph(path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)

  def exists(self) -> bool:
    return self.kg_handler.knowledge_graph_exists()

  def clear(self):
    self.kg_handler.clear_knowledge_graph()
    self.kg = None
