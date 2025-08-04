"""Service for managing and interacting with Knowledge Graphs in Neo4j."""

from pathlib import Path
from typing import Optional

from prometheus.app.services.base_service import BaseService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j import knowledge_graph_handler


class KnowledgeGraphService(BaseService):
    """Manages the lifecycle and operations of Knowledge Graphs.

    This service handles the creation, persistence, and management of Knowledge Graphs
    that represent the whole codebase structures. It provides capabilities for building graphs
    from codebase, storing them in Neo4j, and managing their lifecycle.
    """

    def __init__(
        self,
        neo4j_service: Neo4jService,
        neo4j_batch_size: int,
        max_ast_depth: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """Initializes the Knowledge Graph service.

        Args:
          neo4j_service: Service providing Neo4j database access.
          neo4j_batch_size: Number of nodes to process in each Neo4j batch operation.
          max_ast_depth: Maximum depth to traverse when building AST representations.
          chunk_size: Chunk size for processing text files.
          chunk_overlap: Overlap size for processing text files.
        """
        self.kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
            neo4j_service.neo4j_driver, neo4j_batch_size
        )
        self.max_ast_depth = max_ast_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kg = self._load_existing_knowledge_graph()

    def _load_existing_knowledge_graph(self) -> Optional[KnowledgeGraph]:
        """Attempts to load an existing Knowledge Graph from Neo4j.

        Returns:
            KnowledgeGraph if one exists in the database, None otherwise.
        """
        if self.kg_handler.knowledge_graph_exists():
            return self.kg_handler.read_knowledge_graph()
        return None

    def get_local_path(self) -> Path:
        if self.kg:
            return self.kg.get_local_path()
        return None

    def build_and_save_knowledge_graph(
        self, path: Path, https_url: Optional[str] = None, commit_id: Optional[str] = None
    ):
        """Builds a new Knowledge Graph from source code and saves it to Neo4j.

        Creates a new Knowledge Graph representation of the codebase at the specified path,
        optionally associating it with a repository URL and commit. Any existing
        Knowledge Graph will be cleared before building the new one.

        Args:
            path: Path to the source code directory to analyze.
            https_url: Optional HTTPS URL of the repository.
            commit_id: Optional commit identifier for version tracking.
        """
        if self.exists():
            self.clear()

        kg = KnowledgeGraph(self.max_ast_depth, self.chunk_size, self.chunk_overlap)
        kg.build_graph(path, https_url, commit_id)
        self.kg = kg
        self.kg_handler.write_knowledge_graph(kg)

    def exists(self) -> bool:
        return self.kg_handler.knowledge_graph_exists()

    def clear(self):
        self.kg_handler.clear_knowledge_graph()
        self.kg = None

    def close(self):
        """Clear the knowledge graph before closing the service."""
        self.clear()
