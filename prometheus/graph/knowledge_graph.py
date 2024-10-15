from collections import deque
import logging
from pathlib import Path

from prometheus.graph.file_graph_builder import FileGraphBuilder
from prometheus.graph.graph_types import (
  FileNode,
  KnowledgeGraphEdge,
  KnowledgeGraphEdgeType,
  KnowledgeGraphNode,
)


class KnowledgeGraph:
  def __init__(self, root_dir: Path, max_ast_depth: int):
    self.max_ast_depth = max_ast_depth
    self._next_node_id = 0
    self._root_node = None
    self._knowledge_graph_nodes = []
    self._knowledge_graph_edges = []
    self._file_graph_builder = FileGraphBuilder(max_ast_depth)
    self._logger = logging.getLogger("prometheus.graph.knowledge_graph")

    self._build_graph(root_dir)

  def _build_graph(self, root_dir: Path):
    root_dir_node = FileNode(basename=root_dir.name, relative_path=".")
    kg_root_dir_node = KnowledgeGraphNode(self._next_node_id, root_dir_node)
    self._next_node_id += 1
    self._knowledge_graph_nodes.append(kg_root_dir_node)
    self._root_node = kg_root_dir_node

    file_stack = deque()
    file_stack.append((root_dir, kg_root_dir_node))

    while file_stack:
      file, kg_file_path_node = file_stack.pop()

      if file.is_dir():
        for child_file in sorted(file.iterdir()):
          child_file_node = FileNode(
            basename=child_file.name,
            relative_path=str(child_file.relative_to(root_dir)),
          )
          kg_child_file_node = KnowledgeGraphNode(self._next_node_id, child_file_node)
          self._next_node_id += 1
          self._knowledge_graph_nodes.append(kg_child_file_node)
          self._knowledge_graph_edges.append(
            KnowledgeGraphEdge(
              kg_file_path_node, kg_child_file_node, KnowledgeGraphEdgeType.has_file
            )
          )

          file_stack.append((child_file, kg_child_file_node))
        continue

      if self._file_graph_builder.supports_file(file):
        next_node_id, kg_nodes, kg_edges = self._file_graph_builder.build_file_graph(
          kg_file_path_node, file, self._next_node_id
        )
        self._next_node_id = next_node_id
        self._knowledge_graph_nodes.extend(kg_nodes)
        self._knowledge_graph_edges.extend(kg_edges)
        continue

      self._logger.info(f"Skip parsing {file} because it is not supported.")
