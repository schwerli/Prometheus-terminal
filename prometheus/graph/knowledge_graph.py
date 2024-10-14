from collections import deque
from pathlib import Path

from prometheus.graph.types import FileNode, KnowledgeGraphEdge, KnowledgeGraphEdgeType, KnowledgeGraphNode
from prometheus.parser import tree_sitter_parser


class KnowledgeGraph:

  def __init__(self, max_ast_depth: int):
    self.max_ast_depth = max_ast_depth
    self._next_node_id = 0
    self._root_node = None
    self._knowledge_graph_nodes = []
    self._knowledge_graph_edges = []

  def build_graph(self, root_dir: Path):
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
          child_file_node = FileNode(basename=child_file.name,
                                     relative_path=str(child_file.relative_to(root_dir)))
          kg_child_file_node = KnowledgeGraphNode(self._next_node_id, child_file_node)
          self._next_node_id += 1
          self._knowledge_graph_nodes.append(kg_child_file_node)
          self._knowledge_graph_edges.append(KnowledgeGraphEdge(kg_file_path_node, kg_child_file_node, KnowledgeGraphEdgeType.has_file))

          file_stack.append((child_file, kg_child_file_node))
        continue
        


