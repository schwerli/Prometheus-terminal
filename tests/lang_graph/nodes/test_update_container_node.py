from unittest.mock import Mock

from prometheus.docker.general_container import GeneralContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode


def test_update_container_node():
  mocked_container = Mock(spec=GeneralContainer)
  mocked_container.container = Mock()
  mocked_container.update_files.return_value = None
  mocked_kg = Mock(spec=KnowledgeGraph)
  project_path = "/path/to/project"
  mocked_kg.get_local_path.return_value = "/path/to/project"
  update_container_node = UpdateContainerNode(mocked_container, mocked_kg)

  update_container_node(None)

  assert mocked_container.update_files.call_count == 1
  mocked_container.update_files.assert_called_with(new_project_path=project_path)
