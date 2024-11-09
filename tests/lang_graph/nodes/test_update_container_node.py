from unittest.mock import Mock

from prometheus.docker.general_container import GeneralContainer
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode


def test_update_container_node():
  mocked_container = Mock(spec=GeneralContainer)
  mocked_container.update_files.return_value = None
  update_container_node = UpdateContainerNode(mocked_container)

  project_path = "/path/to/project"
  state = {
    "project_path": project_path,
  }

  update_container_node(state)

  assert mocked_container.update_files.call_count == 1
  mocked_container.update_files.assert_called_with(new_project_path=project_path)
