import logging
from pathlib import Path

from prometheus.docker.python_container import PythonContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class RunTestNode:
  def __init__(self, test_state_attr: str):
    self.test_state_attr = test_state_attr
    self._logger = logging.getLogger("prometheus.agents.context_provider_node")

  def __call__(self, state: IssueAnswerAndFixState):
    container = PythonContainer(Path(state["project_path"]))
    self._logger.info(f"Starting running the test at state {self.test_state_attr}")
    output = container.run_tests()
    self._logger.debug(f"Test output:\n{output}")
    container.cleanup()
    return {self.test_state_attr: output}
