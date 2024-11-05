from pathlib import Path

from prometheus.docker.python_container import PythonContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class RunTestNode:
  def __init__(self, project_path: Path, test_state_attr: str):
    self.project_path = project_path
    self.test_state_attr = test_state_attr

  def __call__(self, state: IssueAnswerAndFixState):
    container = PythonContainer(self.project_path)
    output = container.run_tests()
    container.cleanup()
    return {self.test_state_attr: output}
