from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class UpdateContainerNode:
  def __init__(self, container: BaseContainer):
    self.container = container

  def __call__(self, state: IssueAnswerAndFixState):
    self.container.update_files(new_project_path=state["project_path"])
