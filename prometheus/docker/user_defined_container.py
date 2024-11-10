import uuid
from pathlib import Path
from typing import Sequence

from prometheus.docker.base_container import BaseContainer


class UserDefinedContainer(BaseContainer):
  def __init__(
    self,
    project_path: Path,
    dockerfile_content: str,
    build_commands: Sequence[str],
    test_commands: Sequence[str],
  ):
    super().__init__(project_path)
    self.tag_name = f"prometheus_user_defined_container_{uuid.uuid4().hex[:10]}"
    self.dockerfile_content = dockerfile_content
    self.build_commands = build_commands
    self.test_commands = test_commands

  def get_dockerfile_content(self) -> str:
    return self.dockerfile_content

  def run_build(self) -> str:
    command_output = ""
    for build_command in self.build_commands:
      command_output += f"$ {build_command}\n"
      command_output += f"{self.execute_command(build_command)}\n"
    return command_output

  def run_test(self) -> str:
    command_output = ""
    for test_command in self.test_commands:
      command_output += f"$ {test_command}\n"
      command_output += f"{self.execute_command(test_command)}\n"
    return command_output
