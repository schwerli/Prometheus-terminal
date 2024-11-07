from pydantic import BaseModel, Field

from prometheus.docker.general_container import GeneralContainer


class RunCommandInput(BaseModel):
  command: str = Field("The command to be run in the container")


READ_FILE_DESCRIPTION = """\
Run a command in the container and return the result of the command. You are always at the root
of the codebase.
"""


def run_command(command: str, container: GeneralContainer) -> str:
  return container.execute_command(command)
