import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import docker


class BaseContainer(ABC):
  client: docker.DockerClient = docker.from_env()
  tag_name: str
  container: docker.models.containers.Container
  project_path: Path
  logger: logging.Logger

  def __init__(self, project_path: Path):
    temp_dir = Path(tempfile.mkdtemp())
    temp_project_path = temp_dir / project_path.name
    shutil.copytree(project_path, temp_project_path)
    self.project_path = temp_project_path.absolute()
    self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

  @abstractmethod
  def create_dockerfile(self) -> Path:
    pass

  def build_docker_image(self):
    dockerfile_path = self.create_dockerfile()
    self.logger.info(f"Building docker image {self.tag_name}")
    self.client.images.build(
      path=str(self.project_path), dockerfile=dockerfile_path.name, tag=self.tag_name
    )

  def start_container(self):
    self.logger.info(f"Starting container from image {self.tag_name}")
    self.container = self.client.containers.run(self.tag_name, detach=True, tty=True)

  @abstractmethod
  def run_build(self):
    pass

  @abstractmethod
  def run_test(self):
    pass

  def execute_command(self, command: str) -> str:
    self.logger.debug(f"Running command in container: {command}")
    exec_result = self.container.exec_run(command, workdir="/app")
    exec_result_str = exec_result.output.decode("utf-8")
    self.logger.debug(f"Command output:\n{exec_result_str}")
    return exec_result_str

  def cleanup(self):
    self.logger.info("Cleaning up container and temporary files")
    if self.container:
      self.container.stop()
      self.container.remove()
      self.client.images.remove(self.tag_name)
      self.container = None

    shutil.rmtree(self.project_path)
