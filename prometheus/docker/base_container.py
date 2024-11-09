import logging
import shutil
import tarfile
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
  def get_dockerfile_content(self) -> str:
    pass

  def build_docker_image(self):
    dockerfile_content = self.get_dockerfile_content()
    dockerfile_path = self.project_path / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    self.logger.info(f"Building docker image {self.tag_name}")
    self.client.images.build(
      path=str(self.project_path), dockerfile=dockerfile_path.name, tag=self.tag_name
    )

  def start_container(self):
    self.logger.info(f"Starting container from image {self.tag_name}")
    self.container = self.client.containers.run(self.tag_name, detach=True, tty=True)

  def update_files(self, new_project_path: str, container_path: str = "/app"):
    self.logger.info(f"Updating files in running container with files from {new_project_path}")

    self.execute_command("rm -rf ./*")

    with tempfile.NamedTemporaryFile() as temp_tar:
      with tarfile.open(fileobj=temp_tar, mode="w") as tar:
        abs_project_path = Path(new_project_path).absolute()
        for path in abs_project_path.rglob("*"):
          rel_path = path.relative_to(abs_project_path)
          tar.add(str(path), arcname=str(rel_path))

      temp_tar.seek(0)

      self.container.put_archive(container_path, temp_tar.read())

    self.logger.info("Files updated successfully")

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
      self.container = None
      self.client.images.remove(self.tag_name)

    shutil.rmtree(self.project_path)
