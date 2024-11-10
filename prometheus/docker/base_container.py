import logging
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import docker


class BaseContainer(ABC):
  """An abstract base class for managing Docker containers with file synchronization capabilities.

  This class provides core functionality for creating, managing, and interacting with Docker
  containers. It handles container lifecycle operations including building images, starting
  containers, updating files, and cleanup. The class is designed to be extended for specific
  container implementations that specifies the Dockerfile, how to build and how to run the test.
  """

  client: docker.DockerClient = docker.from_env()
  tag_name: str
  container: docker.models.containers.Container
  project_path: Path
  logger: logging.Logger

  def __init__(self, project_path: Path):
    """Initialize the container with a project directory.

    Creates a temporary copy of the project directory to work with.

    Args:
      project_path: Path to the project directory to be containerized.
    """
    temp_dir = Path(tempfile.mkdtemp())
    temp_project_path = temp_dir / project_path.name
    shutil.copytree(project_path, temp_project_path)
    self.project_path = temp_project_path.absolute()
    self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

  @abstractmethod
  def get_dockerfile_content(self) -> str:
    """Get the content of the Dockerfile for building the container image.

    Returns:
        str: Content of the Dockerfile as a string.
    """
    pass

  def build_docker_image(self):
    """Build a Docker image using the Dockerfile content.

    Creates a Dockerfile in the project directory and builds a Docker image
    using the specified tag name.
    """
    dockerfile_content = self.get_dockerfile_content()
    dockerfile_path = self.project_path / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    self.logger.info(f"Building docker image {self.tag_name}")
    self.client.images.build(
      path=str(self.project_path), dockerfile=dockerfile_path.name, tag=self.tag_name
    )

  def start_container(self):
    """Start a Docker container from the built image.

    Starts a detached container with TTY enabled and mounts the Docker socket.
    """
    self.logger.info(f"Starting container from image {self.tag_name}")
    self.container = self.client.containers.run(
      self.tag_name,
      detach=True,
      tty=True,
      volumes={"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}},
    )

  def update_files(self, new_project_path: str, container_path: str = "/app"):
    """Update files in the running container with files from a local directory.

    Creates a tar archive of the new files and copies them into the container.

    Args:
        new_project_path: Path to the directory containing new files.
        container_path: Path in container where files should be copied. Defaults to "/app".
    """
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
    """Run build commands in the container.

    This method should be implemented by subclasses to define build steps.
    """
    pass

  @abstractmethod
  def run_test(self):
    """Run test commands in the container.

    This method should be implemented by subclasses to define test steps.
    """
    pass

  def execute_command(self, command: str) -> str:
    """Execute a command in the running container.

    Args:
        command: Command to execute in the container.

    Returns:
        str: Output of the command as a string.
    """
    self.logger.debug(f"Running command in container: {command}")
    exec_result = self.container.exec_run(command, workdir="/app")
    exec_result_str = exec_result.output.decode("utf-8")
    self.logger.debug(f"Command output:\n{exec_result_str}")
    return exec_result_str

  def cleanup(self):
    """Clean up container resources and temporary files.

    Stops and removes the container, removes the Docker image,
    and deletes temporary project files.
    """
    self.logger.info("Cleaning up container and temporary files")
    if self.container:
      self.container.stop()
      self.container.remove()
      self.container = None
      self.client.images.remove(self.tag_name)

    shutil.rmtree(self.project_path)
