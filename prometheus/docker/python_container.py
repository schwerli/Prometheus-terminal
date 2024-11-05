import dataclasses
import shutil
import tempfile
import uuid
from pathlib import Path

import docker


@dataclasses.dataclass
class PythonProjectConfig:
  has_setup_py: bool
  has_pyproject_toml: bool
  has_requirements_txt: bool
  has_pytest: bool
  has_unittest: bool


class PythonContainer:
  def __init__(self, project_path: Path):
    temp_dir = Path(tempfile.mkdtemp())
    temp_project_path = temp_dir / "project"
    shutil.copytree(project_path, temp_project_path)
    self.project_path = temp_project_path.absolute()
    self.project_config = self._get_project_config()
    self.client = docker.from_env()
    self.tag_name = f"prometheus_python_container_{uuid.uuid4().hex[:8]}"
    self.container = None

  def _get_project_config(self) -> PythonProjectConfig:
    has_setup_py = (self.project_path / "setup.py").exists()
    has_pyproject_toml = (self.project_path / "pyproject.toml").exists()
    has_requirements_txt = (self.project_path / "requirements.txt").exists()

    has_pytest = False
    has_unittest = False
    for file in self.project_path.rglob("*.py"):
      if "test" in file.name:
        with open(file) as f:
          content = f.read()
          if "pytest" in content:
            has_pytest = True
          if "unittest" in content:
            has_unittest = True

    return PythonProjectConfig(
      has_setup_py=has_setup_py,
      has_pyproject_toml=has_pyproject_toml,
      has_requirements_txt=has_requirements_txt,
      has_pytest=has_pytest,
      has_unittest=has_unittest,
    )

  def _create_dockerfile(self):
    DOCKERFILE_TEMPLATE = """\
FROM {base_image_name}

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install git and build essentials for potential requirements
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential

    # Copy project files
COPY . /app/

# Create and activate virtual environment
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install pytest pytest-cov

RUN {install_requirements_cmd}
"""
    if self.project_config.has_requirements_txt:
      install_requirements_cmd = "pip install -r requirements.txt"
    else:
      install_requirements_cmd = "pip install ."
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
      base_image_name="python:3.11-slim",
      install_requirements_cmd=install_requirements_cmd,
    )
    dockerfile_path = self.project_path / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    return dockerfile_path

  def _build_docker_image(self):
    dockerfile_path = self._create_dockerfile()
    self.client.images.build(
      path=str(self.project_path), dockerfile=dockerfile_path.name, tag=self.tag_name
    )

  def _start_container(self):
    self.container = self.client.containers.run(
      self.tag_name,
      detach=True,
      tty=True,
      volumes={str(self.project_path): {"bind": "/app", "mode": "rw"}},
    )

  def run_tests(self) -> str:
    if not self.container:
      self._build_docker_image()
      self._start_container()

    if self.project_config.has_pytest:
      return self._execute_command("pytest -v")
    return self._execute_command("python -m unittest discover -v")

  def _execute_command(self, command: str) -> str:
    exec_result = self.container.exec_run(
      command, workdir="/app", environment={"PYTHONPATH": "/app"}
    )
    return exec_result.output.decode("utf-8")

  def cleanup(self):
    if self.container:
      self.container.stop()
      self.container.remove()

    # Not work on windows
    #shutil.rmtree(self.project_path.parent)
