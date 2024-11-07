import dataclasses
import uuid
from pathlib import Path

from prometheus.docker.base_container import BaseContainer


@dataclasses.dataclass
class PythonProjectConfig:
  has_setup_py: bool
  has_pyproject_toml: bool
  has_requirements_txt: bool
  has_pytest: bool
  has_unittest: bool


class PythonContainer(BaseContainer):
  def __init__(self, project_path: Path):
    super().__init__(project_path)
    self.tag_name = f"prometheus_python_container_{uuid.uuid4().hex[:10]}"
    self.project_config = self._get_project_config()

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

  def create_dockerfile(self) -> Path:
    DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy project files
COPY . /app/

RUN pip install pytest pytest-cov build

RUN {install_requirements_cmd}
"""
    if self.project_config.has_requirements_txt:
      install_requirements_cmd = "pip install -r requirements.txt"
    else:
      install_requirements_cmd = "pip install ."

    dockerfile_content = DOCKERFILE_TEMPLATE.format(
      install_requirements_cmd=install_requirements_cmd,
    )
    dockerfile_path = self.project_path / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    return dockerfile_path

  def run_build(self):
    self.logger.info("Running Python build")
    return self.execute_command("python -m build")

  def run_test(self):
    self.logger.info("Running Python tests")
    if self.project_config.has_pytest:
      return self.execute_command("python -m pytest -v")
    return self.execute_command("python -m unittest discover -v")
