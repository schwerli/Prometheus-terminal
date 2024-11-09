import shutil
import tempfile
from pathlib import Path

import pytest

from prometheus.docker.general_container import GeneralContainer


@pytest.fixture
def temp_project_dir():
  # Create a temporary directory with some test files
  temp_dir = Path(tempfile.mkdtemp())
  test_file = temp_dir / "test.txt"
  test_file.write_text("test content")

  yield temp_dir

  # Cleanup
  shutil.rmtree(temp_dir)


@pytest.fixture
def container(temp_project_dir):
  return GeneralContainer(temp_project_dir)


def test_initialization(container, temp_project_dir):
  """Test that the container is initialized correctly"""
  assert isinstance(container.tag_name, str)
  assert container.tag_name.startswith("prometheus_general_container_")
  assert len(container.tag_name.split("_")[-1]) == 10  # UUID length
  assert container.project_path != temp_project_dir  # Should be in a new temp directory
  assert (container.project_path / "test.txt").exists()  # Files should be copied


def test_get_dockerfile_content(container):
  dockerfile_content = container.get_dockerfile_content()

  assert dockerfile_content

  # Check for key elements in the Dockerfile
  assert "FROM ubuntu:22.04" in dockerfile_content
  assert "WORKDIR /app" in dockerfile_content
  assert "RUN apt-get update" in dockerfile_content
  assert "COPY . /app/" in dockerfile_content


def test_run_build_raises_not_implemented(container):
  """Test that run_build raises NotImplementedError"""
  with pytest.raises(NotImplementedError):
    container.run_build()


def test_run_test_raises_not_implemented(container):
  """Test that run_test raises NotImplementedError"""
  with pytest.raises(NotImplementedError):
    container.run_test()
