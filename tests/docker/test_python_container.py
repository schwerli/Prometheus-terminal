import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from prometheus.docker.python_container import PythonContainer, PythonProjectConfig


@pytest.fixture
def temp_project_dir():
  """Create a temporary project directory for testing."""
  temp_dir = Path(tempfile.mkdtemp())
  project_path = temp_dir / "test_project"
  project_path.mkdir(parents=True)
  yield project_path
  shutil.rmtree(temp_dir)


@pytest.fixture
def mock_docker():
  """Mock docker client and container."""
  mock_container = MagicMock()
  mock_container.exec_run.return_value.output = b"Test output"

  # Mock the API client
  mock_api = MagicMock()
  mock_api.remove_image = MagicMock()

  # Create mock client
  mock_client = MagicMock()
  mock_client.containers.run.return_value = mock_container
  mock_client.api = mock_api
  mock_client.images.remove = MagicMock()

  # Patch docker.from_env to return our mock client
  with patch("prometheus.docker.base_container.docker.from_env", return_value=mock_client):
    yield {"client": mock_client, "container": mock_container, "docker_client": mock_client}


@pytest.fixture
def container(temp_project_dir, mock_docker):
  """Create a PythonContainer instance with mocked dependencies."""
  container_temp_dir = Path(tempfile.mkdtemp())
  container_project_dir = container_temp_dir / "project"
  container_project_dir.mkdir(parents=True)

  def mock_copytree(src, dst, *args, **kwargs):
    dst.mkdir(parents=True, exist_ok=True)
    return dst

  with patch("shutil.copytree", side_effect=mock_copytree):
    container = PythonContainer(temp_project_dir)
    # Ensure the project directory exists
    Path(container.project_path).mkdir(parents=True, exist_ok=True)
    container.container = mock_docker["container"]
    return container


def test_init_creates_temp_directory(temp_project_dir):
  def mock_copytree(src, dst, *args, **kwargs):
    dst.mkdir(parents=True, exist_ok=True)

  with patch("shutil.copytree", side_effect=mock_copytree) as mock_copy:
    container = PythonContainer(temp_project_dir)
    mock_copy.assert_called_once()
    assert Path(container.project_path).exists()


@pytest.mark.parametrize(
  "files,content,expected",
  [
    (
      ["setup.py", "requirements.txt"],
      "import pytest",
      {
        "has_setup_py": True,
        "has_pyproject_toml": False,
        "has_requirements_txt": True,
        "has_pytest": True,
        "has_unittest": False,
      },
    ),
    (
      ["pyproject.toml"],
      "import unittest",
      {
        "has_setup_py": False,
        "has_pyproject_toml": True,
        "has_requirements_txt": False,
        "has_pytest": False,
        "has_unittest": True,
      },
    ),
  ],
)
def test_get_project_config(temp_project_dir, files, content, expected):
  def mock_copytree(src, dst, *args, **kwargs):
    dst.mkdir(parents=True, exist_ok=True)

  def create_path_mock(name):
    mock = MagicMock()
    mock.name = name
    # Important: Set exists() based on whether the file is in our files list
    mock.exists.return_value = str(name) in files  # Changed to handle Path objects
    mock.absolute.return_value = mock
    mock.__truediv__ = lambda self, other: create_path_mock(other)
    mock.rglob = lambda pattern: [create_path_mock("test_file.py")]
    return mock

  mocked_content = mock_open(read_data=content)

  with (
    patch("shutil.copytree", side_effect=mock_copytree),
    patch("builtins.open", mocked_content),
    patch("prometheus.docker.base_container.Path", create_path_mock),
    patch("prometheus.docker.python_container.Path", create_path_mock),
  ):
    container = PythonContainer(temp_project_dir)
    config = container._get_project_config()

    for key, value in expected.items():
      assert getattr(config, key) == value


def test_get_dockerfile_content(container):
  dockerfile_content = container.get_dockerfile_content()
  assert dockerfile_content

  # Check for key elements in the Dockerfile
  assert "FROM python:3.11-slim" in dockerfile_content
  assert "WORKDIR /app" in dockerfile_content
  assert "COPY . /app/" in dockerfile_content


def test_build_docker_image(container, mock_docker):
  Path(container.project_path).mkdir(parents=True, exist_ok=True)

  # Force the client to be our mocked one
  container.client = mock_docker["client"]

  container.build_docker_image()

  # Verify the build was called with correct arguments
  mock_docker["client"].images.build.assert_called_once()
  build_args = mock_docker["client"].images.build.call_args[1]
  assert build_args["tag"] == container.tag_name


@pytest.mark.parametrize(
  "test_framework,expected_command",
  [(True, "python -m pytest -v"), (False, "python -m unittest discover -v")],
)
def test_run_test(
  container, mock_docker, test_framework, expected_command
):  # Changed from run_tests
  container.project_config = PythonProjectConfig(
    has_setup_py=False,
    has_pyproject_toml=False,
    has_requirements_txt=False,
    has_pytest=test_framework,
    has_unittest=not test_framework,
  )

  result = container.run_test()  # Changed from run_tests

  mock_docker["container"].exec_run.assert_called_with(expected_command, workdir="/app")
  assert result == "Test output"


def test_run_build(container, mock_docker):
  result = container.run_build()

  mock_docker["container"].exec_run.assert_called_with("python -m build", workdir="/app")
  assert result == "Test output"


def test_cleanup(container, mock_docker):
  project_path = Path(container.project_path)

  container.client = mock_docker["client"]
  container.container = mock_docker["container"]

  with patch("shutil.rmtree") as mock_rmtree:
    container.cleanup()

    mock_docker["container"].stop.assert_called_once()
    mock_docker["container"].remove.assert_called_once()
    mock_docker["client"].images.remove.assert_called_once_with(container.tag_name)
    mock_rmtree.assert_called_once_with(project_path)
